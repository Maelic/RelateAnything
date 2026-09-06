"""bench_deploy_family.py — deployment-ready end-to-end latency for the ViT family.

What separates this from deploy/bench_pipeline_family.py: that one times the RESEARCH
path (relsgg.api on raw checkpoints, sequential). This one times what actually ships —
the release vocabulary baked from `predicate_bank.npz`, the decode included, and the
two levers a deployment has that a research path does not:

  CONCURRENT BACKBONE   The relation backbone consumes only the IMAGE, never the boxes,
                        so it does not depend on the detector and can run beside it.
                        `PipelineConfig.overlap` puts it on a worker THREAD -- not a
                        CUDA stream, which was measured to buy exactly nothing: both
                        branches are CPU-dispatch bound and a stream adds no CPU thread.
  GRAPH INFERENCE       torch.compile ("default" / "reduce-overhead") or ONNX Runtime.
                        Inductor changes reduction order, so a compiled arm is a
                        DEPLOYMENT number and never a reported metric.

The ONNX arm is necessarily sequential: the exported graph takes image AND boxes as one
signature, so the backbone cannot be started before the detector has finished. Splitting
it into backbone -> features and features+boxes -> logits is what an ONNX deployment
would need to claim the overlap win, and this benchmark is what says whether that is
worth building.

    python deploy/bench_deploy_family.py --models "ViT-S/16+=<run_dir>=<bundle>" \
        --pack runs/packed/vg150/test --out runs/benchmark/cost/deploy_<gpu>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

# (label, backend, overlap, compile_mode)
ARMS = [
    ("torch-eager-seq",        "torch", False, ""),
    ("torch-eager-concurrent", "torch", True,  ""),
    ("torch-compile-seq",      "torch", False, "default"),
    ("torch-compile-concurrent", "torch", True, "default"),
    ("torch-cudagraph-concurrent", "torch", True, "reduce-overhead"),
    ("onnx-cuda-seq",          "onnx",  False, ""),
]


def summarize(v):
    a = np.asarray(v, dtype=float)
    med = float(np.median(a))
    return {"mean": float(a.mean()), "p50": med, "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()),
            "fps_p50": 1000.0 / max(med, 1e-6),
            # A deployment cares about the worst frame, not only the median: a
            # CUDA-graph arm that re-records on a new detection count shows up
            # here and nowhere else.
            "stalls_gt_3x_median": int((a > 3 * med).sum()),
            "worst_over_median": float(a.max() / max(med, 1e-6))}


def run_torch(arm, model_run, bundle, det, frames, a):
    import torch
    from deploy.pipeline import PipelineConfig, ParallelScenePipeline
    _, _, overlap, mode = arm
    cfg = PipelineConfig(
        ckpt=os.path.join(model_run, "checkpoint_last.pth"),
        vocab_npz=os.path.join(bundle, "predicate_bank.npz"),
        det_weights=det, det_conf=a.conf, det_imgsz=a.imgsz, masks=False,
        max_objects=a.max_objects, geo_budget=a.max_objects * (a.max_objects - 1),
        final_budget=a.max_objects * (a.max_objects - 1),
        device="cuda", overlap=overlap, compile=mode,
        overlap_stream=not a.no_overlap_stream)
    pipe = ParallelScenePipeline(cfg)
    warm_s = pipe.warmup()
    # A second warm frame at a REAL detection count: a CUDA-graph arm records its
    # first true shape here instead of inside the timed region.
    pipe(frames[0], top_k=a.top_k)
    torch.cuda.synchronize()
    for f in frames[: a.num_warmup]:
        pipe(f, top_k=a.top_k)
    torch.cuda.synchronize()

    det_ms, bb_ms, rel_ms, tot_ms, nobj = [], [], [], [], []
    for f in frames[a.num_warmup:]:
        r = pipe(f, top_k=a.top_k)
        torch.cuda.synchronize()
        det_ms.append(r.timing.det); bb_ms.append(r.timing.backbone)
        rel_ms.append(r.timing.relation); tot_ms.append(r.timing.total)
        nobj.append(len(r.labels))
    out = {"warmup_s": round(warm_s, 1), "boxes_per_img": float(np.mean(nobj)),
           "distinct_object_counts": len(set(nobj)),
           "det_ms": float(np.mean(det_ms)), "backbone_ms": float(np.mean(bb_ms)),
           "relation_ms": float(np.mean(rel_ms)), "total": summarize(tot_ms)}
    del pipe
    torch.cuda.empty_cache()
    return out


def run_onnx(arm, onnx_rel, onnx_det, bundle, frames, a):
    """The ONNX host is torch-free: its own letterbox + NMS in numpy, its own decode.

    The detector's input size comes from the exported graph's sidecar json, not from
    --imgsz, so the two backends are only comparable if the ONNX detector was exported
    at the same size the torch arm is run at -- which is why the export size is echoed
    into the result rather than assumed.
    """
    from deploy.runtime import ScenePipeline, DetectorConfig
    providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    pipe = ScenePipeline(dist_dir=bundle, detector=onnx_det, relation=onnx_rel,
                         bank=os.path.join(bundle, "predicate_bank.npz"),
                         backend="onnx", providers=providers, threads=a.threads,
                         det_cfg=DetectorConfig(conf=a.conf, max_det=a.max_objects))
    used = list(pipe.rel.sess.get_providers()) if hasattr(pipe.rel, "sess") else []
    for f in frames[: a.num_warmup]:
        pipe(f)
    tot_ms, det_ms, rel_ms, dec_ms, nobj = [], [], [], [], []
    for f in frames[a.num_warmup:]:
        t0 = time.perf_counter()
        r = pipe(f)
        tot_ms.append((time.perf_counter() - t0) * 1e3)
        det_ms.append(r.det_ms); rel_ms.append(r.rel_ms); dec_ms.append(r.dec_ms)
        nobj.append(len(r.labels))
    return {"providers": used, "boxes_per_img": float(np.mean(nobj)),
            "det_imgsz": getattr(pipe.det, "imgsz", None),
            "det_ms": float(np.mean(det_ms)), "relation_ms": float(np.mean(rel_ms)),
            "decode_ms": float(np.mean(dec_ms)), "total": summarize(tot_ms)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True,
                    help="NAME=run_dir=bundle_dir")
    ap.add_argument("--onnx_dir", default="runs/benchmark/deploy_onnx")
    ap.add_argument("--det", default="checkpoints/detectors/yolov8m-worldv2_megasg497.pt")
    ap.add_argument("--onnx_det",
                    default="checkpoints/detectors/yolov8m-worldv2_megasg497.onnx")
    ap.add_argument("--pack", default="runs/packed/vg150/test")
    ap.add_argument("--num_images", type=int, default=120)
    ap.add_argument("--num_warmup", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--max_objects", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=12)
    ap.add_argument("--threads", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0),
                    help="ORT intra-op threads. Must be set explicitly under a SLURM "
                         "cpuset: with 0 (auto) onnxruntime sizes its pool from the "
                         "machine's core count and every pinning call fails with "
                         "pthread_setaffinity_np EINVAL against the allocated mask.")
    ap.add_argument("--arms", nargs="*", default=[l for l, *_ in ARMS])
    ap.add_argument("--torch_threads", type=int, default=0,
                    help="torch.set_num_threads(). 0 = leave torch's default, which "
                         "is the MACHINE core count -- with the concurrent backbone "
                         "that is two intra-op pools sized for 96 cores inside an "
                         "8-core cpuset.")
    ap.add_argument("--no_overlap_stream", action="store_true",
                    help="run the backbone worker on the default CUDA stream")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import cv2
    meta = json.load(open(os.path.join(a.pack, "meta.json")))
    names = json.load(open(os.path.join(a.pack, "file_names.json")))
    frames = []
    for n in names[: a.num_warmup + a.num_images]:
        f = cv2.imread(os.path.join(meta["img_dir"], n))
        if f is not None:
            frames.append(f)
    import torch
    if a.torch_threads:
        torch.set_num_threads(a.torch_threads)
    gpu = torch.cuda.get_device_name(0)
    print(f"[deploy] {gpu} | {len(frames)} frames | det {os.path.basename(a.det)}")

    out = {"gpu": gpu, "node": platform.node(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "pack": a.pack, "batch_size": 1,
           "n_images": len(frames) - a.num_warmup, "n_warmup": a.num_warmup,
           "det": a.det, "onnx_det": a.onnx_det, "imgsz": a.imgsz, "conf": a.conf,
           "max_objects": a.max_objects, "ort_threads": a.threads,
           "torch_threads": torch.get_num_threads(),
           "overlap_stream": not a.no_overlap_stream, "rows": []}

    for spec in a.models:
        mname, run_dir, bundle = spec.split("=", 2)
        onnx_rel = os.path.join(
            a.onnx_dir, "relateanything_" + bundle.rstrip("/").split("/")[-1]
.replace("relsgg-", "") + ".onnx")
        print(f"\n=== {mname} ===")
        for arm in ARMS:
            label, backend, overlap, mode = arm
            if label not in a.arms:
                continue
            try:
                if backend == "torch":
                    r = run_torch(arm, run_dir, bundle, a.det, frames, a)
                else:
                    if not os.path.exists(onnx_rel):
                        print(f"  {label:<28} skipped: no {onnx_rel}")
                        continue
                    r = run_onnx(arm, onnx_rel, a.onnx_det, bundle, frames, a)
            except Exception as exc:                              # noqa: BLE001
                print(f"  {label:<28} FAILED {exc!r}")
                out["rows"].append({"model": mname, "arm": label, "error": repr(exc)})
                continue
            row = {"model": mname, "arm": label, "backend": backend,
                   "concurrent_backbone": overlap, "graph_mode": mode or "eager",
                   **r}
            out["rows"].append(row)
            t = r["total"]
            print(f"  {label:<28} p50 {t['p50']:6.1f}  p95 {t['p95']:6.1f}  "
                  f"max {t['max']:6.1f} ms  {t['fps_p50']:5.1f} FPS  "
                  f"stalls {t['stalls_gt_3x_median']}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
