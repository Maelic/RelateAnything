"""diag_overlap.py — is the concurrent backbone actually possible on this machine?

The overlap claim rests on the two branches being independent AND on the host being
able to drive both at once. Timing the composite alone cannot distinguish "the overlap
worked" from "the overlap serialized and added thread overhead", so this times each
branch in ISOLATION first and states the ceiling explicitly:

    theoretical best = sequential - min(det, backbone)      (perfect overlap)
    realized         = sequential - concurrent

A negative realized number means the worker thread cost more than it saved, which is
what a GIL-bound host looks like: both branches spend their time in Python issuing
kernel launches, so they contend for one interpreter lock instead of running side by side.
"""
from __future__ import annotations

import argparse, json, os, statistics, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def med(v):
    return float(np.median(np.asarray(v, dtype=float)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--det", default="checkpoints/detectors/yolov8m-worldv2_megasg497.pt")
    ap.add_argument("--pack", default="runs/packed/vg150/test")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import cv2, torch
    from deploy.pipeline import PipelineConfig, ParallelScenePipeline

    meta = json.load(open(os.path.join(a.pack, "meta.json")))
    names = json.load(open(os.path.join(a.pack, "file_names.json")))
    frames = [cv2.imread(os.path.join(meta["img_dir"], n))
              for n in names[: a.warmup + a.n]]
    frames = [f for f in frames if f is not None]

    def build(overlap):
        cfg = PipelineConfig(ckpt=os.path.join(a.run, "checkpoint_last.pth"),
                             vocab_npz=os.path.join(a.bundle, "predicate_bank.npz"),
                             det_weights=a.det, det_conf=0.10, masks=False,
                             max_objects=32, geo_budget=992, final_budget=992,
                             device="cuda", overlap=overlap, compile="")
        return ParallelScenePipeline(cfg)

    pipe = build(False)
    gpu = torch.cuda.get_device_name(0)
    print(f"[diag] {gpu} | OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')} "
          f"| torch threads {torch.get_num_threads()} | {len(frames)} frames")

    @torch.no_grad()
    def det_only(f):
        pipe._detect(f)

    @torch.no_grad()
    def bb_only(f):
        x = pipe._prep_image(f)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=True):
            pipe.model.backbone.extract(pipe.model.backbone.preprocess(x))

    def timeit(fn):
        for f in frames[: a.warmup]:
            fn(f)
        torch.cuda.synchronize()
        out = []
        for f in frames[a.warmup:]:
            t0 = time.perf_counter()
            fn(f)
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
        return out

    d = timeit(det_only)
    b = timeit(bb_only)

    def whole(p):
        for f in frames[: a.warmup]:
            p(f, top_k=12)
        torch.cuda.synchronize()
        out = []
        for f in frames[a.warmup:]:
            t0 = time.perf_counter()
            p(f, top_k=12)
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
        return out

    seq = whole(pipe)
    del pipe; torch.cuda.empty_cache()
    pcon = build(True)
    con = whole(pcon)
    del pcon; torch.cuda.empty_cache()

    D, B, S, C = med(d), med(b), med(seq), med(con)
    ceiling = S - min(D, B)
    res = {"gpu": gpu, "omp": os.environ.get("OMP_NUM_THREADS"),
           "torch_threads": torch.get_num_threads(),
           "det_only_ms": D, "backbone_only_ms": B,
           "sequential_ms": S, "concurrent_ms": C,
           "perfect_overlap_ms": ceiling,
           "realized_saving_ms": S - C,
           "fraction_of_theoretical": (S - C) / max(S - ceiling, 1e-6)}
    print(f"  detector alone   {D:6.1f} ms")
    print(f"  backbone alone   {B:6.1f} ms")
    print(f"  sequential total {S:6.1f} ms")
    print(f"  concurrent total {C:6.1f} ms")
    print(f"  perfect overlap would be {ceiling:6.1f} ms "
          f"(saving {min(D, B):.1f}); realized saving {S - C:+.1f} ms "
          f"= {100 * res['fraction_of_theoretical']:.0f}% of theoretical")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
