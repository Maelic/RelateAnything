"""bench_pipeline_family.py — end-to-end detector + RelateAnything latency, as a matrix.

Same protocol as deploy/bench_react_compare.py (batch 1, real benchmark images, N
warm-up then CUDA-event timing of detector -> relation head, preprocessing included on
our side), but it sweeps MODELS x DETECTORS in one process. The relation model is the
expensive thing to construct, so it is loaded once and every detector is timed against
it; running the cross product as separate processes would spend most of its wall time
rebuilding DINOv3.

WHY A MATRIX AND NOT A SINGLE NUMBER
------------------------------------
RelateAnything is relation-only: it consumes boxes and cannot produce them. Its
deployed cost is therefore detector + head, and the detector is a free choice that
moves the total by more than the head does. Reporting one "pipeline latency" would
hide that choice; the matrix makes the trade explicit.

Two detector regimes, and the difference between them is NOT just weights:
  closed-set   a fixed class list baked into the head (yolo11m, yolo26m @ 80 COCO)
  open-vocab   the class head is built from text at set-up time, so its WIDTH is a
               deployment choice -- 150 pack categories, 497 MegaSG, or YOLOE's
               4,585-class prompt-free head. Class-head width is a first-order
               latency term, so each open-vocab detector is timed at the width it is
               actually deployed with.

    python deploy/bench_pipeline_family.py \
        --runs "ViT-S/16=runs/train/<run>" --dets "yolo11m=checkpoints/detectors/yolo11m.pt" \
        --pack runs/packed/vg150/test --out runs/benchmark/cost/pipeline_<gpu>.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def build_detector(spec: str, categories, imgsz: int):
    """spec = <path>[:prompt] where prompt is 'pack' | 'native'.

    'pack' prompts an open-vocabulary detector with the benchmark's own categories
    (training/detect_boxes.py --set_classes); 'native' leaves the head as shipped —
    COCO-80 for a closed-set model, 497 for a reparameterised YOLO-World, 4,585 for
    YOLOE prompt-free. The prompt is part of the measurement, so it is reported.
    """
    from ultralytics import YOLO
    path, _, prompt = spec.partition(":")
    prompt = prompt or "native"
    y = YOLO(path)
    if prompt == "pack":
        names = list(categories)
        try:
            y.set_classes(names, y.get_text_pe(names))      # YOLOE
        except (AttributeError, TypeError):
            y.set_classes(names)                            # YOLO-World
    return y, path, prompt, len(y.names)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir")
    ap.add_argument("--dets", nargs="+", required=True, help="NAME=path[:pack|native]")
    ap.add_argument("--pack", default="runs/packed/vg150/test")
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--num_warmup", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--max_det", type=int, default=100)
    ap.add_argument("--max_boxes", type=int, default=60)
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--synthetic_vocab", action="store_true",
                    help="stand-in vocabulary matrix when the checkpoint's text student "
                         "is absent; exact for cost, meaningless for accuracy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from relsgg.api import RelateAnything
    from benchmark.vocab_stub import stub_vocabulary

    meta = json.load(open(os.path.join(a.pack, "meta.json")))
    names = json.load(open(os.path.join(a.pack, "file_names.json")))
    preds, cats = meta["predicates"], meta["categories"]
    paths = [os.path.join(meta["img_dir"], n)
             for n in names[: a.num_warmup + a.num_images]]
    imgs = [Image.open(p).convert("RGB") for p in paths]

    dev = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    print(f"[bench] {gpu} | {len(imgs)} images | V={len(preds)} | "
          f"{len(cats)} pack categories")

    # Detectors are built once and reused across relation models: prompting YOLOE
    # runs its text encoder over the whole category list, which is slow and would
    # otherwise be repeated per model for no reason.
    dets = {}
    for spec in a.dets:
        name, _, s = spec.partition("=")
        try:
            y, path, prompt, ncls = build_detector(s, cats, a.imgsz)
        except Exception as exc:                                  # noqa: BLE001
            print(f"!! detector {name}: {exc!r}")
            continue
        y.model.to(dev).eval()
        dets[name] = (y, path, prompt, ncls)
        print(f"  detector {name:<22} {os.path.basename(path):<24} "
              f"prompt={prompt:<7} classes={ncls}")

    out = {"gpu": gpu, "node": platform.node(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "pack": a.pack, "V": len(preds),
           "n_images": a.num_images, "n_warmup": a.num_warmup,
           "imgsz": a.imgsz, "conf": a.conf, "max_det": a.max_det,
           "max_boxes": a.max_boxes, "batch_size": 1,
           "synthetic_vocab": bool(a.synthetic_vocab), "rows": []}

    for spec in a.runs:
        mname, _, run_dir = spec.partition("=")
        ck = os.path.join(run_dir, "checkpoint_last.pth")
        if not os.path.exists(ck):
            print(f"!! {mname}: no {ck}")
            continue
        E = None
        if a.synthetic_vocab:
            _, E = stub_vocabulary(preds, 512, dev)
        ra = RelateAnything.from_checkpoint(ck, preds, device=dev, embeddings=E)
        print(f"\n[{mname}] {run_dir}")

        for dname, (det, dpath, prompt, ncls) in dets.items():
            def run(img):
                e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                e[0].record()
                r = det.predict(img, imgsz=a.imgsz, conf=a.conf, max_det=a.max_det,
                                device=0, verbose=False)[0]
                boxes = r.boxes.xyxy.float().cpu().numpy()
                confs = r.boxes.conf.float().cpu().numpy()
                e[1].record()
                if len(boxes) >= 2:
                    ra.predict(img, boxes, box_scores=confs, topk=a.topk,
                               max_boxes=a.max_boxes)
                e[2].record()
                torch.cuda.synchronize()
                return (e[0].elapsed_time(e[1]), e[1].elapsed_time(e[2]),
                        e[0].elapsed_time(e[2]), len(boxes))

            for im in imgs[: a.num_warmup]:
                run(im)
            d_ms, r_ms, t_ms, nb = [], [], [], []
            for im in imgs[a.num_warmup:]:
                d, r_, t, n = run(im)
                d_ms.append(d); r_ms.append(r_); t_ms.append(t); nb.append(n)

            row = {"model": mname, "run_dir": run_dir, "detector": dname,
                   "det_weights": dpath, "det_prompt": prompt, "det_classes": ncls,
                   "boxes_per_img": float(np.mean(nb)),
                   "det_ms": statistics.fmean(d_ms), "rel_ms": statistics.fmean(r_ms),
                   "total_ms": statistics.fmean(t_ms),
                   "total_ms_p50": float(np.percentile(t_ms, 50)),
                   "total_ms_p95": float(np.percentile(t_ms, 95)),
                   "total_ms_std": float(np.std(t_ms)),
                   "fps": 1000.0 / statistics.fmean(t_ms)}
            out["rows"].append(row)
            print(f"  {dname:<22} boxes/img {row['boxes_per_img']:5.1f}  "
                  f"det {row['det_ms']:6.1f}  rel {row['rel_ms']:6.1f}  "
                  f"total {row['total_ms']:6.1f} ms  ({row['fps']:.1f} FPS)")

        del ra
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
