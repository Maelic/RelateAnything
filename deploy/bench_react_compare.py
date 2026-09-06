"""End-to-end latency of detector + RelateAnything, REACT's protocol.

Mirrors sgg_benchmark tools/evaluate.py --skip-eval: batch size 1, real
benchmark images, N warm-up iterations, then per-image CUDA-event timing of
the whole detector -> relation-head path (image preprocessing included on our
side, which is conservative for us: REACT times only the model call).

    python deploy/bench_react_compare.py --checkpoint <ckpt> --pack runs/packed/vg150/test \
        --det.../yolov8m_vg150.pt --num-images 200 --num-warmup 20
"""
from __future__ import annotations
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pack", required=True, help="packed split dir (meta.json + file_names.json)")
    ap.add_argument("--det", required=True, help="ultralytics detector weights (closed-set YOLO)")
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--num-warmup", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--max-det", type=int, default=100, help="REACT profile --cap-dets 100")
    ap.add_argument("--max-boxes", type=int, default=60, help="boxes fed to the relation head")
    ap.add_argument("--topk", type=int, default=100)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from ultralytics import YOLO
    from relsgg.api import RelateAnything

    meta = json.load(open(os.path.join(args.pack, "meta.json")))
    names = json.load(open(os.path.join(args.pack, "file_names.json")))
    preds = meta["predicates"]
    paths = [os.path.join(meta["img_dir"], n) for n in names[: args.num_warmup + args.num_images]]

    dev = torch.device("cuda")
    det = YOLO(args.det)
    ra = RelateAnything.from_checkpoint(args.checkpoint, preds, device=dev)
    print(f"[bench] {torch.cuda.get_device_name(0)}  det={os.path.basename(args.det)}  "
          f"rel={os.path.basename(os.path.dirname(args.checkpoint))}  V={len(preds)}")

    def run(img):
        s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
        s2 = torch.cuda.Event(enable_timing=True)
        s0.record()
        r = det.predict(img, imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
                        device=0, verbose=False)[0]
        boxes = r.boxes.xyxy.float().cpu().numpy()
        confs = r.boxes.conf.float().cpu().numpy()
        s1.record()
        n = len(boxes)
        if n >= 2:
            ra.predict(img, boxes, box_scores=confs, topk=args.topk, max_boxes=args.max_boxes)
        s2.record()
        torch.cuda.synchronize()
        return s0.elapsed_time(s1), s1.elapsed_time(s2), s0.elapsed_time(s2), n

    for p in paths[: args.num_warmup]:
        run(Image.open(p).convert("RGB"))
    det_ms, rel_ms, tot_ms, nbox = [], [], [], []
    for p in paths[args.num_warmup:]:
        img = Image.open(p).convert("RGB")
        d, r_, t, n = run(img)
        det_ms.append(d); rel_ms.append(r_); tot_ms.append(t); nbox.append(n)
    res = {
        "gpu": torch.cuda.get_device_name(0), "n_images": len(tot_ms),
        "det": args.det, "checkpoint": args.checkpoint, "pack": args.pack,
        "mean_boxes": float(np.mean(nbox)),
        "det_ms_mean": statistics.mean(det_ms), "rel_ms_mean": statistics.mean(rel_ms),
        "total_ms_mean": statistics.mean(tot_ms), "total_ms_median": statistics.median(tot_ms),
        "total_ms_p95": float(np.percentile(tot_ms, 95)), "total_ms_std": statistics.pstdev(tot_ms),
        "fps": 1000.0 / statistics.mean(tot_ms),
    }
    print(f"[bench] {res['n_images']} imgs  boxes/img {res['mean_boxes']:.1f}  "
          f"det {res['det_ms_mean']:.1f} ms  rel {res['rel_ms_mean']:.1f} ms  "
          f"total {res['total_ms_mean']:.1f} ms (median {res['total_ms_median']:.1f}, "
          f"p95 {res['total_ms_p95']:.1f}, std {res['total_ms_std']:.1f})  {res['fps']:.1f} FPS")
    if args.out:
        json.dump(res, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
