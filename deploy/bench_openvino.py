"""End-to-end latency + agreement for the deploy bundle across engines.

Runs the full ScenePipeline (detector -> relation -> decode) on REAL images
and reports the per-stage split. With --compare it also measures what
quantization/conversion did to the OUTPUT: top-1 triplet agreement and top-K
set overlap against a baseline variant, image by image — the release evidence
that a faster artifact still says the same things.

On the Intel laptop:
    python deploy/bench_openvino.py --dist deploy/dist/relsgg-vits16plus \
        --variant int8 --device GPU --images ~/photos --n 30
    python deploy/bench_openvino.py --dist deploy/dist/relsgg-vits16plus \
        --variant int8 fp16 --device CPU --compare onnx --images...

Variants: onnx (onnxruntime fp32 baseline), fp16 / int8 / w4 (OpenVINO IR).
Devices apply to OpenVINO variants only; onnx always runs its CPU EP here.
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from deploy.postprocess import ThresholdConfig      # noqa: E402
from deploy.runtime import ScenePipeline            # noqa: E402


def build(dist: str, variant: str, device: str, threads: int,
          rel_device: str = "") -> ScenePipeline:
    thr = ThresholdConfig()
    if variant == "onnx":
        return ScenePipeline(dist, backend="onnx", threads=threads, thr_cfg=thr)
    rel = os.path.join(dist, f"relateanything_{variant}.xml")
    det = os.path.join(dist, f"detector_{variant}.xml")
    if not os.path.exists(det):                    # w4 is relation-only
        det = os.path.join(dist, "detector_fp16.xml")
    return ScenePipeline(dist, backend="openvino", relation=rel, detector=det,
                         device=device, rel_device=rel_device,
                         threads=threads, thr_cfg=thr)


def load_images(spec: str, n: int) -> list:
    import cv2
    paths = ([spec] if os.path.isfile(spec) else
             sorted(glob.glob(os.path.join(spec, "*.jpg"))
                    + glob.glob(os.path.join(spec, "*.png"))))
    if not paths:
        raise SystemExit(f"no images at {spec}")
    frames = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            frames.append((os.path.basename(p), img))
        if len(frames) >= n:
            break
    return frames


def _iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / max(ua, 1e-9)


def _same(a, b) -> bool:
    """Semantic triplet identity, robust to box renumbering between variants:
    same labels + predicate, and both endpoint boxes actually overlap. Keying
    on box INDEX instead makes one extra detection renumber everything and
    read as total disagreement (measured: fp16 'top-1 0.75' that was really
    1.00)."""
    return (a.predicate == b.predicate
            and a.subject_label == b.subject_label
            and a.object_label == b.object_label
            and _iou(a.subject_box, b.subject_box) >= 0.5
            and _iou(a.object_box, b.object_box) >= 0.5)


def _agreement(base, got):
    """(top1, jaccard) via greedy matching of two ranked triplet lists."""
    used = set()
    hits = 0
    top1 = 0.0
    for i, tb in enumerate(base):
        for j, tg in enumerate(got):
            if j in used:
                continue
            if _same(tb, tg):
                used.add(j)
                hits += 1
                if i == 0 and j == 0:
                    top1 = 1.0
                break
    union = len(base) + len(got) - hits
    return top1, hits / max(union, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--variant", nargs="+", default=["fp16"],
                    choices=["onnx", "fp16", "int8", "w4"])
    ap.add_argument("--device", default="CPU", help="CPU / GPU / NPU / AUTO")
    ap.add_argument("--rel-device", default="",
                    help="override --device for the relation head only "
                         "(e.g. CPU, when --device GPU hits the Intel GPU "
                         "plugin's 'Unsupported gather axis: 4')")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--images", required=True, help="image file or directory")
    ap.add_argument("--n", type=int, default=20, help="images to run")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--compare", default="",
                    choices=["", "onnx", "fp16", "int8", "w4"],
                    help="baseline variant for triplet-agreement metrics")
    ap.add_argument("--topk", type=int, default=20)
    args = ap.parse_args()
    os.chdir(REPO)

    frames = load_images(args.images, args.n)
    print(f"[bench] {len(frames)} images   device={args.device}   "
          f"threads={args.threads or 'auto'}")

    baseline = None
    if args.compare:
        base_pipe = build(args.dist, args.compare, args.device, args.threads,
                          args.rel_device)
        baseline = {}
        for name, img in frames:
            r = base_pipe(img)
            baseline[name] = r.triplets[:args.topk]
        del base_pipe

    for variant in args.variant:
        pipe = build(args.dist, variant, args.device, args.threads,
                    args.rel_device)
        for _, img in frames[:args.warmup]:
            pipe(img)                                   # warmup + GPU compile
        det, rel, dec, agree1, jac = [], [], [], [], []
        for name, img in frames:
            r = pipe(img)
            det.append(r.det_ms); rel.append(r.rel_ms); dec.append(r.dec_ms)
            if baseline is not None and baseline[name]:
                t1, j = _agreement(baseline[name], r.triplets[:args.topk])
                agree1.append(t1)
                jac.append(j)
        m = statistics.median
        total = m(det) + m(rel) + m(dec)
        line = (f"[bench] {variant:5s}  det {m(det):7.1f}  rel {m(rel):7.1f}  "
                f"dec {m(dec):5.2f}  total {total:7.1f} ms  "
                f"({1000.0 / total:4.1f} FPS)")
        if agree1:
            line += (f"   vs {args.compare}: top1 {np.mean(agree1):.3f}  "
                     f"top{args.topk}-jaccard {np.mean(jac):.3f} "
                     f"({len(agree1)} imgs)")
        print(line)
        del pipe


if __name__ == "__main__":
    main()
