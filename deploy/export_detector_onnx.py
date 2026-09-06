"""Export a MEGASG-497 re-parameterized YOLO-World detector to ONNX.

`training/reparam_detectors.py` bakes the 497 MEGASG category embeddings into
the detector head (zero text-encoder overhead at inference). This script turns
that checkpoint into ONNX so the laptop side needs no torch and no ultralytics
— just onnxruntime, opencv and numpy (`deploy/runtime.py` reimplements
letterbox + NMS in numpy).

    python deploy/export_detector_onnx.py --size s --imgsz 640

Output: checkpoints/detectors/yolov8{size}-worldv2_megasg497.onnx
Raw YOLOv8 head layout, [1, 4 + 497, 8400]: cxcywh box coords in letterboxed
pixel space, then per-class scores (already sigmoid'd). No NMS baked in — it
runs host-side, which keeps the confidence threshold dynamic (same reasoning as
the relation head's score threshold; see deploy/export_onnx.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="s", choices=["s", "m", "x"],
                    help="YOLO-World v2 scale (s is ~20x cheaper than x)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--weights", default="", help="override checkpoint path")
    args = ap.parse_args()

    os.chdir(REPO)
    from ultralytics import YOLOWorld

    w = args.weights or f"checkpoints/detectors/yolov8{args.size}-worldv2_megasg497.pt"
    if not os.path.exists(w):
        sys.exit(f"missing {w} — run training/reparam_detectors.py first")

    m = YOLOWorld(w)
    names = [m.names[i] for i in range(len(m.names))]
    print(f"[det-export] {w}: {len(names)} classes @ {args.imgsz}px")

    p = m.export(format="onnx", imgsz=args.imgsz, opset=args.opset,
                 simplify=False, dynamic=False, nms=False)
    print(f"[det-export] wrote {p}")

    meta = {"classes": names, "imgsz": args.imgsz,
            "source": os.path.basename(w), "layout": "yolov8_raw_cxcywh"}
    mp = os.path.splitext(str(p))[0] + ".json"
    json.dump(meta, open(mp, "w"), indent=2)
    print(f"[det-export] wrote {mp}")


if __name__ == "__main__":
    main()
