"""Re-parameterize an open-vocabulary object detector to a fixed vocabulary.

YOLO-World / YOLOE carry a text-conditioned detection head. `set_classes` bakes
a chosen vocabulary's text embeddings into the head so that, at inference, the
detector runs with ZERO language-model overhead — the exact same trick the
relation head uses. Reparameterized weights load
and run offline.

Run once (needs internet the first time to fetch the base detector + its CLIP /
MobileCLIP text encoder):

    python deploy/reparam_detector.py --arch yolo-world --out yoloworld_demo.pt
    python deploy/reparam_detector.py --arch yoloe       --out yoloe_demo.pt

Then the webcam demo loads the reparameterized weights directly.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.vocab import OBJECT_VOCAB  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["yolo-world", "yoloe"], default="yolo-world")
    ap.add_argument("--base", default="",
                    help="base weights (default: yolov8x-worldv2.pt / yoloe-11l-seg.pt)")
    ap.add_argument("--out", default="", help="output.pt (default derived from arch)")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="override object vocabulary (default: deploy/vocab.OBJECT_VOCAB)")
    args = ap.parse_args()

    names = args.classes if args.classes else OBJECT_VOCAB
    print(f"[reparam_detector] {args.arch}: baking {len(names)} classes")

    if args.arch == "yolo-world":
        from ultralytics import YOLOWorld
        base = args.base or "yolov8x-worldv2.pt"
        out = args.out or "yoloworld_demo.pt"
        m = YOLOWorld(base)          # auto-downloads if missing
        m.set_classes(names)
    else:
        from ultralytics import YOLOE
        base = args.base or "yoloe-11l-seg.pt"
        out = args.out or "yoloe_demo.pt"
        m = YOLOE(base)
        m.set_classes(names, m.get_text_pe(names))

    m.save(out)
    print(f"[reparam_detector] saved -> {out}  (vocabulary baked in; runs offline)")


if __name__ == "__main__":
    main()
