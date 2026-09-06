"""Re-parametrize open-vocab detectors on the MEGASG 497-category vocabulary.

Run ONCE on the login node (needs internet for weight + CLIP/MobileCLIP
downloads).  Produces prompt-baked checkpoints that the offline GPU jobs can
load without any text encoder:

    checkpoints/detectors/yolov8x-worldv2_megasg497.pt   (YOLO-World v2)
    checkpoints/detectors/yoloe-11l-megasg497.pt         (YOLOE-11L)

Usage:  cd checkpoints/detectors &&../../.venv/bin/python../../training/reparam_detectors.py
(cwd matters: ultralytics drops CLIP/MobileCLIP weights into the cwd)
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DET = REPO / "checkpoints" / "detectors"
DET.mkdir(parents=True, exist_ok=True)

names = json.load(open(REPO / "runs/packed/megasg/train/meta.json"))["categories"]
assert len(names) == 497, len(names)

from ultralytics import YOLOWorld, YOLOE

print("=== YOLO-World v2 (x) ===")
m = YOLOWorld(str(DET / "yolov8x-worldv2.pt"))  # auto-downloads if missing
m.set_classes(names)
m.save(str(DET / "yolov8x-worldv2_megasg497.pt"))
print("saved yolov8x-worldv2_megasg497.pt")

print("=== YOLOE-11L ===")
me = YOLOE(str(DET / "yoloe-11l-seg.pt"))  # auto-downloads if missing
me.set_classes(names, me.get_text_pe(names))
me.save(str(DET / "yoloe-11l-megasg497.pt"))
print("saved yoloe-11l-megasg497.pt")
