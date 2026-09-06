"""Convert SpatialSense into the mix (positives) + a negatives side file.

SpatialSense (Yang et al., ICCV'19; Zenodo 8104370) is adversarially collected:
9 spatial predicates over Flickr+NYU images, with VERIFIED NEGATIVE pairs
(label=false) — the only source in the mix with real "this relation does NOT
hold" supervision. The current pack format has no negative channel, so:

  * runs/datamix/spatialsense_train_coco.json — label=true relations from the
    dataset's own train+valid splits (their test split is held out entirely as
    a future spatial probe), minus registry-excluded images (86 hash-verified
    duplicates of PSG/VG150-eval photos).
  * runs/datamix/spatialsense_negatives.jsonl — every annotation (pos+neg, all
    splits) with boxes/names/split, for future PU-negative supervision and the
    held-out probe.

bbox convention in source: [y0, y1, x0, x1] pixels -> converted to xywh.
file_name is "flickr/<name>.jpg" | "nyu/<name>.jpg" relative to images/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.convert_datamix import CocoSGGWriter  # noqa: E402
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).
SS = DATASETS / "SpatialSense"
MIX = DATAMIX



def main() -> None:
    reg = json.load(open(MIX / "registry.json"))
    excluded = set(reg.get("spatialsense_exclude", []))

    # basename -> relative path index of the on-disk images
    disk = {}
    for sub in ("flickr", "nyu"):
        d = SS / "images" / sub
        for f in os.listdir(d):
            if f.endswith(".jpg") and not f.startswith("._"):
                disk[f] = f"{sub}/{f}"

    data = json.load(open(SS / "annotations.json"))
    from collections import Counter
    split_counts = Counter(r["split"] for r in data)
    print(f"records per split: {dict(split_counts)}")

    w = CocoSGGWriter()
    neg_out = open(MIX / "spatialsense_negatives.jsonl", "w")
    n_img = n_skip_excl = n_skip_disk = n_pos = n_neg = 0
    for img_id, r in enumerate(data):
        base = os.path.basename(r["url"])
        rel_path = disk.get(base)
        # negatives side file gets everything (incl. test split + excluded imgs
        # are still skipped — they are eval photos)
        if base in excluded:
            n_skip_excl += 1
            continue
        if rel_path is None:
            n_skip_disk += 1
            continue
        W, H = r["width"], r["height"]
        for a in r["annotations"]:
            rec = {"file_name": rel_path, "width": W, "height": H,
                   "split": r["split"], "predicate": a["predicate"].lower(),
                   "label": bool(a["label"]),
                   "subject": {"name": a["subject"]["name"].lower(),
                               "bbox_yyxx": a["subject"]["bbox"]},
                   "object": {"name": a["object"]["name"].lower(),
                              "bbox_yyxx": a["object"]["bbox"]}}
            neg_out.write(json.dumps(rec) + "\n")
            n_pos += int(a["label"]); n_neg += int(not a["label"])

        if r["split"] not in ("train", "valid"):
            continue                      # test split: probe only, never train
        boxes, cats, rels = [], [], []
        key_to_idx = {}

        def bidx(ent):
            y0, y1, x0, x1 = ent["bbox"]
            key = (round(x0), round(y0), round(x1), round(y1))
            if key not in key_to_idx:
                key_to_idx[key] = len(boxes)
                boxes.append([float(x0), float(y0),
                              max(float(x1 - x0), 1.0), max(float(y1 - y0), 1.0)])
                cats.append(ent["name"].strip().lower() or "object")
            return key_to_idx[key]

        for a in r["annotations"]:
            if not a["label"]:
                continue
            si, oi = bidx(a["subject"]), bidx(a["object"])
            rels.append((si, oi, a["predicate"].lower()))
        if rels:
            w.add_image(img_id, rel_path, W, H, boxes, cats, rels)
            n_img += 1

    neg_out.close()
    w.write(MIX / "spatialsense_train_coco.json")
    print(f"train imgs used: {n_img} | skipped: {n_skip_excl} excluded, "
          f"{n_skip_disk} not on disk")
    print(f"negatives file: {n_pos} positive + {n_neg} negative annotations "
          f"(all splits) -> spatialsense_negatives.jsonl")


if __name__ == "__main__":
    main()
