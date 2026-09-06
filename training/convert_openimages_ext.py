"""Convert the Open Images V6 extension (DATASETS/MEGASG_OI_ext) into the mix.

Source: 62,612 images downloaded from the public s3://open-images-dataset
bucket — OI V6 "visual relationships" (VRD) annotations, images with >=2 true
relationship rows (RelationshipLabel != "is", which is an attribute triple:
same box for subject and object, not a two-entity relation) that are NOT
already present in MEGASG's own Open Images subset. All 3 OI splits
(train/val/test) are combined into ONE train-only mix source, matching how
svg_vg/gqa/spatialsense are added (OI's own benchmark split is irrelevant
here; we're not evaluating against the OI VRD challenge).

bbox convention in source: XMin/XMax/YMin/YMax normalized [0,1] -> xywh px
(actual pixel size read from the downloaded file, not assumed). LabelName1/2
are Open Images MID codes -> resolved to display names via
oidv6-class-descriptions.csv. file_name is "<split>/<id>.jpg" relative to
DATASETS/MEGASG_OI_ext/ (same relative-path convention as SpatialSense's
flickr/nyu split dirs).

Eval-leak guard: 21 images are exact content-hash (dHash d=0) duplicates of
protected PSG/VG150 eval photos (runs/datamix/oi_ext_hash_audit.json, built
by training/audit_image_overlap.py --sets megasg_oi_ext) -- excluded here.
3,958 additional d<=8 "suspects" were flagged but NOT excluded (same policy
as the original MEGASG audit: d<=8 is a high-recall screen for manual/256-bit
review, not itself proof of duplication -- in the original audit only 2/many
suspects survived refinement).

Usage:
    python training/convert_openimages_ext.py
"""
from __future__ import annotations

import csv
import json

import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.convert_datamix import CocoSGGWriter  # noqa: E402
from relsgg.paths import DATASETS, DATAMIX  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).
OI = DATASETS / "MEGASG_OI_ext"
MIX = DATAMIX
SCRATCH = DATASETS / "MEGASG_OI_ext/meta"   # oidv6-*.csv downloaded from Open Images

csv.field_size_limit(2**28)


def load_class_names() -> dict[str, str]:
    names = {}
    with open(SCRATCH / "oidv6-class-descriptions.csv", newline="") as f:
        for row in csv.DictReader(f):
            names[row["LabelName"]] = row["DisplayName"].strip().lower()
    return names


def load_manifest() -> dict[str, set]:
    by_split: dict[str, set] = defaultdict(set)
    with open(OI / "downloaded_manifest.tsv", newline="") as f:
        for line in f:
            split, iid, _size = line.rstrip("\n").split("\t")
            by_split[split].add(iid)
    return by_split


def main() -> None:
    class_names = load_class_names()
    print(f"{len(class_names)} MID->name mappings loaded")

    downloaded = load_manifest()
    for s, ids in downloaded.items():
        print(f"downloaded[{s}] = {len(ids)}")

    audit = json.load(open(MIX / "oi_ext_hash_audit.json"))
    exact_dupes = set(audit["oi_ext_exact_dupes_excluded"])
    print(f"excluding {len(exact_dupes)} exact-hash eval duplicates")

    vrd_files = {"train": "oi_v6_train_vrd.csv", "val": "oi_v6_val_vrd.csv",
                 "test": "oi_v6_test_vrd.csv"}

    w = CocoSGGWriter()
    n_img_total = n_missing_file = n_bad_dim = 0

    for split, fn in vrd_files.items():
        rows_by_img: dict[str, list] = defaultdict(list)
        with open(SCRATCH / fn, newline="") as f:
            for row in csv.DictReader(f):
                if row["RelationshipLabel"] == "is":
                    continue
                iid = row["ImageID"]
                if iid not in downloaded[split] or iid in exact_dupes:
                    continue
                rows_by_img[iid].append(row)

        print(f"[{split}] {len(rows_by_img)} candidate images "
              f"(>=1 qualifying row after filtering)")

        n_img_split = 0
        for iid, rows in rows_by_img.items():
            path = OI / split / f"{iid}.jpg"
            try:
                with Image.open(path) as im:
                    W, H = im.size
            except Exception:
                n_missing_file += 1
                continue
            if W <= 0 or H <= 0:
                n_bad_dim += 1
                continue

            boxes, cats, rels = [], [], []
            key_to_idx: dict[tuple, int] = {}

            def bidx(mid: str, xmin: float, xmax: float, ymin: float, ymax: float) -> int:
                x = xmin * W
                y = ymin * H
                bw = max((xmax - xmin) * W, 1.0)
                bh = max((ymax - ymin) * H, 1.0)
                key = (round(x), round(y), round(x + bw), round(y + bh))
                if key not in key_to_idx:
                    key_to_idx[key] = len(boxes)
                    boxes.append([x, y, bw, bh])
                    cats.append(class_names.get(mid, mid))
                return key_to_idx[key]

            for r in rows:
                si = bidx(r["LabelName1"], float(r["XMin1"]), float(r["XMax1"]),
                          float(r["YMin1"]), float(r["YMax1"]))
                oi = bidx(r["LabelName2"], float(r["XMin2"]), float(r["XMax2"]),
                          float(r["YMin2"]), float(r["YMax2"]))
                rels.append((si, oi, r["RelationshipLabel"]))

            img_id = n_img_total  # sequential; this pack is self-contained
            before = len(w.images)
            w.add_image(img_id, f"{split}/{iid}.jpg", W, H, boxes, cats, rels)
            n_img_total += 1
            n_img_split += len(w.images) - before

        print(f"[{split}] {n_img_split} images added")

    out_path = MIX / "openimages_ext_train_coco.json"
    w.write(out_path)
    print(f"skipped: {n_missing_file} unreadable/missing files, "
          f"{n_bad_dim} bad dimensions")


if __name__ == "__main__":
    main()
