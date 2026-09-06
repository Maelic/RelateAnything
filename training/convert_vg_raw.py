"""Convert RAW Visual Genome relationships.json into a dedup-filtered COCO-SGG json.

Why raw VG rather than GQA: GQA's scene graphs ARE these same VG annotations,
normalized down to 307 predicate classes. Raw VG keeps all 36,549 surface
forms, and this model is built to exploit surface-form diversity (the
synonym-preserving policy) — so GQA's normalization destroys exactly the
signal we want. Raw VG also yields more leak-free usable images (41k vs 33k)
and 85.9% of its object names already sit in the union taxonomy.

Filtering, in order:
  1. image must be physically on disk (VG150_coco_format/train),
  2. image must NOT be eval-protected — routed through load_registry(), the
     same path every other datamix source uses (note: `pack_megasg.py --preset
     vg150` does NOT apply this, which is why runs/packed/vg150/train is 43%
     eval-protected and must never be trained on),
  3. predicates and object names longer than --max_words (default 5) are
     dropped, matching datagen/sgg_postprocess.py:88's run-on filter used for
     MEGASG.

Predicate strings are otherwise kept verbatim (lowercased/stripped only) — the
synonym-preserving policy applies to new sources too.

    python training/convert_vg_raw.py
    python training/pack_megasg.py --name vg_raw --splits train \
        --train_ann runs/datamix/vg_raw_train_coco.json \
        --train_img_dir $RA_DATASETS/VG150_coco_format/train \
        --out runs/packed/vg_raw
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.convert_datamix import (CocoSGGWriter,  # noqa: E402
                                      disk_ids, load_registry)
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

VG_META = DATASETS / "VG_metadata"
IMG_ROOT = DATASETS / "VG150_coco_format/train"


def obj_name(o: dict) -> str:
    """VG is inconsistent: subjects carry `name` (str), objects carry `names`
    (list). Accept either."""
    n = o.get("name")
    if not n:
        names = o.get("names") or []
        n = names[0] if names else ""
    return " ".join(str(n).strip().lower().split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_words", type=int, default=5,
                    help="Drop predicates / object names with more than this "
                         "many words (matches the MEGASG run-on filter).")
    ap.add_argument("--out", default=str(DATAMIX / "vg_raw_train_coco.json"))
    a = ap.parse_args()

    prot_coco, prot_vg, vg2coco, _ = load_registry()
    allowed = disk_ids(IMG_ROOT)
    print(f"[vg_raw] images on disk: {len(allowed):,} | protected vg ids: "
          f"{len(prot_vg):,} | banned coco ids: {len(prot_coco):,}")

    # width/height are not in relationships.json — they live in image_data.json
    dims = {str(im["image_id"]): (im["width"], im["height"])
            for im in json.load(open(VG_META / "image_data.json"))}
    print(f"[vg_raw] image_data dims: {len(dims):,}")

    data = json.load(open(VG_META / "relationships.json"))
    print(f"[vg_raw] relationships.json: {len(data):,} image records")

    w = CocoSGGWriter()
    drop = Counter()
    for rec in data:
        vid = str(rec["image_id"])
        if vid not in allowed:
            drop["image_not_on_disk"] += 1
            continue
        if vid in prot_vg or (vid in vg2coco and vg2coco[vid] in prot_coco):
            drop["eval_protected"] += 1
            continue
        wh = dims.get(vid)
        if wh is None:
            drop["no_dims"] += 1
            continue

        box_of: dict[int, int] = {}     # VG object_id -> local box index
        boxes: list[list[float]] = []
        cats: list[str] = []
        rels: list[tuple[int, int, str]] = []

        def bidx(o: dict) -> int | None:
            name = obj_name(o)
            if not name:
                drop["obj_no_name"] += 1
                return None
            if len(name.split()) > a.max_words:
                drop["obj_name_too_long"] += 1
                return None
            oid = o.get("object_id")
            if oid in box_of:
                return box_of[oid]
            box_of[oid] = len(boxes)
            boxes.append([float(o["x"]), float(o["y"]),
                          max(float(o["w"]), 1.0), max(float(o["h"]), 1.0)])
            cats.append(name)
            return box_of[oid]

        for r in rec.get("relationships") or []:
            p = " ".join(str(r.get("predicate") or "").strip().lower().split())
            if not p:
                drop["pred_empty"] += 1
                continue
            if len(p.split()) > a.max_words:
                drop["pred_too_long"] += 1
                continue
            si = bidx(r["subject"])
            oi = bidx(r["object"])
            if si is None or oi is None or si == oi:
                continue
            rels.append((si, oi, p))

        if len(boxes) < 2 or not rels:
            drop["image_no_usable_rels"] += 1
            continue
        w.add_image(int(vid), f"{vid}.jpg", wh[0], wh[1], boxes, cats, rels)

    print("[vg_raw] drops: " + ", ".join(f"{k}={v:,}" for k, v in drop.most_common()))
    w.write(Path(a.out))

    # Leakage assertion — cheap, and the single most damaging thing to get wrong.
    kept = {str(im["id"]) for im in w.images}
    bad = {v for v in kept
           if v in prot_vg or (v in vg2coco and vg2coco[v] in prot_coco)}
    assert not bad, f"LEAK: {len(bad)} eval-protected images survived: {list(bad)[:5]}"
    print(f"[vg_raw] leakage check PASSED — 0 of {len(kept):,} kept images are "
          f"eval-protected")


if __name__ == "__main__":
    main()
