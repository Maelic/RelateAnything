"""Convert SpatialSense's held-out VALID+TEST splits into packs + labelled cells.

SpatialSense (Yang, Russakovsky & Deng, ICCV'19) is the one benchmark built so
that spatial relations cannot be guessed from priors: annotators were shown an
image and asked to produce relations a model would get WRONG, giving verified
NEGATIVE (subject, predicate, object) triples. The test split is exactly
balanced (1,379 true / 1,379 false), so chance is 50% and language/frequency
priors buy nothing.

WHY THIS IS A CLEAN ZERO-SHOT PROBE:
  - The released models train on megasg_clean, vg_raw and hicodet. None of
    those packs contains a SpatialSense image (0/5,976 train, 0/1,126 valid,
    0/1,920 test by filename).
  - Test images do not appear in SpatialSense's own train or valid split, so
    the valid split is an uncontaminated set for choosing the decision
    threshold, which is their protocol and what makes the accuracy comparable
    to their published numbers.
  (`training/convert_spatialsense.py` writes their train split into the mixture
  for a training arm that no released model uses.)

Emits per split:
  DATASETS/SpatialSense/spatialsense_<split>_coco.json  — images, deduped boxes,
      rel_annotations = the label=true cells (so packers/recall evals work)
  runs/datamix/spatialsense_<split>_cells.json — {"predicates": [...],
      "by_image_id": {img_id: [[sub_local, obj_local, pred_id, label],...]}}
      carrying BOTH labels, which is what eval_spatialsense.py scores.

Source bbox convention is [y0, y1, x0, x1] in pixels -> COCO xywh.

    python training/convert_spatialsense_test.py
    python training/pack_megasg.py --train_ann /dev/null \
        --val_ann.../spatialsense_test_coco.json \
        --val_img_dir.../SpatialSense/images --out runs/packed/spatialsense_test \
        --splits val --min_rels 0
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).
SS = DATASETS / "SpatialSense"
MIX = DATAMIX


def main() -> None:
    recs = [json.loads(l) for l in open(MIX / "spatialsense_negatives.jsonl")]
    by_split = defaultdict(list)
    for r in recs:
        by_split[r["split"]].append(r)

    for split in ("valid", "test"):
        rows = by_split[split]
        per_img = defaultdict(list)
        for r in rows:
            per_img[r["file_name"]].append(r)

        images, annotations, rel_annotations = [], [], []
        cells_by_img: dict = {}
        cat_ids: dict = {}
        pred_ids: dict = {}
        ann_gid = 0
        for img_id, (fname, anns) in enumerate(sorted(per_img.items())):
            W, H = anns[0]["width"], anns[0]["height"]
            local, key_to_local = [], {}

            def add_box(ent):
                y0, y1, x0, x1 = [float(v) for v in ent["bbox_yyxx"]]
                key = (round(x0), round(y0), round(x1), round(y1), ent["name"])
                if key in key_to_local:
                    return key_to_local[key]
                i = len(local)
                key_to_local[key] = i
                local.append(([x0, y0, x1 - x0, y1 - y0], ent["name"]))
                return i

            cells = []
            for a in anns:
                s = add_box(a["subject"])
                o = add_box(a["object"])
                p = pred_ids.setdefault(a["predicate"], len(pred_ids))
                cells.append([s, o, p, int(bool(a["label"]))])

            images.append({"id": img_id, "file_name": fname,
                           "width": W, "height": H})
            gid_of = {}
            for i, (bbox, cat) in enumerate(local):
                cid = cat_ids.setdefault(cat, len(cat_ids) + 1)
                gid_of[i] = ann_gid
                annotations.append({"id": ann_gid, "image_id": img_id,
                                    "bbox": bbox, "category_id": cid})
                ann_gid += 1
            for s, o, p, lab in cells:
                if lab:
                    rel_annotations.append({"image_id": img_id,
                                            "subject_id": gid_of[s],
                                            "object_id": gid_of[o],
                                            "predicate_id": p})
            cells_by_img[str(img_id)] = cells

        coco = {"images": images, "annotations": annotations,
                "categories": [{"id": i, "name": n} for n, i in cat_ids.items()],
                "rel_categories": [{"id": i, "name": n} for n, i in
                                   sorted(pred_ids.items(), key=lambda t: t[1])],
                "rel_annotations": rel_annotations}
        json.dump(coco, open(SS / f"spatialsense_{split}_coco.json", "w"))
        json.dump({"predicates": [n for n, _ in sorted(pred_ids.items(),
                                                       key=lambda t: t[1])],
                   "by_image_id": cells_by_img},
                  open(MIX / f"spatialsense_{split}_cells.json", "w"))
        n_cells = sum(len(v) for v in cells_by_img.values())
        n_pos = sum(c[3] for v in cells_by_img.values() for c in v)
        print(f"{split:6s}: {len(images):,} images  {len(annotations):,} boxes  "
              f"{n_cells:,} cells ({n_pos:,} true / {n_cells - n_pos:,} false)  "
              f"{len(pred_ids)} predicates  {len(cat_ids):,} categories")


if __name__ == "__main__":
    main()
