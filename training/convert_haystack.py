"""Convert the Haystack dataset into a COCO-SGG json + a NEGATIVES sidecar.

Haystack (Lorenz et al., "Haystack: A Panoptic Scene Graph Dataset to Evaluate
Rare Predicate Classes") is the only benchmark we have with EXPLICIT negative
relation annotations. Everywhere else, an unlisted relation is merely
unlabelled, so false positives on rare predicates are invisible.

Two properties drive this converter:

1. It is FEDERATED at the (pair, predicate) level, exactly like LVIS is at the
   (image, category) level. A pair carries a mean of 1.49 labelled predicates
   out of 56 (2.7%); everything else is unlabelled and must NOT be scored.
   Hence the negatives go to a sidecar rather than into the pack: the pack
   format has no cell-level "unknown" state.

2. Its images are SA-1B, not COCO or VG, so there is no overlap with anything
   we train on — no exclusion registry needed here (unlike every other source;
   see training/convert_vg_raw.py). Its predicate list is byte-identical to
   runs/packed/psg's 56, and its taxonomy is PSG's 133 (80 thing + 53 stuff).

MOST IMAGES HAVE NO POSITIVE RELATIONS (9,151 of 11,368) — they exist to carry
negatives. Pack with --min_rels 0 or they are silently dropped and their
negatives become unscoreable.

    python training/convert_haystack.py
    python training/pack_megasg.py --name haystack --splits test \
        --test_ann runs/datamix/haystack_test_coco.json \
        --test_img_dir $RA_DATASETS/Haystack/images \
        --min_rels 0 --max_objects 100 --out runs/packed/haystack
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.convert_datamix import CocoSGGWriter  # noqa: E402
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

HAY = DATASETS / "Haystack"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default=str(HAY / "haystack_v1/annotation_v1.json"))
    ap.add_argument("--img_root", default=str(HAY / "images"))
    ap.add_argument("--out", default=str(DATAMIX / "haystack_test_coco.json"))
    ap.add_argument("--neg_out", default=str(DATAMIX / "haystack_negatives.json"))
    a = ap.parse_args()

    d = json.load(open(a.ann))
    preds = d["predicate_classes"]
    cats = d["thing_classes"] + d["stuff_classes"]
    print(f"[haystack] {len(d['data']):,} records | {len(preds)} predicates | "
          f"{len(cats)} categories")

    w = CocoSGGWriter()
    negatives: dict[str, list] = {}
    drop = Counter()
    n_pos = n_neg = 0
    for r in d["data"]:
        iid = int(r["image_id"])
        anns = r["annotations"]
        if len(anns) < 2:
            drop["fewer_than_2_boxes"] += 1
            continue
        if not os.path.exists(os.path.join(a.img_root, r["file_name"])):
            drop["image_not_on_disk"] += 1
            continue
        # detectron2 BoxMode 0 == XYXY_ABS; the writer wants COCO xywh.
        boxes, names = [], []
        for o in anns:
            x1, y1, x2, y2 = (float(v) for v in o["bbox"])
            assert o.get("bbox_mode", 0) == 0, f"unexpected bbox_mode {o.get('bbox_mode')}"
            boxes.append([x1, y1, max(x2 - x1, 1.0), max(y2 - y1, 1.0)])
            names.append(cats[int(o["category_id"])])

        pos = [(int(s), int(o), preds[int(p)]) for s, o, p in r["relations"]]
        # Negatives carry predicate NAMES, not Haystack's ids: the pack
        # assigns its own predicate order (first-appearance), so joining by
        # id would silently mislabel every negative cell.
        neg = [[int(s), int(o), preds[int(p)]] for s, o, p in r["neg_relations"]]
        if not pos and not neg:
            drop["no_labelled_cells"] += 1
            continue
        n_pos += len(pos); n_neg += len(neg)
        # allow_empty_rels: 9,151 images have ONLY negatives; they must survive
        # into the pack or their negatives can never be scored. Pair it with
        # --min_rels 0 at pack time.
        w.add_image(iid, r["file_name"], int(r["width"]), int(r["height"]),
                    boxes, names, pos, allow_empty_rels=True)
        if neg:
            negatives[str(iid)] = neg

    print("[haystack] drops: " + (", ".join(f"{k}={v:,}" for k, v in drop.most_common())
                                  or "none"))
    print(f"[haystack] kept {len(w.images):,} images | {n_pos:,} positive cells | "
          f"{n_neg:,} negative cells ({n_neg / max(n_pos, 1):.1f}x)")
    w.write(Path(a.out))
    Path(a.neg_out).write_text(json.dumps(
        {"predicates": preds, "by_image_id": negatives}))
    print(f"[haystack] negatives → {a.neg_out} ({len(negatives):,} images)")


if __name__ == "__main__":
    main()
