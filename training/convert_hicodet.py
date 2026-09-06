"""Convert the HF `zhimeng/hico_det` TEST parquets into (a) a COCO-SGG json for
pack_megasg.py and (b) a Haystack-style explicit-negatives json for fAP.

DECODED SCHEMA (verified against list_action.csv):
- `objects[].id` is the 1-BASED HOI-CATEGORY id (row of list_action.csv), NOT an
  instance id: each entry is one annotated instance of that (object, verb)
  category with its human/object boxes (id 246 = bench/sit_on, 132 = horse/hold).
- positive/negative/ambiguous_objects are the same category ids 0-based, parallel
  to the caption tuples. NEGATIVE captions carry no boxes — they are IMAGE-LEVEL
  statements ("nobody inspects a bench here").
- 80 of the 600 categories are `no_interaction`: gold pair-level negatives.

Mapping:
- POSITIVES: visible entries with a real verb -> (human box, object box, gerund
  predicate). Predicate string = vname_ing with underscores -> spaces.
- NEGATIVES json (federated, Haystack format):
  (a) each image-level negative caption (obj, action) expands to every
      (human box, box of category obj) pair in the image;
  (b) each visible `no_interaction` entry contributes its OWN pair x every real
      verb the 600-list defines for that object class.
  Cells that coincide with a positive or an ambiguous caption are dropped.
- AMBIGUOUS captions are excluded from both sides (LVIS-style unlabelled).
- Boxes come as [x1, x2, y1, y2] (original.mat convention) -> COCO xywh.
- invis entries skipped; per-image cap 40 boxes = packer max_objects.
- V2 (2026-08-26, --iou_merge, default 0.5): HICO-DET stores one human box and one
  object box PER HOI INSTANCE, so the same bicycle appears 2-4x with slightly
  different coordinates. v1 deduped on exact coordinates only -> 37 % of test
  boxes / 34 % of train boxes were IoU>=0.7 same-class duplicates, which
  depressed HICO precision (.32 ->.47 when identities are merged) and made the
  same-class distractor test read as chance. Now same-category boxes with
  IoU >= --iou_merge are merged (union-find, mean box); the threshold comes from
  PSG, where genuinely distinct same-class instances exceed IoU 0.5 in 0.2 % of
  pairs, and equals HICO-DET's own matching criterion. Pairs whose two ends
  merge into one box (self-loops) are dropped and counted. --iou_merge 0
  reproduces v1.

    python training/convert_hicodet.py
    python training/pack_megasg.py --train_ann /dev/null \
        --val_ann $RA_DATASETS/HICO_DET/hicodet_test_coco.json \
        --val_img_dir $RA_DATASETS/HICO_DET/test_images \
        --out runs/packed/hicodet --splits val
    mv runs/packed/hicodet/val runs/packed/hicodet/test
"""
from __future__ import annotations

import ast
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import sys
import pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).
SRC = DATASETS / "HICO_DET"
# --split train: the 13 train parquets live under train_dl/data/ (HF
# snapshot_download of zhimeng/hico_det, 2026-08-21). Train is NEVER an eval
# split here — it is the GT-scored testbed for the model-in-the-loop pilot and
# a candidate verb-supervision source; HICO test stays zero-shot.
SPLIT = (sys.argv[sys.argv.index("--split") + 1]
         if "--split" in sys.argv else "test")
IOU_MERGE = (float(sys.argv[sys.argv.index("--iou_merge") + 1])
             if "--iou_merge" in sys.argv else 0.5)
PARQ_DIR = SRC / "train_dl" / "data" if SPLIT == "train" else SRC
IMG_DIR = SRC / f"{SPLIT}_images"


def _iou_xywh(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    iw = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    ih = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = iw * ih
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def merge_boxes(local_boxes, thr):
    """Union-find merge of same-category boxes with IoU >= thr.
    Returns (merged_boxes, remap old_idx -> new_idx). Representative = mean box
    of the members (what a detector would emit for the instance)."""
    n = len(local_boxes)
    parent = list(range(n))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for i in range(n):
        for j in range(i + 1, n):
            if local_boxes[i][1] == local_boxes[j][1] and \
                    _iou_xywh(local_boxes[i][0], local_boxes[j][0]) >= thr:
                parent[find(j)] = find(i)
    groups: dict = {}
    remap = [0] * n
    merged = []
    for i in range(n):
        r = find(i)
        if r not in groups:
            groups[r] = len(merged)
            merged.append([[0.0, 0.0, 0.0, 0.0], local_boxes[i][1], 0])
        g = groups[r]
        remap[i] = g
        for d in range(4):
            merged[g][0][d] += local_boxes[i][0][d]
        merged[g][2] += 1
    out = [([v / m for v in bb], cat) for bb, cat, m in merged]
    return out, remap


def main() -> None:
    IMG_DIR.mkdir(exist_ok=True)
    acts = list(csv.DictReader(open(SRC / "list_action.csv")))
    hoi_obj = [r["nname"] for r in acts]
    hoi_verb = [r["vname"] for r in acts]
    hoi_pred = [r["vname_ing"].replace("_", " ") for r in acts]
    verbs_of_obj: dict = defaultdict(set)
    for r in acts:
        if r["vname"] != "no_interaction":
            verbs_of_obj[r["nname"]].add(r["vname_ing"].replace("_", " "))

    images, annotations, rel_annotations = [], [], []
    negatives_by_img: dict = {}
    cat_ids: dict = {"person": 1}
    pred_ids: dict = {}
    ann_gid = img_id = 0
    n_pos = n_neg_a = n_neg_b = n_invis = 0
    n_box_raw = n_box_merged = n_selfloop = n_pos_dup = 0

    for shard in sorted(PARQ_DIR.glob(f"{SPLIT}-*.parquet")):
        t = pq.ParquetFile(shard)
        for rg in range(t.num_row_groups):
            batch = t.read_row_group(rg).to_pydict()
            for k in range(len(batch["image"])):
                row = {c: batch[c][k] for c in batch}
                for c in ["objects", "positive_captions", "negative_captions",
                          "ambiguous_captions"]:
                    if isinstance(row[c], str):
                        row[c] = ast.literal_eval(row[c])

                im = Image.open(io.BytesIO(row["image"]["bytes"]))
                W, H = im.size
                fname = Path(row["image"]["path"]
                             or f"hico_{SPLIT}_{img_id:06d}.jpg").name
                fp = IMG_DIR / fname
                if not fp.exists():
                    im.convert("RGB").save(fp, quality=95)

                box_key_to_local: dict = {}
                local_boxes: list = []

                def add_box(b, cat):
                    x1, x2, y1, y2 = [float(v) for v in b]
                    assert x2 >= x1 and y2 >= y1, (fname, b)
                    key = (round(x1), round(x2), round(y1), round(y2))
                    if key in box_key_to_local:
                        return box_key_to_local[key]
                    li = len(local_boxes)
                    box_key_to_local[key] = li
                    local_boxes.append(([x1, y1, x2 - x1, y2 - y1], cat))
                    return li

                pos, noint_pairs = [], []
                humans: set = set()
                obj_boxes_of: dict = defaultdict(set)
                for e in row["objects"]:
                    cid = int(e["id"])
                    if not (1 <= cid <= 600) or e.get("invis"):
                        n_invis += 1
                        continue
                    oname, vname = hoi_obj[cid - 1], hoi_verb[cid - 1]
                    h = add_box(e["bbox_human"], "person")
                    o = add_box(e["bbox_object"], oname)
                    humans.add(h)
                    obj_boxes_of[oname].add(o)
                    if vname == "no_interaction":
                        noint_pairs.append((h, o, oname))
                    else:
                        pos.append((h, o, hoi_pred[cid - 1]))

                # ---- V2: merge near-duplicate same-category boxes, remap
                n_box_raw += len(local_boxes)
                if IOU_MERGE > 0 and len(local_boxes) > 1:
                    local_boxes, remap = merge_boxes(local_boxes, IOU_MERGE)
                    n_box_merged += len(local_boxes)
                    _pos, _seen = [], set()
                    for h, o, pr in pos:
                        trip = (remap[h], remap[o], pr)
                        if trip[0] == trip[1]:
                            n_selfloop += 1
                            continue
                        if trip in _seen:
                            n_pos_dup += 1
                            continue
                        _seen.add(trip)
                        _pos.append(trip)
                    pos = _pos
                    _ni, _seen = [], set()
                    for h, o, on in noint_pairs:
                        trip = (remap[h], remap[o], on)
                        if trip[0] != trip[1] and trip not in _seen:
                            _seen.add(trip)
                            _ni.append(trip)
                    noint_pairs = _ni
                    humans = {remap[h] for h in humans}
                    obj_boxes_of = defaultdict(set, {k: {remap[o] for o in v}
                                                     for k, v in obj_boxes_of.items()})
                else:
                    n_box_merged += len(local_boxes)

                pos_cells = set(pos)
                amb = {(o_, a_.replace("_", " "))
                       for o_, a_ in row["ambiguous_captions"]}
                neg = set()
                for oname, action in row["negative_captions"]:
                    if action == "no_interaction":
                        continue
                    pred = action.replace("_", " ")
                    # gerund lookup via the object's verb table
                    for r in acts:
                        if r["nname"] == oname and r["vname"] == action:
                            pred = r["vname_ing"].replace("_", " ")
                            break
                    if (oname, pred) in amb:
                        continue
                    for h in humans:
                        for o in obj_boxes_of.get(oname, ()):
                            if (h, o, pred) not in pos_cells:
                                neg.add((h, o, pred))
                n_neg_a += len(neg)
                for h, o, oname in noint_pairs:
                    for pred in verbs_of_obj[oname]:
                        if (h, o, pred) not in pos_cells \
                                and (oname, pred) not in amb:
                            neg.add((h, o, pred))
                            n_neg_b += 1

                if not pos and not neg:
                    continue
                images.append({"id": img_id, "file_name": fname,
                               "width": W, "height": H})
                gid_of_local = {}
                for li, (bbox, cat) in enumerate(local_boxes[:40]):
                    cid = cat_ids.setdefault(cat, len(cat_ids) + 1)
                    gid_of_local[li] = ann_gid
                    annotations.append({"id": ann_gid, "image_id": img_id,
                                        "bbox": bbox, "category_id": cid})
                    ann_gid += 1
                for h, o, p in pos:
                    if h not in gid_of_local or o not in gid_of_local:
                        continue
                    pid = pred_ids.setdefault(p, len(pred_ids))
                    rel_annotations.append({"image_id": img_id,
                                            "subject_id": gid_of_local[h],
                                            "object_id": gid_of_local[o],
                                            "predicate_id": pid})
                    n_pos += 1
                nn = [[h, o, p] for h, o, p in sorted(neg)
                      if h in gid_of_local and o in gid_of_local and h != o]
                if nn:
                    # keyed by the HICO file NUMBER: eval_haystack joins on
                    # int(stem.split("_")[-1]) of the pack's file names, which
                    # survives images the packer drops (the running img_id does
                    # not — the 08-21 train-split edit had switched to it).
                    negatives_by_img[str(int(Path(fname).stem.split("_")[-1]))] = nn
                img_id += 1

    for p in {p for v in negatives_by_img.values() for _, _, p in v}:
        pred_ids.setdefault(p, len(pred_ids))

    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i, "name": n} for n, i in cat_ids.items()],
            "rel_categories": [{"id": i, "name": n}
                               for n, i in sorted(pred_ids.items(),
                                                  key=lambda t: t[1])],
            "rel_annotations": rel_annotations}
    _sfx = "" if SPLIT == "test" else f"_{SPLIT}"
    json.dump(coco, open(SRC / f"hicodet_{SPLIT}_coco.json", "w"))
    json.dump({"predicates": sorted(pred_ids), "by_image_id": negatives_by_img},
              open(DATAMIX / f"hicodet_negatives{_sfx}.json", "w"))
    n_neg = sum(len(v) for v in negatives_by_img.values())
    print(f"{img_id:,} images  {len(annotations):,} boxes  {n_pos:,} positives  "
          f"{n_neg:,} federated negative cells (caption-expanded {n_neg_a:,}, "
          f"no_interaction-derived {n_neg_b:,})  invalid/invis {n_invis:,}")
    print(f"categories {len(cat_ids)}  verb predicates {len(pred_ids)}")
    print(f"iou_merge={IOU_MERGE}: boxes {n_box_raw:,} -> {n_box_merged:,} "
          f"({n_box_raw - n_box_merged:,} merged); positives collapsed as exact "
          f"duplicates {n_pos_dup:,}; self-loop pairs dropped {n_selfloop:,}")


if __name__ == "__main__":
    main()
