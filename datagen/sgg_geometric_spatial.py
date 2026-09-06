#!/usr/bin/env python3
"""
sgg_geometric_spatial.py — deterministic spatial-layout layer from bounding boxes.

Rationale (see memory/sgg-spatial-pass-findings): the LLM spatial pass is ~94%
consistent with the 2D-centroid convention but pays for vacuous body-part spatial,
a depth-vs-height confound, same-label floods, and inverse dups — and a 26B model
isn't actually better than geometry on the box-computable axes. This module
produces the spatial layer directly from the COCO boxes:

  inside          containment (A mostly within B)
  to the left/right of   horizontal dominant, depth-independent (most reliable)
  above / below   vertical stack (horizontal overlap), with a depth-confound guard
  in front of / behind   occlusion: overlapping boxes, nearer = lower bottom edge / larger

One canonical relation per unordered pair (left/upper/front/contained object is the
subject), so inverse duplicates are impossible by construction. Body-part and
optional same-label pairs are filtered. Density is controllable.

Use as a library  ── pair_relation(boxA, boxB) / spatial_for_image(objs)
or as a CLI        ── writes a spatial-layer jsonl alongside a run for comparison.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
import os
from pathlib import Path

# ── tunable thresholds ─────────────────────────────────────────────────────────
CONTAIN_T   = 0.60   # frac of inner box inside outer to call "inside"
OCC_IOU     = 0.10   # min IoU for an occlusion (in front of / behind) judgment
H_STACK_T   = 0.25   # min horizontal overlap to treat a vertical pair as a real stack
DEPTH_RATIO = 0.60   # higher box smaller than this * lower box area ⇒ it's farther (behind)
BOT_EPS     = 0.02   # bottom-edge tie tolerance (frac of image)

# body-part labels — their spatial arrangement is anatomically fixed (vacuous)
PARTS = frozenset({
    "human face", "human hair", "human hand", "human arm", "human head",
    "human body", "human foot", "human leg", "human nose", "human eye",
    "human mouth", "human ear", "face", "hair", "human hand", "human beard",
})


# ── box helpers ────────────────────────────────────────────────────────────────
def _area(b):  return b[2] * b[3]
def _ctr(b):   return (b[0] + b[2] / 2, b[1] + b[3] / 2)
def _bottom(b): return b[1] + b[3]

def _contain_frac(inner, outer):
    ix, iy, iw, ih = inner; ox, oy, ow, oh = outer
    x1, y1 = max(ix, ox), max(iy, oy)
    x2, y2 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / (iw * ih) if iw * ih else 0.0

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2]); y2 = min(a[1] + a[3], b[1] + b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)

def _h_overlap(a, b):
    x1 = max(a[0], b[0]); x2 = min(a[0] + a[2], b[0] + b[2])
    if x2 <= x1:
        return 0.0
    return (x2 - x1) / min(a[2], b[2])


def pair_relation(A, B, emit_inside=True):
    """Return (subject_is_A: bool, predicate) for the single canonical spatial
    relation between boxes A and B, or None if no clear relation.

    emit_inside=False: 2D box containment is a poor proxy for 3D containment
    (worn/held items and occluded objects sit "inside" a person's box) — with it
    off, contained pairs fall through to the occlusion branch (in front/behind),
    which IS what the 2D signal actually shows."""
    cfa, cfb = _contain_frac(A, B), _contain_frac(B, A)   # A in B, B in A
    if emit_inside and max(cfa, cfb) >= CONTAIN_T:
        return (True, "inside") if cfa >= cfb else (False, "inside")

    if _iou(A, B) >= OCC_IOU:                              # occlusion → depth
        ba, bb = _bottom(A), _bottom(B)
        if abs(ba - bb) > BOT_EPS * max(ba, bb, 1):
            front_is_A = ba > bb                           # lower edge = nearer
        else:
            front_is_A = _area(A) >= _area(B)              # tie → larger = nearer
        return (front_is_A, "in front of")

    ax, ay = _ctr(A); bx, by = _ctr(B)
    dx, dy = bx - ax, by - ay
    if abs(dx) >= abs(dy):                                 # horizontal dominant
        return (dx > 0, "to the left of")                 # dx>0 ⇒ A is left of B
    # vertical dominant — guard the depth confound
    higher_is_A = ay < by
    higher, lower = (A, B) if higher_is_A else (B, A)
    if _h_overlap(A, B) < H_STACK_T and _area(higher) < DEPTH_RATIO * _area(lower):
        return (not higher_is_A, "in front of")           # higher+smaller ⇒ farther; lower in front
    return (higher_is_A, "above")                         # genuine vertical stack


def spatial_for_image(objs, drop_parts=True, drop_same_label=False,
                      max_per_subject=0, emit_inside=True, skip_pairs=None):
    """objs: list of dicts with 1-based 'id', 'label', 'bbox' (x,y,w,h).
    Returns a list of relation dicts (subject_id/label, predicate, object_id/label,
    spatial=True, source='geometric').

    skip_pairs: optional set of unordered (id, id) tuples to leave out — e.g.
    pairs already covered by a semantic relation, whose layout is implied."""
    rels = []
    n = len(objs)
    per_subj = defaultdict(int)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = objs[i], objs[j]
            la, lb = a["label"], b["label"]
            if skip_pairs and ((a["id"], b["id"]) in skip_pairs
                               or (b["id"], a["id"]) in skip_pairs):
                continue
            if drop_parts and (la in PARTS or lb in PARTS):
                continue
            if drop_same_label and la == lb:
                continue
            res = pair_relation(tuple(a["bbox"]), tuple(b["bbox"]),
                                emit_inside=emit_inside)
            if res is None:
                continue
            subj_is_a, pred = res
            s, o = (a, b) if subj_is_a else (b, a)
            if max_per_subject and per_subj[s["id"]] >= max_per_subject:
                continue
            per_subj[s["id"]] += 1
            rels.append({
                "subject_id": s["id"], "subject_label": s["label"],
                "predicate": pred,
                "object_id": o["id"], "object_label": o["label"],
                "spatial": True, "source": "geometric",
            })
    return rels


# ── CLI: build a geometric spatial layer for the images in a run ───────────────
def _load_coco(coco_path):
    data = json.load(open(coco_path))
    cat = {c["id"]: c["name"] for c in data["categories"]}
    by_img = defaultdict(list)
    for a in data["annotations"]:
        by_img[a["image_id"]].append(a)
    objs_by_img = {}
    for iid, anns in by_img.items():
        s = sorted(anns, key=lambda a: a["id"])
        objs_by_img[iid] = [{"id": k + 1, "label": cat.get(a["category_id"], "?"),
                             "bbox": a["bbox"]} for k, a in enumerate(s)]
    return objs_by_img


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True,
                    help="run dir (reads shard_0000.jsonl for the image list)")
    ap.add_argument("--coco", type=Path,
                    default=Path(os.environ.get(
                        "MEGASG_COCO",
                        "runs/packed/megasg/train/_annotations.coco.json")))
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--keep_parts", action="store_true")
    ap.add_argument("--drop_same_label", action="store_true")
    ap.add_argument("--max_per_subject", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    print("Loading COCO …")
    objs_by_img = _load_coco(args.coco)
    out = args.out or (args.run / "geometric_spatial.jsonl")
    n_img = n_rel = 0
    with open(args.run / "shard_0000.jsonl") as f, open(out, "w") as w:
        for i, line in enumerate(f):
            if i >= args.limit:
                break
            r = json.loads(line)
            objs = objs_by_img.get(r["img_id"], [])
            rels = spatial_for_image(objs, drop_parts=not args.keep_parts,
                                     drop_same_label=args.drop_same_label,
                                     max_per_subject=args.max_per_subject)
            w.write(json.dumps({"img_id": r["img_id"], "file_name": r["file_name"],
                                "relations": rels}) + "\n")
            n_img += 1; n_rel += len(rels)
    print(f"  {n_img:,} images, {n_rel:,} spatial rels ({n_rel/max(n_img,1):.2f}/img)")
    print(f"  → {out}")


if __name__ == "__main__":
    _main()
