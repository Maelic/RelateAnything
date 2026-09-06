#!/usr/bin/env python3
"""
Audit (measure, don't filter) the geometric plausibility of contact/attention
predicates in a generated run, using the COCO bboxes.

This is a QUALITY METRIC, not a cleanup step — use it to compare prompts:
a lower "disjoint wearing" rate means the model is grounding the relation in
actual pixels instead of riding the label prior.

Usage:
  python datagen/audit_contact_geometry.py --run <run_dir> [--limit 10000]
  python datagen/audit_contact_geometry.py --run A --run B   # side-by-side
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
import os
from pathlib import Path

# Override with MEGASG_COCO; the default assumes the dataset root layout
# documented in docs/data.md.
COCO = Path(os.environ.get("MEGASG_COCO",
                           "datasets/MEGASG/train/_annotations.coco.json"))

# predicates whose object should be CONTAINED in the subject (worn / part of)
CONTAINMENT = {"wearing", "part of"}
# predicates that require the two boxes to at least TOUCH (overlap)
CONTACT = {"holding", "carrying", "riding", "sitting on", "lying on", "standing on",
           "resting on", "leaning against", "mounted on", "eating", "drinking from",
           "attached to", "covering", "hanging from"}
ATTENTION = {"looking at", "watching", "pointing at"}


def _boxes_by_image(coco: Path = None) -> dict[int, dict[int, tuple]]:
    data = json.load(open(coco or COCO))
    by_img = defaultdict(list)
    for a in data["annotations"]:
        by_img[a["image_id"]].append(a)
    out = {}
    for iid, anns in by_img.items():
        s = sorted(anns, key=lambda a: a["id"])
        out[iid] = {i + 1: tuple(a["bbox"]) for i, a in enumerate(s)}
    return out


def contain_frac(inner, outer) -> float:
    ix, iy, iw, ih = inner; ox, oy, ow, oh = outer
    x1, y1 = max(ix, ox), max(iy, oy)
    x2, y2 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return (x2 - x1) * (y2 - y1) / (iw * ih) if iw * ih else 0.0


def iou(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2]); y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    return inter / (b1[2] * b1[3] + b2[2] * b2[3] - inter)


def audit(run: Path, boxes: dict, limit: int) -> dict:
    cont = Counter(); cont_tot = 0
    contact = Counter(); contact_tot = 0
    attn = Counter(); attn_tot = 0
    n_seen = 0
    for shard in sorted(run.glob("shard_*.jsonl")):        # audit ALL shards
        if n_seen >= limit:
            break
        with open(shard) as f:
            for line in f:
                if n_seen >= limit:
                    break
                n_seen += 1
                r = json.loads(line)
                bx = boxes.get(r["img_id"], {})
                for x in r["relations"]:
                    s, o, p = x["subject_id"], x["object_id"], x["predicate"]
                    if s not in bx or o not in bx:
                        continue
                    sb, ob = bx[s], bx[o]
                    if p in CONTAINMENT:
                        cont_tot += 1
                        cf = contain_frac(ob, sb)
                        cont["ok>=0.7" if cf >= 0.7 else
                             "partial0.3-0.7" if cf >= 0.3 else
                             "touch<0.3" if cf > 0 else "DISJOINT(wrong)"] += 1
                    elif p in CONTACT:
                        contact_tot += 1
                        contact["overlap" if iou(sb, ob) > 0 else "DISJOINT(suspect)"] += 1
                    elif p in ATTENTION:
                        attn_tot += 1
                        attn["overlap" if iou(sb, ob) > 0 else "disjoint(unverifiable)"] += 1
    return dict(cont=cont, cont_tot=cont_tot, contact=contact, contact_tot=contact_tot,
                attn=attn, attn_tot=attn_tot)


def show(name: str, a: dict):
    print(f"\n### {name}")
    ct = a["cont_tot"]
    print(f"  CONTAINMENT (wearing/part of)  n={ct}")
    for k in ["ok>=0.7", "partial0.3-0.7", "touch<0.3", "DISJOINT(wrong)"]:
        v = a["cont"].get(k, 0)
        print(f"    {v:6,} ({100*v/max(ct,1):5.1f}%)  {k}")
    cc = a["contact_tot"]
    print(f"  CONTACT (sitting on/holding/…)  n={cc}")
    for k in ["overlap", "DISJOINT(suspect)"]:
        v = a["contact"].get(k, 0)
        print(f"    {v:6,} ({100*v/max(cc,1):5.1f}%)  {k}")
    at = a["attn_tot"]
    print(f"  ATTENTION (looking at/…)  n={at}  [geometry can't verify gaze]")
    for k in ["overlap", "disjoint(unverifiable)"]:
        v = a["attn"].get(k, 0)
        print(f"    {v:6,} ({100*v/max(at,1):5.1f}%)  {k}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, type=Path,
                    help="run dir (repeat for side-by-side comparison)")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--coco", type=Path, default=None,
                    help="override the COCO annotation path (defaults to local, "
                         "then the cluster copy)")
    args = ap.parse_args()
    print("Loading COCO bboxes …")
    boxes = _boxes_by_image(args.coco)
    for run in args.run:
        show(run.name, audit(run, boxes, args.limit))


if __name__ == "__main__":
    main()
