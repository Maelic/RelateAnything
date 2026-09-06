"""Does the TRAINING DATA support the priors we expect for `wearing`?

Two claims a human would call obvious, tested against the packed mixture the
shipped checkpoint actually trained on (megasg_clean 0.919 + vg_raw 0.081
per-image):

  P1 FUNCTIONALITY   one garment has exactly one wearer (fan-in == 1), and a
                     person wears several garments (fan-out >= 1 is fine).
  P2 ATTACHMENT      subject and object boxes touch. Measured three ways
                     because IoU alone understates it: a shirt inside a person
                     box has small IoU but containment ~1.0.

Both are measured on the rows the training loader really yields, i.e. AFTER
    nb = min(nb, max_objects)  and  keep = (sub < nb) & (obj < nb)
(data/relation_dataset.py:266,308) — a relation pointing past the box budget is
never seen, so counting it would describe a dataset nobody trained on.

Usage:
    python training/probe_wearing_prior.py                       # both packs
    python training/probe_wearing_prior.py --packs runs/packed/psg --split test
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def pair_geometry(sb: np.ndarray, ob: np.ndarray) -> dict:
    """IoU, containment of object in subject, and normalized center gap."""
    ix1 = np.maximum(sb[:, 0], ob[:, 0])
    iy1 = np.maximum(sb[:, 1], ob[:, 1])
    ix2 = np.minimum(sb[:, 2], ob[:, 2])
    iy2 = np.minimum(sb[:, 3], ob[:, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    sa = np.clip((sb[:, 2] - sb[:, 0]) * (sb[:, 3] - sb[:, 1]), 1e-9, None)
    oa = np.clip((ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1]), 1e-9, None)
    iou = inter / (sa + oa - inter)
    # gap = shortest distance between the two rectangles, 0 when they overlap,
    # in units of the image diagonal (boxes are normalized, so diag = sqrt(2)).
    dx = np.maximum(np.maximum(ob[:, 0] - sb[:, 2], sb[:, 0] - ob[:, 2]), 0)
    dy = np.maximum(np.maximum(ob[:, 1] - sb[:, 3], sb[:, 1] - ob[:, 3]), 0)
    gap = np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2.0)
    return {
        "iou": iou,
        "contain_obj": inter / oa,   # how much of the garment is in the person
        "contain_sub": inter / sa,
        "gap": gap,
        "obj_over_sub_area": oa / sa,
    }


def q(x: np.ndarray, name: str) -> str:
    if x.size == 0:
        return f"{name:>16s}: (empty)"
    p = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return (f"{name:>16s}: mean {x.mean():.4f} | p1 {p[0]:.4f} p5 {p[1]:.4f} "
            f"p25 {p[2]:.4f} p50 {p[3]:.4f} p75 {p[4]:.4f} p95 {p[5]:.4f} "
            f"p99 {p[6]:.4f}")


def analyse(root: Path, split: str, pattern: str, max_objects: int,
            exclude_ids: set | None, dump: Path | None) -> dict:
    d = root / split
    meta = json.load(open(d / "meta.json"))
    preds = meta["predicates"]
    cats = meta.get("categories", [])
    img_meta = np.load(d / "img_meta.npy")
    boxes = np.load(d / "boxes.npy", mmap_mode="r")
    box_cats = np.load(d / "box_cats.npy", mmap_mode="r")
    rels = np.load(d / "rels.npy", mmap_mode="r")
    file_names = json.load(open(d / "file_names.json"))

    rx = re.compile(pattern, re.I)
    target = [i for i, p in enumerate(preds) if rx.search(p)]
    target_set = set(target)
    counts = meta.get("predicate_counts", {})
    print(f"\n{'='*78}\n{root.name}/{split}  "
          f"{len(img_meta)} imgs, {len(rels)} rel rows, {len(preds)} predicates")
    print(f"matched /{pattern}/ -> {len(target)} surface forms:")
    for i in sorted(target, key=lambda j: -counts.get(preds[j], 0))[:25]:
        print(f"    {counts.get(preds[i], 0):>9,}  {preds[i]!r}")

    keep_img = np.ones(len(img_meta), dtype=bool)
    if exclude_ids:
        keep_img = np.array([Path(f).stem not in exclude_ids for f in file_names])
        print(f"  excluded {int((~keep_img).sum())} held-out images")

    # ---- walk images, applying the loader's own truncation ---------------
    sub_boxes, obj_boxes, sub_cats, obj_cats, pred_ids = [], [], [], [], []
    img_ids = []
    fanin = defaultdict(int)     # (img, obj) -> n distinct subjects
    fanout = defaultdict(int)    # (img, sub) -> n distinct objects
    n_rel_total = n_rel_kept = n_rel_trunc = 0
    n_wear_imgs = 0

    for i in range(len(img_meta)):
        if not keep_img[i]:
            continue
        b0, nb, r0, nr = (int(v) for v in img_meta[i][3:7])
        if nr == 0:
            continue
        r = np.asarray(rels[r0:r0 + nr])
        m = np.isin(r[:, 2], target)
        if not m.any():
            continue
        r = r[m]
        n_rel_total += len(r)
        nbk = min(nb, max_objects)
        ok = (r[:, 0] < nbk) & (r[:, 1] < nbk)
        n_rel_trunc += int((~ok).sum())
        r = r[ok]
        if not len(r):
            continue
        n_rel_kept += len(r)
        n_wear_imgs += 1
        bx = np.asarray(boxes[b0:b0 + nbk], dtype=np.float64)
        bc = np.asarray(box_cats[b0:b0 + nbk])
        # distinct (sub,obj) pairs — a pair carrying two synonym forms is ONE
        # edge in the scene, and counting it twice would fake a fan-in of 2.
        seen = set()
        for s, o, p in r[:,:3]:
            if (s, o) in seen:
                continue
            seen.add((s, o))
            fanin[(i, o)] += 1
            fanout[(i, s)] += 1
            sub_boxes.append(bx[s]); obj_boxes.append(bx[o])
            sub_cats.append(bc[s]); obj_cats.append(bc[o])
            pred_ids.append(p); img_ids.append(i)

    if not sub_boxes:
        print("  no wearing edges survive")
        return {}

    sb = cxcywh_to_xyxy(np.array(sub_boxes))
    ob = cxcywh_to_xyxy(np.array(obj_boxes))
    g = pair_geometry(sb, ob)
    n = len(sb)
    fi = np.array(list(fanin.values()))
    fo = np.array(list(fanout.values()))

    print(f"\n  rows matching  {n_rel_total:,}   "
          f"dropped by max_objects={max_objects}: {n_rel_trunc:,} "
          f"({100*n_rel_trunc/max(n_rel_total,1):.1f}%)   "
          f"kept {n_rel_kept:,} -> {n:,} distinct pairs in {n_wear_imgs:,} imgs")

    print(f"\n  --- P1 FUNCTIONALITY ---")
    print(f"  fan-in  (subjects per garment): mean {fi.mean():.3f}  "
          f"max {fi.max()}  ==1: {100*(fi==1).mean():.2f}%  "
          f">=2: {int((fi>=2).sum()):,} ({100*(fi>=2).mean():.2f}%)")
    print(f"  fan-out (garments per person):  mean {fo.mean():.3f}  "
          f"max {fo.max()}  ==1: {100*(fo==1).mean():.2f}%  "
          f">=2: {int((fo>=2).sum()):,} ({100*(fo>=2).mean():.2f}%)")
    bc = np.bincount(fi, minlength=6)[:6]
    print(f"  fan-in histogram 1..5+: " +
          "  ".join(f"{k}:{bc[k]:,}" for k in range(1, 6)))

    print(f"\n  --- P2 ATTACHMENT ---")
    print("  " + q(g["iou"], "IoU"))
    print("  " + q(g["contain_obj"], "obj-in-sub"))
    print("  " + q(g["gap"], "gap/diag"))
    print("  " + q(g["obj_over_sub_area"], "area ratio"))
    for thr in (0.0, 0.01, 0.05, 0.1, 0.3, 0.5):
        f = float((g["iou"] <= thr).mean())
        print(f"    IoU <= {thr:<5}: {100*f:6.2f}%   "
              f"(of those, obj-in-sub >= 0.5: "
              f"{100*float((g['contain_obj'][g['iou']<=thr] >= 0.5).mean() if (g['iou']<=thr).any() else 0):5.2f}%)")
    disj = g["iou"] <= 0.0
    print(f"    DISJOINT (zero overlap): {int(disj.sum()):,} / {n:,} "
          f"= {100*disj.mean():.3f}%")
    if disj.any():
        print("    " + q(g["gap"][disj], "  their gap"))
        far = disj & (g["gap"] > 0.05)
        print(f"    disjoint AND gap>0.05 diag: {int(far.sum()):,} "
              f"({100*far.mean():.4f}% of all wearing edges)")

    # ---- who wears what --------------------------------------------------
    if cats:
        sc = np.array(sub_cats); oc = np.array(obj_cats)
        top_s = np.bincount(sc[sc >= 0], minlength=len(cats))
        top_o = np.bincount(oc[oc >= 0], minlength=len(cats))
        print(f"\n  top subject cats: " + ", ".join(
            f"{cats[i]} {top_s[i]:,}" for i in np.argsort(-top_s)[:8]))
        print(f"  top object  cats: " + ", ".join(
            f"{cats[i]} {top_o[i]:,}" for i in np.argsort(-top_o)[:8]))

    if dump is not None:
        np.savez_compressed(
            dump, iou=g["iou"], contain_obj=g["contain_obj"], gap=g["gap"],
            area_ratio=g["obj_over_sub_area"], fanin=fi, fanout=fo,
            img=np.array(img_ids), pred=np.array(pred_ids),
            sub_box=sb, obj_box=ob,
            sub_cat=np.array(sub_cats), obj_cat=np.array(obj_cats),
)
        print(f"\n  dumped -> {dump}")

    return {
        "pack": root.name, "n_pairs": n, "n_imgs": n_wear_imgs,
        "fanin_ge2_frac": float((fi >= 2).mean()),
        "fanin_ge2_n": int((fi >= 2).sum()),
        "fanout_mean": float(fo.mean()),
        "iou_p5": float(np.percentile(g["iou"], 5)),
        "iou_median": float(np.median(g["iou"])),
        "disjoint_frac": float(disj.mean()),
        "contain_obj_median": float(np.median(g["contain_obj"])),
        "trunc_frac": float(n_rel_trunc / max(n_rel_total, 1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+",
                    default=["runs/packed/megasg_clean", "runs/packed/vg_raw"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--pattern", default=r"wear|dressed in|has on|clothed")
    ap.add_argument("--max_objects", type=int, default=40)
    ap.add_argument("--exclude_ids", default="runs/datamix/indoorvg_holdout.json")
    ap.add_argument("--dump_dir", default=None)
    args = ap.parse_args()

    ex = None
    if args.exclude_ids and Path(args.exclude_ids).exists():
        raw = json.load(open(args.exclude_ids))
        ex = set(raw if isinstance(raw, list) else raw.get("ids", []))

    rows = []
    for p in args.packs:
        dump = None
        if args.dump_dir:
            Path(args.dump_dir).mkdir(parents=True, exist_ok=True)
            dump = Path(args.dump_dir) / f"wearing_{Path(p).name}_{args.split}.npz"
        r = analyse(Path(p), args.split, args.pattern, args.max_objects, ex, dump)
        if r:
            rows.append(r)

    print(f"\n{'='*78}\nSUMMARY")
    hdr = ("pack", "pairs", "fanin>=2", "fanout", "IoU p5", "IoU med",
           "disjoint", "obj-in-sub med")
    print("{:<16}{:>10}{:>12}{:>9}{:>9}{:>9}{:>10}{:>16}".format(*hdr))
    for r in rows:
        print("{:<16}{:>10,}{:>11.2f}%{:>9.3f}{:>9.4f}{:>9.4f}{:>9.3f}%{:>16.4f}"
.format(r["pack"], r["n_pairs"], 100 * r["fanin_ge2_frac"],
                      r["fanout_mean"], r["iou_p5"], r["iou_median"],
                      100 * r["disjoint_frac"], r["contain_obj_median"]))


if __name__ == "__main__":
    main()
