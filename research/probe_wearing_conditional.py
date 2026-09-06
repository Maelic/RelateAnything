"""P(annotated `wearing` | person-garment pair) as a function of overlap.

probe_wearing_prior.py measures what wearing edges LOOK like, which is
conditioned on the edge existing — it can say "annotated wearing overlaps", but
not "a disjoint pair is not wearing". This script measures the other direction:
enumerate every (person-ish, garment-ish) box pair in an image and ask how often
it carries a wearing annotation, bucketed by IoU / containment / gap.

That curve IS the prior the model was supposed to learn, and it is the fair
comparison for the model's own P(emit wearing | pair, geometry).

Incompleteness caveat, stated once: an unannotated pair is not proven negative
(see relsgg-haystack-federated-precision). It biases the rate DOWN uniformly, so
the SHAPE across geometry bins — the only thing claimed here — survives it, and
the disjoint bucket is the one place where "unannotated" and "false" coincide
for physical reasons.

Usage:
    python training/probe_wearing_conditional.py --dump_dir runs/analysis/wearing
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PERSON = re.compile(
    r"^(person|man|woman|men|women|boy|girl|guy|lady|player|child|kid|people|"
    r"skier|snowboarder|surfer|rider|pedestrian|worker|chef|officer|policeman|"
    r"soldier|athlete|catcher|batter|pitcher|skateboarder|biker|cyclist|"
    r"male|female|human|adult|toddler|baby|gentleman|dude|teenager|"
    r"passenger|customer|spectator|crowd|couple|driver|student|doctor|nurse)s?$",
    re.I)

GARMENT = re.compile(
    r"^(clothing|clothes|shirt|t-?shirt|tshirt|pants|trousers|jeans|shorts|"
    r"jacket|coat|hat|cap|helmet|glasses|sunglasses|goggles|shoe|shoes|"
    r"sneaker|sneakers|boot|boots|sandal|sandals|footwear|sock|socks|"
    r"dress|skirt|suit|tie|necktie|scarf|glove|gloves|belt|watch|"
    r"necklace|bracelet|earring|earrings|ring|backpack|bag|apron|uniform|"
    r"jersey|sweater|hoodie|vest|blouse|robe|gown|swimsuit|bikini|"
    r"headband|bandana|visor|mask|wetsuit|costume|outfit|attire|"
    r"kneepad|wristband|armband|lanyard|badge|sleeve|collar|hood|"
    r"heel|heels|slipper|slippers|flip-?flop|cleat|cleats|skate|skates)s?$",
    re.I)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


def cross_geometry(sb, ob):
    """All-pairs [S,O] IoU, containment of obj in sub, and rectangle gap."""
    ix1 = np.maximum(sb[:, None, 0], ob[None,:, 0])
    iy1 = np.maximum(sb[:, None, 1], ob[None,:, 1])
    ix2 = np.minimum(sb[:, None, 2], ob[None,:, 2])
    iy2 = np.minimum(sb[:, None, 3], ob[None,:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    sa = np.clip((sb[:, 2] - sb[:, 0]) * (sb[:, 3] - sb[:, 1]), 1e-9, None)
    oa = np.clip((ob[:, 2] - ob[:, 0]) * (ob[:, 3] - ob[:, 1]), 1e-9, None)
    iou = inter / (sa[:, None] + oa[None,:] - inter)
    dx = np.maximum(np.maximum(ob[None,:, 0] - sb[:, None, 2],
                               sb[:, None, 0] - ob[None,:, 2]), 0)
    dy = np.maximum(np.maximum(ob[None,:, 1] - sb[:, None, 3],
                               sb[:, None, 1] - ob[None,:, 3]), 0)
    gap = np.sqrt(dx ** 2 + dy ** 2) / np.sqrt(2.0)
    return iou, inter / oa[None,:], gap


BINS = np.array([-1e-9, 1e-9, 0.01, 0.03, 0.05, 0.10, 0.20, 0.35, 0.50, 1.01])
BIN_NAMES = ["IoU = 0", "(0,.01]", "(.01,.03]", "(.03,.05]", "(.05,.10]",
             "(.10,.20]", "(.20,.35]", "(.35,.50]", "(.50,1]"]


def analyse(root: Path, split: str, pattern: str, max_objects: int,
            exclude_ids, limit, dump):
    d = root / split
    meta = json.load(open(d / "meta.json"))
    preds, cats = meta["predicates"], meta.get("categories", [])
    img_meta = np.load(d / "img_meta.npy")
    boxes = np.load(d / "boxes.npy", mmap_mode="r")
    box_cats = np.load(d / "box_cats.npy", mmap_mode="r")
    rels = np.load(d / "rels.npy", mmap_mode="r")
    file_names = json.load(open(d / "file_names.json"))

    rx = re.compile(pattern, re.I)
    wear_ids = np.array([i for i, p in enumerate(preds) if rx.search(p)])
    is_person = np.array([bool(PERSON.match(c)) for c in cats])
    is_garment = np.array([bool(GARMENT.match(c)) for c in cats])
    print(f"\n{'='*78}\n{root.name}/{split}: "
          f"{int(is_person.sum())} person cats, {int(is_garment.sum())} garment "
          f"cats of {len(cats)}; {len(wear_ids)} wearing forms")

    keep_img = np.ones(len(img_meta), dtype=bool)
    if exclude_ids:
        keep_img = np.array([Path(f).stem not in exclude_ids for f in file_names])

    n_pair = np.zeros(len(BINS) - 1, dtype=np.int64)
    n_wear = np.zeros(len(BINS) - 1, dtype=np.int64)
    # same, bucketed by containment of the garment in the person box
    CBINS = np.array([-1e-9, 1e-9, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01])
    c_pair = np.zeros(len(CBINS) - 1, dtype=np.int64)
    c_wear = np.zeros(len(CBINS) - 1, dtype=np.int64)
    # gap bins for the disjoint subset only
    GBINS = np.array([-1e-9, 1e-9, 0.02, 0.05, 0.10, 0.20, 1.01])
    g_pair = np.zeros(len(GBINS) - 1, dtype=np.int64)
    g_wear = np.zeros(len(GBINS) - 1, dtype=np.int64)

    n_imgs = 0
    for i in range(len(img_meta)):
        if not keep_img[i]:
            continue
        if limit and n_imgs >= limit:
            break
        b0, nb, r0, nr = (int(v) for v in img_meta[i][3:7])
        nb = min(nb, max_objects)
        if nb < 2:
            continue
        bc = np.asarray(box_cats[b0:b0 + nb])
        pi = np.flatnonzero(is_person[bc])
        oi = np.flatnonzero(is_garment[bc])
        if not len(pi) or not len(oi):
            continue
        n_imgs += 1
        bx = cxcywh_to_xyxy(np.asarray(boxes[b0:b0 + nb], dtype=np.float64))
        iou, cont, gap = cross_geometry(bx[pi], bx[oi])

        lab = np.zeros(iou.shape, dtype=bool)
        if nr:
            r = np.asarray(rels[r0:r0 + nr])
            r = r[np.isin(r[:, 2], wear_ids) & (r[:, 0] < nb) & (r[:, 1] < nb)]
            if len(r):
                pmap = {int(v): k for k, v in enumerate(pi)}
                omap = {int(v): k for k, v in enumerate(oi)}
                for s, o in r[:,:2]:
                    a, b = pmap.get(int(s)), omap.get(int(o))
                    if a is not None and b is not None:
                        lab[a, b] = True

        bi = np.digitize(iou.ravel(), BINS) - 1
        np.add.at(n_pair, bi, 1)
        np.add.at(n_wear, bi, lab.ravel().astype(np.int64))
        ci = np.digitize(cont.ravel(), CBINS) - 1
        np.add.at(c_pair, ci, 1)
        np.add.at(c_wear, ci, lab.ravel().astype(np.int64))
        dis = iou.ravel() <= 0.0
        if dis.any():
            gi = np.digitize(gap.ravel()[dis], GBINS) - 1
            np.add.at(g_pair, gi, 1)
            np.add.at(g_wear, gi, lab.ravel()[dis].astype(np.int64))

    tot_p, tot_w = n_pair.sum(), n_wear.sum()
    print(f"  {n_imgs:,} imgs with >=1 person and >=1 garment; "
          f"{tot_p:,} candidate pairs, {tot_w:,} annotated wearing "
          f"(base rate {100*tot_w/max(tot_p,1):.2f}%)")

    print(f"\n  P(wearing | person-garment pair) by IoU")
    print(f"  {'bin':<12}{'pairs':>12}{'wearing':>11}{'rate':>9}{'lift':>8}")
    base = tot_w / max(tot_p, 1)
    for k, nm in enumerate(BIN_NAMES):
        if n_pair[k] == 0:
            continue
        r = n_wear[k] / n_pair[k]
        print(f"  {nm:<12}{n_pair[k]:>12,}{n_wear[k]:>11,}{100*r:>8.2f}%"
              f"{r/max(base,1e-12):>8.2f}x")

    print(f"\n... by containment of garment in person box")
    for k in range(len(CBINS) - 1):
        if c_pair[k] == 0:
            continue
        lo, hi = CBINS[k], CBINS[k + 1]
        nm = "cont = 0" if k == 0 else f"({max(lo,0):.1f},{min(hi,1.0):.1f}]"
        r = c_wear[k] / c_pair[k]
        print(f"  {nm:<12}{c_pair[k]:>12,}{c_wear[k]:>11,}{100*r:>8.2f}%"
              f"{r/max(base,1e-12):>8.2f}x")

    print(f"\n... DISJOINT pairs only, by gap (fraction of image diagonal)")
    for k in range(len(GBINS) - 1):
        if g_pair[k] == 0:
            continue
        lo, hi = GBINS[k], GBINS[k + 1]
        nm = "touching" if k == 0 else f"({max(lo,0):.2f},{min(hi,1.0):.2f}]"
        r = g_wear[k] / g_pair[k]
        print(f"  {nm:<12}{g_pair[k]:>12,}{g_wear[k]:>11,}{100*r:>8.3f}%"
              f"{r/max(base,1e-12):>8.2f}x")

    if dump:
        np.savez(dump, n_pair=n_pair, n_wear=n_wear, bins=BINS,
                 c_pair=c_pair, c_wear=c_wear, cbins=CBINS,
                 g_pair=g_pair, g_wear=g_wear, gbins=GBINS)
        print(f"\n  dumped -> {dump}")
    return dict(pack=root.name, base=base, n_pair=int(tot_p), n_wear=int(tot_w),
                rate_disjoint=float(n_wear[0] / max(n_pair[0], 1)),
                rate_high=float(n_wear[-1] / max(n_pair[-1], 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+",
                    default=["runs/packed/megasg_clean", "runs/packed/vg_raw"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--pattern", default=r"wear|dressed in|has on|clothed")
    ap.add_argument("--max_objects", type=int, default=40)
    ap.add_argument("--exclude_ids", default="runs/datamix/indoorvg_holdout.json")
    ap.add_argument("--limit", type=int, default=0)
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
            dump = Path(args.dump_dir) / f"cond_{Path(p).name}_{args.split}.npz"
        rows.append(analyse(Path(p), args.split, args.pattern, args.max_objects,
                            ex, args.limit, dump))

    print(f"\n{'='*78}\nSUMMARY  P(wearing | person-garment pair)")
    print(f"{'pack':<16}{'pairs':>12}{'base':>9}{'disjoint':>11}{'IoU>.5':>10}")
    for r in rows:
        print(f"{r['pack']:<16}{r['n_pair']:>12,}{100*r['base']:>8.2f}%"
              f"{100*r['rate_disjoint']:>10.3f}%{100*r['rate_high']:>9.2f}%")


if __name__ == "__main__":
    main()
