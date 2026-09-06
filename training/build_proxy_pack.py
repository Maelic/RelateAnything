"""Build the 50K tail-preserving PROXY pack for cheap experiment iteration.

A full run costs about 12 GPU-hours, and the effects worth ablating are
visible at a fraction of the data. The proxy brings a loss or architecture arm
down to 2-3 GPU-hours — but only once it is validated: a known configuration
change whose signature is understood at full scale (mean recall up on PSG while
micro recall and the `on` class collapse, spatial classes recovering on
IndoorVG) has to reproduce in direction on the proxy first.

Selection: rank images by tail value = sum over the image's relations of
1/global_count(predicate), take the top K. An image carrying a count-1 predicate
scores >= 1.0 and is effectively guaranteed in; head-only images fill the rest
of the budget. This is "keep the long tail" without per-class quotas. A random-K
baseline is scored alongside purely for the report, to show what the selection
buys.

The pack format makes this safe: img_meta rows carry ABSOLUTE (offset, count)
pairs into boxes/box_cats/rels (see data/relation_dataset.py's exclude_ids
path, which filters img_meta the same way). So the proxy is: filtered img_meta
+ filtered file_names + symlinked array files. No offsets are rewritten.

IndoorVG holdout stems are excluded at BUILD time so they don't consume budget;
train jobs should still pass --exclude_ids as belt-and-braces.

    python training/build_proxy_pack.py \
        --src runs/packed/megasg_clean --split train \
        --exclude runs/datamix/indoorvg_holdout.json \
        --k 50000 --out runs/packed/megasg_proxy50k
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def bucket_shares(counts_full: np.ndarray, pred_of_rel: np.ndarray) -> dict:
    """Relation share per frequency bucket (buckets defined on the FULL pack)."""
    edges = {"tail(<100)": (0, 100), "mid(100-10k)": (100, 10_000),
             "head(>=10k)": (10_000, np.inf)}
    out = {}
    n = len(pred_of_rel)
    for name, (lo, hi) in edges.items():
        m = (counts_full[pred_of_rel] >= lo) & (counts_full[pred_of_rel] < hi)
        out[name] = float(m.sum() / max(n, 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="runs/packed/megasg_clean")
    ap.add_argument("--split", default="train")
    ap.add_argument("--exclude", default="runs/datamix/indoorvg_holdout.json")
    ap.add_argument("--k", type=int, default=50_000)
    ap.add_argument("--out", default="runs/packed/megasg_proxy50k")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src) / args.split
    meta = json.load(open(src / "meta.json"))
    im = np.load(src / "img_meta.npy")
    rels = np.load(src / "rels.npy", mmap_mode="r")
    file_names = json.load(open(src / "file_names.json"))
    V = len(meta["predicates"])
    assert rels[:, 2].max() < V

    # ---- holdout exclusion (same stem matching as relation_dataset.py) ----
    stems_obj = json.load(open(args.exclude))
    stems = {str(s) for s in (stems_obj["stems"] if isinstance(stems_obj, dict)
                              else stems_obj)}
    keep_img = np.array([Path(f).stem not in stems for f in file_names], bool)
    print(f"[exclude] {int((~keep_img).sum()):,} holdout images removed "
          f"({len(stems):,} stems)")

    pred_of_rel = np.asarray(rels[:, 2])
    img_of_rel = np.repeat(np.arange(len(im)), im[:, 6])
    rel_ok = keep_img[img_of_rel]

    # ---- global predicate counts on the eligible pool ----
    counts = np.bincount(pred_of_rel[rel_ok], minlength=V).astype(np.float64)

    # ---- tail value per image: sum of 1/count over its relations ----
    inv = np.zeros(V)
    nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    score = np.bincount(img_of_rel[rel_ok], weights=inv[pred_of_rel[rel_ok]],
                        minlength=len(im))
    score[~keep_img] = -1.0

    k = min(args.k, int(keep_img.sum()))
    sel = np.sort(np.argpartition(-score, k - 1)[:k])
    in_sel = np.zeros(len(im), bool)
    in_sel[sel] = True

    rng = np.random.default_rng(args.seed)
    rand = rng.choice(np.flatnonzero(keep_img), size=k, replace=False)
    in_rand = np.zeros(len(im), bool)
    in_rand[rand] = True

    # ---- report ----
    def stats(mask_img: np.ndarray, tag: str) -> None:
        m = mask_img[img_of_rel] & rel_ok
        p = pred_of_rel[m]
        c = np.bincount(p, minlength=V)
        alive = int((c > 0).sum())
        n_img = int(mask_img.sum())
        # retention per FULL-pack tail predicate (count < 100)
        tail_ids = np.flatnonzero((counts > 0) & (counts < 100))
        ret = c[tail_ids] / counts[tail_ids]
        print(f"[{tag}] {n_img:,} imgs  {int(m.sum()):,} rels "
              f"({m.sum()/n_img:.1f}/img)  predicates alive {alive:,}/"
              f"{int(nz.sum()):,}")
        print(f"        tail(<100) retention: median {np.median(ret):.2f}  "
              f"p25 {np.percentile(ret,25):.2f}  dead {int((ret==0).sum()):,}"
              f"/{len(tail_ids):,}")
        print(f"        relation share {bucket_shares(counts, p)}")

    full_rel = pred_of_rel[rel_ok]
    print(f"[full]  {int(keep_img.sum()):,} imgs  {len(full_rel):,} rels  "
          f"predicates {int(nz.sum()):,}")
    print(f"        relation share {bucket_shares(counts, full_rel)}")
    stats(in_sel, "proxy")
    stats(in_rand, "random-k (report only)")

    # ---- write pack: filtered index + symlinked arrays ----
    out = Path(args.out) / args.split
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "img_meta.npy", im[sel])
    json.dump([file_names[i] for i in sel], open(out / "file_names.json", "w"))
    for f in ["boxes.npy", "box_cats.npy", "rels.npy"]:
        dst = out / f
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(os.path.relpath(src / f, out), dst)

    sel_rel = in_sel[img_of_rel] & rel_ok
    c_sel = np.bincount(pred_of_rel[sel_rel], minlength=V)
    meta2 = dict(meta)
    meta2["dataset"] = meta["dataset"] + "_proxy50k"
    meta2["num_images"] = int(len(sel))
    meta2["num_rels"] = int(sel_rel.sum())
    meta2["num_boxes"] = int(im[sel, 4].sum())
    meta2["predicate_counts"] = {meta["predicates"][i]: int(c_sel[i])
                                 for i in np.flatnonzero(c_sel)}
    meta2["proxy_of"] = str(Path(args.src).resolve())
    meta2["proxy_selection"] = ("top-k by sum(1/global_count(pred)); "
                                f"k={k}; holdout excluded at build")
    json.dump(meta2, open(out / "meta.json", "w"))
    print(f"\nwrote {out}  (arrays symlinked to {src})")


if __name__ == "__main__":
    main()
