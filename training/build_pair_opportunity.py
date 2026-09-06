"""Per-category-pair INTERACTION RATE, for statistically weighted existence negatives.

WHY
---
The relatedness head is supervised, but every non-GT pair gets the same flat
`neg_weight` (sampler.py). Measured consequence on v38_entity: AUC 0.724 — it separates
— yet 89.4% of pairs score above sigmoid 0.5, none above 0.9, and the whole distribution
sits in a 0.2-wide band. It has never been shown a confident negative, so it never
learns a threshold, so the model never stops emitting.

The flat weight is a hedge against the PU problem (unannotated != unrelated) priced for
the worst case and then applied to everything. But the worst case is rare: most category
pairs relate on only a small fraction of the instance pairs they present. This table
prices the hedge per category pair instead.

    r(cs,co) = relations(cs,co) / opportunities(cs,co)
    opportunities = SUM over images of n_cs * n_co    (ordered INSTANCE pairs)

`1 - r` approximates P(a given unannotated candidate of this category pair is a genuine
negative). Note a binary "never relates" rule is NOT enough: measured on megasg, only 511
category pairs co-occur >=50 times and never relate. The signal is in the graded rate.

CAREFUL: the denominator must be instance pairs, not per-image category co-occurrence.
Three people wearing three shirts is one co-occurrence but three relations, and using the
image-level count produces "rates" above 1.0.

    python training/build_pair_opportunity.py \\
        --pack runs/packed/megasg_clean/train \\
        --cooc runs/packed/datamix_v22/pair_cooc_student.npz \\
        --out runs/packed/datamix_v22/pair_opportunity.npz
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True)
    p.add_argument("--cooc", required=True,
                   help="pair_cooc npz — supplies pair_count (relations per pair)")
    p.add_argument("--limit", type=int, default=0, help="images to scan (0 = all)")
    p.add_argument("--min_support", type=int, default=50,
                   help="opportunities below which the rate is untrustworthy and the "
                        "flat hedge is kept")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    z = np.load(a.cooc, allow_pickle=False)
    C = int(z["num_cats"])

    root = Path(a.pack)
    im = np.load(root / "img_meta.npy")
    box_cats = np.load(root / "box_cats.npy", mmap_mode="r")
    rels_arr = np.load(root / "rels.npy", mmap_mode="r")
    # Numerator and denominator MUST come from the same corpus. The cooc table's
    # `pair_count` was built over datamix_v22 (5,993,105 relations) while these
    # opportunities are counted over this pack (megasg_clean, 4,183,035) — mixing them
    # inflates every rate. Recount relations here, in the same pass.
    rel = np.zeros(C * C, np.int64)
    n_img = len(im)
    idx = range(n_img) if not a.limit else np.linspace(0, n_img - 1, a.limit).astype(int)

    # Dense [C, C] int64 = 66 MB at C=2884. The previous attempt allocated a C*C
    # bincount PER IMAGE, which is what made it take 12 minutes for 60k images.
    opp = np.zeros((C, C), np.int64)
    t0, seen = time.time(), 0
    for k, i in enumerate(idx):
        _, _, _, b0, nb, r0, nr = im[i]
        nb = min(int(nb), 400)
        if nb < 2:
            continue
        cats_all = np.asarray(box_cats[b0:b0 + nb], dtype=np.int64)
        # relations of this image, mapped to their endpoint CATEGORIES
        if int(nr):
            r = np.asarray(rels_arr[r0:r0 + int(nr)], dtype=np.int64)
            r = r[(r[:, 0] < nb) & (r[:, 1] < nb)]
            if len(r):
                cs, co = cats_all[r[:, 0]], cats_all[r[:, 1]]
                m = (cs >= 0) & (cs < C) & (co >= 0) & (co < C)
                np.add.at(rel, cs[m] * C + co[m], 1)
        cats = cats_all[(cats_all >= 0) & (cats_all < C)]
        if len(cats) < 2:
            continue
        u, cnt = np.unique(cats, return_counts=True)
        # ordered instance pairs: n_a * n_b, minus self-pairings on the diagonal
        blk = np.outer(cnt, cnt)
        np.fill_diagonal(blk, cnt * (cnt - 1))
        opp[np.ix_(u, u)] += blk
        seen += 1
        if k and k % 50000 == 0:
            print(f"  {k}/{len(idx)}  {time.time()-t0:.0f}s", flush=True)

    scale = n_img / max(1, seen) if a.limit else 1.0
    opp_flat = (opp.reshape(-1) * scale).astype(np.int64)
    rel = (rel * scale).astype(np.int64)   # same extrapolation as the denominator
    rate = np.zeros(C * C, np.float32)
    ok = opp_flat > 0
    rate[ok] = np.minimum(1.0, rel[ok] / opp_flat[ok])

    trusted = opp_flat >= a.min_support
    print(f"\nscanned {seen} images in {time.time()-t0:.0f}s")
    print(f"category pairs with any opportunity: {int(ok.sum()):,}")
    print(f"...with >= {a.min_support} opportunities: {int(trusted.sum()):,}")
    r = rate[trusted]
    if len(r):
        for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99):
            print(f"    interaction rate q{q:.2f}: {np.quantile(r, q):.4f}")
        for thr in (0.01, 0.02, 0.05, 0.10):
            n = int((r <= thr).sum())
            print(f"    rate <= {thr:.2f}: {n:,} pairs ({100*n/len(r):.1f}%) "
                  f"-> negatives worth >= {1-thr:.2f}")
    np.savez_compressed(
        a.out, rate=rate, opportunities=opp_flat, relations=rel,
        num_cats=np.int32(C), min_support=np.int32(a.min_support),
        meta=np.asarray([json.dumps({
            "pack": str(root), "cooc": a.cooc, "images_scanned": int(seen),
            "extrapolated": bool(a.limit),
            "note": "rate = relations / ordered INSTANCE-pair opportunities; "
                    "1-rate approximates P(unannotated candidate is a true negative)",
        })]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
