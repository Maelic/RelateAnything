"""v42.1: support-weight the kernel positives in soft_supervision.npz.

THE MEASURED DEFECT (full-scale v42 + proxy, [[relsgg-v42-soft-supervision]]):
the isotonic kernel spreads positive mass by cosine alone, so high-support heads
(`on`, `near`) — which have the LARGEST synonym clouds — leave the exact form the
smallest share of its own positive mass exactly where GT frequency is highest.
Result: `on` -69..-89% on transfer while macro rises.

THE FIX, with no new hand constants: the kernel estimates P(j synonymous with g).
What the positive target actually needs is P(j synonymous) x P(annotator would
WRITE j). The second factor is estimable from the same pooled GT counts the
builder already uses: the capped MLE ratio

    w_new[g, j] = kernel[g, j] * min(1, count[j] / count[g])        diag stays 1

- g=`on`, j=rare scaffold variant: ratio ~0 -> mass collapses back onto `on`.
- g=rare variant, j=`on`: ratio >= 1 -> capped, kernel value KEPT -> the region
  alignment that bought the tail-transfer wins is untouched.
- comparable-count synonyms: unchanged.
Down-weight only: the support factor never claims more synonymy than the kernel.
count[j]=0 (never annotated in the mix) drops the member entirely.

Everything else (neg_lw, sym, inverse kernel, w_cooc) is copied verbatim.

    python training/build_soft_supervision_sw.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))
TS = PROJ / "runs/packed/datamix_v22/text_space"

from training.estimate_false_negatives import load_pack  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(TS / "soft_supervision.npz"))
    ap.add_argument("--packs", nargs="+",
                    default=["runs/packed/megasg", "runs/packed/vg_raw"])
    ap.add_argument("--out", default=str(TS / "soft_supervision_sw.npz"))
    a = ap.parse_args()

    z = dict(np.load(a.src, allow_pickle=False))
    names = [str(n) for n in z["predicates"]]
    idx = {n: i for i, n in enumerate(names)}
    V = len(names)

    # same pooled-count basis as build_soft_supervision.py
    counts = np.zeros(V, np.int64)
    for root in a.packs:
        pnames, rels, _ = load_pack(Path(root), "train")
        remap = np.array([idx.get(n, -1) for n in pnames], np.int64)
        g = remap[rels[:, 2]]
        np.add.at(counts, g[g >= 0], 1)
        print(f"[packs] {root}: {(g >= 0).sum():,} relations")

    gi, gj = z["pos_i"].astype(np.int64), z["pos_j"].astype(np.int64)
    w = z["pos_w"].astype(np.float32)
    ratio = np.minimum(1.0, counts[gj] / np.maximum(counts[gi], 1)).astype(np.float32)
    ratio[counts[gj] == 0] = 0.0
    w_new = w * ratio
    keep = (w_new > 1e-4) | (gi == gj)
    print(f"[pos] {len(w):,} kernel entries -> {int(keep.sum()):,} after support "
          f"weighting ({(1 - keep.sum()/len(w))*100:.1f}% dropped)")

    # review table: exact-form share of its own positive mass, before/after
    def mass(gis, ws, tid):
        m = gis == tid
        return float(ws[m].sum())
    print(f"\n{'predicate':>14s} {'count':>8s} {'mass_old':>9s} {'mass_new':>9s} "
          f"{'exact_share old->new':>22s}")
    for name in ["on", "near", "in front of", "behind", "above",
                 "resting beside", "sipping from"]:
        if name not in idx:
            continue
        t = idx[name]
        mo, mn = mass(gi, w, t) + 1.0, mass(gi[keep], w_new[keep], t) + 1.0
        print(f"{name:>14s} {counts[t]:>8,d} {mo:>9.1f} {mn:>9.1f} "
              f"{1/mo:>10.3f} -> {1/mn:.3f}")

    out = {k: v for k, v in z.items() if not k.startswith("pos_")}
    out.update(pos_i=gi[keep].astype(z["pos_i"].dtype),
               pos_j=gj[keep].astype(z["pos_j"].dtype),
               pos_w=w_new[keep].astype(z["pos_w"].dtype))
    np.savez_compressed(a.out, **out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
