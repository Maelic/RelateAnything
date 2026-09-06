"""Export the isotonic synonym kernel P(synonym | cos_v2) as a dense curve.

The curve is fit EXACTLY as in training/build_soft_supervision.py's kernel
section (same lexical ground truth from scaffolding-stripped identity buckets,
same count-weighted marginal negative sampling, same seed), then dumped on a
0.001 cosine grid so consumers (codebook decoding in eval_zeroshot.py) can
np.interp it without reloading packs or refitting.

    python training/export_syn_kernel.py
writes runs/packed/datamix_v22/text_space/syn_kernel_v2.npz {cos, p}.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from training.estimate_false_negatives import load_pack  # noqa: E402
from training.build_soft_supervision import load_embeds, norm  # noqa: E402

TS = PROJ / "runs/packed/datamix_v22/text_space"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packs", nargs="+",
                    default=["runs/packed/megasg", "runs/packed/vg_raw"])
    ap.add_argument("--union_preds", default=str(TS / "union_predicates.json"))
    ap.add_argument("--kernel_embeds",
                    default=str(TS / "pred_embeds_studentv2_photo.npz"))
    ap.add_argument("--neg_per_pos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corpus", default="",
                    help="Distillation corpus.json: adds its labelled "
                         "synonym_pairs as positives (the 108 cross-lemma "
                         "pairs populate the 0.6-0.8 cosine band the "
                         "scaffold-identity truth leaves empty) and its "
                         "inverse_pairs as guaranteed hard negatives (so "
                         "lifting the curve there cannot admit antonyms). "
                         "Same labelled set tau_eval=0.955 was calibrated "
                         "against — an established ground truth, not a new "
                         "hand input.")
    ap.add_argument("--out", default=str(TS / "syn_kernel_v2.npz"))
    a = ap.parse_args()
    rs = np.random.RandomState(a.seed)

    import json
    names = [str(n) for n in json.load(open(a.union_preds))]
    idx = {n: i for i, n in enumerate(names)}
    V = len(names)
    E2 = load_embeds(Path(a.kernel_embeds), names)

    counts = np.zeros(V, np.int64)
    for root in a.packs:
        pnames, rels, _ = load_pack(Path(root), "train")
        remap = np.array([idx.get(n, -1) for n in pnames], np.int64)
        g = remap[rels[:, 2]]
        np.add.at(counts, g[g >= 0], 1)

    from sklearn.isotonic import IsotonicRegression
    buckets: dict = {}
    for n in names:
        buckets.setdefault(norm(n), []).append(n)
    lex = np.array([[idx[x], idx[y]] for v in buckets.values() if len(v) > 1
                    for x, y in itertools.combinations(v, 2)], np.int64)
    hard_neg = np.zeros((0, 2), np.int64)
    if a.corpus:
        import json as _json
        cj = _json.load(open(a.corpus))
        cs = cj["strings"]
        syn = np.array([[idx[cs[i]], idx[cs[j]]]
                        for i, j in cj["synonym_pairs"]
                        if cs[i] in idx and cs[j] in idx], np.int64)
        hard_neg = np.array([[idx[cs[i]], idx[cs[j]]]
                             for i, j in cj["inverse_pairs"]
                             if cs[i] in idx and cs[j] in idx], np.int64)
        print(f"[corpus] +{len(syn)} labelled synonym positives "
              f"(+{len(hard_neg)} inverse hard negatives) from {a.corpus}")
        lex = np.concatenate([lex, syn])
    marg = np.bincount(lex.reshape(-1), minlength=V).astype(np.float64)
    marg = marg * (counts + 1.0)
    marg /= marg.sum()
    neg = rs.choice(V, size=(len(lex) * a.neg_per_pos * 2, 2), p=marg)
    neg = neg[neg[:, 0] != neg[:, 1]]
    seen = {tuple(sorted(t)) for t in lex}
    neg = np.array([t for t in neg if tuple(sorted(t)) not in seen])[
:len(lex) * a.neg_per_pos]
    if len(hard_neg):
        neg = np.concatenate([neg, hard_neg])
    cs = np.concatenate([(E2[lex[:, 0]] * E2[lex[:, 1]]).sum(-1),
                         (E2[neg[:, 0]] * E2[neg[:, 1]]).sum(-1)])
    ys = np.concatenate([np.ones(len(lex)), np.zeros(len(neg))])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cs, ys)

    grid = np.arange(0.0, 1.0005, 0.001)
    p = iso.predict(grid).astype(np.float32)
    np.savez_compressed(a.out, cos=grid.astype(np.float32), p=p)
    nz = grid[np.argmax(p > 0.01)]
    print(f"[kernel] {len(lex):,} lexical pairs; P>0.01 above cos {nz:.3f}; "
          f"P at 0.75/0.85/0.95 = {iso.predict([0.75])[0]:.3f}/"
          f"{iso.predict([0.85])[0]:.3f}/{iso.predict([0.95])[0]:.3f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
