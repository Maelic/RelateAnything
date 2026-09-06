"""Calibrate the open-vocabulary synonym-matching threshold (--tau_eval).

The open-vocab protocol (eval_zeroshot.py --open_vocab) accepts a predicted
predicate as correct when its text cosine to the GT predicate is >= tau. That
threshold is the whole metric: too low and the model gets credit for saying
anything vaguely relational, too high and a synonym-preserving model is
punished for its own vocabulary. It must be calibrated, not guessed.

Oracle: runs/packed/text_student/corpus.json ships LABELLED synonym_pairs and
inverse_pairs (built by training/distill/build_corpus.py), so tau can be
chosen against ground truth rather than by inspection.

A THRESHOLD IS A PROPERTY OF ONE EMBEDDING SPACE, and carrying a number
between spaces silently changes what it tests. Two text spaces for the same
vocabulary:

                 synonym cos   inverse   random    separation
    an earlier space   0.961     0.279    0.805         0.156
    the shipped one    0.755    -0.003    0.167         0.588

The shipped space separates synonyms from random pairs 3.8 times better and
pushes inverses to zero, but the same threshold means something else in it:

      tau     synonym recall, earlier    shipped
    0.955                       72.3%       0.6%
    0.720                      100.0%      64.4%   <- the calibrated point

A threshold carried from one space into another therefore rejected 99.4% of
true synonyms while looking unchanged. So --pred_embeds defaults to the space
in use, the grid extends down to 0.30, and the chosen tau is written into the
json together with the space it was fitted in, which lets eval_zeroshot.py
refuse to mix them.

Measured on the 19,103-predicate union vocabulary in the shipped space:

      tau  syn recall  inv leak  rand FPR   accepted/GT
    0.700       66.7%     0.00%    0.122%          23.3
    0.720       64.4%     0.00%    0.099%          19.0   <- chosen
    0.750       57.6%     0.00%    0.074%          14.2
    0.800       44.1%     0.00%    0.047%           8.9

The chosen point reproduces v1's INTENDED strictness (64.4% synonym recall vs
v1's 63.3% at its own chosen tau), so this is a relocation of one operating
point into the correct space, not a loosening of the metric.

Two findings worth keeping:
  * INVERSE LEAKAGE IS ZERO at every tau >= 0.90 (inverse cos 0.279 vs
    synonym 0.961). The antonym-aware distillation did its job, and the
    inverse mask in build_cross_match_matrix is belt-and-braces, not load
    bearing.
  * RANDOM-PAIR COSINE AVERAGES 0.805 — the space is highly anisotropic, so
    synonym (0.961) and random are only ~0.16 apart. That narrow gap, not the
    threshold, is what caps this metric's precision; a hubness correction
    (CSLS/whitening) on the matcher is the obvious next lever.

Selection rule: highest synonym recall subject to inverse leakage == 0 and
random FPR <= 0.1% (i.e. at most ~19 wrong spellings accepted per GT class).

    python training/calibrate_match_tau.py
    python training/calibrate_match_tau.py --max_fpr 0.0005   # stricter
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Wide enough for any text space's operating point: a grid that cannot reach
# the answer silently returns its own lowest entry, which reads as a choice.
TAUS = [round(0.30 + 0.01 * i, 2) for i in range(70)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/packed/text_student/corpus.json")
    ap.add_argument("--pred_embeds",
                    default="runs/packed/datamix_v22/text_space/"
                            "pred_embeds_studentv2_photo.npz")
    ap.add_argument("--max_fpr", type=float, default=0.001,
                    help="Max tolerated random-pair false-positive rate.")
    ap.add_argument("--n_random", type=int, default=400_000)
    ap.add_argument("--out", default="runs/packed/text_student/tau_calibration.json")
    a = ap.parse_args()

    c = json.load(open(a.corpus))
    strings = c["strings"]
    z = np.load(a.pred_embeds)
    preds = [str(p) for p in z["predicates"]]
    E = torch.nn.functional.normalize(
        torch.from_numpy(z["embeddings"]).float(), dim=-1)
    idx = {p: i for i, p in enumerate(preds)}

    def mapped(pairs):
        out = []
        for x, y in pairs:
            i, j = idx.get(strings[x]), idx.get(strings[y])
            if i is not None and j is not None and i != j:
                out.append((i, j))
        return out

    syn, inv = mapped(c["synonym_pairs"]), mapped(c["inverse_pairs"])
    cos = lambda P: (E[[i for i, _ in P]] * E[[j for _, j in P]]).sum(-1).numpy()
    c_syn, c_inv = cos(syn), cos(inv)

    rng = np.random.default_rng(0)
    ra, rb = (rng.integers(0, len(preds), a.n_random),
              rng.integers(0, len(preds), a.n_random))
    keep = ra != rb
    c_rnd = (E[ra[keep]] * E[rb[keep]]).sum(-1).numpy()

    print(f"vocab {len(preds):,} | labelled: {len(syn)} synonym, {len(inv)} inverse")
    print(f"cos means — synonym {c_syn.mean():.3f}  inverse {c_inv.mean():.3f}  "
          f"random {c_rnd.mean():.3f}")
    print(f"\n{'tau':>7} {'syn recall':>11} {'inv leak':>9} {'rand FPR':>9} {'accept/GT':>10}")
    rows, best = [], None
    for t in TAUS:
        sr = float((c_syn >= t).mean())
        il = float((c_inv >= t).mean())
        fp = float((c_rnd >= t).mean())
        rows.append({"tau": t, "syn_recall": sr, "inv_leak": il,
                     "rand_fpr": fp, "accept_per_gt": fp * len(preds)})
        print(f"{t:>7.3f} {100*sr:>10.1f}% {100*il:>8.2f}% {100*fp:>8.3f}% "
              f"{fp*len(preds):>10.1f}")
        if il == 0.0 and fp <= a.max_fpr and (best is None or sr > best["syn_recall"]):
            best = rows[-1]

    if best is None:
        raise SystemExit(f"no tau satisfies inv_leak==0 and fpr<={a.max_fpr}")
    print(f"\nCHOSEN tau={best['tau']}  syn recall {100*best['syn_recall']:.1f}%  "
          f"random FPR {100*best['rand_fpr']:.3f}%  "
          f"(~{best['accept_per_gt']:.0f} accepted spellings per GT class)")
    json.dump({"pred_embeds": a.pred_embeds,   # the space this tau is valid in
               "chosen": best, "curve": rows, "max_fpr": a.max_fpr,
               "cos_means": {"synonym": float(c_syn.mean()),
                             "inverse": float(c_inv.mean()),
                             "random": float(c_rnd.mean())},
               "n_labelled": {"synonym": len(syn), "inverse": len(inv)}},
              open(a.out, "w"), indent=2)
    print(f"saved → {a.out}")


if __name__ == "__main__":
    main()
