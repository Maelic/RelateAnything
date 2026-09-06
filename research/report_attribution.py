"""report_attribution.py — read the lesion ladder, and join it against FREQ.

Two tables.

TABLE 1, the lesion ladder: what each input channel is worth on the per-edge
protocol, with the pixel-free FREQ baseline printed on the same scale.

TABLE 2, THE COMPLEMENTARITY JOIN, which is the one that matters. FREQ
outscoring the model in aggregate does NOT mean vision is worthless -- it means
either (a) vision is redundant with counting, or (b) vision is right in a
DIFFERENT place than counting, and the aggregate hides it. Only an edge-by-edge
join separates those. So for every edge we cross-tabulate

               FREQ right    FREQ wrong
  model right      both        MODEL ONLY   <- what vision genuinely adds
  model wrong   FREQ ONLY        neither

and report the oracle union. If "model only" is large, vision carries
information the label prior does not have, whatever the aggregate says. Note
the asymmetry that makes this fair to read: FREQ is handed ORACLE entity
categories, while the model never sees a category label at inference
(entity_labels are training-loss only, sampler.py:368) and must recover them
from pixels.

Usage:
    python training/report_attribution.py [--towers...] [--split test]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# training/ is not a package (no __init__.py), so this is a flat import off the
# path insert above, not `from training.bias_baselines import...`.
from research.bias_baselines import FreqBaseline, load_split  # noqa: E402

B = "relsgg-vits16plus-zeroshot"
S = "_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5"
TOWERS = [("ViT-S", f"{B}_vits16{S}"), ("ViT-S+", f"{B}_vits16plus{S}"),
          ("ViT-B", f"{B}{S}")]
LESIONS = ("full", "compose0", "nogeo", "imgshuf", "prior")


def freq_ranks(root: str, split: str):
    """FREQ rank per rels.npy row of `split`, fit on that pack's train split."""
    meta = json.load(open(os.path.join(root, "train", "meta.json")))
    n_cat, n_pred = len(meta["categories"]), len(meta["predicates"])
    tr = load_split(root, "train")
    ev = load_split(root, split)
    f = FreqBaseline(n_cat, n_pred)
    f.fit(tr[0], tr[1], tr[2])
    # rank_of is 0-indexed; the probe reports 1-indexed. Align to 1-indexed.
    return np.array([f.rank_of(s, o, p) + 1
                     for s, o, p in zip(ev[0], ev[1], ev[2])])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--sources", nargs="+",
                    default=["vg150", "psg", "indoorvg"])
    args = ap.parse_args()

    fr = {}
    for src in args.sources:
        root = f"runs/packed/{src}"
        if os.path.isdir(os.path.join(root, args.split)):
            fr[src] = freq_ranks(root, args.split)
            print(f"[freq] {src}: fit, {len(fr[src]):,} edges")

    for name, run in TOWERS:
        f = f"runs/train/{run}/attrib/attribution_{args.split}.json"
        if not os.path.exists(f):
            print(f"\n### {name}: no attribution json yet ({f})")
            continue
        d = json.load(open(f))
        print(f"\n{'='*78}\n### {name}\n{'='*78}")
        for src, cell in d["results"].items():
            print(f"\n-- {src} ({args.split}) "
                  f"cover {cell.get('full', {}).get('pair_coverage', 0):.4f}")
            print(f"   {'channel':<26} {'Acc@1':>7} {'dAcc%':>7} "
                  f"{'MRR':>7} {'mAcc@1':>7} {'MedRank':>7}")
            base = cell.get("full")
            for k in LESIONS:
                if k not in cell:
                    continue
                m = cell[k]
                dd = (100 * (m["Acc@1"] - base["Acc@1"]) / base["Acc@1"]
                      if base and base["Acc@1"] else float("nan"))
                print(f"   {k:<26} {m['Acc@1']:>7.4f} {dd:>7.1f} "
                      f"{m['MRR']:>7.4f} {m['mAcc@1']:>7.4f} "
                      f"{m['MedRank']:>7.0f}")
            if src in fr:
                bl = json.load(open(
                    f"runs/packed/{src}/bias_baselines_{args.split}.json"))
                b = bl["freq_baseline"]["all"]
                # Print FREQ's MACRO too. Its micro win is produced by always
                # answering a category pair's MAJORITY predicate, so micro alone
                # is the one axis on which counting is guaranteed to look good;
                # mAcc@1 is where a visual model can actually be ahead.
                dm = (100 * (base["mAcc@1"] - b["mAcc@1"]) / b["mAcc@1"]
                      if b.get("mAcc@1") else float("nan"))
                print(f"   {'FREQ (oracle labels, 0 px)':<26} "
                      f"{b['R@1']:>7.4f} {'':>7} {b['MRR']:>7.4f} "
                      f"{b.get('mAcc@1', float('nan')):>7.4f}")
                print(f"   {'  -> model vs FREQ':<26} "
                      f"{100*(base['Acc@1']-b['R@1'])/b['R@1']:>+7.1f}% micro"
                      f"{'':>8} {dm:>+7.1f}% MACRO")
            if "attribution" in cell:
                a = cell["attribution"]
                shares = "  ".join(
                    f"{k.replace('var_share_', '')} {a[k]*100:.1f}%"
                    for k in a if k.startswith("var_share_"))
                print(f"   semantic-expert Var_v: {shares}  "
                      f"cov {a['cov_mass']*100:.1f}%  "
                      f"(fp32 residual {a['residual_max']:.1e})")

            # ---- the join ----
            ef = (f"runs/train/{run}/attrib/"
                  f"edges_{src}_{args.split}.npz")
            if not (src in fr and os.path.exists(ef)):
                continue
            z = np.load(ef)
            row, rk = z["full_row"], z["full_rank"]
            ok = row >= 0
            row, rk = row[ok], rk[ok]
            fq = fr[src][row]
            mr, fq1 = rk == 1, fq == 1
            n = len(row)
            both = int((mr & fq1).sum())
            m_only = int((mr & ~fq1).sum())
            f_only = int((~mr & fq1).sum())
            neither = int((~mr & ~fq1).sum())
            print(f"\n   COMPLEMENTARITY on {n:,} joined edges (top-1 correct?)")
            print(f"     both right          {both:>7,}  {100*both/n:>5.1f}%")
            print(f"     MODEL only          {m_only:>7,}  {100*m_only/n:>5.1f}%"
                  f"   <- what vision adds over counting")
            print(f"     FREQ only           {f_only:>7,}  {100*f_only/n:>5.1f}%")
            print(f"     neither             {neither:>7,}  {100*neither/n:>5.1f}%")
            print(f"     model {100*mr.mean():>5.1f}%   FREQ {100*fq1.mean():>5.1f}%"
                  f"   ORACLE UNION {100*(mr | fq1).mean():>5.1f}%")
            if m_only + f_only:
                print(f"     of the {m_only + f_only:,} edges exactly one gets "
                      f"right, the model takes "
                      f"{100*m_only/(m_only+f_only):.1f}%")


if __name__ == "__main__":
    main()
