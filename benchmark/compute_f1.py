"""F1@K = harmonic mean of R@K and mR@K (SGG-Benchmark, Maelic et al.).

    F1@K = 2 * R@K * mR@K / (R@K + mR@K)

WHY IT IS WORTH REPORTING. R@K and mR@K are in tension by construction: the
head-heavy prediction that maximises R@K flattens mR@K, and the tail-boosting
one does the reverse. Reporting them side by side lets a run look good on
whichever one it happens to win — a change that took PSG mean recall +19% while
collapsing `on` by 81% reads as a win on one of the two. The harmonic mean
weights the smaller of the two, so a run only scores well by sacrificing
neither.

RETROACTIVE COMPUTATION IS EXACT, NOT AN APPROXIMATION. F1@K is a pure function
of the two aggregate numbers, both of which are already stored in every
history.json and eval json — so nothing needs re-running on a GPU.

THE ONE REAL PITFALL: R@K and mR@K must come from the SAME protocol. Mixing a
graph-constrained mR with an unconstrained R produces a number that is not
F1 of anything. This
script therefore only pairs values found in the same metrics dict, and refuses
to combine across files.

    python training/compute_f1.py --runs                # training histories
    python training/compute_f1.py --evals               # benchmark eval jsons
    python training/compute_f1.py --runs --k 20 50 100
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Optional


def f1(r: Optional[float], mr: Optional[float]) -> Optional[float]:
    """Harmonic mean; None if either side is missing, 0.0 if both are 0."""
    if r is None or mr is None:
        return None
    if r + mr <= 0:
        return 0.0
    return 2.0 * r * mr / (r + mr)


def _get(d: dict, key: str, prefix: str = "") -> Optional[float]:
    v = d.get(prefix + key)
    return float(v) if isinstance(v, (int, float)) else None


def runs(paths, ks, prefix, select_on):
    hdr = (f"{'run':52s} {'ep':>3s}" +
           "".join(f"{'R@'+str(k):>9s}{'mR@'+str(k):>9s}{'F1@'+str(k):>9s}"
                   for k in ks))
    print(hdr)
    print("-" * len(hdr))
    for p in paths:
        name = os.path.basename(os.path.dirname(p))
        try:
            h = json.load(open(p))
        except Exception as e:
            print(f"{name:52s}  unreadable: {e}")
            continue
        # Pick the epoch F1 would have selected, and the one mR selected.
        scored = []
        for e in h:
            row = {f"F1@{k}": f1(_get(e, f"R@{k}", prefix),
                                 _get(e, f"mR@{k}", prefix))
                   for k in ks}
            scored.append((e, row))
        by_f1 = max(scored, key=lambda t: (t[1].get(select_on) or -1))
        by_mr = max(scored, key=lambda t: (_get(t[0], f"mR@{select_on.split('@')[1]}",
                                                prefix) or -1))
        for tag, (e, row) in (("F1", by_f1), ("mR", by_mr)):
            cells = ""
            for k in ks:
                r, mr = _get(e, f"R@{k}", prefix), _get(e, f"mR@{k}", prefix)
                cells += (f"{r:9.4f}" if r is not None else f"{'-':>9s}")
                cells += (f"{mr:9.4f}" if mr is not None else f"{'-':>9s}")
                cells += (f"{row[f'F1@{k}']:9.4f}" if row[f"F1@{k}"] is not None else f"{'-':>9s}")
            star = " <- best F1" if tag == "F1" else " <- best mR (what we selected)"
            print(f"{name[:52]:52s} {e['epoch']:3d}{cells}{star}")
            if by_f1[0]["epoch"] == by_mr[0]["epoch"]:
                break  # same epoch, no need to print twice
        print()


def evals(paths, ks):
    hdr = (f"{'eval file':64s}" +
           "".join(f"{'R@'+str(k):>9s}{'mR@'+str(k):>9s}{'F1@'+str(k):>9s}"
                   for k in ks))
    print(hdr)
    print("-" * len(hdr))
    for p in sorted(paths):
        try:
            d = json.load(open(p))
        except Exception:
            continue
        m = d.get("metrics", d)
        if not isinstance(m, dict):
            continue
        if not any(f"R@{k}" in m for k in ks):
            continue
        cells = ""
        for k in ks:
            r, mr = _get(m, f"R@{k}"), _get(m, f"mR@{k}")
            v = f1(r, mr)
            cells += (f"{r:9.4f}" if r is not None else f"{'-':>9s}")
            cells += (f"{mr:9.4f}" if mr is not None else f"{'-':>9s}")
            cells += (f"{v:9.4f}" if v is not None else f"{'-':>9s}")
        rel = os.path.relpath(p)
        print(f"{rel[-64:]:64s}{cells}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", action="store_true", help="scan training histories")
    p.add_argument("--evals", action="store_true", help="scan benchmark eval jsons")
    p.add_argument("--glob", default="runs/train/*/history.json")
    p.add_argument("--eval_glob", default="runs/train/*/zeroshot_*.json")
    p.add_argument("--k", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--prefix", default="dev_",
                   help="metric prefix in history.json ('dev_' = the "
                        "out-of-domain selection split, '' = in-domain val)")
    p.add_argument("--select_on", default="F1@50")
    a = p.parse_args()
    if not (a.runs or a.evals):
        a.runs = True
    if a.runs:
        print(f"=== training histories (prefix '{a.prefix}') ===")
        runs(sorted(glob.glob(a.glob)), a.k, a.prefix, a.select_on)
    if a.evals:
        print("=== benchmark eval jsons ===")
        evals(glob.glob(a.eval_glob), a.k)


if __name__ == "__main__":
    main()
