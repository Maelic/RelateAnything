"""summarize_ovr_boxsource.py — Table 5 (tab:ovrleaderboard) by DETECTOR BACKBONE.

Table 5 places our rows on OvSGTR's own OvR-SGG leaderboard, but our rows run on the
baseline's Swin-T boxes while the two strongest published rows (OvSGTR Swin-B, INOVA)
use Swin-B. This prints, side by side:

  * the reproduction gate  — OvSGTR's own checkpoint scored by OvSGTR's own matcher,
    against its published figures. If Swin-T does not reproduce, nothing else counts.
  * the box-source ceiling — object recall and PAIR recall of each box set. Pair
    recall is the upper bound on detection-mode relation recall for ANY model given
    those boxes, so it is what makes "Swin-T boxes vs Swin-B boxes" quantitative.
  * our rows on each box set, scored with the baseline's matcher.

    python benchmark/ovsgtr/summarize_ovr_boxsource.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BM = os.path.join(ROOT, "runs", "benchmark", "ovr_sgg")

# OvSGTR README, OvR-SGG table (SGDet, VG150 test): Base+Novel then Novel R@20/50/100.
PUBLISHED = {
    "vg-ovr-swint": ([15.85, 20.50, 23.90], [10.17, 13.47, 16.20]),
    "vg-ovr-swinb": ([None, 22.89, 26.65], [None, 16.39, 19.72]),
}
BOXSETS = ["vg-ovr-swint", "vg-ovr-swinb"]
OURS = [("zeroshot_S+", "RelateAnything (ViT-S/16+), zero-shot"),
        ("official_S+", "RelateAnything (ViT-S/16+)")]
K = [20, 50, 100]


def read(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def row(label, bn, nv, extra=""):
    def f(v):
        return "  --  " if v is None else f"{v:6.2f}"
    print(f"  {label:<46}" + " ".join(f(v) for v in bn)
          + "  |" + " ".join(f(v) for v in nv) + ("  " + extra if extra else ""))


def header(title):
    print(f"\n{title}")
    print(f"  {'':<46}{'Base+Novel R@20/50/100':^22}  |{'Novel R@20/50/100':^22}")
    print("  " + "-" * 92)


def main() -> None:
    print("=" * 96)
    print("OvR-SGG leaderboard by detector backbone  (VG150 test, SGDet, OvSGTR's matcher)")
    print("=" * 96)

    header("1. REPRODUCTION GATE — OvSGTR scored by its own matcher (ours) vs published")
    for ck in BOXSETS:
        d = read(os.path.join(BM, f"native_protocol_{ck}.json"))
        pub_bn, pub_nv = PUBLISHED[ck]
        row(f"{ck}  [published]", pub_bn, pub_nv)
        if d is None:
            row(f"{ck}  [ours]", [None] * 3, [None] * 3, "MISSING")
            continue
        m = d["protocols"]["ovsgtr"]
        bn = [m[f"R@{k}"] for k in K]
        nv = [m[f"novel_R@{k}"] for k in K]
        worst = max(abs(a - b) for a, b in zip(bn + nv, pub_bn + pub_nv)
                    if a is not None and b is not None)
        row(f"{ck}  [ours]", bn, nv, f"max|delta| = {worst:.2f}")

    print("\n2. BOX-SOURCE CEILING — upper bound on relation recall given those boxes")
    print(f"  {'box set':<46}{'boxes/img':>10}{'obj recall':>12}{'PAIR recall':>13}")
    print("  " + "-" * 92)
    for ck in BOXSETS:
        d = read(os.path.join(BM, f"ceiling_ovsgtr_{ck}.json"))
        if d is None or not d.get("rows"):
            print(f"  {ck:<46}{'MISSING':>10}")
            continue
        r = d["rows"][0]
        print(f"  {ck:<46}{r['det_per_img']:>10.1f}"
              f"{100 * r['obj_recall']:>11.1f}%{100 * r['pair_recall']:>12.1f}%")

    for ck in BOXSETS:
        header(f"3. OUR ROWS on {ck} boxes (baseline's matcher, every pair scored)")
        for tag, label in OURS:
            d = (read(os.path.join(BM, f"native_protocol_ours_{tag}_on_{ck}_geo10000.json"))
                 or read(os.path.join(BM, f"native_protocol_ours_{tag}_on_{ck}.json")))
            if d is None:
                row(label, [None] * 3, [None] * 3, "MISSING")
                continue
            m = d["protocols"]["ovsgtr"]
            row(label, [m[f"R@{k}"] for k in K], [m[f"novel_R@{k}"] for k in K])

    print("\n4. DELTA for our rows, Swin-T boxes -> Swin-B boxes")
    print("  " + "-" * 92)
    for tag, label in OURS:
        vals = {}
        for ck in BOXSETS:
            d = (read(os.path.join(BM, f"native_protocol_ours_{tag}_on_{ck}_geo10000.json"))
                 or read(os.path.join(BM, f"native_protocol_ours_{tag}_on_{ck}.json")))
            if d:
                vals[ck] = d["protocols"]["ovsgtr"]
        if len(vals) < 2:
            print(f"  {label:<46}  (needs both box sets; have "
                  f"{sorted(vals) or 'none'})")
            continue
        t, b = vals["vg-ovr-swint"], vals["vg-ovr-swinb"]
        d_bn = [b[f"R@{k}"] - t[f"R@{k}"] for k in K]
        d_nv = [b[f"novel_R@{k}"] - t[f"novel_R@{k}"] for k in K]
        row(label, d_bn, d_nv)
    print()


if __name__ == "__main__":
    sys.exit(main())
