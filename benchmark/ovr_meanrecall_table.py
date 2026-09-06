"""ovr_meanrecall_table.py — the OvR-SGG comparison, reported as MEAN recall.

WHY NOT R@K. VG150's "novel" subset is not a balanced held-out set: `on` alone is
62.3% of its GT relations by instance count, `of` 19.1% and `in` 11.2% -- 92.6% in
three predicates. Instance-weighted Novel R@K is therefore close to a measurement of
`on`, and a model that transfers to `on` and to nothing else wins the column outright.
Measured on Swin-T boxes, OvSGTR is non-zero on 3 of the 15 novel predicates and on
13 of all 50; it beats us on `on` (18.7 vs 11.9) and loses on `of` (4.8 vs 11.7), `in`
(4.7 vs 16.3) and `riding` (0.0 vs 47.5). Micro hands them the Novel column; mean
recall, which gives every predicate one vote, reverses it ~3.8x.

mR@K = mean_c (tp_c / gt_c) over classes with GT support -- the definition already
used by eval_ovsgtr_novel.py and the SGG convention generally, for exactly this
reason.

Both systems are scored by score_native_protocol.py --per_class, which transcribes
OvSGTR's own recall code; the box sets are byte-identical (their detections, consumed
with det_conf 0.0) and `labels` differ only by index base, which each file declares
and the scorer subtracts. Protocol `ovsgtr` is MANY-TO-MANY, as published: a GT object
covered by k duplicate detections gives k chances (duplicate_detection_factor 2.99
here), so the absolute numbers are inflated for both models equally.

    python benchmark/ovr_meanrecall_table.py --boxes vg-ovr-swint
"""
from __future__ import annotations

import argparse
import json
import os


def load(path: str):
    if not os.path.exists(path):
        return None
    return json.load(open(path))["per_class"]


def mean_recall(pc: dict, k: int, novel_only: bool):
    gt, nv, tp = pc["gt_support"], set(pc["novel"]), pc["tp"]["ovsgtr"][str(k)]
    cls = [p for p, g in gt.items() if g > 0 and (p in nv if novel_only else True)]
    if not cls:
        return None
    r = [tp.get(p, 0) / gt[p] for p in cls]
    return (100 * sum(r) / len(r), len(cls), sum(x > 0 for x in r))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--boxes", default="vg-ovr-swint")
    p.add_argument("--dir", default="runs/benchmark/ovr_sgg")
    p.add_argument("--systems", nargs="+",
                   default=["ours=heldout", "OvSGTR=ovsgtr"],
                   help="label=tag, reading perclass_<tag>_<boxsuffix>.json")
    p.add_argument("--topk", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--out", default="")
    a = p.parse_args()

    suf = a.boxes.replace("vg-ovr-", "")
    systems = []
    for spec in a.systems:
        label, _, tag = spec.partition("=")
        pc = load(os.path.join(a.dir, f"perclass_{tag}_{suf}.json"))
        if pc is None:
            print(f"  (missing perclass_{tag}_{suf}.json)")
            continue
        systems.append((label, pc))
    if len(systems) < 2:
        raise SystemExit("need at least two systems")

    print(f"\nOvR-SGG on {a.boxes} boxes — MEAN RECALL (mR@K = mean_c tp_c/gt_c)")
    print("protocol `ovsgtr` (their code, many-to-many); identical box set\n")
    hdr = f"{'subset':<18}{'K':>5}" + "".join(f"{l:>12}" for l, _ in systems) \
        + f"{'ratio':>8}" + "".join(f"{l[:6]+' nz':>12}" for l, _ in systems)
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for label, novel_only in (("Base+Novel (50)", False), ("Novel (15)", True)):
        for k in a.topk:
            vals = [mean_recall(pc, k, novel_only) for _, pc in systems]
            if any(v is None for v in vals):
                continue
            ratio = vals[0][0] / vals[1][0] if vals[1][0] else float("inf")
            print(f"{label:<18}{k:>5}" + "".join(f"{v[0]:>12.2f}" for v in vals)
                  + f"{ratio:>7.1f}x"
                  + "".join(f"{v[2]:>9}/{v[1]}" for v in vals))
            rows[f"{label}@{k}"] = {l: {"mR": v[0], "n_classes": v[1],
                                        "nonzero": v[2]}
                                    for (l, _), v in zip(systems, vals)}

    # Per-class novel detail: the table above is only credible with this beside it.
    print(f"\nper-predicate Novel R@50 (GT support, then recall per system)")
    gt = systems[0][1]["gt_support"]
    nv = [p for p in systems[0][1]["novel"] if gt.get(p, 0) > 0]
    nv.sort(key=lambda p: -gt[p])
    tot = sum(gt[p] for p in nv)
    h2 = f"{'predicate':<15}{'GT':>8}{'share':>8}" + "".join(f"{l:>10}" for l, _ in systems)
    print(h2)
    print("-" * len(h2))
    for pr in nv:
        cells = "".join(f"{100 * pc['tp']['ovsgtr']['50'].get(pr, 0) / gt[pr]:>10.1f}"
                        for _, pc in systems)
        print(f"{pr:<15}{gt[pr]:>8,}{gt[pr] / tot:>8.1%}{cells}")
    print(f"\n`on` is {gt.get('on', 0) / tot:.1%} of the novel GT — which is why the "
          f"instance-weighted\ncolumn tracks one predicate and mean recall does not.")

    if a.out:
        json.dump({"boxes": a.boxes, "mean_recall": rows}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
