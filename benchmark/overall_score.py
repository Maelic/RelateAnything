"""OVS — one balance-enforcing scalar over the OV-SGG axes.

WHY THIS EXISTS, GIVEN SPEC.md ONCE EXCLUDED IT
------------------------------------------------
SPEC.md rejected "a single aggregate score" because averaging raw numbers across
sources with different vocabularies invites the corpus-match gaming the benchmark
exists to expose. That objection is against an ARITHMETIC MEAN OF RAW METRICS,
and it stands. This is a different object, built so the objection does not apply:

  1. Every cell is CHANCE-CORRECTED before it is combined, so a metric whose
     floor is 0.5 (SpatialSense AUC) cannot outweigh one whose floor is 1/V
     (recall over a V-class vocabulary). Raw averages get this catastrophically
     wrong: AUC 0.66 and mR@50 0.22 are not "0.44 on average", they are
     0.32 and 0.20 above their respective floors.
  2. Axes are combined with a HARMONIC mean, which is minimised by imbalance.
     Being excellent on one axis and useless on another cannot produce a good
     score — the precedent is generalised zero-shot learning, where the harmonic
     mean of seen/unseen accuracy replaced the arithmetic mean precisely because
     the latter rewarded models that ignored unseen classes.
  3. It NEVER replaces the vector. `aggregate.py` remains the headline; this is a
     summary of it, and the per-axis components are printed with every score.

So corpus match now BUYS LESS, not more: a model that matches VG150's annotation
style gains on one A1 cell out of four and nothing on A2/A4/A6, and the harmonic
mean drags it back toward its weakest axis.

DEFINITION
----------
Per cell:  norm = clip((x - chance) / (1 - chance), 0, 1)
Per axis:  arithmetic mean of its cells (same capability, different sources)
Overall:   OVS = harmonic mean of the COMPOSITE axes (A1, A2, A4, A5, A6)

A3 is measured, reported, and out of the composite: the baseline cannot be run
on it at all -- its predicate vocabulary is a caption capped at 512 word pieces
-- so a composite containing A3 exists for one of the two models being compared
and the head-to-head cell is a dash.

A5 costs one judge run per model, so a run without one is scored on the
remaining axes and prints its axis set. Do not compare an OVS over four axes
with an OVS over five.
Reported beside it: OVS_arith (arithmetic mean of axes), the WEAKEST axis, and
`balance` = OVS / OVS_arith in (0, 1] — 1.0 exactly when all axes are equal, so
it reads directly as "how specialised is this model".

CHANCE LEVELS (derived, never hand-set — cf.)
  A1 recall     1/V, V = the benchmark's own vocabulary size (a uniform-random
                predicate under the graph constraint)
  A2 fAP        the dataset's positive prevalence (AP of a random scorer)
  A3 open-vocab 1/|deployed vocab| ~ 5e-5, taken as 0   (reported, not composed)
  A4 wR@50      1/V of the source, after dividing by the measured pair-recall
                ceiling of the shared detector
  A5 bits       0, and the cell is already a share of the annotation's own
                information, so no further correction applies
  A6 AUC        0.5

CAVEAT THAT MUST TRAVEL WITH THE NUMBER: OVS is only comparable between models
scored on the SAME AXIS SET. Adding an axis changes every score. The axis set is
printed in the output and stored in the json.

    python benchmark/overall_score.py --out runs/benchmark/ovs.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# A1 sources -> vocabulary size (chance = 1/V under the graph constraint)
A1 = {"vg150": 50, "psg": 56, "indoorvg": 37, "hicodet": 116}
A3 = ["vg150", "psg", "indoorvg"]

# A4 DEPLOYMENT. Detection-mode wR@50 on the shared open-vocabulary detector,
# divided by the MEASURED pair-recall ceiling before it is chance-corrected. The
# raw number is bounded by the detector, which no relation model controls and
# which bounds every model identically; dividing by the ceiling turns it into
# "share of the recoverable pairs recovered", which is the model's part. The
# ceiling is imported rather than repeated -- aggregate.py measured it.
A4_SOURCE = "psg"
A4_FILES = ("detbox/detbox_psg_yoloworld_gc.json",
            "zeroshot_detbox_psg_yoloworld_gc.json")
A4_METRIC = "wR@50"   # support-weighted; the same cell tab:sixaxes reports
# One copy of the ceiling: aggregate.py holds the measured value.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.aggregate import CEILING as _CEILING  # noqa: E402
CEILING = _CEILING[("A4", "psg/test")]

# A5 GRAPH QUALITY. Eq. (4) of the report -- surprisal of the relations a judge
# accepted -- divided by the same quantity on the images' own annotations, at
# matched graph depth. Written by a5b_annotation_reference.py.
#
# The share of the annotation's information is a property of one model on its
# own: it has a floor of 0 (a system whose claims the judge rejects earns no
# bits), needs no chance correction, and is computable without a baseline.
A5_FILE = os.path.join("runs", "benchmark", "a5b",
                       "annotation_reference_top10_matched.json")
A5_KEY = {"ours": "RelateAnything-pack", "base": "OvSGTR"}

# The composite spans these five. A3 stays out: the baseline cannot be run on it
# at all (its vocabulary is one caption capped at 512 tokens), so a composite
# containing A3 exists for one of the two models being compared and the
# head-to-head cell is a dash.
COMPOSITE_AXES = ("A1 transfer", "A2 precision", "A4 detector",
                  "A5 graph quality", "A6 spatial")
# A2 prevalence: positives / labelled cells, from the packs' own annotations.
A2_PREVALENCE = {"haystack": 1.0 / (1.0 + 8.1),   # 8.1:1 neg:pos, SPEC.md sec.3
                 "hicodet": 18954.0 / 309895.0}     # pack v1 (fallback)
# The HICO pack exists in two shapes (18,846 positives over 100,449 labelled
# cells, and 18,954 over 309,895 before duplicate boxes were merged), so the
# chance level depends on which one a run was scored against. The fAP json
# records n_cells, and prevalence is resolved per run from it.
HICO_PREVALENCE_BY_CELLS = {309895: 18954.0 / 309895.0, 100449: 18846.0 / 100449.0}


USE_F1 = False


def f1_or(metrics, mr_key, r_key, use_f1):
    """A1/A3 cell value: mR@K alone, or F1@K = 2*R*mR/(R+mR).

    WHY F1 IS AN OPTION HERE. A1 and A3 are the two recall axes, and each is
    gameable in one direction on mR alone: A1 rewards tail-boosting that
    collapses the head (a change that took PSG mean recall +19% cost `on` 81%), and A3
    rewards head collapse, because generic predicates sit inside nearly every
    accepted synonym set (SPEC.md §2). F1 weights the SMALLER of R and mR, so
    neither trick pays. A2 (fAP) and A6 (AUC) are NOT recall pairs and are
    left alone — forcing F1 onto them would be meaningless.

    ORDER OF OPERATIONS: this returns the RAW F1, which the caller then
    chance-corrects once. F1(norm(R), norm(mR)) != norm(F1(R, mR)) because a
    harmonic mean does not commute with an affine map; correcting once keeps
    the underlying quantity identical to SGG-Benchmark's definition, hence
    citable. Both R and mR share the same 1/V floor (a uniform-random
    predicate scores 1/V expected recall on EVERY class, so micro and macro
    coincide at chance), so there is no floor-mismatch reason to prefer the
    other order.
    """
    mr = metrics.get(mr_key)
    if not use_f1:
        return mr
    r = metrics.get(r_key)
    if r is None or mr is None:
        return None
    return 0.0 if (r + mr) <= 0 else 2.0 * r * mr / (r + mr)


def norm(x, chance):
    if x is None:
        return None
    return max(0.0, min(1.0, (x - chance) / (1.0 - chance)))


def parse_fap(logs, section):
    out, arm = {}, None
    hdr = re.compile(r"#+ +(\S+) — " + section)
    kv = re.compile(r"(\w[\w@]*): +([\d.]+)")
    for log in logs:
        if not Path(log).exists():
            continue
        for line in Path(log).read_text().splitlines():
            m = hdr.search(line)
            if m:
                arm = m.group(1)
                continue
            if arm and "mfAP:" in line and "coverage:" in line:
                out[arm] = {k: float(v) for k, v in kv.findall(line)}
                arm = None
    return out


def harmonic(vals):
    vals = [v for v in vals if v is not None]
    if not vals or min(vals) <= 0:
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--spec_log", nargs="*", default=[],
                   help="optional stdout logs of benchmark/eval_haystack.py; the "
                        "per-run haystack_sigmoid.json is read first")
    p.add_argument("--hico_log", nargs="*", default=[],
                   help="optional stdout logs of benchmark/eval_hico_map.py; the "
                        "per-run hico_fap/haystack_sigmoid.json is read first")
    p.add_argument("--latency", default="runs/benchmark/latency.json",
                   help="from benchmark/latency.py. Reported BESIDE "
                        "OVS, never inside it — see the LATENCY note below.")
    p.add_argument("--runs", nargs="*", default=None,
                   help="run directories to score (default: every directory under "
                        "runs/train that carries the evaluation artifacts)")
    p.add_argument("--out", default="runs/benchmark/ovs.json")
    p.add_argument("--head_to_head", default=None,
                   help="also score the released tower against the baseline on "
                        "the cells both have, e.g. runs/benchmark/ovs_head2head.json")
    p.add_argument("--metric", choices=["mR", "F1"], default="mR",
                   help="cell metric for the two RECALL axes (A1, A3). "
                        "'F1' = harmonic mean of R@K and mR@K per SGG-Benchmark, "
                        "which neither tail-boosting nor head-collapse can game. "
                        "NOTE OVS values are NOT comparable across this choice — "
                        "it changes the axis definition, so recompute every arm.")
    a = p.parse_args()
    global USE_F1
    USE_F1 = (a.metric == "F1")

    # LATENCY IS A COST, NOT AN AXIS. It is deliberately kept out of the
    # harmonic mean: any cost term is optimised by doing less work, so folding
    # speed into a capability composite would let "fast and useless" outrank
    # "slow and correct". It is printed as its own column, plus an explicit
    # efficiency view (OVS per 100 ms) for anyone who wants the ratio — stated
    # rather than smuggled into the score.
    lat = {}
    if os.path.exists(a.latency):
        lj = json.load(open(a.latency))
        lat = {k: v for k, v in lj.get("runs", {}).items()}
        lat_meta = f"{lj.get('device', '?')}, bs1, budget {lj.get('eval_budget')}"
    else:
        lat_meta = ""

    vg_fap = parse_fap(a.spec_log, r"A2 Haystack fAP")
    hi_fap = parse_fap(a.hico_log, r"HICO A2")

    runs = ([Path(r) for r in a.runs] if a.runs
            else sorted(q for q in Path("runs/train").glob("*") if q.is_dir()))
    rows = []
    for d in runs:
        label = d.name
        if not d.exists():
            continue
        cells, axes = {}, {}

        a1 = []
        for src, V in A1.items():
            f = d / f"zeroshot_{src}_test_gc.json"
            if f.exists():
                x = f1_or(json.load(open(f))["metrics"], "mR@50", "R@50", USE_F1)
                if x is None:
                    continue
                n = norm(x, 1.0 / V)
                cells[f"A1/{src}"] = (x, n)
                a1.append(n)
        if a1:
            axes["A1 transfer"] = sum(a1) / len(a1)

        a2 = []
        for src, prev in A2_PREVALENCE.items():
            # PREFER THE JSON, fall back to the log. eval_haystack.py writes
            # metrics.mfAP into haystack_sigmoid.json next to the checkpoint, and
            # that value is identical to the one it prints (verified: 0.75846 in
            # the json vs "mfAP: 0.7585" in the log for the ViT-B full run). The
            # log parser needs a "<arm> — A2 Haystack fAP" banner, so ANY caller
            # that prints a different banner silently loses A2 and its arm drops
            # to INCOMPLETE — which is exactly what happened to the resolution
            # sweep cells. Reading the file the eval actually produced removes
            # that coupling between a metric and a log format.
            jf = d / ("haystack_sigmoid.json" if src == "haystack"
                      else "hico_fap/haystack_sigmoid.json")
            m = None
            if jf.exists():
                m = json.load(open(jf)).get("metrics")
            if not m:
                m = (vg_fap if src == "haystack" else hi_fap).get(arm)
            if m:
                x = m.get("mfAP_sup5", m.get("mfAP"))
                if src == "hicodet":
                    nc = int(m.get("n_cells", 0) or 0)
                    if nc in HICO_PREVALENCE_BY_CELLS:
                        prev = HICO_PREVALENCE_BY_CELLS[nc]
                    elif nc:
                        print(f"!! {arm}: HICO fAP over {nc} cells — unknown pack, "
                              f"using v1 prevalence")
                n = norm(x, prev)
                cells[f"A2/{src}"] = (x, n)
                a2.append(n)
        if a2:
            axes["A2 precision"] = sum(a2) / len(a2)

        # A3 provenance guard. The open-vocabulary matcher's threshold is
        # only meaningful in the text space it was calibrated in, and a
        # threshold carried across spaces can silently reduce A3 to exact
        # string matching. An A3 number is admitted only if it was produced
        # after its space's calibration was fitted; an older one is dropped,
        # which leaves the run incomplete rather than silently comparable.
        cal_mtime = max((os.path.getmtime(p) for p in
                         Path("runs/benchmark").glob("tau_calibration_*.json")),
                        default=0.0)
        a3, a3_stale = [], []
        for src in A3:
            f = d / f"zeroshot_{src}_test_gc_ov.json"
            if not f.exists():
                continue
            j = json.load(open(f))
            fresh = ("tau_eval" in j) or (os.path.getmtime(f) >= cal_mtime)
            x = f1_or(j["metrics"], "SoftmR@50", "SoftR@50", USE_F1)
            if x is None:
                continue
            n = norm(x, 0.0)
            cells[f"A3/{src}"] = (x, n)
            (a3 if fresh else a3_stale).append(n)
        if a3 and not a3_stale:
            axes["A3 open-vocab"] = sum(a3) / len(a3)
        elif a3_stale:
            cells["A3/STALE_TAU"] = (float("nan"), float("nan"))

        for name in A4_FILES:
            f = d / name
            if not f.exists():
                continue
            j = json.load(open(f))
            # "lenient" is the box-matching mode tab:sixaxes reports; a baseline
            # record written by the interchange scorer carries "metrics".
            blk = j.get("lenient") or j.get("metrics") or {}
            x = blk.get(A4_METRIC)
            if x is None:
                continue
            n = norm(x / CEILING, 1.0 / A1[A4_SOURCE])
            cells[f"A4/{A4_SOURCE}"] = (x, n)
            axes["A4 detector"] = n
            break

        f = d / "spatialsense.json"
        if f.exists():
            # MACRO (mean of per-predicate AUC), not the pooled AUC over all
            # 2,758 cells. SpatialSense's predicate mix is dominated by `on`
            # (807) and `behind` (406), the two the model already handles, so
            # the pooled figure largely re-measures them and can rank two
            # models the opposite way round from the macro. Every other axis
            # here is macro; this makes A6 consistent.
            ss = json.load(open(f))
            pp = ss.get("per_predicate") or {}
            x = (sum(v["AUC"] for v in pp.values()) / len(pp)) if pp else ss["AUC"]
            n = norm(x, 0.5)
            cells["A6/spatialsense"] = (x, n)
            cells["A6/spatialsense_pooled"] = (ss["AUC"], norm(ss["AUC"], 0.5))
            axes["A6 spatial"] = n

        if not axes:
            continue
        # Every summary statistic is over the COMPOSITE axes. A3 stays in
        # `axes` because it is measured and reported; it does not enter here.
        comp = {k: axes[k] for k in COMPOSITE_AXES if k in axes}
        if not comp:
            continue
        vals = list(comp.values())
        ovs = harmonic(vals)
        arith = sum(vals) / len(vals)
        weakest = min(comp, key=comp.get)
        row = {"arm": arm, "label": label, "cells": cells, "axes": axes,
               "composite_axes": list(comp), "OVS": ovs, "OVS_arith": arith,
               "balance": (ovs / arith) if arith else 0.0,
               "weakest_axis": weakest, "weakest_value": comp[weakest]}
        L = lat.get(arm)
        if L:
            o = L.get("open_eval_bs1", {})
            c = L.get("closed_eval_bs1", {})
            row["latency"] = {
                "open_ms_mean": o.get("mean"), "open_ms_min": o.get("min"),
                "open_ms_max": o.get("max"), "open_ms_p95": o.get("p95"),
                "closed_ms_mean": c.get("mean"),
                "img_s_batch32": L.get("open_batch32_img_s"),
                "n_params_M": L.get("n_params_M"),
                "stages_ms": L.get("open_stages_ms")}
            if o.get("mean"):
                row["OVS_per_100ms"] = ovs / (o["mean"] / 100.0)
        rows.append(row)

    axis_names = sorted({k for r in rows for k in r["axes"]})
    for r in rows:
        r["missing_axes"] = [n for n in COMPOSITE_AXES if n not in r["axes"]]
        r["complete"] = not r["missing_axes"]
    print(f"composite: {list(COMPOSITE_AXES)}   (OVS is comparable only within "
          f"this set)\nalso measured, not in the composite: "
          f"{[n for n in axis_names if n not in COMPOSITE_AXES]}\n")
    w = max(len(r["label"]) for r in rows)

    def lat_cell(r):
        L = r.get("latency")
        if not L or L.get("open_ms_mean") is None:
            return f"{'--':>22s}"
        return (f"{L['open_ms_mean']:7.1f} "
                f"{'[%.1f-%.1f]' % (L['open_ms_min'], L['open_ms_max']):>14s}")

    def show(rs):
        for r in sorted(rs, key=lambda x: -x["OVS"]):
            print(f"{r['label']:{w}s} {r['OVS']:7.4f} {r['OVS_arith']:7.4f} "
                  f"{r['balance']:6.3f} "
                  + "".join(f"{r['axes'][n]:7.3f}" if n in r["axes"] else f"{'--':>7s}"
                            for n in axis_names)
                  + lat_cell(r)
                  + f"   {r['weakest_axis']} ({r['weakest_value']:.3f})")

    hdr = (f"{'arm':{w}s} {'OVS':>7s} {'arith':>7s} {'bal':>6s} "
           + "".join(f"{n.split()[0]:>7s}" for n in axis_names)
           + f"{'ms/img':>8s}{'[min-max]':>15s}   weakest")
    print(hdr)
    show([r for r in rows if r["complete"]])
    part = [r for r in rows if not r["complete"]]
    if part:
        # Scored on a SUBSET of the axes, so their OVS is not comparable to the
        # rows above (a harmonic mean over fewer axes cannot be penalised by the
        # axis that is absent). Printed separately rather than interleaved.
        print(f"\n-- INCOMPLETE (fewer axes; NOT comparable to the block above) --")
        show(part)

    timed = [r for r in rows if r.get("latency")]
    if timed:
        print(f"\n-- LATENCY ({lat_meta}) — a COST, reported beside OVS and "
              f"never inside its harmonic mean --")
        print(f"{'arm':{w}s} {'ms/img':>8s} {'p95':>7s} {'closed':>8s} "
              f"{'img/s@32':>9s} {'OVS/100ms':>10s}   stages (ms)")
        for r in sorted(timed, key=lambda x: x["latency"]["open_ms_mean"]):
            L = r["latency"]
            st = L.get("stages_ms") or {}
            print(f"{r['label']:{w}s} {L['open_ms_mean']:8.1f} "
                  f"{L['open_ms_p95']:7.1f} {L['closed_ms_mean']:8.1f} "
                  f"{L['img_s_batch32']:9.1f} {r.get('OVS_per_100ms', 0):10.4f}   "
                  + " ".join(f"{k} {v:.1f}" for k, v in
                             sorted(st.items(), key=lambda x: -x[1])))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"axis_set": list(COMPOSITE_AXES), "measured_axes": axis_names,
               "rows": rows,
               "definition": "norm=(x-chance)/(1-chance) per cell; arithmetic "
                             "within axis; HARMONIC across the composite axes",
               "chance": {"A1": "1/V per benchmark vocab", "A2": A2_PREVALENCE,
                          "A3": 0.0, "A4": f"1/{A1[A4_SOURCE]} after / ceiling "
                                           f"{CEILING}", "A6": 0.5}},
              open(a.out, "w"), indent=2)
    print(f"\nsaved → {a.out}")

    if a.head_to_head:
        h = head_to_head(USE_F1)
        Path(a.head_to_head).parent.mkdir(parents=True, exist_ok=True)
        json.dump(h, open(a.head_to_head, "w"), indent=2)
        print("\nhead-to-head (" + h["metric"] + ", cells both models have):")
        for n, m in h["models"].items():
            print(f"  {n:<16} OVS {100*m['OVS']:5.1f}   "
                  + " ".join(f"{k.split()[0]} {v:.3f}" for k, v in m["axes"].items())
                  + (f"   MISSING {m['missing_axes']}" if m["missing_axes"] else ""))
        print(f"saved → {a.head_to_head}")



# ========================================================= head-to-head ====
# The composite row of the paper's six-axis table. `main()` above scores OUR
# arms; this scores our released tower and the baseline side by side, on the
# CELLS BOTH MODELS HAVE. Two deliberate differences from the per-arm loop:
#
#   * A2 uses the haystack cell alone. Our arms average haystack with a HICO-DET
#     fAP cell; the baseline was never run on HICO fAP, and averaging a two-cell
#     axis against a one-cell axis compares different quantities. Restricting to
#     the common cell RAISES our A2 (0.608 -> 0.692), so it is stated here
#     rather than left implicit.
#   * A1's HICO-DET cell comes from the ZERO-SHOT tower for us, because the
#     released tower saw HICO-DET train and the baseline did not. This matches
#     the A1 rows of the same table; the per-arm loop above scores each arm on
#     its own HICO number, which is the right thing for arm selection and the
#     wrong thing for a head-to-head.
# The head-to-head compares one released model against the baseline. The
# HICO cells come from the zero-shot sibling, which never saw HICO-DET.
OURS_RUN = os.environ.get("OVS_RUN", "runs/train/relsgg-vits16plus")
OURS_ZS_RUN = os.environ.get("OVS_RUN_ZEROSHOT", "runs/train/relsgg-vits16plus-zeroshot")
BASE_DIR = os.environ.get("OVS_BASELINE", "runs/ovsgtr")


def _metrics(path, block="metrics"):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    return j.get(block) or j.get("lenient") or j.get("metrics")


def _macro_auc(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    pp = j.get("per_predicate") or {}
    return (sum(v["AUC"] for v in pp.values()) / len(pp)) if pp else j.get("AUC")


def head_to_head(use_f1: bool) -> dict:
    def a1(src, ours):
        if ours:
            run = OURS_ZS_RUN if src == "hicodet" else OURS_RUN
            m = _metrics(f"{run}/zeroshot_{src}_test_gc.json")
        else:
            m = _metrics(f"{BASE_DIR}/ovdr_mega_{src}_test_gtbox_gc.json")
        return f1_or(m, "mR@50", "R@50", use_f1) if m else None

    def a2(ours):
        m = (_metrics(f"{OURS_RUN}/haystack_sigmoid.json") if ours else
             _metrics(f"{BASE_DIR}/ovdr_mega_haystack_test_gtbox_haystack.json"))
        return m.get("mfAP_sup5", m.get("mfAP")) if m else None

    def a4(ours):
        m = (_metrics(f"{OURS_RUN}/zeroshot_detbox_psg_yoloworld_gc.json", "lenient")
             if ours else _metrics("runs/sgdet/ovdr_mega_psg_test_yoloworld_gc.json"))
        return m.get(A4_METRIC) if m else None

    def a5(ours):
        j = json.load(open(A5_FILE)) if os.path.exists(A5_FILE) else None
        if not j:
            return None
        v = j["per_system"].get(A5_KEY["ours" if ours else "base"], {})
        # Can exceed 1 at deployed depth (the annotation is sparse); the matched
        # depth used here does not, and the composite clips anyway.
        return v.get("share_of_annotation")

    def a6(ours):
        return _macro_auc(f"{OURS_RUN}/spatialsense.json" if ours else
                          f"{BASE_DIR}/ovdr_mega_spatialsense_test.json")

    out = {"axis_set": list(COMPOSITE_AXES), "metric": "F1" if use_f1 else "mR",
           "a2_cells": ["haystack"], "models": {}}
    for name, ours in (("RelateAnything", True), ("OvSGTR", False)):
        cells, axes = {}, {}
        vals = [(f"A1/{s}", a1(s, ours), 1.0 / V) for s, V in A1.items()]
        got = [(k, x, c) for k, x, c in vals if x is not None]
        if got:
            for k, x, c in got:
                cells[k] = (x, norm(x, c))
            axes["A1 transfer"] = sum(cells[k][1] for k, _, _ in got) / len(got)
        for key, axis, x, chance in (
                ("A2/haystack", "A2 precision", a2(ours), A2_PREVALENCE["haystack"]),
                ("A5/psg_matched", "A5 graph quality", a5(ours), 0.0),
                ("A6/spatialsense", "A6 spatial", a6(ours), 0.5)):
            if x is not None:
                cells[key] = (x, norm(x, chance))
                axes[axis] = cells[key][1]
        x = a4(ours)
        if x is not None:
            cells[f"A4/{A4_SOURCE}"] = (x, norm(x / CEILING, 1.0 / A1[A4_SOURCE]))
            axes["A4 detector"] = cells[f"A4/{A4_SOURCE}"][1]
        comp = {k: axes[k] for k in COMPOSITE_AXES if k in axes}
        out["models"][name] = {
            "cells": cells, "axes": axes, "OVS": harmonic(list(comp.values())),
            "missing_axes": [k for k in COMPOSITE_AXES if k not in axes],
            "weakest_axis": min(comp, key=comp.get) if comp else None}
    return out


if __name__ == "__main__":
    main()
