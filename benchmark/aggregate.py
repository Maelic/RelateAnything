"""Assemble the OV-SGG benchmark table: every source x protocol, both models, one
evaluator.

DESIGN (LVIS's strategy, translated to relations)
-------------------------------------------------
LVIS's three ideas are federated annotation (absence != negative), a support-based
rare/common/frequent split, and AP over labelled cells only. The relation analogue
needs a fourth thing LVIS does not: a way to tell corpus match apart from
understanding, because SGG benchmarks share predicate vocabularies with their training
corpora. So the benchmark is FOUR AXES, each answering a question no other axis can:

  A1 TRANSFER      recall across sources with DIFFERENT annotation styles, each cell
                   annotated with its overlap statistic so in-domain results can be
                   discounted rather than believed.
  A2 PRECISION     Haystack's explicit negatives — the only axis that can see false
                   positives on rare predicates (fAP / P-AUC).
  A3 OPEN-VOCAB    the full training vocabulary deployed, synonym-tolerant matching at
                   a calibrated tau: does the model MEAN the right relation.
  A4 DEPLOYMENT    SGDet on a SHARED open-vocab detector, reported against the measured
                   pair-recall ceiling that bounds every model identically.
  A5 GRAPH QUALITY an LLM oracle judging the whole graph with NO ground truth, which is
                   the only axis that can credit a relation that is true but unlabelled.
                   Reported ONLY if its controls pass, and always beside the GT-free
                   graph-shape statistics, which hold regardless of the verdict.

There is deliberately NO single headline scalar. Aggregating across sources with
different vocabularies and different matcher breadth is not meaningful (see
); LVIS itself reports a vector (AP/APr/APc/APf) and so do
we. The headline is wR@50 + the bucket split; micro R@50 is reported only for
comparability with the literature and is the metric most inflated by corpus match.

    python benchmark/aggregate.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Which of our runs is "ours" in the table. Overridable with --ours so a new arm can be
# aggregated without editing this file (the SGDet path is derived from it too).
OURS = os.environ.get("AGG_OURS", "relsgg-vits16plus")
OV = "runs/ovsgtr"

# axis, source, protocol label, ours path, ovsgtr path, [subkey]
CELLS = [
    ("A1", "vg150/test", "GT-box, closed vocab",
     f"runs/train/{OURS}/zeroshot_vg150_test_gc.json",
     f"{OV}/ovdr_mega_vg150_test_gtbox_gc.json", None),
    ("A1", "psg/test", "GT-box, closed vocab",
     f"runs/train/{OURS}/zeroshot_psg_test_gc.json",
     f"{OV}/ovdr_mega_psg_test_gtbox_gc.json", None),
    ("A1", "indoorvg/test", "GT-box, closed vocab",
     f"runs/train/{OURS}/zeroshot_indoorvg_test_gc.json",
     f"{OV}/ovdr_mega_indoorvg_test_gtbox_gc.json", None),
    ("A4", "psg/test", "SGDet, shared YOLO-World",
     f"runs/sgdet/{OURS}_psg_test_yoloworld.json",
     "runs/sgdet/ovdr_mega_psg_test_yoloworld_gc.json", "lenient"),
]
HAYSTACK = [
    ("A2", "haystack", "federated negatives (fAP)",
     f"runs/train/{OURS}/haystack_sigmoid.json",
     f"{OV}/ovdr_mega_haystack_test_gtbox_haystack.json", None),
]

RECALL_KEYS = ["R@50", "mR@50", "R@50_rare", "wR@50"]
FED_KEYS = ["mfAP_sup5", "mfAP", "fAP_rare", "mPAUC", "coverage"]

# Measured by benchmark/annotation_overlap.py, per corpus each model was
# trained on: ours = RA-4M, OvSGTR = vg150/train.
OVERLAP = {
    "vg150/test": {"ours": 0.491, "ovsgtr": 1.000},
    "psg/test": {"ours": 0.353, "ovsgtr": 0.573},
    "indoorvg/test": {"ours": 0.454, "ovsgtr": 0.957},
    "haystack": {"ours": 0.353, "ovsgtr": 0.573},
}
CEILING = {("A4", "psg/test"): 0.696}

# A5: pairwise LLM oracle, one file per vocabulary arm, plus the judge-free graph shape.
JUDGE = [("pack", "runs/judge/verdicts_psg_yoloworld_pack.json"),
         ("train", "runs/judge/verdicts_psg_yoloworld_train.json")]
GRAPH_STATS = "runs/judge/graph_stats_psg.json"
# Controls certify the JUDGE, not any one head-to-head, so they pool across runs of the
# same judge configuration. A single 200-image run yields only ~9 DECISIVE scramble
# comparisons (the judge commits on well under half), which is too thin to certify;
# top-up runs at other seeds are pooled here. The sample size of each top-up was fixed
# in advance — growing it until the gate passes would be optional stopping.
CONTROL_POOL = ["runs/judge/verdicts_psg_yoloworld_pack.json",
                "runs/judge/controls_topup_seed1.json"]


def pooled_controls(paths):
    """Pool scramble/padding controls and the primacy statistic across judge runs."""
    scr, pad, first, total, cfgs = [], [], 0, 0, set()
    for p in paths:
        f = Path(p)
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        s = d.get("summary", {})
        cfgs.add((s.get("model"), s.get("votes_per_comparison"), s.get("majority")))
        for r in d.get("records", []):
            (scr if r["kind"] == "scramble" else pad if r["kind"] == "pad" else []).append(r)
            first += r.get("n_first_position", 0)
            total += len(r.get("votes", []))
    if len(cfgs) > 1:
        # Pooling across different judges or vote settings would be meaningless.
        return {"error": f"heterogeneous judge configs pooled: {sorted(cfgs)}"}

    def acc(rs):
        w = sum(r["verdict"] == "A" for r in rs)
        l = sum(r["verdict"] == "B" for r in rs)
        return (w / (w + l) if w + l else 0.0), w + l, len(rs)

    c_acc, c_dec, c_n = acc(scr)
    p_acc, p_dec, p_n = acc(pad)
    primacy = first / total if total else float("nan")
    return {"control_accuracy": c_acc, "control_decisive": c_dec, "control_n": c_n,
            "padding_resistance": p_acc, "padding_decisive": p_dec, "padding_n": p_n,
            "primacy_rate": primacy, "n_runs_pooled": len([p for p in paths
                                                           if Path(p).exists()]),
            "valid": c_acc >= 0.75 and primacy <= 0.60 and c_dec >= 10}


def load(path, subkey=None):
    """subkey applies ONLY to our detbox results, which nest per-protocol dicts
    ('lenient'/'strict'); the interchange results always use 'metrics'."""
    p = Path(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if subkey and subkey in d:
        m = dict(d[subkey])
    else:
        m = dict(d.get("metrics", d))
    # Results written before MIN_POS_FOR_HEADLINE existed lack mfAP_sup5; recover it
    # from the persisted per-class support rather than re-running the GPU eval.
    if "mfAP_sup5" not in m and isinstance(d.get("per_class"), dict):
        sup = [v["fAP"] for v in d["per_class"].values()
               if isinstance(v, dict) and v.get("n_pos", 0) >= 5 and "fAP" in v]
        if sup:
            m["mfAP_sup5"] = sum(sup) / len(sup)
    return m


def fmt(v):
    return "  —  " if v is None else f"{v:.4f}"


def delta(a, b):
    if a is None or b is None or not a:
        return "  —  "
    return f"{(b - a) / a * 100:+.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ours", default="",
                    help="training run to report as ours (default: relsgg-vits16plus, "
                         "or AGG_OURS in the environment)")
    ap.add_argument("--out_json", default="runs/benchmark/ovsgg_benchmark.json")
    ap.add_argument("--out_md", default="runs/benchmark/ovsgg_benchmark.md")
    args = ap.parse_args()
    if args.ours:
        # Re-derive the cell paths for the requested run.
        global OURS, CELLS, HAYSTACK
        _prev, OURS = OURS, args.ours
        CELLS = [(ax, src, pr, po.replace(_prev, OURS), pv, sub)
                 for ax, src, pr, po, pv, sub in CELLS]
        HAYSTACK = [(ax, src, pr, po.replace(_prev, OURS), pv, sub)
                    for ax, src, pr, po, pv, sub in HAYSTACK]

    lines, blob = [], {}
    lines.append("# OV-SGG benchmark — head-to-head\n")
    lines.append("Both models scored by the SAME evaluator, same images, same "
                 "vocabulary, graph-constrained.\n")
    lines.append("`overlap` = fraction of that model's TRAINING relation mass landing "
                 "on the benchmark's exact predicate strings; high = in-domain, so "
                 "micro R@50 there measures corpus match more than understanding.\n")

    for axis, src, proto, po, pv, sub in CELLS:
        mo, mv = load(po, sub), load(pv, sub)
        blob[f"{axis}|{src}|{proto}"] = {"ours": mo, "ovsgtr": mv}
        ov_ours = OVERLAP.get(src, {}).get("ours")
        ov_ovs = OVERLAP.get(src, {}).get("ovsgtr")
        lines.append(f"\n## [{axis}] {src} — {proto}")
        ceil = CEILING.get((axis, src))
        if ceil:
            lines.append(f"\npair-recall ceiling **{ceil:.3f}** (bounds both models)\n")
        lines.append(f"\noverlap: ours **{ov_ours:.1%}** vs OvSGTR **{ov_ovs:.1%}**"
                     f"  → gap {(ov_ovs - ov_ours) * 100:.0f} pts in OvSGTR's favour\n"
                     if ov_ours is not None else "")
        lines.append(f"\n| metric | OvSGTR | ours | delta |")
        lines.append("|---|---|---|---|")
        for k in RECALL_KEYS:
            a = (mv or {}).get(k)
            b = (mo or {}).get(k)
            lines.append(f"| {k} | {fmt(a)} | {fmt(b)} | {delta(a, b)} |")

    for axis, src, proto, po, pv, sub in HAYSTACK:
        mo, mv = load(po, sub), load(pv, sub)
        blob[f"{axis}|{src}|{proto}"] = {"ours": mo, "ovsgtr": mv}
        lines.append(f"\n## [{axis}] {src} — {proto}")
        lines.append("\nOnly axis that measures FALSE POSITIVES. `coverage` is not "
                     "symmetric: OvSGTR enumerates every pair (~100%), our sampler "
                     "prunes (~88%), and unsampled cells score 0 — this favours "
                     "OvSGTR.\n")
        lines.append(f"\n| metric | OvSGTR | ours | delta |")
        lines.append("|---|---|---|---|")
        for k in FED_KEYS:
            a = (mv or {}).get(k)
            b = (mo or {}).get(k)
            lines.append(f"| {k} | {fmt(a)} | {fmt(b)} | {delta(a, b)} |")

    # ---- A5: graph shape (judge-free) then the oracle itself -------------------
    gs = Path(GRAPH_STATS)
    if gs.exists():
        g = json.loads(gs.read_text())
        blob["A5|psg/test|graph shape"] = g
        lines.append("\n## [A5] psg/test — graph shape, shared YOLO-World boxes")
        lines.append("\nNo ground truth and no judge involved, so these hold even if the "
                     "oracle's controls fail. `distinct` is the headline: a model can "
                     "bank R@K while saying one thing over and over, because under a "
                     "graph constraint a collapsed argmax lands on `on`/`in`/`has`.\n")
        lines.append("\n| model | rel/graph | distinct preds | modal share | vocab used |")
        lines.append("|---|---|---|---|---|")
        for k, v in g["models"].items():
            lines.append(f"| {k} | {v['rel_per_graph_mean']:.1f} | "
                         f"{v['distinct_pred_per_graph_mean']:.1f} | "
                         f"{v['modal_predicate_share_mean']:.2f} | {v['vocab_used']} |")
        cap = {k: v["at_cap_frac"] for k, v in g["models"].items()}
        lines.append(f"\nat the `max_rel={g['max_rel']}` cap: "
                     + ", ".join(f"{k} {v:.0%}" for k, v in cap.items())
                     + ". A model pinned at the cap has its coverage UNDERSTATED.\n")

    for arm, path in JUDGE:
        p = Path(path)
        if not p.exists():
            continue
        s = dict(json.loads(p.read_text())["summary"])
        # Certification is a property of the JUDGE, so it comes from the pooled controls
        # rather than from whatever this one arm happened to sample.
        cert = pooled_controls(CONTROL_POOL)
        if "error" not in cert:
            s.update({k: v for k, v in cert.items() if k != "n_runs_pooled"})
            s["controls_pooled_over_runs"] = cert["n_runs_pooled"]
        blob[f"A5|psg/test|oracle {arm}"] = s
        lines.append(f"\n## [A5] psg/test — LLM oracle, vocab={arm} ({s['model']})")
        if s.get("controls_pooled_over_runs", 1) > 1:
            lines.append(f"\n*Controls pooled over {s['controls_pooled_over_runs']} runs "
                         f"of the same judge configuration.*\n")
        # An accuracy without its DECISIVE sample size is not a control: the judge
        # commits on well under half of comparisons, so n_decisive is the real n.
        lines.append(
            f"\ncontrols — scramble {s['control_accuracy']:.2f} "
            f"({s.get('control_decisive', '?')} decisive of {s['control_n']}), "
            f"padding resistance {s['padding_resistance']:.2f} "
            f"({s.get('padding_decisive', '?')} of {s['padding_n']}), "
            f"primacy {s.get('primacy_rate', float('nan')):.2f} "
            f"(0.5 = no position effect), "
            f"undecided {s.get('undecided_rate', float('nan')):.2f}\n")
        if not s["valid"]:
            lines.append("\n**CONTROLS FAILED — win rate withheld: the judge cannot "
                         "reliably tell an intact graph from a scrambled one, or its "
                         "verdict moves with presentation order.**\n")
            continue
        lines.append(f"\n{s['a_name']} wins **{s['win_rate_a']:.1%}** of decisive "
                     f"comparisons ({s['wins_a']}-{s['wins_b']}, {s['ties']} ties/"
                     f"unstable, n={s['n_compared']}). Graph length "
                     f"{s['mean_rel_a']:.1f} vs {s['mean_rel_b']:.1f}.\n")
        if s["padding_resistance"] < 0.5:
            lines.append("\n*The judge showed a net preference for padded graphs, so "
                         "the stratum where the winner was SHORTER is the trustworthy "
                         "one.*\n")
        lines.append("\n| stratum | n | win rate (ours) |")
        lines.append("|---|---|---|")
        for name, st in s["by_length"].items():
            wr = "  —  " if st["win_rate_a"] is None else f"{st['win_rate_a']:.1%}"
            lines.append(f"| {name} | {st['n']} | {wr} |")

    md = "\n".join(lines) + "\n"
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(md)
    Path(args.out_json).write_text(json.dumps(blob, indent=2, sort_keys=True))
    print(md)
    # A5 entries are not two-sided cells, so only check the head-to-head ones.
    missing = [k for k, v in blob.items()
               if "ours" in v and (not v["ours"] or not v["ovsgtr"])]
    missing += [f"A5|{arm}" for arm, p in JUDGE if not Path(p).exists()]
    if missing:
        print("PENDING cells (result file not on disk yet):")
        for k in missing:
            print("  ", k)
    print(f"wrote {args.out_md} and {args.out_json}")


if __name__ == "__main__":
    main()
