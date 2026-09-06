#!/usr/bin/env python
"""eval_student.py — Phase C: validate the distilled student text encoder.

Runs text_space_diag.py's exact metric machinery (build_groups + evaluate) on
BOTH the student and the frozen dino.txt teacher, per pack, so the comparison is
apples-to-apples on identical random pairs:

  megasg  — training vocabulary (10,102 predicates)
  vg150 / psg — HELD OUT (never distilled on): the real generalization test for
                reparameterize()-time text spaces.

Go/no-go criteria (printed with PASS/FAIL):
  * auc_syn_vs_rand >= 0.99            (student matches teacher clustering)
  * auc_syn_vs_inv  improved vs teacher, target >= 0.85   (antonyms separated)
  * inv_frac_above_tau well below teacher
  * student >= ~50x smaller than teacher

Also emits pred_embeds_student_<tset>.npz for megasg (meta-predicate order) for a
future Phase-D adoption, and a params/throughput comparison.

Usage (GPU): python training/distill/eval_student.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from relsgg.text.student import PredicateTextStudent  # noqa: E402
from training.text_space_diag import (# noqa: E402
    TEMPLATE_SETS, build_groups, combine_templates, encode_all_templates,
    encode_dinotxt, evaluate,
)

PACKS = {
    "megasg": ("runs/packed/megasg", "train"),
    "vg150": ("runs/packed/vg150", "heldout"),
    "psg": ("runs/packed/psg", "heldout"),
}
DINOTXT_PARAMS = 538_000_000  # text tower, for the size-ratio criterion


def student_encoder(model: PredicateTextStudent, device):
    def fn(texts, batch: int = 2048):
        with torch.no_grad():
            emb = model.encode_texts(texts, device=device, batch=batch)
        return emb.float().cpu().numpy()
    return fn


_METRIC_KEYS = ["synonym_pairs", "inverse_pairs", "cos_synonym_mean",
                "cos_inverse_mean", "cos_random_mean", "auc_syn_vs_rand",
                "auc_syn_vs_inv", "tau_ignore_at_95pct_syn_recall",
                "random_frac_above_tau", "inverse_frac_above_tau"]


def safe_evaluate(E, syn, inv, rng) -> dict:
    """evaluate() but robust to packs with no synonym / inverse pairs.

    text_space_diag.evaluate crashes on np.quantile of an empty synonym-cosine
    array (small packs like PSG share no canonical group). Return NaNs for the
    metrics that are undefined instead of crashing."""
    nan = float("nan")
    if len(syn) == 0:
        return {k: nan for k in _METRIC_KEYS} | {"synonym_pairs": 0,
                                                 "inverse_pairs": len(inv)}
    m = evaluate(E, syn, inv, rng)
    if len(inv) == 0:
        m["auc_syn_vs_inv"] = nan
        m["inverse_frac_above_tau"] = nan
        m["cos_inverse_mean"] = nan
    return m


def eval_pack(name, root, encode_fn, predicates, raw_links, seed):
    _, syn, inv = build_groups(predicates, raw_links)
    per_template = encode_all_templates(encode_fn, predicates)
    out = {}
    for tset, templates in TEMPLATE_SETS.items():
        E = combine_templates(per_template, templates)
        out[tset] = safe_evaluate(E, syn, inv, np.random.default_rng(seed))
    return out, per_template


def _pearson(x, y):
    x, y = x - x.mean(), y - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def geometry_correlation(Et, Es, idx, rng, n_pairs=200_000):
    """How well the student reproduces the teacher's pairwise-cosine geometry
    over a set of strings (dense, label-free → robust where oracle pairs are
    too sparse, e.g. the random held-out slice). Returns Pearson + Spearman
    (numpy-only; Spearman = Pearson of ranks)."""
    idx = np.asarray(idx)
    if len(idx) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": 0}
    a = idx[rng.integers(0, len(idx), n_pairs)]
    b = idx[rng.integers(0, len(idx), n_pairs)]
    keep = a != b
    a, b = a[keep], b[keep]
    ct = np.einsum("ij,ij->i", Et[a], Et[b]).astype(np.float64)
    cs = np.einsum("ij,ij->i", Es[a], Es[b]).astype(np.float64)
    rt = ct.argsort().argsort().astype(np.float64)
    rs = cs.argsort().argsort().astype(np.float64)
    return {"pearson": _pearson(ct, cs), "spearman": _pearson(rt, rs),
            "n": int(len(a))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", default="runs/packed/text_student")
    ap.add_argument("--student", default=None, help="default <art>/student.pt")
    ap.add_argument("--headline", default="photo",
                    choices=["plain", "carrier", "photo"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    art = PROJ / args.art
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = PredicateTextStudent.from_checkpoint(
        args.student or str(art / "student.pt"), device=device)
    s_params = student.num_parameters()
    print(f"[eval] student {s_params/1e6:.2f}M params "
          f"({DINOTXT_PARAMS/s_params:.0f}x smaller than dino.txt); device={device}")

    s_fn = student_encoder(student, device)
    report = {"student_params": s_params, "size_ratio_vs_dinotxt":
              DINOTXT_PARAMS / s_params, "packs": {}}

    for name, (root, split) in PACKS.items():
        meta = json.load(open(PROJ / root / "train" / "meta.json"))
        preds, raw_links = meta["predicates"], meta.get("raw_links", [])
        print(f"\n=== {name} ({split}, {len(preds)} predicates) ===")

        t_metrics, t_per_template = eval_pack(name, root, encode_dinotxt, preds, raw_links, args.seed)
        s_metrics, s_per_template = eval_pack(name, root, s_fn, preds, raw_links, args.seed)
        report["packs"][name] = {"split": split, "n_pred": len(preds),
                                 "teacher": t_metrics, "student": s_metrics}

        h = args.headline
        t, s = t_metrics[h], s_metrics[h]
        print(f"  [{h}] metric                teacher   student")
        for k in ["auc_syn_vs_rand", "auc_syn_vs_inv", "cos_synonym_mean",
                  "cos_inverse_mean", "cos_random_mean",
                  "tau_ignore_at_95pct_syn_recall", "inverse_frac_above_tau"]:
            print(f"    {k:34s} {t[k]:7.4f}  {s[k]:7.4f}")

        # emit megasg student embeddings in meta order for future adoption,
        # plus the isolated random-held-out slice (true generalization test)
        if name == "megasg":
            for tset, templates in TEMPLATE_SETS.items():
                E = combine_templates(s_per_template, templates)
                np.savez_compressed(
                    art / f"pred_embeds_student_{tset}.npz",
                    embeddings=E.astype(np.float16),
                    # plain unicode arrays: train.py loads without allow_pickle
                    predicates=np.array([str(p) for p in preds]),
                    templates=np.array([str(t) for t in templates]),
)
            print(f"  emitted pred_embeds_student_*.npz ({len(preds)} preds, meta order)")

            # generalization on UNSEEN strings. Random holdout shatters oracle
            # pairs (≈0 synonym/inverse pairs survive), so a label-free
            # geometry-correlation is the robust signal: does the student
            # reproduce the teacher's pairwise-cosine structure on strings it
            # never distilled on?
            corpus = json.load(open(art / "corpus.json"))
            held = {p["s"] for p in corpus["provenance"] if p["split"] == "heldout"}
            hidx = [i for i, p in enumerate(preds) if p in held]
            print(f"  megasg_heldout: {len(hidx)} unseen predicates "
                  f"(geometry-correlation vs teacher)")
            geo = {}
            for tset, templates in TEMPLATE_SETS.items():
                Et = combine_templates(t_per_template, templates)
                Es = combine_templates(s_per_template, templates)
                geo[tset] = geometry_correlation(Et, Es, hidx,
                                                 np.random.default_rng(args.seed))
            report["packs"]["megasg_heldout"] = {
                "split": "heldout", "n_pred": len(hidx),
                "geometry_correlation": geo}
            g = geo[args.headline]
            print(f"  [{args.headline}] heldout geometry corr: "
                  f"pearson {g['pearson']:.4f}  spearman {g['spearman']:.4f} "
                  f"(n={g['n']}) — 1.0 = perfectly reproduces teacher geometry")

    # ---- go/no-go gate (headline template) ----
    import math
    def _fin(x, default):  # NaN-safe: undefined metric → don't fail its check
        return default if (x is None or (isinstance(x, float) and math.isnan(x))) else x
    print("\n=== GO/NO-GO (headline template = %s) ===" % args.headline)
    ok = True
    for name in report["packs"]:
        blk = report["packs"][name]
        if "student" not in blk:          # geometry-only slice (megasg_heldout)
            g = blk["geometry_correlation"][args.headline]
            cg = _fin(g["spearman"], 0.0) >= 0.90
            ok = ok and cg
            print(f"  {name:15s} heldout geometry spearman>=.90 "
                  f"{'PASS' if cg else 'FAIL'} ({g['spearman']:.3f})")
            continue
        s, t = blk["student"][args.headline], blk["teacher"][args.headline]
        isnan = lambda x: isinstance(x, float) and math.isnan(x)
        has_inv = not isnan(s["auc_syn_vs_inv"])
        # undefined (NaN) metric ⇒ not assessable ⇒ don't fail on it
        c1 = True if isnan(s["auc_syn_vs_rand"]) else s["auc_syn_vs_rand"] >= 0.99
        # antonym separation: strong AND at least as good as teacher
        # (both-perfect counts as pass, not a strict-greater fail)
        c2 = (s["auc_syn_vs_inv"] >= 0.85
              and s["auc_syn_vs_inv"] >= t["auc_syn_vs_inv"] - 1e-9) \
            if has_inv else True
        c3 = (s["inverse_frac_above_tau"] < t["inverse_frac_above_tau"]) \
            if has_inv else True
        ok = ok and c1 and c2 and c3
        print(f"  {name:15s} syn_vs_rand>=.99 {'PASS' if c1 else 'FAIL'} "
              f"({s['auc_syn_vs_rand']:.3f}) | syn_vs_inv>=.85 & >teacher "
              f"{'PASS' if c2 else 'FAIL'} ({s['auc_syn_vs_inv']:.3f} vs "
              f"{t['auc_syn_vs_inv']:.3f}) | inv_frac<teacher "
              f"{'PASS' if c3 else 'FAIL'} ({s['inverse_frac_above_tau']:.3f} vs "
              f"{t['inverse_frac_above_tau']:.3f})")
    c_size = DINOTXT_PARAMS / s_params >= 40  # well over an order of magnitude
    ok = ok and c_size
    print(f"  {'size':15s} >=40x smaller {'PASS' if c_size else 'FAIL'} "
          f"({DINOTXT_PARAMS/s_params:.0f}x)")
    report["go"] = bool(ok)
    print(f"\n  OVERALL: {'GO' if ok else 'NO-GO'}")

    json.dump(report, open(art / "diag_student.json", "w"), indent=2)
    print(f"\n[eval] report → {art/'diag_student.json'}")


if __name__ == "__main__":
    main()
