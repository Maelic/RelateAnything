"""Is the baseline just predicting `on` everywhere? Measure it, don't eyeball it.

The qualitative panels suggest a collapse onto one predicate. Ten hand-viewed images
cannot establish that, so this quantifies it three ways that fail differently:

  1. DEPLOYED graph  — what the model actually emits (top-K by its own score).
  2. ALL PAIRS       — the argmax over every candidate pair, so the finding cannot be an
                       artifact of the top-K cut.
  3. RANK of `on`    — how often `on` is merely top-1 versus dominating the whole
                       ranking. A model that ranks `on` first but a sensible predicate
                       second is differently broken from one that has a single mode.

Each is reported against the GT predicate prior for the same split: over-prediction is
only meaningful relative to how often the predicate is genuinely correct. And each is
reported for BOTH the GT-box and shared-detector arms, because a collapse that appears
only with detector boxes would be a detection artifact, not a relation-head property.

    python benchmark/predicate_bias.py --pack runs/packed/psg/test \\
        --pred OvSGTR-gtbox=runs/ovsgtr/ovdr_mega_psg_test_gtbox.npz \\
               OvSGTR-sgdet=runs/sgdet/ovdr_mega_psg_test_yoloworld.npz \\
        --deployed RelSGG=runs/judge/relsgg_psg_test_yoloworld_pack.npz \\
        --out runs/benchmark/predicate_bias_psg.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.llm_judge import load_npz, triplets  # noqa: E402


def entropy_eff(counts):
    """exp(H) — the effective number of predicates in use. 1.0 = says one thing."""
    n = sum(counts.values())
    if not n:
        return 0.0
    p = np.array([c / n for c in counts.values()], dtype=np.float64)
    return float(np.exp(-(p * np.log(p)).sum()))


def gt_prior(pack):
    """Predicate distribution of the ground truth for this split."""
    rels = np.load(Path(pack) / "rels.npy", mmap_mode="r")
    meta = json.loads((Path(pack) / "meta.json").read_text())
    preds = list(meta["predicates"])
    c = Counter()
    ids, cnt = np.unique(np.asarray(rels[:, 2]), return_counts=True)
    for i, k in zip(ids, cnt):
        if 0 <= int(i) < len(preds):
            c[preds[int(i)]] = int(k)
    return c, preds


def analyse_full(d, preds, top_n=5):
    """All-pairs argmax + rank statistics. Requires the full pair x V score matrix."""
    if d["_meta"].get("emit") == "topk":
        return None
    bg = int(d["_meta"].get("bg_column", -1))
    sc = d["rel_scores"]
    if sc.dtype != np.float32:
        sc = sc.astype(np.float32)
    if bg >= 0:
        sc = np.delete(sc, bg, axis=1)
    names = [str(x) for x in d["predicates"]]
    arg = sc.argmax(1)
    cnt = Counter()
    ids, k = np.unique(arg, return_counts=True)
    for i, n in zip(ids, k):
        cnt[names[int(i)]] = int(n)

    # Rank of the globally-modal predicate within each pair's own ranking.
    modal = cnt.most_common(1)[0][0]
    mi = names.index(modal)
    order = np.argsort(-sc, axis=1)
    rank = (order == mi).argmax(1)          # 0-based rank of `modal` per pair
    # Runner-up: what the model says when `modal` is removed from contention.
    second = order[:, 1]
    scnt = Counter()
    ids, k = np.unique(second, return_counts=True)
    for i, n in zip(ids, k):
        scnt[names[int(i)]] = int(n)
    # The obvious objection to an all-pairs count is that most candidate pairs are
    # genuinely unrelated, so their argmax is meaningless. Two restrictions answer it:
    # pairs the model prefers over background, and its own most-confident decile.
    strata = {}
    best = sc.max(1)
    bgcol = (d["rel_scores"][:, bg].astype(np.float32) if bg >= 0 else None)
    for label, mask in (("asserted", best > bgcol if bgcol is not None
                         else np.ones(len(best), bool)),
                        ("top_decile", best >= np.quantile(best, 0.9))):
        sub = arg[mask]
        c2 = Counter()
        ids2, k2 = np.unique(sub, return_counts=True)
        for i, n in zip(ids2, k2):
            c2[names[int(i)]] = int(n)
        t2 = max(1, sum(c2.values()))
        strata[label] = {
            "n_pairs": int(mask.sum()),
            "modal_share": round(c2[cnt.most_common(1)[0][0]] / t2, 4),
            "effective_vocab": round(entropy_eff(c2), 2),
            "top3": [(p, round(c / t2, 4)) for p, c in c2.most_common(3)],
        }

    tot = int(sc.shape[0])
    return {
        "n_pairs": tot, "strata": strata,
        "top5": [(p, round(c / tot, 4)) for p, c in cnt.most_common(top_n)],
        "modal": modal, "modal_share": round(cnt[modal] / tot, 4),
        "effective_vocab": round(entropy_eff(cnt), 2),
        "n_predicates_ever_argmax": len(cnt),
        "modal_mean_rank": round(float(rank.mean()), 3),
        "modal_top1_frac": round(float((rank == 0).mean()), 4),
        "modal_top3_frac": round(float((rank < 3).mean()), 4),
        "runner_up_top5": [(p, round(c / tot, 4)) for p, c in scnt.most_common(top_n)],
    }


def analyse_deployed(d, cat_names, max_rel, rel_frac, limit=0):
    n = len(d["image_index"])
    if limit:
        n = min(n, limit)
    cnt, homog, sizes = Counter(), [], []
    for i in range(n):
        t = triplets(d, i, max_rel, rel_frac, cat_names)
        if not t:
            continue
        c = Counter(x[1] for x in t)
        cnt.update(c)
        homog.append(c.most_common(1)[0][1] / len(t))
        sizes.append(len(t))
    tot = sum(cnt.values())
    homog = np.array(homog)
    return {
        "n_images": len(sizes), "n_relations": tot,
        "top5": [(p, round(c / tot, 4)) for p, c in cnt.most_common(5)],
        "modal": cnt.most_common(1)[0][0] if cnt else None,
        "modal_share": round(cnt.most_common(1)[0][1] / tot, 4) if tot else 0.0,
        "effective_vocab": round(entropy_eff(cnt), 2),
        "vocab_used": len(cnt),
        "mean_within_image_modal_share": round(float(homog.mean()), 4),
        "frac_images_single_predicate": round(float((homog == 1.0).mean()), 4),
        "frac_images_90pct_one_predicate": round(float((homog >= 0.9).mean()), 4),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", required=True)
    p.add_argument("--pred", nargs="*", default=[], help="NAME=path (full matrices)")
    p.add_argument("--deployed", nargs="*", default=[], help="NAME=path (top-k emit)")
    p.add_argument("--max_rel", type=int, default=20)
    p.add_argument("--rel_frac", type=float, default=0.7)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    gt, preds = gt_prior(a.pack)
    tot_gt = sum(gt.values())
    cat_names = list(json.loads((Path(a.pack) / "meta.json").read_text())["categories"])

    out = {"pack": a.pack, "max_rel": a.max_rel, "rel_frac": a.rel_frac,
           "gt": {"n_relations": tot_gt, "n_predicates": len(gt),
                  "effective_vocab": round(entropy_eff(gt), 2),
                  "top5": [(k, round(v / tot_gt, 4)) for k, v in gt.most_common(5)]},
           "models": {}}
    print(f"GROUND TRUTH  {tot_gt} relations, effective vocab "
          f"{out['gt']['effective_vocab']}, top: "
          + ", ".join(f"{k} {v:.1%}" for k, v in out["gt"]["top5"]))
    print()

    for spec in list(a.pred) + list(a.deployed):
        name, _, path = spec.partition("=")
        d = load_npz(path)
        e = {"source": path,
             "deployed": analyse_deployed(d, cat_names, a.max_rel, a.rel_frac)}
        full = analyse_full(d, preds)
        if full:
            e["all_pairs"] = full
        out["models"][name] = e

        dep = e["deployed"]
        print(f"{name}")
        print(f"  deployed: modal `{dep['modal']}` {dep['modal_share']:.1%} of all "
              f"emitted relations | effective vocab {dep['effective_vocab']} | "
              f"{dep['vocab_used']} predicates used")
        print(f"             images that are >=90% one predicate: "
              f"{dep['frac_images_90pct_one_predicate']:.1%} "
              f"(100%: {dep['frac_images_single_predicate']:.1%})")
        if full:
            print(f"  all pairs: modal `{full['modal']}` is argmax on "
                  f"{full['modal_share']:.1%} of {full['n_pairs']:,} pairs | "
                  f"effective vocab {full['effective_vocab']} | "
                  f"{full['n_predicates_ever_argmax']} predicates ever win")
            print(f"             `{full['modal']}` mean rank "
                  f"{full['modal_mean_rank']:.2f}, in top-3 on "
                  f"{full['modal_top3_frac']:.1%} of pairs")
            print(f"             runner-up: " + ", ".join(
                f"{k} {v:.1%}" for k, v in full["runner_up_top5"][:3]))
        print()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
