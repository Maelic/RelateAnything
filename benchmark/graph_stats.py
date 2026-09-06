"""A5 companion: descriptive statistics of the graphs each model actually emits.

These are GT-free and judge-free, so they stay valid even if the LLM oracle's controls
fail. They describe the *shape* of a deployed graph — how much it says, how many
different things it says, how often it repeats itself — which is precisely what
per-triplet recall cannot see.

The headline is `distinct_pred_per_graph`. A model can score respectably on R@K while
emitting one predicate over and over: under a graph constraint the argmax of a collapsed
model lands on `on`/`in`/`has`, which are correct often enough to bank recall. Measured
on PSG/test with shared YOLO-World boxes, OvSGTR emits 1.7 distinct predicates across a
12-relation graph. That is the head collapse A1-A4 only ever saw indirectly.

    python benchmark/graph_stats.py \
        --graphs RelSGG=runs/judge/relsgg_psg_test_yoloworld_pack.npz \
                 OvSGTR=runs/sgdet/ovdr_mega_psg_test_yoloworld.npz \
        --pack runs/packed/psg/test --out runs/judge/graph_stats_psg.json
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


def stats(D, cat_names, max_rel, rel_frac, limit=0):
    n = len(D["image_index"])
    if limit:
        n = min(n, limit)
    lens, div, rep, top1, preds = [], [], [], [], Counter()
    for i in range(n):
        t = triplets(D, i, max_rel, rel_frac, cat_names)
        if not t:
            continue
        c = Counter(x[1] for x in t)
        lens.append(len(t))
        div.append(len(c))
        # share of the graph spent on its single most-used predicate
        rep.append(c.most_common(1)[0][1] / len(t))
        top1.append(c.most_common(1)[0][0])
        preds.update(c)
    lens, div, rep = np.array(lens), np.array(div), np.array(rep)
    tot = sum(preds.values())
    return {
        "n_images": int(len(lens)),
        "rel_per_graph_mean": float(lens.mean()),
        "rel_per_graph_median": float(np.median(lens)),
        "at_cap_frac": float((lens >= max_rel).mean()),
        "distinct_pred_per_graph_mean": float(div.mean()),
        "distinct_pred_per_graph_median": float(np.median(div)),
        # 1.0 = the graph says one thing over and over
        "modal_predicate_share_mean": float(rep.mean()),
        "vocab_used": int(len(preds)),
        "top5_predicates": [(p, round(c / tot, 4)) for p, c in preds.most_common(5)],
        "modal_predicate_agreement": float(Counter(top1).most_common(1)[0][1] / len(top1)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", nargs="+", required=True, help="NAME=path.npz")
    p.add_argument("--pack", required=True)
    p.add_argument("--max_rel", type=int, default=20)
    p.add_argument("--rel_frac", type=float, default=0.7)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cat_names = list(json.loads((Path(a.pack) / "meta.json").read_text())["categories"])
    out = {"max_rel": a.max_rel, "rel_frac": a.rel_frac, "pack": a.pack, "models": {}}
    for spec in a.graphs:
        name, _, path = spec.partition("=")
        out["models"][name] = stats(load_npz(path), cat_names,
                                    a.max_rel, a.rel_frac, a.limit)
        out["models"][name]["source"] = path

    w = max(len(k) for k in out["models"])
    print(f"{'model':<{w}}  rel/graph  distinct  modal-share  vocab-used")
    for k, v in out["models"].items():
        print(f"{k:<{w}}  {v['rel_per_graph_mean']:9.1f}  "
              f"{v['distinct_pred_per_graph_mean']:8.1f}  "
              f"{v['modal_predicate_share_mean']:11.2f}  {v['vocab_used']:10d}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
