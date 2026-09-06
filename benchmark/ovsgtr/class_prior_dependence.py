"""class_prior_dependence.py — how much of a model's relation output is a
function of the OBJECT-CATEGORY PAIR alone?

Motivation. OvSGTR consumes object categories twice: its detector is prompted with
the 150 VG150 nouns, so its object queries are grounded to category text, and the
relation head reads those queries. Our model never receives an object label. The
question is whether that channel is doing the work.

An intervention (scrambling the prompt) also destroys detection, confounding the
answer. This measures the same thing observationally, on predictions we already
have, with NO GPU and no confound: both models are run on the SAME boxes with the
SAME predicted labels, so the category pairs are identical and only the relation
head differs.

Three statistics, all on the top-K ranked pairs per image (the ones that decide R@K):

  self_lookup    Build a table (sub_cat, obj_cat) -> the model's own most common
                 predicted predicate, on one half of the images; measure how often
                 it reproduces the model's prediction on the OTHER half. This is
                 the direct question: how much of the model's output is recoverable
                 from category identity, with the pixels thrown away. Held out, so
                 it cannot be inflated by memorising the eval set.
  freq_agree     Agreement with the Neural-MOTIFS FREQ table built from VG150 TRAIN
                 annotations -- the classic pixel-free baseline.
  H_cond/H       Conditional entropy of the model's predicate given the category
                 pair, normalised by its unconditional entropy. 0 = fully determined
                 by the pair, 1 = the pair says nothing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_preds(path, topk):
    """Top-`topk` ranked (sub_cat, obj_cat, predicate) per image from an
    interchange npz. Order is the stored one, which both producers set to
    graph_infer's pred*conf*conf sort, so the two models are compared on the
    pairs each actually ranks highest."""
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    info = json.loads(str(d["meta"][0]))
    lb = int(info.get("label_base", 1))
    pair_ptr, box_ptr = d["pair_ptr"], d["box_ptr"]
    out = []
    for i in range(len(d["image_index"])):
        a, b = int(pair_ptr[i]), int(pair_ptr[i + 1])
        if b <= a:
            out.append(np.zeros((0, 3), np.int32))
            continue
        e = min(b, a + topk)
        pairs = d["pairs"][a:e].astype(np.int64)
        pp = d["rel_scores"][a:e].astype(np.float32)[:, 1:].argmax(1)
        bA, bB = int(box_ptr[i]), int(box_ptr[i + 1])
        cls = d["labels"][bA:bB].astype(np.int64) - lb
        out.append(np.stack([cls[pairs[:, 0]], cls[pairs[:, 1]], pp], 1).astype(np.int32))
    return out, [str(x) for x in d["predicates"]]


def freq_table_from_pack(pack: Path):
    """Neural-MOTIFS FREQ: argmax_p count(p | sub_cat, obj_cat) over TRAIN."""
    img_meta = np.load(pack / "img_meta.npy")
    rels = np.load(pack / "rels.npy")
    cats = np.load(pack / "box_cats.npy")
    cnt = defaultdict(Counter)
    for row in img_meta:
        _i, _w, _h, b0, nb, r0, nr = (int(x) for x in row)
        r = rels[r0:r0 + nr]
        r = r[(r[:, 0] < nb) & (r[:, 1] < nb)]
        c = cats[b0:b0 + nb]
        for s, o, p in r[:,:3]:
            cnt[(int(c[s]), int(c[o]))][int(p)] += 1
    return {k: v.most_common(1)[0][0] for k, v in cnt.items()}


def entropy(counter):
    n = sum(counter.values())
    if n == 0:
        return 0.0
    p = np.array([v / n for v in counter.values()], dtype=np.float64)
    return float(-(p * np.log2(p)).sum())


def analyse(preds, freq, n_pred):
    even = [preds[i] for i in range(0, len(preds), 2)]
    odd = [preds[i] for i in range(1, len(preds), 2)]

    def build(rows):
        t = defaultdict(Counter)
        for a in rows:
            for s, o, p in a:
                t[(int(s), int(o))][int(p)] += 1
        return t

    res = {}
    # self-lookup, both directions, then averaged (each half is held out once)
    accs, cov = [], []
    for build_on, test_on in ((even, odd), (odd, even)):
        tab = {k: v.most_common(1)[0][0] for k, v in build(build_on).items()}
        hit = tot = seen = 0
        for a in test_on:
            for s, o, p in a:
                tot += 1
                key = (int(s), int(o))
                if key in tab:
                    seen += 1
                    hit += int(tab[key] == int(p))
        accs.append(hit / max(tot, 1))
        cov.append(seen / max(tot, 1))
    res["self_lookup"] = float(np.mean(accs))
    res["self_lookup_pair_coverage"] = float(np.mean(cov))

    # agreement with the TRAIN-annotation FREQ table
    hit = tot = 0
    for a in preds:
        for s, o, p in a:
            tot += 1
            hit += int(freq.get((int(s), int(o)), -1) == int(p))
    res["freq_agree"] = hit / max(tot, 1)
    res["n_edges"] = tot

    # normalised conditional entropy of the prediction given the category pair
    full = build(preds)
    marg = Counter()
    for c in full.values():
        marg.update(c)
    H = entropy(marg)
    N = sum(marg.values())
    Hc = sum(sum(c.values()) / N * entropy(c) for c in full.values())
    res["H_pred"] = H
    res["H_pred_given_pair"] = Hc
    res["H_ratio"] = Hc / H if H > 0 else 0.0
    res["distinct_predicates_used"] = len(marg)
    res["n_pred_vocab"] = n_pred
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preds", nargs="+", required=True,
                   help="NAME=path.npz pairs (interchange format)")
    p.add_argument("--train_pack", default="runs/packed/vg150/train")
    p.add_argument("--topk", type=int, default=50)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    print(f"building FREQ table from {args.train_pack}...", flush=True)
    freq = freq_table_from_pack(Path(args.train_pack))
    print(f"  {len(freq)} category pairs seen in train")

    rows = {}
    for spec in args.preds:
        name, path = spec.split("=", 1)
        preds, pred_names = load_preds(path, args.topk)
        rows[name] = analyse(preds, freq, len(pred_names))
        rows[name]["source"] = path
        print(f"  scored {name}", flush=True)

    hdr = (f"{'model':<26}{'self-lookup':>12}{'FREQ agree':>12}"
           f"{'H(p|pair)/H':>13}{'#preds used':>12}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, r in rows.items():
        print(f"{name:<26}{100*r['self_lookup']:>11.1f}%{100*r['freq_agree']:>11.1f}%"
              f"{r['H_ratio']:>13.3f}{r['distinct_predicates_used']:>12}")
    print(f"\ntop-{args.topk} ranked pairs per image; "
          f"{list(rows.values())[0]['n_edges']:,} edges per model")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"topk": args.topk, "train_pack": args.train_pack, "models": rows},
            indent=2, sort_keys=True))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
