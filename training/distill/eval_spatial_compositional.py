"""Does the text student place compositional spatial phrasings correctly?

Scores exactly the property a spatial-only deployment depends on. The head
scores cos(q_spa, w_p) against frozen text embeddings, so a novel spatial
predicate transfers about as well as its cosine to a trained one — with the
right polarity. For each string from spatial_grammar.py:

  axis_top1   nearest base form over ALL axes has the string's own axis
  polarity    within its own axis, the nearer side is the correct polarity
  margin      cos(best same-polarity base) - cos(best opposite-polarity base)

`margin` is the metric that matters: it is the headroom the relation head has
to rank the right predicate first, and it goes NEGATIVE exactly on the failures
that motivated the grammar ("on the underside of" sitting closer to inside/on
than to under).

Reported per grammar family and split by whether the family was held out of
distillation (build_corpus.py --spatial_holdout), so the held-out rows measure
compositional generalization rather than memorization.

    python training/distill/eval_spatial_compositional.py \
        --student runs/packed/text_student/student.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

PROJ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJ))

from relsgg.text.student import encode_texts_student              # noqa: E402
from training.distill.spatial_grammar import AXES, NEW_BASES, OPPOSITE, generate   # noqa: E402

TRAIN_TEMPLATES = ["{p}", "one object is {p} another object",
                   "a photo of something {p} something"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="runs/packed/text_student/student.pt")
    ap.add_argument("--corpus", default="",
                    help="corpus.json — marks which families were held out. "
                         "Optional; without it every row reports as 'train'.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    entries, bases = generate()

    split_of: dict = {}
    if args.corpus:
        for p in json.load(open(args.corpus))["provenance"]:
            if "grammar" in p:
                split_of[p["s"]] = p["split"]

    # (axis, polarity) -> column block of the base matrix
    base_key = [(ax, pol) for ax, d in AXES.items() for pol in d
                for _ in d[pol]]
    base_str = [b for ax, d in AXES.items() for pol in d for b in d[pol]]

    texts = [e["s"] for e in entries] + base_str
    E = encode_texts_student(texts, str(PROJ / args.student),
                             templates=TRAIN_TEMPLATES, device=device)
    E = F.normalize(E, dim=-1)
    G, B = E[:len(entries)], E[len(entries):]
    cos = (G @ B.T).cpu()                                   # [n_gen, n_base]

    rows = []
    for i, e in enumerate(entries):
        ax, pol = e["axis"], e["polarity"]
        same = [j for j, k in enumerate(base_key) if k == (ax, pol)]
        oppo = [j for j, k in enumerate(base_key) if k == (ax, OPPOSITE[pol])]
        if not same or not oppo:
            continue
        best_same = float(cos[i, same].max())
        best_oppo = float(cos[i, oppo].max())
        # axis_top1 ranks only against bases the relation head can actually
        # score (NEW_BASES are new vocabulary, not valid targets).
        pool = [j for j, b in enumerate(base_str) if b not in NEW_BASES]
        top = pool[int(cos[i, pool].argmax())]
        nearest_axis = base_key[top][0]
        rows.append({
            "s": e["s"], "family": e["family"], "axis": ax, "polarity": pol,
            "split": split_of.get(e["s"], "train"),
            "axis_top1": nearest_axis == ax,
            "polarity_ok": best_same > best_oppo,
            "margin": best_same - best_oppo,
            "nearest": base_str[top],
            "cos_nearest": float(cos[i, top]),
        })

    by: dict = defaultdict(list)
    for r in rows:
        by[(r["family"], r["split"])].append(r)

    print(f"student: {args.student}   ({len(rows)} generated strings)\n")
    print(f'{"family":<13}{"split":<9}{"n":>4}{"axis_top1":>11}'
          f'{"polarity":>10}{"margin":>9}')
    for key in sorted(by):
        g = by[key]
        n = len(g)
        print(f"{key[0]:<13}{key[1]:<9}{n:>4}"
              f"{sum(r['axis_top1'] for r in g) / n:>11.2f}"
              f"{sum(r['polarity_ok'] for r in g) / n:>10.2f}"
              f"{sum(r['margin'] for r in g) / n:>9.3f}")
    n = len(rows)
    print(f"\n{'ALL':<13}{'':<9}{n:>4}"
          f"{sum(r['axis_top1'] for r in rows) / n:>11.2f}"
          f"{sum(r['polarity_ok'] for r in rows) / n:>10.2f}"
          f"{sum(r['margin'] for r in rows) / n:>9.3f}")

    worst = sorted(rows, key=lambda r: r["margin"])[:12]
    print("\nworst margins (negative = wrong side of its own axis):")
    for r in worst:
        print(f"  {r['margin']:+.3f}  {r['s']:<28} -> {r['nearest']} "
              f"({r['cos_nearest']:.2f})  [{r['family']}/{r['split']}]")

    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
        print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()
