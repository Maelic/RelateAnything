"""build_union_vocab.py — construct the shared predicate vocabulary + text-space
artifacts for a multi-source training mixture.

The mixture trains one relation head over several packed sources whose predicate
strings only partly overlap. RelationDataset already remaps each source's local
predicate/category ids to a shared vocab at load time (rel_cat_to_idx / OOV→-1),
so all that is needed is:

  1. a single ordered union predicate list — used as BOTH the W row order and the
     dataset rel_cat_to_idx. MEGASG's predicates are kept first, at their existing
     indices (0..V_megasg-1), so the pre-existing cooc/analysis/W stay index-
     comparable; every predicate string present only in the other sources is
     appended in a deterministic (sorted) order.
  2. the student text embeddings for that union order (the training-time W).
  3. canonical_groups.json over the union (synonym/inverse logic), reusing the
     exact build_groups used for MEGASG (MEGASG raw_links still enrich it; the
     other sources contribute canonicalization-only grouping).

The category vocabulary is unioned too (base first, stable indices, like
predicates) — but ONLY from sources passed via ``--category_roots``, which
should exclude any source whose "categories" aren't a real reusable
taxonomy. svg_vg is the deliberate exclusion: its 142K "category" strings are
free-form per-box region captions ("clock in green, tall", "arm in raised") —
essentially unique per box, not a vocabulary — so unioning it in would bloat
the space with one-off phrases and its boxes should keep mapping to -1
(RelationDataset's OOV path), which is the CORRECT behavior for a source with
no real category system, not a bug. GQA (309 cats) and SpatialSense (9 cats)
are real small taxonomies with genuine semantic overlap to MEGASG, so folding
them in gives their boxes real category ids instead of -1 — which matters
because -1 forces BatchLocalInfoNCE's cooc soft-mask to treat EVERY negative
for that pair as soft (unlabeled-not-false), diluting the contrastive signal
on however much of the mixture has unknown categories (measured: soft_neg_frac
0.38 base vs 0.81 mixed, when only megasg had real categories).

Usage (GPU node, for the student encode):
    python training/build_union_vocab.py \
        --base_root runs/packed/megasg \
        --extra_roots runs/packed/svg_vg runs/packed/gqa runs/packed/spatialsense \
        --category_roots runs/packed/gqa runs/packed/spatialsense \
        --student runs/packed/text_student/student.pt \
        --out_dir runs/packed/datamix_v22/text_space
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.text_space_diag import build_groups          # noqa: E402
from relsgg.text.student import encode_texts_student        # noqa: E402

# The template ensemble the vocabulary matrix is built with.
PHOTO_TEMPLATES = ["{p}", "one object is {p} another object",
                   "a photo of something {p} something"]


def load_meta(root: str) -> dict:
    return json.load(open(Path(root) / "train" / "meta.json"))


def load_predicates(root: str) -> list[str]:
    return load_meta(root)["predicates"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_root", default="runs/packed/megasg",
                    help="Source whose predicate/category order is preserved first.")
    ap.add_argument("--extra_roots", nargs="+", required=True,
                    help="All non-base sources, for the PREDICATE union.")
    ap.add_argument("--category_roots", nargs="+", default=None,
                    help="Subset of extra_roots (or any roots) to union into "
                         "the CATEGORY vocab. Default: same as --extra_roots. "
                         "Pass an explicit subset to exclude sources whose "
                         "'categories' aren't a real taxonomy (see module "
                         "docstring — svg_vg is the canonical example).")
    ap.add_argument("--student", default="runs/packed/text_student/student.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch", type=int, default=1024)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- union predicate order: base first (stable indices), then new sorted --
    base_meta = load_meta(args.base_root)
    base_preds = base_meta["predicates"]
    base_set = set(base_preds)
    assert len(base_set) == len(base_preds), "base predicates not unique"

    # aggregate per-predicate relation counts across all sources (by name) so
    # the union ontology's zero-count clamp reflects the whole mixture.
    agg_counts: Counter = Counter(base_meta.get("predicate_counts", {}))
    extra_new: set[str] = set()
    per_source_counts = {os.path.basename(args.base_root): len(base_preds)}
    for r in args.extra_roots:
        m = load_meta(r)
        ps = m["predicates"]
        extra_new.update(p for p in ps if p not in base_set)
        for p, c in m.get("predicate_counts", {}).items():
            agg_counts[p] += c
        per_source_counts[os.path.basename(r)] = len(ps)

    union = list(base_preds) + sorted(extra_new)
    V = len(union)
    assert len(set(union)) == V, "union predicates not unique"
    print(f"[union] base={len(base_preds)}  +new={len(extra_new)}  → V={V}")
    print(f"[union] per-source predicate counts: {per_source_counts}")

    json.dump(union, open(out_dir / "union_predicates.json", "w"))

    # ---- union category order: base first, then real-taxonomy sources only ---
    base_cats = base_meta["categories"]
    base_cat_set = set(base_cats)
    assert len(base_cat_set) == len(base_cats), "base categories not unique"

    category_roots = args.category_roots if args.category_roots is not None \
        else args.extra_roots
    cat_new: set[str] = set()
    per_source_cat_counts = {os.path.basename(args.base_root): len(base_cats)}
    for r in category_roots:
        cs = load_meta(r)["categories"]
        new = [c for c in cs if c not in base_cat_set]
        cat_new.update(new)
        per_source_cat_counts[os.path.basename(r)] = len(cs)
        print(f"[union-cat] {os.path.basename(r)}: {len(cs)} cats, "
              f"{len(cs) - len(new)} exact-match base, {len(new)} new")

    union_categories = list(base_cats) + sorted(cat_new)
    Ccat = len(union_categories)
    assert len(set(union_categories)) == Ccat, "union categories not unique"
    print(f"[union-cat] base={len(base_cats)}  +new={len(cat_new)}  → C={Ccat}  "
          f"(category_roots={[os.path.basename(r) for r in category_roots]})")
    json.dump(union_categories, open(out_dir / "union_categories.json", "w"))

    # ---- canonical groups over the union (MEGASG raw_links still enrich) ------
    base_meta = json.load(open(Path(args.base_root) / "train" / "meta.json"))
    canon, syn_pairs, inv_pairs = build_groups(union, base_meta["raw_links"])
    n_groups = len(set(canon))
    print(f"[union] {V} predicates → {n_groups} canonical groups; "
          f"{len(syn_pairs)} syn pairs, {len(inv_pairs)} inverse pairs")
    json.dump({p: c for p, c in zip(union, canon)},
              open(out_dir / "canonical_groups.json", "w"), indent=1)

    # ---- student embeddings for the union order (the training W) --------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[union] encoding {V} predicates with student on {device}...")
    E = encode_texts_student(union, args.student,
                             templates=PHOTO_TEMPLATES, device=device,
                             batch=args.batch).numpy().astype(np.float16)
    assert E.shape[0] == V, (E.shape, V)
    np.savez_compressed(
        out_dir / "pred_embeds_student_photo.npz",
        embeddings=E,
        predicates=np.array(union),           # plain unicode — np.load-safe
        templates=np.array(PHOTO_TEMPLATES),
)
    print(f"[union] wrote pred_embeds_student_photo.npz  {E.shape} "
          f"({(out_dir / 'pred_embeds_student_photo.npz').stat().st_size/1e6:.1f} MB)")

    # ---- synthetic union meta.json (for PredicateOntology.from_artifacts, and
    # as the categories source for the mixture dataset / object-aux loss) ------
    # raw_links come from the base source only (only MEGASG carries them).
    union_meta = {
        "dataset": "datamix_union",
        "predicates": union,
        "predicate_counts": {p: int(agg_counts[p]) for p in union if agg_counts[p]},
        "categories": union_categories,
        "raw_links": base_meta.get("raw_links", []),
    }
    json.dump(union_meta, open(out_dir / "union_meta.json", "w"))
    print(f"[union] wrote union_meta.json  ({V} predicates, "
          f"{len(union_meta['categories'])} categories)")
    print(f"[union] artifacts → {out_dir}")


if __name__ == "__main__":
    main()
