"""Two-graph decomposition: split one relation forward pass into a SPATIAL
graph and a SEMANTIC graph.

Why this is a post-processing layer and not a second model: the measured
decomposed evaluation (benchmark/eval_decomposed.py, runs/train/v43_full_5ep/
decomposed*.json) runs ONE forward pass and partitions the vocabulary columns
by predicate type — mask the other type to -inf, take one argmax edge per
pair, rank each stream independently. Everything it needs is already in the
checkpoint outputs, so inference gets it for free.

Type sources, and why there are two:
  * CORPUS flags — majority vote of the training corpus's per-relation
    spatial bit per predicate string. This is the split the measured 6/6
    result used. Caveat that must travel with it: the flag is provenance-
    contaminated, not a clean ontology (`on` reads 0.985 spatial while its
    synonym `resting on` reads 0.001), so near-identical predicates can land
    in different streams. Covers only strings the corpus has seen.
  * GATE alpha — the checkpoint's own spatialness router, a function of the
    text embedding, so it covers ANY string. Measured caveat: it under-routes
    NOVEL spatial predicates (alpha <= 0.09 on unseen spatial strings), so
    gate-only would quietly drain the spatial stream for user vocabulary.

The shipped rule is therefore hybrid: corpus flag when the string is known,
gate alpha >= 0.5 otherwise, with the source recorded per row
(`type_source`). The corpus-vs-alpha A/B on the release checkpoint decides
the default before shipping (interim default = corpus, the measured winner).

Synonyms are never collapsed (project rule): they simply compete within
their stream at argmax time.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def corpus_spatial_map(pack: str, split: str = "val",
                       resolution: int = 224) -> Dict[str, bool]:
    """predicate string -> is_spatial, by majority over the corpus's own flags.

    Moved verbatim from benchmark/eval_decomposed.py (which now imports it from
    here) so deploy-time bank building and eval share one definition.
    """
    from data import RelationDataset

    ds = RelationDataset(root=pack, split=split, resolution=resolution,
                         max_objects=40)
    rels = np.asarray(ds.rels)
    acc: Dict[int, list] = defaultdict(lambda: [0, 0])
    for p, fl in zip(rels[:, 2], rels[:, 3]):
        acc[int(p)][0] += int(bool(fl & 1))
        acc[int(p)][1] += 1
    names = ds.predicate_names
    out, gap = {}, []
    for k, (a, b) in acc.items():
        f = a / b
        out[names[k]] = f > 0.5
        if 0.05 < f < 0.79:
            gap.append((names[k], f, b))
    # MOSTLY bimodal but provenance-contaminated (see module docstring).
    # Report rather than assert — the partition is still usable; the caveat
    # must travel with the numbers.
    if gap:
        print("  [warn] type map not cleanly bimodal; majority vote applied to: "
              + ", ".join(f"{n}({f:.2f}, n={c})" for n, f, c in sorted(gap)))
    return out


def type_vector(names: Sequence[str],
                corpus_map: Optional[Dict[str, bool]] = None,
                alpha: Optional[np.ndarray] = None,
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-predicate (is_spatial, type_source) for a vocabulary.

    Hybrid rule: corpus flag when the string is in the map, gate alpha >= 0.5
    otherwise. Returns (bool [V], source [V] of {"corpus","gate","default"}).
    "default" (semantic) is only used when NEITHER source knows the string —
    counted and worth printing by callers, because a mostly-default vector
    means the split is guessing.
    """
    V = len(names)
    is_sp = np.zeros(V, dtype=bool)
    src = np.full(V, "default", dtype=object)
    for i, n in enumerate(names):
        if corpus_map is not None and n in corpus_map:
            is_sp[i] = corpus_map[n]
            src[i] = "corpus"
        elif alpha is not None:
            is_sp[i] = bool(alpha[i] >= 0.5)
            src[i] = "gate"
    return is_sp, src


def split_ranked(pred_score: np.ndarray,     # [K, V] fused per-pair scores
                 sub_idx: np.ndarray,        # [K]
                 obj_idx: np.ndarray,        # [K]
                 valid_mask: np.ndarray,     # [K] bool
                 is_spatial: np.ndarray,     # [V] bool
                 topk: int = 20,
                 ) -> Dict[str, List[Tuple[int, int, int, float]]]:
    """One forward pass -> two independently-ranked edge streams.

    Exactly the DecomposedEvaluator semantics (the measured 6/6 win): within
    each type, mask the other type's columns to -inf, take the argmax
    predicate per pair (one edge per pair — the graph constraint), rank pairs
    by that score, cut at topk PER STREAM. Returns
    {"spatial"|"semantic": [(sub, obj, pred_idx, score), ...]} sorted by
    score descending. 2*topk edges total by construction; a stream whose
    type has no columns is empty.
    """
    pred_score = np.asarray(pred_score, np.float32)
    if pred_score.ndim == 3:                          # drop batch dim
        pred_score = pred_score[0]
        sub_idx, obj_idx = sub_idx[0], obj_idx[0]
        valid_mask = valid_mask[0]
    valid = np.asarray(valid_mask, bool)
    is_spatial = np.asarray(is_spatial, bool)

    out: Dict[str, List[Tuple[int, int, int, float]]] = {}
    sc = pred_score[valid]
    s_l = np.asarray(sub_idx)[valid]
    o_l = np.asarray(obj_idx)[valid]
    for tag, sel in (("spatial", is_spatial), ("semantic", ~is_spatial)):
        if not sel.any() or sc.size == 0:
            out[tag] = []
            continue
        masked = np.where(sel[None, :], sc, -np.inf)
        arg = masked.argmax(axis=-1)                  # one edge per pair
        best = masked[np.arange(len(arg)), arg]
        order = np.argsort(-best)[:topk]
        out[tag] = [(int(s_l[i]), int(o_l[i]), int(arg[i]), float(best[i]))
                    for i in order if np.isfinite(best[i])]
    return out
