"""Two graphs from one forward pass: a spatial layout graph and a semantic
interaction graph.

The vocabulary columns are partitioned by predicate type; within each type
the other type's columns are masked, each pair keeps its best predicate, and
the two streams are ranked independently. A pair may therefore appear in
both graphs, holding a layout relation and an interaction at the same time.

Predicate types come from two sources: the training corpus's own spatial
flag when the string is known (``relsgg/corpus_type_map.json``), and the
head's routing weight ``alpha`` (spatial when >= 0.5) for any other string.
The gate under-routes unseen spatial strings, so the corpus flag has priority.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def corpus_spatial_map(pack: str, split: str = "val", resolution: int = 224) -> Dict[str, bool]:
    """predicate string -> is_spatial, by majority over a pack's relation flags."""
    from.data import RelationDataset

    ds = RelationDataset(root=pack, split=split, resolution=resolution, max_objects=40)
    rels = np.asarray(ds.rels)
    acc: Dict[int, list] = defaultdict(lambda: [0, 0])
    for p, fl in zip(rels[:, 2], rels[:, 3]):
        acc[int(p)][0] += int(bool(fl & 1))
        acc[int(p)][1] += 1
    names = ds.predicate_names
    return {names[k]: (a / b) > 0.5 for k, (a, b) in acc.items()}


def type_vector(names: Sequence[str], corpus_map: Optional[Dict[str, bool]] = None,
                alpha: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Per-predicate ``(is_spatial, source)`` with source in
    ``{"corpus", "gate", "default"}``; ``default`` (semantic) only when
    neither source knows the string."""
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


def split_ranked(pred_score: np.ndarray, sub_idx: np.ndarray, obj_idx: np.ndarray,
                 valid_mask: np.ndarray, is_spatial: np.ndarray, topk: int = 20,
) -> Dict[str, List[Tuple[int, int, int, float]]]:
    """``pred_score [K, V]`` fused scores -> two ranked edge lists
    ``{"spatial" | "semantic": [(sub, obj, pred_idx, score),...]}``, each of
    at most ``topk`` edges, one edge per pair and stream."""
    pred_score = np.asarray(pred_score, np.float32)
    if pred_score.ndim == 3:
        pred_score = pred_score[0]
        sub_idx, obj_idx, valid_mask = sub_idx[0], obj_idx[0], valid_mask[0]
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
        masked = np.where(sel[None,:], sc, -np.inf)
        arg = masked.argmax(axis=-1)
        best = masked[np.arange(len(arg)), arg]
        order = np.argsort(-best)[:topk]
        out[tag] = [(int(s_l[i]), int(o_l[i]), int(arg[i]), float(best[i]))
                    for i in order if np.isfinite(best[i])]
    return out
