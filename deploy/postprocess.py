"""Host-side postprocessing for the ONNX relation head — where thresholding lives.

The exported graph emits raw scores only. Everything that a user might want to
turn a knob on happens here, in numpy, on a [K, V] array (128x29 by default):

  * a global score threshold                       (dynamic, per frame)
  * per-predicate thresholds                       (dynamic, per predicate)
  * the pair-existence weight                      (dynamic)
  * top-k, and whether a pair may emit >1 predicate

This costs microseconds against the backbone's tens of milliseconds, so making
it dynamic is free. Baking it into the graph instead would fix the values at
export time AND make the output shape data-dependent — see the module docstring
of ``deploy/export_onnx.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Triplet:
    subject_idx: int
    predicate: str
    score: float
    object_idx: int
    subject_box: Optional[np.ndarray] = None
    object_box: Optional[np.ndarray] = None
    subject_label: Optional[str] = None
    object_label: Optional[str] = None

    def __repr__(self) -> str:
        s = self.subject_label or f"obj{self.subject_idx}"
        o = self.object_label or f"obj{self.object_idx}"
        return f"({s}) --{self.predicate} [{self.score:.2f}]--> ({o})"


@dataclass
class ThresholdConfig:
    """Every field is live — mutate between frames, no re-export."""

    threshold: float = 0.40
    """Global floor on the RELATION score, pred * pair^w — the model's own
    confidence that this predicate holds. Detector confidence deliberately does
    NOT enter the threshold: conf(sub)*conf(obj) is typically ~0.1, which would
    drag every score to a tenth of its meaning and make the number you tune
    unrelated to the number displayed. It affects ranking only (see
    `box_score_weight`)."""

    per_predicate: Dict[str, float] = field(default_factory=dict)
    """Overrides keyed by predicate name. Raise 'near'/'next to' to suppress
    the chatty spatial predicates without touching the interaction ones."""

    pair_weight: float = 1.0
    """Weight on the pair-existence LOGIT: sigmoid(a*(pred + w*rel) + b).
    1.0 is the trained fusion, 0.0 drops relatedness, >1 sharpens it.

    The weight is on the logit, not an exponent on the probability: the two
    rank pairs differently, and only this one matches what the evaluator
    scores. See relsgg/scoring.py."""

    calib_a: float = 1.0
    calib_b: float = 0.0
    """Deployment calibration, from the checkpoint's calibration.json. Identity
    means UNCALIBRATED, in which case ~97% of scores land in [0.9, 1.0) and
    `threshold` is very nearly a no-op — the default 0.40 would keep almost
    everything. Monotone, so ranking and topk are unaffected either way."""

    def contract(self) -> "ScoreContract":
        from relsgg.scoring import ScoreContract
        return ScoreContract(calib_a=self.calib_a, calib_b=self.calib_b,
                             pair_weight=self.pair_weight)

    topk: int = 20
    max_per_pair: int = 1
    """How many predicates a single (sub, obj) pair may emit."""

    box_score_weight: bool = True
    """Rank triplets by score * conf(sub) * conf(obj) — the SGDet convention,
    which suppresses pairs built on shaky boxes. Ranking only: the reported
    score and the threshold stay in relation-confidence units."""

    def thresholds_vector(self, predicates: Sequence[str]) -> np.ndarray:
        """[V] per-predicate thresholds, global value where unspecified."""
        v = np.full(len(predicates), self.threshold, dtype=np.float32)
        for i, p in enumerate(predicates):
            if p in self.per_predicate:
                v[i] = self.per_predicate[p]
        return v


def decode(
    pred_logits: np.ndarray,         # [K, V]  RAW predicate logits
    pair_logits: np.ndarray,         # [K]     RAW pair-existence logits
    sub_idx: np.ndarray,             # [K]
    obj_idx: np.ndarray,             # [K]
    valid_mask: np.ndarray,          # [K] bool
    predicates: Sequence[str],
    cfg: ThresholdConfig,
    boxes_xyxy: Optional[np.ndarray] = None,
    box_scores: Optional[np.ndarray] = None,
    box_labels: Optional[Sequence[str]] = None,
) -> List[Triplet]:
    """Raw graph outputs -> ranked triplets, under a fully dynamic threshold.

    Takes logits, not probabilities: the score contract cannot be recovered
    from two separate sigmoids, and the inverse is where the precision has
    already gone.
    """
    pred_logits = np.asarray(pred_logits, np.float32)
    if pred_logits.ndim == 3:                      # drop batch
        pred_logits, pair_logits = pred_logits[0], np.asarray(pair_logits)[0]
        sub_idx, obj_idx, valid_mask = sub_idx[0], obj_idx[0], valid_mask[0]
    pair_logits = np.asarray(pair_logits, np.float32)

    # The graph is built for a FIXED box count and zero-padded, so pair slots
    # that are invalid (or reference padding) can carry indices >= the real box
    # count. Bound them before any gather, and drop them via `in_range` below —
    # never trust valid_mask alone to keep the indices addressable.
    n_boxes = None
    if box_scores is not None:
        n_boxes = len(np.asarray(box_scores).reshape(-1))
    elif boxes_xyxy is not None:
        n_boxes = len(boxes_xyxy)

    if n_boxes is not None:
        in_range = (sub_idx < n_boxes) & (obj_idx < n_boxes)
        sub_idx = np.clip(sub_idx, 0, n_boxes - 1)
        obj_idx = np.clip(obj_idx, 0, n_boxes - 1)
    else:
        in_range = np.ones_like(valid_mask, dtype=bool)

    # relation confidence — what gets thresholded AND what gets displayed.
    # THE one contract, shared with relsgg/evaluator.py so eval and deploy
    # cannot drift again: sigmoid(a * (pred + w * rel) + b).
    score = cfg.contract().scores(pred_logits, pair_logits)         # [K, V]

    # ranking score — same thing, optionally discounted by box confidence
    rank = score
    if cfg.box_score_weight and box_scores is not None:
        bs = np.asarray(box_scores, np.float32).reshape(-1)
        rank = score * (bs[sub_idx] * bs[obj_idx])[:, None]

    # --- the dynamic part: a [V] threshold vector compared elementwise -----
    thr = cfg.thresholds_vector(predicates)                          # [V]
    keep = (score >= thr[None,:]) & valid_mask[:, None]
    keep &= (in_range & (sub_idx != obj_idx))[:, None]

    if cfg.max_per_pair == 1:
        # keep only each pair's argmax predicate
        best = score.argmax(axis=1)                                  # [K]
        only_best = np.zeros_like(keep)
        only_best[np.arange(len(best)), best] = True
        keep &= only_best

    k_i, v_i = np.nonzero(keep)
    if len(k_i) == 0:
        return []
    s = score[k_i, v_i]                    # reported (relation confidence)
    order = np.argsort(-rank[k_i, v_i])[: cfg.topk]   # ranked (SGDet weighting)

    out: List[Triplet] = []
    for j in order:
        k, v = int(k_i[j]), int(v_i[j])
        si, oi = int(sub_idx[k]), int(obj_idx[k])
        out.append(Triplet(
            subject_idx=si, object_idx=oi,
            predicate=predicates[v], score=float(s[j]),
            subject_box=None if boxes_xyxy is None else boxes_xyxy[si],
            object_box=None if boxes_xyxy is None else boxes_xyxy[oi],
            subject_label=None if box_labels is None else box_labels[si],
            object_label=None if box_labels is None else box_labels[oi],
))
    return out


def decode_decomposed(
    pred_logits: np.ndarray,         # [K, V]  RAW predicate logits
    pair_logits: np.ndarray,         # [K]     RAW pair-existence logits
    sub_idx: np.ndarray,
    obj_idx: np.ndarray,
    valid_mask: np.ndarray,
    predicates: Sequence[str],
    is_spatial: np.ndarray,          # [V] bool, from the predicate bank
    cfg: ThresholdConfig,
    boxes_xyxy: Optional[np.ndarray] = None,
    box_scores: Optional[np.ndarray] = None,
    box_labels: Optional[Sequence[str]] = None,
) -> Dict[str, List[Triplet]]:
    """Two graphs from the SAME graph outputs: spatial + semantic.

    Type-stratified graph constraint (the measured protocol of
    benchmark/eval_decomposed.py): within each stream the other type's columns
    are masked out, each pair contributes its argmax predicate, and the two
    streams are thresholded and ranked INDEPENDENTLY. A pair may legitimately
    appear in both — carrying a layout relation and an interaction at once is
    the point. Pure numpy; needs only the bank's `is_spatial` vector, so the
    ONNX graph is untouched (pred_score already carries every column).
    """
    pred_logits = np.asarray(pred_logits, np.float32)
    if pred_logits.ndim == 3:
        pred_logits, pair_logits = pred_logits[0], np.asarray(pair_logits)[0]
        sub_idx, obj_idx, valid_mask = sub_idx[0], obj_idx[0], valid_mask[0]
    is_spatial = np.asarray(is_spatial, bool)
    out: Dict[str, List[Triplet]] = {}
    for tag, sel in (("spatial", is_spatial), ("semantic", ~is_spatial)):
        if not sel.any():
            out[tag] = []
            continue
        # -inf in LOGIT space, which the contract maps to score 0 for any
        # (a > 0, b). Masking a probability with -inf would not survive it.
        masked = np.where(sel[None,:], pred_logits, -np.inf)
        out[tag] = decode(masked, pair_logits, sub_idx, obj_idx, valid_mask,
                          predicates, cfg, boxes_xyxy=boxes_xyxy,
                          box_scores=box_scores, box_labels=box_labels)
    return out
