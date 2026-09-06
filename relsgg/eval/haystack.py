"""Federated evaluation against Haystack's explicit negative annotations.

Everywhere else in SGG an unlisted relation is merely UNLABELLED, so false
positives on rare predicates cannot be measured at all. Haystack annotates
(pair, predicate) cells as positive OR negative, leaving the rest unknown —
LVIS's federated design moved down one level. This evaluator scores only the
labelled cells.

NO MODEL CHANGE IS REQUIRED, and that is deliberate. The upstream protocol
wants a dense (N, 3+56) array over Haystack's chosen pairs, which would mean
forcing pairs past our relatedness sampler. We don't do that, because an
unsampled pair is not missing data — it is the model predicting "no relation",
which is exactly what the deployed system does when the sampler drops a pair.
Unsampled cells are therefore scored 0.0: a correct rejection for a negative,
a genuine miss for a positive. Forcing pairs through would measure a model we
do not ship. ``coverage`` is reported so the assumption stays visible (at
eval_budget 500, 77.8% of Haystack images have every ordered pair scored, and
PairRecall on our other benchmarks runs 0.9966-0.9972).

Metrics, all per predicate then averaged over predicates:
  fAP       Federated Average Precision — area under precision/recall over the
            labelled cells. THE HEADLINE. Haystack is 8.1:1 negative:positive,
            and ROC-AUC is optimistic under that imbalance while PR-AP is not
            (the standard detection argument, cf. LVIS).
  P-AUC     roc_auc_score, the upstream metric, for comparability with their
            paper. Only defined where a predicate has >3 positives.
  PDD       1 - mean_k recall(labels, rank < k)     "discrimination disadvantage"
  PDO       1 - mean_k precision(labels, rank < k)  "dominance overestimation"
            where rank is the target predicate's position after sorting that
            pair's scores descending.

IMPORTANT CAVEAT for the paper: Haystack's negatives come from a
model-assisted pipeline that deliberately hunts hard cases, so these numbers
measure discrimination against ADVERSARIAL negatives — not deployment
precision. Do not convert fAP into a claimed real-world precision, and do not
compute calibration (ECE/Brier) on this data; hard-mining would bias it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

from.evaluator import FREQ_BUCKETS

# Minimum labelled positives for a predicate to enter the headline fAP mean. Haystack
# is deliberately rare-focused: 31 of its 56 predicates have <50 positives and one has
# a single positive, where AP is essentially a coin flip.
MIN_POS_FOR_HEADLINE = 5


class HaystackEvaluator:
    """Score the labelled (pair, predicate) cells of a Haystack pack.

    Args:
        neg_by_index: row index in the pack -> list of [sub, obj, predicate_id]
                      negative cells (built by the runner from the sidecar's
                      image ids and the pack's file_names order).
        num_predicates: size of the deployed vocabulary (56 for Haystack).
        score_mode: "sigmoid" matches our training/deploy contract;
                    "softmax" reproduces the upstream normalisation. Per-
                    predicate AUC/AP depend on CROSS-PAIR ranking, which
                    softmax changes, so the two are not interchangeable.
    """

    def __init__(self, neg_by_index: Dict[int, List[List[int]]],
                 num_predicates: int, score_mode: str = "sigmoid",
                 use_pair: bool = True) -> None:
        self.neg = {int(k): v for k, v in neg_by_index.items()}
        self.V = int(num_predicates)
        assert score_mode in ("sigmoid", "softmax")
        self.score_mode = score_mode
        # fAP compares scores across pairs, so the pair-existence logit (the
        # only cross-pair term) dominates it; ``use_pair=False`` scores the
        # predicate logits alone as a control.
        self.use_pair = use_pair
        self.reset()

    def reset(self) -> None:
        self._scores: Dict[int, List[float]] = defaultdict(list)
        self._labels: Dict[int, List[int]] = defaultdict(list)
        self._ranks: Dict[int, List[int]] = defaultdict(list)
        self._n_cells = 0
        self._n_covered = 0

    @torch.no_grad()
    def update(self, out: dict, targets: List[dict]) -> None:
        logits = out["logits"]
        sub_idx, obj_idx = out["sub_idx"], out["obj_idx"]
        valid_mask = out["valid_mask"]
        pl = out.get("pair_logits") if self.use_pair else None
        if self.score_mode == "sigmoid":
            lg = logits.float()
            if pl is not None:
                lg = lg + pl.float().unsqueeze(-1)
            scores = torch.sigmoid(lg)
        else:
            scores = torch.softmax(logits.float(), dim=-1)
            if pl is not None:
                scores = scores * torch.sigmoid(pl.float()).unsqueeze(-1)

        for b in range(logits.shape[0]):
            idx = int(targets[b]["index"])
            mask = valid_mask[b]
            # (sub, obj) -> row of the score matrix for that pair
            pair_row = {}
            if int(mask.sum()):
                s = sub_idx[b][mask].tolist()
                o = obj_idx[b][mask].tolist()
                for i, (si, oi) in enumerate(zip(s, o)):
                    pair_row.setdefault((si, oi), i)
                sc = scores[b][mask]
            # positive cells come from the pack's GT relations
            cells = [(int(r[0]), int(r[1]), int(r[2]), 1)
                     for r in targets[b].get("relations", [])]
            cells += [(s, o, p, 0) for s, o, p in self.neg.get(idx, [])]

            for si, oi, p, lab in cells:
                self._n_cells += 1
                row = pair_row.get((si, oi))
                if row is None:
                    # Unsampled == the deployed system emits nothing for this
                    # pair. Score 0, and it ranks last among the V predicates.
                    self._scores[p].append(0.0)
                    self._ranks[p].append(self.V - 1)
                else:
                    self._n_covered += 1
                    v = sc[row]
                    self._scores[p].append(float(v[p]))
                    self._ranks[p].append(int((v > v[p]).sum()))
                self._labels[p].append(lab)

    def compute(self) -> Dict[str, float]:
        from sklearn.metrics import (average_precision_score, precision_score,
                                     recall_score, roc_auc_score)
        per: Dict[int, Dict[str, float]] = {}
        n_pos: Dict[int, int] = {}
        for p, lab in self._labels.items():
            y = np.asarray(lab)
            s = np.asarray(self._scores[p], dtype=np.float64)
            r = np.asarray(self._ranks[p])
            n_pos[p] = int(y.sum())
            # Both classes must be present for any of these to be defined.
            if n_pos[p] == 0 or n_pos[p] == len(y):
                continue
            m = {"fAP": float(average_precision_score(y, s))}
            if n_pos[p] > 3:                     # upstream's own guard
                m["PAUC"] = float(roc_auc_score(y, s))
            rec = [recall_score(y, r < k, zero_division=0) for k in range(1, self.V + 1)]
            pre = [precision_score(y, r < k, zero_division=0) for k in range(1, self.V + 1)]
            m["PDD"] = float(1.0 - np.mean(rec))
            m["PDO"] = float(1.0 - np.mean(pre))
            per[p] = m

        out: Dict[str, float] = {}
        if not per:
            return out
        for key in ("fAP", "PAUC", "PDD", "PDO"):
            vals = [v[key] for v in per.values() if key in v]
            if vals:
                out[f"m{key}"] = float(np.mean(vals))
                out[f"n_cls_{key}"] = float(len(vals))
        # HEADLINE fAP excludes barely-supported predicates. AP over a handful of
        # positives is nearly bimodal — `attached to` has ONE positive on Haystack and
        # scores fAP 1.00 if that single cell happens to outrank its negatives — so the
        # unfiltered mean is dominated by classes that carry no evidence. `mfAP` is kept
        # alongside for continuity with numbers recorded before this filter existed.
        sup = [v["fAP"] for p, v in per.items() if n_pos[p] >= MIN_POS_FOR_HEADLINE]
        if sup:
            out[f"mfAP_sup{MIN_POS_FOR_HEADLINE}"] = float(np.mean(sup))
            out[f"n_cls_fAP_sup{MIN_POS_FOR_HEADLINE}"] = float(len(sup))
        # Frequency buckets over POSITIVE support, matching the LVIS-style
        # split used by the recall metrics so the tables line up.
        for name, lo, hi in FREQ_BUCKETS:
            sel = [p for p in per if lo <= n_pos[p] < hi]
            if sel:
                out[f"fAP_{name}"] = float(np.mean([per[p]["fAP"] for p in sel]))
                out[f"n_cls_{name}"] = float(len(sel))
        out["coverage"] = (self._n_covered / self._n_cells) if self._n_cells else 0.0
        out["n_cells"] = float(self._n_cells)
        self.per_class = {int(p): v for p, v in per.items()}
        self.n_pos = {int(p): v for p, v in n_pos.items()}
        return out
