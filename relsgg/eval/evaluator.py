"""Evaluators for relation prediction.

    SGClsEvaluator      R@K and mR@K with exact predicate matching
                        (oracle boxes, the SGCls protocol of Xu et al. 2017).
    SoftSGClsEvaluator  SoftR@K / SoftmR@K: a prediction matches when its
                        predicate is the annotated one or a synonym of it
                        (text cosine at or above tau_eval, never an inverse).
                        On a synonym-preserving vocabulary exact matching
                        counts correct synonyms as misses.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch


def _harmonic(r: float, mr: float) -> float:
    """F1@K = 2*R@K*mR@K / (R@K + mR@K), 0 when both are 0.

    Both arguments MUST come from the same evaluator pass — pairing a
    graph-constrained mR with an unconstrained R gives a number that is the
    F1 of nothing (unconstrained R@K runs 12-19 points high).
    """
    return 0.0 if (r + mr) <= 0 else float(2.0 * r * mr / (r + mr))


class SGClsEvaluator:
    """R@K and mR@K for scene graph classification with oracle boxes.

    Protocol (Xu et al. 2017):
        1. For each image, generate all scored (sub, obj, pred) triplets
           by expanding K sampled pairs × V predicates with their softmax
           probability as score.
        2. Sort all image-level triplets by score descending.
        3. R@K  = fraction of unique GT relations found in top-K.
        4. mR@K = R@K averaged over predicate classes (mean recall).

    Args:
        topk:           K values to evaluate, e.g. ``[20, 50, 100]``.
        num_predicates: Number of predicate classes (used for book-keeping
                        only; does not affect the computation).

    Usage::

        ev = SGClsEvaluator(topk=[20, 50, 100])
        for images, boxes, box_counts, targets in val_loader:
            out = model(images, boxes, box_counts, targets=None)
            ev.update(out, targets)
        metrics = ev.compute()  # {"R@20":..., "mR@20":...,...}
    """

    def __init__(
        self,
        topk: List[int] = (20, 50, 100),
        num_predicates: int = 50,
        score_mode: str = "softmax",
        subsets: Optional[Dict[str, List[int]]] = None,
        graph_constraint: bool = False,
        contract: Optional["ScoreContract"] = None,
):
        """score_mode: "softmax" (classic SGCls protocol) or "sigmoid"
        (synonym-trained models — softmax over a synonym-sharing vocabulary
        deflates every synonym's probability).

        subsets: optional {name: [class_id,...]}. When given, ``compute``
        additionally reports ``{name}_R@K`` — the standard image-macro recall
        restricted to GT triplets whose predicate falls in that subset (used
        for the OvSGTR base/novel-relation split; opt-in, no effect on the
        default metrics).

        graph_constraint: if True, each object pair contributes only its
        single highest-scoring predicate to the ranked list ("with graph
        constraint", the default convention for the R@K numbers reported in
        most of the SGG literature). If False (default here), the top-K is
        taken over the full pairs x predicates score matrix, so one pair may
        occupy several slots — the "unconstrained" convention, which yields
        systematically higher numbers. Cross-paper comparisons must match
        this setting.

        contract: the deployment score contract (relsgg/scoring.py). Default
        is the uncalibrated identity, which is the right choice for every
        metric here — R@K/mR@K are RANKING metrics and the contract is
        monotone, so a calibration cannot move them. Pass the shipped one when
        you want eval and the product to be byte-for-byte the same function
        (benchmark/eval_deploy_metrics.py does)."""
        from relsgg.scoring import ScoreContract
        self.topk = list(topk)
        self.num_predicates = num_predicates
        assert score_mode in ("softmax", "sigmoid")
        self.score_mode = score_mode
        self.graph_constraint = graph_constraint
        self.contract = contract or ScoreContract()
        self._subsets = {n: set(ids) for n, ids in (subsets or {}).items()}
        self.reset()

    def reset(self) -> None:
        self._recall_hits: Dict[int, List[float]] = {k: [] for k in self.topk}
        self._per_class_recall: Dict[int, Dict[int, List[float]]] = {
            k: defaultdict(list) for k in self.topk
        }
        # Raw cumulative counts for per-class TP/GT reporting
        self._per_class_tp: Dict[int, Dict[int, int]] = {k: defaultdict(int) for k in self.topk}
        self._per_class_gt: Dict[int, int] = defaultdict(int)
        # Per-image recall restricted to each named predicate subset
        self._subset_recall: Dict[str, Dict[int, List[float]]] = {
            n: {k: [] for k in self.topk} for n in self._subsets
        }

    @torch.no_grad()
    def update(self, out: dict, targets: List[dict]) -> None:
        """Accumulate one batch.

        Args:
            out:     Model output dict with keys ``logits`` [B,K,V],
                     ``sub_idx`` [B,K], ``obj_idx`` [B,K],
                     ``valid_mask`` [B,K].
            targets: List of per-image target dicts, each with key
                     ``relations`` LongTensor [R, 3].
        """
        logits     = out["logits"]      # [B, K, V]
        sub_idx    = out["sub_idx"]     # [B, K]
        obj_idx    = out["obj_idx"]     # [B, K]
        valid_mask = out["valid_mask"]  # [B, K]

        B = logits.shape[0]
        if self.score_mode == "sigmoid":
            # THE shared contract (relsgg/scoring.py) — the same object the
            # ONNX host and deploy/pipeline.py use, so eval and the product
            # cannot drift apart again. Padded slots carry rel_logit=-inf,
            # hence score 0.
            scores = self.contract.scores(
                logits.float(),
                None if out.get("pair_logits") is None
                else out["pair_logits"].float())        # [B, K, V]
        else:
            scores = torch.softmax(logits.float(), dim=-1)
            if out.get("pair_logits") is not None:
                scores = scores * torch.sigmoid(out["pair_logits"].float()).unsqueeze(-1)
        max_k = max(self.topk)

        for b in range(B):
            mask  = valid_mask[b]
            if int(mask.sum()) == 0:
                continue

            # GT relations
            rels = targets[b].get("relations", None)
            if rels is None or len(rels) == 0:
                continue

            # Global top-max_k over the k_valid×V score matrix ON DEVICE —
            # a full argsort of k×V (4M entries at V=10K) on CPU dominated
            # eval wall-clock before.
            probs = scores[b][mask]                     # [k_valid, V]
            # Standard SGDet triplet score = pred_score · conf(sub) · conf(obj).
            # Opt-in via targets[b]["box_scores"] (detector box confidences);
            # for oracle-box eval it is absent → pure predicate ranking as
            # before. Down-weights low-confidence detector boxes so their pairs
            # do not crowd out well-localized pairs under the fixed top-K budget.
            bscore = targets[b].get("box_scores")
            if bscore is not None:
                bscore = bscore.to(probs.device, torch.float32)
                n_bs = bscore.shape[0]
                si = sub_idx[b][mask].long().clamp(max=n_bs - 1)
                oi = obj_idx[b][mask].long().clamp(max=n_bs - 1)
                probs = probs * (bscore[si] * bscore[oi]).unsqueeze(-1)
            V = probs.shape[1]
            if self.graph_constraint:
                # One triplet per pair: its arg-max predicate, pairs then
                # ranked by that score.
                best_score, best_pred = probs.max(dim=-1)   # [k_valid]
                n_top = min(max_k, best_score.numel())
                _, pair_ids = best_score.topk(n_top)
                all_p = best_pred[pair_ids].cpu()
            else:
                flat = probs.reshape(-1)
                n_top = min(max_k, flat.numel())
                _, top_flat = flat.topk(n_top)
                pair_ids = top_flat // V
                all_p = (top_flat % V).cpu()
            all_s = sub_idx[b][mask][pair_ids].cpu()
            all_o = obj_idx[b][mask][pair_ids].cpu()

            rels = rels.cpu()  # [R, 3]
            gt_set = {(int(r[0]), int(r[1]), int(r[2])) for r in rels}
            R_total = len(gt_set)

            # Accumulate raw GT counts once per image (k-independent)
            for r in rels:
                self._per_class_gt[int(r[2])] += 1

            for k in self.topk:
                hits = 0
                per_class_hits: Dict[int, int] = defaultdict(int)
                per_class_total: Dict[int, int] = defaultdict(int)
                for r in rels:
                    per_class_total[int(r[2])] += 1

                for i in range(min(k, n_top)):
                    trip = (int(all_s[i]), int(all_o[i]), int(all_p[i]))
                    if trip in gt_set:
                        hits += 1
                        per_class_hits[trip[2]] += 1

                # An image with no GT in the evaluated vocabulary carries no
                # recall information — scoring it 0.0 would deflate R@K by the
                # empty fraction. Skip it, matching the `sub_gt > 0` guard the
                # subset recalls below already use. Only reachable when the
                # vocabulary is restricted (RelationDataset rel_cat_to_idx:
                # eval_zeroshot --spatial_only empties 34.6% of VG150 test,
                # eval_ovsgtr_novel restricts to novel predicates); packs
                # enforce min_rels, so full-vocabulary eval is unaffected.
                if R_total == 0:
                    continue
                self._recall_hits[k].append(hits / R_total)
                for cls_id, cls_total in per_class_total.items():
                    self._per_class_recall[k][cls_id].append(
                        per_class_hits.get(cls_id, 0) / cls_total
)
                # Accumulate raw TP counts
                for cls_id, n_hits in per_class_hits.items():
                    self._per_class_tp[k][cls_id] += n_hits
                # Image-macro recall restricted to each named subset
                for name, ids in self._subsets.items():
                    sub_gt = sum(t for c, t in per_class_total.items() if c in ids)
                    if sub_gt > 0:
                        sub_hit = sum(h for c, h in per_class_hits.items() if c in ids)
                        self._subset_recall[name][k].append(sub_hit / sub_gt)

    def _frequency_metrics(self, k: int) -> Dict[str, float]:
        return frequency_metrics(self._per_class_tp[k], self._per_class_gt, k)

    def compute(self, oov_indices: Optional[List[int]] = None) -> Dict[str, float]:
        """Return aggregated metrics dict.

        Args:
            oov_indices: If provided, also compute ``OOV_mR@K`` restricted to
                         these predicate class indices (zero-shot transfer metric).
        """
        metrics: Dict[str, float] = {}
        for k in self.topk:
            arr = self._recall_hits[k]
            metrics[f"R@{k}"] = float(np.mean(arr)) if arr else 0.0

            class_recalls = [
                float(np.mean(vals))
                for vals in self._per_class_recall[k].values()
            ]
            metrics[f"mR@{k}"] = float(np.mean(class_recalls)) if class_recalls else 0.0

            if oov_indices is not None:
                oov_recalls = [
                    float(np.mean(self._per_class_recall[k][cls_id]))
                    for cls_id in oov_indices
                    if cls_id in self._per_class_recall[k]
                ]
                metrics[f"OOV_mR@{k}"] = float(np.mean(oov_recalls)) if oov_recalls else 0.0

            # F1@K — harmonic mean of R@K and mR@K (SGG-Benchmark). R@K and
            # mR@K trade off against each other by construction, so either one
            # alone can be won by skewing head or tail; the harmonic mean
            # weights the SMALLER of the two and cannot be gamed that way.
            metrics[f"F1@{k}"] = _harmonic(metrics[f"R@{k}"], metrics[f"mR@{k}"])

            metrics.update(self._frequency_metrics(k))

            for name in self._subsets:
                arr = self._subset_recall[name][k]
                metrics[f"{name}_R@{k}"] = float(np.mean(arr)) if arr else 0.0

        return metrics

    def compute_per_class(
        self, k: int = 50, pred_names: Optional[List[str]] = None
) -> List[dict]:
        """Return per-class cumulative TP / GT counts and recall at @K.

        Returns a list sorted by GT count descending, one entry per predicate
        class that appeared in at least one GT annotation::

            [{"class_id": 3, "name": "on", "tp": 52, "gt": 120, "recall": 0.433},...]

        Args:
            k:          The @K threshold — must be in ``self.topk``.
            pred_names: Optional predicate name list; index is the class ID.
        """
        if k not in self.topk:
            raise ValueError(f"k={k} is not in topk={self.topk}.")
        tp_dict = self._per_class_tp.get(k, {})
        rows = []
        for cls_id, gt_count in sorted(self._per_class_gt.items(), key=lambda x: -x[1]):
            tp = tp_dict.get(cls_id, 0)
            name = (
                pred_names[cls_id]
                if pred_names is not None and cls_id < len(pred_names)
                else str(cls_id)
)
            rows.append({
                "class_id": cls_id,
                "name":     name,
                "tp":       int(tp),
                "gt":       int(gt_count),
                "recall":   tp / gt_count if gt_count > 0 else 0.0,
            })
        return rows


class SoftSGClsEvaluator:
    """Synonym-aware SGCls recall (SoftR@K / SoftmR@K).

    A predicted triplet (s, o, p̂) matches GT (s, o, g) when
    ``match_matrix[p̂, g]`` is True. Build the matrix from the predicate
    ontology: same canonical group OR text-embedding cosine ≥ tau_eval
    (never for spatial inverses). Mean recall is computed over CANONICAL
    GROUPS, not raw predicate ids — with 10K synonym-preserving classes,
    per-string mR is statistically meaningless.

    Also reports the strict/spatial/semantic splits when ``rel_flags`` are
    present in targets (bit0 = spatial — see pack meta flags_legend).

    Args:
        match_matrix: [V, V] bool torch tensor (pred, gt).
        group_of:     [V] int64 canonical-group id per predicate.
        topk:         K values.
        score_mode:   "sigmoid" (default — synonym-trained models) or
                      "softmax".
    """

    def __init__(
        self,
        match_matrix: torch.Tensor,
        group_of: torch.Tensor,
        topk: List[int] = (20, 50, 100),
        score_mode: str = "sigmoid",
        graph_constraint: bool = False,
        contract: Optional["ScoreContract"] = None,
):
        from relsgg.scoring import ScoreContract
        self.topk = list(topk)
        self.M = match_matrix.bool()
        self.group_of = group_of.long()
        assert score_mode in ("softmax", "sigmoid")
        self.score_mode = score_mode
        # Same contract semantics as SGClsEvaluator: default identity is
        # correct for ranking metrics (monotone), pass the shipped one only
        # for byte-identical eval-vs-deploy comparisons.
        self.contract = contract or ScoreContract()
        # Rank one triplet per object pair (its arg-max predicate) instead of
        # a flat top-K over pairs x predicates. Optional in-domain, but
        # MANDATORY in open-vocabulary mode: with V=19,103 the unconstrained
        # top-50 is filled by a single pair's near-synonyms ("on", "on top
        # of", "resting on",...), so R@K would measure synonym density
        # rather than coverage of the image's relations.
        self.graph_constraint = graph_constraint
        self.reset()

    def reset(self) -> None:
        self._img_recall: Dict[int, List[float]] = {k: [] for k in self.topk}
        self._img_recall_split: Dict[str, Dict[int, List[float]]] = {
            s: {k: [] for k in self.topk} for s in ("spatial", "semantic")
        }
        self._group_tp: Dict[int, Dict[int, int]] = {k: defaultdict(int) for k in self.topk}
        self._group_gt: Dict[int, int] = defaultdict(int)
        # Fraction of GT pairs (regardless of predicate) present in the
        # sampled top-max_k — the eval-time sampler-recall gate (M0 ≥ 0.98
        # is measured against the raw sampled pair set, this is the
        # score-ranked variant visible without model internals).
        self._pair_recall: List[float] = []
        # Within-pair rank of the GT predicate (soft: best-matching predicate
        # counts). R@K is a step function that sits at 0 until alignment is
        # strong; reciprocal rank moves from the first optimizer step.
        self._gt_ranks: List[float] = []

    @torch.no_grad()
    def update(self, out: dict, targets: List[dict]) -> None:
        logits     = out["logits"]
        sub_idx    = out["sub_idx"]
        obj_idx    = out["obj_idx"]
        valid_mask = out["valid_mask"]
        device = logits.device
        M = self.M.to(device)

        if self.score_mode == "sigmoid":
            # THE shared contract — see SGClsEvaluator.update
            scores = self.contract.scores(
                logits.float(),
                None if out.get("pair_logits") is None
                else out["pair_logits"].float())
        else:
            scores = torch.softmax(logits.float(), dim=-1)
            if out.get("pair_logits") is not None:
                scores = scores * torch.sigmoid(out["pair_logits"].float()).unsqueeze(-1)
        max_k = max(self.topk)
        B = logits.shape[0]

        for b in range(B):
            mask = valid_mask[b]
            rels = targets[b].get("relations", None)
            if int(mask.sum()) == 0 or rels is None or len(rels) == 0:
                continue
            rels = rels.to(device)                       # [R, 3]
            flags = targets[b].get("rel_flags")
            spatial = (flags.to(device) & 1).bool() if flags is not None else None

            probs = scores[b][mask]                      # [k_valid, V]
            V = probs.shape[1]
            if self.graph_constraint:
                best_score, best_pred = probs.max(dim=-1)      # [k_valid]
                n_top = min(max_k, best_score.numel())
                _, pair_ids = best_score.topk(n_top)
                p_top = best_pred[pair_ids]                    # [n]
            else:
                flat = probs.reshape(-1)
                n_top = min(max_k, flat.numel())
                _, top_flat = flat.topk(n_top)
                pair_ids = top_flat // V
                p_top = top_flat % V                           # [n]
            s_top = sub_idx[b][mask][pair_ids]           # [n]
            o_top = obj_idx[b][mask][pair_ids]

            # Sampler coverage: GT pairs present among ALL sampled pairs
            # (not just the top-n triplets) — the true pair-recall gate.
            s_all = sub_idx[b][mask]
            o_all = obj_idx[b][mask]
            sampled = (s_all.unsqueeze(1) == rels[:, 0].unsqueeze(0)) & \
                      (o_all.unsqueeze(1) == rels[:, 1].unsqueeze(0))
            self._pair_recall.append(float(sampled.any(0).float().mean()))

            # Within-pair GT-predicate rank (1 = best). Soft: the score of the
            # best M-matching predicate is the reference.
            found_r, slot_of = sampled.max(0)            # [R] slot index per GT
            for r_i in torch.nonzero(found_r).flatten().tolist():
                row = probs[slot_of[r_i]]                # [V]
                ok = M[:, rels[r_i, 2]]                  # predicates matching GT
                best = row[ok].max()
                self._gt_ranks.append(float((row > best).sum()) + 1.0)

            # [n, R]: prediction i matches GT r
            pair_ok = (s_top.unsqueeze(1) == rels[:, 0].unsqueeze(0)) & \
                      (o_top.unsqueeze(1) == rels[:, 1].unsqueeze(0))
            pred_ok = M[p_top.unsqueeze(1), rels[:, 2].unsqueeze(0)]
            match = pair_ok & pred_ok                    # [n, R]

            # first matching rank per GT (n if never matched)
            ranks = torch.where(
                match.any(0),
                match.float().argmax(0),
                torch.full((len(rels),), n_top, device=device),
)                                            # [R]

            gt_groups = self.group_of.to(device)[rels[:, 2]]
            for g in gt_groups.tolist():
                self._group_gt[g] += 1

            for k in self.topk:
                found = ranks < min(k, n_top)            # [R]
                self._img_recall[k].append(float(found.float().mean()))
                if spatial is not None:
                    for name, sel in (("spatial", spatial), ("semantic", ~spatial)):
                        if sel.any():
                            self._img_recall_split[name][k].append(
                                float(found[sel].float().mean())
)
                for g, f in zip(gt_groups.tolist(), found.tolist()):
                    if f:
                        self._group_tp[k][g] += 1

    def compute(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for k in self.topk:
            arr = self._img_recall[k]
            metrics[f"SoftR@{k}"] = float(np.mean(arr)) if arr else 0.0
            group_recalls = [
                self._group_tp[k].get(g, 0) / total
                for g, total in self._group_gt.items() if total > 0
            ]
            metrics[f"SoftmR@{k}"] = float(np.mean(group_recalls)) if group_recalls else 0.0
            metrics[f"SoftF1@{k}"] = _harmonic(metrics[f"SoftR@{k}"],
                                               metrics[f"SoftmR@{k}"])
            metrics.update(frequency_metrics(self._group_tp[k], self._group_gt,
                                             k, prefix="Soft"))
            for name in ("spatial", "semantic"):
                arr = self._img_recall_split[name][k]
                if arr:
                    metrics[f"SoftR@{k}_{name}"] = float(np.mean(arr))
        if self._pair_recall:
            metrics["PairRecall"] = float(np.mean(self._pair_recall))
        if self._gt_ranks:
            ranks = np.asarray(self._gt_ranks)
            metrics["GT_MRR"] = float((1.0 / ranks).mean())
            metrics["GT_MedRank"] = float(np.median(ranks))
        return metrics


class FanoutEvaluator:
    """Run several evaluators over the same predictions; merge their metrics."""

    def __init__(self, evaluators: List) -> None:
        self.evaluators = list(evaluators)

    def update(self, out: dict, targets: List[dict]) -> None:
        for e in self.evaluators:
            e.update(out, targets)

    def compute(self, **kw) -> Dict[str, float]:
        merged: Dict[str, float] = {}
        for e in self.evaluators:
            try:
                merged.update(e.compute(**kw))
            except TypeError:
                merged.update(e.compute())
        return merged


# Relation-count cut points for the frequency buckets, mirroring LVIS's
# AP_r / AP_c / AP_f split (there: 1-10 / 11-100 / >100 IMAGES per category).
# Absolute counts, not percentiles, so a bucket means the same thing across
# benchmarks of different sizes; the class count per bucket is reported
# alongside so the split stays interpretable.
FREQ_BUCKETS = (("rare", 0, 50), ("common", 50, 500), ("freq", 500, 1 << 62))


def frequency_metrics(tp: Dict[int, int], gt: Dict[int, int],
                      k: int, prefix: str = "") -> Dict[str, float]:
    """Tail-aware aggregates that R@K and mR@K miss on their own: a model
    can gain micro recall while losing the tail, and only a split by class
    frequency shows it.

    Both aggregates use pooled tp/gt per class rather than the mean of
    per-image recalls, so each class contributes once at its own support:

      R@K_{rare,common,freq}  LVIS-style buckets by GT relation count.
      wR@K                    IDF-weighted recall, w_c ∝ log(N/n_c)
                              normalised to sum 1 — CIDEr's weighting. One
                              scalar that grades the tail continuously
                              instead of by hard buckets. Tail weight
                              raises variance, so wR is the noisiest of the
                              three and is reported as a supporting number.
    """
    out: Dict[str, float] = {}
    gt = {c: n for c, n in gt.items() if n > 0}
    if not gt:
        return out
    rec = {c: tp.get(c, 0) / n for c, n in gt.items()}
    for name, lo, hi in FREQ_BUCKETS:
        sel = [c for c, n in gt.items() if lo <= n < hi]
        if sel:
            out[f"{prefix}R@{k}_{name}"] = float(np.mean([rec[c] for c in sel]))
            out[f"{prefix}n_cls_{name}"] = float(len(sel))
    N = sum(gt.values())
    w = {c: math.log(N / n) for c, n in gt.items()}
    tot = sum(w.values())
    if tot > 0:
        out[f"{prefix}wR@{k}"] = float(sum(w[c] / tot * rec[c] for c in gt))
    return out


def build_cross_match_matrix(
    train_preds: List[str],
    bench_preds: List[str],
    E_train: np.ndarray,
    E_bench: np.ndarray,
    tau_eval: float = 0.9,
    inverse_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[V_train, V_bench] bool — does a prediction from the FULL training
    vocabulary count as the benchmark's ground-truth predicate?

    This is the open-vocabulary scoring contract. The closed-vocabulary
    protocol reparameterizes the head down to the benchmark's own 37-56
    predicates, so the model can only answer in the benchmark's words. Here
    the deployed vocabulary stays at its full size and the model answers in
    its own words; a prediction is correct when it MEANS the GT predicate.
    Without this, a model trained for synonym diversity is punished for
    saying "on top of" where the benchmark wrote "on".

    Match = exact string OR cosine(E_train[p], E_bench[g]) >= tau_eval,
    minus inverses. Inverse removal matters more here than anywhere else:
    the whole point of the distilled text space was to separate
    "above"/"below", and a soft matcher that let them match would silently
    hand back the directionality the model was trained to learn. Inverses
    are transferred through benchmark predicates whose exact string also
    exists in the training vocabulary (the usual case for these packs).
    """
    tr_idx = {p: i for i, p in enumerate(train_preds)}
    A = torch.nn.functional.normalize(
        torch.from_numpy(np.asarray(E_train, dtype=np.float32)), dim=-1)
    B = torch.nn.functional.normalize(
        torch.from_numpy(np.asarray(E_bench, dtype=np.float32)), dim=-1)
    M = torch.zeros(len(train_preds), len(bench_preds), dtype=torch.bool)
    step = 2048
    for i in range(0, A.shape[0], step):
        M[i:i + step] = (A[i:i + step] @ B.T) >= tau_eval
    for g, name in enumerate(bench_preds):          # exact string always matches
        if name in tr_idx:
            M[tr_idx[name], g] = True
    if inverse_mask is not None:
        inv = inverse_mask.bool()
        for g, name in enumerate(bench_preds):
            j = tr_idx.get(name)
            if j is not None:
                M[:, g] &= ~inv[:, j]
                M[tr_idx[name], g] = True           # never mask the identity
    return M


def build_match_matrix(
    group_of: torch.Tensor,
    embeddings: Optional[np.ndarray] = None,
    tau_eval: float = 0.9,
    inverse_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[V, V] bool: same canonical group OR cosine ≥ tau_eval, never inverses."""
    same_group = group_of.unsqueeze(0) == group_of.unsqueeze(1)
    M = same_group
    if embeddings is not None:
        E = torch.from_numpy(np.asarray(embeddings, dtype=np.float32))
        E = torch.nn.functional.normalize(E, dim=-1)
        V = E.shape[0]
        cos_ok = torch.zeros(V, V, dtype=torch.bool)
        step = 1024
        for i in range(0, V, step):
            cos_ok[i:i + step] = (E[i:i + step] @ E.T) >= tau_eval
        M = M | cos_ok
    if inverse_mask is not None:
        M = M & ~inverse_mask.bool()
    return M
