"""Pair sampler: which ordered (subject, object) pairs the head scores.

Two stages, both batched and traceable for export:

1. A geometry pre-scorer (two-layer MLP on the 19 pair features, no visual
   input) ranks every ordered pair and keeps ``geo_budget`` of them.
2. A learned relatedness score ``<f_s(v_i), f_o(v_j)> / sqrt(d)`` ranks the
   survivors and keeps ``final_budget``. It is trained with a focal binary
   cross-entropy in which unlabelled pairs count less than annotated ones,
   because an unannotated pair is unlabelled rather than negative. Its logit
   is the pair-existence term of the deployed score.

During training, annotated pairs and their swapped copies always survive both
stages, which is what the direction hinge in the loss relies on.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from.geometry import RelGeomEncoder


class RelatednessPairSampler(nn.Module):
    def __init__(self, geo_budget: int = 400, final_budget: int = 128,
                 feat_dim: int = 768, rel_dim: int = 256, neg_weight: float = 0.3,
                 swap_include: bool = True):
        super().__init__()
        self.geo_budget = geo_budget
        self.final_budget = final_budget
        self.neg_weight = neg_weight
        self.swap_include = swap_include
        self.rel_dim = rel_dim
        self.num_cats = 0
        # Per-category-pair interaction rates (training only, see
        # ``set_negative_rates``); never saved with the model.
        self.register_buffer("neg_rate", None, persistent=False)
        self.register_buffer("neg_trusted", None, persistent=False)

        self.geo_scorer = nn.Sequential(
            nn.Linear(RelGeomEncoder.NUM_GEO, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.f_sub = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, rel_dim), nn.GELU(), nn.Linear(rel_dim, rel_dim))
        self.f_obj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, rel_dim), nn.GELU(), nn.Linear(rel_dim, rel_dim))

    def set_negative_rates(self, rate: torch.Tensor, trusted: torch.Tensor,
                           num_cats: int) -> None:
        """Install per-(subject category, object category) interaction rates.

        ``1 - rate`` estimates the probability that an unannotated pair of
        that category pair is a genuine negative; ``neg_weight`` remains the
        floor for untrusted pairs.
        """
        self.neg_rate = rate
        self.neg_trusted = trusted
        self.num_cats = int(num_cats)

    def _pu_neg_weight(self, cs: Optional[torch.Tensor], co: Optional[torch.Tensor],
                       like: torch.Tensor) -> torch.Tensor:
        neg_w = torch.full_like(like, self.neg_weight)
        if self.neg_rate is None or cs is None:
            return neg_w
        known = (cs >= 0) & (co >= 0) & (cs < self.num_cats) & (co < self.num_cats)
        flat = (cs.clamp(min=0) * self.num_cats + co.clamp(min=0))
        flat = flat.clamp(max=self.neg_rate.numel() - 1)
        rate = self.neg_rate[flat].float()
        trusted = self.neg_trusted[flat] & known
        return torch.where(trusted, (1.0 - rate).clamp(min=self.neg_weight, max=1.0), neg_w)

    @staticmethod
    def _gt_grid(targets, B: int, N: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """``is_gt [B, N*N]`` and a predicate label per pair (-1 where none;
        the last predicate wins for multi-predicate pairs)."""
        is_gt = torch.zeros(B, N * N, dtype=torch.bool, device=device)
        labels = torch.full((B, N * N), -1, dtype=torch.long, device=device)
        if targets is None:
            return is_gt, labels
        for b, t in enumerate(targets):
            rels = t.get("relations")
            if rels is None or len(rels) == 0:
                continue
            keep = (rels[:, 0] < N) & (rels[:, 1] < N) & (rels[:, 0] != rels[:, 1])
            rels = rels[keep]
            if len(rels) == 0:
                continue
            flat = rels[:, 0] * N + rels[:, 1]
            is_gt[b].scatter_(0, flat, True)
            labels[b].scatter_(0, flat, rels[:, 2])
        return is_gt, labels

    def forward(self, boxes: torch.Tensor, obj_feats: torch.Tensor,
                box_counts: Optional[torch.Tensor] = None,
                targets: Optional[List[dict]] = None,
                entity_labels: Optional[torch.Tensor] = None):
        """Returns ``(sub_idx, obj_idx, valid_mask, pred_labels, geo_loss,
        rel_loss, rel_logits)``; the index tensors are ``[B, K]``.

        ``entity_labels`` (``[B, N]``, -1 unknown) only weights the training
        loss; inference never passes it."""
        B, N, C = obj_feats.shape
        device = boxes.device
        K = self.final_budget
        training = targets is not None
        if box_counts is None:
            box_counts = torch.full((B,), N, dtype=torch.long, device=device)

        ar = torch.arange(N, device=device)
        valid_box = ar.unsqueeze(0) < box_counts.unsqueeze(1)
        not_self = ar.unsqueeze(0) != ar.unsqueeze(1)
        pair_valid = (valid_box.unsqueeze(2) & valid_box.unsqueeze(1) & not_self).reshape(B, N * N)

        geo_feats = RelGeomEncoder.features(
            boxes.unsqueeze(2).expand(B, N, N, 4),
            boxes.unsqueeze(1).expand(B, N, N, 4)).reshape(B, N * N, RelGeomEncoder.NUM_GEO)
        geo_scores = self.geo_scorer(geo_feats).squeeze(-1)

        is_gt, flat_labels = self._gt_grid(targets, B, N, device)
        force = is_gt
        if training and self.swap_include:
            swapped = is_gt.reshape(B, N, N).transpose(1, 2).reshape(B, N * N)
            force = is_gt | (swapped & pair_valid)

        NEG = torch.finfo(geo_scores.dtype).min
        sel_scores = geo_scores.masked_fill(~pair_valid, NEG)
        if training:
            sel_scores = sel_scores.masked_fill(force & pair_valid, float("inf"))
        K1 = min(self.geo_budget, N * N)
        _, top1 = sel_scores.topk(K1, dim=1)
        alive1 = torch.gather(pair_valid, 1, top1)

        zs, zo = self.f_sub(obj_feats), self.f_obj(obj_feats)
        sub_i1, obj_i1 = top1 // N, top1 % N
        z_s = torch.gather(zs, 1, sub_i1.unsqueeze(-1).expand(-1, -1, self.rel_dim))
        z_o = torch.gather(zo, 1, obj_i1.unsqueeze(-1).expand(-1, -1, self.rel_dim))
        rel_scores1 = (z_s * z_o).sum(-1) / (self.rel_dim ** 0.5)

        geo_loss = boxes.new_zeros(())
        rel_loss = boxes.new_zeros(())
        if training:
            n_valid = pair_valid.float().sum().clamp(min=1.0)
            geo_bce = F.binary_cross_entropy_with_logits(geo_scores, is_gt.float(), reduction="none")
            geo_loss = (geo_bce * pair_valid.float()).sum() / n_valid

            gt1 = torch.gather(is_gt, 1, top1).float()
            cs = co = None
            if self.neg_rate is not None and entity_labels is not None:
                cs = torch.gather(entity_labels, 1, sub_i1)
                co = torch.gather(entity_labels, 1, obj_i1)
            neg_w = self._pu_neg_weight(cs, co, rel_scores1)
            w = torch.where(gt1.bool(), torch.ones_like(rel_scores1), neg_w)
            p = torch.sigmoid(rel_scores1)
            p_t = gt1 * p + (1 - gt1) * (1 - p)
            rel_bce = F.binary_cross_entropy_with_logits(
                rel_scores1, gt1, reduction="none") * (1 - p_t).pow(2.0)
            m1 = alive1.float()
            rel_loss = (rel_bce * w * m1).sum() / m1.sum().clamp(min=1.0)

        sel2 = rel_scores1.masked_fill(~alive1, NEG)
        if training:
            sel2 = sel2.masked_fill(torch.gather(force & pair_valid, 1, top1), float("inf"))
        K2 = min(K, K1)
        _, top2 = sel2.topk(K2, dim=1)

        flat_sel = torch.gather(top1, 1, top2)
        sub_idx, obj_idx = flat_sel // N, flat_sel % N
        valid_mask = torch.gather(alive1, 1, top2)
        pred_labels = torch.gather(flat_labels, 1, flat_sel)
        pred_labels = torch.where(valid_mask, pred_labels, torch.full_like(pred_labels, -1))
        rel_logits = torch.gather(rel_scores1, 1, top2)

        if K2 < K:
            pad = K - K2
            sub_idx = F.pad(sub_idx, (0, pad))
            obj_idx = F.pad(obj_idx, (0, pad))
            valid_mask = F.pad(valid_mask, (0, pad))
            pred_labels = F.pad(pred_labels, (0, pad), value=-1)
            rel_logits = F.pad(rel_logits, (0, pad), value=float("-inf"))
        return sub_idx, obj_idx, valid_mask, pred_labels, geo_loss, rel_loss, rel_logits
