"""Pair samplers.

CascadePairSampler (legacy, P1 baseline):
  Stage 1 — Geometry pre-scorer (trainable 2-layer MLP).
             Operates on the 15 raw geometry features. No visual features.
             Selects the geo_budget highest-scoring pairs via topk.
             Trained with binary cross-entropy: GT pairs = 1, rest = 0.
  Stage 2 — Attention proxy: cosine(v_sub, v_obj) on Stage-1 survivors.
             Selects the final_budget highest-scoring pairs.
  Per-image Python loop; GT pairs force-included by tail replacement.

RelatednessPairSampler (P2, plan D4):
  Same two-stage shape, fully batched/vectorized (no .item(), traceable for
  ONNX), with Stage 2 replaced by a LEARNED asymmetric relatedness score
  s(i,j) = <f_s(v_i), f_o(v_j)>/sqrt(d) (Scene-Graph ViT style). The cosine
  proxy selects *similar* objects, not *interacting* ones — the
  interaction-vs-non-interaction confusion is the main OV-SGG noise source
  (Li et al., NeurIPS'25). Relatedness is trained with focal BCE using
  down-weighted negatives (absent relations are unlabeled, not false — the
  LLM GT is positive-unlabeled), and answers "what supervises no-relation":
  relatedness scores pair existence, the vocab head scores predicate
  identity, final triplet score = sigmoid(rel) * sigmoid(pred).
  GT inclusion at train time via score override (+inf) — vectorized, and
  optionally also includes SWAPPED GT pairs (swap_include) to power the
  ARO-style directional supervision in the loss.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import GeoEncoder


class CascadePairSampler(nn.Module):
    """Two-stage cascade pair sampler.

    Args:
        geo_budget:   Number of pairs kept after Stage 1 (geometry filter).
        final_budget: Number of pairs kept after Stage 2 (attention proxy).
        iou_threshold: Minimum IoU to consider an external box as matching a
                       GT box when building GT pair targets. Not used here
                       since training assumes GT boxes are passed directly.
    """

    def __init__(
        self,
        geo_budget: int = 400,
        final_budget: int = 128,
        geo_squash: bool = False,
    ):
        super().__init__()
        self.geo_budget = geo_budget
        self.final_budget = final_budget
        self.geo_squash = geo_squash  # must match RelGeomEncoder's setting

        # Small scorer: geometry features only, ~4K parameters
        self.geo_scorer = nn.Sequential(
            nn.Linear(GeoEncoder.NUM_GEO, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        boxes: torch.Tensor,
        obj_feats: torch.Tensor,
        box_counts: Optional[torch.Tensor] = None,
        targets: Optional[List[dict]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample pairs for each image in the batch.

        Args:
            boxes:      [B, max_N, 4] normalized cxcywh, zero-padded.
            obj_feats:  [B, max_N, C] per-object visual features.
            box_counts: [B] actual number of valid boxes per image.
                        Defaults to max_N for all images if None.
            targets:    List[dict] with keys ``relations`` (LongTensor [R, 3]:
                        sub_idx, obj_idx, pred_label) for each image.
                        Pass None at inference.
        Returns:
            sub_idx:     [B, final_budget] subject indices into ``boxes``.
            obj_idx:     [B, final_budget] object  indices into ``boxes``.
            valid_mask:  [B, final_budget] bool — True for real pairs.
            pred_labels: [B, final_budget] int64 — GT predicate or -1.
            geo_loss:    scalar — binary cross-entropy on the geo scorer.
                         Zero tensor if targets is None.
        """
        B, max_N, C = obj_feats.shape
        device = boxes.device
        K = self.final_budget

        if box_counts is None:
            box_counts = torch.full((B,), max_N, dtype=torch.long, device=device)

        all_sub, all_obj, all_valid, all_labels = [], [], [], []
        geo_losses: List[torch.Tensor] = []

        for b in range(B):
            N = int(box_counts[b].item())
            rels = targets[b].get("relations") if targets is not None else None

            sub_b, obj_b, valid_b, labels_b, geo_loss_b = self._sample_one(
                boxes=boxes[b, :N],          # [N, 4]
                feats=obj_feats[b, :N],      # [N, C]
                N=N,
                K=K,
                relations=rels,
                device=device,
            )
            all_sub.append(sub_b)
            all_obj.append(obj_b)
            all_valid.append(valid_b)
            all_labels.append(labels_b)
            if geo_loss_b is not None:
                geo_losses.append(geo_loss_b)

        geo_loss = (
            torch.stack(geo_losses).mean()
            if geo_losses
            else boxes.new_zeros(1).squeeze()
        )

        return (
            torch.stack(all_sub),    # [B, K]
            torch.stack(all_obj),    # [B, K]
            torch.stack(all_valid),  # [B, K]
            torch.stack(all_labels), # [B, K]
            geo_loss,
        )

    def _sample_one(
        self,
        boxes: torch.Tensor,
        feats: torch.Tensor,
        N: int,
        K: int,
        relations,
        device: torch.device,
    ):
        """Sample pairs for a single image.

        Returns (sub_idx[K], obj_idx[K], valid_mask[K], pred_labels[K], geo_loss).
        """
        # Allocate output tensors (padding = 0-indexed, covered by valid_mask=False)
        sub_out = boxes.new_zeros(K, dtype=torch.long)
        obj_out = boxes.new_zeros(K, dtype=torch.long)
        valid = boxes.new_zeros(K, dtype=torch.bool)
        labels = boxes.new_full((K,), -1, dtype=torch.long)

        if N < 2:
            return sub_out, obj_out, valid, labels, None

        # Build all N*(N-1) ordered pairs: exclude self-pairs
        ii, jj = torch.meshgrid(
            torch.arange(N, device=device),
            torch.arange(N, device=device),
            indexing="ij",
        )
        pair_mask = ii != jj
        ii_flat = ii[pair_mask]  # [P]
        jj_flat = jj[pair_mask]  # [P]
        P = ii_flat.shape[0]

        # Flat-index lookup: pair_to_flat[i, j] → position in ii_flat/jj_flat
        pair_to_flat = torch.full((N, N), -1, dtype=torch.long, device=device)
        pair_to_flat[ii_flat, jj_flat] = torch.arange(P, device=device)

        # --- Geometry features for all pairs [P, 15] ---
        sub_boxes_all = boxes[ii_flat]  # [P, 4]
        obj_boxes_all = boxes[jj_flat]  # [P, 4]
        geo_feats_all = GeoEncoder.features(sub_boxes_all, obj_boxes_all,
                                            squash=self.geo_squash)  # [P, 19]

        # --- Stage 1: geometry pre-scorer ---
        geo_scores = self.geo_scorer(geo_feats_all).squeeze(-1)  # [P], has grad

        # Build GT targets for geo_scorer (positive = GT relation pair)
        gt_flat_indices: List[int] = []
        flat_to_pred: dict = {}
        if relations is not None and len(relations) > 0:
            for rel in relations:
                s_gt, o_gt, pred = int(rel[0]), int(rel[1]), int(rel[2])
                if s_gt >= N or o_gt >= N or s_gt == o_gt:
                    continue
                flat_idx = int(pair_to_flat[s_gt, o_gt].item())
                if flat_idx >= 0:
                    gt_flat_indices.append(flat_idx)
                    flat_to_pred[flat_idx] = pred

        with torch.no_grad():
            geo_targets = geo_scores.new_zeros(P)
            if gt_flat_indices:
                geo_targets[torch.tensor(gt_flat_indices, device=device)] = 1.0

        geo_loss = F.binary_cross_entropy_with_logits(
            geo_scores, geo_targets
        ) if relations is not None else None

        # Discrete pair selection — no grad needed below this point
        with torch.no_grad():
            K1 = min(self.geo_budget, P)
            _, top1 = geo_scores.detach().topk(K1)

            # Stage 2: cosine attention proxy on Stage-1 survivors
            v_sub = feats[ii_flat[top1]]  # [K1, C]
            v_obj = feats[jj_flat[top1]]  # [K1, C]
            attn = F.cosine_similarity(v_sub, v_obj, dim=-1)  # [K1]

            K2 = min(K, K1)
            _, top2 = attn.topk(K2)
            selected_flat = top1[top2].tolist()  # flat pair indices [K2]

            # Guarantee inclusion of GT pairs not already selected
            selected_set = set(selected_flat)
            extra: List[int] = []
            for flat_idx in gt_flat_indices:
                if flat_idx not in selected_set:
                    extra.append(flat_idx)

            # Replace tail of selected with GT pairs (GT takes priority)
            for flat_idx in extra:
                if len(selected_flat) < K:
                    selected_flat.append(flat_idx)
                else:
                    selected_flat[-1] = flat_idx  # replace lowest-scoring slot

            for k, flat_idx in enumerate(selected_flat[:K]):
                sub_out[k] = ii_flat[flat_idx]
                obj_out[k] = jj_flat[flat_idx]
                valid[k] = True
                if flat_idx in flat_to_pred:
                    labels[k] = flat_to_pred[flat_idx]

        return sub_out, obj_out, valid, labels, geo_loss


class RelatednessPairSampler(nn.Module):
    """Vectorized two-stage sampler with a learned relatedness head (plan D4).

    Args:
        geo_budget:   Pairs kept after the geometry stage.
        final_budget: Pairs kept after the relatedness stage (K).
        feat_dim:     Visual feature dim of ``obj_feats`` (backbone dim).
        rel_dim:      Projection dim of the relatedness head.
        neg_weight:   Weight of non-GT pairs in the relatedness BCE
                      (positive-unlabeled: absent != false).
        swap_include: Also force-include the swapped (obj, sub) slot of every
                      GT pair at train time — enables directional (ARO-style)
                      negatives in the classification loss.
    """

    def __init__(
        self,
        geo_budget: int = 400,
        final_budget: int = 128,
        feat_dim: int = 768,
        rel_dim: int = 256,
        neg_weight: float = 0.3,
        swap_include: bool = True,
        neg_rate: Optional[torch.Tensor] = None,
        neg_trusted: Optional[torch.Tensor] = None,
        num_cats: int = 0,
        geo_squash: bool = False,
        geo_pu: bool = False,
    ):
        super().__init__()
        self.geo_budget = geo_budget
        self.final_budget = final_budget
        self.neg_weight = neg_weight
        self.swap_include = swap_include
        self.num_cats = int(num_cats)
        self.geo_squash = geo_squash  # must match RelGeomEncoder's setting
        # geo_pu: apply the SAME PU-priced negative weighting to the geometry
        # pre-scorer's BCE that the relatedness head already gets. The geo
        # scorer is the FIRST hard gate (stage-1 topk) and was trained with
        # full-weight negatives on positive-unlabeled data — i.e. to predict
        # annotation propensity, and its mistakes are unrecoverable downstream.
        # Same data, same "absent != false" argument (Elkan & Noto 2008),
        # now applied to both heads. False = prior runs bit-identical.
        self.geo_pu = geo_pu
        # Per-(sub_cat, obj_cat) interaction rate, from build_pair_opportunity.py.
        # `1 - rate` approximates P(an unannotated candidate of this category pair is a
        # GENUINE negative), so it prices the PU hedge per pair instead of applying one
        # worst-case constant to everything. Non-persistent: 17 MB that is reproducible
        # from the pack, and checkpoints should not carry it.
        self.register_buffer("neg_rate", neg_rate, persistent=False)
        self.register_buffer("neg_trusted", neg_trusted, persistent=False)

        self.geo_scorer = nn.Sequential(
            nn.Linear(GeoEncoder.NUM_GEO, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.f_sub = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, rel_dim),
            nn.GELU(), nn.Linear(rel_dim, rel_dim),
        )
        self.f_obj = nn.Sequential(
            nn.LayerNorm(feat_dim), nn.Linear(feat_dim, rel_dim),
            nn.GELU(), nn.Linear(rel_dim, rel_dim),
        )
        self.rel_dim = rel_dim

    def _pu_neg_weight(self, cs: torch.Tensor, co: torch.Tensor,
                       like: torch.Tensor) -> torch.Tensor:
        """Per-pair negative weight from the opportunity table.

        ``1 - rate`` approximates P(an unannotated candidate of this category
        pair is a genuine negative); ``neg_weight`` stays the FLOOR for
        untrusted / unknown pairs. Returns a tensor shaped like ``like``.
        """
        neg_w = torch.full_like(like, self.neg_weight)
        if self.neg_rate is None or cs is None:
            return neg_w
        known = (cs >= 0) & (co >= 0) & (cs < self.num_cats) & (co < self.num_cats)
        flat = (cs.clamp(min=0) * self.num_cats + co.clamp(min=0))
        flat = flat.clamp(max=self.neg_rate.numel() - 1)
        rate = self.neg_rate[flat].float()
        trusted = self.neg_trusted[flat] & known
        return torch.where(
            trusted,
            (1.0 - rate).clamp(min=self.neg_weight, max=1.0),
            neg_w)

    @staticmethod
    def _gt_grid(targets, B: int, N: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        """(is_gt [B, N*N] bool, labels [B, N*N] int64, -1 where none).

        Multi-predicate pairs keep the LAST predicate here (labels feed the
        contrastive terms); the classification loss builds full multi-hot
        targets from the raw relations independently.
        """
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

    def forward(
        self,
        boxes: torch.Tensor,       # [B, N, 4] normalized cxcywh, zero-padded
        obj_feats: torch.Tensor,   # [B, N, C]
        box_counts: Optional[torch.Tensor] = None,
        targets: Optional[List[dict]] = None,
        entity_labels: Optional[torch.Tensor] = None,   # [B, N] int64, -1 = unknown
    ):
        """Returns (sub_idx, obj_idx, valid_mask, pred_labels, geo_loss,
        rel_loss, rel_logits) — all [B, K] except the two scalar losses.
        ``rel_logits`` are the raw relatedness scores of the selected pairs
        (sigmoid → pair existence probability).

        ``entity_labels`` is used ONLY to weight the training loss; inference never
        receives it and the forward path does not branch on it."""
        B, N, C = obj_feats.shape
        device = boxes.device
        K = self.final_budget
        training = targets is not None

        if box_counts is None:
            box_counts = torch.full((B,), N, dtype=torch.long, device=device)

        ar = torch.arange(N, device=device)
        valid_box = ar.unsqueeze(0) < box_counts.unsqueeze(1)          # [B, N]
        # Off-diagonal mask via arange comparison rather than ~torch.eye(bool):
        # bit-identical, and exports to ONNX Equal instead of a bool EyeLike,
        # which onnxruntime has no CPU kernel for.
        not_self = ar.unsqueeze(0) != ar.unsqueeze(1)                  # [N,N]
        pair_valid = (valid_box.unsqueeze(2) & valid_box.unsqueeze(1)  # [B,N,N]
                      & not_self)
        pair_valid = pair_valid.reshape(B, N * N)

        # ---- geometry scores for ALL ordered pairs -----------------------
        geo_feats = GeoEncoder.features(
            boxes.unsqueeze(2).expand(B, N, N, 4),
            boxes.unsqueeze(1).expand(B, N, N, 4),
            squash=self.geo_squash,
        ).reshape(B, N * N, GeoEncoder.NUM_GEO)
        geo_scores = self.geo_scorer(geo_feats).squeeze(-1)            # [B, N*N]

        is_gt, flat_labels = self._gt_grid(targets, B, N, device)
        force = is_gt
        if training and self.swap_include:
            swapped = is_gt.reshape(B, N, N).transpose(1, 2).reshape(B, N * N)
            force = is_gt | (swapped & pair_valid)

        NEG = torch.finfo(geo_scores.dtype).min
        sel_scores = geo_scores.masked_fill(~pair_valid, NEG)
        if training:
            # GT (and swapped) pairs always survive stage 1: score override
            # keeps everything batched — no per-image index surgery.
            sel_scores = sel_scores.masked_fill(force & pair_valid, float("inf"))

        K1 = min(self.geo_budget, N * N)
        _, top1 = sel_scores.topk(K1, dim=1)                           # [B, K1]
        alive1 = torch.gather(pair_valid, 1, top1)                     # [B, K1]

        # ---- learned relatedness on stage-1 survivors ---------------------
        zs = self.f_sub(obj_feats)                                     # [B, N, d]
        zo = self.f_obj(obj_feats)
        sub_i1 = top1 // N
        obj_i1 = top1 % N
        z_s = torch.gather(zs, 1, sub_i1.unsqueeze(-1).expand(-1, -1, self.rel_dim))
        z_o = torch.gather(zo, 1, obj_i1.unsqueeze(-1).expand(-1, -1, self.rel_dim))
        rel_scores1 = (z_s * z_o).sum(-1) / (self.rel_dim ** 0.5)      # [B, K1]

        # ---- losses --------------------------------------------------------
        # geo_loss and rel_loss are returned SEPARATELY: they used to be summed
        # into one aux scalar weighted by lambda_geo=0.1, which trained the
        # relatedness head — half the deployed score, and the term the beta
        # fusion feeds on — at 10% weight as a side effect of a hyperparameter
        # named after geometry.
        geo_loss = boxes.new_zeros(())
        rel_loss = boxes.new_zeros(())
        if training:
            n_valid = pair_valid.float().sum().clamp(min=1.0)
            geo_bce = F.binary_cross_entropy_with_logits(
                geo_scores, is_gt.float(), reduction="none")
            w_geo = pair_valid.float()
            if self.geo_pu and entity_labels is not None:
                cs_all = entity_labels.unsqueeze(2).expand(B, N, N).reshape(B, N * N)
                co_all = entity_labels.unsqueeze(1).expand(B, N, N).reshape(B, N * N)
                negw = self._pu_neg_weight(cs_all, co_all, geo_scores)
                w_geo = w_geo * torch.where(is_gt, torch.ones_like(negw), negw)
            geo_loss = (geo_bce * w_geo).sum() / n_valid

            gt1 = torch.gather(is_gt, 1, top1).float()                 # [B, K1]
            # Negative weight, per pair rather than constant.
            #
            # A flat `neg_weight` is a hedge against the PU problem (unannotated !=
            # unrelated) priced for the worst case and then charged to every pair.
            # Measured on megasg: the median well-supported category pair relates on
            # only 13% of the instance pairs it presents, so the median genuine negative
            # deserves ~0.87, not 0.3 — the head was receiving about a third of the
            # gradient it should from the negatives it can be most sure about, which is
            # why it never learns a threshold and the model never stops emitting.
            #
            # `neg_weight` stays the FLOOR: pairs whose categories genuinely interact
            # often, or that lack support, keep exactly today's hedge. The change only
            # ever raises the weight of negatives the statistics vouch for.
            cs = co = None
            if self.neg_rate is not None and entity_labels is not None:
                cs = torch.gather(entity_labels, 1, sub_i1)            # [B, K1]
                co = torch.gather(entity_labels, 1, obj_i1)
            neg_w = self._pu_neg_weight(cs, co, rel_scores1)
            w = torch.where(gt1.bool(), torch.ones_like(rel_scores1), neg_w)
            p = torch.sigmoid(rel_scores1)
            p_t = gt1 * p + (1 - gt1) * (1 - p)
            rel_bce = F.binary_cross_entropy_with_logits(
                rel_scores1, gt1, reduction="none") * (1 - p_t).pow(2.0)
            m1 = alive1.float()
            rel_loss = (rel_bce * w * m1).sum() / m1.sum().clamp(min=1.0)

        # ---- stage 2 selection --------------------------------------------
        sel2 = rel_scores1.masked_fill(~alive1, NEG)
        if training:
            force1 = torch.gather(force & pair_valid, 1, top1)
            sel2 = sel2.masked_fill(force1, float("inf"))
        K2 = min(K, K1)
        _, top2 = sel2.topk(K2, dim=1)                                 # [B, K2]

        flat_sel = torch.gather(top1, 1, top2)                         # [B, K2]
        sub_idx = flat_sel // N
        obj_idx = flat_sel % N
        valid_mask = torch.gather(alive1, 1, top2)
        pred_labels = torch.gather(flat_labels, 1, flat_sel)
        pred_labels = torch.where(valid_mask, pred_labels,
                                  torch.full_like(pred_labels, -1))
        rel_logits = torch.gather(rel_scores1, 1, top2)

        if K2 < K:  # pad to the fixed budget
            pad = K - K2
            sub_idx = F.pad(sub_idx, (0, pad))
            obj_idx = F.pad(obj_idx, (0, pad))
            valid_mask = F.pad(valid_mask, (0, pad))
            pred_labels = F.pad(pred_labels, (0, pad), value=-1)
            rel_logits = F.pad(rel_logits, (0, pad), value=float("-inf"))

        return (sub_idx, obj_idx, valid_mask, pred_labels,
                geo_loss, rel_loss, rel_logits)
