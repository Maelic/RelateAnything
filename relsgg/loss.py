"""Relation loss functions.

Two components, weighted and summed:

  1. Classification loss (cross-entropy on matched GT slots).
     Provides a strong, direct gradient signal on the correct predicate.

  2. Focal loss (sigmoid, per-class binary).
     Operates on all valid slots (GT and negatives).
     Suppresses false positives for non-GT predicates.
     Standard focal loss (Lin et al., 2017) with learnable-friendly defaults.

Only slots where ``pred_labels >= 0`` (GT pairs) contribute to the
cross-entropy term. All valid slots contribute to the focal term (GT slots
as positives, the rest as negatives).

Predicate-Conditioned Spatial Grounding (PCSG) loss — ``RelSpatialGroundingLoss``
----------------------------------------------------------------------------------
Two additional components that supervise the backbone patch grid directly via
geometry-derived interaction zones (the union box of each subject–object pair):

  IZSA — Interaction Zone Spatial Alignment.
    InfoNCE between the coverage-weighted average of backbone patch features
    inside the union box (projected to text embedding space) and the GT
    predicate text embedding.  Forces the ViT LoRA adapters to encode
    predicate-discriminative features at the pixel level, before any
    transformer enrichment.

  CZSC — Cross-Zone Supervised Contrastive.
    SupCon (Khosla et al., NeurIPS 2020) over GT zone fingerprints within the
    batch: pulls interaction-zone features from pairs sharing the same
    predicate together, pushes different predicates apart.  Operates on
    detatched zone fingerprints to avoid conflicting gradient pulls.

  AA — Attention Anchoring.
    KL divergence between the soft geometric prior (coverage mask of the union
    box over the patch grid) and the average cross-attention map from the last
    cross-attention layer of RelationTransformer.  Encourages the transformer
    to attend to the geometrically relevant spatial zone when constructing pair
    representations.  Requires ``attn_weights`` to be passed from the
    transformer (Phase 2, enabled when ``attn_weights`` is not None).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelSGGLoss(nn.Module):
    """Combined classification + focal loss for relation prediction.

    Args:
        lambda_cls:    Weight for the cross-entropy classification term.
        lambda_focal:  Weight for the sigmoid focal loss term.
        focal_alpha:   Focal loss alpha (foreground weight).
        focal_gamma:   Focal loss gamma (focusing parameter).
    """

    def __init__(
        self,
        lambda_cls: float = 1.0,
        lambda_focal: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_focal = lambda_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self,
        logits: torch.Tensor,
        valid_mask: torch.Tensor,
        pred_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined relation loss.

        Args:
            logits:      [B, K, V] raw (unactivated) cosine scores.
            valid_mask:  [B, K]    True for real (non-padding) slots.
            pred_labels: [B, K]    GT predicate index per slot, -1 if no GT.
        Returns:
            Dict with keys ``loss``, ``loss_cls``, ``loss_focal``.
            ``loss`` is the weighted sum and has gradients.
        """
        B, K, V = logits.shape
        device = logits.device

        has_gt = (pred_labels >= 0) & valid_mask  # [B, K]

        # ------------------------------------------------------------------
        # 1. Cross-entropy on GT-matched slots
        # ------------------------------------------------------------------
        if has_gt.any():
            gt_logits = logits[has_gt]       # [M, V]
            gt_labels = pred_labels[has_gt]  # [M]
            loss_cls = F.cross_entropy(gt_logits, gt_labels)
        else:
            loss_cls = logits.new_zeros(1).squeeze()

        # ------------------------------------------------------------------
        # 2. Sigmoid focal loss over all valid slots
        # ------------------------------------------------------------------
        # Build dense target tensor: one-hot where GT exists, else zeros
        targets = torch.zeros(B, K, V, device=device)
        if has_gt.any():
            targets[has_gt] = F.one_hot(pred_labels[has_gt], V).float()

        p = torch.sigmoid(logits)  # [B, K, V]
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        alpha_t = targets * self.focal_alpha + (1.0 - targets) * (1.0 - self.focal_alpha)
        focal = alpha_t * (1.0 - p_t).pow(self.focal_gamma) * ce  # [B, K, V]

        n_valid = valid_mask.float().sum().clamp(min=1.0)
        loss_focal = (focal * valid_mask.unsqueeze(-1).float()).sum() / n_valid

        total = self.lambda_cls * loss_cls + self.lambda_focal * loss_focal

        return {
            "loss": total,
            "loss_cls": loss_cls.detach(),
            "loss_focal": loss_focal.detach(),
        }


# ---------------------------------------------------------------------------
# Predicate-Conditioned Spatial Grounding Loss (PCSG)
# ---------------------------------------------------------------------------

class RelSpatialGroundingLoss(nn.Module):
    """Pixel-level predicate grounding via interaction-zone contrastive losses.

    Two loss components (+ optional Attention Anchoring in Phase 2):

    IZSA — Interaction Zone Spatial Alignment
        InfoNCE between zone fingerprints (coverage-weighted pool of backbone
        patches inside the union box, projected to text embedding space) and
        the GT predicate text embeddings ``W``.  Only GT-matched slots
        (``pred_labels >= 0``) contribute.

    CZSC — Cross-Zone Supervised Contrastive (SupCon)
        Pulls zone fingerprints of pairs sharing the same predicate together
        and pushes other predicates apart, at the batch level.  Both IZSA and
        CZSC contribute gradients through ``zone_proj``.

    AA — Attention Anchoring (Phase 2)
        KL divergence between the soft geometric prior derived from the union
        box coverage mask and the mean cross-attention map from the last
        cross-attention layer of RelationTransformer.  Pass ``attn_weights``
        from the transformer to activate this term.

    Args:
        zone_temp:   Temperature for the IZSA InfoNCE. Default: 0.07 (CLIP-matched).
        czsc_temp:   Temperature for CZSC SupCon. Default: 0.1.
    """

    def __init__(
        self,
        zone_temp: float = 0.07,
        czsc_temp: float = 0.1,
    ):
        super().__init__()
        self.zone_temp = zone_temp
        self.czsc_temp = czsc_temp

    # ------------------------------------------------------------------
    # IZSA — InfoNCE between zone fingerprint and predicate text embedding
    # ------------------------------------------------------------------

    def _izsa(
        self,
        z: torch.Tensor,        # [M, t_dim] L2-normalized zone fingerprints, GT slots only
        W: torch.Tensor,        # [V, t_dim] L2-normalized text embeddings
        labels: torch.Tensor,   # [M] GT predicate indices
    ) -> torch.Tensor:
        """Interaction Zone Spatial Alignment loss.

        For each GT pair i, maximises cos(z_i, W[p_i]) / τ against all V
        predicate text embeddings — identical in form to the InfoNCE in
        VocabHead but operating on raw, pre-transformer zone features.
        """
        logits = z @ W.T / self.zone_temp  # [M, V]
        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # CZSC — SupCon on zone fingerprints
    # ------------------------------------------------------------------

    def _czsc(
        self,
        z: torch.Tensor,        # [M, t_dim] L2-normalized zone fingerprints
        labels: torch.Tensor,   # [M] GT predicate indices
    ) -> torch.Tensor:
        """Cross-Zone Supervised Contrastive loss (SupCon, Khosla et al. 2020).

        For each anchor i, treats other GT pairs in the batch with the same
        predicate as positives and all others as negatives.  Degenerate batches
        where no positive pair exists return zero.
        """
        M = z.shape[0]
        if M < 2:
            return z.new_zeros(1).squeeze()

        # Similarity matrix [M, M], masked diagonal
        sim = z @ z.T / self.czsc_temp  # [M, M]
        # Positive mask: same predicate, excluding self
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~torch.eye(
            M, dtype=torch.bool, device=z.device
        )

        if not pos_mask.any():
            return z.new_zeros(1).squeeze()

        # Log-softmax denominator excludes self (standard SupCon)
        diag_mask = torch.eye(M, dtype=torch.bool, device=z.device)
        sim_no_diag = sim.masked_fill(diag_mask, float("-inf"))
        log_denom = torch.logsumexp(sim_no_diag, dim=1, keepdim=True)  # [M, 1]

        n_pos = pos_mask.float().sum(dim=1).clamp(min=1.0)  # [M]
        loss_per_anchor = -(
            (pos_mask.float() * (sim - log_denom)).sum(dim=1) / n_pos
        )
        # Average only over anchors that have at least one positive
        has_pos = pos_mask.any(dim=1)
        return loss_per_anchor[has_pos].mean()

    # ------------------------------------------------------------------
    # AA — Attention Anchoring (Phase 2, activated when attn_weights ≠ None)
    # ------------------------------------------------------------------

    def _aa(
        self,
        attn_weights: torch.Tensor,   # [B, K, h*w] cross-attn map (scene tokens only)
        zone_masks: torch.Tensor,      # [B, K, h*w] coverage-weighted union box mask
        valid_mask: torch.Tensor,      # [B, K] True = real slot
        has_gt: torch.Tensor,          # [B, K] True = GT-matched slot
    ) -> torch.Tensor:
        """Attention Anchoring loss.

        KL( P_geo || A ) where P_geo is the softmax-normalised coverage mask
        of the union box over the patch grid and A is the mean cross-attention
        map averaged over heads from the last cross-attention layer.

        Using KL(prior || prediction) rather than reverse KL to avoid mode-
        seeking towards a single patch — the coverage mask spreads mass over the
        entire interaction zone evenly, encouraging the transformer to cover
        the full zone rather than collapsing to a single salient point.
        """
        eps = 1e-8
        # Normalise masks to probability distributions
        p_geo = zone_masks / (zone_masks.sum(dim=-1, keepdim=True) + eps)  # [B, K, h*w]
        p_attn = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + eps)  # [B, K, h*w]

        # KL(p_geo || p_attn) = sum p_geo * log(p_geo / p_attn)
        kl = (p_geo * (torch.log(p_geo + eps) - torch.log(p_attn + eps))).sum(dim=-1)  # [B, K]

        n_gt = has_gt.float().sum().clamp(min=1.0)
        return (kl * has_gt.float()).sum() / n_gt

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,                            # [B, K, t_dim] zone fingerprints (L2-normed)
        W: torch.Tensor,                            # [V, t_dim] text embeddings (L2-normed)
        pred_labels: torch.Tensor,                  # [B, K] GT predicate index, -1 if no GT
        valid_mask: torch.Tensor,                   # [B, K] True = real slot
        attn_weights: Optional[torch.Tensor] = None,  # [B, K, h*w] Phase 2
        zone_masks: Optional[torch.Tensor] = None,    # [B, K, h*w] Phase 2
    ) -> Dict[str, torch.Tensor]:
        """Compute PCSG losses.

        Args:
            z:            [B, K, t_dim] L2-normalized zone fingerprints.
            W:            [V, t_dim]    L2-normalized predicate text embeddings.
            pred_labels:  [B, K]        GT predicate index per slot, -1 if no GT.
            valid_mask:   [B, K]        True for real (non-padding) slots.
            attn_weights: [B, K, h*w]   Per-head-averaged cross-attention weights
                          from the last cross-attention layer (Phase 2 only).
            zone_masks:   [B, K, h*w]   Coverage-weighted union box masks (Phase 2).
        Returns:
            Dict with keys ``loss_izsa``, ``loss_czsc``, optionally ``loss_aa``.
            Values at GT-absent batches are zero tensors with gradients.
        """
        has_gt = (pred_labels >= 0) & valid_mask  # [B, K]

        if has_gt.any():
            z_gt = z[has_gt]                        # [M, t_dim]
            labels_gt = pred_labels[has_gt]         # [M]

            loss_izsa = self._izsa(z_gt, W, labels_gt)
            loss_czsc = self._czsc(z_gt, labels_gt)
        else:
            loss_izsa = z.new_zeros(1).squeeze()
            loss_czsc = z.new_zeros(1).squeeze()

        out = {
            "loss_izsa": loss_izsa,
            "loss_czsc": loss_czsc,
        }

        if attn_weights is not None and zone_masks is not None:
            out["loss_aa"] = self._aa(attn_weights, zone_masks, valid_mask, has_gt)

        return out
