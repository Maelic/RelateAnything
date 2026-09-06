"""ROI-align-free spatial feature extraction via soft cross-attention pooling.

SoftSpatialPool
    For each box, constructs a coordinate-injected query by adding a
    BoxPromptEncoder token to a learnable base query. The query then attends
    to backbone patch tokens via a single cross-attention layer, producing a
    soft-pooled feature that is spatially aware without any grid resampling.

    Replaces the old coverage-weighted ``masked_avg_pool``.

masked_avg_pool
    Kept as a lightweight fallback (no learnable parameters) that computes
    coverage-weighted averages over patch cells. Used when SoftSpatialPool is
    not initialized.

union_box
    Unchanged utility: tight bounding box of a subject–object pair.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import BoxPromptEncoder


class SoftSpatialPool(nn.Module):
    """Coordinate-aware soft spatial pooling via cross-attention.

    For each detected box, a box-position-encoded query attends to all
    backbone patch tokens (the full scene) through a single multi-head
    cross-attention layer. The corner tokens from BoxPromptEncoder are added
    to the base query so the attention heads can learn to focus on the correct
    spatial region — differentiably, with no ROI-Align required.

    Args:
        d_model:     Feature dimension (must match backbone d_model).
        n_heads:     Number of attention heads.
        num_freqs:   Fourier frequency bands passed to BoxPromptEncoder.
    """

    def __init__(
        self,
        d_model: int = 768,
        n_heads: int = 8,
        num_freqs: int = 64,
        max_octave: "float | None" = None,
        scene_pe: bool = False,
        role_queries: bool = False,
        mode_gated: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        # Mode gate (see RelSGGConfig.mode_gated): a per-image bit m multiplies
        # cov_lambda and adds m * mode_embed to the region query, so m=0 (box)
        # is the box-only forward EXACTLY whatever the mask path has learned.
        self.mode_embed = (nn.Parameter(torch.zeros(1, 1, d_model))
                           if mode_gated else None)

        # One learnable base query per call (broadcast over batch and boxes)
        self.base_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.base_query, std=0.02)

        # Gated absolute PE on the pooling KEYS. This is the site where it
        # matters most: the query encodes the box position but the keys were
        # raw content — there was NO mechanism to attend "inside the box"
        # except whatever absolute-position leakage survives in the backbone
        # features. See geometry.ScenePosEnc (zero-init gate).
        self.scene_pe = None
        if scene_pe:
            from .geometry import ScenePosEnc
            self.scene_pe = ScenePosEnc(d_model)

        # Per-ROLE query biases (object / union / contact). The same
        # base_query previously served three semantically different pooling
        # roles, distinguishable only by box coordinates. Zero-init => old
        # checkpoints and epoch 0 are bit-identical; roles differentiate in
        # training.
        self.role_bias = (nn.Parameter(torch.zeros(3, d_model))
                          if role_queries else None)

        # Positional encoder — produces a d_model offset for the TL corner
        # (we use just the single fused token average of TL + BR for pooling)
        self.box_pe = BoxPromptEncoder(d_model=d_model, num_freqs=num_freqs,
                                       max_octave=max_octave)

        # Single cross-attention layer: box query (Q) ← patch tokens (K/V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            bias=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.n_heads = n_heads

        # Region-shape gate. The pooled query only ever knew the box CORNERS,
        # so it could learn to suppress background but never saw region SHAPE.
        # This adds lambda * log(coverage) to the attention logits: at 0 the
        # module is bit-identical to the box-only model, at 1 attention is
        # confined to the region and weighted by per-patch coverage. One
        # scalar per head, zero-init — if it stays at 0 after training, the
        # model does not want shape in the pooling, and that is a real answer
        # for 8 parameters (see the SAM-style conv alternative in roi docs).
        self.cov_lambda = nn.Parameter(torch.zeros(n_heads))

    def _boxes_cxcywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack(
            [
                (cx - w * 0.5).clamp(0.0, 1.0),
                (cy - h * 0.5).clamp(0.0, 1.0),
                (cx + w * 0.5).clamp(0.0, 1.0),
                (cy + h * 0.5).clamp(0.0, 1.0),
            ],
            dim=-1,
        )

    def forward(
        self,
        F_map: torch.Tensor,
        boxes: torch.Tensor,
        cov: "torch.Tensor | None" = None,
        roles: "torch.Tensor | None" = None,
        mode: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Extract per-region features via coordinate-aware cross-attention.

        Args:
            F_map:  [B, h, w, d_model]  backbone spatial features.
            boxes:  [B, N, 4]           boxes in normalized cxcywh ∈ [0, 1].
            cov:    [B, N, h*w] optional per-region coverage in [0, 1], already
                    at the patch-grid resolution. None => box-only behaviour,
                    bit-identical to the pre-mask model.
            roles:  [N] long, optional — 0 object / 1 union / 2 contact, index
                    into role_bias (only when built with role_queries=True).
                    None = all object.
        Returns:
            [B, N, d_model]  one feature vector per region.
        """
        B, h, w, d = F_map.shape
        N = boxes.shape[1]

        # Flatten patch tokens: [B, h*w, d_model]
        patches = F_map.reshape(B, h * w, d)
        if self.scene_pe is not None:
            patches = patches + self.scene_pe(h, w, patches.device,
                                              patches.dtype)

        # Build coordinate-aware query: [B, N, d_model]
        boxes_xyxy = self._boxes_cxcywh_to_xyxy(boxes)       # [B, N, 4]
        corner_tokens = self.box_pe(boxes_xyxy)               # [B, N, 2, d_model]
        # Average TL + BR corner embeddings as the positional offset
        pos_offset = corner_tokens.mean(dim=2)                # [B, N, d_model]
        query = self.base_query.expand(B, N, d) + pos_offset  # [B, N, d_model]
        if self.role_bias is not None:
            # roles=None means "all object" (role 0) — indexed explicitly so
            # role_bias[0] trains too (and the dead-param audit stays quiet).
            if roles is None:
                roles = torch.zeros(N, dtype=torch.long, device=boxes.device)
            query = query + self.role_bias[roles].unsqueeze(0)  # [1, N, d]
        if self.mode_embed is not None and mode is not None:
            # m * e_mode: untouched for box images (m=0).
            query = query + mode.view(B, 1, 1).to(query.dtype) * self.mode_embed

        # Region-shape bias over the patch grid, added to the attention logits.
        # log() so the bias is scale-free in coverage and lambda interpolates
        # smoothly from "ignore shape" (0) to "coverage-weighted" (1).
        attn_mask = None
        if cov is not None:
            log_cov = torch.log(cov.clamp(0.0, 1.0) + 1e-4)          # [B, N, hw]
            lam = self.cov_lambda.view(1, -1, 1, 1)                  # [1,H,1,1]
            if mode is not None:
                # m * lambda: a box image gets NO shape bias, whatever lambda
                # has learned for masks -- the box path stays the box-only net.
                lam = lam * mode.view(B, 1, 1, 1).to(lam.dtype)      # [B,H,1,1]
            attn_mask = (lam * log_cov.unsqueeze(1)                  # [B,H,N,hw]
                         ).expand(B, self.n_heads, N, h * w)
            attn_mask = attn_mask.reshape(B * self.n_heads, N, h * w).to(patches.dtype)

        # Cross-attention: N box queries attend to hw shared patch tokens.
        # PyTorch MHA natively supports query/key with different sequence lengths
        # — no tiling required.  query=[B, N, d], key=value=[B, hw, d] → [B, N, d].
        attn_out, _ = self.cross_attn(
            query=query,
            key=patches,
            value=patches,
            attn_mask=attn_mask,
        )  # [B, N, d]

        out = self.norm(attn_out)  # [B, N, d_model]
        return out


# ---------------------------------------------------------------------------
# Legacy fallback: coverage-weighted average pooling (no learnable parameters)
# ---------------------------------------------------------------------------

def box_coverage_weights(
    boxes: torch.Tensor,
    h: int,
    w: int,
) -> torch.Tensor:
    """Normalized per-box coverage weights over the patch grid.

    Returns the bilinear-area intersection between each box and each patch
    cell, normalized to sum to 1 over the spatial dimension.  This is the
    weight map used internally by :func:`masked_avg_pool` — exposed here so
    callers that need the raw ``[B, N, h*w]`` distribution (e.g. the AA loss
    zone mask) can obtain it without a dummy feature-map trick.

    Args:
        boxes: [B, N, 4]  normalized cxcywh ∈ [0, 1].
        h:     patch-grid height.
        w:     patch-grid width.
    Returns:
        [B, N, h*w]  normalized coverage weights (sum to 1 over last dim).
    """
    device = boxes.device
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw * 0.5).clamp(0.0, 1.0)
    y1 = (cy - bh * 0.5).clamp(0.0, 1.0)
    x2 = (cx + bw * 0.5).clamp(0.0, 1.0)
    y2 = (cy + bh * 0.5).clamp(0.0, 1.0)

    gx1 = torch.arange(w, device=device, dtype=torch.float32) / w
    gx2 = (torch.arange(w, device=device, dtype=torch.float32) + 1) / w
    gy1 = torch.arange(h, device=device, dtype=torch.float32) / h
    gy2 = (torch.arange(h, device=device, dtype=torch.float32) + 1) / h

    ix = (torch.minimum(x2.unsqueeze(-1), gx2) - torch.maximum(x1.unsqueeze(-1), gx1)).clamp(min=0.0)
    iy = (torch.minimum(y2.unsqueeze(-1), gy2) - torch.maximum(y1.unsqueeze(-1), gy1)).clamp(min=0.0)

    weights = (iy.unsqueeze(-1) * ix.unsqueeze(-2)).reshape(boxes.shape[0], -1, h * w)
    return weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)


def masked_avg_pool(
    F_map: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    """Per-box coverage-weighted average pooling (parameter-free fallback).

    Each patch token is weighted by how much of its cell area overlaps the box.

    Args:
        F_map:  [B, h, w, C]  spatial feature map.
        boxes:  [B, N, 4]     boxes in normalized cxcywh ∈ [0, 1].
    Returns:
        [B, N, C]  per-box feature vectors.
    """
    B, h, w, C = F_map.shape

    weights = box_coverage_weights(boxes, h, w)  # [B, N, h*w]

    F_flat = F_map.reshape(B, h * w, C)
    return torch.bmm(weights, F_flat)


# ---------------------------------------------------------------------------
# Union box utility (unchanged)
# ---------------------------------------------------------------------------

def union_box(
    boxes_i: torch.Tensor,
    boxes_j: torch.Tensor,
) -> torch.Tensor:
    """Tight bounding box of a subject–object pair (the union box).

    Args:
        boxes_i: [B, K, 4] normalized cxcywh.
        boxes_j: [B, K, 4] normalized cxcywh.
    Returns:
        [B, K, 4] the union box in normalized cxcywh.
    """

    def _to_xyxy(b: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], -1)

    def _to_cxcywh(b: torch.Tensor) -> torch.Tensor:
        x1, y1, x2, y2 = b.unbind(-1)
        return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], -1)

    bi = _to_xyxy(boxes_i)
    bj = _to_xyxy(boxes_j)
    u = torch.stack(
        [
            torch.minimum(bi[..., 0], bj[..., 0]),
            torch.minimum(bi[..., 1], bj[..., 1]),
            torch.maximum(bi[..., 2], bj[..., 2]),
            torch.maximum(bi[..., 3], bj[..., 3]),
        ],
        dim=-1,
    )
    return _to_cxcywh(u)
