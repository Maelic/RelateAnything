"""Geometry encoders for bounding-box coordinate injection.

Two components:

BoxPromptEncoder
    Converts a bounding box into two learnable corner tokens (top-left and
    bottom-right), each encoded with sinusoidal Fourier positional encoding
    projected to d_model. These tokens are appended to the cross-attention
    memory in RelationTransformer, acting as spatial attractors that bias
    attention toward the correct image region — without ROI-Align.

RelGeomEncoder
    Encodes the *relationship* between a subject–object pair as a dense
    vector of 15 scale-invariant geometry features, then projects to d_model
    via a two-layer MLP. This captures explicit spatial predicates such as
    "above" or "inside". Slot-dropped into the pair representation.

    (Replaces the old GeoEncoder, which is aliased for compatibility.)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fourier positional encoding helper
# ---------------------------------------------------------------------------

def _fourier_pe(coords: torch.Tensor, num_freqs: int,
                max_octave: "float | None" = None) -> torch.Tensor:
    """Encode scalar coordinates in [0, 1] with sinusoidal Fourier features.

    Args:
        coords:     [...] float32 in [0, 1].
        num_freqs:  Number of frequency bands.
        max_octave: Top frequency = 2**max_octave. None = LEGACY ladder
                    2**arange(num_freqs) — with the historical num_freqs=64
                    that tops out at 2^63, where float32's ULP exceeds the
                    sine's period around band ~19: bands 20-63 are
                    deterministic quantization noise, NOT smooth in the input
                    (nearby boxes decorrelate there). Kept only so existing
                    checkpoints reproduce bit-exactly. New runs should set a
                    geometric ladder 2**linspace(0, max_octave, num_freqs):
                    max_octave=7 tops at 128 cycles/image, ~2x the 56-cell
                    grid Nyquist (NeRF uses L=10 on far finer geometry;
                    Tancik et al. 2020 — excess bandwidth = noise-fitting).
    Returns:
        [..., 2 * num_freqs] float32.
    """
    if max_octave is None:
        freqs = 2.0 ** torch.arange(num_freqs, device=coords.device,
                                    dtype=coords.dtype)
    else:
        freqs = 2.0 ** torch.linspace(0.0, float(max_octave), num_freqs,
                                      device=coords.device, dtype=coords.dtype)
    # [..., num_freqs]
    angles = coords.unsqueeze(-1) * freqs * math.pi
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


# ---------------------------------------------------------------------------
# ScenePosEnc
# ---------------------------------------------------------------------------

class ScenePosEnc(nn.Module):
    """Absolute Fourier positional encoding for the scene patch grid.

    WHY: the cross-attention memories (RelationTransformer, the interaction
    block's grounding stage, SoftSpatialPool's keys) carried NO positional
    signal — scene keys were pure content, so a query could not address "near
    (x, y)" content-independently. The only spatially-addressable tokens were
    the box-corner tokens (geometry with no content), which is exactly the
    shortcut the attention collapsed onto (measured scene mass 0.000 under
    LoRA, runs/analysis/relation_attn_stats.json). DETR adds spatial PE to
    the keys at every layer for precisely this reason; the box corner tokens
    are already Fourier encodings of coordinates, so giving scene keys the
    same encoding family makes location matching a learnable dot product.

    Added as ``tokens + gamma * proj(fourier(patch_centers))`` with GAMMA
    ZERO-INIT: old checkpoints rebuild bit-identically and training grows
    into the PE. Deviation from DETR noted: nn.TransformerDecoderLayer uses
    memory as K AND V, so the PE lands on values too — the gate lets the
    model regulate that trade itself.

    This module is NEW (no checkpoint legacy), so it always uses the fixed
    geometric frequency ladder — never the legacy 2**arange(64) one whose
    upper bands are float32 noise.
    """

    def __init__(self, d_model: int, num_freqs: int = 16,
                 max_octave: float = 7.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_octave = float(max_octave)
        self.proj = nn.Linear(4 * num_freqs, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gamma = nn.Parameter(torch.zeros(d_model))  # LayerScale at 0

    def forward(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Returns [1, h*w, d_model], broadcastable over the batch."""
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")          # [h, w]
        pe = torch.cat([_fourier_pe(xx.reshape(-1), self.num_freqs, self.max_octave),
                        _fourier_pe(yy.reshape(-1), self.num_freqs, self.max_octave)],
                       dim=-1)                                   # [h*w, 4*nf]
        return (self.gamma * self.proj(pe)).unsqueeze(0)         # [1, h*w, d]


# ---------------------------------------------------------------------------
# BoxPromptEncoder
# ---------------------------------------------------------------------------

class BoxPromptEncoder(nn.Module):
    """Box → two spatial corner tokens for cross-attention injection.

    Each box (x1, y1, x2, y2) normalized to [0, 1] is encoded as two tokens:
      - TL token: Fourier PE of (x1, y1) + learned "top-left corner" bias
      - BR token: Fourier PE of (x2, y2) + learned "bottom-right corner" bias

    The Fourier encoding for a single (x, y) point uses ``num_freqs`` bands
    per coordinate, so the raw PE dimension is 4 * num_freqs (sin+cos of x,
    sin+cos of y). This is projected to d_model with a linear layer.

    Returned tokens are ready to be concatenated to the cross-attention K/V
    memory in ``RelationTransformer``.

    Args:
        d_model:   Output token dimension.
        num_freqs: Number of Fourier frequency bands per coordinate.
    """

    def __init__(self, d_model: int = 512, num_freqs: int = 64,
                 max_octave: "float | None" = None):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_octave = max_octave  # None = legacy 2**arange ladder
        pe_dim = 4 * num_freqs  # sin+cos for x, sin+cos for y

        self.proj = nn.Linear(pe_dim, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # Learned additive bias distinguishing TL vs BR corners
        self.corner_bias = nn.Embedding(2, d_model)
        nn.init.normal_(self.corner_bias.weight, std=0.02)

    def _encode_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Encode a (x, y) point to a d_model vector.

        Args:
            x, y: [...] float32 in [0, 1].
        Returns:
            [..., d_model].
        """
        pe_x = _fourier_pe(x, self.num_freqs, self.max_octave)  # [..., 2*nf]
        pe_y = _fourier_pe(y, self.num_freqs, self.max_octave)  # [..., 2*nf]
        return self.proj(torch.cat([pe_x, pe_y], dim=-1))  # [..., d_model]

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """Encode boxes to corner token pairs.

        Args:
            boxes: [B, N, 4] normalized xyxy in [0, 1].
        Returns:
            tokens: [B, N, 2, d_model] — axis-2 is (TL_token, BR_token).
        """
        x1, y1, x2, y2 = boxes.unbind(-1)      # each [B, N]

        tl = self._encode_point(x1, y1)         # [B, N, d_model]
        br = self._encode_point(x2, y2)         # [B, N, d_model]

        # Add corner-type learned bias
        device = boxes.device
        tl = tl + self.corner_bias(torch.zeros(1, dtype=torch.long, device=device))
        br = br + self.corner_bias(torch.ones(1, dtype=torch.long, device=device))

        return torch.stack([tl, br], dim=2)     # [B, N, 2, d_model]

    def encode_pairs(
        self,
        sub_boxes: torch.Tensor,
        obj_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Encode subject + object boxes into a flat sequence of corner tokens.

        Produces 4 tokens per pair: (sub_TL, sub_BR, obj_TL, obj_BR).

        Args:
            sub_boxes: [B, K, 4] normalized xyxy.
            obj_boxes: [B, K, 4] normalized xyxy.
        Returns:
            [B, K, 4, d_model]
        """
        sub_tokens = self.forward(sub_boxes)  # [B, K, 2, d_model]
        obj_tokens = self.forward(obj_boxes)  # [B, K, 2, d_model]
        return torch.cat([sub_tokens, obj_tokens], dim=2)  # [B, K, 4, d_model]


# ---------------------------------------------------------------------------
# RelGeomEncoder  (updated GeoEncoder)
# ---------------------------------------------------------------------------

class RelGeomEncoder(nn.Module):
    """Pairwise geometry encoder for explicit spatial relation features.

    Computes 15 hand-crafted, scale-invariant geometry features for each
    ordered subject–object pair, then projects them to d_model via a small
    MLP. This captures explicit spatial predicates (above, inside, next-to).

    Feature list (indices 0–14):
      0  dx          relative horizontal displacement (normalised by subject width)
      1  dy          relative vertical displacement   (normalised by subject height)
      2  log_wr      log width ratio  (obj / sub)
      3  log_hr      log height ratio (obj / sub)
      4  log_ar      log area  ratio  (obj / sub)
      5  log_as      log subject area (relative to image)
      6  log_ao      log object  area (relative to image)
      7  iou         intersection-over-union
      8  s_in        fraction of subject box covered by the intersection
      9  o_in        fraction of object  box covered by the intersection
      10 asp_s       log aspect ratio of subject  (w / h)
      11 asp_o       log aspect ratio of object   (w / h)
      12 cos_theta   cosine of the directed sub→obj angle
      13 sin_theta   sine   of the directed sub→obj angle
      14 delta_cy    raw vertical-centre offset (obj_cy − sub_cy)
      15 fill_s      subject region area / its box area   (1.0 for a box)
      16 fill_o      object  region area / its box area   (1.0 for a box)
      17 r_iou       region-vs-region IoU                 (= box IoU for boxes)
      18 r_contact   region intersection / min(area)      (= box value for boxes)

    Features 15-18 are NOT gated by a "has mask" flag and need no null
    embedding, because a box is a genuine region with genuine values for all
    four: it fills its own bounding box exactly (1.0), and its region overlap
    IS its box overlap. Contrast SAM's prompt encoder, which does carry a
    `no_mask_embed` — there the mask is refinement feedback, so its absence is
    real absence. Here box and mask describe the same region at different
    fidelity, so `fill` doubles as a *graded* modality signal: a face-on book
    at 0.95 sits near the box regime, a bicycle at 0.25 far from it.
    """

    NUM_GEO: int = 19
    NUM_BOX_GEO: int = 15          # features 0-14; unchanged since v33

    def __init__(self, d_model: int = 512, squash: bool = False,
                 mode_gated: bool = False, region_adjacency: bool = False):
        super().__init__()
        # squash=True replaces the hard clamp(-10,10) with 10*tanh(x/10):
        # the clamp zeroes gradient at the rails, and the rails are COMMON —
        # dx = (o_cx-s_cx)/s_w exceeds 10 for any small subject with a distant
        # object, and log_as = log(1e-6) = -13.8 pins every tiny box's area
        # features. tanh keeps ordering and gradient everywhere. False =
        # prior runs bit-identical (values differ a few % in |x| in [3,10]).
        self.squash = bool(squash)
        self.mlp = nn.Sequential(
            nn.Linear(self.NUM_GEO, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
        )
        # Zero the columns of the 4 region features so a warm start from a
        # 15-feature checkpoint is bit-exact and epoch 0 reproduces the
        # box-only model. Training moves them off zero from the first step.
        with torch.no_grad():
            self.mlp[0].weight[:, self.NUM_BOX_GEO:].zero_()
        # Mode-gated region path (RelSGGConfig.mode_gated). The released tower
        # reads masks ONLY through columns 15-18, whose weights were fitted on
        # BOX overlap statistics; mask overlap is systematically lower for the
        # same pair, and that alone costs 29% R@50 on PSG masks. Here the 19
        # columns ALWAYS carry the box-derived values (a box image is the
        # box-only network exactly) and mask information enters as
        #     m * W_delta @ (fill_s-1, fill_o-1, r_iou-iou_box, r_contact-contact_box)
        # added to the first layer's pre-activation, zero-init.
        self.mode_gated = bool(mode_gated)
        # region_adjacency: a 5th adapter column, region BOUNDARY ADJACENCY.
        # Why it is needed: r_contact (col 18) is an INTERSECTION, and for masks
        # that partition the image -- panoptic/PSG -- exact intersection is
        # identically zero for every pair (measured: 0.0% of 765 PSG test pairs
        # overlap at all, against 19% for SAM box-prompted masks). The feature
        # is therefore degenerate exactly where contact predicates live, while
        # the box column it was trained against reads ~0.19. Adjacency --
        # overlap after dilating one region by a cell -- restores the signal
        # (fires on 29.9% of PSG pairs, mean 0.16) and is NOT redundant with the
        # box value (corr 0.46; on 2.6% of pairs the boxes overlap heavily while
        # the masks do not touch at all).
        # It enters ONLY here, in the zero-init mode-gated adapter, never in the
        # 19 columns of features(): NUM_GEO stays 19, so every existing
        # checkpoint loads and the box path stays bit-identical.
        self.region_adjacency = bool(region_adjacency)
        self.region_delta = None
        if mode_gated:
            self.region_delta = nn.Linear(5 if self.region_adjacency else 4,
                                          d_model // 2, bias=False)
            nn.init.zeros_(self.region_delta.weight)

    @staticmethod
    def features(
        sub_boxes: torch.Tensor,
        obj_boxes: torch.Tensor,
        region: "tuple | None" = None,
        squash: bool = False,
    ) -> torch.Tensor:
        """Compute raw geometry features.

        Args:
            sub_boxes: [..., 4] normalized cxcywh.
            obj_boxes: [..., 4] normalized cxcywh.
            region:    optional ``(fill_s, fill_o, r_iou, r_contact)``, each
                       broadcastable to ``sub_boxes[..., 0]``. When None the
                       box-equivalent values are synthesised, which is exactly
                       what a box region would produce — not a placeholder.
        Returns:
            [..., 19] float32, clamped to [-10, 10].
        """
        eps = 1e-6
        s_cx, s_cy, s_w, s_h = sub_boxes.unbind(-1)
        o_cx, o_cy, o_w, o_h = obj_boxes.unbind(-1)

        dx = (o_cx - s_cx) / (s_w + eps)
        dy = (o_cy - s_cy) / (s_h + eps)
        log_wr = torch.log((o_w + eps) / (s_w + eps))
        log_hr = torch.log((o_h + eps) / (s_h + eps))
        log_ar = torch.log((o_w * o_h + eps) / (s_w * s_h + eps))
        log_as = torch.log(s_w * s_h + eps)
        log_ao = torch.log(o_w * o_h + eps)

        s_x1, s_y1 = s_cx - s_w * 0.5, s_cy - s_h * 0.5
        s_x2, s_y2 = s_cx + s_w * 0.5, s_cy + s_h * 0.5
        o_x1, o_y1 = o_cx - o_w * 0.5, o_cy - o_h * 0.5
        o_x2, o_y2 = o_cx + o_w * 0.5, o_cy + o_h * 0.5

        iw = (torch.minimum(s_x2, o_x2) - torch.maximum(s_x1, o_x1)).clamp(min=0.0)
        ih = (torch.minimum(s_y2, o_y2) - torch.maximum(s_y1, o_y1)).clamp(min=0.0)
        intersection = iw * ih

        s_area = (s_w * s_h).clamp(min=eps)
        o_area = (o_w * o_h).clamp(min=eps)
        iou = intersection / (s_area + o_area - intersection + eps)
        s_in = intersection / s_area
        o_in = intersection / o_area

        asp_s = torch.log((s_w / (s_h + eps)).clamp(min=eps))
        asp_o = torch.log((o_w / (o_h + eps)).clamp(min=eps))

        dist = ((o_cx - s_cx).pow(2) + (o_cy - s_cy).pow(2)).clamp(min=eps).sqrt()
        cos_theta = (o_cx - s_cx) / dist
        sin_theta = (o_cy - s_cy) / dist
        delta_cy = o_cy - s_cy

        if region is None:
            # A box IS a region: it fills its own bounding box, and its region
            # overlap is its box overlap. These are the true values, so the
            # box-only path stays numerically identical to v33-v44.
            ones = torch.ones_like(iou)
            fill_s, fill_o = ones, ones
            r_iou = iou
            r_contact = intersection / torch.minimum(s_area, o_area).clamp(min=eps)
        else:
            fill_s, fill_o, r_iou, r_contact = region

        feats = torch.stack(
            [
                dx, dy, log_wr, log_hr, log_ar,
                log_as, log_ao, iou, s_in, o_in,
                asp_s, asp_o, cos_theta, sin_theta, delta_cy,
                fill_s.expand_as(iou), fill_o.expand_as(iou),
                r_iou.expand_as(iou), r_contact.expand_as(iou),
            ],
            dim=-1,
        )
        if squash:
            return 10.0 * torch.tanh(feats / 10.0)
        return feats.clamp(-10.0, 10.0)

    def forward(
        self,
        sub_boxes: torch.Tensor,
        obj_boxes: torch.Tensor,
        region: "tuple | None" = None,
        mode: "torch.Tensor | None" = None,
        region_adj: "tuple | None" = None,
    ) -> torch.Tensor:
        """Project geometry features to d_model.

        Args:
            sub_boxes: [..., 4] normalized cxcywh.
            obj_boxes: [..., 4] normalized cxcywh.
            region:    optional (fill_s, fill_o, r_iou, r_contact); see features().
            mode:      mode-gated only: per-image bit broadcastable to
                       [..., 1] (1 = the region tuple is a mask, 0 = box).
            region_adj: region_adjacency only -- the boundary-adjacency DELTA
                       (region minus its own box), already differenced in
                       relsgg/model.py. It must be computed there because both
                       sides have to come from the SAME rasteriser: an analytic
                       box reference disagrees with a max-pooled raster by ~0.4
                       on box regions (dilating fractional edge coverage grows
                       the support by up to two cells, not one), which would
                       make the column a rasterisation artefact instead of a
                       mask-vs-box difference.
        Returns:
            [..., d_model].
        """
        if not self.mode_gated:
            return self.mlp(self.features(sub_boxes, obj_boxes, region,
                                          squash=self.squash))
        # Box-derived 19 columns regardless of `region`: the box forward.
        feats = self.features(sub_boxes, obj_boxes, None, squash=self.squash)
        h = self.mlp[0](feats)
        if region is not None and mode is not None:
            raw = self.features(sub_boxes, obj_boxes, None, squash=False)
            iou_b = raw[..., 7]
            con_b = torch.maximum(raw[..., 8], raw[..., 9])  # inter / min(area)
            fill_s, fill_o, r_iou, r_contact = region
            cols = [fill_s.expand_as(iou_b) - 1.0,
                    fill_o.expand_as(iou_b) - 1.0,
                    r_iou.expand_as(iou_b) - iou_b,
                    r_contact.expand_as(iou_b) - con_b]
            if self.region_adjacency:
                if region_adj is None:
                    # rasters present but adjacency not supplied: contribute
                    # nothing rather than silently shifting the column.
                    cols.append(torch.zeros_like(iou_b))
                else:
                    cols.append(region_adj.expand_as(iou_b))
            d = torch.stack(cols, dim=-1)
            m = mode.to(h.dtype)
            while m.dim() < h.dim():
                m = m.unsqueeze(-1)
            h = h + m * self.region_delta(d.to(h.dtype))
        for layer in self.mlp[1:]:
            h = layer(h)
        return h


# Backward-compatible alias
GeoEncoder = RelGeomEncoder
