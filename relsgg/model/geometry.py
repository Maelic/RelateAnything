"""Box and scene geometry encoders.

BoxPromptEncoder   a box becomes two corner tokens (top-left, bottom-right),
                   each a Fourier encoding of its coordinates projected to
                   ``d_model``. They join the cross-attention memory of the
                   relation transformer.
ScenePosEnc        gated absolute Fourier encoding of the patch grid, added to
                   the scene tokens at every cross-attention site.
RelGeomEncoder     19 scale-invariant features of an ordered (subject, object)
                   pair, projected to ``d_model`` by a small MLP.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def fourier_pe(coords: torch.Tensor, num_freqs: int, max_octave: float) -> torch.Tensor:
    """Sinusoidal features of coordinates in [0, 1].

    Frequencies form a geometric ladder ``2**linspace(0, max_octave,
    num_freqs)``; with ``max_octave=7`` the top band is 128 cycles per image,
    about twice the Nyquist rate of the 28x28 patch grid at 448 px.
    Returns ``[..., 2 * num_freqs]``.
    """
    freqs = 2.0 ** torch.linspace(0.0, float(max_octave), num_freqs,
                                  device=coords.device, dtype=coords.dtype)
    angles = coords.unsqueeze(-1) * freqs * math.pi
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ScenePosEnc(nn.Module):
    """Absolute positional encoding of the patch grid, added to scene tokens.

    ``tokens + gamma * proj(fourier(patch_centres))`` with ``gamma`` zero at
    initialisation, so the model decides during training how much position
    the scene keys carry.
    """

    def __init__(self, d_model: int, num_freqs: int = 16, max_octave: float = 7.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_octave = float(max_octave)
        self.proj = nn.Linear(4 * num_freqs, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gamma = nn.Parameter(torch.zeros(d_model))

    def forward(self, h: int, w: int, device, dtype) -> torch.Tensor:
        """Returns ``[1, h*w, d_model]``, broadcastable over the batch."""
        ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) / h
        xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) / w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        pe = torch.cat([fourier_pe(xx.reshape(-1), self.num_freqs, self.max_octave),
                        fourier_pe(yy.reshape(-1), self.num_freqs, self.max_octave)],
                       dim=-1)
        return (self.gamma * self.proj(pe)).unsqueeze(0)


class BoxPromptEncoder(nn.Module):
    """A box in normalised xyxy becomes a top-left and a bottom-right token."""

    def __init__(self, d_model: int = 512, num_freqs: int = 16,
                 max_octave: float = 7.0):
        super().__init__()
        self.num_freqs = num_freqs
        self.max_octave = float(max_octave)
        self.proj = nn.Linear(4 * num_freqs, d_model)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.corner_bias = nn.Embedding(2, d_model)
        nn.init.normal_(self.corner_bias.weight, std=0.02)

    def _encode_point(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        pe_x = fourier_pe(x, self.num_freqs, self.max_octave)
        pe_y = fourier_pe(y, self.num_freqs, self.max_octave)
        return self.proj(torch.cat([pe_x, pe_y], dim=-1))

    def forward(self, boxes: torch.Tensor) -> torch.Tensor:
        """``[B, N, 4]`` xyxy in [0, 1] -> ``[B, N, 2, d_model]`` (TL, BR)."""
        x1, y1, x2, y2 = boxes.unbind(-1)
        tl = self._encode_point(x1, y1)
        br = self._encode_point(x2, y2)
        device = boxes.device
        tl = tl + self.corner_bias(torch.zeros(1, dtype=torch.long, device=device))
        br = br + self.corner_bias(torch.ones(1, dtype=torch.long, device=device))
        return torch.stack([tl, br], dim=2)

    def encode_pairs(self, sub_boxes: torch.Tensor, obj_boxes: torch.Tensor) -> torch.Tensor:
        """Four tokens per pair: (sub_TL, sub_BR, obj_TL, obj_BR) -> ``[B, K, 4, d_model]``."""
        return torch.cat([self.forward(sub_boxes), self.forward(obj_boxes)], dim=2)


class RelGeomEncoder(nn.Module):
    """Pairwise geometry features of an ordered (subject, object) pair.

    Features 0-14 are box geometry; 15-18 describe the regions inside the
    boxes and take their box values when no region (mask) is given:

      0  dx           horizontal displacement over subject width
      1  dy           vertical displacement over subject height
      2  log_wr       log width ratio (object / subject)
      3  log_hr       log height ratio
      4  log_ar       log area ratio
      5  log_as       log subject area
      6  log_ao       log object area
      7  iou          box intersection over union
      8  s_in         share of the subject box inside the intersection
      9  o_in         share of the object box inside the intersection
      10 asp_s        log subject aspect ratio
      11 asp_o        log object aspect ratio
      12 cos_theta    direction subject -> object
      13 sin_theta
      14 delta_cy     vertical centre offset
      15 fill_s       subject region area over its box area (1 for a box)
      16 fill_o       object region area over its box area
      17 r_iou        region intersection over union (box IoU for boxes)
      18 r_contact    region intersection over the smaller region

    Values pass through ``10 * tanh(x / 10)``, which keeps the ordering of the
    hard clamp at +-10 while leaving a gradient at the rails.
    """

    NUM_GEO: int = 19
    NUM_BOX_GEO: int = 15

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self.NUM_GEO, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.LayerNorm(d_model),
)
        with torch.no_grad():
            self.mlp[0].weight[:, self.NUM_BOX_GEO:].zero_()

    @staticmethod
    def features(sub_boxes: torch.Tensor, obj_boxes: torch.Tensor,
                 region: "tuple | None" = None) -> torch.Tensor:
        """Raw features, ``[..., 19]``. ``region`` is an optional
        ``(fill_s, fill_o, r_iou, r_contact)`` tuple of tensors broadcastable
        to ``sub_boxes[..., 0]``; without it the box values are used."""
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
            ones = torch.ones_like(iou)
            fill_s, fill_o = ones, ones
            r_iou = iou
            r_contact = intersection / torch.minimum(s_area, o_area).clamp(min=eps)
        else:
            fill_s, fill_o, r_iou, r_contact = region

        feats = torch.stack(
            [dx, dy, log_wr, log_hr, log_ar,
             log_as, log_ao, iou, s_in, o_in,
             asp_s, asp_o, cos_theta, sin_theta, delta_cy,
             fill_s.expand_as(iou), fill_o.expand_as(iou),
             r_iou.expand_as(iou), r_contact.expand_as(iou)],
            dim=-1)
        return 10.0 * torch.tanh(feats / 10.0)

    def forward(self, sub_boxes: torch.Tensor, obj_boxes: torch.Tensor,
                region: "tuple | None" = None) -> torch.Tensor:
        """``[..., 4]`` cxcywh pairs -> ``[..., d_model]``."""
        return self.mlp(self.features(sub_boxes, obj_boxes, region))
