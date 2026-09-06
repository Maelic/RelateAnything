"""Box-anchored deformable read of the scene feature map.

After the relation transformer, each pair query predicts sampling offsets
around four anchors (subject, object, union and contact box centres), reads
the projected scene map at those points with bilinear sampling, and adds the
weighted read back to the query through a zero-initialised gate.

  * Offsets are expressed in units of the anchor's half-extent and are not
    limited to the box: evidence for a relation often lies outside both boxes
    (the road under a parked car, the hook a towel hangs from). Sampled
    positions are clamped to the image.
  * Several heads sample independent point sets over channel slices, so the
    read covers ``heads x points`` locations per anchor at the cost of one.
  * Per (head, anchor), ``null_slots`` learnable vectors compete in the same
    softmax as the sampled points without reading the image. They let a pair
    attenuate the read instead of pointing off-frame to fetch zeros.
  * Each (head, point) starts at a distinct angle and radius so the points
    receive different gradients from the first step.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableRelRead(nn.Module):
    N_ANCHORS = 4  # subject, object, union, contact

    def __init__(self, d_model: int, n_points: int = 4, heads: int = 8,
                 null_slots: int = 2, null_logit_bias: float = -2.0):
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model {d_model} not divisible by heads {heads}")
        self.n_points = n_points
        self.heads = heads
        self.null_slots = null_slots
        # Diagnostics: with ``capture`` set, forward() keeps the sampled
        # positions, weights and anchors of its last call (visualisation only).
        self.capture = False
        self.last_pos = self.last_w = self.last_null_w = self.last_anchors = None
        A, P, H, S = self.N_ANCHORS, n_points, heads, null_slots

        self.norm = nn.LayerNorm(d_model)
        self.offset_mlp = nn.Linear(d_model, H * A * P * 2)
        self.weight_mlp = nn.Linear(d_model, H * A * (P + S))
        if S:
            self.null_vec = nn.Parameter(torch.zeros(H, A, S, d_model // H))
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.offset_mlp.weight)
        nn.init.zeros_(self.weight_mlp.weight)
        nn.init.zeros_(self.weight_mlp.bias)
        if S:
            # A logit of -2 gives the null slots about 6 percent of the read
            # at initialisation (S=2, P=4) instead of the third an all-zero
            # bias would hand them.
            with torch.no_grad():
                self.weight_mlp.bias.view(H, A, P + S)[..., P:] = null_logit_bias
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gamma = nn.Parameter(torch.zeros(d_model))

        # Ring initialisation: head h owns an angular sector, its points fan
        # across the sector with a growing radius (0.35 half-extents per step).
        bias = torch.zeros(H, A, P, 2)
        for h in range(H):
            for p in range(P):
                theta = 2.0 * math.pi * (h + p / max(P, 1)) / H
                r = 0.35 * (p + 1)
                bias[h,:, p, 0] = math.cos(theta) * r
                bias[h,:, p, 1] = math.sin(theta) * r
        with torch.no_grad():
            self.offset_mlp.bias.copy_(bias.reshape(-1))

    def forward(self, queries: torch.Tensor, scene: torch.Tensor,
                anchors_cxcywh: torch.Tensor) -> torch.Tensor:
        """``queries [B, K, d]``, ``scene [B, d, h, w]``, ``anchors [B, K, 4, 4]``."""
        B, K, d = queries.shape
        A, P, H, S = self.N_ANCHORS, self.n_points, self.heads, self.null_slots
        dh = d // H

        qn = self.norm(queries)
        off = self.offset_mlp(qn).view(B, K, H, A, 1, P, 2)
        centers = anchors_cxcywh[...,:2].view(B, K, 1, A, 1, 1, 2)
        half = (anchors_cxcywh[..., 2:].clamp_min(0.05) * 0.5).view(B, K, 1, A, 1, 1, 2)
        pos = (centers + off * half).clamp(0.0, 1.0)                 # [B,K,H,A,1,P,2]

        lh, lw = scene.shape[-2:]
        scene_h = scene.view(B, H, dh, lh, lw).reshape(B * H, dh, lh, lw)
        # Slice rather than index the level axis: the Slice+Squeeze trace is
        # what the OpenVINO GPU plugin accepts.
        grid = (pos[:,:,:,:, 0:1].squeeze(4)
.permute(0, 2, 1, 3, 4, 5)
.reshape(B * H, K, A * P, 2) * 2.0 - 1.0)
        vals = F.grid_sample(scene_h, grid, mode="bilinear", align_corners=False,
                             padding_mode="zeros").view(B, H, dh, K, A, P)

        w = F.softmax(self.weight_mlp(qn).view(B, K, H, A * (P + S)), dim=-1)
        if self.capture:
            self.last_pos = pos.detach()
            wv = w.view(B, K, H, A, P + S)
            self.last_w = wv[...,:P].reshape(B, K, H, A * P).detach()
            self.last_null_w = wv[..., P:].sum(-1).detach() if S else None
            self.last_anchors = anchors_cxcywh.detach()
        if S:
            nulls = (self.null_vec.view(1, H, A, S, dh)
.permute(0, 1, 4, 2, 3)
.unsqueeze(3).expand(B, H, dh, K, A, S))
            vals = torch.cat([vals, nulls], dim=-1)
        vals = vals.reshape(B * H, dh, K, A * (P + S))
        wh = w.permute(0, 2, 1, 3).reshape(B * H, K, A * (P + S))
        read = torch.einsum("bdkp,bkp->bkd", vals, wh)
        read = read.view(B, H, K, dh).permute(0, 2, 1, 3).reshape(B, K, d)
        return queries + self.gamma * self.out_proj(read)
