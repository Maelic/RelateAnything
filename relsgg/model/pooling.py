"""Region pooling and box utilities."""
from __future__ import annotations

import torch
import torch.nn as nn

from.geometry import BoxPromptEncoder, ScenePosEnc


class SoftSpatialPool(nn.Module):
    """One feature vector per region by cross-attention over the patch grid.

    The query for a box is a learned base query plus the mean of its two
    corner tokens; it attends to every patch token of the image, so the
    receptive field is the whole scene and no ROI-Align is involved.

    ``cov`` (per-region coverage of the patch grid, from a mask) adds
    ``cov_lambda * log(coverage)`` to the attention logits. ``cov_lambda`` is
    zero at initialisation and stays zero in models trained on boxes, so masks
    given at inference change the geometry features but not the pooling.
    """

    def __init__(self, d_model: int = 768, n_heads: int = 8,
                 num_freqs: int = 16, max_octave: float = 7.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.base_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.base_query, std=0.02)
        self.scene_pe = ScenePosEnc(d_model)
        self.box_pe = BoxPromptEncoder(d_model=d_model, num_freqs=num_freqs,
                                       max_octave=max_octave)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True, bias=True)
        self.norm = nn.LayerNorm(d_model)
        self.cov_lambda = nn.Parameter(torch.zeros(n_heads))

    @staticmethod
    def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([(cx - w * 0.5).clamp(0.0, 1.0), (cy - h * 0.5).clamp(0.0, 1.0),
                            (cx + w * 0.5).clamp(0.0, 1.0), (cy + h * 0.5).clamp(0.0, 1.0)],
                           dim=-1)

    def forward(self, F_map: torch.Tensor, boxes: torch.Tensor,
                cov: "torch.Tensor | None" = None) -> torch.Tensor:
        """``F_map [B, h, w, d]``, ``boxes [B, N, 4]`` cxcywh, optional
        ``cov [B, N, h*w]`` coverage in [0, 1] -> ``[B, N, d]``."""
        B, h, w, d = F_map.shape
        N = boxes.shape[1]
        patches = F_map.reshape(B, h * w, d)
        patches = patches + self.scene_pe(h, w, patches.device, patches.dtype)

        corner_tokens = self.box_pe(self._cxcywh_to_xyxy(boxes))     # [B, N, 2, d]
        query = self.base_query.expand(B, N, d) + corner_tokens.mean(dim=2)

        attn_mask = None
        if cov is not None:
            log_cov = torch.log(cov.clamp(0.0, 1.0) + 1e-4)            # [B, N, hw]
            lam = self.cov_lambda.view(1, -1, 1, 1)
            attn_mask = (lam * log_cov.unsqueeze(1)).expand(B, self.n_heads, N, h * w)
            attn_mask = attn_mask.reshape(B * self.n_heads, N, h * w).to(patches.dtype)

        attn_out, _ = self.cross_attn(query=query, key=patches, value=patches,
                                      attn_mask=attn_mask)
        return self.norm(attn_out)


def union_box(boxes_i: torch.Tensor, boxes_j: torch.Tensor) -> torch.Tensor:
    """Tight box around two cxcywh boxes, in cxcywh."""

    def _to_xyxy(b):
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5], -1)

    bi, bj = _to_xyxy(boxes_i), _to_xyxy(boxes_j)
    x1 = torch.minimum(bi[..., 0], bj[..., 0])
    y1 = torch.minimum(bi[..., 1], bj[..., 1])
    x2 = torch.maximum(bi[..., 2], bj[..., 2])
    y2 = torch.maximum(bi[..., 3], bj[..., 3])
    return torch.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], -1)
