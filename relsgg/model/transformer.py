"""Relation transformer over the sampled pairs.

``n_self`` self-attention layers let pairs inform each other, then
``n_cross`` cross-attention layers read an extended memory made of the scene
patch tokens and the four box-corner tokens of every pair. The corner tokens
give the queries positions to attend to directly, so the backbone does not
have to learn spatial routing on its own.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from.geometry import ScenePosEnc


class CrossAttentionLayer(nn.Module):
    """Pre-norm decoder layer: self-attention, cross-attention, feed-forward."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True, bias=True)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True, bias=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model), nn.Dropout(dropout))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.norm1(tgt)
        sa_out, _ = self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask,
                                   need_weights=False)
        tgt = tgt + sa_out
        x = self.norm2(tgt)
        ca_out, _ = self.cross_attn(x, memory, memory,
                                    key_padding_mask=memory_key_padding_mask,
                                    need_weights=False)
        tgt = tgt + ca_out
        return tgt + self.ffn(self.norm3(tgt))


class RelationTransformer(nn.Module):
    def __init__(self, d_model: int = 512, backbone_dim: int = 768,
                 n_self: int = 2, n_cross: int = 2, n_heads: int = 8,
                 ffn_ratio: float = 2.0, dropout: float = 0.2):
        super().__init__()
        ffn_dim = int(d_model * ffn_ratio)
        self.scene_proj = nn.Linear(backbone_dim, d_model)
        nn.init.xavier_uniform_(self.scene_proj.weight)
        nn.init.zeros_(self.scene_proj.bias)
        self.scene_pe = ScenePosEnc(d_model)

        self.self_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(n_self)])
        # All cross layers but the last are stock decoder layers; the last one
        # is the same computation with explicit sub-modules.
        self.cross_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
                                       dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True)
            for _ in range(max(n_cross - 1, 0))])
        self.last_cross = CrossAttentionLayer(d_model=d_model, nhead=n_heads,
                                              dim_feedforward=ffn_dim, dropout=dropout)

    def forward(self, pair_feat: torch.Tensor, scene_feat: torch.Tensor,
                box_tokens: Optional[torch.Tensor] = None,
                pair_padding_mask: Optional[torch.Tensor] = None,
                box_token_drop: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            pair_feat:         ``[B, K, d_model]`` sampled pair representations.
            scene_feat:        ``[B, h, w, backbone_dim]`` patch tokens.
            box_tokens:        ``[B, K, 4, d_model]`` corner tokens per pair.
            pair_padding_mask: ``[B, K]`` True where the slot is padding.
            box_token_drop:    ``[B]`` True hides that image's box tokens
                               (training-time modality dropout).
        """
        B, h, w, _ = scene_feat.shape
        n_scene = h * w
        scene = self.scene_proj(scene_feat.reshape(B, n_scene, -1))
        scene = scene + self.scene_pe(h, w, scene.device, scene.dtype)

        # Box tokens of padding slots are masked out of the memory, otherwise
        # a prediction would depend on how many boxes the batch was padded to.
        memory_key_padding_mask = None
        if box_tokens is not None:
            K, T = box_tokens.shape[1], box_tokens.shape[2]
            memory = torch.cat([scene, box_tokens.reshape(B, K * T, -1)], dim=1)
            if pair_padding_mask is not None or box_token_drop is not None:
                scene_keep = torch.zeros(B, n_scene, dtype=torch.bool, device=scene.device)
                bt_mask = (pair_padding_mask.repeat_interleave(T, dim=1)
                           if pair_padding_mask is not None
                           else torch.zeros(B, K * T, dtype=torch.bool, device=scene.device))
                if box_token_drop is not None:
                    bt_mask = bt_mask | box_token_drop.unsqueeze(1)
                memory_key_padding_mask = torch.cat([scene_keep, bt_mask], dim=1)
        else:
            memory = scene

        x = pair_feat
        for layer in self.self_layers:
            x = layer(x, src_key_padding_mask=pair_padding_mask)
        for layer in self.cross_layers:
            x = layer(tgt=x, memory=memory, tgt_key_padding_mask=pair_padding_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return self.last_cross(x, memory, tgt_key_padding_mask=pair_padding_mask,
                               memory_key_padding_mask=memory_key_padding_mask)
