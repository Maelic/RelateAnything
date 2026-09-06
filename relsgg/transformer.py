"""Relation Transformer.

Processes K sampled pair representations through:
  1. N_self layers of self-attention among pairs  (pairs inform each other).
  2. N_cross layers of cross-attention to an extended memory that contains
     both scene patch tokens AND bounding-box corner tokens.

The box corner tokens (4 per pair: sub_TL, sub_BR, obj_TL, obj_BR) are
produced by BoxPromptEncoder and appended to the scene patch sequence before
cross-attention. They act as learnable spatial attractors: the pair queries
can attend directly to the spatially relevant positions, so the backbone does
not need to learn spatial routing implicitly.

This is the v1 architecture — predicate queries carry no explicit entity-type
conditioning. Visual-only features drive predicate recognition.

Uses standard PyTorch TransformerEncoderLayer / TransformerDecoderLayer with
batch_first=True, pre-norm (norm_first=True), and no dropout by default.

Phase 2 (PCSG Attention Anchoring):
  The last cross-attention layer is replaced by ``ExposedCrossAttentionLayer``,
  which calls nn.MultiheadAttention with ``need_weights=True`` and returns the
  average cross-attention map over heads alongside the layer output.
  ``RelationTransformer.forward`` returns a ``(output, attn_weights)`` tuple
  when ``return_attn=True`` is passed (default: False, zero overhead).
  Only the *scene* portion of the memory (first ``h*w`` tokens) is exposed;
  box-corner token attention is excluded to keep the geometric prior clean.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Optional, Tuple


@torch.no_grad()
def depth_scaled_residual_init_(stacks: Iterable[nn.Module]) -> int:
    """GPT-2-style depth-scaled init: shrink every residual-branch output
    projection by 1/sqrt(N), where N is the total number of residual additions
    along the query path. Keeps the residual stream's variance ~constant with
    depth at init instead of growing linearly, which smooths the first
    optimizer steps of the from-scratch stack.

    Covers the three layer types used in this codebase; call AFTER default
    construction, BEFORE any checkpoint load. Returns N (for logging).
    """
    projs: list[torch.Tensor] = []
    for stack in stacks:
        if stack is None:
            continue
        for m in stack.modules():
            if isinstance(m, nn.TransformerEncoderLayer):
                projs += [m.self_attn.out_proj.weight, m.linear2.weight]
            elif isinstance(m, nn.TransformerDecoderLayer):
                projs += [m.self_attn.out_proj.weight,
                          m.multihead_attn.out_proj.weight, m.linear2.weight]
            elif isinstance(m, ExposedCrossAttentionLayer):
                projs += [m.self_attn.out_proj.weight,
                          m.cross_attn.out_proj.weight, m.ffn[3].weight]
    scale = len(projs) ** -0.5
    for w in projs:
        w.mul_(scale)
    return len(projs)


# ---------------------------------------------------------------------------
# Phase-2 helper: drop-in replacement for the last nn.TransformerDecoderLayer
# that additionally exposes per-pair cross-attention weights over scene tokens.
# ---------------------------------------------------------------------------

class ExposedCrossAttentionLayer(nn.Module):
    """Cross-attention layer that optionally returns averaged attention weights.

    Matches the interface of ``nn.TransformerDecoderLayer`` (pre-norm, batch_first)
    but exposes ``attn_weights [B, K, n_scene]`` averaged over heads when
    ``need_weights=True`` is passed to ``forward``.

    Self-attention sub-layer (pairs attending to themselves) uses a standard
    ``nn.MultiheadAttention`` and is not exposed — only the cross-attention
    weights to scene tokens are returned.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Diagnostics (visualization only, never set during training — the
        # deformable.capture pattern): per_head_attn=True returns weights
        # [B, nhead, K, n_scene] instead of head-averaged; capture_memory=True
        # stashes the memory tensor + FULL (scene+box-token) attention of the
        # last call so value-norm-weighted maps can be computed offline.
        self.per_head_attn = False
        self.capture_memory = False
        self.last_memory: Optional[torch.Tensor] = None
        self.last_full_attn: Optional[torch.Tensor] = None
        # Self-attention sub-layer (tgt × tgt)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True, bias=True,
        )
        # Cross-attention sub-layer (tgt × memory)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead,
            dropout=dropout, batch_first=True, bias=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        n_scene: int,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            tgt:                  [B, K, d_model]
            memory:               [B, h*w + K*T, d_model] extended memory
            n_scene:              number of scene patch tokens (h*w); used to
                                  slice out scene-only attention weights
            tgt_key_padding_mask: [B, K] True = invalid (padding) slot
            need_weights:         if True, return mean attention map [B, K, n_scene]
            memory_key_padding_mask: [B, h*w + K*T] True = ignore this memory
                                  token. Hides the box tokens of padding pair
                                  slots (see RelationTransformer.forward).
        Returns:
            (output [B, K, d_model], attn_weights [B, K, n_scene] or None)
        """
        # Pre-norm self-attention
        x = self.norm1(tgt)
        sa_out, _ = self.self_attn(
            x, x, x, key_padding_mask=tgt_key_padding_mask, need_weights=False,
        )
        tgt = tgt + sa_out

        # Pre-norm cross-attention
        x = self.norm2(tgt)
        ca_out, attn_w = self.cross_attn(
            x, memory, memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=need_weights,
            # default head-average → [B, K, mem_len]; per_head → [B, H, K, mem_len]
            average_attn_weights=not self.per_head_attn,
        )
        tgt = tgt + ca_out

        # Pre-norm FFN
        tgt = tgt + self.ffn(self.norm3(tgt))

        if self.capture_memory:
            self.last_memory = memory.detach()
            self.last_full_attn = attn_w.detach() if attn_w is not None else None
        scene_attn = attn_w[..., :n_scene] if need_weights else None
        return tgt, scene_attn

    @classmethod
    def from_decoder_layer(cls, layer: nn.TransformerDecoderLayer) -> "ExposedCrossAttentionLayer":
        """Copy weights from an existing TransformerDecoderLayer into this module.

        ``TransformerDecoderLayer`` (pre-norm, batch_first) stores:
          self_attn, multihead_attn, linear1, linear2, norm1, norm2, norm3,
          dropout1, dropout2 (activation embedded).
        """
        d_model = layer.self_attn.embed_dim
        nhead = layer.self_attn.num_heads
        dim_feedforward = layer.linear1.out_features
        dropout = layer.dropout1.p if hasattr(layer, "dropout1") else 0.0

        new = cls(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)

        # Copy self-attention weights
        new.self_attn.load_state_dict(layer.self_attn.state_dict())
        # Copy cross-attention weights
        new.cross_attn.load_state_dict(layer.multihead_attn.state_dict())
        # Copy FFN weights (linear1 → ffn[0], linear2 → ffn[3])
        new.ffn[0].load_state_dict(layer.linear1.state_dict())
        new.ffn[3].load_state_dict(layer.linear2.state_dict())
        # Copy norms
        new.norm1.load_state_dict(layer.norm1.state_dict())
        new.norm2.load_state_dict(layer.norm2.state_dict())
        new.norm3.load_state_dict(layer.norm3.state_dict())

        return new


class RelationTransformer(nn.Module):
    """Self-attention + cross-attention over sampled relation pairs.

    Args:
        d_model:      Pair feature dimension (must match pair_proj output in
                      RelSGG, default 512).
        backbone_dim: Backbone patch feature dimension (768 for ViT-B).
        n_self:       Number of self-attention layers among pairs.
        n_cross:      Number of cross-attention layers to extended memory.
        n_heads:      Number of attention heads.
        ffn_ratio:    FFN hidden dimension multiplier.
        dropout:      Attention dropout (0 for deployment).
    """

    def __init__(
        self,
        d_model: int = 512,
        backbone_dim: int = 768,
        n_self: int = 2,
        n_cross: int = 2,
        n_heads: int = 8,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        scene_pe: bool = False,
    ):
        super().__init__()
        ffn_dim = int(d_model * ffn_ratio)

        # Project backbone patch features to model dimension once
        self.scene_proj = nn.Linear(backbone_dim, d_model)
        nn.init.xavier_uniform_(self.scene_proj.weight)
        nn.init.zeros_(self.scene_proj.bias)

        # Gated absolute PE on the scene keys (see geometry.ScenePosEnc for
        # the full rationale — the memory was positionally unaddressable and
        # attention collapsed onto the box-token shortcut). Zero-init gate =>
        # scene_pe=False AND fresh scene_pe=True both start bit-identical.
        self.scene_pe = None
        if scene_pe:
            from .geometry import ScenePosEnc
            self.scene_pe = ScenePosEnc(d_model)

        # Self-attention among the K pairs
        self.self_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=ffn_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_self)
            ]
        )

        # Cross-attention: pairs (Q) ← extended memory (K / V)
        # Memory = [scene_patch_tokens ; box_corner_tokens]
        # All layers except the last use standard TransformerDecoderLayer.
        # The last layer uses ExposedCrossAttentionLayer to optionally expose
        # cross-attention weights for the Attention Anchoring (AA) loss.
        standard_cross = [
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(max(n_cross - 1, 0))
        ]
        last_cross = ExposedCrossAttentionLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
        )
        self.cross_layers = nn.ModuleList(standard_cross)
        self.last_cross = last_cross  # separated for clarity; logically part of cross stack

    def forward(
        self,
        pair_feat: torch.Tensor,
        scene_feat: torch.Tensor,
        box_tokens: torch.Tensor | None = None,
        pair_padding_mask: torch.Tensor | None = None,
        return_attn: bool = False,
        box_token_drop: torch.Tensor | None = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pair_feat:        [B, K, d_model]  sampled pair representations.
            scene_feat:       [B, h, w, backbone_dim]  backbone patch tokens.
            box_tokens:       [B, K, T, d_model]  box corner tokens from
                              BoxPromptEncoder.encode_pairs(), where T=4.
                              If None, attention is over scene tokens only.
            pair_padding_mask:[B, K] bool — True means the slot is INVALID
                              (padding). Passed as src_key_padding_mask.
            return_attn:      If True, also return scene cross-attention weights
                              [B, K, h*w] from the last cross-attention layer.
                              Used only for the AA loss during training; no
                              overhead when False.
            box_token_drop:   [B] bool — True hides that sample's ENTIRE
                              box-token block from the memory (per-sample
                              modality dropout; train-time only).
        Returns:
            [B, K, d_model] enriched pair representations, or
            ([B, K, d_model], [B, K, h*w]) when return_attn=True.
        """
        B, h, w, _ = scene_feat.shape
        n_scene = h * w

        # Project and flatten scene tokens: [B, h*w, d_model]
        scene = self.scene_proj(scene_feat.reshape(B, n_scene, -1))
        if self.scene_pe is not None:
            scene = scene + self.scene_pe(h, w, scene.device, scene.dtype)

        # Build extended memory: [B, h*w + K*T, d_model]
        #
        # The box tokens of PADDING pair slots must be masked out of this
        # memory. Without it, every valid pair cross-attends to the corner
        # tokens of filler pairs — and filler slots are chosen by a topk over
        # scores that are all exactly finfo.min, i.e. an arbitrary tie-break.
        # The result is a prediction that depends on how many boxes you padded
        # to and on which backend broke the tie (torch and onnxruntime break it
        # differently, which is how this was found). Scene tokens are never
        # padding, so only the box-token block is masked.
        memory_key_padding_mask = None
        if box_tokens is not None:
            K, T = box_tokens.shape[1], box_tokens.shape[2]
            bt_flat = box_tokens.reshape(B, K * T, -1)
            memory = torch.cat([scene, bt_flat], dim=1)  # [B, h*w + K*T, d_model]
            if pair_padding_mask is not None or box_token_drop is not None:
                scene_keep = torch.zeros(B, n_scene, dtype=torch.bool,
                                         device=scene.device)
                bt_mask = (pair_padding_mask.repeat_interleave(T, dim=1)
                           if pair_padding_mask is not None else
                           torch.zeros(B, K * T, dtype=torch.bool,
                                       device=scene.device))       # [B, K*T]
                if box_token_drop is not None:
                    # Per-sample modality dropout: hide the WHOLE box-token
                    # block for dropped samples. Scene tokens stay, so no
                    # memory row is ever fully masked (no NaN softmax).
                    bt_mask = bt_mask | box_token_drop.unsqueeze(1)
                memory_key_padding_mask = torch.cat([scene_keep, bt_mask], dim=1)
        else:
            memory = scene  # [B, h*w, d_model]

        x = pair_feat  # [B, K, d_model]

        # 1. Self-attention among pairs
        for layer in self.self_layers:
            x = layer(x, src_key_padding_mask=pair_padding_mask)

        # 2. Cross-attention — all-but-last layers
        for layer in self.cross_layers:
            x = layer(
                tgt=x,
                memory=memory,
                tgt_key_padding_mask=pair_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )

        # 3. Last cross-attention layer (always ExposedCrossAttentionLayer)
        x, attn_weights = self.last_cross(
            tgt=x,
            memory=memory,
            n_scene=n_scene,
            tgt_key_padding_mask=pair_padding_mask,
            need_weights=return_attn,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        if return_attn:
            return x, attn_weights  # attn_weights: [B, K, h*w]
        return x
