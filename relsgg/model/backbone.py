"""DINOv3 backbone and the relation interaction block."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import Normalize

from.geometry import ScenePosEnc

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]
# Hidden states fused into the scene map, relative to the last block.
LAYER_OFFSETS = [-6, -3, -1]


class Backbone(nn.Module):
    """DINOv3 ViT, fully fine-tuned, read at three depths.

    The three tapped hidden states are LayerNorm-ed and combined with a
    softmax over three learned scalars, so their share of the scene map does
    not depend on the growth of activation norms with depth.

    Args:
        model_name: Hugging Face id or local directory of the DINOv3 tower.
        pretrained: load the pretrained weights (training); ``False`` builds
                    the architecture from ``config.json`` only, for released
                    checkpoints that carry every weight themselves.
    """

    def __init__(self, model_name: str, patch_size: int = 16, pretrained: bool = True,
                 layer_offsets: List[int] = LAYER_OFFSETS):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        self.patch_size = patch_size
        self.layer_offsets = list(layer_offsets)
        if pretrained:
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(model_name))
        # The taps read hidden states, so the final norm and the mask token
        # never receive a gradient; freeze them so the audit does not report
        # them as dead.
        for attr in ("norm", "layernorm"):
            mod = getattr(self.model, attr, None)
            if mod is not None:
                mod.requires_grad_(False)
        emb = getattr(self.model, "embeddings", None)
        mask_token = getattr(emb, "mask_token", None) if emb is not None else None
        if mask_token is not None:
            mask_token.requires_grad_(False)

        self.d_model: int = self.model.config.hidden_size
        self._normalize = Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)
        self.layer_weights = nn.Parameter(torch.zeros(len(self.layer_offsets)))

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Float images in [0, 1] -> ImageNet-normalised."""
        return self._normalize(images)

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """Normalised ``[B, 3, H, W]`` -> fused patch features ``[B, H/p, W/p, d]``."""
        B, _, H, W = images.shape
        h, w = H // self.patch_size, W // self.patch_size
        hidden_states = self.model(pixel_values=images, output_hidden_states=True).hidden_states
        total = len(hidden_states)
        indices = [total + off if off < 0 else off for off in self.layer_offsets]
        weights = F.softmax(self.layer_weights, dim=0)
        n_patch = h * w
        # Patch tokens are the last h*w tokens (CLS and register tokens first).
        taps = [F.layer_norm(hidden_states[i][:, -n_patch:,:], (self.d_model,))
                for i in indices]
        fused = sum(weights[i] * t for i, t in enumerate(taps))
        return fused.reshape(B, h, w, self.d_model)


class RelationInteractionBlock(nn.Module):
    """Refinement of the pair queries after the relation transformer.

    Stage 1 (``n_dep`` self-attention layers): pairs attend to each other, so
    co-occurring relations can support or suppress one another.
    Stage 2 (``n_gnd`` joint layers): the queries and the scene tokens form one
    sequence passed through a self-attention layer, and the query positions
    are read back; queries and scene update each other.
    """

    def __init__(self, d_model: int, scene_dim: Optional[int] = None, n_dep: int = 2,
                 n_gnd: int = 1, n_heads: int = 8, ffn_ratio: float = 2.0,
                 dropout: float = 0.2) -> None:
        super().__init__()
        ffn_dim = int(d_model * ffn_ratio)
        if scene_dim is not None and scene_dim != d_model:
            self.scene_proj: nn.Module = nn.Linear(scene_dim, d_model, bias=False)
            nn.init.xavier_uniform_(self.scene_proj.weight)
        else:
            self.scene_proj = nn.Identity()
        self.scene_pe = ScenePosEnc(d_model)

        def _layer():
            return nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)

        self.dep_layers = nn.ModuleList([_layer() for _ in range(n_dep)])
        self.gnd_layers = nn.ModuleList([_layer() for _ in range(n_gnd)])

    def forward(self, queries: torch.Tensor, image_tokens: torch.Tensor,
                query_padding_mask: Optional[torch.Tensor] = None,
                grid_hw: Optional[tuple] = None) -> torch.Tensor:
        x = queries
        mem = self.scene_proj(image_tokens)
        h, w = grid_hw
        mem = mem + self.scene_pe(h, w, mem.device, mem.dtype)
        for layer in self.dep_layers:
            x = layer(x, src_key_padding_mask=query_padding_mask)
        n_q = x.shape[1]
        for layer in self.gnd_layers:
            joint = torch.cat([x, mem], dim=1)
            joint_mask = None
            if query_padding_mask is not None:
                scene_no_pad = torch.zeros(x.shape[0], mem.shape[1], dtype=torch.bool,
                                           device=x.device)
                joint_mask = torch.cat([query_padding_mask, scene_no_pad], dim=1)
            x = layer(joint, src_key_padding_mask=joint_mask)[:,:n_q]
        return x
