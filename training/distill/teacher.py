"""The dino.txt text tower, the teacher the predicate student is distilled from.

Meta's dino.txt checkpoint (vision head + text encoder) carries a 24-block
causal text transformer aligned with DINOv3's visual features. It is needed
only to build distillation targets (``build_corpus.py --encode-teacher``) and
to compare the student against it (``eval_student.py``); inference and
training of the relation head use the distilled student instead.

The transformer is implemented here so neither the ``dinov3`` package nor
SL-HOI is required. Its shape follows the checkpoint's keys: token embedding
[49408, 1280], positional embedding [77, 1280], 24 pre-norm causal blocks
(width 1280, 20 heads, feed-forward 5120), a final norm and a [2048, 1280]
projection. Tokenisation is CLIP's byte-pair encoding, the same one the
checkpoint was trained with.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalBlock(nn.Module):
    """Single pre-norm causal transformer block matching dino.txt weights."""

    def __init__(self, dim: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(dim)
        # QKV as a single linear, no bias (weight only: [3*dim, dim])
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, dim)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention (causal)
        residual = x
        x = self.attention_norm(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # each [B, N, heads, head_dim]
        q = q.transpose(1, 2)    # [B, heads, N, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.fc2(F.gelu(self.fc1(x)))
        return residual + x


class DinoTxtEncoder(nn.Module):
    """Causal text transformer from the dino.txt checkpoint.

    Architecture: 24-layer pre-norm causal ViT-like transformer,
    dim=1280, 20 heads, FFN=5120, ctx=77, vocab=49408.
    Output: 2048-dim embeddings via a final linear projection.

    Loaded from the combined dino.txt checkpoint
    (``dinov3_vitl16_dinotxt_vision_head_and_text_encoder-*.pth``).
    """

    DIM = 1280
    NUM_HEADS = 20
    FFN_DIM = 5120
    NUM_LAYERS = 24
    CTX_LEN = 77
    VOCAB_SIZE = 49408
    OUT_DIM = 2048

    def __init__(self) -> None:
        super().__init__()
        d = self.DIM
        self.token_embedding = nn.Embedding(self.VOCAB_SIZE, d)
        self.positional_embedding = nn.Parameter(torch.empty(self.CTX_LEN, d))
        self.blocks = nn.ModuleList([
            CausalBlock(d, self.NUM_HEADS, self.FFN_DIM)
            for _ in range(self.NUM_LAYERS)
        ])
        self.ln_final = nn.LayerNorm(d)
        self.linear_projection = nn.Linear(d, self.OUT_DIM, bias=False)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "DinoTxtEncoder":
        """Load weights from the combined dino.txt checkpoint."""
        model = cls().to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)

        sd = {}
        for k, v in ckpt.items():
            if not k.startswith("text_model."):
                continue
            # Strip "text_model.backbone." and "text_model.head." prefixes
            k = k[len("text_model."):]
            if k.startswith("backbone."):
                k = k[len("backbone."):]
                # Remap block submodule names
                if ".attention.qkv." in k:
                    k = k.replace(".attention.qkv.", ".qkv.")
                elif ".attention.proj." in k:
                    k = k.replace(".attention.proj.", ".proj.")
                elif ".feed_forward.fc1." in k:
                    k = k.replace(".feed_forward.fc1.", ".fc1.")
                elif ".feed_forward.fc2." in k:
                    k = k.replace(".feed_forward.fc2.", ".fc2.")
            elif k.startswith("head."):
                k = k[len("head."):]  # "linear_projection.weight"
            sd[k] = v

        model.load_state_dict(sd, strict=True)
        return model

    @torch.inference_mode()
    def encode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Encode tokenised text to 2048-dim normalised embeddings.

        Args:
            token_ids: [V, 77] int64 token IDs (CLIP BPE, padded with 0).
        Returns:
            [V, 2048] float32, L2-normalised.
        """
        x = self.token_embedding(token_ids) + self.positional_embedding  # [V, 77, 1280]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        # EOS pooling: take the output at the EOS position.
        # CLIP convention: EOS token has the highest ID in the sequence,
        # so argmax over token_ids gives its position.
        eos_positions = token_ids.argmax(dim=-1)  # [V]
        x = x[torch.arange(len(eos_positions)), eos_positions]  # [V, 1280]
        x = self.linear_projection(x)             # [V, 2048]
        return F.normalize(x.float(), dim=-1)


@torch.no_grad()
def encode_texts_dinotxt(
    texts: List[str],
    dinotxt_weights: str,
    templates: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    batch: int = 512,
) -> torch.Tensor:
    """Encode arbitrary strings with the frozen dino.txt text tower.

    Standalone (no VocabHead state touched) — used for e.g. object-category
    embeddings for the compositional aux loss. Returns [len(texts), 2048]
    L2-normalized float32 on CPU.
    """
    from transformers import CLIPTokenizer

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DinoTxtEncoder.from_checkpoint(dinotxt_weights, device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def _encode(strs: List[str]) -> torch.Tensor:
        outs = []
        for i in range(0, len(strs), batch):
            ids = tokenizer(strs[i:i + batch], return_tensors="pt",
                            padding="max_length", truncation=True,
                            max_length=DinoTxtEncoder.CTX_LEN)["input_ids"].to(device)
            outs.append(encoder.encode(ids).cpu())
        return torch.cat(outs)

    if templates:
        emb = sum(_encode([t.format(p=p) for p in texts]) for t in templates)
        emb = F.normalize(emb, dim=-1)
    else:
        emb = _encode(texts)
    return emb.float()


