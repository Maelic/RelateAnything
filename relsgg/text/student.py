"""PredicateTextStudent: the text encoder that embeds predicate strings.

A 6-block bidirectional transformer (about 9M parameters) distilled from the
dino.txt text tower on relation phrases, with two properties the head needs:
antonyms are far apart (``above`` / ``below``), and any string can be encoded
because the token table is the full CLIP byte-pair vocabulary (49,408 rows,
factorised through a 128-wide table). Output: L2-normalised ``[N, out_dim]``.

Every released model ships its student as ``text_student.pt`` next to
``model.pth``, with the tokenizer files beside it, so encoding a vocabulary
needs no network access.
"""
from __future__ import annotations

import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_VOCAB_SIZE = 49408
CLIP_TOKENIZER_ID = "openai/clip-vit-base-patch32"
#: Files ``CLIPTokenizer.from_pretrained`` needs from a directory. Which subset
#: ``save_pretrained`` writes depends on the transformers version, so either
#: layout is accepted.
CLIP_TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "merges.txt",
                        "tokenizer_config.json", "special_tokens_map.json")
CLIP_TOKENIZER_LAYOUTS = (("tokenizer.json",), ("vocab.json", "merges.txt"))


def _colocated_tokenizer(path: str) -> Optional[str]:
    """The directory of ``path`` when it holds the tokenizer files."""
    d = os.path.dirname(os.path.abspath(path))
    ok = any(all(os.path.exists(os.path.join(d, f)) for f in layout)
             for layout in CLIP_TOKENIZER_LAYOUTS)
    return d if ok else None


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ffn(self.norm2(x))


class PredicateTextStudent(nn.Module):
    """
    Args:
        vocab_size: token table size (the CLIP BPE vocabulary).
        d_tok:      width of the factorised token table (``None`` for a plain
                    ``[vocab_size, dim]`` table).
        token_init: optional initial token table, e.g. a projection of CLIP's.
    """

    PAD = 0

    def __init__(self, vocab_size: int = CLIP_VOCAB_SIZE, d_tok: Optional[int] = 128,
                 dim: int = 256, depth: int = 6, heads: int = 4, ffn_dim: int = 1024,
                 out_dim: int = 512, max_len: int = 32,
                 token_init: Optional[torch.Tensor] = None):
        super().__init__()
        emb_dim = d_tok if (d_tok and d_tok < dim) else dim
        self.cfg = dict(vocab_size=vocab_size, dim=dim, depth=depth, heads=heads,
                        ffn_dim=ffn_dim, out_dim=out_dim, max_len=max_len,
                        d_tok=(emb_dim if emb_dim < dim else None))
        self.max_len = max_len
        self.out_dim = out_dim
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.tok_proj = nn.Linear(emb_dim, dim, bias=False) if emb_dim < dim else nn.Identity()
        self.positional = nn.Parameter(torch.zeros(max_len, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads, ffn_dim) for _ in range(depth)])
        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim, bias=False)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional, std=0.02)
        if token_init is not None:
            assert token_init.shape == self.token_embedding.weight.shape
            with torch.no_grad():
                self.token_embedding.weight.copy_(token_init.float())
        self._tokenizer = None
        self.tokenizer_src: Optional[str] = None   # local directory, else the hub id

    def forward(self, ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[N, L]`` token ids -> ``[N, out_dim]`` unit vectors (masked mean pooling)."""
        if pad_mask is None:
            pad_mask = ids == self.PAD
        L = ids.shape[1]
        x = self.tok_proj(self.token_embedding(ids)) + self.positional[:L]
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        x = self.ln_final(x)
        keep = (~pad_mask).float().unsqueeze(-1)
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return F.normalize(self.head(pooled), dim=-1)

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(self.tokenizer_src or CLIP_TOKENIZER_ID)
        return self._tokenizer

    def tokenize(self, texts: List[str], device=None):
        """Strings -> (ids ``[N, max_len]``, pad_mask ``[N, max_len]``)."""
        tok = self._ensure_tokenizer()
        enc = tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
        N = len(texts)
        ids = torch.full((N, self.max_len), self.PAD, dtype=torch.long)
        pad = torch.ones((N, self.max_len), dtype=torch.bool)
        for r, seq in enumerate(enc):
            seq = seq[: self.max_len]
            ids[r,: len(seq)] = torch.tensor(seq, dtype=torch.long)
            pad[r,: len(seq)] = False
        if device is not None:
            ids, pad = ids.to(device), pad.to(device)
        return ids, pad

    @torch.no_grad()
    def encode_texts(self, texts: List[str], device=None, batch: int = 1024) -> torch.Tensor:
        device = device or next(self.parameters()).device
        outs = []
        for i in range(0, len(texts), batch):
            ids, pad = self.tokenize(texts[i:i + batch], device=device)
            outs.append(self(ids, pad))
        return torch.cat(outs) if outs else torch.zeros(0, self.out_dim, device=device)

    def save(self, path: str) -> None:
        torch.save({"cfg": self.cfg, "state_dict": self.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path: str, device=None) -> "PredicateTextStudent":
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = {k: v for k, v in ck["cfg"].items() if k != "full"}
        model = cls(**cfg)
        model.load_state_dict(ck["state_dict"])
        model.float()
        model.tokenizer_src = _colocated_tokenizer(path)
        if device is not None:
            model.to(device)
        return model.eval()

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def resolve_student_path(path: str, near: Optional[str] = None) -> str:
    """The student a checkpoint names: ``path`` itself, else ``text_student.pt``
    next to ``near`` (the released layout), else an ``hf://owner/repo/file``
    reference fetched from the Hub. Anything else is returned unchanged."""
    if os.path.exists(path):
        return path
    if near:
        sibling = os.path.join(os.path.dirname(os.path.abspath(near)), "text_student.pt")
        if os.path.exists(sibling):
            return sibling
    if path.startswith("hf://"):
        owner, _, rest = path[len("hf://"):].partition("/")
        name, _, filename = rest.partition("/")
        from huggingface_hub import hf_hub_download
        return hf_hub_download(f"{owner}/{name}", filename or "text_student.pt")
    return path


def encode_texts_student(texts: List[str], ckpt_path: str,
                         templates: Optional[List[str]] = None, device=None,
                         batch: int = 1024) -> torch.Tensor:
    """Encode strings with a student checkpoint, averaging over ``templates``
    (each with a ``{p}`` slot) and re-normalising. Returns float32 on CPU."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PredicateTextStudent.from_checkpoint(ckpt_path, device)

    def _enc(strs: List[str]) -> torch.Tensor:
        return model.encode_texts(strs, device=device, batch=batch).cpu()

    if templates:
        emb = F.normalize(sum(_enc([t.format(p=p) for p in texts]) for t in templates), dim=-1)
    else:
        emb = _enc(texts)
    return emb.float()
