"""PredicateTextStudent — a small, antonym-aware text encoder distilled from
dino.txt, specialised to predicate/relation strings.

Open-vocabulary by construction
--------------------------------
The whole point of the head is runtime reparameterization to *arbitrary* new
predicate strings. So the student keeps the **full CLIP-BPE vocabulary**
(49,408 tokens) — any subword tokenises to a real row, never a dead <unk>.
To stay small it uses a **factorised** embedding: a compact ``d_tok`` table
projected up to ``dim`` (ALBERT-style), and the table is **initialised from
CLIP's own token embeddings** so subwords the predicate corpus never contained
still start with meaningful semantics.  ~9M params, ~55x smaller than dino.txt.

(A legacy compact-vocab mode — ``clip_ids`` given — is retained for ablation;
it drops out-of-corpus subwords to <unk> and is not the default.)

Bidirectional pre-norm transformer + masked mean pooling (predicate phrases are
short and order-light).  Final linear → ``out_dim``, L2-normalised — same
contract as ``_DinoTxtEncoder.encode`` so it drops into
``VocabHead.set_vocabulary_matrix`` and ``text_space_diag``'s registry unchanged.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_VOCAB_SIZE = 49408
CLIP_TOKENIZER_ID = "openai/clip-vit-base-patch32"
#: files CLIPTokenizer.from_pretrained needs from a local directory — a few MB,
#: versus the 1.2 GB the HF cache holds for that repo (the rest is CLIP's
#: image tower, which the student never touches). Which subset save_pretrained
#: actually writes is transformers-version dependent: current versions emit the
#: merged tokenizer.json, older ones the vocab.json + merges.txt pair. Accept
#: either, and copy whatever exists.
CLIP_TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "merges.txt",
                        "tokenizer_config.json", "special_tokens_map.json")
#: a directory is a usable tokenizer if it satisfies one of these layouts
CLIP_TOKENIZER_LAYOUTS = (("tokenizer.json",), ("vocab.json", "merges.txt"))


def _colocated_tokenizer(path: str) -> Optional[str]:
    """A student checkpoint may ship its tokenizer alongside it, which is what
    makes an embedded deploy bundle work with no network and no HF cache."""
    d = os.path.dirname(os.path.abspath(path))
    ok = any(all(os.path.exists(os.path.join(d, f)) for f in layout)
             for layout in CLIP_TOKENIZER_LAYOUTS)
    return d if ok else None


class _Block(nn.Module):
    """Bidirectional pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim)
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class PredicateTextStudent(nn.Module):
    """Compact predicate-string text encoder.

    Args:
        vocab_size:  token table size. Full mode = 49408 (CLIP BPE); compact
                     mode passes the reduced size from token_vocab.json.
        clip_ids:    None  → full-vocab mode (raw CLIP ids, no OOV).
                     list  → legacy compact mode (compact_id-2 -> CLIP id).
        d_tok:       factorised embedding width (< dim ⇒ table [V,d_tok] + a
                     Linear(d_tok,dim); None/>=dim ⇒ plain [V,dim] table).
        token_init:  optional [vocab_size, d_tok-or-dim] tensor to initialise
                     the embedding table (e.g. PCA of CLIP token embeddings).
    """

    PAD = 0  # manual pad id; pad positions are always masked out explicitly

    def __init__(
        self,
        vocab_size: int = CLIP_VOCAB_SIZE,
        clip_ids: Optional[List[int]] = None,
        d_tok: Optional[int] = 128,
        dim: int = 256,
        depth: int = 4,
        heads: int = 4,
        ffn_dim: int = 1024,
        out_dim: int = 512,
        max_len: int = 32,
        token_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.full = clip_ids is None
        emb_dim = d_tok if (d_tok and d_tok < dim) else dim
        self.cfg = dict(vocab_size=vocab_size, dim=dim, depth=depth, heads=heads,
                        ffn_dim=ffn_dim, out_dim=out_dim, max_len=max_len,
                        d_tok=(emb_dim if emb_dim < dim else None), full=self.full)
        self.max_len = max_len
        self.out_dim = out_dim

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.tok_proj = nn.Linear(emb_dim, dim, bias=False) if emb_dim < dim \
            else nn.Identity()
        self.positional = nn.Parameter(torch.zeros(max_len, dim))
        self.blocks = nn.ModuleList(
            [_Block(dim, heads, ffn_dim) for _ in range(depth)]
        )
        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim, bias=False)

        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional, std=0.02)
        if token_init is not None:
            assert token_init.shape == self.token_embedding.weight.shape, (
                f"token_init {tuple(token_init.shape)} != "
                f"{tuple(self.token_embedding.weight.shape)}")
            with torch.no_grad():
                self.token_embedding.weight.copy_(token_init.float())

        # tokenizer mapping (compact mode only); lazy tokenizer
        self._clip_ids = list(clip_ids) if clip_ids is not None else None
        self._compact_of_clip = (
            {c: i + 2 for i, c in enumerate(self._clip_ids)}
            if self._clip_ids is not None else None
        )
        self._tokenizer = None
        #: local dir holding vocab.json/merges.txt; None -> download by id
        self.tokenizer_src: Optional[str] = None

    # ------------------------------------------------------------------
    # Forward / encode
    # ------------------------------------------------------------------

    def forward(self, ids: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ids: [N, L] token ids. Returns [N, out_dim] L2-normed."""
        if pad_mask is None:
            pad_mask = ids == self.PAD
        L = ids.shape[1]
        x = self.tok_proj(self.token_embedding(ids)) + self.positional[:L]
        for blk in self.blocks:
            x = blk(x, key_padding_mask=pad_mask)
        x = self.ln_final(x)
        keep = (~pad_mask).float().unsqueeze(-1)                 # [N, L, 1]
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)  # [N, dim]
        return F.normalize(self.head(pooled), dim=-1)

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _ensure_tokenizer(self):
        if self._tokenizer is None:
            from transformers import CLIPTokenizer
            self._tokenizer = CLIPTokenizer.from_pretrained(
                self.tokenizer_src or CLIP_TOKENIZER_ID)
        return self._tokenizer

    def tokenize(self, texts: List[str], device=None):
        """Templated strings → (ids [N,max_len], pad_mask [N,max_len] bool).

        Full mode uses raw CLIP ids (no OOV). Compact mode maps to the reduced
        table, unseen subwords → <unk> (=1)."""
        tok = self._ensure_tokenizer()
        enc = tok(texts, truncation=True, max_length=self.max_len)["input_ids"]
        N = len(texts)
        ids = torch.full((N, self.max_len), self.PAD, dtype=torch.long)
        pad = torch.ones((N, self.max_len), dtype=torch.bool)
        for r, seq in enumerate(enc):
            seq = seq[: self.max_len]
            for c, cid in enumerate(seq):
                ids[r, c] = int(cid) if self.full \
                    else self._compact_of_clip.get(int(cid), 1)
                pad[r, c] = False
        if device is not None:
            ids, pad = ids.to(device), pad.to(device)
        return ids, pad

    @torch.no_grad()
    def encode_texts(self, texts: List[str], device=None, batch: int = 1024
                     ) -> torch.Tensor:
        """Convenience: strings → [N, out_dim] normalised embeddings."""
        device = device or next(self.parameters()).device
        outs = []
        for i in range(0, len(texts), batch):
            ids, pad = self.tokenize(texts[i:i + batch], device=device)
            outs.append(self(ids, pad))
        return torch.cat(outs) if outs else torch.zeros(0, self.out_dim, device=device)

    # ------------------------------------------------------------------
    # (De)serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        torch.save({"cfg": self.cfg, "clip_ids": self._clip_ids,
                    "state_dict": self.state_dict()}, path)

    @classmethod
    def from_checkpoint(cls, path: str, device=None) -> "PredicateTextStudent":
        ck = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = dict(ck["cfg"])
        cfg.pop("full", None)                       # derived from clip_ids
        model = cls(clip_ids=ck["clip_ids"], **cfg)
        model.load_state_dict(ck["state_dict"])
        # fp16-stored students (deploy bundles) upcast to fp32 for compute
        model.float()
        model.tokenizer_src = _colocated_tokenizer(path)
        if device is not None:
            model.to(device)
        return model.eval()

    @classmethod
    def full_vocab(cls, token_init: Optional[torch.Tensor] = None, **kw
                   ) -> "PredicateTextStudent":
        """Construct in full-CLIP-vocab mode (the default, OOV-free)."""
        return cls(vocab_size=CLIP_VOCAB_SIZE, clip_ids=None,
                   token_init=token_init, **kw)

    @classmethod
    def from_token_vocab(cls, vocab_path: str, **kw) -> "PredicateTextStudent":
        """Legacy compact-vocab construction (ablation only)."""
        v = json.load(open(vocab_path))
        return cls(vocab_size=v["size"], clip_ids=v["clip_ids"], **kw)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


#: Name of the precomputed bank that stands in for a missing ``student.pt``:
#: a checkpoint's own ``vocab_head.W`` exported as ``{predicate: embedding}``
#: next to where the student would live; see ``_bank_lookup``.
VOCAB_BANK_NAME = "pred_embeds_union.npz"

_BANK_CACHE: dict = {}


def _bank_lookup(texts: List[str], bank_path: str,
                 templates: Optional[List[str]]) -> torch.Tensor:
    """Resolve embeddings by name from a precomputed template-ensembled bank.

    Only valid when the bank's ``templates`` match the requested ensemble --
    the rows ARE one ensemble's output and cannot be re-ensembled. Any
    unresolved name is fatal: a silently partial vocabulary produces numbers
    that look fine and mean nothing.
    """
    if bank_path not in _BANK_CACHE:
        import numpy as np
        z = np.load(bank_path, allow_pickle=True)
        _BANK_CACHE[bank_path] = (
            {str(p): i for i, p in enumerate(z["predicates"])},
            torch.from_numpy(z["embeddings"]).float(),
            [str(t) for t in z["templates"]],
        )
    idx, W, bank_templates = _BANK_CACHE[bank_path]

    want = list(templates) if templates else None
    if want is not None and want != bank_templates:
        raise RuntimeError(
            f"{bank_path} was ensembled over {bank_templates}, but this call "
            f"asked for {want}. The bank stores ensembled rows and cannot be "
            "re-ensembled -- recover the real student checkpoint instead.")

    missing = [t for t in texts if t not in idx]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(texts)} strings are absent from "
            f"{bank_path} (e.g. {missing[:8]}). The bank covers only the "
            "vocabulary the checkpoint was trained over; encoding genuinely "
            "new strings needs the student checkpoint.")
    return W[torch.tensor([idx[t] for t in texts], dtype=torch.long)].clone()


@torch.no_grad()
def resolve_student_path(path: str, near: Optional[str] = None) -> str:
    """Locate the text student a checkpoint names.

    Order: ``path`` itself; ``text_student.pt`` next to ``near`` (the layout of
    every released model repository, where ``model.pth`` and the student sit
    side by side); an ``hf://<repo_id>/<filename>`` reference fetched with
    ``huggingface_hub``. Anything else is returned unchanged so the caller's
    error message names the original path.
    """
    if os.path.exists(path):
        return path
    if near:
        sibling = os.path.join(os.path.dirname(os.path.abspath(near)), "text_student.pt")
        if os.path.exists(sibling):
            return sibling
    if path.startswith("hf://"):
        repo_id, _, filename = path[len("hf://"):].partition("/")
        owner, _, rest = filename.partition("/")
        # hf://<owner>/<name>/<file>: the repo id has two segments
        repo_id, filename = f"{repo_id}/{owner}", rest or "text_student.pt"
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id, filename)
    return path


def encode_texts_student(
    texts: List[str],
    ckpt_path: str,
    templates: Optional[List[str]] = None,
    device=None,
    batch: int = 1024,
) -> torch.Tensor:
    """Encode arbitrary strings with a frozen distilled student checkpoint.

    Mirror of relsgg.vocab.encode_texts_dinotxt: loads the student lazily,
    optionally averages a template ensemble ({p} slots), re-normalizes.
    Returns [len(texts), out_dim] L2-normalized float32 on CPU.

    If ``ckpt_path`` is absent but a sibling ``pred_embeds_union.npz`` exists,
    embeddings are looked up there instead. That bank is the checkpoint's own
    frozen ``vocab_head.W``, i.e. this student's recorded output for the
    training vocabulary -- exact, not an approximation, but limited to strings
    it already contains.
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(ckpt_path):
        bank = os.path.join(os.path.dirname(ckpt_path), VOCAB_BANK_NAME)
        if os.path.exists(bank):
            print(f"[text_student] {ckpt_path} absent -- resolving "
                  f"{len(texts)} strings from {bank} (checkpoint vocab_head.W)")
            return _bank_lookup(texts, bank, templates)
    model = PredicateTextStudent.from_checkpoint(ckpt_path, device)

    def _enc(strs: List[str]) -> torch.Tensor:
        return model.encode_texts(strs, device=device, batch=batch).cpu()

    if templates:
        emb = sum(_enc([t.format(p=p) for p in texts]) for t in templates)
        emb = F.normalize(emb, dim=-1)
    else:
        emb = _enc(texts)
    return emb.float()
