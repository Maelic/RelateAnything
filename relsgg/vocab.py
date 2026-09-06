"""Open-vocabulary predicate head with re-parametrization and InfoNCE loss.

Design principle
----------------
For large-scale pretraining followed by zero-shot transfer (e.g. MegaSG →
PSG / VG150 via ``reparameterize()``), the visual pair features must live in
a *universal* embedding space — one aligned with all of CLIP's semantic
geometry, not just the gravity wells of the training predicates.

Therefore the head has exactly ONE learned transformation on the
classification path: ``proj: Linear(d_model, d_model)`` on the *visual* side.
No ``text_proj`` is used.  The predicate weight matrix W holds raw, normalised
CLIP text embeddings and is never modified by learned parameters.

Consequences:
- InfoNCE pushes ``proj(r)`` toward the exact CLIP text direction of the GT
  predicate.  There is no intermediate learned subspace that could only
  represent the training vocabulary.
- Zero-shot: ``encode_vocabulary(["wears", "has part", ...])`` embeds new
  predicates via CLIP and stores them directly as W.  No part of the network
  needs to have seen those words during training.
- ``reparameterize()`` simply seals W and verifies normalisation.  The
  forward pass is a single matmul + scale — zero LM overhead at runtime.

API:
    head.encode_vocabulary(["above", "behind", "next to", "holding"])
    head.reparameterize()         # seal vocabulary for inference
    logits = head(r)              # [B, K, V] — no text encoder at runtime
    head.encode_vocabulary([...]) # swap vocabulary at any time; re-param again
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Standalone dino.txt text encoder
# ---------------------------------------------------------------------------
# Implements the text transformer from the dino.txt checkpoint without
# requiring the external dinov3 or SL-HOI package.  Architecture inferred
# from checkpoint keys:
#   token_embedding [49408, 1280], positional_embedding [77, 1280],
#   24 causal pre-norm blocks (dim=1280, heads=20, FFN=5120),
#   ln_final [1280], linear_projection [2048, 1280]

class _CausalBlock(nn.Module):
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


class _DinoTxtEncoder(nn.Module):
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
            _CausalBlock(d, self.NUM_HEADS, self.FFN_DIM)
            for _ in range(self.NUM_LAYERS)
        ])
        self.ln_final = nn.LayerNorm(d)
        self.linear_projection = nn.Linear(d, self.OUT_DIM, bias=False)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device) -> "_DinoTxtEncoder":
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
    encoder = _DinoTxtEncoder.from_checkpoint(dinotxt_weights, device).eval()
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    def _encode(strs: List[str]) -> torch.Tensor:
        outs = []
        for i in range(0, len(strs), batch):
            ids = tokenizer(strs[i:i + batch], return_tensors="pt",
                            padding="max_length", truncation=True,
                            max_length=_DinoTxtEncoder.CTX_LEN)["input_ids"].to(device)
            outs.append(encoder.encode(ids).cpu())
        return torch.cat(outs)

    if templates:
        emb = sum(_encode([t.format(p=p) for p in texts]) for t in templates)
        emb = F.normalize(emb, dim=-1)
    else:
        emb = _encode(texts)
    return emb.float()


class VocabHead(nn.Module):
    """Re-parametrizable open-vocabulary predicate scoring head.

    Args:
        d_model:         Visual pair feature dimension.
        text_dim:        Output dimension of the text encoder.  ``None``
                         defaults to ``d_model`` (compatible with
                         CLIP-B/32 at d_model=512).  Set to ``2048`` when
                         using dino.txt text embeddings.
        text_model_name: HuggingFace identifier for the CLIP-based text
                         encoder used by :meth:`encode_vocabulary`.
                         Ignored when calling
                         :meth:`encode_vocabulary_dinotxt`.
        logit_scale_init: Initial value for the learnable logit scale.
        infonce_temp:    Temperature for the InfoNCE contrastive loss.
    """

    def __init__(
        self,
        d_model: int = 512,
        text_dim: Optional[int] = None,
        text_model_name: str = "openai/clip-vit-base-patch32",
        logit_scale_init: float = 1.0 / 0.07,
        infonce_temp: float = 0.07,
        logit_bias_init: float = 0.0,
        proj_layers: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.text_dim = text_dim if text_dim is not None else d_model
        self.text_model_name = text_model_name
        self.infonce_temp = infonce_temp

        # Learned projection: visual d_model → text embedding space.
        # proj_layers=1: single Linear (square + identity-init for CLIP,
        # Xavier for dino.txt). proj_layers>=2: MLP — one matrix is too thin
        # to express relation semantics in a 2048-d text space.
        if proj_layers <= 1:
            self.proj = nn.Linear(d_model, self.text_dim, bias=False)
            if d_model == self.text_dim:
                nn.init.eye_(self.proj.weight)
            else:
                nn.init.xavier_uniform_(self.proj.weight)
        else:
            hidden = max(d_model * 2, self.text_dim // 2)
            layers: list[nn.Module] = []
            in_dim = d_model
            for _ in range(proj_layers - 1):
                layers += [nn.Linear(in_dim, hidden), nn.GELU()]
                in_dim = hidden
            layers += [nn.LayerNorm(in_dim), nn.Linear(in_dim, self.text_dim, bias=False)]
            self.proj = nn.Sequential(*layers)
            for m in self.proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Learnable log-scale (shared CLIP convention)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(logit_scale_init))
        )
        # Learnable additive bias (SigLIP): init ≈ -log(V) so at cold start
        # every class's sigmoid ≈ 1/V — negatives are born suppressed and the
        # gradient budget goes to positives. A constant shift cancels in
        # softmax, so the legacy CE path is unaffected; sigmoid ranking is
        # monotone in cosine, so eval ordering is unaffected too.
        self.logit_bias = nn.Parameter(torch.tensor(float(logit_bias_init)))

        # The predicate embedding matrix — register as buffer so it is saved
        # and moved with .to(device) automatically.
        # W is exposed through the class property below (buffer by default,
        # normalised view of the trainable W_param after make_W_trainable()).
        # register_buffer() would refuse the name because the property exists.
        self._buffers["W"] = torch.empty(0)
        self.pred_names: List[str] = []
        self.is_reparameterized: bool = False

        # Dual-projection spatial head (optional): a spatialness gate on TEXT
        # embeddings routes each predicate between two visual experts.
        # Two generations coexist:
        #   gate_u/gate_b  frozen logistic-probe direction (legacy, and the
        #                  warm-start target for the MLP)
        #   gate_mlp       trainable MLP gate (v34+) — alpha computed LIVE
        #                  from W during training so gradients shape the
        #                  routing; baked into the alpha buffer only at
        #                  reparameterize()/vocab-swap time.
        # The alpha BUFFER must never be read during training when gate_mlp
        # exists: ModelEMA copies buffers verbatim each step, so the EMA
        # shadow's buffer holds the train model's values — the shadow instead
        # recomputes from its own EMA-averaged gate_mlp via current_alpha().
        self.register_buffer("gate_u", torch.empty(0))
        self.register_buffer("gate_b", torch.zeros(()))
        self.register_buffer("alpha", torch.empty(0))
        # Per-predicate relatedness weight (build_beta_mlp); empty = disabled.
        self.register_buffer("beta", torch.empty(0))
        self.beta_mlp = None
        self.gate_mlp: Optional[nn.Module] = None
        # Inference-time constant routing (set_alpha_override): pins every
        # predicate to one expert regardless of the gate. alpha=1 is the
        # single-expert *spatial-only* deployment — the semantic query then
        # multiplies by zero, so vocab_head.proj can be dropped at export.
        self.alpha_override: Optional[float] = None

    def build_gate_mlp(self, hidden: int = 128) -> None:
        """Create the trainable spatialness gate (call before optimizer/DDP)."""
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.text_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.gate_mlp.to(self.logit_scale.device)

    def build_beta_mlp(self, hidden: int = 128) -> None:
        """Per-predicate weight on the pair-existence (relatedness) logit.

        MEASURED MOTIVATION ([[relsgg-relatedness-contact-prior]]): the
        relatedness head is trained on "was this pair annotated at all", which is
        an ANNOTATION-PROPENSITY signal correlated with contact/overlap. Fusing
        it with one global weight is wrong in opposite directions — on
        SpatialSense, removing it costs `on` -0.116 AUC but GAINS `in front of`
        +0.138, `above` +0.128, `to the left of` +0.118. One scalar cannot serve
        both, so the weight becomes per-predicate and is READ OFF THE TEXT
        EMBEDDING exactly as the spatialness gate is — it therefore generalizes
        to vocabularies never seen in training and bakes into reparameterization
        as a per-column constant.

        Initialised so beta ~ 1 (sigmoid(2.0) = 0.88) — i.e. it starts close to
        today's behaviour and has to LEARN to switch the term off, rather than
        starting from a guess about which predicates are contact-like.
        """
        self.beta_mlp = nn.Sequential(
            nn.Linear(self.text_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        for m in self.beta_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.beta_mlp[-1].bias, 2.0)
        self.beta_mlp.to(self.logit_scale.device)

    def current_beta(self) -> Optional[torch.Tensor]:
        """[V] per-predicate relatedness weight, or None when disabled."""
        if getattr(self, "beta_mlp", None) is None:
            return self.beta if self.beta.numel() else None
        if not self.is_reparameterized:
            return torch.sigmoid(self.beta_mlp(self.W).squeeze(-1))
        return self.beta if self.beta.numel() else None

    def warm_start_gate_mlp(self, steps: int = 300, lr: float = 1e-2) -> float:
        """Regress gate_mlp onto the fitted logistic-probe alphas so training
        starts from the probe's routing instead of random. Returns final MSE."""
        assert self.gate_mlp is not None and self.gate_u.numel() and self.W.numel()
        with torch.no_grad():
            target = torch.sigmoid(self.W @ self.gate_u + self.gate_b)
        opt = torch.optim.Adam(self.gate_mlp.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            pred = torch.sigmoid(self.gate_mlp(self.W).squeeze(-1))
            loss = F.mse_loss(pred, target)
            loss.backward()
            opt.step()
        self._update_alpha()  # bake warm-started alpha so EMA buffers match
        return float(loss.detach())

    def current_alpha(self) -> torch.Tensor:
        """Per-predicate routing weight for dual-expert scoring.

        Override path (set_alpha_override): a constant for every predicate —
        the single-expert inference configuration, gate bypassed entirely.
        Trainable path (gate_mlp present, vocabulary not sealed): computed
        live from W — differentiable w.r.t. gate_mlp. Otherwise the baked
        buffer (probe-derived, or MLP output frozen at reparameterize()).
        """
        if self.alpha_override is not None:
            return torch.full((self.W.shape[0],), self.alpha_override,
                              device=self.W.device, dtype=self.W.dtype)
        if self.gate_mlp is not None and not self.is_reparameterized:
            return torch.sigmoid(self.gate_mlp(self.W).squeeze(-1))
        return self.alpha

    @torch.no_grad()
    def set_alpha_override(self, value: Optional[float]) -> None:
        """Pin routing to a constant (1.0 = spatial expert only, 0.0 =
        semantic only, None = restore the gate).

        Evaluated *after* the gate, so it survives vocabulary swaps: the
        trained gate does not recognise unseen spatial predicates as spatial
        (measured: alpha <= 0.09 on novel spatial strings), which is why a
        spatial-only deployment must pin alpha rather than trust the routing.
        """
        self.alpha_override = None if value is None else float(value)
        self._update_alpha()

    # ------------------------------------------------------------------
    # Spatialness gate (dual-projection head)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def set_spatial_gate(self, u: torch.Tensor, b: torch.Tensor) -> None:
        """Install the fitted text-space spatialness direction.

        ``u`` [text_dim], ``b`` scalar come from a logistic probe fitted on
        the training vocabulary's text embeddings against per-predicate
        spatial flags (train.py). Frozen — never trained by backprop.
        """
        device = self.logit_scale.device
        self.gate_u = u.float().to(device).clone()
        self.gate_b = torch.as_tensor(float(b), device=device)
        self._update_alpha()

    @torch.no_grad()
    def _update_alpha(self) -> None:
        """Bake per-predicate routing weights for the current W into the
        buffer — from the trainable MLP gate when present, else the probe."""
        if self.W.numel() == 0:
            return
        if self.alpha_override is not None:
            self.alpha = torch.full((self.W.shape[0],), self.alpha_override,
                                    device=self.W.device, dtype=self.W.dtype)
        elif self.gate_mlp is not None:
            self.alpha = torch.sigmoid(
                self.gate_mlp(self.W).squeeze(-1)).clone()
        elif self.gate_u.numel():
            self.alpha = torch.sigmoid(
                self.W @ self.gate_u.to(self.W.device) + self.gate_b
            ).clone()
        if getattr(self, "beta_mlp", None) is not None:
            self.beta = torch.sigmoid(self.beta_mlp(self.W).squeeze(-1)).clone()

    # ------------------------------------------------------------------
    # Vocabulary management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_vocabulary(self, pred_names: List[str]) -> None:
        """Encode predicate names with the frozen text encoder.

        This method loads the text encoder **lazily** (only when called) and
        discards it afterwards — it is never kept as a persistent attribute.

        Args:
            pred_names: List of predicate strings, e.g.
                        ``["above", "below", "next to", "holding"]``.
        """
        from transformers import AutoTokenizer, AutoModel

        tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
        device = self.logit_scale.device
        text_encoder = AutoModel.from_pretrained(self.text_model_name).eval().to(device)

        inputs = tokenizer(
            pred_names,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # CLIP is a joint vision-language model — extract only the
        # text sub-module so we don't need dummy pixel_values.
        encoder = getattr(text_encoder, "text_model", text_encoder)

        with torch.autocast(device.type if device.type != "cpu" else "cpu",
                            enabled=False):
            outputs = encoder(**inputs)

        # CLIP exposes pooler_output directly; generic models fall back to
        # last_hidden_state CLS token.
        if hasattr(outputs, "text_model_output"):
            embeds = outputs.text_model_output.pooler_output
        elif hasattr(outputs, "text_embeds"):
            embeds = outputs.text_embeds
        elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeds = outputs.pooler_output
        else:
            # CLS token fallback (BERT-style)
            embeds = outputs.last_hidden_state[:, 0, :]

        embeds = F.normalize(embeds.float(), dim=-1)  # [V, text_dim]

        # Store raw normalised embeddings — no learned text-side transform.
        # .clone() ensures the buffer is a regular tensor (not an inference
        # tensor), which allows load_state_dict to do its inplace copy.
        self.W = embeds.to(device).clone()  # [V, text_dim]
        self.pred_names = list(pred_names)
        self.is_reparameterized = False
        self._update_alpha()

    # ------------------------------------------------------------------
    # dino.txt vocabulary encoding
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # W: frozen buffer (the open-vocabulary contract) or, for the closed-set
    # fine-tune comparison only, a trainable parameter read through an
    # L2-normalised view so every scoring site keeps seeing unit rows.
    # ------------------------------------------------------------------
    @property
    def W(self) -> torch.Tensor:
        p = self._parameters.get("W_param")
        if p is not None:
            return F.normalize(p, dim=-1)
        return self._buffers["W"]

    @W.setter
    def W(self, value: torch.Tensor) -> None:
        # Only reached in trainable mode: while "W" is a buffer, nn.Module's
        # __setattr__ writes the buffer directly and never calls this setter.
        p = self._parameters.get("W_param")
        if p is None:
            self._buffers["W"] = value
            return
        with torch.no_grad():
            p.data = value.detach().to(p.device, p.dtype).clone()

    def make_W_trainable(self) -> None:
        """Turn the installed vocabulary matrix into a learnable parameter
        (closed-set classifier ablation). Call BEFORE DDP/EMA/optimizer
        construction; the state_dict then carries ``W_param`` instead of ``W``
        (benchmark/eval_zeroshot.py maps it back on load)."""
        if "W_param" in self._parameters:
            return
        W = self._buffers.pop("W").detach().clone()
        if W.numel() == 0:
            raise RuntimeError("install a vocabulary before make_W_trainable()")
        self.W_param = nn.Parameter(W)

    @torch.no_grad()
    def set_vocabulary_matrix(self, pred_names: List[str], W) -> None:
        """Install a precomputed text-embedding matrix directly.

        Use with the artifacts of training/text_space_diag.py so training jobs
        share the exact (template-ensembled) embeddings the ontology masks
        were calibrated on — and never load a text encoder at all.

        Args:
            pred_names: Predicate strings, index = label id.
            W:          [V, text_dim] numpy array or tensor; will be
                        L2-normalised and cast to float32.
        """
        W = torch.as_tensor(np.asarray(W), dtype=torch.float32) \
            if not isinstance(W, torch.Tensor) else W.float()
        if W.shape != (len(pred_names), self.text_dim):
            raise ValueError(
                f"W shape {tuple(W.shape)} != ({len(pred_names)}, {self.text_dim})"
            )
        self.W = F.normalize(W, dim=-1).to(self.logit_scale.device).clone()
        self.pred_names = list(pred_names)
        self.is_reparameterized = False
        self._update_alpha()

    @torch.no_grad()
    def encode_vocabulary_dinotxt(
        self,
        pred_names: List[str],
        dinotxt_weights: str,
        bpe_path_or_url: str = (
            "https://dl.fbaipublicfiles.com/dinov3/thirdparty/"
            "bpe_simple_vocab_16e6.txt.gz"
        ),
        backbone_weights: Optional[str] = None,  # unused; kept for API compat
        templates: Optional[List[str]] = None,
    ) -> None:
        """Encode predicates using the frozen dino.txt text transformer.

        Produces 2048-dim embeddings trained jointly with DINOv3 visual
        features — a better match for DINOv3 backbones than CLIP-B/32.
        Does **not** require the external dinov3 package; the text transformer
        is implemented inline and weights are loaded directly from the combined
        checkpoint.

        Args:
            pred_names:      Predicate strings to encode.
            dinotxt_weights: Path to the dino.txt combined checkpoint
                             (vision head + text encoder, *not* the backbone).
                             Filename matches ``dinov3_vitl16_dinotxt_*.pth``.
            bpe_path_or_url: Unused; tokenisation uses HuggingFace
                             ``CLIPTokenizer`` (same BPE, no extra download).
            backbone_weights: Ignored (kept for API compatibility).
            templates:       Optional prompt templates with ``{p}`` slots;
                             per-template embeddings are averaged then
                             re-normalised (must match the template set the
                             ontology/diagnostic artifacts were built with).

        Note:
            ``self.text_dim`` must be ``2048`` at construction time.
        """
        from transformers import CLIPTokenizer

        device = self.logit_scale.device

        # Load text transformer directly from checkpoint (no external package).
        encoder = _DinoTxtEncoder.from_checkpoint(dinotxt_weights, device).eval()

        # HuggingFace CLIPTokenizer uses the same BPE (vocab_size=49408, ctx=77).
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        def _encode(texts: List[str]) -> torch.Tensor:
            embs = []
            for i in range(0, len(texts), 512):
                ids = tokenizer(
                    texts[i:i + 512],
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=_DinoTxtEncoder.CTX_LEN,
                )["input_ids"].to(device)
                embs.append(encoder.encode(ids))
            return torch.cat(embs)

        if templates:
            embeds = sum(_encode([t.format(p=p) for p in pred_names])
                         for t in templates)
            embeds = F.normalize(embeds, dim=-1)
        else:
            embeds = _encode(pred_names)  # [V, 2048], already L2-normalised

        if embeds.shape[-1] != self.text_dim:
            raise ValueError(
                f"dino.txt output dim ({embeds.shape[-1]}) != self.text_dim "
                f"({self.text_dim}).  Construct VocabHead with "
                f"text_dim={embeds.shape[-1]}."
            )

        self.W = embeds.to(device).clone()
        self.pred_names = list(pred_names)
        self.is_reparameterized = False
        self._update_alpha()
        del encoder  # free VRAM

    # ------------------------------------------------------------------
    # Vocabulary sealing
    # ------------------------------------------------------------------

    def reparameterize(self) -> None:
        """Seal the vocabulary for inference.

        Verifies that W is normalised and marks the head as reparameterized.
        After this call the forward pass is a single matmul + logit scale —
        zero language model overhead at runtime.

        Can be called multiple times (e.g. after swapping vocabulary via
        ``encode_vocabulary()``).
        """
        if self.W.numel() == 0:
            raise RuntimeError("Call encode_vocabulary() before reparameterize().")
        with torch.no_grad():
            self.W = F.normalize(self.W, dim=-1)  # idempotent; ensures correctness
            self._update_alpha()
        self.is_reparameterized = True

    # ------------------------------------------------------------------
    # InfoNCE contrastive loss
    # ------------------------------------------------------------------

    def compute_infonce_loss(
        self,
        r: torch.Tensor,
        pred_labels: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Region-text InfoNCE contrastive loss.

        For each valid pair slot that has a GT predicate label, push the visual
        representation toward the corresponding text embedding and away from all
        other predicate embeddings. This is the YOLO-World / GLIP-style
        region-text alignment loss adapted for predicates.

        Args:
            r:           [B, K, d_model]  visual pair representations (pre-proj).
            pred_labels: [B, K]           GT predicate index, -1 if no GT.
            valid_mask:  [B, K]           True for real (non-padding) slots.
        Returns:
            Scalar loss (0 if no GT slots present in this batch).
        """
        if self.W.numel() == 0:
            return r.new_zeros(1).squeeze()

        has_gt = (pred_labels >= 0) & valid_mask  # [B, K]
        if not has_gt.any():
            return r.new_zeros(1).squeeze()

        # W holds raw normalised text embeddings [V, text_dim] — no text_proj.
        proj_r = F.normalize(self.proj(r[has_gt]), dim=-1)  # [M, text_dim]
        labels = pred_labels[has_gt]                         # [M]

        # Cosine similarities against all V training predicates: [M, V]
        logits = proj_r @ self.W.T / self.infonce_temp

        return F.cross_entropy(logits, labels)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def score_query(self, q: torch.Tensor) -> torch.Tensor:
        """Score ALREADY-COMPOSED text-space queries against the vocabulary.

        Used by the compositional path (model builds q = proj(r) + subject +
        object semantic embeddings); this only normalizes and matmuls the
        frozen W — same contract as forward(), no projection applied.

        Args:
            q: [..., text_dim] unnormalized queries.
        Returns:
            [..., V] scaled cosine logits.
        """
        if self.W.numel() == 0:
            raise RuntimeError("No vocabulary loaded. Call encode_vocabulary() first.")
        q = F.normalize(q, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return q @ self.W.T * scale + self.logit_bias

    def score_query_dual(
        self, q_sem: torch.Tensor, q_spa: torch.Tensor
    ) -> torch.Tensor:
        """Two-expert scoring: each predicate's cosine is a per-predicate
        mixture of the semantic and spatial query experts,

            cos_p = (1 - alpha_p) * cos(q_sem, w_p) + alpha_p * cos(q_spa, w_p)

        with alpha read off the TEXT embedding by the fitted spatialness gate
        (so routing generalizes to unseen vocabularies and bakes into
        reparameterization as a per-column constant).

        Args:
            q_sem, q_spa: [..., text_dim] unnormalized expert queries.
        Returns:
            [..., V] scaled mixed cosine logits.
        """
        alpha = self.current_alpha()
        if alpha.numel() != self.W.shape[0]:
            raise RuntimeError(
                "Spatialness gate not fitted for current vocabulary — call "
                "set_spatial_gate() or build_gate_mlp() after installing W."
            )
        cos_sem = F.normalize(q_sem, dim=-1) @ self.W.T
        cos_spa = F.normalize(q_spa, dim=-1) @ self.W.T
        cos = (1.0 - alpha) * cos_sem + alpha * cos_spa
        scale = self.logit_scale.exp().clamp(max=100.0)
        return cos * scale + self.logit_bias

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """Score pair representations against the predicate vocabulary.

        The same code path executes whether or not the head has been
        re-parametrized; the only difference is whether W is in text-space
        (and projected on the fly) or already in d_model space.

        Args:
            r: [B, K, d_model] pair representations from the transformer.
        Returns:
            logits: [B, K, V]  unnormalised cosine scores scaled by
                    ``logit_scale``.  Apply softmax or sigmoid for probabilities.
        """
        if self.W.numel() == 0:
            raise RuntimeError(
                "No vocabulary loaded. Call encode_vocabulary() first."
            )

        # W holds raw normalised text embeddings [V, text_dim] — no text_proj.
        proj_r = F.normalize(self.proj(r), dim=-1)  # [B, K, text_dim]
        scale = self.logit_scale.exp().clamp(max=100.0)
        return torch.einsum("bkd,vd->bkv", proj_r, self.W) * scale + self.logit_bias
