"""Open-vocabulary predicate head.

The vocabulary is a matrix ``W`` of L2-normalised text embeddings, one row
per predicate, produced by the text student (``relsgg.text``) and installed
with ``set_vocabulary_matrix``. Scoring is a scaled cosine between a query in
text space and every row, so swapping the vocabulary is a matrix swap and
inference carries no text encoder.

Two query experts are mixed per predicate: a semantic query and a spatial
query. The mixing weight ``alpha`` is read off the predicate's text embedding
by a small gate MLP, so unseen predicates are routed too.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VocabHead(nn.Module):
    def __init__(self, d_model: int = 512, text_dim: int = 512,
                 logit_scale_init: float = 5.0, proj_layers: int = 2,
                 gate_hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.text_dim = text_dim

        if proj_layers <= 1:
            self.proj = nn.Linear(d_model, text_dim, bias=False)
            nn.init.xavier_uniform_(self.proj.weight)
        else:
            hidden = max(d_model * 2, text_dim // 2)
            layers: list = []
            in_dim = d_model
            for _ in range(proj_layers - 1):
                layers += [nn.Linear(in_dim, hidden), nn.GELU()]
                in_dim = hidden
            layers += [nn.LayerNorm(in_dim), nn.Linear(in_dim, text_dim, bias=False)]
            self.proj = nn.Sequential(*layers)
            for m in self.proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_init)))
        self.logit_bias = nn.Parameter(torch.zeros(()))

        self.register_buffer("W", torch.empty(0))       # [V, text_dim]
        self.register_buffer("alpha", torch.empty(0))   # [V] routing weights
        self.pred_names: List[str] = []
        self.is_reparameterized: bool = False

        self.gate_mlp = nn.Sequential(nn.Linear(text_dim, gate_hidden), nn.GELU(),
                                      nn.Linear(gate_hidden, 1))
        for m in self.gate_mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # -- vocabulary -------------------------------------------------------

    @torch.no_grad()
    def set_vocabulary_matrix(self, pred_names: List[str], W) -> None:
        """Install ``W`` (``[V, text_dim]``, any float array) as the vocabulary."""
        W = torch.as_tensor(W, dtype=torch.float32) if not isinstance(W, torch.Tensor) else W.float()
        if W.shape != (len(pred_names), self.text_dim):
            raise ValueError(f"W shape {tuple(W.shape)} != ({len(pred_names)}, {self.text_dim})")
        self.W = F.normalize(W, dim=-1).to(self.logit_scale.device).clone()
        self.pred_names = list(pred_names)
        self.is_reparameterized = False
        self._update_alpha()

    def reparameterize(self) -> None:
        """Seal the vocabulary: bake the routing weights and mark the head
        ready for inference. Idempotent; call again after a vocabulary swap."""
        if self.W.numel() == 0:
            raise RuntimeError("install a vocabulary before reparameterize()")
        with torch.no_grad():
            self.W = F.normalize(self.W, dim=-1)
            self._update_alpha()
        self.is_reparameterized = True

    # -- routing gate -----------------------------------------------------

    def current_alpha(self) -> torch.Tensor:
        """Per-predicate spatial weight. Computed live from ``W`` while the
        head trains (so the gate receives gradient); the baked buffer after
        ``reparameterize``."""
        if not self.is_reparameterized:
            return torch.sigmoid(self.gate_mlp(self.W).squeeze(-1))
        return self.alpha

    @torch.no_grad()
    def _update_alpha(self) -> None:
        if self.W.numel():
            self.alpha = torch.sigmoid(self.gate_mlp(self.W).squeeze(-1)).clone()

    def warm_start_gate(self, target_alpha: torch.Tensor, steps: int = 300,
                        lr: float = 1e-2) -> float:
        """Regress the gate onto a target routing (training start). Returns
        the final mean squared error."""
        target = target_alpha.to(self.W.device).float()
        opt = torch.optim.Adam(self.gate_mlp.parameters(), lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss = F.mse_loss(torch.sigmoid(self.gate_mlp(self.W).squeeze(-1)), target)
            loss.backward()
            opt.step()
        self._update_alpha()
        return float(loss.detach())

    # -- scoring ----------------------------------------------------------

    def score_query_dual(self, q_sem: torch.Tensor, q_spa: torch.Tensor) -> torch.Tensor:
        """Per-predicate mixture of the two experts' cosines:
        ``cos_p = (1 - alpha_p) cos(q_sem, w_p) + alpha_p cos(q_spa, w_p)``,
        scaled and shifted. ``[..., text_dim]`` -> ``[..., V]``."""
        if self.W.numel() == 0:
            raise RuntimeError("no vocabulary installed")
        alpha = self.current_alpha()
        cos_sem = F.normalize(q_sem, dim=-1) @ self.W.T
        cos_spa = F.normalize(q_spa, dim=-1) @ self.W.T
        cos = (1.0 - alpha) * cos_sem + alpha * cos_spa
        scale = self.logit_scale.exp().clamp(max=100.0)
        return cos * scale + self.logit_bias
