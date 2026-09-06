"""Training objective of the predicate head.

The vocabulary keeps every surface form the corpus uses (10,102 predicates,
synonyms included), so a softmax over it would treat every synonym of the
annotated predicate as a negative. The objective instead contrasts each
labelled pair against a small set built from the batch, with per-column
weights estimated from the corpus (``soft_supervision.npz``, built by
``training/build_soft_supervision.py``):

  pos_w[g, v]   how much column v counts as a positive of predicate g
                (a fitted P(synonym | text cosine); 1 on the diagonal)
  neg_lw[g, v]  log(1 - P(v also true | g annotated)), added to the
                denominator logits so plausibly-true columns weigh less
  sym[g]        P(the relation is reciprocal), which weights the direction
                hinge
  inverses      spatial opposites (above / below,...) that always stay
                full-weight negatives of each other
"""
from __future__ import annotations

import json
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PredicateOntology:
    """Per-vocabulary supervision tables loaded from ``soft_supervision.npz``."""

    def __init__(self, predicates: List[str], pos_w: torch.Tensor, neg_lw: torch.Tensor,
                 sym: torch.Tensor, inverse_mask: torch.Tensor) -> None:
        V = len(predicates)
        self.predicates = predicates
        self.pos_w = pos_w                       # [V, V] fp16
        self.neg_lw = neg_lw                     # [V, V] fp16
        self.sym = sym                           # [V]
        self.pos_mask = pos_w.float() > 0        # [V, V] bool
        self.inverse_mask = inverse_mask         # [V, V] bool
        self.group_of = torch.arange(V)          # each predicate is its own group

    @classmethod
    def from_soft_supervision(cls, meta_path: str, npz_path: str) -> "PredicateOntology":
        meta = json.load(open(meta_path))
        z = np.load(npz_path, allow_pickle=False)
        names = [str(x) for x in z["predicates"]]
        assert names == meta["predicates"], f"{npz_path} vocabulary != {meta_path}"
        V = len(names)

        pos_w = np.zeros((V, V), dtype=np.float16)
        pos_w[z["pos_i"], z["pos_j"]] = z["pos_w"]
        np.fill_diagonal(pos_w, 1.0)
        neg_lw = np.zeros((V, V), dtype=np.float16)
        neg_lw[z["neg_i"], z["neg_j"]] = np.maximum(
            z["neg_lw"].astype(np.float32), math.log(1e-6)).astype(np.float16)
        np.fill_diagonal(neg_lw, 0.0)
        # Inverse evidence is accumulated with max (seed expansions overlap)
        # and vetoes any down-weighting: the estimator cannot tell opposites
        # apart, so they must remain full negatives.
        inv_w = np.zeros((V, V), dtype=np.float32)
        np.maximum.at(inv_w, (z["inv_i"], z["inv_j"]), z["inv_w"].astype(np.float32))
        inv_w = np.maximum(inv_w, inv_w.T)
        neg_lw[inv_w > 0] = 0.0
        return cls(names, torch.from_numpy(pos_w), torch.from_numpy(neg_lw),
                   torch.from_numpy(z["sym"].astype(np.float32)),
                   torch.from_numpy(inv_w > 0.5))

    def stats(self) -> dict:
        negw = 1.0 - torch.exp(self.neg_lw.float())
        return {
            "V": len(self.predicates),
            "avg_pos_members": float((self.pos_w > 0).float().sum(1).mean()),
            "avg_pos_weight_mass": float(self.pos_w.float().sum(1).mean()),
            "avg_neg_downweight_mass": float(negw.sum(1).mean()),
            "inverse_pairs_strong": int(self.inverse_mask.sum()) // 2,
            "sym_gt_half": int((self.sym > 0.5).sum()),
        }


# ---------------------------------------------------------------------------
# slot targets
# ---------------------------------------------------------------------------

def _flat_pair_lookup(sub_idx: torch.Tensor, obj_idx: torch.Tensor, valid_mask: torch.Tensor):
    """Sync-free lookup from (image, sub, obj) keys to flat slot indices.
    Key layout: ``b * 2**20 + sub * 1024 + obj`` (boxes < 1024)."""
    B, K = sub_idx.shape
    base = (torch.arange(B, device=sub_idx.device) << 20).unsqueeze(1)
    keys = (base + sub_idx * 1024 + obj_idx).masked_fill(~valid_mask, -1).reshape(-1)
    order = keys.argsort()
    sk = keys[order]

    def lookup(q: torch.Tensor) -> torch.Tensor:
        pos = torch.searchsorted(sk, q).clamp(max=sk.numel() - 1)
        hit = sk[pos] == q
        return torch.where(hit, order[pos], torch.full_like(q, -1))

    return lookup


def _cat_relations(targets: List[dict], device):
    """Concatenate per-image relations: ``(b_ids [R], rels [R, 3], weights [R])``."""
    per = [t.get("relations") for t in targets]
    lens = [0 if r is None else len(r) for r in per]
    if sum(lens) == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, torch.zeros(0, 3, dtype=torch.long, device=device), torch.zeros(0, device=device)
    rels = torch.cat([r for r in per if r is not None and len(r)])
    b_ids = torch.repeat_interleave(torch.arange(len(targets), device=device),
                                    torch.tensor(lens, device=device))
    ws = []
    for t, n in zip(targets, lens):
        if n == 0:
            continue
        w = t.get("rel_weights")
        ws.append(w.to(device) if w is not None else torch.ones(n, device=device))
    return b_ids, rels.to(device), torch.cat(ws)


def build_slot_targets(sub_idx: torch.Tensor, obj_idx: torch.Tensor, valid_mask: torch.Tensor,
                       targets: List[dict], V: int):
    """Multi-hot predicate targets per sampled slot, ``[B, K, V]`` bool, and
    the mean relation weight per slot ``[B, K]`` (1 where unlabelled). A pair
    keeps every predicate annotated on it."""
    B, K = sub_idx.shape
    device = sub_idx.device
    multi_hot = torch.zeros(B * K, V, dtype=torch.bool, device=device)
    slot_w = torch.ones(B * K, device=device)
    b_ids, rels, rw = _cat_relations(targets, device)
    if rels.numel():
        lookup = _flat_pair_lookup(sub_idx, obj_idx, valid_mask)
        slot = lookup((b_ids << 20) + rels[:, 0] * 1024 + rels[:, 1])
        hit = slot >= 0
        multi_hot[slot[hit], rels[hit, 2]] = True
        slot_w.scatter_reduce_(0, slot[hit], rw[hit], reduce="mean", include_self=False)
        slot_w.clamp_(min=1e-3)
    return multi_hot.view(B, K, V), slot_w.view(B, K)


def swap_direction_hinge(q_sem: torch.Tensor, q_spa: torch.Tensor, alpha: torch.Tensor,
                         W: torch.Tensor, sub_idx: torch.Tensor, obj_idx: torch.Tensor,
                         valid_mask: torch.Tensor, targets: List[dict],
                         sym: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    """``relu(margin + cos(swapped slot, g) - cos(annotated slot, g))`` for
    every relation whose swapped pair was also sampled, weighted by
    ``1 - sym[g]`` so reciprocal predicates are not pushed. Relations
    annotated in both directions in the same image are skipped."""
    device = W.device
    b_ids, rels, _ = _cat_relations(targets, device)
    if rels.numel() == 0:
        return W.new_zeros(())
    g = rels[:, 2]
    lookup = _flat_pair_lookup(sub_idx, obj_idx, valid_mask)
    f_slot = lookup((b_ids << 20) + rels[:, 0] * 1024 + rels[:, 1])
    b_slot = lookup((b_ids << 20) + rels[:, 1] * 1024 + rels[:, 0])
    keep = (f_slot >= 0) & (b_slot >= 0)
    V_bits = int(W.shape[0]).bit_length()
    key_f = (((b_ids << 20) + rels[:, 0] * 1024 + rels[:, 1]) << V_bits) | g
    key_b = (((b_ids << 20) + rels[:, 1] * 1024 + rels[:, 0]) << V_bits) | g
    keep &= ~torch.isin(key_b, key_f)
    f_slot, b_slot = f_slot.clamp(min=0), b_slot.clamp(min=0)
    keep_f = keep.float() * (1.0 - sym[g])

    w_g = W[g]
    a_g = alpha[g].unsqueeze(-1)
    D = q_sem.shape[-1]
    flat_s, flat_p = q_sem.reshape(-1, D), q_spa.reshape(-1, D)

    def _cos(slots):
        c = (F.normalize(flat_s[slots], dim=-1) * w_g).sum(-1, keepdim=True)
        cp = (F.normalize(flat_p[slots], dim=-1) * w_g).sum(-1, keepdim=True)
        return ((1.0 - a_g) * c + a_g * cp).squeeze(-1)

    hinge = F.relu(margin + _cos(b_slot) - _cos(f_slot))
    return (hinge * keep_f).sum() / keep_f.sum().clamp(min=1.0)


# ---------------------------------------------------------------------------
# batch-local contrastive loss
# ---------------------------------------------------------------------------

class BatchLocalInfoNCE(nn.Module):
    """Region-text contrastive loss over a set built per batch.

    The contrast set S is the batch's annotated predicates, their spatial
    inverses, the ``n_neg * hard_frac`` most confusable columns that the
    corpus statistics mark as safely false, and uniform random columns up to
    ``n_neg``. For each labelled slot and each of its predicates g, the loss
    is the weighted mean over the positive columns of g of
    ``logsumexp(denominator) - logit(column)``, where the denominator logits
    carry ``neg_lw[g]`` and positive and inverse columns are exempt from
    down-weighting.
    """

    def __init__(self, ontology: PredicateOntology, temp: float = 0.07,
                 n_neg: int = 512, hard_frac: float = 0.5) -> None:
        super().__init__()
        self.temp = temp
        self.n_neg = n_neg
        self.hard_frac = hard_frac
        # Plain attributes (not buffers): V x V tables must not be copied by
        # EMA or saved with the model.
        self.pos_mask = ontology.pos_mask
        self.inverse_mask = ontology.inverse_mask
        self.pos_w = ontology.pos_w
        self.neg_lw = ontology.neg_lw

    def _ensure_device(self, device: torch.device) -> None:
        if self.pos_mask.device != device:
            self.pos_mask = self.pos_mask.to(device)
            self.inverse_mask = self.inverse_mask.to(device)
            self.pos_w = self.pos_w.to(device)
            self.neg_lw = self.neg_lw.to(device)

    @torch.no_grad()
    def build_set(self, labels: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Contrast set S as unique predicate ids ``[n_S]``."""
        V = W.shape[0]
        device = labels.device
        classes = labels.unique()
        parts = [classes, self.inverse_mask[classes].any(0).nonzero().flatten()]
        n_hard = int(self.n_neg * self.hard_frac)
        if n_hard > 0:
            sim = W[classes] @ W.T
            score = sim.amax(0) + self.neg_lw[classes].amin(0).float()
            score = score.masked_fill(self.pos_mask[classes].any(0), -2.0)
            parts.append(score.topk(min(n_hard, V)).indices)
        n_rand = self.n_neg - n_hard
        if n_rand > 0:
            parts.append(torch.randint(0, V, (n_rand,), device=device))
        return torch.cat(parts).unique()

    def forward(self, feats: torch.Tensor, labels: torch.Tensor, W: torch.Tensor,
                feats_spa: Optional[torch.Tensor] = None,
                alpha: Optional[torch.Tensor] = None,
                weights: Optional[torch.Tensor] = None,
                col_allow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feats:     ``[M, D]`` semantic queries of the labelled slots.
            labels:    ``[M, V]`` bool multi-hot targets.
            W:         ``[V, D]`` normalised vocabulary.
            feats_spa: ``[M, D]`` spatial queries, mixed per column by ``alpha``.
            weights:   ``[M]`` per-slot loss weights.
            col_allow: ``[M, V]`` bool, columns a slot may be contrasted
                       against (source-aware masking); positives always count.
        """
        if feats.numel() == 0:
            return feats.new_zeros(())
        self._ensure_device(feats.device)
        flat = labels.nonzero(as_tuple=True)[1]
        S = self.build_set(flat, W)
        allow_S = col_allow[:, S] if col_allow is not None else None

        feats = F.normalize(feats, dim=-1)
        cos = feats @ W[S].T
        if feats_spa is not None:
            a = alpha[S]
            cos = (1.0 - a) * cos + a * (F.normalize(feats_spa, dim=-1) @ W[S].T)
        logits = cos / self.temp
        neg_inf = torch.finfo(logits.dtype).min

        a_idx, g_idx = labels.nonzero(as_tuple=True)
        hot = labels.float()
        pos_any = (hot @ self.pos_mask[:, S].float()) > 0
        inv_any = (hot @ self.inverse_mask[:, S].float()) > 0
        # Denominator weights: summing log(1 - p) over a slot's predicates is
        # the log of the union probability under independence.
        lw = (hot @ self.neg_lw[:, S].float()).masked_fill(pos_any | inv_any, 0.0)
        den_logits = logits + lw
        ign = torch.zeros_like(pos_any)
        if allow_S is not None:
            ign = ign | (~allow_S & ~pos_any)
        den_lse = den_logits.masked_fill(ign, neg_inf).logsumexp(-1)          # [M]

        # One row per (slot, predicate): every predicate of a slot must beat
        # the shared denominator on its own.
        w = self.pos_w[g_idx][:, S].to(logits.dtype)                             # [M', n_S]
        per = den_lse[a_idx].unsqueeze(-1) - logits[a_idx]
        loss = (per * w).sum(-1) / w.sum(-1).clamp(min=1e-6)
        n_lab = labels.sum(-1).clamp(min=1)
        wt = weights if weights is not None else torch.ones_like(n_lab, dtype=loss.dtype)
        w_row = (wt / n_lab)[a_idx]
        return (loss * w_row).sum() / w_row.sum().clamp(min=1e-6)
