"""Synonym-aware relation losses for the 10K open-vocabulary predicate set.

Why this module exists (plan D2): the vocabulary deliberately preserves
synonym diversity (10,102 surface forms, see memory
keep-predicate-synonym-diversity). Softmax CE / one-hot focal over that
vocabulary punishes every synonym of the GT as a negative — actively
destroying the text-embedding geometry the data was built to teach. These
losses replace them:

  SynonymAwareRelLoss   masked sigmoid multi-label with focal modulation:
      target 1   for the GT's whole canonical group (multi-positive),
      IGNORED    for near-synonyms above tau_ignore (cannot be trusted as
                 negatives — the LLM annotation is positive-unlabeled),
      target 0   elsewhere, down-weighted by neg_weight (unlabeled ≠ false),
      target 0   at FULL weight for spatial-inverse predicates of the GT
                 ("above" vs "below" have high text cosine but must stay
                 hard negatives or directionality is never learned).

  MultiPositiveInfoNCE  alignment loss: logsumexp over the canonical group
      as the positive mass; tau-synonyms removed from the denominator;
      inverses always kept in the denominator.

Both consume a PredicateOntology built once from pack + diagnostic artifacts
(canonical_groups.json, pred_embeds_*.npz from training/text_space_diag.py).
Canonical forms steer LOSS LOGIC only — emitted labels are never rewritten.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Spatial inverse map over canonical forms (mirror of training/invert_spatial.py,
# duplicated here so the relsgg package stays importable without training/).
CANONICAL_INVERSE = {
    "above": "below", "below": "above",
    "to the left of": "to the right of", "to the right of": "to the left of",
    "in front of": "behind", "behind": "in front of",
    "on top of": "beneath", "beneath": "on top of",
}


class PredicateOntology:
    """Precomputed masks over the predicate vocabulary.

    TWO WAYS TO DECIDE WHAT COUNTS AS A POSITIVE
    --------------------------------------------
    ``canon_map`` (LEGACY, runs v34-v39): a hand-written synonym table
    (datagen/sgg_canon.py ``_SPATIAL`` -> canonical_groups.json) declares groups, and
    every group member is a positive of every other. Measured 2026-07-28, this is the
    direct cause of the spatial 0.000s. The positive term is a logsumexp over the group,
    which is satisfied by whichever single member is easiest to reach in the text space;
    the rest get almost no positive gradient while still taking full negative pressure
    from other anchors, so they are driven far down the ranking. On megasg val the exact
    GT string ended at median rank 338/1972 for `in front of`, 262 for `next to`, 148 for
    `to the left of`, while the group's chosen member sat at rank 1. The winner is not
    even the frequent form: `before` (11 relations) beat `in front of` (414,961), because
    the hand table asserts they are synonyms while the text encoder puts them 0.607 apart
    — no closer than `beside` at 0.606. Of 536 multi-member groups, `in front of` is the
    least text-coherent in the whole vocabulary, and it is the predicate that failed.

    ``canon_map=None`` (DEFAULT): no table, nothing asserted. The GT string is its own
    and only positive, so it always receives the positive gradient and can never be
    starved by a synonym. The two learned signals are used only to decide what to
    ABSTAIN on — which columns are too unreliable to serve as negatives:

        text cosine >= tau_ignore                                     (as before), OR
        context similarity >= tau_ctx AND text cosine >= tau_ctx_floor

    The second clause is the distributional statistic from
    training/build_predicate_context.py: predicates used between the same kinds of
    subject and object are plausibly interchangeable, so penalising one when the other
    was annotated is unsound (the annotation is positive-unlabelled). ``tau_ctx_floor``
    is what makes it safe: antonyms have near-identical context distributions —
    `to the left of` vs `to the right of` scores 0.892, the highest in the vocabulary —
    and would otherwise be abstained on, destroying the direction signal. The student
    text encoder was distilled to separate antonyms and puts that pair at 0.323, so the
    floor vetoes it. Each signal covers precisely the other's blind spot.

    Why abstain rather than assert: every attempt to declare two predicates equivalent
    has produced a wrong equivalence, hand-written or learned. The conjunction's
    surviving pairs at text >= 0.90 are dominated by preposition swaps that are NOT
    interchangeable (`sitting on`/`sitting in` 0.989, `parked on`/`parked in` 0.989).
    Abstaining costs a little gradient; asserting teaches a falsehood.

    Args:
        predicates:    Vocabulary in label-id order (pack meta order).
        canon_map:     LEGACY hand-written groups, or None (default) for
                       identity positives.
        embeddings:    [V, D] text embeddings (same order), or None to disable
                       cosine-based ignore masking.
        tau_ignore:    Cosine at/above which a column is dropped from the negative
                       set (text-space diagnostic, 95% synonym recall).
        ctx_sim/ctx_ids: [n, n] context-similarity block and the predicate ids
                       it is indexed by (build_predicate_context.py).
        tau_ctx:       Context similarity at/above which to abstain.
        tau_ctx_floor: Text cosine BELOW which the context clause is refused —
                       the antonym veto. Do not lower without re-checking that
                       `to the left of`/`to the right of` stays a negative.
        class_weight_cap / class_weight_pow: long-tail weighting,
                       w = min(cap, (f/median)^-pow).
        counts:        Optional per-predicate label counts for the weights.

    Exposes (torch tensors, moved lazily to the loss device):
        pos_mask     [V, V] bool  — positives of g (identity unless canon_map)
        ignore_mask  [V, V] bool  — columns of g to drop from the loss
        inverse_mask [V, V] bool  — spatial inverses of g (forced negatives)
        class_weight [V] float    — per-GT-class loss weight
    """

    def __init__(
        self,
        predicates: List[str],
        canon_map: Optional[Dict[str, str]] = None,
        embeddings: Optional[np.ndarray] = None,
        tau_ignore: float = 0.9,
        class_weight_pow: float = 0.5,
        class_weight_cap: float = 3.0,
        counts: Optional[Dict[str, int]] = None,
        ctx_sim: Optional[np.ndarray] = None,
        ctx_ids: Optional[np.ndarray] = None,
        tau_ctx: float = 0.5,
        tau_ctx_floor: float = 0.85,
        group_positives: bool = False,
    ) -> None:
        V = len(predicates)
        self.predicates = predicates
        self.soft = False
        self.group_positives = bool(group_positives and canon_map is not None)

        E = None
        if embeddings is not None:
            E = embeddings.astype(np.float32)
            E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-8

        inv = np.zeros((V, V), dtype=bool)
        if canon_map is not None:
            canon = [canon_map.get(p, p) for p in predicates]
            canon_ids: Dict[str, int] = {}
            group_of = np.empty(V, dtype=np.int64)
            for i, c in enumerate(canon):
                group_of[i] = canon_ids.setdefault(c, len(canon_ids))
            self.num_groups = len(canon_ids)
            # Groups always expand the INVERSE seeds; whether they also define
            # POSITIVES is the thing under test.
            for a_c, b_c in CANONICAL_INVERSE.items():
                ga, gb = canon_ids.get(a_c), canon_ids.get(b_c)
                if ga is not None and gb is not None:
                    inv |= (group_of[:, None] == ga) & (group_of[None, :] == gb)
            pos = (group_of[:, None] == group_of[None, :] if self.group_positives
                   else np.eye(V, dtype=bool))
        else:
            group_of = np.arange(V, dtype=np.int64)
            pos = np.eye(V, dtype=bool)
            self.num_groups = V
            # No group table: mark the seed strings only. Expanding them through a
            # text-space cosine ball was tried and REJECTED — the student space has
            # local defects that a ball propagates wholesale. `beneath` (a seed for
            # the on-top-of/beneath pair) sits within 0.94 of `reflected in`,
            # `depicted on` and `depicts`, which all became "inverses of above";
            # and cos(`above`, `in front of`) = 0.961, so the vertical and depth
            # seeds expanded into each other's balls and produced identical rows.
            # Pass canon_groups_path to get the tested 455-pair expansion instead.
            idx = {p: i for i, p in enumerate(predicates)}
            for a_s, b_s in CANONICAL_INVERSE.items():
                ia, ib = idx.get(a_s), idx.get(b_s)
                if ia is not None and ib is not None:
                    inv[ia, ib] = True
        self.group_of = torch.from_numpy(group_of)
        inv |= inv.T

        if E is not None:
            # Chunked V×V cosine to keep peak memory modest
            ign = np.zeros((V, V), dtype=bool)
            step = 1024
            for i in range(0, V, step):
                ign[i:i + step] = (E[i:i + step] @ E.T) >= tau_ignore
            if ctx_sim is not None and ctx_ids is not None:
                # Abstain on distributionally interchangeable pairs too, but only
                # where the text space agrees they are not opposites.
                sel = np.asarray(ctx_ids, dtype=np.int64)
                blk = np.asarray(ctx_sim) >= tau_ctx
                blk &= (E[sel] @ E[sel].T) >= tau_ctx_floor
                add = np.zeros((V, V), dtype=bool)
                add[np.ix_(sel, sel)] = blk
                self.n_ctx_ignored = int((add & ~ign).sum())
                ign |= add
            ign &= ~pos
            ign &= ~inv  # inverses stay negatives no matter how close in text
        else:
            ign = np.zeros((V, V), dtype=bool)

        self.pos_mask = torch.from_numpy(pos)
        self.ignore_mask = torch.from_numpy(ign)
        self.inverse_mask = torch.from_numpy(inv)
        self.tau_ignore = tau_ignore

        if counts:
            group_count = np.zeros(self.num_groups, dtype=np.float64)
            for i, p in enumerate(predicates):
                group_count[group_of[i]] += counts.get(p, 0)
            med = max(np.median(group_count[group_count > 0]), 1.0)
            # clip(min=1): zero-count predicates never appear as GT, their
            # weight is irrelevant — but keep the array finite and warning-free
            w = np.minimum(class_weight_cap,
                           (group_count[group_of].clip(min=1.0) / med)
                           ** (-class_weight_pow))
            self.class_weight = torch.from_numpy(w.astype(np.float32))
        else:
            self.class_weight = torch.ones(V)

    # ------------------------------------------------------------------

    @classmethod
    def from_artifacts(
        cls,
        meta_path: str,
        canon_groups_path: Optional[str] = None,
        embeds_path: Optional[str] = None,
        tau_ignore: float = 0.9,
        context_path: Optional[str] = None,
        **kw,
    ) -> "PredicateOntology":
        """Build from pack meta.json + text_space_diag.py artifacts.

        ``canon_groups_path`` still expands the four INVERSE seeds (direction
        supervision, which measures fine: opposite-share 0.0% on every spatial
        predicate). Whether those same groups also define POSITIVES is
        ``group_positives``, default False — pass True only to reproduce v34-v39.
        """
        meta = json.load(open(meta_path))
        canon_map = json.load(open(canon_groups_path)) if canon_groups_path else None
        emb = None
        if embeds_path:
            z = np.load(embeds_path)
            emb_preds = [str(p) for p in z["predicates"]]
            assert emb_preds == meta["predicates"], (
                "embedding npz predicate order != pack meta order — regenerate "
                "with training/text_space_diag.py on this pack"
            )
            emb = z["embeddings"]
        ctx_sim = ctx_ids = None
        if context_path:
            c = np.load(context_path)
            # Same index-space trap as everywhere else: the table is written in the
            # ontology vocabulary, so refuse it outright if that is not this one.
            ctx_preds = [str(p) for p in c["predicates"]]
            assert ctx_preds == meta["predicates"], (
                f"{context_path} was built against a different vocabulary "
                f"({len(ctx_preds)} predicates vs this ontology's "
                f"{len(meta['predicates'])}) — rebuild it with "
                "training/build_predicate_context.py --ontology_meta <this meta>"
            )
            ctx_sim, ctx_ids = c["ctx_sim"], c["pred_ids"]
        return cls(
            predicates=meta["predicates"],
            canon_map=canon_map,
            embeddings=emb,
            tau_ignore=tau_ignore,
            counts=meta.get("predicate_counts"),
            ctx_sim=ctx_sim,
            ctx_ids=ctx_ids,
            **kw,
        )

    # ------------------------------------------------------------------

    @classmethod
    def from_soft_supervision(
        cls,
        meta_path: str,
        npz_path: str,
        class_weight_pow: float = 0.5,
        class_weight_cap: float = 3.0,
    ) -> "PredicateOntology":
        """Build from a soft_supervision.npz (training/build_soft_supervision.py).

        Every semantic constant is ESTIMATED there and validated on held-out data;
        this classmethod only unpacks. The ontology it returns carries, beyond the
        legacy boolean masks (kept for the sampler and the exemption logic):

          pos_w   [V, V] fp16  — per-member positive weights (diag 1.0), the
                  isotonic P(synonym | cos_v2). Replaces group positives AND the
                  pos_member_weight scalar.
          neg_lw  [V, V] fp16  — log(1 - P(also-true)), additive denominator
                  weight. p -> 1 reproduces the old ignore mask as a limit;
                  replaces tau_ignore, the hard_lo band and the 0.3 constants.
          sym     [V]  float   — P(reciprocal) from reverse-annotation EM;
                  replaces the alpha<=0.5 swap-hinge eligibility rule.
          inv_elig[V]  float   — continuous inverse-eligibility (max kernel
                  inverse weight), for the hinge.
          w_cooc  float        — fitted replacement for soft_neg_weight.
        """
        meta = json.load(open(meta_path))
        z = np.load(npz_path, allow_pickle=False)
        names = [str(x) for x in z["predicates"]]
        assert names == meta["predicates"], (
            f"{npz_path} vocabulary != {meta_path} — rebuild the artifact")
        V = len(names)

        self = cls.__new__(cls)
        self.predicates = names
        self.group_positives = False
        self.soft = True
        self.num_groups = V
        self.group_of = torch.arange(V)

        pos_w = np.zeros((V, V), dtype=np.float16)
        pos_w[z["pos_i"], z["pos_j"]] = z["pos_w"]
        np.fill_diagonal(pos_w, 1.0)
        neg_lw = np.zeros((V, V), dtype=np.float16)
        neg_lw[z["neg_i"], z["neg_j"]] = np.maximum(
            z["neg_lw"].astype(np.float32), math.log(1e-6)).astype(np.float16)
        np.fill_diagonal(neg_lw, 0.0)
        # MAX-accumulate, not assign: the four seed expansions overlap, and fancy
        # assignment keeps the LAST write — which demoted the (above, below) seed
        # pair itself to 0.097 via its weak appearance in the on-top-of/beneath
        # expansion, silently dropping it from the exemption set.
        inv_w = np.zeros((V, V), dtype=np.float32)
        np.maximum.at(inv_w, (z["inv_i"], z["inv_j"]), z["inv_w"].astype(np.float32))
        inv_w = np.maximum(inv_w, inv_w.T)
        # Direction safety, baked into the weights: the ctx feature is antonym-
        # blind, so the estimator marks direction-opposites "also true" (measured:
        # above|below and on|under at full removal). ANY inverse-kernel evidence
        # vetoes the down-weight — the old `ign &= ~inv` principle, made soft.
        neg_lw[inv_w > 0] = 0.0

        self.pos_w = torch.from_numpy(pos_w)
        self.neg_lw = torch.from_numpy(neg_lw)
        self.sym = torch.from_numpy(z["sym"].astype(np.float32))
        self.inv_elig = torch.from_numpy(inv_w.max(1).astype(np.float32))
        self.w_cooc = float(z["w_cooc"])
        # boolean views for the sampler / exemptions (>0.5 = the strong pairs;
        # weaker evidence acts through the continuous weights, never as structure)
        self.pos_mask = torch.from_numpy(pos_w.astype(np.float32) > 0)
        self.ignore_mask = torch.zeros(V, V, dtype=torch.bool)
        self.inverse_mask = torch.from_numpy(inv_w > 0.5)
        self.tau_ignore = None

        counts = meta.get("predicate_counts") or {}
        c = np.array([counts.get(p, 0) for p in names], dtype=np.float64)
        med = max(np.median(c[c > 0]), 1.0) if (c > 0).any() else 1.0
        w = np.minimum(class_weight_cap,
                       (c.clip(min=1.0) / med) ** (-class_weight_pow))
        self.class_weight = torch.from_numpy(w.astype(np.float32))
        return self

    def stats(self) -> dict:
        V = len(self.predicates)
        if getattr(self, "soft", False):
            negw = 1.0 - torch.exp(self.neg_lw.float())
            return {
                "V": V, "positives": "kernel soft weights",
                "avg_pos_members": float((self.pos_w > 0).float().sum(1).mean()),
                "avg_pos_weight_mass": float(self.pos_w.float().sum(1).mean()),
                "avg_neg_downweight_mass": float(negw.sum(1).mean()),
                "effectively_removed_per_class": float(
                    (negw > 0.94).float().sum(1).mean()),
                "inverse_pairs_strong": int(self.inverse_mask.sum()) // 2,
                "sym_gt_half": int((self.sym > 0.5).sum()),
                "w_cooc": self.w_cooc,
            }
        return {
            "V": V,
            "positives": "canonical groups" if self.group_positives else "identity",
            "canonical_groups": self.num_groups,
            "avg_group_size": float(self.pos_mask.float().sum(1).mean()),
            "avg_ignored_per_class": float(self.ignore_mask.float().sum(1).mean()),
            "ctx_added_to_ignore": getattr(self, "n_ctx_ignored", 0),
            "inverse_pairs": int(self.inverse_mask.sum()) // 2,
        }


def _flat_pair_lookup(
    sub_idx: torch.Tensor,       # [B, K]
    obj_idx: torch.Tensor,       # [B, K]
    valid_mask: torch.Tensor,    # [B, K]
):
    """Sync-free lookup from global pair keys to flat slot indices.

    Sampled pairs are unique per image, so each (b, sub, obj) key maps to at
    most one slot. Returns ``lookup(keys [N]) -> flat idx into B*K (or -1)``.
    Key layout: b * 2^20 + sub * 1024 + obj (boxes ≤ 100 → safe).
    """
    B, K = sub_idx.shape
    base = (torch.arange(B, device=sub_idx.device) << 20).unsqueeze(1)
    keys = (base + sub_idx * 1024 + obj_idx).masked_fill(~valid_mask, -1)
    keys = keys.reshape(-1)                                    # [B*K]
    order = keys.argsort()
    sk = keys[order]

    def lookup(q: torch.Tensor) -> torch.Tensor:
        pos = torch.searchsorted(sk, q).clamp(max=sk.numel() - 1)
        hit = sk[pos] == q
        return torch.where(hit, order[pos], torch.full_like(q, -1))

    return lookup


def _cat_relations(targets: List[dict], device):
    """Concat per-image relations with batch ids. Pure tensor ops (no sync).

    Returns (b_ids [Rt], rels [Rt, 3], weights [Rt]) — empty tensors if the
    batch has no relations at all.
    """
    per = [t.get("relations") for t in targets]
    lens = [0 if r is None else len(r) for r in per]
    if sum(lens) == 0:
        z = torch.zeros(0, dtype=torch.long, device=device)
        return z, torch.zeros(0, 3, dtype=torch.long, device=device), \
            torch.zeros(0, device=device)
    rels = torch.cat([r for r in per if r is not None and len(r)])
    b_ids = torch.repeat_interleave(
        torch.arange(len(targets), device=device),
        torch.tensor(lens, device=device))
    ws = []
    for t, n in zip(targets, lens):
        if n == 0:
            continue
        w = t.get("rel_weights")
        ws.append(w.to(device) if w is not None
                  else torch.ones(n, device=device))
    return b_ids, rels.to(device), torch.cat(ws)


def build_slot_targets(
    sub_idx: torch.Tensor,       # [B, K]
    obj_idx: torch.Tensor,       # [B, K]
    valid_mask: torch.Tensor,    # [B, K]
    targets: List[dict],
    V: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Multi-hot GT per slot from raw relations (NOT from pred_labels, which
    keeps only the last predicate the sampler saw for a multi-predicate pair).

    Fully batched — the earlier per-image loop cost two host syncs per image
    (`.any()` early-outs), ~13% of step time at bs128.

    Returns:
        slot_multi_hot [B, K, V] bool
        slot_weight    [B, K]    float — mean rel_weight of the slot's GT
                       relations (geometric-source downweighting), 1 if none.
    """
    B, K = sub_idx.shape
    device = sub_idx.device
    multi_hot = torch.zeros(B * K, V, dtype=torch.bool, device=device)
    slot_w = torch.ones(B * K, device=device)

    b_ids, rels, rw = _cat_relations(targets, device)
    if rels.numel():
        lookup = _flat_pair_lookup(sub_idx, obj_idx, valid_mask)
        slot = lookup((b_ids << 20) + rels[:, 0] * 1024 + rels[:, 1])  # [Rt]
        hit = slot >= 0
        multi_hot[slot[hit], rels[hit, 2]] = True
        slot_w.scatter_reduce_(0, slot[hit], rw[hit],
                               reduce="mean", include_self=False)
        slot_w.clamp_(min=1e-3)

    return multi_hot.view(B, K, V), slot_w.view(B, K)


def swap_direction_hinge(
    queries: List[tuple],        # [(q_sem, q_spa|None), ...] per head; q_* [B, K, D]
    alpha: Optional[torch.Tensor],  # [V] dual-head routing (None = single query)
    W: torch.Tensor,             # [V, D] normalized text embeddings
    sub_idx: torch.Tensor,       # [B, K]
    obj_idx: torch.Tensor,       # [B, K]
    valid_mask: torch.Tensor,    # [B, K]
    targets: List[dict],
    inverse_mask: torch.Tensor,  # [V, V] bool (on device)
    margin: float = 0.05,
    sym: Optional[torch.Tensor] = None,   # [V] P(reciprocal) — soft mode
) -> torch.Tensor:
    """Cross-slot direction hinge: relu(m + cos(bwd, g) - cos(fwd, g)).

    The swap probe showed direction knowledge is intact WITHIN a pair
    (InvTop 0.90-0.93) but scores are not comparable ACROSS the two directed
    slots (SwapAcc ~0.5): nothing in the batch-local InfoNCE ever pushes
    score(o,s,g) below score(s,o,g) — unannotated swapped slots simply are
    not anchors. This hinge adds that constraint directly on cosine scale,
    for every GT relation whose swapped slot was force-included.

    Eligibility: semantic predicates (alpha<=0.5) and spatial predicates
    WITH a defined inverse. Symmetric spatial predicates (near/beside — no
    inverse) are excluded: their annotation direction is arbitrary.

    Fully batched — no per-image loop, no host syncs. Ineligible / unmatched
    relations contribute 0 via masking, so the graph shape is data-driven
    but the host never blocks on the GPU.
    """
    device = W.device
    b_ids, rels, _ = _cat_relations(targets, device)
    if rels.numel() == 0:
        return W.new_zeros(())
    g = rels[:, 2]

    lookup = _flat_pair_lookup(sub_idx, obj_idx, valid_mask)
    f_slot = lookup((b_ids << 20) + rels[:, 0] * 1024 + rels[:, 1])
    b_slot = lookup((b_ids << 20) + rels[:, 1] * 1024 + rels[:, 0])
    keep = (f_slot >= 0) & (b_slot >= 0)                          # [Rt]
    # Per-instance reciprocal skip — EXACT, no estimation: when (o,s,g) is ALSO
    # annotated in this image, both directions are true, the annotation order is
    # arbitrary, and the two relations' hinges are pure gradient conflict
    # (measured: 12,310 reversed `looking at` pairs, 5,684 `next to`). Skip both.
    V_bits = int(W.shape[0]).bit_length()
    key_f = (((b_ids << 20) + rels[:, 0] * 1024 + rels[:, 1]) << V_bits) | g
    key_b = (((b_ids << 20) + rels[:, 1] * 1024 + rels[:, 0]) << V_bits) | g
    keep &= ~torch.isin(key_b, key_f)
    # clamp for safe gathering; masked rows are zeroed below
    f_slot = f_slot.clamp(min=0)
    b_slot = b_slot.clamp(min=0)

    w_g = W[g]                                                    # [Rt, D]
    a_g = alpha[g].unsqueeze(-1) if alpha is not None else None
    if sym is not None:
        # v42: direction supervision for every predicate, weighted by how
        # NON-reciprocal it measures in the data (reverse-annotation EM).
        # Replaces the two-clause eligibility rule, whose alpha<=0.5 branch
        # wrongly hinged mutual predicates (`kissing` alpha=0.131) and only
        # protected `hugging` by a routing accident (alpha=0.739).
        keep_f = keep.float() * (1.0 - sym[g])
    else:
        eligible = inverse_mask[g].any(-1)
        if alpha is not None:
            eligible |= alpha[g] <= 0.5
        keep_f = (keep & eligible).float()

    def _cos(q_pair, slots):
        q_s, q_p = q_pair
        D = q_s.shape[-1]
        flat_s = q_s.reshape(-1, D)
        c = (F.normalize(flat_s[slots], dim=-1) * w_g).sum(-1, keepdim=True)
        if q_p is not None and a_g is not None:
            flat_p = q_p.reshape(-1, D)
            cp = (F.normalize(flat_p[slots], dim=-1) * w_g).sum(-1, keepdim=True)
            c = (1.0 - a_g) * c + a_g * cp
        return c.squeeze(-1)                                      # [Rt]

    total = W.new_zeros(())
    denom = keep_f.sum().clamp(min=1.0)
    for q_pair in queries:
        hinge = F.relu(margin + _cos(q_pair, b_slot) - _cos(q_pair, f_slot))
        total = total + (hinge * keep_f).sum() / denom
    return total / max(len(queries), 1)


class SynonymAwareRelLoss(nn.Module):
    """Masked sigmoid multi-label focal loss over the open vocabulary.

    Args:
        ontology:    PredicateOntology with pos/ignore/inverse masks.
        neg_weight:  Weight for unlabeled negatives (positive-unlabeled
                     correction; plan D2). Inverse-of-GT negatives always get
                     weight 1.
        focal_gamma: Focusing parameter (0 disables modulation).
        focal_alpha: Foreground weight in [0,1]; 0.25 as in RelSGGLoss.
    """

    def __init__(
        self,
        ontology: PredicateOntology,
        neg_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        swap_negatives: bool = True,
    ) -> None:
        super().__init__()
        self.ont = ontology
        self.neg_weight = neg_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        # ARO-style directional supervision: on slot (j,i), the predicates of
        # GT relations (i,j,g) become FULL-weight negatives ("horse riding
        # man" is a certain negative, not an unlabeled one) — unless they are
        # positives/ignored there (symmetric predicates are annotated in both
        # directions and land in pos first). No-op when the sampler never
        # selects swapped pairs.
        self.swap_negatives = swap_negatives
        # Deliberately NOT registered as buffers: ModelEMA copies every buffer
        # each step and deepcopy would duplicate ~300MB of V×V masks on GPU.
        # Plain attributes + lazy device move instead.
        self.pos_mask = ontology.pos_mask
        self.ignore_mask = ontology.ignore_mask
        self.inverse_mask = ontology.inverse_mask
        self.class_weight = ontology.class_weight

    def _ensure_device(self, device: torch.device) -> None:
        if self.pos_mask.device != device:
            self.pos_mask = self.pos_mask.to(device)
            self.ignore_mask = self.ignore_mask.to(device)
            self.inverse_mask = self.inverse_mask.to(device)
            self.class_weight = self.class_weight.to(device)

    def forward(
        self,
        logits: torch.Tensor,       # [B, K, V]
        valid_mask: torch.Tensor,   # [B, K]
        sub_idx: torch.Tensor,      # [B, K]
        obj_idx: torch.Tensor,      # [B, K]
        targets: List[dict],
    ) -> Dict[str, torch.Tensor]:
        B, K, V = logits.shape
        self._ensure_device(logits.device)
        gt_hot, slot_w = build_slot_targets(sub_idx, obj_idx, valid_mask,
                                            targets, V)   # [B,K,V], [B,K]

        # Expand GT to canonical groups (multi-positive); collect per-slot
        # ignore / inverse sets as the union over the slot's GT predicates.
        # Gather only the GT rows of the V×V masks and scatter-accumulate —
        # avoids materialising float V×V matmuls every step.
        def expand(hot_bkv: torch.Tensor, mask_vv: torch.Tensor) -> torch.Tensor:
            acc = torch.zeros(B * K, V, dtype=torch.uint8, device=logits.device)
            hot = hot_bkv.view(B * K, V).nonzero(as_tuple=True)
            if hot[0].numel():
                acc.index_put_((hot[0],), mask_vv[hot[1]].to(torch.uint8),
                               accumulate=True)
            return (acc > 0).view(B, K, V)

        pos = expand(gt_hot, self.pos_mask)             # [B, K, V]
        ign = expand(gt_hot, self.ignore_mask)
        inv = expand(gt_hot, self.inverse_mask)
        ign &= ~pos
        inv &= ~pos

        has_gt = gt_hot.any(-1) & valid_mask            # [B, K]

        tgt = pos.float()
        w = torch.full_like(tgt, self.neg_weight)
        w = torch.where(pos, torch.ones_like(w), w)
        w = torch.where(inv, torch.ones_like(w), w)     # inverses: full-weight negatives
        if self.swap_negatives:
            # Predicates of the REVERSED pair's GT relations — and their
            # synonym neighborhoods — are certain negatives here ("horse
            # riding man"), so they override both neg_weight AND the ignore
            # mask. Symmetric predicates are annotated in both directions,
            # land in pos, and are excluded.
            swap_hot, _ = build_slot_targets(obj_idx, sub_idx, valid_mask,
                                             targets, V)
            swap_neg = (expand(swap_hot, self.pos_mask)
                        | expand(swap_hot, self.ignore_mask)) & ~pos
            w = torch.where(swap_neg, torch.ones_like(w), w)
            ign &= ~swap_neg
        w = torch.where(ign, torch.zeros_like(w), w)    # near-synonyms: ignored

        # Slots with no GT: every predicate is an unlabeled negative.
        w = w * torch.where(has_gt, slot_w, torch.ones_like(slot_w)).unsqueeze(-1)
        w = w * valid_mask.unsqueeze(-1)

        # Long-tail class weighting on the positive entries
        w = torch.where(pos, w * self.class_weight.unsqueeze(0).unsqueeze(0), w)

        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
        if self.focal_gamma > 0:
            p_t = tgt * p + (1 - tgt) * (1 - p)
            alpha_t = tgt * self.focal_alpha + (1 - tgt) * (1 - self.focal_alpha)
            ce = alpha_t * (1 - p_t).pow(self.focal_gamma) * ce

        n_valid = valid_mask.float().sum().clamp(min=1.0)
        loss = (ce * w).sum() / n_valid

        return {
            "loss": loss,
            "loss_cls": loss.detach(),
            "n_gt_slots": has_gt.float().sum().detach(),
        }


class MultiPositiveInfoNCE(nn.Module):
    """InfoNCE with canonical-group positives and synonym-masked denominator.

    positives  = logsumexp over the GT's canonical group,
    denominator = all predicates except tau-ignored near-synonyms
                  (inverses always stay in — they are the hard negatives that
                  carry directionality).
    """

    def __init__(self, ontology: PredicateOntology, temp: float = 0.07) -> None:
        super().__init__()
        self.temp = temp
        # Plain attributes (not buffers) — see SynonymAwareRelLoss.__init__.
        self.pos_mask = ontology.pos_mask
        self.ignore_mask = ontology.ignore_mask

    def forward(
        self,
        feats: torch.Tensor,        # [M, D] visual features projected to text dim
        W: torch.Tensor,            # [V, D] normalised text embeddings
        labels: torch.Tensor,       # [M] GT predicate index
    ) -> torch.Tensor:
        if feats.numel() == 0:
            return feats.new_zeros(1).squeeze()
        if self.pos_mask.device != feats.device:
            self.pos_mask = self.pos_mask.to(feats.device)
            self.ignore_mask = self.ignore_mask.to(feats.device)
        feats = F.normalize(feats, dim=-1)
        logits = feats @ W.T / self.temp                # [M, V]

        pos = self.pos_mask[labels]                     # [M, V]
        ign = self.ignore_mask[labels] & ~pos           # [M, V]

        neg_inf = torch.finfo(logits.dtype).min
        pos_lse = logits.masked_fill(~pos, neg_inf).logsumexp(-1)
        den_lse = logits.masked_fill(ign, neg_inf).logsumexp(-1)
        return (den_lse - pos_lse).mean()


class BatchLocalInfoNCE(nn.Module):
    """GLIP/YOLO-World-style batch-local region-text contrastive loss.

    Instead of supervising cosine logits against all V=10K text embeddings
    (a "10K-way distribution" no batch can meaningfully constrain), each step
    contrasts only against a small set built from the batch:

        S = unique GT predicates in the batch
          ∪ sampled HARD negatives — text-space neighbors of those predicates
            with cosine in (hard_lo, tau_ignore): confusable but known-distinct
          ∪ spatial inverses of the batch predicates (always in — they carry
            the directionality signal the text space itself lacks)
          ∪ uniform random fill up to ``n_neg``

    tau-synonyms of each anchor's GT (ignore_mask) are removed from its
    denominator; positives are the anchor's whole canonical group ∩ S.
    The full vocabulary is used only at eval/inference (cosine ranking).

    Hard/soft negatives (v34): with a pair co-occurrence table
    (training/build_pair_cooc.py), negatives whose predicate group was ever
    observed between the anchor's (subject-category, object-category) pair —
    or whose category pair has too little support to judge — are SOFT:
    plausibly-true-but-unlabeled (the GT is positive-unlabeled), down-weighted
    by ``soft_neg_weight`` in the denominator. Never-seen-with-support
    negatives are HARD (full weight). Positive columns and spatial-inverse
    columns are always exempt from softening (inverses carry the
    directionality signal and must stay full-weight negatives).

    Args:
        ontology:  PredicateOntology (pos/ignore/inverse masks).
        temp:      InfoNCE temperature.
        n_neg:     Total sampled negatives added to the batch's own classes.
        hard_frac: Fraction of ``n_neg`` drawn from hard-negative pools
                   (rest uniform over V).
        hard_lo:   Lower cosine bound for the hard-negative pool.
        cooc_path: Optional pair_cooc npz (build_pair_cooc.py) enabling
                   hard/soft weighting; forward() then expects ``pair_cats``.
        soft_neg_weight: Denominator weight of soft negatives (log-additive).
        min_support: Min train relations for a category pair before its
                   never-seen predicates count as HARD.
    """

    def __init__(
        self,
        ontology: PredicateOntology,
        temp: float = 0.07,
        n_neg: int = 256,
        hard_frac: float = 0.5,
        hard_lo: float = 0.5,
        cooc_path: Optional[str] = None,
        soft_neg_weight: float = 0.3,
        min_support: int = 30,
        pos_agg: str = "lse",
        pos_member_weight: float = 0.0,
    ) -> None:
        super().__init__()
        assert pos_agg in ("lse", "mean"), pos_agg
        self.pos_agg = pos_agg
        self.pos_member_weight = float(pos_member_weight)
        self.temp = temp
        self.n_neg = n_neg
        self.hard_frac = hard_frac
        self.hard_lo = hard_lo
        self.tau_ignore = ontology.tau_ignore
        # Plain attrs (not buffers) — EMA/deepcopy must not duplicate V×V masks.
        self.pos_mask = ontology.pos_mask
        self.ignore_mask = ontology.ignore_mask
        self.inverse_mask = ontology.inverse_mask
        # SOFT-SUPERVISION MODE (v42): estimated continuous weights replace the
        # banded machinery. See PredicateOntology.from_soft_supervision.
        self.soft = bool(getattr(ontology, "soft", False))
        if self.soft:
            self.pos_w = ontology.pos_w        # [V, V] fp16, diag 1
            self.neg_lw = ontology.neg_lw      # [V, V] fp16, log(1-p)
            soft_neg_weight = ontology.w_cooc  # fitted, replaces the hand 0.3

        self.soft_neg_weight = soft_neg_weight
        self.min_support = min_support
        # diagnostics, set each forward — kept as a 0-dim TENSOR (not
        # float(...)) so this assignment doesn't force a host sync / graph
        # break under torch.compile; callers .detach()/.item() it themselves.
        self.last_soft_frac: torch.Tensor = torch.tensor(0.0)
        if cooc_path:
            z = np.load(cooc_path)
            self.cooc_bits = torch.from_numpy(z["bits"])              # [R, nB] u8
            self.cooc_row = torch.from_numpy(z["row_of_pair"]).long() # [C*C]
            self.cooc_count = torch.from_numpy(z["pair_count"]).long()
            # the npz's OWN predicate→group mapping (id order differs from
            # ontology.group_of — never mix them)
            self.cooc_group_of = torch.from_numpy(z["group_of"]).long()
            self.cooc_C = int(z["num_cats"])
        else:
            self.cooc_bits = None

    def _pos_term(
        self,
        logits: torch.Tensor,     # [R, n_S]
        den_lse: torch.Tensor,    # [R]
        pos: torch.Tensor,        # [R, n_S] bool — the GT's canonical group ∩ S
        gt: torch.Tensor,         # [R] GT predicate id per row
        S: torch.Tensor,          # [n_S]
        neg_inf: float,
    ) -> torch.Tensor:
        """den_lse - (positive mass). Returns [R].

        WHY THERE ARE TWO AGGREGATIONS. v39 and v40 got exactly half each of what is
        needed, and the halves are measurable and opposite:

        `lse` (v34-v39) — logsumexp over the whole canonical group. Satisfied by ONE
        member, so the model aligns to a REGION of text space rather than a point. That
        region alignment is what survives re-parameterising the head onto a benchmark's
        own phrasing, which is why it transfers: VG150 mR@50 0.1433, PSG 0.1335,
        IndoorVG 0.1970. But nothing forces the GT string itself to be the peak within
        the region, so the non-chosen members are buried — median rank 338/1972 for
        `in front of`, 262 for `next to`, 148 for `to the left of`.

        `mean` with pos_member_weight=0 (v40) — identity positives. Fixes the ranking
        completely (those three go to 0, 2, 0; top-20 from ~0% to 91-98%) and costs the
        tail everywhere, because a point cannot generalise across phrasings: mR@50
        -11% / -10% / -18% and rare-class recall -33% / -51% / -68%. Micro R@50 barely
        moves, which is what localises the damage to tail transfer specifically.

        `mean` with 0 < pos_member_weight < 1 — every group member is pulled up (region
        preserved, tail supervision pooled) while the GT string is pulled hardest
        (weight 1, so it becomes the peak). This is the formulation that can have both;
        pos_member_weight is the dial between the two measured endpoints.
        """
        if self.pos_agg == "lse":
            return den_lse - logits.masked_fill(~pos, neg_inf).logsumexp(-1)
        # Weighted mean of per-member InfoNCE terms: each positive must beat the
        # SHARED denominator on its own, exactly as the multi-label branch already
        # requires of each GT predicate. No max anywhere, so no winner-take-all.
        if self.soft:
            # v42: per-member weights are the fitted kernel P(synonym | cos_v2) —
            # coherent members pull hard, incoherent ones barely. The GT column
            # is 1.0 by construction (pos_w diagonal), no masked_fill needed.
            w = self.pos_w[gt][:, S].to(logits.dtype)
        else:
            w = pos.to(logits.dtype) * self.pos_member_weight
            w = w.masked_fill(S.unsqueeze(0) == gt.unsqueeze(1), 1.0)
        per = den_lse.unsqueeze(-1) - logits                      # [R, n_S]
        return (per * w).sum(-1) / w.sum(-1).clamp(min=1e-6)

    def _ensure_device(self, device: torch.device) -> None:
        if self.pos_mask.device != device:
            self.pos_mask = self.pos_mask.to(device)
            self.ignore_mask = self.ignore_mask.to(device)
            self.inverse_mask = self.inverse_mask.to(device)
            if self.soft:
                self.pos_w = self.pos_w.to(device)
                self.neg_lw = self.neg_lw.to(device)
        if self.cooc_bits is not None and self.cooc_bits.device != device:
            self.cooc_bits = self.cooc_bits.to(device)
            self.cooc_row = self.cooc_row.to(device)
            self.cooc_count = self.cooc_count.to(device)
            self.cooc_group_of = self.cooc_group_of.to(device)

    @torch.no_grad()
    def _soft_neg_mask(
        self, pair_cats: torch.Tensor, S: torch.Tensor
    ) -> torch.Tensor:
        """[M, n_S] bool — True where the S-column is a SOFT negative for the
        anchor (predicate group seen between the anchor's category pair, or
        pair support below min_support, or categories unknown). Positive- and
        inverse-column exemptions are applied by the caller."""
        sub, obj = pair_cats[:, 0], pair_cats[:, 1]
        valid = (sub >= 0) & (obj >= 0)
        key = (sub.clamp(min=0) * self.cooc_C + obj.clamp(min=0))
        row = self.cooc_row[key]                                      # [M]
        count = self.cooc_count[key]
        low_support = ~valid | (count < self.min_support)             # [M]

        g = self.cooc_group_of[S]                                     # [n_S]
        byte_idx, bit_idx = g >> 3, g & 7
        rows = self.cooc_bits[row]                                    # [M, nB]
        seen = ((rows[:, byte_idx].long() >> bit_idx) & 1).bool()     # [M, n_S]
        return seen | low_support.unsqueeze(-1)

    @torch.no_grad()
    def build_set(self, labels: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Return the contrast set S (unique predicate ids, [n_S])."""
        V = W.shape[0]
        device = labels.device
        classes = labels.unique()                                    # [U]

        parts = [classes]
        inv = self.inverse_mask[classes].any(0).nonzero().flatten()
        parts.append(inv)

        n_hard = int(self.n_neg * self.hard_frac)
        if n_hard > 0:
            sim = W[classes] @ W.T                                   # [U, V]
            if self.soft:
                # v42, band-free: the most confusable columns that are SAFELY
                # false — cosine rank discounted by the estimated P(also-true).
                # No (hard_lo, tau_ignore) window: unsafe columns are already
                # down-weighted in the denominator, so sampling them is merely
                # wasteful, and the (1-p) factor removes the waste.
                score = sim.amax(0)
                score = score + self.neg_lw[classes].amin(0).float()
                score = score.masked_fill(self.pos_mask[classes].any(0), -2.0)
                parts.append(score.topk(min(n_hard, V)).indices)
            else:
                pool = ((sim > self.hard_lo) & (sim < self.tau_ignore)).any(0)
                pool &= ~self.pos_mask[classes].any(0)
                pool_idx = pool.nonzero().flatten()
                if pool_idx.numel():
                    take = torch.randperm(pool_idx.numel(), device=device)[:n_hard]
                    parts.append(pool_idx[take])

        n_rand = self.n_neg - n_hard
        if n_rand > 0:
            parts.append(torch.randint(0, V, (n_rand,), device=device))

        return torch.cat(parts).unique()

    def forward(
        self,
        feats: torch.Tensor,   # [M, D] projected visual queries (pre-normalize ok)
        labels: torch.Tensor,  # [M] GT predicate ids  OR  [M, V] bool multi-hot
        W: torch.Tensor,       # [V, D] frozen normalized text embeddings
        feats_spa: Optional[torch.Tensor] = None,  # [M, D] spatial-expert queries
        alpha: Optional[torch.Tensor] = None,      # [V] per-predicate routing
        weights: Optional[torch.Tensor] = None,    # [M] per-anchor loss weights
        pair_cats: Optional[torch.Tensor] = None,  # [M, 2] (sub_cat, obj_cat),
                                                   # -1 unknown; enables the
                                                   # cooc hard/soft weighting
        col_allow: Optional[torch.Tensor] = None,  # [M, V] bool: columns this
                                                   # anchor may be contrasted
                                                   # against (source-aware
                                                   # negative masking); None =
                                                   # all. Positives never masked.
    ) -> torch.Tensor:
        if feats.numel() == 0:
            return feats.new_zeros(())
        self._ensure_device(feats.device)

        # Multi-label anchors: a (sub,obj) pair routinely carries several GT
        # predicates ("holding" + "looking at"); single-label supervision made
        # the extras negatives of each other. Positives = union of all GT
        # predicates' canonical groups; ignore = union of their τ-synonym
        # neighborhoods.
        multi = labels.dim() == 2
        flat = labels.nonzero(as_tuple=True)[1] if multi else labels

        S = self.build_set(flat, W)                                  # [n_S]
        # Source-aware negative masking: an anchor from a single-vocabulary
        # source (HICO: one verb per pair, spatial relations never annotated)
        # must not push down columns its source could never have labelled —
        # "riding" is not evidence against "on"/"above". Disallowed columns
        # leave the denominator entirely (ignore), positives are kept.
        allow_S = col_allow[:, S] if col_allow is not None else None   # [M, n_S]
        feats = F.normalize(feats, dim=-1)
        cos = feats @ W[S].T                                         # [M, n_S]
        if feats_spa is not None:
            # Dual-projection head: per-predicate expert mixture. The gate
            # routes gradients automatically — spatial-predicate columns
            # (alpha≈1) train the spatial expert, semantic columns (alpha≈0)
            # the semantic one; no explicit family split of the batch needed.
            a = alpha[S]
            cos = (1.0 - a) * cos + a * (F.normalize(feats_spa, dim=-1) @ W[S].T)
        logits = cos / self.temp                                     # [M, n_S]

        neg_inf = torch.finfo(logits.dtype).min

        # Cooc hard/soft negative weighting: additive log-weight on the
        # DENOMINATOR logits only (pos_lse stays on raw logits). Applied on
        # logits, not cos — adding to cos before the temperature division
        # would exponentiate the weight by 1/temp into a de-facto hard mask.
        soft = None
        if self.cooc_bits is not None and pair_cats is not None:
            soft = self._soft_neg_mask(pair_cats, S)                  # [M, n_S]

        if multi:
            # Per-positive-group aggregation: one row per (anchor, GT pred).
            # A single logsumexp over the UNION of positives is satisfied by
            # the easiest one — on pairs carrying a frequent + a tail
            # predicate, the frequent one absorbs the whole gradient and the
            # tail predicate starves (measured: v3 union-lse cost -21% tail
            # SoftmR vs single-label). Each GT predicate's canonical group
            # must beat the shared denominator on its own.
            a_idx, g_idx = labels.nonzero(as_tuple=True)             # [M']
            hot = labels.float()                                     # [M, V]
            pos_any = (hot @ self.pos_mask[:, S].float()) > 0
            inv_any = (hot @ self.inverse_mask[:, S].float()) > 0
            den_logits = logits
            if self.soft:
                # v42: continuous denominator weights. Summing log(1-p) over the
                # anchor's GT rows IS the independence union p_u = 1-∏(1-p_g).
                # Positives and inverses are exempt (inverses carry direction and
                # must stay full-weight; positives are handled by _pos_term).
                lw = (hot @ self.neg_lw[:, S].float())
                lw = lw.masked_fill(pos_any | inv_any, 0.0)
                den_logits = den_logits + lw
                ign = torch.zeros_like(pos_any)
            else:
                ign = ((hot @ self.ignore_mask[:, S].float()) > 0)   # [M, n_S]
                ign &= ~pos_any
            if soft is not None:
                soft = soft & ~pos_any & ~inv_any
                self.last_soft_frac = soft.float().mean()
                den_logits = den_logits + soft * math.log(self.soft_neg_weight)
            if allow_S is not None:
                ign = ign | (~allow_S & ~pos_any)
            den_lse = den_logits.masked_fill(ign, neg_inf).logsumexp(-1)  # [M]

            pos_g = self.pos_mask[g_idx][:, S]                       # [M', n_S]
            loss = self._pos_term(logits[a_idx], den_lse[a_idx], pos_g,
                                  g_idx, S, neg_inf)                 # [M']
            # keep per-anchor mass constant: split the anchor's weight
            # across its labels
            n_lab = labels.sum(-1).clamp(min=1)                      # [M]
            w = (weights if weights is not None
                 else torch.ones_like(n_lab, dtype=loss.dtype))
            w_row = (w / n_lab)[a_idx]
            return (loss * w_row).sum() / w_row.sum().clamp(min=1e-6)

        pos = self.pos_mask[labels][:, S]                            # [M, n_S]
        inv_cols = self.inverse_mask[labels][:, S]
        den_logits = logits
        if self.soft:
            lw = self.neg_lw[labels][:, S].float()
            lw = lw.masked_fill(pos | inv_cols, 0.0)
            den_logits = den_logits + lw
            ign = torch.zeros_like(pos)
        else:
            ign = self.ignore_mask[labels][:, S] & ~pos
        if soft is not None:
            soft = soft & ~pos & ~inv_cols
            self.last_soft_frac = soft.float().mean()
            den_logits = den_logits + soft * math.log(self.soft_neg_weight)
        if allow_S is not None:
            ign = ign | (~allow_S & ~pos)
        den_lse = den_logits.masked_fill(ign, neg_inf).logsumexp(-1)
        # every anchor's own class is in S by construction → pos never empty
        loss = self._pos_term(logits, den_lse, pos, labels, S, neg_inf)  # [M]
        if weights is not None:
            return (loss * weights).sum() / weights.sum().clamp(min=1e-6)
        return loss.mean()
