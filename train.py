#!/usr/bin/env python3
"""RelAnything — SGCls training entry point.

Uses ground-truth boxes as oracle input; only relation prediction is
evaluated (SGCls task).  Add new datasets by registering them in
``build_datasets()``.

Usage — single GPU:
    python train.py --dataset vg150 \\
        --data_root /path/to/VG150_coco_format \\
        --output_dir ./runs/exp1 \\
        --backbone_type dinov3 --lora_rank 8

Usage — multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train.py --dataset vg150 ...
"""

from __future__ import annotations

import argparse
import datetime
import functools
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent))

from data import collate_fn
from data.relation_dataset import RelationDataset
from relsgg.embed_analyzer import EmbeddingAnalyzer
from relsgg.loss_synonym import PredicateOntology
from relsgg.evaluator import (FanoutEvaluator, KeySwapEvaluator,
                              SGClsEvaluator, SoftSGClsEvaluator,
                              build_match_matrix)
from relsgg.model import RelSGG, RelSGGConfig
from relsgg.train_engine import (collect_embeddings, evaluate, evaluate_loss,
                                 train_one_epoch)


# ==========================================================================
# Dataset factory
# ==========================================================================

def _load_exclude_ids(path: Optional[str]) -> Optional[set]:
    """Read a held-out image list (see training/build_indoorvg_holdout.py).

    Accepts the {"stems": [...]} payload that script writes, or a bare list.
    """
    if not path:
        return None
    obj = json.load(open(path))
    stems = obj["stems"] if isinstance(obj, dict) else obj
    stems = {str(s) for s in stems}
    print(f"[exclude] {len(stems):,} held-out image stems from {path}")
    return stems


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, List[str]]:
    """Return ``(train_dataset, val_dataset, predicate_names)``.

    Single-source: any root with ``train/`` + ``val/``. Multi-source mixture
    (``--data_roots``): a ConcatDataset over the union predicate vocab (order
    taken from the --pred_embeds npz so it matches W row-for-row) AND the union
    category vocab (taken from --ontology_meta's "categories" field, written by
    build_union_vocab.py — base categories first, then any --category_roots
    source not already covered); the base root's val split is shared.
    """
    if args.data_roots:
        from data.multipack import build_mixture_datasets
        z = np.load(args.pred_embeds)
        union_predicates = [str(p) for p in z["predicates"]]
        # OvR-SGG protocol arm (--drop_predicates). Removing a predicate STRING from
        # the union vocabulary is enough to hold it out completely: rel_cat_to_idx is
        # built from this list, RelationDataset drops every relation whose predicate is
        # absent (counted per source as n_rels_oov_dropped), and the vocabulary matrix
        # is matched to the dataset's predicate_names by name, so the head never gets a
        # row for it either. That is OvSGTR's "never seen with a label during training".
        #
        # The predicate is still scorable AT INFERENCE, because the vocabulary enters
        # as text at test time -- which is the whole open-vocabulary claim, and the
        # reason this arm is meaningful rather than merely smaller.
        if args.drop_predicates:
            drop = json.load(open(args.drop_predicates)) \
                if args.drop_predicates.endswith(".json") \
                else [x.strip() for x in args.drop_predicates.split(",") if x.strip()]
            drop = {str(d) for d in drop}
            missing = drop - set(union_predicates)
            kept = [p for p in union_predicates if p not in drop]
            n_removed = len(union_predicates) - len(kept)
            print(f"[drop_predicates] holding out {n_removed} of "
                  f"{len(union_predicates)} union predicates: "
                  f"{sorted(drop - missing)}")
            if missing:
                # Not fatal -- a corpus legitimately may not use a string -- but it
                # must be visible, because a typo here silently produces a model that
                # was NOT held out and a Novel number that is quietly contaminated.
                print(f"[drop_predicates] WARNING: {len(missing)} requested strings "
                      f"absent from the union vocabulary, nothing held out for them: "
                      f"{sorted(missing)}")
            # The invariant is NOT "this flag removed something" -- it is "none of
            # these predicates is in the final vocabulary". Both spellings of the
            # holdout satisfy it: filtering here, or pointing --pred_embeds at a bank
            # that was already rebuilt without them (which is required anyway, since
            # union_meta / soft_supervision are position-keyed to the same list --
            # see training/build_heldout_text_space.py). Enforcing the invariant
            # rather than the side effect keeps the flag idempotent, so it can stay
            # on the command line as provenance even when the bank is pre-filtered.
            still_present = drop & set(kept)
            if still_present:
                raise SystemExit(f"!! --drop_predicates left {sorted(still_present)} "
                                 "in the vocabulary")
            if not drop:
                raise SystemExit("!! --drop_predicates is empty; refusing to train a "
                                 "run that would be mislabelled as held-out")
            if not n_removed:
                print("[drop_predicates] all already absent from --pred_embeds; the "
                      "holdout is carried by the bank itself, flag kept as provenance")
            union_predicates = kept
        ontology_meta = (args.ontology_meta
                         or os.path.join(args.data_roots[0], "train", "meta.json"))
        union_categories = json.load(open(ontology_meta))["categories"]
        train_ds, val_ds, source_of_index, source_names = build_mixture_datasets(
            base_root=args.data_roots[0],
            extra_roots=args.data_roots[1:],
            union_predicates=union_predicates,
            resolution=args.img_size,
            max_objects=args.max_objects,
            union_categories=union_categories,
            val_root=args.val_root,
            augment=args.augment,
            geometric_weight=args.geometric_weight,
            drop_geometric_roots=args.drop_geometric,
            cat_aliases=(json.load(open(args.cat_aliases))
                         if args.cat_aliases else None),
            exclude_ids=_load_exclude_ids(args.exclude_ids),
            rasters=args.rasters,
            mask_dropout=args.mask_dropout,
        )
        # Data-scaling arms: restrict TRAIN to a deterministic random subset of
        # the images (val untouched). The seed is fixed and rank-independent so
        # every DDP rank selects the identical subset — otherwise the run would
        # silently see the whole corpus spread across ranks.
        if args.train_subset_frac < 1.0:
            n = len(train_ds)
            k = int(round(n * args.train_subset_frac))
            keep = np.sort(
                np.random.default_rng(args.train_subset_seed).permutation(n)[:k])
            train_ds = torch.utils.data.Subset(train_ds, keep.tolist())
            source_of_index = source_of_index[keep]
            print(f"[subset] train restricted to {k:,}/{n:,} images "
                  f"(frac={args.train_subset_frac}, seed={args.train_subset_seed})")

        # stash for the sampler + spatial-flags aggregation
        args._source_of_index = source_of_index
        args._source_names = source_names
        args._union_categories = union_categories
        return train_ds, val_ds, union_predicates

    train_ds = RelationDataset(
        root=args.data_root,
        split="train",
        resolution=args.img_size,
        max_objects=args.max_objects,
        augment=args.augment,          # train only — val must stay deterministic
        rasters=args.rasters,
        mask_dropout=args.mask_dropout,
    )
    val_ds = RelationDataset(
        root=args.data_root,
        split="val",
        resolution=args.img_size,
        max_objects=args.max_objects,
        # Share category mappings from train for consistency
        cat_to_idx=train_ds.cat_to_idx,
        rel_cat_to_idx=train_ds.rel_cat_to_idx,
        rasters=args.rasters,
    )
    return train_ds, val_ds, train_ds.predicate_names


def union_spatial_flags(roots: List[str], union_predicates: List[str]) -> np.ndarray:
    """Per-union-predicate spatial label = spatial-majority in ANY source.

    The FLAG_SPATIAL bit is near-deterministic per predicate WITHIN a source,
    but the other sources often leave it unset for names MEGASG marks spatial
    ("on"): pooling raw counts across sources would let a large unflagged
    source flip the vote. OR-ing per-source majorities keeps a predicate
    spatial if any source considers it so."""
    idx = {p: i for i, p in enumerate(union_predicates)}
    flag = np.zeros(len(union_predicates), dtype=np.int64)
    for r in roots:
        meta = json.load(open(os.path.join(r, "train", "meta.json")))
        local = meta["predicates"]
        rels = np.load(os.path.join(r, "train", "rels.npy"), mmap_mode="r")
        pid = np.asarray(rels[:, 2])
        sbit = (np.asarray(rels[:, 3]) & 1).astype(np.float64)
        lcnt = np.bincount(pid, minlength=len(local))
        lspa = np.bincount(pid, weights=sbit, minlength=len(local))
        maj = lspa >= 0.5 * np.maximum(lcnt, 1)
        for li, name in enumerate(local):
            ui = idx.get(name)
            if ui is not None and lcnt[li] > 0 and maj[li]:
                flag[ui] = 1
    return flag


# ==========================================================================
# Model / optimiser / scheduler helpers
# ==========================================================================

def fit_spatial_gate(W: np.ndarray, is_spatial: np.ndarray):
    """Fit the text-space spatialness direction for the dual-projection head.

    Balanced logistic regression (the positive class is ~21 predicates out of
    10K — unweighted fitting would push every alpha to ~0). Measured on this
    vocabulary: spatial-vs-semantic AUC 0.998 on dino.txt embeddings.

    Returns (u [D] tensor, b scalar tensor).
    """
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(W, is_spatial)
    u = torch.from_numpy(clf.coef_[0].astype(np.float32))
    b = torch.tensor(float(clf.intercept_[0]))
    alpha = torch.sigmoid(torch.from_numpy(W.astype(np.float32)) @ u + b)
    spa, sem = alpha[is_spatial == 1], alpha[is_spatial == 0]
    print(f"Spatial gate: {int(is_spatial.sum())} spatial preds | "
          f"alpha spatial {spa.mean():.3f} (min {spa.min():.3f}) | "
          f"alpha semantic {sem.mean():.3f} (max {sem.max():.3f}) | "
          f"ambiguous(0.2-0.8): {int(((alpha > 0.2) & (alpha < 0.8)).sum())}")
    return u, b


def predicate_spatial_flags(data_root: str, V: int) -> np.ndarray:
    """Per-predicate spatial label by majority vote over packed train
    relations (rel flags bit0). The flag is near-deterministic per predicate:
    6/1939 well-supported predicates are mixed."""
    rels = np.load(os.path.join(data_root, "train", "rels.npy"), mmap_mode="r")
    pred_id = np.asarray(rels[:, 2])
    spatial_bit = (np.asarray(rels[:, 3]) & 1).astype(np.float64)
    cnt = np.bincount(pred_id, minlength=V)
    spa = np.bincount(pred_id, weights=spatial_bit, minlength=V)
    return (spa >= 0.5 * np.maximum(cnt, 1)).astype(np.int64)


# Keys --init_from copies from the source checkpoint's args (see main()).
INIT_INHERIT_KEYS = [
    # everything build_model() reads
    "backbone_model", "backbone_type", "beta_relatedness", "bg_agg", "bg_topk",
    "box_token_dropout", "cfa_alpha", "cfa_mode", "cfa_partner", "cfa_prob",
    "compose_query", "deformable_border", "deformable_clamp", "deformable_gain",
    "deformable_heads", "deformable_nulls", "deformable_points", "deformable_ring",
    "deformable_v2", "depth_scaled_init", "d_model", "dropout", "drop_path",
    "dual_spatial_head", "fast_bilinear", "final_budget", "geo_budget", "geo_pu",
    "geo_squash", "lambda_bg", "lambda_czsc", "lambda_fast", "lambda_geo",
    "lambda_infonce", "lambda_rel", "lambda_sigmoid", "lambda_swap",
    "logit_bias_init", "logit_scale_init", "lora_layers", "lora_rank", "loss_type",
    "ms_deconv_level", "ms_depth_levels", "ms_pool_level", "n_cross_layers",
    "n_dep_layers", "n_gnd_layers", "norm_taps", "n_self_layers", "pe_max_octave",
    "pe_num_freqs", "pool_role_queries", "proj_layers", "rel_neg_weight",
    "role_obj_loss", "sampler_type", "scene_pe", "stage_s2d", "stage_weight_init",
    "swap_include", "swap_margin", "text_dim", "tucker_query", "use_rel_interaction",
    "gate_mlp", "region_adjacency", "contact_field", "mask_adapter_dim",
    # data view + recipe knobs that are part of "the same model"
    "text_student", "img_size", "max_objects", "static_shapes", "augment",
    "multi_scale", "multi_scale_n", "tau_eval", "pos_agg", "n_neg", "hard_lo",
    "soft_neg_weight", "min_support", "tau_ignore", "geometric_weight",
    "lambda_obj", "eval_budget",
]


def build_model(args, pred_names: List[str],
                obj_names: Optional[List[str]] = None) -> RelSGG:
    cfg = RelSGGConfig(
        backbone_type=args.backbone_type,
        lora_rank=args.lora_rank,
        lora_layers=args.lora_layers,
        **({"backbone_model": args.backbone_model} if args.backbone_model else {}),
        d_model=args.d_model,
        text_dim=args.text_dim,
        geo_budget=args.geo_budget,
        final_budget=args.final_budget,
        n_self_layers=args.n_self_layers,
        n_cross_layers=args.n_cross_layers,
        lambda_infonce=args.lambda_infonce,
        lambda_czsc=args.lambda_czsc,
        logit_scale_init=args.logit_scale_init,
        logit_bias_init=(args.logit_bias_init if args.logit_bias_init is not None
                         else (-10.0 if args.loss_type == "synonym" else 0.0)),
        sampler_type=args.sampler_type,
        rel_neg_weight=args.rel_neg_weight,
        neg_rate_table=args.neg_rate_table,
        swap_include=args.swap_include,
        proj_layers=args.proj_layers,
        # compose_query was historically DERIVED from loss_type, which made it
        # impossible to ablate independently of the loss. --compose_query now
        # overrides; None keeps the legacy coupling so prior recipes are
        # unchanged (batch_infonce runs, incl. v34, have always had this ON).
        compose_query=(args.compose_query if args.compose_query is not None
                       else (args.loss_type == "batch_infonce")),
        tucker_query=args.tucker_query,
        dropout=args.dropout,
        box_token_dropout=args.box_token_dropout,
        deformable_points=args.deformable_points,
        deformable_heads=args.deformable_heads,
        deformable_nulls=args.deformable_nulls,
        deformable_clamp=args.deformable_clamp,
        deformable_v2=args.deformable_v2,
        deformable_ring=args.deformable_ring,
        deformable_gain=args.deformable_gain,
        deformable_border=args.deformable_border,
        ms_depth_levels=args.ms_depth_levels,
        ms_pool_level=args.ms_pool_level,
        ms_deconv_level=args.ms_deconv_level,
        depth_scaled_init=args.depth_scaled_init,
        drop_path=args.drop_path,
        norm_taps=args.norm_taps,
        stage_s2d=args.stage_s2d,
        stage_weight_init=args.stage_weight_init,
        pe_num_freqs=args.pe_num_freqs,
        pe_max_octave=args.pe_max_octave,
        geo_squash=args.geo_squash,
        geo_pu=args.geo_pu,
        lambda_geo=args.lambda_geo,
        lambda_rel=args.lambda_rel,
        lambda_bg=args.lambda_bg,
        bg_topk=args.bg_topk,
        bg_agg=args.bg_agg,
        scene_pe=args.scene_pe,
        pool_role_queries=args.pool_role_queries,
        mode_gated=getattr(args, "mode_gated", False),
        region_adjacency=getattr(args, "region_adjacency", False),
        contact_field=getattr(args, "contact_field", False),
        mask_adapter_dim=getattr(args, "mask_adapter_dim", 0),
        role_obj_loss=args.role_obj_loss,
        dual_spatial_head=args.dual_spatial_head,
        beta_relatedness=args.beta_relatedness,
        lambda_sigmoid=args.lambda_sigmoid,
        fast_bilinear_head=args.fast_bilinear,
        lambda_fast=args.lambda_fast,
        lambda_swap=args.lambda_swap,
        swap_margin=args.swap_margin,
        use_rel_interaction=args.use_rel_interaction,
        n_dep_layers=args.n_dep_layers,
        n_gnd_layers=args.n_gnd_layers,
        cfa_mode=args.cfa_mode,
        cfa_prob=args.cfa_prob,
        cfa_alpha=args.cfa_alpha,
        cfa_partner=args.cfa_partner,
    )
    model = RelSGG(cfg)

    # ---- Vocabulary embeddings ----
    # Fail fast if masks were requested: a silent box-mode run that REPORTS as
    # a mask run is the worst outcome, and is exactly what happened in jobs
    # 6922420/6922421 (DDP scatter dropped the rasters; cov_lambda stayed
    # bit-exact 0.0 for 12 epochs).
    model.expect_region = bool(args.rasters)

    if args.pred_embeds:
        # Reuse the (template-ensembled) embeddings from text_space_diag.py —
        # identical to what the ontology masks were calibrated on, and no
        # text encoder load at job start.
        z = np.load(args.pred_embeds)
        npz_preds = [str(p) for p in z["predicates"]]
        assert npz_preds == list(pred_names), (
            f"{args.pred_embeds} predicate order != dataset vocabulary — "
            "regenerate with training/text_space_diag.py on this pack"
        )
        print(f"Installing precomputed W from {args.pred_embeds} "
              f"(templates={list(z['templates'])})")
        model.vocab_head.set_vocabulary_matrix(pred_names, z["embeddings"])
    elif args.dinotxt_weights:
        print(f"Encoding vocabulary with dino.txt  ({args.dinotxt_weights})")
        model.vocab_head.encode_vocabulary_dinotxt(
            pred_names,
            dinotxt_weights=args.dinotxt_weights,
            bpe_path_or_url=args.dinotxt_bpe,
            backbone_weights=None,  # text-only encoding; skip backbone download
        )
    else:
        model.encode_vocabulary(pred_names)

    # ---- Dual-projection spatial head: fit the text-side routing gate ----
    if args.dual_spatial_head and getattr(args, "init_from", ""):
        # Closed-set fine-tune: the probe (gate_u/gate_b) and the trained
        # gate_mlp come from the source checkpoint in apply_init_from(); the
        # benchmark packs carry no spatial flags to fit a probe on anyway.
        if args.gate_mlp:
            model.vocab_head.build_gate_mlp()
        print("[init_from] spatial gate: probe fit skipped, loaded from checkpoint")
    elif args.dual_spatial_head:
        if args.data_roots:
            is_spatial = union_spatial_flags(args.data_roots, pred_names)
        else:
            is_spatial = predicate_spatial_flags(args.data_root, len(pred_names))
        u, b = fit_spatial_gate(
            model.vocab_head.W.cpu().numpy(), is_spatial
        )
        model.vocab_head.set_spatial_gate(u, b)
        if args.gate_mlp:
            # v34: trainable MLP gate, warm-started from the probe so
            # training begins with the probe's routing. Must run BEFORE
            # optimizer/DDP/EMA creation (new parameters + baked buffer).
            model.vocab_head.build_gate_mlp()
            mse = model.vocab_head.warm_start_gate_mlp()
            a = model.vocab_head.alpha
            print(f"gate_mlp warm-start: mse={mse:.5f}  "
                  f"alpha>0.5: {int((a > 0.5).sum())}/{a.numel()}")

    # ---- Synonym-aware losses (plan D2; --loss_type ce = legacy ablation) ----
    if args.loss_type in ("synonym", "batch_infonce"):
        if args.canon_groups and not os.path.isfile(args.canon_groups):
            raise FileNotFoundError(
                f"--canon_groups {args.canon_groups!r} not found. Generate it "
                "with training/text_space_diag.py, pass '' for identity "
                "positives (the default), or use --loss_type ce."
            )
        if args.pred_context and not os.path.isfile(args.pred_context):
            raise FileNotFoundError(
                f"--pred_context {args.pred_context!r} not found. Build it with "
                "training/build_predicate_context.py --packs <train split(s)> "
                "--ontology_meta <the same meta this run uses>."
            )
        ontology_meta = (args.ontology_meta
                         or os.path.join(args.data_root, "train", "meta.json"))
        if args.soft_supervision:
            # v42: every semantic constant estimated (build_soft_supervision.py).
            # The legacy knobs are not read on this path — say so loudly rather
            # than silently ignoring them.
            ontology = PredicateOntology.from_soft_supervision(
                meta_path=ontology_meta, npz_path=args.soft_supervision)
            print("[soft-supervision] tau_ignore / hard_lo / neg_weight / "
                  "soft_neg_weight / tau_ctx* / --group_positives / "
                  "--pos_member_weight are NOT read in this mode — positives, "
                  "denominator weights, w_cooc and hinge eligibility all come "
                  f"from {args.soft_supervision}")
        else:
            ontology = PredicateOntology.from_artifacts(
                meta_path=ontology_meta,
                canon_groups_path=args.canon_groups or None,
                embeds_path=args.pred_embeds or None,
                tau_ignore=args.tau_ignore,
                context_path=args.pred_context or None,
                tau_ctx=args.tau_ctx,
                tau_ctx_floor=args.tau_ctx_floor,
                group_positives=args.group_positives,
            )
        print(f"Ontology: {ontology.stats()}")
        model.set_ontology(
            ontology, neg_weight=args.neg_weight,
            mode=("batch" if args.loss_type == "batch_infonce" else "vwide"),
            n_neg=args.n_neg, lambda_obj=args.lambda_obj,
            hard_lo=args.hard_lo,
            cooc_path=args.pair_cooc or None,
            soft_neg_weight=args.soft_neg_weight,
            min_support=args.min_support,
            pos_agg=args.pos_agg,
            pos_member_weight=args.pos_member_weight,
        )
        print(f"Positive aggregation: {args.pos_agg}"
              + (f" (member weight {args.pos_member_weight})"
                 if args.pos_agg == "mean" else " — logsumexp over the group"))
        if args.pair_cooc:
            print(f"Cooc hard/soft negatives: {args.pair_cooc} "
                  f"(soft_w={args.soft_neg_weight}, min_support={args.min_support})")

    # ---- Object-category text embeddings (compositional aux loss) ----
    # Cache is keyed by encoder AND dim-checked: a stale cache from another
    # encoder (e.g. 2048-d dinotxt when training in 768-d student space)
    # would otherwise load silently and crash _object_text_loss mid-step.
    if args.loss_type == "batch_infonce" and obj_names:
        _t_dim = cfg.text_dim if cfg.text_dim is not None else cfg.d_model
        # The tag must distinguish student CHECKPOINTS, not just student-vs-
        # dinotxt: v1 and v2 students are both 768-d over the same category
        # names, so neither assert below would catch a cross-load.
        enc_tag = (f"student-{os.path.basename(os.path.dirname(args.text_student))}"
                   if args.text_student else "dinotxt")
        # --canon_groups is optional now, so it can no longer be relied on to name
        # the text-space directory; fall back to wherever the embeddings live.
        # --soft_supervision leads the chain: in that mode canon_groups is unread
        # but still DEFAULTS to the megasg path, whose cached obj_embeds carries
        # the old category list (v42 smoke 6919683 died on exactly that).
        _ts_dir = os.path.dirname(args.soft_supervision or args.canon_groups
                                  or args.pred_embeds or args.ontology_meta or ".")
        cache = os.path.join(_ts_dir, f"obj_embeds_{enc_tag}.npz")
        if os.path.isfile(cache):
            zo = np.load(cache)
            assert [str(n) for n in zo["names"]] == list(obj_names), \
                f"stale {cache} — delete it to re-encode"
            W_obj = torch.from_numpy(zo["embeddings"].astype(np.float32))
        else:
            if args.text_student:
                from relsgg.text_student import encode_texts_student
                print(f"Encoding {len(obj_names)} object categories with "
                      f"the student ({args.text_student})")
                W_obj = encode_texts_student(
                    obj_names, args.text_student,
                    templates=["{p}", "a photo of a {p}"],
                )
            else:
                from relsgg.vocab import encode_texts_dinotxt
                print(f"Encoding {len(obj_names)} object categories with dino.txt")
                W_obj = encode_texts_dinotxt(
                    obj_names, args.dinotxt_weights,
                    templates=["{p}", "a photo of a {p}"],
                )
            np.savez_compressed(cache, embeddings=W_obj.numpy().astype(np.float16),
                                names=obj_names)
        assert W_obj.shape[1] == _t_dim, (
            f"{cache}: dim {W_obj.shape[1]} != text_dim {_t_dim} — "
            "delete the cache to re-encode with the current encoder"
        )
        model.set_object_vocabulary(obj_names, W_obj)
    return model


def inherit_init_args(args) -> None:
    """--init_from: copy the network + recipe keys from the source run's
    saved args (see INIT_INHERIT_KEYS). Data paths, LR/schedule, epochs and
    the vocabulary artefacts stay as given on the command line, so a
    fine-tune launch cannot retype the architecture wrong."""
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False,
                    mmap=True)
    ca = ck.get("args") or {}
    ca = ca if isinstance(ca, dict) else vars(ca)
    changed = []
    for k in INIT_INHERIT_KEYS:
        if k in ca and getattr(args, k, None) != ca[k]:
            changed.append((k, getattr(args, k, None), ca[k]))
            setattr(args, k, ca[k])
    print(f"[init_from] inherited {len(INIT_INHERIT_KEYS)} recipe keys from "
          f"{args.init_from}; overrides: "
          + (", ".join(f"{k}: {o!r}->{n!r}" for k, o, n in changed) or "none"))


#: The mode adapter (--mode_gated). A source checkpoint trained without it
#: lacks mode_embed / region_delta; they stay at their zero init, which is a
#: no-op, so their absence is not architecture drift.
MODE_ADAPTER_PREFIXES = ("spatial_pool.cov_lambda", "spatial_pool.mode_embed",
                         "geo_encoder.region_delta", "mask_adapter")


def apply_init_from(model: RelSGG, path: str, learn_W: bool = False) -> None:
    """Weights-only init from a finished run (EMA weights when present).

    Vocabulary-sized tensors (W, alpha, object-text banks) keep the values
    installed for THIS run's vocabulary; every other tensor must load
    strictly — any other missing/unexpected key means the inherited
    architecture drifted and the run aborts instead of training a different
    network. Must run BEFORE DDP/EMA/optimizer construction (learn_W adds a
    parameter).
    """
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck["ema_model"] if "ema_model" in ck else ck["model"]
    if "vocab_head.W_param" in sd:                      # source was itself learn_W
        sd = dict(sd)
        sd["vocab_head.W"] = F.normalize(sd.pop("vocab_head.W_param").float(), dim=-1)
    vocab_sized = ("vocab_head.W", "vocab_head.alpha", "vocab_head.beta")
    # Lazily-sized buffers (gate_u/gate_b, ...) are empty until first use:
    # resize them to the checkpoint's shape so they load (same trick as
    # benchmark/eval_zeroshot.py's builder). Vocabulary-sized ones stay.
    buffers = dict(model.named_buffers())
    for k, v in sd.items():
        if k in buffers and k not in vocab_sized and tuple(buffers[k].shape) != tuple(v.shape):
            mod = model
            *parts, leaf = k.split(".")
            for pth in parts:
                mod = getattr(mod, pth)
            # ON THE MODEL'S DEVICE: the checkpoint tensor is a CPU tensor and a
            # CPU buffer inside a CUDA model breaks DDP's initial state sync
            # (job 6967201). Invisible to the CPU smoke by construction.
            setattr(mod, leaf, torch.empty_like(v, device=buffers[k].device))
    own = model.state_dict()
    keep, dropped = {}, []
    for k, v in sd.items():
        if k in own and tuple(own[k].shape) == tuple(v.shape):
            keep[k] = v
        else:
            dropped.append(k)
    bad = [k for k in dropped
           if not (k in vocab_sized or "W_obj" in k or "obj_text" in k or k not in own)]
    unexpected_src = [k for k in dropped if k not in own]
    if bad or unexpected_src:
        raise RuntimeError(f"--init_from shape/key drift: dropped={bad} "
                           f"unexpected={unexpected_src}")
    missing = [k for k in own if k not in keep and k not in vocab_sized
               and not k.startswith(MODE_ADAPTER_PREFIXES)]
    adapter_fresh = [k for k in own if k not in keep
                     and k.startswith(MODE_ADAPTER_PREFIXES)]
    if adapter_fresh:
        print(f"[init_from] mode adapter absent from source, kept at zero init: "
              f"{adapter_fresh}")
    if missing:
        raise RuntimeError(f"--init_from: checkpoint lacks {missing[:8]}"
                           f"{' ...' if len(missing) > 8 else ''}")
    model.load_state_dict(keep, strict=False)
    with torch.no_grad():
        model.vocab_head._update_alpha()      # routing for THIS vocab from the loaded gate
    src = "ema_model" if "ema_model" in ck else "model"
    print(f"[init_from] loaded {len(keep)} tensors from {path} ({src}); "
          f"kept this run's vocabulary tensors {dropped}")
    if learn_W:
        model.vocab_head.make_W_trainable()
        print("[init_from] W is TRAINABLE (closed-set classifier ablation; "
              "state_dict carries vocab_head.W_param)")
    del ck


def build_optimizer(model: RelSGG, args) -> torch.optim.Optimizer:
    """AdamW with a lower LR for the backbone than the rest of the model.

    --backbone_layer_decay < 1.0 enables layer-wise LR decay (LLRD) over the
    ViT blocks: block i (of L) trains at backbone_lr * decay^(L-1-i), the
    embeddings at backbone_lr * decay^L, the final norm at backbone_lr. The
    flat backbone LR is the measured razor's edge of full FT (+8.0% at 1e-5,
    noise at 5e-5); LLRD widens it by moving late blocks — where the relation
    semantics live — faster than the generic early blocks.
    --backbone_weight_decay overrides WD for the backbone group(s); pretrained
    weights decayed toward zero is the wrong prior for a fine-tune.
    """
    backbone_params, other_params = [], []
    llrd_groups: dict[int, list] = {}
    decay = args.backbone_layer_decay
    n_blocks = None
    if decay < 1.0:
        idx = [int(m.group(1)) for n, _ in model.named_parameters()
               if (m := re.search(r"backbone\..*\.layer\.(\d+)\.", n))]
        n_blocks = max(idx) + 1 if idx else 0
    w_params: list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("vocab_head.W_param"):
            # learn_W: unit-row text matrix, head LR, never decayed (decay
            # would shrink the very rows the normalised view rescales).
            w_params.append(param)
            continue
        # `backbone.stage_proj.*` is the ConvNeXt multi-scale fusion: 1x1 convs
        # trained FROM SCRATCH, not pretrained weights being adapted. The name
        # match would hand them backbone_lr (1e-5), i.e. 40x too low for random
        # init, and the first ConvNeXt arm trained them that way — its -12% vs
        # LoRA measured a starved adapter, not the backbone family. They belong
        # with the rest of the freshly-initialised model at args.lr.
        # `backbone.stage_norm.*` is the ConvNeXt per-stage fusion LayerNorm —
        # ALSO from scratch, so it belongs with stage_proj for exactly the same
        # reason. It was missed on the first pass and the arms paid for it: this
        # is the generalised lesson from the stage_proj starvation restated —
        # ANY new from-scratch module added under an existing name prefix
        # inherits that prefix's LR group silently.
        #
        # `layer_weights` stays in the backbone group ON THE ViT PATH, where it
        # is a zero-init scalar combiner starting at an already-sane uniform and
        # where every run to date trained it there — moving it would break
        # comparability for no benefit. On the CONVNEXT path it is moved to the
        # head group, because there it is the PRIMARY EXPERIMENTAL READOUT (which
        # feature level does the task want?) and backbone_lr cannot answer it:
        # measured, the ViT combiner needs ~47K steps at 5e-5 to develop a real
        # preference (logits +-0.497 -> 0.202/0.261/0.537 on the full pack) and
        # still only reaches +-0.079 -> 0.310/0.329/0.361 on the 6K-step proxy.
        # A ConvNeXt proxy arm at 1e-5 has ~39x less total logit budget than that
        # full ViT run, so it CANNOT move regardless of what the model wants, and
        # reading "no preference" off it would be reading the LR.
        # No ViT run changes; the 8 prior ConvNeXt arms all sat at uniform under
        # the starved setting, so nothing comparable is lost either.
        # [[relsgg-convnext-fusion-flaw]]
        from_scratch_fusion = ("stage_proj" in name or "stage_norm" in name
                               or (name.endswith("backbone.layer_weights")
                                   and args.backbone_type == "dinov3_convnext"))
        is_backbone = "backbone" in name and not from_scratch_fusion
        if not is_backbone:
            other_params.append(param)
        elif decay >= 1.0 or n_blocks == 0:
            backbone_params.append(param)
        else:
            m = re.search(r"backbone\..*\.layer\.(\d+)\.", name)
            if m:
                depth = n_blocks - 1 - int(m.group(1))   # 0 at last block
            elif "embeddings" in name:
                depth = n_blocks                          # below block 0
            else:
                depth = 0   # final norm, layer_weights: top scale
            llrd_groups.setdefault(depth, []).append(param)

    bwd = (args.backbone_weight_decay
           if args.backbone_weight_decay is not None else args.weight_decay)

    # No-decay split: AdamW's DECOUPLED decay shrinks a parameter toward zero
    # every step regardless of its gradient, so decaying the zero-init gates
    # (deformable gamma, cov_lambda, compose_gate, layer_weights, logit_scale/
    # bias) systematically biased every "the gate stayed at 0, so the model
    # doesn't want the mechanism" reading. ndim<=1 catches all of those plus
    # every bias and LayerNorm affine — the standard ViT-recipe exclusion.
    # Content tensors (ndim>=2, incl. base_query/corner_bias/null_vec) keep WD.
    # --decay_all restores the legacy single-group behaviour; it is also
    # REQUIRED when resuming a checkpoint written before this split (the
    # optimizer state_dict group layout changed).
    groups: list = []

    def _emit(params: list, lr: float, wd: float) -> None:
        if getattr(args, "decay_all", False):
            if params:
                groups.append({"params": params, "lr": lr, "weight_decay": wd})
            return
        dec = [p for p in params if p.ndim > 1]
        nod = [p for p in params if p.ndim <= 1]
        if dec:
            groups.append({"params": dec, "lr": lr, "weight_decay": wd})
        if nod:
            groups.append({"params": nod, "lr": lr, "weight_decay": 0.0})

    _emit(other_params, args.lr, args.weight_decay)
    if w_params:
        groups.append({"params": w_params, "lr": args.lr, "weight_decay": 0.0})
        print(f"[optim] vocab_head.W_param trainable ({w_params[0].shape[0]} rows, lr {args.lr}, wd 0)")
    _emit(backbone_params, args.backbone_lr, bwd)
    for depth, params in sorted(llrd_groups.items()):
        _emit(params, args.backbone_lr * decay ** depth, bwd)
    if not getattr(args, "decay_all", False):
        n_nod = sum(len(g["params"]) for g in groups if g["weight_decay"] == 0.0)
        print(f"[optim] weight decay excluded on {n_nod} ndim<=1 tensors "
              f"(biases / norms / gates); --decay_all restores legacy")
    if llrd_groups:
        lrs = [args.backbone_lr * decay ** d for d in llrd_groups]
        print(f"[optim] LLRD over {n_blocks} blocks, decay {decay}: "
              f"backbone lr {min(lrs):.2e}..{max(lrs):.2e}")

    return torch.optim.AdamW(
        groups,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )


def build_scheduler(optimizer, args, steps_per_epoch: int):
    """Cosine decay with linear warm-up, stepped once per OPTIMIZER STEP.

    The old epoch-granular version made lr_lambda(0) ≈ 0 — the entire first
    epoch trained at ~zero LR (44 wasted GPU-minutes per full-data run).
    """
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = min(args.warmup_steps, total_steps // 10 + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(args.min_lr_factor, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def zeroshot_dev_metrics(eval_model, dev: dict, args,
                         device) -> Tuple[dict, List[dict]]:
    """Score the current weights on an OUT-OF-DOMAIN dev split each epoch.

    Why this exists: ``checkpoint_best`` used to maximise in-domain
    SoftmR@50 on megasg val, but that signal is measurably the wrong one.
    The v36 half-data arm showed the in-domain slope overstates zero-shot
    transfer by ~5x, and v35 beat v34 on *every* transfer metric while
    LOSING in-domain R@50 (0.3955 vs 0.4437). Selecting on in-domain
    therefore rewards fitting MegaSG's annotation style, which is not the
    product. This runs the real contract instead — swap the vocabulary,
    reparameterize, score under the graph constraint — on a split that is
    never trained on.

    The vocabulary swap is done in place and undone in a ``finally``.
    Restoring is MANDATORY, not hygiene: ``ModelEMA.update`` syncs buffers
    with ``copy_()``, which throws on the shape mismatch that a left-over
    [V_dev, text_dim] ``W`` would create on the next optimizer step. The
    four saved attributes are exactly what ``set_vocabulary_matrix`` and
    ``reparameterize`` touch.

    Every rank runs this on the full dev split, matching the val_loader
    policy above (evaluator.py never all-reduces, so sharding would score
    only rank 0's slice). It is ~1k images against ~50k train draws, and
    all ranks run it in parallel, so it costs no meaningful wall clock.
    """
    raw = eval_model.module if hasattr(eval_model, "module") else eval_model
    head = raw.vocab_head
    # `beta` must be saved with `alpha`: both are per-predicate buffers sized to
    # the CURRENT vocabulary, and this block swaps the head to the dev set's.
    # Restoring alpha but not beta leaves beta at the dev width while W returns
    # to the training width — the next training step then fails with a
    # 56-vs-19103 broadcast error.
    saved = (head.W, head.alpha, head.beta, head.pred_names,
             head.is_reparameterized)
    was_training = eval_model.training
    try:
        head.set_vocabulary_matrix(dev["pred_names"], dev["E"])
        raw.reparameterize()
        ev = SGClsEvaluator(
            topk=[20, 50, 100], num_predicates=len(dev["pred_names"]),
            score_mode=dev["score_mode"],
            # Graph-constrained: one triplet per pair, the convention behind
            # published R@K. Unconstrained ranking inflates R@K by 12-19 pts
            # and would make selection chase a metric we do not report.
            graph_constraint=True,
        )
        m = evaluate(eval_model, dev["loader"], device, args, ev,
                     eval_budget=dev["budget"])
        # Per-class recall from THIS evaluator, not the in-training val one.
        # This is the only per-epoch per-class readout that is trustworthy:
        # constrained AND scored against the dev pack's own vocabulary. The
        # megasg-val table ranks over all 19,103 training columns (every
        # determiner variant: `behind a`, `are behind`, `behind the`), which
        # suppresses spatial classes by ~14x — reading `in front of` = 0.000
        # out of a training log has cost us real time twice now.
        dev_per_class = ev.compute_per_class(50, dev["pred_names"])
    finally:
        (head.W, head.alpha, head.beta, head.pred_names,
         head.is_reparameterized) = saved
        eval_model.train(was_training)
    return ({f"dev_{k}": float(v) for k, v in m.items()}, dev_per_class)


class ModelEMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of the model whose parameters are a running
    weighted average of the training model.  The EMA model is used for
    evaluation and checkpointing; the training model receives gradients.

    After each optimizer step call ``ema.update(model)``.

    Args:
        model:  The training model to shadow (unwrapped, no DDP).
        decay:  EMA decay rate.  0.9998 is a good default for models trained
                for tens of epochs.  Higher = slower update = more smoothing.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        import copy
        self.decay = decay
        self.updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        # Disable gradients on the shadow copy — it is never trained directly.
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights: ema = d * ema + (1 - d) * model.

        d ramps from 0 toward ``decay`` (timm-style warmup): a fixed 0.9998
        after only ~1K steps leaves the shadow ~80% random init, which made
        short-run evals look dead.
        """
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        # Unwrap DDP if necessary
        src = model.module if hasattr(model, "module") else model
        for ema_p, src_p in zip(
            self.ema_model.parameters(), src.parameters()
        ):
            ema_p.mul_(d).add_(src_p.data, alpha=1.0 - d)
        # Keep buffers (e.g. BN running stats, vocab W) in sync too
        for ema_b, src_b in zip(
            self.ema_model.buffers(), src.buffers()
        ):
            ema_b.copy_(src_b)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema_model.load_state_dict(state)


def save_checkpoint(state: dict, output_dir: str, name: str) -> None:
    torch.save(state, os.path.join(output_dir, name))


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# ==========================================================================
# Argument parsing
# ==========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RelAnything SGCls trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset ----
    p.add_argument("--data_root", default="",
                   help="Path to a COCO-format SGG dataset root (must contain "
                        "train/ and val/ sub-folders with _annotations.coco.json). "
                        "Legacy single-source path; prefer --data_roots.")
    p.add_argument("--data_roots", nargs="+", default=None,
                   help="Multi-source mixture: pack roots trained jointly over a "
                        "shared union predicate vocab (first = base, whose "
                        "categories + val split are used). Overrides --data_root. "
                        "Requires --pred_embeds (union W, defines the order) and "
                        "--ontology_meta (union_meta.json).")
    p.add_argument("--mix_fractions", nargs="+", type=float, default=None,
                   help="Target per-source sampling fractions (one per data_root, "
                        "renormalized). Default: equal fractions across sources.")
    p.add_argument("--mix_temperature", type=float, default=None,
                   help="Derive mixture fractions from source SIZE instead of "
                        "hand-setting them: frac_s ∝ N_s**alpha. 1.0 = strictly "
                        "proportional (no source oversampled — every image "
                        "equally likely), 0.5 = sqrt sampling, 0.0 = equal per "
                        "source (max oversampling of small ones). Overrides "
                        "--mix_fractions. The hand-set 0.50/0.20/0.25/0.05 draws "
                        "spatialsense images ~11x more often per image than "
                        "megasg's, which memorizes small sources over a long run.")
    p.add_argument("--mix_max_passes", type=float, default=None,
                   help="Hard cap on per-epoch passes over any one source "
                        "(surplus redistributed to uncapped sources). Composable "
                        "with either fractions or --mix_temperature.")
    p.add_argument("--cat_aliases", default=None,
                   help="JSON {source_category_string: union_category_name} "
                        "from training/build_cat_aliases.py. Maps a source's "
                        "free-form box strings onto EXISTING union categories "
                        "so they stop being OOV (entity_label -1), which "
                        "otherwise forces all of that pair's negatives soft. "
                        "For svg_vg this lifts both-endpoints-in-vocab from "
                        "29.8%% to 77.1%% of its relations. Adds spellings "
                        "only — the taxonomy, W_obj and cooc are unchanged.")
    p.add_argument("--dev_root", default=None, metavar="PACK",
                   help="Packed root of an OUT-OF-DOMAIN dev split scored "
                        "every epoch (e.g. runs/packed/psg). Reported as "
                        "dev_* in history.json. Use a val split and keep the "
                        "matching test split for reporting.")
    p.add_argument("--dev_split", default="val")
    p.add_argument("--dev_budget", type=int, default=500,
                   help="Pair budget for the dev eval (matches "
                        "eval_zeroshot.py's default so the numbers line up).")
    p.add_argument("--dev_select", action="store_true",
                   help="Select checkpoint_best on --dev_metric instead of "
                        "in-domain SoftmR@50. The in-domain metric overstates "
                        "transfer ~5x (v36 half-data arm) and ranked v34 above "
                        "v35 even though v35 won every zero-shot metric.")
    p.add_argument("--dev_metric", default="mR@50",
                   help="Dev metric driving --dev_select. mR@50 is the tail "
                        "metric this model differentiates on; R@50 selects for "
                        "head recall instead.")
    p.add_argument("--exclude_ids", default=None, metavar="JSON",
                   help="Held-out image list (from "
                        "training/build_indoorvg_holdout.py): every TRAIN "
                        "source drops images whose file-name stem appears in "
                        "it. Needed because IndoorVG's eval images are Visual "
                        "Genome photos that also sit in vg_raw (506 of them) "
                        "and, under their COCO twin names, in megasg_clean "
                        "(12) — the registry's protected_vg_ids does NOT "
                        "cover IndoorVG. Applies to train only; val is "
                        "untouched.")
    p.add_argument("--drop_geometric", nargs="*", default=None, metavar="SOURCE",
                   help="Source names (pack basenames, e.g. 'gqa') whose "
                        "source=geometric relations are REMOVED from training "
                        "entirely, or 'all'. For GQA these are exactly its 1.6M "
                        "auto-derived left/right edges (88.5%% of that source), "
                        "which push left/right to 29.5%% of all mixture edges. "
                        "Removing them raises the mixture's mass on PSG's test "
                        "vocabulary from 27.3%% to 39.4%%. Stronger than "
                        "--geometric_weight, which only reweights the loss.")
    p.add_argument("--geometric_weight", type=float, default=1.0,
                   help="Loss weight for relations flagged source=geometric "
                        "(auto-derived from box geometry, not human/LLM "
                        "judged). 1.0 = no downweight (the historical default; "
                        "the plumbing existed but was never set). Matters for "
                        "the mixture: GQA is 88.5%% geometric left/right, which "
                        "pushes left/right to 29.5%% of all mixture edges vs "
                        "13.6%% for megasg alone. Train split only.")
    p.add_argument("--train_subset_frac", type=float, default=1.0,
                   help="Train on a deterministic random fraction of the "
                        "training images (val unchanged). Used for data-scaling "
                        "arms; 1.0 = full corpus. Subset is identical on every "
                        "DDP rank.")
    p.add_argument("--seed", type=int, default=42,
                   help="Global RNG seed (actual seed is this + DDP rank). "
                        "Default 42 reproduces every run made before this flag "
                        "existed; vary it for seed-replicate arms.")
    p.add_argument("--train_subset_seed", type=int, default=42,
                   help="Seed selecting the --train_subset_frac subset.")
    p.add_argument("--drop_predicates", default="",
                   help="Hold predicates out of TRAINING entirely: a .json list of "
                        "strings, or a comma-separated list. Used for the OvR-SGG "
                        "arm, where OvSGTR holds out 15 of VG150's 50 predicates and "
                        "reports Novel R@K. Relations carrying a dropped predicate are "
                        "removed from supervision and the head gets no row for it; the "
                        "predicate remains scorable at inference from its text "
                        "embedding. See training/ovr_novel_predicates.json.")
    p.add_argument("--samples_per_epoch", type=int, default=0,
                   help="Mixture only: draws per epoch (0 = size of the concat). "
                        "Set e.g. 50000 for a fast scale-controlled ablation.")
    p.add_argument("--ontology_meta", default=None,
                   help="Union meta.json for the mixture ontology "
                        "(build_union_vocab.py). Default: <data_root>/train/meta.json.")
    p.add_argument("--val_root", default=None,
                   help="Mixture only: pack supplying the val split (default: "
                        "first data_root). Use when the base train pack is "
                        "train-only, e.g. --val_root runs/packed/megasg.")
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--static_shapes", action="store_true",
                   help="Pad every batch's box dim to --max_objects instead "
                        "of the batch's own max, making the pair sampler's "
                        "shapes compile-time constants (torch.compile / CUDA "
                        "graphs / ONNX export friendly). No metric change "
                        "(see data/relation_dataset.collate_fn); costs some "
                        "throughput on batches with few real boxes.")

    # ---- Model ----
    p.add_argument("--backbone_type",  default="dinov3",
                   choices=["dinov3", "dinov2", "pe_core", "eupe",
                            "dinov3_convnext"])
    p.add_argument("--backbone_model", default="",
                   help="HF id or local path for the backbone (e.g. "
                        "checkpoints/hf/vitb16_lvd1689m — the facebook/dinov3-* "
                        "hub repos are gated). Empty = the backbone_type default.")
    p.add_argument("--lora_rank",      type=int, default=8,
                   help="LoRA rank; 0 = full fine-tune, -1 = frozen.")
    p.add_argument("--lora_layers",    type=int, default=None,
                   help="Apply LoRA only to the LAST N backbone blocks "
                        "(None = all). The multi-layer taps read [-6,-3,-1], "
                        "so 6 covers every tapped layer while halving the "
                        "backward cost through the backbone.")
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--geo_budget",     type=int, default=400)
    p.add_argument("--final_budget",   type=int, default=128)
    p.add_argument("--n_self_layers",  type=int, default=2)
    p.add_argument("--n_cross_layers", type=int, default=2)
    p.add_argument("--lambda_infonce", type=float, default=0.5)
    p.add_argument("--lambda_czsc",    type=float, default=0.05,
                   help="Weight for CZSC zone SupCon loss. 0.05 is stable at full scale; "
                        "0.0 disables it entirely.")
    p.add_argument("--logit_scale_init", type=float, default=5.0,
                   help="Initial logit scale γ₀ = 1/τ. 5.0 (τ=0.2) avoids AMP overflow "
                        "in early epochs. Increase toward 14.3 (τ=0.07) via --resume if needed.")
    p.add_argument("--text_dim",       type=int,   default=2048,
                   help="Text embedding dim for VocabHead. 2048 = dino.txt (default). "
                        "Set to 512 to fall back to CLIP-B/32.")

    # ---- Loss (plan D2, revised per user: batch-local contrastive) ----
    p.add_argument("--loss_type", default="batch_infonce",
                   choices=["batch_infonce", "synonym", "ce"],
                   help="'batch_infonce' = GLIP/YOLO-World-style batch-local "
                        "InfoNCE with sampled hard negatives; the 10K vocab is "
                        "inference-only. 'synonym' = earlier V-wide masked "
                        "sigmoid (ablation). 'ce' = legacy softmax (ablation).")
    p.add_argument("--n_neg", type=int, default=256,
                   help="Sampled negatives per batch for batch_infonce.")
    p.add_argument("--lambda_obj", type=float, default=0.3,
                   help="Weight of the object-semantics aux loss (compositional "
                        "query training; entity labels used at train only).")
    p.add_argument("--dual_spatial_head", action="store_true",
                   help="Two-expert query head: semantic (compositional) + "
                        "spatial (direct geometry path), mixed per-predicate "
                        "by a spatialness gate fitted on text embeddings — "
                        "removes top-K competition between the 21 mega-"
                        "frequent spatial predicates and the semantic tail.")
    p.add_argument("--fast_bilinear", action="store_true",
                   help="Aux RAM-style separable pair scorer q=LN(P_s v_i + "
                        "P_o v_j) vs W — the all-pairs two-matmul export "
                        "path; eval reports its metrics as fast_*.")
    p.add_argument("--lambda_fast", type=float, default=0.5,
                   help="Weight of the fast bilinear head's aux InfoNCE.")
    p.add_argument("--lambda_swap", type=float, default=0.0,
                   help="Cross-slot direction hinge weight (v3.2). Fixes "
                        "SwapAcc~0.5: nothing else pushes score(o,s,g) below "
                        "score(s,o,g).")
    p.add_argument("--swap_margin", type=float, default=0.05,
                   help="Hinge margin on cosine scale.")
    p.add_argument("--cfa_mode", default="off",
                   choices=["off", "entity", "zone", "both"],
                   help="CFA-analogue feature augmentation (train-only): mix "
                        "pooled pair components with a random batch slot "
                        "sharing the canonical predicate group. 'entity' mixes "
                        "v_sub/v_obj, 'zone' mixes v_union/v_contact. "
                        "Label-preserving by construction.")
    p.add_argument("--cfa_prob", type=float, default=0.0,
                   help="Per-slot probability of applying the CFA mix.")
    p.add_argument("--cfa_alpha", type=float, default=1.0,
                   help="lambda ~ Beta(a, a) for the CFA blend; 1.0 = Uniform.")
    p.add_argument("--cfa_partner", default="group", choices=["group", "random"],
                   help="Partner criterion for the CFA mix. 'group' = CFA "
                        "proper (partner shares the canonical predicate group, "
                        "label-preserving). 'random' = MECHANISM CONTROL: same "
                        "src slots and same lambda, predicate-blind partner — "
                        "isolates whether same-predicate matching is what "
                        "produces the gain.")
    p.add_argument("--proj_layers", type=int, default=2,
                   help="Visual→text projection depth (1 = single linear).")
    p.add_argument("--neg_rate_table", default="",
                   help="pair_opportunity.npz from training/build_pair_opportunity.py. "
                        "Prices the relatedness-BCE negative weight per category pair "
                        "as clamp(1-interaction_rate, --rel_neg_weight, 1.0) instead of "
                        "a flat constant. Training-only; inference is unchanged.")
    p.add_argument("--canon_groups",
                   default="runs/packed/megasg/text_space/canonical_groups.json",
                   help="hand-written synonym groups (text_space_diag.py). Still used "
                        "to expand the four INVERSE seeds — direction supervision, "
                        "which measures correct. Pass '' to drop it entirely (seed "
                        "strings only).")
    p.add_argument("--pos_agg", choices=["lse", "mean"], default="lse",
                   help="how the GT's canonical group is aggregated into the positive "
                        "term. lse (v34-v39): logsumexp — ONE member satisfies it, so "
                        "the model aligns to a text-space REGION (transfers well) but "
                        "the GT string need not be the peak (buried at median rank 338 "
                        "for `in front of`). mean: weighted mean of per-member terms, "
                        "every member pulled up and the GT string pulled hardest — the "
                        "formulation that can have both. See --pos_member_weight.")
    p.add_argument("--pos_member_weight", type=float, default=0.0,
                   help="with --pos_agg mean, the weight of NON-GT group members (the "
                        "GT string is always 1.0). 0.0 reproduces v40's identity "
                        "positives: ranking fixed, tail transfer -11..-18%% mR@50 and "
                        "-33..-68%% rare recall. 1.0 pulls every member equally. This is "
                        "the dial between the two measured endpoints.")
    p.add_argument("--soft_supervision", default="",
                   help="v42+: soft_supervision.npz from "
                        "training/build_soft_supervision.py — positives from the "
                        "fitted synonym kernel, denominator weights from the "
                        "Elkan-Noto-calibrated also-true estimator, hinge "
                        "eligibility from the reverse-annotation symmetry EM, "
                        "w_cooc fitted. Supersedes and IGNORES tau_ignore, "
                        "hard_lo, neg_weight, soft_neg_weight, tau_ctx*, "
                        "--group_positives, --pos_member_weight; use --pos_agg "
                        "mean with it (lse would reintroduce winner-take-all).")
    p.add_argument("--group_positives", action="store_true",
                   help="LEGACY (v34-v39): let --canon_groups also define POSITIVES, "
                        "i.e. logsumexp over the group. This is what buried "
                        "`in front of` at median rank 338/1972 while `before` — 11 "
                        "training relations against its 414,961 — took rank 1: "
                        "logsumexp is satisfied by ONE member and starves the rest. "
                        "Default off; the GT string is its own only positive.")
    p.add_argument("--pred_context", default="",
                   help="pred_context.npz from training/build_predicate_context.py: "
                        "measured distributional interchangeability, used ONLY to "
                        "widen the abstain (ignore) set, never to assert a positive.")
    p.add_argument("--tau_ctx", type=float, default=0.5,
                   help="context similarity at/above which to abstain on a column.")
    p.add_argument("--tau_ctx_floor", type=float, default=0.85,
                   help="ANTONYM VETO: text cosine below which --pred_context is "
                        "refused. Antonyms have the highest context similarity in the "
                        "vocabulary (to the left of / to the right of = 0.892); this "
                        "floor is the only thing keeping them hard negatives. Do not "
                        "lower it without re-checking that pair.")
    p.add_argument("--pred_embeds", default="",
                   help="Optional pred_embeds_*.npz from text_space_diag.py: installs "
                        "W directly (template-ensembled) AND drives the ontology's "
                        "cosine ignore masks.")
    p.add_argument("--tau_ignore", type=float, default=0.9,
                   help="Text-cosine threshold above which non-group predicates are "
                        "excluded from negatives (calibrate via text_space_diag).")
    p.add_argument("--neg_weight", type=float, default=0.3,
                   help="Weight of unlabeled negatives (positive-unlabeled "
                        "correction); spatial inverses always get weight 1.")
    p.add_argument("--tau_eval", type=float, default=0.9,
                   help="Cosine threshold for SoftSGCls predicate matching "
                        "(calibrate against the LLM judge, plan §9a).")
    p.add_argument("--logit_bias_init", type=float, default=None,
                   help="Initial VocabHead logit bias (SigLIP trick). Default: "
                        "-10 for --loss_type synonym (negatives born suppressed "
                        "at V=10K), 0 for ce.")

    # ---- v34: cooc hard/soft negatives + trainable gate + student space ----
    p.add_argument("--pair_cooc", default="",
                   help="pair_cooc npz from training/build_pair_cooc.py — "
                        "enables hard/soft negative weighting in the batch "
                        "InfoNCE (PU-aware: seen (cat,cat,pred-group) combos "
                        "are soft, never-seen-with-support are hard).")
    p.add_argument("--soft_neg_weight", type=float, default=0.3,
                   help="Denominator weight of SOFT (plausibly-unlabeled) "
                        "negatives; hard negatives keep weight 1.")
    p.add_argument("--min_support", type=int, default=30,
                   help="Min train relations for a category pair before its "
                        "never-seen predicates count as hard negatives.")
    p.add_argument("--hard_lo", type=float, default=0.5,
                   help="Lower cosine bound of the InfoNCE hard-negative "
                        "pool. Space-dependent: ~0.5 dinotxt, ~0.85 student "
                        "(random-pair cosine is ~0.78 there).")
    p.add_argument("--restrict_neg_sources", nargs="*", default=[],
                   help="Source names (basename of --data_roots entries, e.g. "
                        "hicodet) whose anchors are contrasted ONLY against "
                        "that source's own predicate vocabulary in the InfoNCE, "
                        "sigmoid-aux and background losses. Single-vocabulary "
                        "sources never label spatial relations, so without this "
                        "'person riding horse' pushes down 'on'/'above' every "
                        "batch (measured: HICO mix costs projective SpatialSense "
                        "AUC -0.05, share-independent).")
    p.add_argument("--lambda_sigmoid", type=float, default=0.0,
                   help="weight of the per-cell sigmoid (SPML) auxiliary; "
                        "trains cross-pair calibration, which InfoNCE cannot")
    p.add_argument("--box_token_dropout", type=float, default=0.0,
                   help="train-time probability of dropping the box-corner "
                        "tokens from the rel_transformer cross-attn memory "
                        "(modality dropout against the geometry-shortcut "
                        "collapse measured in runs/analysis/"
                        "relation_attn_stats.json); 0 = off")
    p.add_argument("--beta_relatedness", action="store_true",
                   help="per-predicate weight on the relatedness logit, "
                        "read off the text embedding like the gate")
    p.add_argument("--gate_mlp", action="store_true",
                   help="Trainable MLP spatialness gate (warm-started from "
                        "the logistic probe) instead of the frozen probe.")
    p.add_argument("--text_student", default="",
                   help="Distilled student checkpoint (relsgg/text_student.py) "
                        "for encoding OBJECT categories when text_dim matches "
                        "the student space; predicate W still comes from "
                        "--pred_embeds.")

    # ---- Pair sampler (plan D4) ----
    p.add_argument("--sampler_type", default="cascade",
                   choices=["cascade", "relatedness"],
                   help="'relatedness' = vectorized sampler with learned "
                        "asymmetric pair-existence head + swapped-GT inclusion "
                        "(P2); 'cascade' = P1 baseline.")
    p.add_argument("--rel_neg_weight", type=float, default=0.3,
                   help="Negative weight in the relatedness BCE (PU-aware).")
    p.add_argument("--no_swap_include", action="store_false", dest="swap_include",
                   default=True,
                   help="Disable swapped-GT pair inclusion (ablation).")

    # ---- Zero-shot cross-dataset eval ----
    p.add_argument("--vg150_root", default="",
                   help="Optional: path to VG150 COCO-format root for a final zero-shot "
                        "evaluation after training completes.")

    # ---- dino.txt text encoder ----
    p.add_argument("--dinotxt_weights", default="dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
                   help="Path or URL to the dino.txt combined checkpoint "
                        "(vision head + text encoder). Required when text_dim=2048. "
                        "Filename matches dinov3_vitl16_dinotxt_*.pth.")
    p.add_argument("--dinotxt_bpe",
                   default="https://dl.fbaipublicfiles.com/dinov3/thirdparty/"
                           "bpe_simple_vocab_16e6.txt.gz",
                   help="Path or URL to the BPE vocabulary used by dino.txt tokenizer.")

    # ---- Relation Interaction Block ----
    p.add_argument("--use_rel_interaction", action="store_true", default=True)
    p.add_argument("--no_rel_interaction",  action="store_false", dest="use_rel_interaction",
                   help="Disable RelationInteractionBlock for ablation.")
    p.add_argument("--n_dep_layers", type=int, default=2,
                   help="Self-attn layers for inter-pair dependency.")
    p.add_argument("--n_gnd_layers", type=int, default=1,
                   help="Cross-attn layers for image grounding.")

    # ---- Training ----
    p.add_argument("--output_dir",   default="./runs/exp")
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--backbone_lr",  type=float, default=1e-5)
    p.add_argument("--multi_scale", default="",
                   help="Multi-scale square resize, e.g. '0.5,1.5' (lo,hi "
                        "relative to --img_size). One resolution per BATCH, "
                        "drawn rank-identically. FREE geometrically: a square "
                        "resize is the identity in normalized cxcywh, so boxes, "
                        "cov rasters and scene_pe are all untouched; only the "
                        "patch grid and object pixel size change. Train only — "
                        "val/dev/zero-shot stay at --img_size. Empty = off.")
    p.add_argument("--multi_scale_n", type=int, default=7,
                   help="Rungs in the scale ladder, snapped to multiples of the "
                        "patch size (16); duplicates collapse.")
    p.add_argument("--backbone_layer_decay", type=float, default=1.0,
                   help="LLRD: block i of L trains at backbone_lr*decay^(L-1-i)"
                        " (embeddings at decay^L). 1.0 = flat (prior runs).")
    p.add_argument("--backbone_weight_decay", type=float, default=None,
                   help="WD override for backbone groups (None = weight_decay)."
                        " 0 stops decaying pretrained weights toward zero.")
    p.add_argument("--deformable_points", type=int, default=0,
                   help="Box-anchored deformable scene read: points per anchor"
                        " (4 anchors), additive behind a zero-init gate. 0=off.")
    p.add_argument("--deformable_heads", type=int, default=1,
                   help="Deformable sampling heads: H heads = H x locations at"
                        " identical memory traffic (d_model/H channels each).")
    p.add_argument("--deformable_nulls", type=int, default=0,
                   help="V3: learnable null slots per (head,anchor) in the "
                        "sampling softmax — clean per-pair attenuation.")
    p.add_argument("--deformable_clamp", action="store_true",
                   help="V3: clamp sampled positions to the image frame.")
    p.add_argument("--deformable_v2", action="store_true",
                   help="Compound v2 arm: ring init + gain gate + border pad "
                        "at once. Measured -6.8%% A6; prefer the flags below.")
    p.add_argument("--deformable_ring", action="store_true", default=None,
                   help="Distinct angle/radius per (head,point). Required for "
                        "heads to specialize (identical points -> identical "
                        "gradients). Unset = follow --deformable_v2.")
    p.add_argument("--deformable_gain", action="store_true", default=None,
                   help="Per-pair sigmoid gain on the read. Redundant with "
                        "--deformable_nulls. Unset = follow --deformable_v2.")
    p.add_argument("--deformable_border", action="store_true", default=None,
                   help="grid_sample padding_mode=border. No-op under "
                        "--deformable_clamp. Unset = follow --deformable_v2.")
    p.add_argument("--ms_depth_levels", type=int, default=0,
                   help="Multi-level deformable read: expose this many backbone"
                        " taps as SEPARATE levels (own LayerNorm + 1x1) instead"
                        " of the fused map. Measured: the fused map is ~0.72 of"
                        " its magnitude from the LAST tap despite a uniform"
                        " combiner, so the depth axis is averaged away. 0=off.")
    p.add_argument("--ms_pool_level", action="store_true",
                   help="Add a stride-32 avg-pooled level (zero params; the "
                        "only level that changes spatial support).")
    p.add_argument("--ms_deconv_level", action="store_true",
                   help="Add a ViTDet-style stride-8 deconv level (1.05M "
                        "params). Learned sharpening, NOT recovered detail — a "
                        "plain ViT/16 has no sub-16 representation.")
    p.add_argument("--depth_scaled_init", action="store_true",
                   help="GPT-2-style 1/sqrt(N) residual out-proj init of the "
                        "from-scratch rel transformer + interaction stacks.")
    p.add_argument("--drop_path", type=float, default=0.0,
                   help="Stochastic-depth rate for the HF backbone (DINOv3 "
                        "native; train-mode only).")
    p.add_argument("--norm_taps", action="store_true",
                   help="LayerNorm on each backbone tap before the "
                        "softmax-weighted fusion (decouples layer importance "
                        "from activation scale). Parameter-free on the ViT "
                        "path; affine + post-projection on the ConvNeXt path, "
                        "where it ALSO makes --layer_weights non-degenerate.")
    p.add_argument("--stage_s2d", action="store_true",
                   help="ConvNeXt only: space-to-depth instead of area-average "
                        "when bringing a stage finer than the patch grid down "
                        "to it (lossless + learned vs a fixed blur).")
    p.add_argument("--stage_weight_init", default="", choices=["", "measured"],
                   help="ConvNeXt only: 'measured' inits the stage combiner to "
                        "the shares the un-normalized design produced, so "
                        "--norm_taps starts as a near-no-op.")
    p.add_argument("--pe_num_freqs", type=int, default=64,
                   help="Fourier bands per coordinate in the box positional "
                        "encoders. Legacy 64 doubling ladder is float32-noise "
                        "past band ~19; recommended 16 with --pe_max_octave 7. "
                        "Changes proj width — old ckpts need the default.")
    p.add_argument("--pe_max_octave", type=float, default=None,
                   help="Top PE frequency = 2**this (geometric ladder). "
                        "None = legacy 2**arange ladder.")
    p.add_argument("--geo_squash", action="store_true",
                   help="10*tanh(x/10) instead of clamp(-10,10) on geometry "
                        "features (gradient survives at the rails).")
    p.add_argument("--geo_pu", action="store_true",
                   help="PU-price the geometry pre-scorer's BCE negatives "
                        "with the opportunity table (needs --neg_rate_table).")
    p.add_argument("--scene_pe", action="store_true",
                   help="Gated absolute Fourier PE on scene keys at all three "
                        "cross-attn sites (rel transformer, interaction "
                        "grounding, spatial pool). Zero-init gates: safe to "
                        "enable on any run.")
    p.add_argument("--mode_gated", action="store_true",
                   help="Gate every mask-sensitive parameter (cov_lambda, a "
                        "query mode-embedding, 4 delta geometry columns) on a "
                        "per-image mode bit, so a box image runs the box-only "
                        "network EXACTLY and masks train a zero-init adapter. "
                        "Needs --rasters to do anything.")
    p.add_argument("--region_adjacency", action="store_true",
                   help="With --mode_gated: give the region adapter a 5th column, "
                        "region BOUNDARY ADJACENCY (overlap after a one-cell "
                        "dilation). r_contact is an intersection and is identically "
                        "zero for masks that PARTITION the image -- panoptic/PSG "
                        "masks never overlap (measured 0%% of test pairs, vs 19%% "
                        "for SAM box-prompted) -- so the contact column is "
                        "degenerate exactly where contact predicates live. "
                        "Adapter-only: NUM_GEO stays 19 and every existing "
                        "checkpoint still loads.")
    p.add_argument("--contact_field", action="store_true",
                   help="Mask-derived coverage bias for the CONTACT half of the "
                        "merged spatial pool. That half currently gets a flat "
                        "raster, making it the one part of the network that is "
                        "identical for boxes and masks by construction; this "
                        "replaces it with min(dil(cov_sub), dil(cov_obj)). "
                        "Mode-gated, so box images stay bit-identical.")
    p.add_argument("--mask_adapter_dim", type=int, default=0,
                   help="Width of a zero-init, mode-gated residual MLP on the "
                        "fused pair input (0 = off). Capacity knob for the "
                        "frozen-trunk retrofit, which otherwise trains only "
                        "1,416 parameters; 64 gives ~1.5e5. Needs --mode_gated.")
    p.add_argument("--freeze_trunk", action="store_true",
                   help="With --init_from --mode_gated: train ONLY the mode "
                        "adapter (spatial_pool.cov_lambda / mode_embed, "
                        "geo_encoder.region_delta). The box path is then "
                        "bit-identical to the source checkpoint by construction.")
    p.add_argument("--pool_role_queries", action="store_true",
                   help="Per-role (object/union/contact) query biases in "
                        "SoftSpatialPool, zero-init.")
    p.add_argument("--lambda_geo", type=float, default=1.0,
                   help="Geometry pre-scorer BCE weight. <=v45 runs used 0.1 "
                        "on the SUM (geo+rel); 0.1 with --lambda_rel 0.1 "
                        "reproduces that weighting.")
    p.add_argument("--lambda_rel", type=float, default=1.0,
                   help="Relatedness-head BCE weight (pair existence — the "
                        "deployed sigmoid(rel) term). 0.1 reproduces <=v45.")
    p.add_argument("--bg_agg", choices=["topk", "lse"], default="topk",
                   help="Aggregate for the background (no-relation) penalty. "
                        "topk = mean softplus over --bg_topk columns (k=5 was "
                        "never justified). lse = softplus(logsumexp_v z_v), the "
                        "NLL of the null under an at-most-one model — matches "
                        "graph-constrained decoding, and its softmax gradient "
                        "protects tail columns exponentially instead of at a "
                        "rank-k cliff. topk is bit-exact with prior runs.")
    p.add_argument("--lambda_bg", type=float, default=0.0,
                   help="Background suppression on valid non-GT slots "
                        "(top-k softplus on the fused logits, PU-weighted). "
                        "The predicate head otherwise never sees a negative "
                        "PAIR. Start 0.05-0.1; 0 = off (legacy).")
    p.add_argument("--bg_topk", type=int, default=5,
                   help="How many top columns per non-GT slot the background "
                        "suppression pushes down (tail-safe self-focusing).")
    p.add_argument("--role_obj_loss", action="store_true",
                   help="Role-disjoint training sets for sub/obj text "
                        "projections (compose path) instead of identical "
                        "ones — gives the compositional query real "
                        "subject/object asymmetry.")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--decay_all", action="store_true",
                   help="Legacy: apply weight decay to ndim<=1 params too "
                        "(biases, norms, zero-init gates). Default excludes "
                        "them. Required when resuming pre-split checkpoints.")
    p.add_argument("--clip_grad",    type=float, default=1.0)
    p.add_argument("--dead_param_audit", type=int, default=100,
                   help="Fail if any trainable param got no gradient within "
                        "this many steps of epoch 0 (0 = off). Catches "
                        "silently-dead modules that find_unused_parameters "
                        "otherwise hides (the logit_scale lesson).")
    p.add_argument("--dead_param_warn", action="store_true",
                   help="Downgrade the dead-param audit from error to warning.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="Micro-batches per optimizer step. bs32 x accum4 on "
                        "ONE GPU reproduces the 4-GPU bs32 reference exactly "
                        "(per-rank InfoNCE contrast sets AND global batch "
                        "preserved); a bigger single-GPU batch would not.")
    p.add_argument("--grad_telemetry", type=int, default=0,
                   help="Every N steps, log each loss term's lambda-scaled "
                        "gradient norm on two shared-trunk tensors "
                        "(gnorm_<term>_{pair,pool}). ~1 extra head-backward "
                        "per term per measurement; 0 = off. Run this ONCE on "
                        "a proxy arm before touching loss weights.")
    p.add_argument("--warmup_epochs",type=int,   default=1,
                   help="DEPRECATED (was epoch-granular = zero-LR first epoch); "
                        "kept for arg compat, use --warmup_steps.")
    p.add_argument("--warmup_steps", type=int,   default=1000,
                   help="Linear LR warmup steps (capped at 10%% of total steps).")
    p.add_argument("--min_lr_factor",type=float, default=0.01,
                   help="Minimum LR = lr × min_lr_factor (end of cosine schedule).")
    p.add_argument("--amp",    action="store_true", default=True)
    p.add_argument("--no_amp", action="store_false", dest="amp")
    p.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"],
                   help="Autocast dtype. bf16 (default) is overflow-immune; "
                        "fp16 kept for GPUs without bf16 (T4/V100).")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--stop_after_epoch", type=int, default=0,
                   help="Stop after this many epochs while keeping the --epochs "
                        "LR SCHEDULE. This is the validated cheap gate: "
                        "training/analyze_epoch_truncation.py measures that the "
                        "in-training ranking of 21 proxy arms is FLAT from epoch "
                        "2 on (sign agreement with the final 12-epoch TEST "
                        "verdict ~80% at every epoch), so a truncated run ranks "
                        "arms as well as a finished one -- but only if the "
                        "schedule is unchanged, since a shorter cosine is a "
                        "different recipe, not a truncation. Resumable to the "
                        "full horizon with --resume. 0 = run all --epochs.")
    p.add_argument("--init_from", default="",
                   help="Weights-only initialisation from a finished run's checkpoint "
                        "(EMA weights; vocabulary-sized buffers are re-installed from "
                        "--pred_embeds). Architecture/recipe keys are inherited from the "
                        "checkpoint's args so the fine-tune cannot silently build a "
                        "different network. Used for the closed-set (VG150/PSG) "
                        "fine-tune comparison against REACT.")
    p.add_argument("--learn_W", action="store_true",
                   help="With --init_from: make the predicate matrix W trainable "
                        "(closed-set classifier ablation; breaks the open-vocab contract).")
    p.add_argument("--resume",      default="",
                   help="Path to a checkpoint to resume from.")

    # ---- Evaluation ----
    p.add_argument("--eval_budget",         type=int, default=500,
                   help="Pair sampling budget during evaluation.")
    p.add_argument("--val_eval_limit",      type=int, default=0,
                   help="Cap the per-epoch recall evaluate() on val to the "
                        "first N images (0 = full split). The full megasg "
                        "val is ~25K images and costs ~5 min/epoch through "
                        "the SGCls+Soft evaluator fan-out, for numbers that "
                        "are diagnostic-only (checkpoint selection reads "
                        "dev_*; see relsgg-intraining-eval-artifact for why "
                        "the in-training val numbers must not be ranked on "
                        "anyway). val_loss and dev are unaffected.")
    p.add_argument("--eval_batch_size",     type=int, default=0,
                   help="Batch size for the val/dev eval loaders (0 = same "
                        "as --batch_size). Eval holds no gradients, so this "
                        "can be several x the train batch size.")
    p.add_argument("--val_loss_batches",    type=int, default=100,
                   help="Val batches used to compute val_loss_* (train/val "
                        "loss-gap diagnostic, distinct from the recall-based "
                        "evaluate() above). 0 disables.")
    p.add_argument("--embed_eval_every",    type=int, default=5,
                   help="Run embedding analysis every N epochs (and on last epoch).")
    p.add_argument("--embed_max_per_class", type=int, default=500)
    p.add_argument("--embed_max_batches",   type=int, default=200)

    # ---- Regularization ----
    # v34 overfits the full 472K pack from ~epoch 8 (val_loss_nce min at ep8,
    # rising after; R@50 peaks at ep7) — these are the levers for that.
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Attention/FFN dropout in the relation transformer AND "
                        "the interaction block. Was hardcoded 0.1 in both (so "
                        "unsweepable); 0.1 reproduces every prior run exactly.")
    p.add_argument("--rasters", default=None,
                   help="Root of precomputed region rasters (see "
                        "datagen/build_mask_rasters.py). Unset = box-only, "
                        "numerically identical to the pre-mask model.")
    p.add_argument("--mask_dropout", type=float, default=0.0,
                   help="Per-IMAGE probability of replacing masks with their "
                        "bounding rectangles during TRAINING. Deployment gets "
                        "detector boxes (a segmenter costs ~150ms against the "
                        "model's 29ms), so the box path must stay first-class; "
                        "0.5 gives both regimes equal supervision.")
    p.add_argument("--augment", type=float, default=0.0,
                   help="Photometric augmentation strength on TRAIN images "
                        "(brightness/contrast/saturation, factors drawn from "
                        "U(1-a, 1+a)); 0 = off, the historical behaviour. "
                        "Geometry-preserving so box coordinates stay valid — no "
                        "flip (directional predicates) and no crop (would need "
                        "box transforms). NOTE: mutually exclusive with any "
                        "backbone-feature caching, since cached features cannot "
                        "be re-augmented.")
    p.add_argument("--compose_query", type=lambda s: s.lower() in ("1", "true", "yes"),
                   default=None,
                   help="Override the compositional query path (q = proj(r) + "
                        "g0*P_s(v_sub) + g1*P_o(v_obj)). Default None = legacy "
                        "behaviour, i.e. ON iff --loss_type batch_infonce. Set "
                        "explicitly to ablate it independently of the loss.")
    p.add_argument("--tucker_query", default="",
                   help="'r12,r3' -> add a MUTAN-style MULTIPLICATIVE pair "
                        "term P((A v_sub) x_G (B v_obj)) into the composed "
                        "query (mode-3 factor = the frozen text bank, so open "
                        "vocab is untouched). Zero-init output: step 0 is "
                        "bit-identical to the baseline. Requires the "
                        "compose_query path. '' = off, prior runs unchanged.")
    p.add_argument("--no_save_best", action="store_true",
                   help="Write only checkpoint_last.pth, not checkpoint_best. "
                        "The best epoch is still tracked and printed. Use for "
                        "any run ranked on the FINAL epoch (our protocol), "
                        "which makes checkpoint_best a ~1-2GB dead copy.")
    p.add_argument("--no_save_checkpoint", action="store_true",
                   help="Skip writing checkpoint_last/best.pth every epoch. "
                        "For throwaway diagnostic runs only (e.g. throughput/"
                        "hyperparam probes) — each checkpoint is ~1.2GB and "
                        "the write to network storage dominates epoch time "
                        "on short runs. Never use for a real training run.")

    # ---- EMA ----
    p.add_argument("--ema_decay", type=float, default=0.9998,
                   help="EMA decay for shadow model used at eval time. "
                        "0 = disable EMA.")

    # ---- DDP ----
    p.add_argument("--local_rank", type=int,
                   default=int(os.environ.get("LOCAL_RANK", 0)))

    # ---- Logging ----
    p.add_argument("--log_every", type=int, default=50,
                   help="Monitor: log iteration losses every N steps "
                        "(metrics/iters.jsonl + plots).")
    p.add_argument("--wandb",          action="store_true", default=True,
                   help="Enable Weights & Biases logging (offline by default "
                        "on compute nodes; sync later with `wandb sync`).")
    p.add_argument("--no_wandb", action="store_false", dest="wandb")
    p.add_argument("--wandb_project",  default="relanything",
                   help="W&B project name.")
    p.add_argument("--wandb_run_name", default="",
                   help="W&B run name (defaults to output_dir basename).")

    return p.parse_args()


# ==========================================================================
# Plotting
# ==========================================================================

def save_training_plots(history: List[dict], output_dir: str) -> None:
    """Save training_curves.png with loss and recall subplots."""
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [row["epoch"] for row in history]
    loss_keys = [k for k in history[0]
                 if k.startswith("loss_") and k not in ("loss_total", "amp_scale")]
    has_val_loss = "val_loss_total" in history[0]

    fig, axes = plt.subplots(1, 3 if has_val_loss else 2, figsize=(21 if has_val_loss else 14, 5))

    # -- Left: individual losses --
    ax = axes[0]
    for k in loss_keys:
        ax.plot(epochs, [row.get(k, float("nan")) for row in history],
                label=k.replace("loss_", ""), linewidth=1.5)
    if "loss_total" in history[0]:
        ax.plot(epochs, [row.get("loss_total", float("nan")) for row in history],
                label="total", linewidth=2, linestyle="--", color="black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # -- Middle (if val_loss_* present): train vs val loss GAP — the direct
    # overfitting readout. Widening gap (val flat/rising while train falls) =
    # overfitting; both falling together = still generalizing, keep training.
    if has_val_loss:
        ax = axes[1]
        ax.plot(epochs, [row.get("loss_total", float("nan")) for row in history],
                label="train", linewidth=2, color="tab:blue")
        ax.plot(epochs, [row.get("val_loss_total", float("nan")) for row in history],
                label="val", linewidth=2, color="tab:red", marker="o", markersize=3)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train vs Val Loss (overfitting readout)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # -- Right: recall metrics --
    # In-domain solid/dashed, out-of-domain dev dotted on the same axes. Both
    # belong in one frame precisely because they diverge: v35 lost in-domain
    # R@50 to v34 while winning every transfer metric, which is why
    # --dev_select exists. Seeing the two curves part is the readout.
    ax = axes[-1]
    for key, ls in [("mR@20", "-"), ("mR@50", "-"), ("mR@100", "-"), ("R@50", "--")]:
        if key in history[0]:
            ax.plot(epochs, [row.get(key, float("nan")) for row in history],
                    label=key, linestyle=ls)
    for key in ("dev_mR@50", "dev_R@50"):
        if key in history[0]:
            ax.plot(epochs, [row.get(key, float("nan")) for row in history],
                    label=key, linestyle=":", linewidth=2, marker="o",
                    markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.set_title("Evaluation Recall (dotted = out-of-domain dev)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    args = parse_args()

    if args.init_from:
        inherit_init_args(args)

    # ---- Distributed setup ----
    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if ddp:
        # Default 600s NCCL watchdog is too tight around rank-0-only blocks
        # (embed analysis, checkpoint I/O) that other ranks don't join —
        # crashed job 6876175 at epoch 4's embedding analysis with a
        # collective timeout while ranks 1-3 waited at the next epoch's
        # first backward. Widen it; this is a safety margin, not a fix for
        # a true deadlock (embed analysis is also skippable via
        # --embed_max_batches 0, which is the actual fix used going forward).
        dist.init_process_group(backend="nccl",
                                timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Reproducibility ----
    seed = args.seed + (dist.get_rank() if ddp else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # ---- Output dir + monitor ----
    monitor = None
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        wandb_run = None
        if args.wandb:
            try:
                import wandb
                run_name = args.wandb_run_name or Path(args.output_dir).name
                # Compute nodes are offline: default to offline mode so runs
                # land in <output_dir>/wandb/ and can be pushed later with
                # `wandb sync <output_dir>/wandb/offline-run-*` (login node,
                # after `wandb login`). WANDB_MODE env still wins.
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=run_name,
                    config=vars(args),
                    dir=args.output_dir,
                    mode=os.environ.get("WANDB_MODE", "offline"),
                    resume="allow",
                )
            except ImportError:
                print("[wandb] wandb not installed; skipping.  pip install wandb")
                args.wandb = False
        from relsgg.monitor import TrainMonitor
        monitor = TrainMonitor(args.output_dir, log_every=args.log_every,
                               wandb_run=wandb_run)

    # ---- Datasets ----
    print(f"Loading dataset from {args.data_root} ...")
    train_ds, val_ds, pred_names = build_datasets(args)
    print(f"  train: {len(train_ds):,} images")
    print(f"  val:   {len(val_ds):,} images")
    print(f"  predicates ({len(pred_names)}): {pred_names[:5]} ...")

    # ---- DataLoaders ----
    if args.data_roots:
        # weighted mixture: realise per-source target fractions regardless of
        # raw source size, sharded across DDP ranks with a shared epoch seed.
        from data.multipack import (DistributedWeightedSampler,
                                     fractions_from_temperature,
                                     sample_weights_from_fractions)
        soi = args._source_of_index
        n_src = len(args._source_names)
        draws = args.samples_per_epoch or len(train_ds)
        counts = np.bincount(soi, minlength=n_src)
        if args.mix_temperature is not None:
            fracs = list(fractions_from_temperature(counts, args.mix_temperature))
            print(f"[mixture] temperature alpha={args.mix_temperature} → "
                  f"size-derived fractions")
        else:
            fracs = args.mix_fractions or [1.0] * n_src
        assert len(fracs) == n_src, (
            f"--mix_fractions has {len(fracs)} values for {n_src} sources")
        weights = sample_weights_from_fractions(
            soi, fracs, max_passes=args.mix_max_passes, draws_per_epoch=draws)
        # Report the REALIZED per-source rate, including any max_passes
        # redistribution — and the per-image repetition relative to the largest
        # source, which is the number that actually drives small-source memorization.
        realized = np.array([weights[soi == s].sum() for s in range(n_src)])
        passes = np.where(counts > 0, draws * realized / np.maximum(counts, 1), 0.0)
        base = passes.max() if passes.max() > 0 else 1.0
        print("[mixture] realized: " + "  ".join(
            f"{args._source_names[s]}={realized[s]:.3f} "
            f"({passes[s]:.3f} passes/ep, {passes[s] / base:.1f}x rarest-rate)"
            for s in range(n_src)))
        train_sampler = DistributedWeightedSampler(
            weights,
            num_replicas=(dist.get_world_size() if ddp else 1),
            rank=(dist.get_rank() if ddp else 0),
            num_samples=(args.samples_per_epoch or len(train_ds)),
            seed=42,   # rank-independent: all ranks draw the same multinomial
        )
    elif ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_ds)
    # evaluator.py never all-reduces/all-gathers across ranks — sharding val
    # with DistributedSampler would silently score only rank 0's local 1/4
    # of val (confirmed: 196/196 batches at batch=32 == 24,964/4), making
    # DDP eval numbers incomparable to every single-GPU reference (full_v1/
    # v2). Every rank evaluates the full val set independently instead;
    # since all 4 GPUs run it in parallel this costs no extra wall-clock,
    # only redundant (already-reserved) compute.
    val_sampler = torch.utils.data.SequentialSampler(val_ds)

    # Static shapes: pad every batch's box dimension to max_objects instead of
    # the batch's own max. Makes the sampler's N*N pair grid and stage-1
    # top-k budget compile-time constants (see data/relation_dataset.collate_fn
    # docstring) — no metric change, just uniform padding. Off by default so
    # existing recipes' memory/throughput profile doesn't shift silently.
    _collate = (functools.partial(collate_fn, pad_to=args.max_objects)
                if args.static_shapes else collate_fn)
    # Multi-scale: one resolution per BATCH, so it has to come from a
    # batch_sampler rather than a transform (see data/multiscale.py). When it is
    # on, DataLoader takes batch_sampler and must NOT also take
    # batch_size/sampler/drop_last. train_sampler is rebound to the wrapper so
    # the existing set_epoch call below reaches it (it forwards to the inner one).
    _ms_res = None
    if args.multi_scale:
        from data.multiscale import MultiScaleBatchSampler, scale_ladder
        try:
            _lo, _hi = (float(x) for x in args.multi_scale.split(","))
        except ValueError:
            raise SystemExit(f"--multi_scale wants 'lo,hi', got {args.multi_scale!r}")
        _ms_res = scale_ladder(args.img_size, _lo, _hi, args.multi_scale_n)
        train_sampler = MultiScaleBatchSampler(
            train_sampler, args.batch_size, _ms_res, drop_last=True, seed=42)
        if is_main_process():
            _tok = [(r // 16) ** 2 for r in _ms_res]
            print(f"[multi-scale] {len(_ms_res)} rungs {_ms_res} "
                  f"(base {args.img_size}); patch tokens {min(_tok)}-{max(_tok)}, "
                  f"mean relative FLOPs "
                  f"{sum(r * r for r in _ms_res) / len(_ms_res) / args.img_size ** 2:.3f}x")
    train_loader = DataLoader(
        train_ds,
        collate_fn=_collate, num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        **({"batch_sampler": train_sampler} if _ms_res else
           {"batch_size": args.batch_size, "sampler": train_sampler,
            "drop_last": True}),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        collate_fn=_collate, num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    # Recall-eval view of val: optionally capped (--val_eval_limit) and at a
    # larger, gradient-free batch size (--eval_batch_size). val_loss keeps
    # using val_loader so the loss-gap diagnostic stays on the same
    # distribution slice it has always used.
    _eval_bs = args.eval_batch_size or args.batch_size
    if args.val_eval_limit > 0 or _eval_bs != args.batch_size:
        _val_eval_ds = (torch.utils.data.Subset(
            val_ds, range(min(args.val_eval_limit, len(val_ds))))
            if args.val_eval_limit > 0 else val_ds)
        val_eval_loader = DataLoader(
            _val_eval_ds, batch_size=_eval_bs,
            sampler=torch.utils.data.SequentialSampler(_val_eval_ds),
            collate_fn=_collate, num_workers=args.num_workers,
            pin_memory=True,
        )
    else:
        val_eval_loader = val_loader

    # ---- Out-of-domain dev split for checkpoint selection ----
    # The target vocabulary is encoded ONCE here: it is fixed for the whole
    # run, and re-loading the text encoder every epoch would be pure waste.
    # It must be encoded with the SAME tower the head was trained against —
    # scoring a student-trained head with teacher embeddings is meaningless
    # (the trap eval_zeroshot.py's --text_student default guards against).
    dev = None
    if args.dev_root:
        from relsgg.api import TRAIN_TEMPLATES
        dev_ds = RelationDataset(
            root=args.dev_root, split=args.dev_split,
            resolution=args.img_size, max_objects=args.max_objects,
            # Dev must match the TRAINING modality. Selecting a mask-trained
            # model on box-mode dev would score it in a regime it never saw.
            # CAVEAT for the dropout arm: training is half box / half mask but
            # selection here is mask-mode, so it tilts toward the mask path —
            # which is why the final checkpoints are scored BOTH ways rather
            # than trusting checkpoint_best alone.
            rasters=args.rasters,
        )
        if not args.text_student:
            raise SystemExit(
                "--dev_root needs --text_student: this run trains in student "
                "text space, and encoding the dev vocabulary with the teacher "
                "tower would score the head against a space it never saw.")
        from relsgg.text_student import encode_texts_student
        dev = {
            "name": os.path.basename(os.path.normpath(args.dev_root)),
            "pred_names": dev_ds.predicate_names,
            "E": encode_texts_student(dev_ds.predicate_names,
                                      args.text_student,
                                      templates=TRAIN_TEMPLATES, device=device),
            "loader": DataLoader(
                dev_ds, batch_size=args.eval_batch_size or args.batch_size,
                sampler=torch.utils.data.SequentialSampler(dev_ds),
                collate_fn=_collate, num_workers=args.num_workers,
                pin_memory=True),
            "budget": args.dev_budget,
            "score_mode": ("sigmoid" if args.loss_type in ("synonym", "batch_infonce")
                           else "softmax"),
        }
        if is_main_process():
            print(f"[dev] {dev['name']}/{args.dev_split}: {len(dev_ds):,} imgs, "
                  f"{len(dev['pred_names'])} predicates — scored every epoch"
                  + (f", SELECTING checkpoint_best on dev_{args.dev_metric}"
                     if args.dev_select else " (reported only)"))

    # ---- Model ----
    if args.data_roots:
        obj_names = args._union_categories       # set in build_datasets above
    else:
        obj_names = train_ds.meta.get("categories")
    model = build_model(args, pred_names, obj_names=obj_names).to(device)
    if args.init_from:
        apply_init_from(model, args.init_from, learn_W=args.learn_W)
    if getattr(args, "freeze_trunk", False):
        assert args.init_from and args.mode_gated, \
            "--freeze_trunk needs --init_from and --mode_gated"
        for _n, _p in model.named_parameters():
            _p.requires_grad_(_n.startswith(MODE_ADAPTER_PREFIXES))
        _tr = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"[freeze_trunk] trainable tensors ({len(_tr)}): {_tr}")
    if ddp:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], find_unused_parameters=True
        )
    raw_model = model.module if ddp else model
    if args.restrict_neg_sources:
        assert args.data_roots, "--restrict_neg_sources needs --data_roots"
        _names = list(args._source_names)
        _idx = {pn: i for i, pn in enumerate(pred_names)}
        _allow = torch.ones(len(_names), len(pred_names), dtype=torch.bool)
        for _s in args.restrict_neg_sources:
            assert _s in _names, f"--restrict_neg_sources {_s!r} not in {_names}"
            _si = _names.index(_s)
            _meta = json.load(open(os.path.join(args.data_roots[_si], "train", "meta.json")))
            _own = [_idx[pn] for pn in _meta["predicates"] if pn in _idx]
            _allow[_si] = False
            _allow[_si, _own] = True
            print(f"[negmask] {_s}: anchors contrast against its own "
                  f"{len(_own)}/{len(pred_names)} predicates only "
                  f"(InfoNCE + sigmoid-aux + bg)")
        raw_model.source_col_allow = _allow.to(device)

    optimizer = build_optimizer(raw_model, args)
    # Optimizer steps per epoch, not micro-batches: under --grad_accum the
    # scheduler steps only at accumulation boundaries.
    scheduler = build_scheduler(
        optimizer, args,
        steps_per_epoch=math.ceil(len(train_loader) / max(args.grad_accum, 1)))
    # bf16 needs no loss scaling (fp32-range exponent); the GradScaler exists
    # only for fp16 — whose overflows NaN'd full-run 6850450 at peak LR.
    args.amp_dtype_t = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = (torch.amp.GradScaler("cuda")
              if args.amp and args.amp_dtype_t is torch.float16
              and device.type == "cuda" else None)

    # ---- EMA ----
    ema: Optional[ModelEMA] = None
    if args.ema_decay > 0:
        ema = ModelEMA(raw_model, decay=args.ema_decay)
        print(f"EMA enabled  decay={args.ema_decay}")

    # ---- Resume ----
    start_epoch = 0
    best_recall = 0.0
    recall_key  = "mR@50"
    if args.resume and os.path.isfile(args.resume):
        # weights_only=False: our checkpoints carry the argparse Namespace and
        # ontology objects, which torch>=2.6 refuses under the new default.
        # Latent until a run actually resumed — a crashed arm left a
        # checkpoint_last.pth behind and every relaunch then died on load.
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_recall = ckpt.get("best_recall", 0.0)
        if ema is not None and "ema_model" in ckpt:
            ema.load_state_dict(ckpt["ema_model"])
        print(f"Resumed from {args.resume!r} (epoch {start_epoch})")

    # ---- Soft (synonym-aware) evaluation setup ----
    # SoftR@K is the primary MEGASG-val metric: strict matching counts a
    # correctly-predicted synonym as a miss on this vocabulary.
    soft_matrix = soft_group_of = None
    _ont = getattr(raw_model, "ontology", None)
    if _ont is not None:
        recall_key = "SoftmR@50"  # strict mR punishes synonyms on this vocab
        _emb = np.load(args.pred_embeds)["embeddings"] if args.pred_embeds else None
        soft_matrix = build_match_matrix(
            _ont.group_of, _emb, tau_eval=args.tau_eval,
            inverse_mask=_ont.inverse_mask,
        )
        soft_group_of = _ont.group_of

    # Out-of-domain selection overrides the in-domain key (see --dev_select).
    if args.dev_select:
        if dev is None:
            raise SystemExit("--dev_select needs --dev_root")
        recall_key = f"dev_{args.dev_metric}"

    # ---- Training loop ----
    history: List[dict] = []
    _hist_path = os.path.join(args.output_dir, "history.json")
    if start_epoch > 0 and os.path.isfile(_hist_path):
        with open(_hist_path) as f:
            history = [r for r in json.load(f) if r["epoch"] < start_epoch]

    stop_at = args.stop_after_epoch or args.epochs
    if stop_at < args.epochs and is_main_process():
        print(f"[gate] stopping after epoch {stop_at} of the {args.epochs}-epoch "
              f"schedule (LR schedule UNCHANGED). Resume with "
              f"--resume <ckpt> --epochs {args.epochs} to finish.")
    for epoch in range(start_epoch, stop_at):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # -- Train -- (scheduler now steps per optimizer step, inside)
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, epoch, args, device,
            ema=ema, scheduler=scheduler, monitor=monitor,
        )

        if is_main_process():
            print(
                f"\n[epoch {epoch}] train  "
                + "  ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items())
            )

        # -- Eval (use EMA model when available) --
        eval_model = ema.ema_model if ema is not None else model
        evaluator = SGClsEvaluator(
            topk=[20, 50, 100], num_predicates=len(pred_names),
            score_mode="sigmoid" if args.loss_type in ("synonym", "batch_infonce") else "softmax",
            # Match the dev evaluator and every reported number. Unconstrained,
            # top-50 is a flat ranking over pairs x predicates, so a few
            # high-scoring pairs fill every slot with their own synonym cloud
            # and spatial classes read ~0 ([[relsgg-intraining-eval-artifact]]).
            # NOTE this only removes the SMALLER half of that artifact; the
            # larger half is the 19,103-column vocabulary below.
            graph_constraint=True,
        )
        eval_fan = evaluator
        _score_mode = ("sigmoid" if args.loss_type in ("synonym", "batch_infonce")
                       else "softmax")
        if soft_matrix is not None:
            _evs = [
                evaluator,
                SoftSGClsEvaluator(
                    soft_matrix, soft_group_of, topk=[20, 50, 100],
                    score_mode=_score_mode, graph_constraint=True,
                ),
            ]
            if args.fast_bilinear:
                # Score the separable head in the same pass — how close does
                # the two-matmul export path get to the transformer head?
                _evs.append(KeySwapEvaluator(SoftSGClsEvaluator(
                    soft_matrix, soft_group_of, topk=[20, 50, 100],
                    score_mode=_score_mode, graph_constraint=True,
                )))
            eval_fan = FanoutEvaluator(_evs)
        eval_metrics = evaluate(
            eval_model, val_eval_loader, device, args, eval_fan,
            eval_budget=args.eval_budget
        )

        # -- Val loss (train/val loss-GAP diagnostic; distinct from the
        # recall-based evaluate() above, which forwards with targets=None and
        # so never measures whether the loss itself generalizes) --
        if args.val_loss_batches > 0:
            val_loss_metrics = evaluate_loss(
                eval_model, val_loader, device, args,
                max_batches=args.val_loss_batches,
            )
            eval_metrics.update(val_loss_metrics)
            if is_main_process():
                print(
                    f"[epoch {epoch}] val_loss "
                    + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(val_loss_metrics.items()))
                )

        # -- Out-of-domain dev eval (the selection signal; see --dev_root) --
        if dev is not None:
            dev_metrics, dev_per_class = zeroshot_dev_metrics(
                eval_model, dev, args, device)
            eval_metrics.update(dev_metrics)
            if is_main_process():
                print(
                    f"[epoch {epoch}] dev({dev['name']}) "
                    + "  ".join(f"{k[4:]}: {v:.4f}"
                                for k, v in sorted(dev_metrics.items())
                                if k[4:] in ("R@20", "R@50", "R@100",
                                             "mR@20", "mR@50", "mR@100"))
                )
                with open(os.path.join(args.output_dir,
                                       "dev_per_class_recall.json"), "w") as f:
                    json.dump({"epoch": epoch, "dev": dev["name"],
                               "classes": dev_per_class}, f, indent=2)
                # Spatial sentinels, printed every epoch. These are the classes
                # the megasg-val table renders as ~0.000; here they are honest.
                _sent = {r["name"]: r for r in dev_per_class}
                _row = "  ".join(
                    f"{n}: {_sent[n]['recall']:.3f}({_sent[n]['gt']})"
                    for n in ("in front of", "behind", "to the left of",
                              "to the right of", "above", "below", "on")
                    if n in _sent)
                if _row:
                    print(f"[epoch {epoch}] dev spatial  {_row}")

        if monitor is not None:
            monitor.log_epoch(epoch, train_metrics, eval_metrics)
            monitor.render()

        if is_main_process():
            print(
                f"[epoch {epoch}] eval   "
                + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(eval_metrics.items()))
            )

        # -- Embedding analysis (main process only) --
        # embed_max_batches=0 opts out entirely (e.g. throughput benchmarks);
        # otherwise forced on the last epoch regardless of embed_eval_every.
        run_embed = (args.embed_max_batches > 0
                     and ((epoch + 1) % args.embed_eval_every == 0
                          or epoch == args.epochs - 1))
        if is_main_process() and run_embed:
            print(f"[epoch {epoch}] running embedding analysis...")
            analyzer = EmbeddingAnalyzer(
                pred_names=pred_names,
                max_per_class=args.embed_max_per_class,
            )
            embed_src = ema.ema_model if ema is not None else raw_model
            collect_embeddings(
                embed_src, val_loader, device, args, analyzer,
                max_batches=args.embed_max_batches,
            )
            embed_metrics = analyzer.compute(args.output_dir, epoch)
            if embed_metrics:
                print(
                    f"[epoch {epoch}] embed  "
                    + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(embed_metrics.items()))
                )
                if args.wandb:
                    import wandb
                    wandb.log({**{f"embed/{k}": v for k, v in embed_metrics.items()},
                               "epoch": epoch})

        # -- Checkpoint (main process only) --
        if is_main_process():
            row = {"epoch": epoch, **train_metrics, **eval_metrics}
            history.append(row)
            with open(os.path.join(args.output_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            save_training_plots(history, args.output_dir)

            # -- Per-class recall @50 --
            per_class_stats = evaluator.compute_per_class(50, pred_names)
            with open(os.path.join(args.output_dir, "per_class_recall.json"), "w") as f:
                json.dump({"epoch": epoch, "classes": per_class_stats}, f, indent=2)
            _lines = [f"[epoch {epoch}] per-class @50  (tp/gt  recall)"]
            for _row in per_class_stats:
                bar = "█" * int(_row["recall"] * 20)
                _lines.append(
                    f"  {_row['name']:24s}  {_row['tp']:>5}/{_row['gt']:<5}"
                    f"  {_row['recall']:.3f}  {bar}"
                )
            print("\n".join(_lines))

            if args.wandb:
                # train/eval scalars are mirrored by TrainMonitor.log_epoch;
                # only the per-class table is logged here.
                import wandb
                wandb.log({"per_class_table": wandb.Table(
                    columns=["predicate", "tp", "gt", "recall@50"],
                    data=[[r["name"], r["tp"], r["gt"], r["recall"]]
                          for r in per_class_stats],
                ), "epoch": epoch})

            current_recall = eval_metrics.get(recall_key, 0.0)
            state = {
                "epoch":       epoch,
                "model":       raw_model.state_dict(),
                "ema_model":   ema.state_dict() if ema is not None else None,
                "optimizer":   optimizer.state_dict(),
                "scheduler":   scheduler.state_dict(),
                "best_recall": best_recall,
                "pred_names":  pred_names,
                "args":        vars(args),
            }
            if not args.no_save_checkpoint:
                save_checkpoint(state, args.output_dir, "checkpoint_last.pth")

            if current_recall > best_recall:
                best_recall = current_recall
                # --no_save_best still TRACKS the best epoch and prints it; it
                # only skips the ~1-2GB write. Correct whenever the run will be
                # ranked on the FINAL epoch, which is our protocol (FINAL
                # predicts OOD better, rho .771 vs .657 —
                # [[relsgg-final-not-best-epoch]]) and what the chained OVS
                # evals pin via CKPT_NAME=checkpoint_last.pth. Exists because
                # the 500G quota has already killed two arms mid-checkpoint-
                # write, and a dead `best` copy is pure quota
                # ([[mimer-disk-quota-cleanup]]).
                if not args.no_save_checkpoint and not args.no_save_best:
                    save_checkpoint(state, args.output_dir, "checkpoint_best.pth")
                print(f"[epoch {epoch}] ★ new best {recall_key}: {best_recall:.4f}")

    if is_main_process():
        print(f"\nTraining complete.  Best {recall_key}: {best_recall:.4f}")
        print(f"Checkpoints in: {args.output_dir}")

    # ---- Zero-shot VG150 eval (optional, main process only) ----
    if is_main_process() and args.vg150_root and os.path.isdir(args.vg150_root):
        print(f"\n[zero-shot] Loading VG150 from {args.vg150_root} ...")
        vg_val = RelationDataset(
            root=args.vg150_root, split="val",
            resolution=args.img_size, max_objects=args.max_objects,
        )
        vg_pred_names = vg_val.predicate_names
        vg_loader = DataLoader(
            vg_val, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
        )
        print(f"[zero-shot] Reparametrizing to VG150 vocabulary ({len(vg_pred_names)} predicates) ...")
        # Use the best EMA / raw model checkpoint (already in best_recall state)
        eval_model = ema.ema_model if ema is not None else raw_model
        if args.dinotxt_weights:
            eval_model.vocab_head.encode_vocabulary_dinotxt(
                vg_pred_names,
                dinotxt_weights=args.dinotxt_weights,
                bpe_path_or_url=args.dinotxt_bpe,
                backbone_weights=None,
            )
        else:
            eval_model.encode_vocabulary(vg_pred_names)
        eval_model.reparameterize()
        vg_ev = SGClsEvaluator(topk=[20, 50, 100], num_predicates=len(vg_pred_names))
        vg_metrics = evaluate(
            eval_model, vg_loader, device, args, vg_ev,
            eval_budget=args.eval_budget,
        )
        print(
            "[zero-shot] VG150 results:  "
            + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(vg_metrics.items()))
        )
        with open(os.path.join(args.output_dir, "vg150_zeroshot.json"), "w") as f:
            json.dump(vg_metrics, f, indent=2)
        print(f"[zero-shot] Saved → {args.output_dir}/vg150_zeroshot.json")

    if is_main_process() and args.wandb:
        import wandb
        wandb.finish()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
