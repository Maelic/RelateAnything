#!/usr/bin/env python3
"""Train RelateAnything on packed relation datasets.

    torchrun --nproc_per_node=4 train.py \
        --data_roots runs/packed/megasg_clean runs/packed/vg_raw runs/packed/hicodet \
        --mix_fractions 0.7274 0.063 0.2096 --restrict_neg_sources hicodet \
        --val_root runs/packed/megasg --dev_root runs/packed/psg \
        --pred_embeds runs/packed/datamix_v22/text_space/pred_embeds_studentv2_512_photo.npz \
        --ontology_meta runs/packed/datamix_v22/text_space/union_meta.json \
        --soft_supervision runs/packed/datamix_v22/text_space/soft_supervision.npz \
        --neg_rate_table runs/packed/datamix_v22/pair_opportunity.npz \
        --text_student runs/packed/text_student_v2_512/student.pt \
        --exclude_ids runs/datamix/indoorvg_holdout.json \
        --output_dir runs/train/relsgg-vits16plus

Every default is the released recipe (``training/configs/``); the backbone
is chosen with ``--backbone_model``. See docs/training.md.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).parent))

from relsgg.config import RelSGGConfig                                   # noqa: E402
from relsgg.data import RelationDataset, collate_fn                        # noqa: E402
from relsgg.data.multipack import (DistributedWeightedSampler,             # noqa: E402
                                   build_mixture_datasets,
                                   sample_weights_from_fractions)
from relsgg.data.multiscale import MultiScaleBatchSampler, scale_ladder    # noqa: E402
from relsgg.eval.evaluator import (FanoutEvaluator, SGClsEvaluator,        # noqa: E402
                                   SoftSGClsEvaluator, build_match_matrix)
from relsgg.text.student import encode_texts_student                       # noqa: E402
from relsgg.training.embeddings import EmbeddingAnalyzer                   # noqa: E402
from relsgg.training.engine import (collect_embeddings, evaluate,          # noqa: E402
                                    evaluate_loss, train_one_epoch)
from relsgg.training.monitor import TrainMonitor                           # noqa: E402
from relsgg.training.setup import (ModelEMA, apply_init_from,              # noqa: E402
                                   build_model, build_optimizer,
                                   build_scheduler, inherit_init_args,
                                   source_column_mask, zeroshot_dev_metrics)
from relsgg.vocabulary import TRAIN_TEMPLATES                              # noqa: E402

_CFG = RelSGGConfig()


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# ==========================================================================
# arguments
# ==========================================================================

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RelateAnything trainer",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--data_roots", nargs="+", required=True,
                   help="Packed training sources, trained jointly over the union vocabulary "
                        "(training/pack_megasg.py). The first is the base pack.")
    g.add_argument("--mix_fractions", nargs="+", type=float, default=None,
                   help="Per-source sampling fractions (default: equal).")
    g.add_argument("--val_root", default=None,
                   help="Pack whose val/ split is the in-domain validation set "
                        "(default: the first data root).")
    g.add_argument("--dev_root", default=None, metavar="PACK",
                   help="Out-of-domain pack scored every epoch (e.g. runs/packed/psg); "
                        "checkpoint_best is selected on it.")
    g.add_argument("--dev_split", default="val")
    g.add_argument("--dev_budget", type=int, default=500)
    g.add_argument("--dev_metric", default="mR@50", help="Selection metric on the dev pack.")
    g.add_argument("--exclude_ids", default=None, metavar="JSON",
                   help="Image stems removed from every training source "
                        "(training/build_indoorvg_holdout.py).")
    g.add_argument("--pred_embeds", required=True,
                   help="Union vocabulary embeddings (training/build_union_vocab.py): "
                        "defines the predicate order and the initial W.")
    g.add_argument("--ontology_meta", default=None,
                   help="union_meta.json of the mixture (default: <base>/train/meta.json).")
    g.add_argument("--soft_supervision", required=True,
                   help="soft_supervision.npz (training/build_soft_supervision.py).")
    g.add_argument("--neg_rate_table", default="",
                   help="pair_opportunity.npz (training/build_pair_opportunity.py): per "
                        "category pair negative weights for the relatedness loss.")
    g.add_argument("--text_student", required=True,
                   help="Text student checkpoint (training/distill/), used to encode object "
                        "categories and the dev vocabulary.")
    g.add_argument("--restrict_neg_sources", nargs="*", default=[],
                   help="Sources (pack basenames) whose anchors are contrasted only against "
                        "their own vocabulary.")
    g.add_argument("--drop_predicates", default="",
                   help="Predicates held out of training entirely: a.json list or a "
                        "comma-separated list (the open-vocabulary relation arm).")
    g.add_argument("--img_size", type=int, default=448)
    g.add_argument("--max_objects", type=int, default=40)
    g.add_argument("--static_shapes", action="store_true",
                   help="Pad every batch to --max_objects boxes (constant shapes).")
    g.add_argument("--augment", type=float, default=0.3,
                   help="Photometric jitter strength on training images (0 = off).")
    g.add_argument("--multi_scale", default="0.5,1.5",
                   help="Per-batch square resize range relative to --img_size ('' = off).")
    g.add_argument("--multi_scale_n", type=int, default=7, help="Rungs in the scale ladder.")
    g.add_argument("--rasters", default=None,
                   help="Root of region rasters (datagen/build_mask_rasters.py) to train "
                        "with masks; unset = boxes.")
    g.add_argument("--mask_dropout", type=float, default=0.0,
                   help="With --rasters: per-image probability of using the boxes instead.")
    g.add_argument("--samples_per_epoch", type=int, default=0,
                   help="Draws per epoch (0 = size of the mixture).")
    g.add_argument("--train_subset_frac", type=float, default=1.0,
                   help="Train on a deterministic random fraction of the images.")
    g.add_argument("--train_subset_seed", type=int, default=42)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--num_workers", type=int, default=16)

    g = p.add_argument_group("model (defaults are the released recipe)")
    g.add_argument("--backbone_model", default=_CFG.backbone_model,
                   help="DINOv3 tower: hub id or a local directory.")
    for name, default in (("d_model", _CFG.d_model), ("text_dim", _CFG.text_dim),
                          ("geo_budget", _CFG.geo_budget), ("final_budget", _CFG.final_budget),
                          ("n_self_layers", _CFG.n_self_layers), ("n_cross_layers", _CFG.n_cross_layers),
                          ("n_dep_layers", _CFG.n_dep_layers), ("n_gnd_layers", _CFG.n_gnd_layers),
                          ("n_heads", _CFG.n_heads), ("deformable_points", _CFG.deformable_points),
                          ("deformable_heads", _CFG.deformable_heads),
                          ("deformable_nulls", _CFG.deformable_nulls),
                          ("pe_num_freqs", _CFG.pe_num_freqs), ("proj_layers", _CFG.proj_layers),
                          ("bg_topk", _CFG.bg_topk)):
        g.add_argument(f"--{name}", type=int, default=default)
    for name, default in (("ffn_ratio", _CFG.ffn_ratio), ("dropout", _CFG.dropout),
                          ("pe_max_octave", _CFG.pe_max_octave),
                          ("logit_scale_init", _CFG.logit_scale_init),
                          ("infonce_temp", _CFG.infonce_temp),
                          ("rel_neg_weight", _CFG.rel_neg_weight),
                          ("box_token_dropout", _CFG.box_token_dropout),
                          ("lambda_geo", _CFG.lambda_geo), ("lambda_rel", _CFG.lambda_rel),
                          ("lambda_obj", _CFG.lambda_obj), ("lambda_swap", _CFG.lambda_swap),
                          ("swap_margin", _CFG.swap_margin),
                          ("lambda_sigmoid", _CFG.lambda_sigmoid), ("lambda_bg", _CFG.lambda_bg),
                          ("cfa_prob", _CFG.cfa_prob), ("cfa_alpha", _CFG.cfa_alpha)):
        g.add_argument(f"--{name}", type=float, default=default)
    g.add_argument("--n_neg", type=int, default=512,
                   help="Sampled negative columns per batch in the contrastive loss.")

    g = p.add_argument_group("optimisation")
    g.add_argument("--output_dir", default="./runs/train/exp")
    g.add_argument("--epochs", type=int, default=12)
    g.add_argument("--batch_size", type=int, default=32, help="Per process.")
    g.add_argument("--lr", type=float, default=4e-4)
    g.add_argument("--backbone_lr", type=float, default=5e-5)
    g.add_argument("--weight_decay", type=float, default=1e-4)
    g.add_argument("--clip_grad", type=float, default=1.0)
    g.add_argument("--warmup_steps", type=int, default=500)
    g.add_argument("--min_lr_factor", type=float, default=0.01)
    g.add_argument("--grad_accum", type=int, default=1,
                   help="Micro-batches per optimiser step (N reproduces an N-GPU run).")
    g.add_argument("--amp", action="store_true", default=True)
    g.add_argument("--no_amp", action="store_false", dest="amp")
    g.add_argument("--amp_dtype", default="bf16", choices=["bf16", "fp16"])
    g.add_argument("--ema_decay", type=float, default=0.9998, help="0 disables the EMA.")
    g.add_argument("--dead_param_audit", type=int, default=100,
                   help="Fail if a trainable parameter gets no gradient in this many steps "
                        "(0 = off).")
    g.add_argument("--dead_param_warn", action="store_true", help="Warn instead of failing.")
    g.add_argument("--init_from", default="",
                   help="Weights-only initialisation from a finished run (fine-tuning); "
                        "the network and recipe keys are inherited from its args.")
    g.add_argument("--resume", default="", help="Checkpoint to resume from.")
    g.add_argument("--stop_after_epoch", type=int, default=0,
                   help="Stop after this many epochs while keeping the --epochs schedule.")

    g = p.add_argument_group("evaluation and logging")
    g.add_argument("--eval_budget", type=int, default=400)
    g.add_argument("--val_eval_limit", type=int, default=0,
                   help="Images of the val split scored per epoch (0 = all).")
    g.add_argument("--eval_batch_size", type=int, default=0, help="0 = --batch_size.")
    g.add_argument("--val_loss_batches", type=int, default=100)
    g.add_argument("--tau_eval", type=float, default=0.72,
                   help="Text cosine at which a predicate counts as a synonym (SoftR@K).")
    g.add_argument("--embed_eval_every", type=int, default=5)
    g.add_argument("--embed_max_per_class", type=int, default=500)
    g.add_argument("--embed_max_batches", type=int, default=0,
                   help="Batches for the embedding analysis (0 = off).")
    g.add_argument("--no_save_best", action="store_true")
    g.add_argument("--no_save_checkpoint", action="store_true")
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--wandb", action="store_true", default=False)
    g.add_argument("--wandb_project", default="relateanything")
    g.add_argument("--wandb_run_name", default="")
    g.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)))
    return p.parse_args(argv)


# ==========================================================================
# data
# ==========================================================================

def _load_exclude_ids(path: Optional[str]) -> Optional[set]:
    if not path:
        return None
    obj = json.load(open(path))
    stems = {str(s) for s in (obj["stems"] if isinstance(obj, dict) else obj)}
    print(f"[exclude] {len(stems):,} held-out image stems from {path}")
    return stems


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, List[str]]:
    """``(train, val, predicate_names)`` over the union vocabulary."""
    z = np.load(args.pred_embeds)
    union_predicates = [str(p) for p in z["predicates"]]
    if args.drop_predicates:
        drop = (json.load(open(args.drop_predicates)) if args.drop_predicates.endswith(".json")
                else [x.strip() for x in args.drop_predicates.split(",") if x.strip()])
        drop = {str(d) for d in drop}
        if not drop:
            raise SystemExit("--drop_predicates is empty")
        missing = drop - set(union_predicates)
        if missing:
            print(f"[drop_predicates] {len(missing)} strings absent from the vocabulary: "
                  f"{sorted(missing)}")
        # Removing the string from the vocabulary holds the predicate out
        # completely: its relations are dropped by the loader and the head
        # gets no row for it, while it stays scorable from text at inference.
        union_predicates = [q for q in union_predicates if q not in drop]
        print(f"[drop_predicates] held out {len(drop - missing)} predicates")
    ontology_meta = args.ontology_meta or os.path.join(args.data_roots[0], "train", "meta.json")
    union_categories = json.load(open(ontology_meta))["categories"]
    train_ds, val_ds, source_of_index, source_names = build_mixture_datasets(
        base_root=args.data_roots[0], extra_roots=args.data_roots[1:],
        union_predicates=union_predicates, resolution=args.img_size,
        max_objects=args.max_objects, union_categories=union_categories,
        val_root=args.val_root, augment=args.augment,
        exclude_ids=_load_exclude_ids(args.exclude_ids),
        rasters=args.rasters, mask_dropout=args.mask_dropout)
    if args.train_subset_frac < 1.0:
        n = len(train_ds)
        k = int(round(n * args.train_subset_frac))
        keep = np.sort(np.random.default_rng(args.train_subset_seed).permutation(n)[:k])
        train_ds = torch.utils.data.Subset(train_ds, keep.tolist())
        source_of_index = source_of_index[keep]
        print(f"[subset] train restricted to {k:,}/{n:,} images")
    args._source_of_index = source_of_index
    args._source_names = source_names
    args._union_categories = union_categories
    return train_ds, val_ds, union_predicates


def save_training_plots(history: List[dict], output_dir: str) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    epochs = [row["epoch"] for row in history]
    loss_keys = [k for k in history[0] if k.startswith("loss_") and k != "loss_total"]
    has_val = "val_loss_total" in history[0]
    fig, axes = plt.subplots(1, 3 if has_val else 2, figsize=(21 if has_val else 14, 5))
    ax = axes[0]
    for k in loss_keys:
        ax.plot(epochs, [row.get(k, float("nan")) for row in history], label=k[5:], linewidth=1.5)
    ax.plot(epochs, [row.get("loss_total", float("nan")) for row in history],
            label="total", linewidth=2, linestyle="--", color="black")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Training losses")
    ax.legend(fontsize=8, ncol=2); ax.grid(True, alpha=0.3)
    if has_val:
        ax = axes[1]
        ax.plot(epochs, [row.get("loss_total", float("nan")) for row in history], label="train")
        ax.plot(epochs, [row.get("val_loss_total", float("nan")) for row in history],
                label="val", marker="o", markersize=3)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Train vs val loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax = axes[-1]
    for key, ls in [("mR@50", "-"), ("R@50", "--"), ("dev_mR@50", ":"), ("dev_R@50", ":")]:
        if key in history[0]:
            ax.plot(epochs, [row.get(key, float("nan")) for row in history], label=key, linestyle=ls)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Recall"); ax.set_title("Recall (dotted: dev)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curves.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)


# ==========================================================================
# main
# ==========================================================================

def main() -> None:
    args = parse_args()
    if args.init_from:
        inherit_init_args(args)

    ddp = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if ddp:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=30))
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = args.seed + (dist.get_rank() if ddp else 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    monitor = None
    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        wandb_run = None
        if args.wandb:
            try:
                import wandb
                wandb_run = wandb.init(project=args.wandb_project,
                                       name=args.wandb_run_name or Path(args.output_dir).name,
                                       config=vars(args), dir=args.output_dir,
                                       mode=os.environ.get("WANDB_MODE", "offline"),
                                       resume="allow")
            except ImportError:
                print("[wandb] not installed; pip install wandb")
                args.wandb = False
        monitor = TrainMonitor(args.output_dir, log_every=args.log_every, wandb_run=wandb_run)

    # ---- data ----
    train_ds, val_ds, pred_names = build_datasets(args)
    print(f"  train: {len(train_ds):,} images   val: {len(val_ds):,} images   "
          f"predicates: {len(pred_names)}")

    soi = args._source_of_index
    n_src = len(args._source_names)
    draws = args.samples_per_epoch or len(train_ds)
    fracs = args.mix_fractions or [1.0] * n_src
    assert len(fracs) == n_src, f"--mix_fractions has {len(fracs)} values for {n_src} sources"
    weights = sample_weights_from_fractions(soi, fracs, draws_per_epoch=draws)
    counts = np.bincount(soi, minlength=n_src)
    realized = np.array([weights[soi == s].sum() for s in range(n_src)])
    print("[mixture] " + "  ".join(
        f"{args._source_names[s]}={realized[s]:.3f} "
        f"({draws * realized[s] / max(counts[s], 1):.2f} passes/epoch)" for s in range(n_src)))
    train_sampler = DistributedWeightedSampler(
        weights, num_replicas=(dist.get_world_size() if ddp else 1),
        rank=(dist.get_rank() if ddp else 0), num_samples=draws, seed=42)
    # Every process scores the full validation set (the evaluators do not
    # reduce across processes).
    val_sampler = torch.utils.data.SequentialSampler(val_ds)
    _collate = (functools.partial(collate_fn, pad_to=args.max_objects)
                if args.static_shapes else collate_fn)

    ms_res = None
    if args.multi_scale:
        lo, hi = (float(x) for x in args.multi_scale.split(","))
        ms_res = scale_ladder(args.img_size, lo, hi, args.multi_scale_n)
        train_sampler = MultiScaleBatchSampler(train_sampler, args.batch_size, ms_res,
                                               drop_last=True, seed=42)
        if is_main_process():
            print(f"[multi-scale] {len(ms_res)} resolutions {ms_res}")
    train_loader = DataLoader(
        train_ds, collate_fn=_collate, num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        **({"batch_sampler": train_sampler} if ms_res else
           {"batch_size": args.batch_size, "sampler": train_sampler, "drop_last": True}))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            collate_fn=_collate, num_workers=args.num_workers,
                            pin_memory=True, persistent_workers=args.num_workers > 0)
    eval_bs = args.eval_batch_size or args.batch_size
    if args.val_eval_limit > 0 or eval_bs != args.batch_size:
        ds = (torch.utils.data.Subset(val_ds, range(min(args.val_eval_limit, len(val_ds))))
              if args.val_eval_limit > 0 else val_ds)
        val_eval_loader = DataLoader(ds, batch_size=eval_bs,
                                     sampler=torch.utils.data.SequentialSampler(ds),
                                     collate_fn=_collate, num_workers=args.num_workers,
                                     pin_memory=True)
    else:
        val_eval_loader = val_loader

    dev = None
    if args.dev_root:
        dev_ds = RelationDataset(root=args.dev_root, split=args.dev_split,
                                 resolution=args.img_size, max_objects=args.max_objects,
                                 rasters=args.rasters)
        dev = {"name": os.path.basename(os.path.normpath(args.dev_root)),
               "pred_names": dev_ds.predicate_names,
               "E": encode_texts_student(dev_ds.predicate_names, args.text_student,
                                         templates=TRAIN_TEMPLATES, device=device),
               "loader": DataLoader(dev_ds, batch_size=eval_bs,
                                    sampler=torch.utils.data.SequentialSampler(dev_ds),
                                    collate_fn=_collate, num_workers=args.num_workers,
                                    pin_memory=True),
               "budget": args.dev_budget}
        if is_main_process():
            print(f"[dev] {dev['name']}/{args.dev_split}: {len(dev_ds):,} images, "
                  f"{len(dev['pred_names'])} predicates; checkpoint_best on dev_{args.dev_metric}")

    # ---- model ----
    model = build_model(args, pred_names, obj_names=args._union_categories).to(device)
    if args.init_from:
        apply_init_from(model, args.init_from)
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    find_unused_parameters=True)
    raw_model = model.module if ddp else model
    allow = source_column_mask(args, pred_names)
    if allow is not None:
        raw_model.source_col_allow = allow.to(device)

    optimizer = build_optimizer(raw_model, args)
    scheduler = build_scheduler(
        optimizer, args, steps_per_epoch=math.ceil(len(train_loader) / max(args.grad_accum, 1)))
    args.amp_dtype_t = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = (torch.amp.GradScaler("cuda") if args.amp and args.amp_dtype_t is torch.float16
              and device.type == "cuda" else None)
    ema = ModelEMA(raw_model, decay=args.ema_decay) if args.ema_decay > 0 else None

    start_epoch, best_recall = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_recall = ckpt.get("best_recall", 0.0)
        if ema is not None and ckpt.get("ema_model"):
            ema.load_state_dict(ckpt["ema_model"])
        print(f"Resumed from {args.resume!r} (epoch {start_epoch})")

    # In-domain validation scores synonyms as matches (SoftR@K); selection
    # uses the dev pack when one is given.
    ont = raw_model.ontology
    soft_matrix = build_match_matrix(ont.group_of, np.load(args.pred_embeds)["embeddings"],
                                     tau_eval=args.tau_eval, inverse_mask=ont.inverse_mask)
    recall_key = f"dev_{args.dev_metric}" if dev is not None else "SoftmR@50"

    history: List[dict] = []
    hist_path = os.path.join(args.output_dir, "history.json")
    if start_epoch > 0 and os.path.isfile(hist_path):
        history = [r for r in json.load(open(hist_path)) if r["epoch"] < start_epoch]

    stop_at = args.stop_after_epoch or args.epochs
    for epoch in range(start_epoch, stop_at):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, epoch, args,
                                        device, ema=ema, scheduler=scheduler, monitor=monitor)
        if is_main_process():
            print(f"\n[epoch {epoch}] train  "
                  + "  ".join(f"{k}: {v:.4f}" for k, v in train_metrics.items()))

        eval_model = ema.ema_model if ema is not None else model
        evaluator = SGClsEvaluator(topk=[20, 50, 100], num_predicates=len(pred_names),
                                   score_mode="sigmoid", graph_constraint=True)
        soft = SoftSGClsEvaluator(soft_matrix, ont.group_of, topk=[20, 50, 100],
                                  score_mode="sigmoid", graph_constraint=True)
        eval_metrics = evaluate(eval_model, val_eval_loader, device, args,
                                FanoutEvaluator([evaluator, soft]), eval_budget=args.eval_budget)
        if args.val_loss_batches > 0:
            eval_metrics.update(evaluate_loss(eval_model, val_loader, device, args,
                                              max_batches=args.val_loss_batches))
        if dev is not None:
            dev_metrics, dev_per_class = zeroshot_dev_metrics(eval_model, dev, args, device)
            eval_metrics.update(dev_metrics)
            if is_main_process():
                with open(os.path.join(args.output_dir, "dev_per_class_recall.json"), "w") as f:
                    json.dump({"epoch": epoch, "dev": dev["name"], "classes": dev_per_class}, f,
                              indent=2)
        if monitor is not None:
            monitor.log_epoch(epoch, train_metrics, eval_metrics)
            monitor.render()
        if is_main_process():
            print(f"[epoch {epoch}] eval   "
                  + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(eval_metrics.items())))

        run_embed = (args.embed_max_batches > 0
                     and ((epoch + 1) % args.embed_eval_every == 0 or epoch == args.epochs - 1))
        if is_main_process() and run_embed:
            analyzer = EmbeddingAnalyzer(pred_names=pred_names,
                                         max_per_class=args.embed_max_per_class)
            collect_embeddings(ema.ema_model if ema is not None else raw_model, val_loader,
                               device, args, analyzer, max_batches=args.embed_max_batches)
            embed_metrics = analyzer.compute(args.output_dir, epoch)
            if embed_metrics:
                print(f"[epoch {epoch}] embed  "
                      + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(embed_metrics.items())))

        if is_main_process():
            row = {"epoch": epoch, **train_metrics, **eval_metrics}
            history.append(row)
            with open(hist_path, "w") as f:
                json.dump(history, f, indent=2)
            save_training_plots(history, args.output_dir)
            per_class = evaluator.compute_per_class(50, pred_names)
            with open(os.path.join(args.output_dir, "per_class_recall.json"), "w") as f:
                json.dump({"epoch": epoch, "classes": per_class}, f, indent=2)

            current = eval_metrics.get(recall_key, 0.0)
            state = {"epoch": epoch, "model": raw_model.state_dict(),
                     "ema_model": ema.state_dict() if ema is not None else None,
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "best_recall": best_recall, "pred_names": pred_names, "args": vars(args)}
            if not args.no_save_checkpoint:
                torch.save(state, os.path.join(args.output_dir, "checkpoint_last.pth"))
            if current > best_recall:
                best_recall = current
                if not args.no_save_checkpoint and not args.no_save_best:
                    torch.save(state, os.path.join(args.output_dir, "checkpoint_best.pth"))
                print(f"[epoch {epoch}] new best {recall_key}: {best_recall:.4f}")

    if is_main_process():
        print(f"\nTraining complete. Best {recall_key}: {best_recall:.4f}")
        print(f"Checkpoints in: {args.output_dir}")
        if args.wandb:
            import wandb
            wandb.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
