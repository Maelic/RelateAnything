"""Training and evaluation loops.

Both work on any DataLoader that yields ``(images, boxes, box_counts,
targets)`` batches (see ``relsgg.data``).
"""
from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def region_kwargs(targets, device) -> dict:
    """The batched region rasters of a batch, as explicit tensor keyword
    arguments. They must not travel on the ``targets`` list: DDP rebuilds
    scattered lists as plain lists and would drop the attributes."""
    cov = getattr(targets, "cov", None)
    if cov is None:
        return {}
    fill = getattr(targets, "fill", None)
    return {"cov": cov.to(device, non_blocking=True),
            "fill": None if fill is None else fill.to(device, non_blocking=True)}


def _check_dead_params(model: nn.Module, grad_seen: set, exempt: set,
                       n_steps: int, strict: bool) -> None:
    """Report trainable parameters that received no gradient in the first
    ``n_steps`` optimiser steps. DDP with ``find_unused_parameters`` turns a
    module the loss never reaches into silence; this makes it an error."""
    raw = model.module if hasattr(model, "module") else model
    dead = [n for n, p in raw.named_parameters()
            if p.requires_grad and n not in grad_seen
            and not any(n.startswith(e) for e in exempt)]
    if not dead:
        print(f"[audit] all trainable parameters received gradient within {n_steps} steps")
        return
    msg = (f"[audit] {len(dead)} trainable parameter tensor(s) received no gradient in the "
           f"first {n_steps} steps:\n  " + "\n  ".join(dead[:20])
           + ("\n..." if len(dead) > 20 else "")
           + "\nEither a module is never used by the loss, or a loss path is broken. "
             "Freeze it, fix the path, or pass --dead_param_warn / --dead_param_audit 0.")
    if strict:
        raise RuntimeError(msg)
    print("WARNING " + msg)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    scaler, epoch: int, args, device: torch.device, ema=None,
                    scheduler=None, monitor=None) -> Dict[str, float]:
    """One pass over ``loader``. Returns the mean of every loss component."""
    model.train()
    metric_logger: Dict[str, list] = defaultdict(list)
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t_start = time.time()
    n_images = 0
    step0 = epoch * len(loader)

    audit_left = int(getattr(args, "dead_param_audit", 100)) if epoch == 0 else 0
    audit_strict = not getattr(args, "dead_param_warn", False)
    grad_seen: set = set()
    audit_exempt: set = set()
    accum = max(int(getattr(args, "grad_accum", 1)), 1)

    for it, (images, boxes, box_counts, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        box_counts = box_counts.to(device, non_blocking=True)
        for t in targets:
            if "relations" in t:
                t["relations"] = t["relations"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp,
                                dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
            out = model(images, boxes, box_counts, targets, **region_kwargs(targets, device))
        loss = out["loss"]

        # Gradient accumulation: the mean of N micro-batch gradients equals
        # the mean over N data-parallel ranks, so one GPU with accum N
        # reproduces an N-GPU run (the contrastive set is built per
        # micro-batch either way).
        boundary = ((it + 1) % accum == 0) or (it + 1 == len(loader))
        if accum > 1:
            loss = loss / accum
        if it % accum == 0:
            optimizer.zero_grad(set_to_none=True)
        sync = (model.no_sync() if (not boundary and hasattr(model, "no_sync"))
                else contextlib.nullcontext())
        with sync:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if boundary:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if ema is not None:
                ema.update(model)

        if audit_left > 0:
            raw = model.module if hasattr(model, "module") else model
            if it == 0 and not region_kwargs(targets, device):
                # Box-only training never passes coverage, so the pooling
                # shape gate is inactive by design.
                audit_exempt.add("spatial_pool.cov_lambda")
            grad_seen.update(n for n, p in raw.named_parameters() if p.grad is not None)
            audit_left -= 1
            if audit_left == 0:
                _check_dead_params(model, grad_seen, audit_exempt,
                                   int(getattr(args, "dead_param_audit", 100)), audit_strict)

        for k, v in out.get("loss_dict", {}).items():
            metric_logger[k].append(float(v))
        if scaler is not None:
            metric_logger["amp_scale"].append(scaler.get_scale())
        n_images += images.shape[0]
        if monitor is not None:
            monitor.log_iter(step0 + it, epoch, max(g["lr"] for g in optimizer.param_groups),
                             {k: float(v) for k, v in out.get("loss_dict", {}).items()})
        if it % 20 == 0:
            pbar.set_postfix(loss=f"{loss.detach().item():.4f}")

    metrics = {k: float(np.mean(v)) for k, v in metric_logger.items()}
    metrics["img_per_s"] = n_images / max(time.time() - t_start, 1e-6)
    if device.type == "cuda":
        metrics["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, args,
             evaluator, eval_budget: int = 500) -> Dict[str, float]:
    """Run inference (``targets=None``, so pair sampling is unbiased) and
    feed one or several evaluators. ``eval_budget`` widens the pair budget
    for the duration of the pass."""
    model.eval()
    evaluators = evaluator if isinstance(evaluator, (list, tuple)) else [evaluator]
    raw = model.module if hasattr(model, "module") else model
    orig_budget = raw.sampler.final_budget
    raw.sampler.final_budget = min(eval_budget, raw.sampler.geo_budget)
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    try:
        for images, boxes, box_counts, targets in tqdm(loader, desc="Evaluating", disable=not is_main):
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            box_counts = box_counts.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=args.amp,
                                    dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
                # Region rasters are inputs, not labels: they are passed even
                # without targets.
                out = model(images, boxes, box_counts, targets=None,
                            **region_kwargs(targets, device))
            for ev in evaluators:
                ev.update(out, targets)
    finally:
        raw.sampler.final_budget = orig_budget
    metrics: Dict[str, float] = {}
    for ev in evaluators:
        metrics.update(ev.compute())
    return metrics


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: torch.device, args,
                  max_batches: int = 100) -> Dict[str, float]:
    """The training loss on held-out data (``val_loss_*``), for the
    train / validation gap."""
    model.eval()
    metric_logger: Dict[str, list] = defaultdict(list)
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    for step, (images, boxes, box_counts, targets) in enumerate(
            tqdm(loader, desc="Val loss", disable=not is_main,
                 total=min(max_batches, len(loader)))):
        if step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        box_counts = box_counts.to(device, non_blocking=True)
        for t in targets:
            if "relations" in t:
                t["relations"] = t["relations"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp,
                                dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
            out = model(images, boxes, box_counts, targets, **region_kwargs(targets, device))
        for k, v in out.get("loss_dict", {}).items():
            metric_logger[f"val_{k}"].append(float(v))
    return {k: float(np.mean(v)) for k, v in metric_logger.items()}


@torch.no_grad()
def collect_embeddings(model: nn.Module, loader: DataLoader, device: torch.device, args,
                       analyzer, max_batches: int = 200) -> None:
    """Fill an ``EmbeddingAnalyzer`` with labelled pair features."""
    model.eval()
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    for step, (images, boxes, box_counts, targets) in enumerate(
            tqdm(loader, desc="Collecting embeddings", disable=not is_main)):
        if step >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        box_counts = box_counts.to(device, non_blocking=True)
        for t in targets:
            if "relations" in t:
                t["relations"] = t["relations"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=args.amp,
                                dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
            out = model(images, boxes, box_counts, targets=targets,
                        **region_kwargs(targets, device))
        entity_labels_batch = [t.get("entity_labels") for t in targets]
        if any(el is None for el in entity_labels_batch):
            entity_labels_batch = None
        analyzer.update(out, entity_labels=entity_labels_batch)
