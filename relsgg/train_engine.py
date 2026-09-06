"""Core training and evaluation loops for RelAnything.

These functions are dataset-agnostic — they operate on any DataLoader
that yields ``(images, boxes, box_counts, targets)`` batches.

Functions
---------
train_one_epoch   — one full pass over the training loader.
evaluate          — compute evaluator metrics on a validation loader.
collect_embeddings — fill an EmbeddingAnalyzer with GT-labelled pair features.
"""

from __future__ import annotations

import contextlib
import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm



def _region_kwargs(targets, device):
    """Pull the batched region rasters off the loader's TargetList.

    They MUST be passed to the model as explicit tensor kwargs rather than
    ridden in on `targets`: DistributedDataParallel scatters its inputs, and
    scatter_map rebuilds any list as a PLAIN list
    (`[list(i) for i in zip(*map(scatter_map, obj))]`), silently dropping the
    TargetList subclass and its attributes. That is exactly how jobs 6922420 /
    6922421 trained with masks requested and never received one — cov_lambda
    came back bit-exact 0.0. Tensors, by contrast, scatter correctly.
    """
    cov = getattr(targets, "cov", None)
    fill = getattr(targets, "fill", None)
    mode = getattr(targets, "mode", None)
    if cov is None:
        return {}
    return {"cov": cov.to(device, non_blocking=True),
            "fill": None if fill is None else fill.to(device, non_blocking=True),
            "mode": None if mode is None else mode.to(device, non_blocking=True)}


def _check_dead_params(model: nn.Module, grad_seen: set, exempt: set,
                       n_steps: int, strict: bool) -> None:
    """Raise (or warn) on trainable params that received NO gradient in the
    first ``n_steps`` optimizer steps.

    WHY THIS EXISTS: DDP runs with find_unused_parameters=True, which converts
    "parameter got no gradient" from a crash into silence — the exact mechanism
    by which logit_scale/logit_bias trained as dead weight for ten versions
    ([[relsgg-untrained-output-head]]) and gate_mlp was silently dropped on the
    deploy path. audit_param_health.py exists but is opt-in; this is the
    default-on version that fails FAST (step ~100, before GPU-hours are spent).

    Frozen-by-design params (requires_grad False) are naturally excluded.
    ``exempt`` holds names that are legitimately inactive under the current
    config (e.g. spatial_pool.cov_lambda in a box-mode run).
    """
    raw = model.module if hasattr(model, "module") else model
    dead = [n for n, p in raw.named_parameters()
            if p.requires_grad and n not in grad_seen
            and not any(n.startswith(e) for e in exempt)]
    if not dead:
        print(f"[audit] all trainable params received gradient within "
              f"{n_steps} steps")
        return
    msg = (f"[audit] {len(dead)} trainable parameter tensor(s) received NO "
           f"gradient in the first {n_steps} steps:\n  "
           + "\n  ".join(dead[:20])
           + ("\n  ..." if len(dead) > 20 else "")
           + "\nEither the config builds a module the loss never uses, or a "
             "loss path is broken. Freeze it (requires_grad_(False)), fix the "
             "path, or pass --dead_param_warn / --dead_param_audit 0.")
    if strict:
        raise RuntimeError(msg)
    print("WARNING " + msg)

@torch.no_grad()
def _term_grad_norms(model: nn.Module, out: dict) -> Dict[str, float]:
    """Per-loss-term gradient norms w.r.t. two shared-trunk tensors.

    GradNorm's measurement trick (Chen et al., ICML 2018): compare terms by
    the gradient they induce on ONE shared layer, not the whole model.
    Refs: ``pair_proj.weight`` (everything downstream of pair fusion) and
    ``spatial_pool.cross_attn.out_proj.weight`` (which the relatedness loss
    also reaches — the geometry BCE reaches neither, by design: its head is
    parameter-isolated). Uses ``torch.autograd.grad(inputs=...)`` which does
    NOT populate ``.grad`` or fire DDP reducer hooks, so it is safe before
    the real backward. Costs roughly one extra head-backward per term —
    enable every N steps, not every step.

    WHY: the ~8 loss lambdas are hand constants; before adopting GradNorm or
    uncertainty weighting we measure what the effective weights actually are
    ([[relsgg-ovs-composite]] house rule: quantify before adopting).
    """
    raw = model.module if hasattr(model, "module") else model
    refs = [("pair", raw.pair_proj.weight),
            ("pool", raw.spatial_pool.cross_attn.out_proj.weight)]
    lambdas = out.get("loss_lambdas", {})
    norms: Dict[str, float] = {}
    with torch.enable_grad():
        for k, t in out.get("loss_terms", {}).items():
            if not (torch.is_tensor(t) and t.requires_grad):
                continue
            for tag, ref in refs:
                g = torch.autograd.grad(t, ref, retain_graph=True,
                                        allow_unused=True)[0]
                gn = 0.0 if g is None else float(g.float().norm())
                norms[f"gnorm_{k}_{tag}"] = gn * float(lambdas.get(k, 1.0))
    return norms


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    args,
    device: torch.device,
    ema=None,
    scheduler=None,
    monitor=None,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model:     The model (may be DDP-wrapped).
        loader:    Training DataLoader yielding
                   ``(images, boxes, box_counts, targets)``.
        optimizer: Optimizer.
        scaler:    AMP GradScaler, or None if AMP is disabled.
        epoch:     Current epoch number (for display only).
        args:      Namespace with ``amp``, ``clip_grad``.
        device:    Target device.
        ema:       Optional ModelEMA instance. Updated after every optimizer step.
        monitor:   Optional relsgg.monitor.TrainMonitor (rank 0 only) — gets
                   per-iteration loss/lr rows.
    Returns:
        Dict of mean loss component values over the epoch.
    """
    model.train()
    metric_logger: Dict[str, list] = defaultdict(list)

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _t_start = time.time()
    _n_images = 0
    _step0 = epoch * len(loader)

    # Dead-param audit (epoch 0 only): record which params get gradients over
    # the first N steps, then fail loudly on any that never did. See
    # _check_dead_params for why this is default-on.
    _audit_left = (int(getattr(args, "dead_param_audit", 100))
                   if epoch == 0 else 0)
    _audit_strict = not getattr(args, "dead_param_warn", False)
    _grad_seen: set = set()
    _audit_exempt: set = set()
    _accum = max(int(getattr(args, "grad_accum", 1)), 1)

    for _it, (images, boxes, box_counts, targets) in enumerate(pbar):
        images     = images.to(device, non_blocking=True)
        boxes      = boxes.to(device, non_blocking=True)
        box_counts = box_counts.to(device, non_blocking=True)

        for t in targets:
            if "relations" in t:
                t["relations"] = t["relations"].to(device, non_blocking=True)

        # bf16 default: fp16's narrow exponent overflowed on long peak-LR
        # schedules (full-run NaN @ job 6850450 — amp_scale 25675→1024→11).
        with torch.amp.autocast("cuda", enabled=args.amp,
                                dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
            out = model(images, boxes, box_counts, targets,
                        **_region_kwargs(targets, device))

        loss = out["loss"]

        _tel = int(getattr(args, "grad_telemetry", 0))
        if _tel > 0 and (_step0 + _it) % _tel == 0:
            _gn = _term_grad_norms(model, out)
            for k, v in _gn.items():
                metric_logger[k].append(v)
            if monitor is not None and _gn:
                monitor.log_iter(_step0 + _it, epoch,
                                 max(g["lr"] for g in optimizer.param_groups),
                                 _gn)

        # Gradient accumulation (--grad_accum): N micro-batches per optimizer
        # step. Mean-of-N-micro-grads == mean-over-N-DDP-ranks, so 1 GPU x
        # bs32 x accum4 reproduces the 4-GPU reference EXACTLY — same per-rank
        # contrast sets in the batch-local InfoNCE (which is built PER BATCH,
        # so a bigger single-GPU batch would change the objective), same
        # global batch, same optimizer-step count and schedule. accum=1 is
        # bit-identical to the pre-accumulation loop (no loss division).
        _boundary = ((_it + 1) % _accum == 0) or (_it + 1 == len(loader))
        if _accum > 1:
            loss = loss / _accum
        if _it % _accum == 0:
            optimizer.zero_grad(set_to_none=True)
        # Skip DDP allreduce on non-boundary micro-batches (no-op off DDP).
        _sync = (model.no_sync() if (not _boundary and hasattr(model, "no_sync"))
                 else contextlib.nullcontext())
        with _sync:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
        if _boundary:
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

        if _audit_left > 0:
            _raw = model.module if hasattr(model, "module") else model
            if _it == 0 and not _region_kwargs(targets, device):
                # Box-mode run: the pooling shape-gate is inactive BY DESIGN
                # (cov is never passed), not dead by bug.
                _audit_exempt.add("spatial_pool.cov_lambda")
            _grad_seen.update(n for n, p in _raw.named_parameters()
                              if p.grad is not None)
            _audit_left -= 1
            if _audit_left == 0:
                _check_dead_params(model, _grad_seen, _audit_exempt,
                                   int(getattr(args, "dead_param_audit", 100)),
                                   _audit_strict)

        for k, v in out.get("loss_dict", {}).items():
            metric_logger[k].append(float(v))
        if scaler is not None:
            metric_logger["amp_scale"].append(scaler.get_scale())
        _n_images += images.shape[0]
        if monitor is not None:
            # max over groups = the head LR (group 0 can be the tiny
            # backbone-LR group)
            monitor.log_iter(
                _step0 + _it, epoch,
                max(g["lr"] for g in optimizer.param_groups),
                {k: float(v) for k, v in out.get("loss_dict", {}).items()})

        # .item() is a host sync on a dispatch-bound model — pay it every 20
        # steps, not every step ([[relsgg-inference-launch-bound]]).
        if _it % 20 == 0:
            pbar.set_postfix(loss=f"{loss.detach().item():.4f}")

    metrics = {k: float(np.mean(v)) for k, v in metric_logger.items()}
    # Utilization visibility (GPU-hour budget is finite — right-size batches):
    metrics["img_per_s"] = _n_images / max(time.time() - _t_start, 1e-6)
    if device.type == "cuda":
        metrics["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
    return metrics


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
    evaluator,
    eval_budget: int = 500,
) -> Dict[str, float]:
    """Evaluate relation prediction with oracle GT boxes.

    Runs model inference with ``targets=None`` (unbiased pair sampling),
    then accumulates metrics into ``evaluator``.

    Args:
        model:       The model (may be DDP-wrapped).
        loader:      Validation DataLoader.
        device:      Target device.
        args:        Namespace with ``amp``.
        evaluator:   A pre-reset evaluator (e.g. ``SGClsEvaluator``)
                     with ``.update(out, targets)`` and ``.compute()``,
                     or a list of them (all fed from the same forward pass;
                     computed metrics are merged).
        eval_budget: Pair budget used during inference.  Temporarily overrides
                     ``model.sampler.final_budget`` for better recall coverage.
    Returns:
        Dict of metric values from ``evaluator.compute()``.
    """
    model.eval()
    evaluators = (evaluator if isinstance(evaluator, (list, tuple))
                  else [evaluator])

    # Temporarily widen the pair budget — handle DDP wrapper
    raw = model.module if hasattr(model, "module") else model
    orig_budget = raw.sampler.final_budget
    raw.sampler.final_budget = min(eval_budget, raw.sampler.geo_budget)

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    try:
        for images, boxes, box_counts, targets in tqdm(
            loader, desc="Evaluating", disable=not is_main
        ):
            images     = images.to(device, non_blocking=True)
            boxes      = boxes.to(device, non_blocking=True)
            box_counts = box_counts.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=args.amp,
                                    dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
                # targets=None keeps pair sampling unbiased, but the region
                # rasters are INPUTS, not labels — they must still be fed or
                # a mask-trained model gets scored in box mode.
                out = model(images, boxes, box_counts, targets=None,
                            **_region_kwargs(targets, device))

            for ev in evaluators:
                ev.update(out, targets)
    finally:
        raw.sampler.final_budget = orig_budget

    metrics: Dict[str, float] = {}
    for ev in evaluators:
        metrics.update(ev.compute())
    return metrics


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
    max_batches: int = 100,
) -> Dict[str, float]:
    """Compute the training loss composite on held-out val data (model.eval(),
    no grad). ``evaluate()`` above deliberately runs with ``targets=None`` (it
    only scores recall), so nothing currently measures whether the loss itself
    generalizes — this fills that gap. Returns ``val_loss_*`` keys, directly
    comparable to ``train_one_epoch``'s ``loss_*`` since it runs the same
    forward/loss computation, just without backprop and on val images.

    Args:
        model:       The model (may be DDP-wrapped; NOT the EMA shadow —
                     compare against the same weights train_metrics came from).
        loader:      Validation DataLoader.
        device:      Target device.
        args:        Namespace with ``amp``.
        max_batches: Cap for wall-clock (val loss is a diagnostic, not a
                     leaderboard metric — a few hundred batches is plenty).
    """
    model.eval()
    metric_logger: Dict[str, list] = defaultdict(list)
    is_main = not dist.is_initialized() or dist.get_rank() == 0
    for step, (images, boxes, box_counts, targets) in enumerate(
        tqdm(loader, desc="Val loss", disable=not is_main, total=min(max_batches, len(loader)))
    ):
        if step >= max_batches:
            break
        images     = images.to(device, non_blocking=True)
        boxes      = boxes.to(device, non_blocking=True)
        box_counts = box_counts.to(device, non_blocking=True)
        for t in targets:
            if "relations" in t:
                t["relations"] = t["relations"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp,
                                dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
            out = model(images, boxes, box_counts, targets,
                        **_region_kwargs(targets, device))

        for k, v in out.get("loss_dict", {}).items():
            metric_logger[f"val_{k}"].append(float(v))

    return {k: float(np.mean(v)) for k, v in metric_logger.items()}


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args,
    analyzer,
    max_batches: int = 200,
) -> None:
    """Fill ``analyzer`` with GT-labelled pair embeddings.

    Runs a separate forward pass *with* targets so that GT predicate labels
    are assigned to sampled pair slots.

    Args:
        model:       The raw (non-DDP) model.
        loader:      DataLoader (val recommended).
        device:      Target device.
        args:        Namespace with ``amp``.
        analyzer:    A pre-reset ``EmbeddingAnalyzer``.
        max_batches: Stop after this many batches to limit runtime.
    """
    model.eval()

    is_main = not dist.is_initialized() or dist.get_rank() == 0
    for step, (images, boxes, box_counts, targets) in enumerate(
        tqdm(loader, desc="Collecting embeddings", disable=not is_main)
    ):
        if step >= max_batches:
            break

        images     = images.to(device, non_blocking=True)
        boxes      = boxes.to(device, non_blocking=True)
        box_counts = box_counts.to(device, non_blocking=True)
        for t in targets:
            if "relations" in t:
                t["relations"] = t["relations"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=args.amp,
                                dtype=getattr(args, "amp_dtype_t", torch.bfloat16)):
            out = model(images, boxes, box_counts, targets=targets,
                        **_region_kwargs(targets, device))

        # Extract per-image entity label tensors for T1-B entity bias metric
        entity_labels_batch = [t.get("entity_labels") for t in targets]
        if any(el is None for el in entity_labels_batch):
            entity_labels_batch = None

        analyzer.update(out, entity_labels=entity_labels_batch)
