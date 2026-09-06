"""Pieces of the training run: model construction, weight initialisation
from another run, optimiser, schedule, EMA and the out-of-domain dev score.
"""
from __future__ import annotations

import copy
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from..config import RelSGGConfig, config_from_args
from..eval.evaluator import SGClsEvaluator
from..model import RelSGG
from..text.student import encode_texts_student
from.engine import evaluate
from.losses import PredicateOntology

#: Keys ``--init_from`` copies from the source run so a fine-tune builds the
#: same network and sees the data the same way.
INIT_INHERIT_KEYS = [f.name for f in RelSGGConfig.__dataclass_fields__.values()
                     if f.name != "backbone_pretrained"] + [
    "text_student", "img_size", "max_objects", "static_shapes", "augment",
    "multi_scale", "multi_scale_n", "tau_eval", "n_neg", "eval_budget"]


def union_spatial_flags(roots: List[str], union_predicates: List[str]) -> np.ndarray:
    """Per-predicate spatial flag: 1 when any source's relations carrying
    that predicate are spatial by majority (relation flag bit 0)."""
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


def spatial_probe_alpha(W: np.ndarray, is_spatial: np.ndarray) -> torch.Tensor:
    """Target routing weights for the gate warm start: a balanced logistic
    probe from text embeddings to the spatial flag (the positive class is a
    few dozen predicates among thousands)."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=5000, class_weight="balanced")
    clf.fit(W, is_spatial)
    u = torch.from_numpy(clf.coef_[0].astype(np.float32))
    b = float(clf.intercept_[0])
    alpha = torch.sigmoid(torch.from_numpy(W.astype(np.float32)) @ u + b)
    spa, sem = alpha[is_spatial == 1], alpha[is_spatial == 0]
    print(f"Spatial probe: {int(is_spatial.sum())} spatial predicates | alpha spatial "
          f"{spa.mean():.3f} (min {spa.min():.3f}) | semantic {sem.mean():.3f} "
          f"(max {sem.max():.3f})")
    return alpha


def build_model(args, pred_names: List[str], obj_names: Optional[List[str]] = None) -> RelSGG:
    """The model of a training run: config from the flags, vocabulary from
    ``--pred_embeds``, gate warm-started on the spatial flags, losses from
    ``--soft_supervision``, negative rates from ``--neg_rate_table``,
    category embeddings for the object alignment term."""
    cfg = config_from_args(vars(args), backbone_pretrained=True)
    model = RelSGG(cfg)

    z = np.load(args.pred_embeds)
    npz_preds = [str(p) for p in z["predicates"]]
    assert npz_preds == list(pred_names), (
        f"{args.pred_embeds} predicate order != dataset vocabulary; rebuild it with "
        "training/build_union_vocab.py on this pack")
    print(f"Installing W from {args.pred_embeds} (templates={list(z['templates'])})")
    model.vocab_head.set_vocabulary_matrix(pred_names, z["embeddings"])

    if getattr(args, "init_from", ""):
        print("[init_from] gate loaded from the source checkpoint; probe fit skipped")
    else:
        is_spatial = union_spatial_flags(args.data_roots, pred_names)
        target = spatial_probe_alpha(model.vocab_head.W.cpu().numpy(), is_spatial)
        mse = model.vocab_head.warm_start_gate(target)
        a = model.vocab_head.alpha
        print(f"gate warm start: mse={mse:.5f}  alpha>0.5: {int((a > 0.5).sum())}/{a.numel()}")

    ontology_meta = args.ontology_meta or os.path.join(args.data_roots[0], "train", "meta.json")
    ontology = PredicateOntology.from_soft_supervision(meta_path=ontology_meta,
                                                       npz_path=args.soft_supervision)
    print(f"Ontology: {ontology.stats()}")
    model.install_losses(ontology, n_neg=args.n_neg)

    if args.neg_rate_table:
        t = np.load(args.neg_rate_table, allow_pickle=False)
        trusted = t["opportunities"] >= int(t["min_support"])
        model.sampler.set_negative_rates(
            torch.from_numpy(t["rate"].astype(np.float16)), torch.from_numpy(trusted),
            int(t["num_cats"]))
        print(f"[sampler] statistical negatives: {int(trusted.sum()):,} trusted category "
              f"pairs, median weight {float(1 - np.median(t['rate'][trusted])):.2f} "
              f"(floor {cfg.rel_neg_weight})")

    if obj_names:
        cache = os.path.join(os.path.dirname(args.soft_supervision), "obj_embeds.npz")
        W_obj = None
        if os.path.isfile(cache):
            zo = np.load(cache)
            if [str(n) for n in zo["names"]] == list(obj_names) and zo["embeddings"].shape[1] == cfg.text_dim:
                W_obj = torch.from_numpy(zo["embeddings"].astype(np.float32))
        if W_obj is None:
            print(f"Encoding {len(obj_names)} object categories with the text student")
            W_obj = encode_texts_student(obj_names, args.text_student,
                                         templates=["{p}", "a photo of a {p}"])
            np.savez_compressed(cache, embeddings=W_obj.numpy().astype(np.float16), names=obj_names)
        model.set_object_vocabulary(obj_names, W_obj)
    return model


def inherit_init_args(args) -> None:
    """``--init_from``: copy the network and recipe keys of the source run."""
    ck = torch.load(args.init_from, map_location="cpu", weights_only=False, mmap=True)
    ca = ck.get("args") or {}
    ca = ca if isinstance(ca, dict) else vars(ca)
    changed = []
    for k in INIT_INHERIT_KEYS:
        if k in ca and getattr(args, k, None) != ca[k]:
            changed.append((k, getattr(args, k, None), ca[k]))
            setattr(args, k, ca[k])
    print(f"[init_from] inherited {len(INIT_INHERIT_KEYS)} keys from {args.init_from}; "
          "overrides: " + (", ".join(f"{k}: {o!r}->{n!r}" for k, o, n in changed) or "none"))


def apply_init_from(model: RelSGG, path: str) -> None:
    """Weights-only initialisation from a finished run (EMA weights when
    present). Vocabulary-sized tensors keep this run's values; every other
    tensor must load, or the run aborts."""
    from..checkpoint import OBSOLETE_KEYS, VOCAB_SIZED
    ck = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    sd = ck["ema_model"] if ck.get("ema_model") else ck["model"]
    sd = {k: v for k, v in sd.items() if k not in OBSOLETE_KEYS}
    own = model.state_dict()
    keep = {k: v for k, v in sd.items()
            if k in own and k not in VOCAB_SIZED and tuple(own[k].shape) == tuple(v.shape)}
    bad = [k for k in sd if k not in keep and k not in VOCAB_SIZED]
    missing = [k for k in own if k not in keep and k not in VOCAB_SIZED]
    if bad or missing:
        raise RuntimeError(f"--init_from architecture mismatch: unmatched={bad[:8]} "
                           f"missing={missing[:8]}")
    model.load_state_dict(keep, strict=False)
    with torch.no_grad():
        model.vocab_head._update_alpha()
    print(f"[init_from] loaded {len(keep)} tensors from {path}")
    del ck


def build_optimizer(model: RelSGG, args) -> torch.optim.Optimizer:
    """AdamW: head at ``--lr``, backbone at ``--backbone_lr``; no weight decay
    on biases, norms and the zero-initialised gates (every tensor with one
    dimension or fewer)."""
    backbone_params, other_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (backbone_params if "backbone" in name else other_params).append(param)
    groups: list = []

    def _emit(params: list, lr: float, wd: float) -> None:
        dec = [p for p in params if p.ndim > 1]
        nod = [p for p in params if p.ndim <= 1]
        if dec:
            groups.append({"params": dec, "lr": lr, "weight_decay": wd})
        if nod:
            groups.append({"params": nod, "lr": lr, "weight_decay": 0.0})

    _emit(other_params, args.lr, args.weight_decay)
    _emit(backbone_params, args.backbone_lr, args.weight_decay)
    return torch.optim.AdamW(groups, weight_decay=args.weight_decay,
                             fused=torch.cuda.is_available())


def build_scheduler(optimizer, args, steps_per_epoch: int):
    """Linear warm-up then cosine decay to ``lr * min_lr_factor``, stepped per
    optimiser step."""
    total_steps = max(args.epochs * steps_per_epoch, 1)
    warmup_steps = min(args.warmup_steps, total_steps // 10 + 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-3, step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(args.min_lr_factor, 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class ModelEMA:
    """Exponential moving average of the weights; evaluated and saved in
    place of the raw weights. The decay ramps up over the first steps."""

    def __init__(self, model: nn.Module, decay: float = 0.9998) -> None:
        self.decay = decay
        self.updates = 0
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        src = model.module if hasattr(model, "module") else model
        for ema_p, src_p in zip(self.ema_model.parameters(), src.parameters()):
            ema_p.mul_(d).add_(src_p.data, alpha=1.0 - d)
        for ema_b, src_b in zip(self.ema_model.buffers(), src.buffers()):
            ema_b.copy_(src_b)

    def state_dict(self) -> dict:
        return self.ema_model.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self.ema_model.load_state_dict(state)


@torch.no_grad()
def zeroshot_dev_metrics(eval_model, dev: dict, args, device) -> Tuple[dict, List[dict]]:
    """Score the current weights on an out-of-domain dev split: swap in the
    dev vocabulary, reparameterize, evaluate graph-constrained, swap back.
    Checkpoint selection reads this rather than the in-domain metric, which
    rewards fitting the training corpus's annotation style."""
    raw = eval_model.module if hasattr(eval_model, "module") else eval_model
    head = raw.vocab_head
    saved = (head.W, head.alpha, head.pred_names, head.is_reparameterized)
    was_training = eval_model.training
    try:
        head.set_vocabulary_matrix(dev["pred_names"], dev["E"])
        raw.reparameterize()
        ev = SGClsEvaluator(topk=[20, 50, 100], num_predicates=len(dev["pred_names"]),
                            score_mode="sigmoid", graph_constraint=True)
        m = evaluate(eval_model, dev["loader"], device, args, ev, eval_budget=dev["budget"])
        dev_per_class = ev.compute_per_class(50, dev["pred_names"])
    finally:
        head.W, head.alpha, head.pred_names, head.is_reparameterized = saved
        eval_model.train(was_training)
    return {f"dev_{k}": float(v) for k, v in m.items()}, dev_per_class


def source_column_mask(args, pred_names: List[str]) -> Optional[torch.Tensor]:
    """``--restrict_neg_sources``: ``[n_sources, V]`` bool, the vocabulary
    columns anchors from each source may be contrasted against. A source that
    annotates one verb per pair never labels spatial relations, so its
    anchors must not push those columns down."""
    if not args.restrict_neg_sources:
        return None
    names = list(args._source_names)
    idx = {pn: i for i, pn in enumerate(pred_names)}
    allow = torch.ones(len(names), len(pred_names), dtype=torch.bool)
    for s in args.restrict_neg_sources:
        assert s in names, f"--restrict_neg_sources {s!r} not in {names}"
        si = names.index(s)
        meta = json.load(open(os.path.join(args.data_roots[si], "train", "meta.json")))
        own = [idx[pn] for pn in meta["predicates"] if pn in idx]
        allow[si] = False
        allow[si, own] = True
        print(f"[negmask] {s}: anchors contrast against its own {len(own)}/{len(pred_names)} predicates")
    return allow
