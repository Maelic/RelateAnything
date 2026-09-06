"""Load a released or training checkpoint into a ``RelSGG`` model.

A released ``model.pth`` holds the EMA weights, the training ``args`` (from
which the config is rebuilt), the backbone's Hugging Face config (so the
tower is built without any download), and the training predicate names. A
training checkpoint holds the same plus optimiser state and names the
pretrained tower instead of embedding its config.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

import torch

from.config import config_from_args
from.model import RelSGG

#: Local converted-backbone directory names and the hub repositories they
#: were converted from (gated; ``huggingface-cli login`` after accepting the
#: DINOv3 licence).
BACKBONE_HF_IDS = {
    "vits16_lvd1689m": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus_lvd1689m": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16_lvd1689m": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16_lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
}

#: State-dict entries older training checkpoints carry that the model no
#: longer defines. They are dropped on load and by ``release/strip_checkpoint.py``.
OBSOLETE_KEYS = ("vocab_head.gate_u", "vocab_head.gate_b", "vocab_head.beta")

#: Buffers whose size depends on the installed vocabulary; they are resized
#: to the checkpoint's shape before loading and replaced by ``set_vocabulary``.
VOCAB_SIZED = ("vocab_head.W", "vocab_head.alpha", "W_obj")


def hub_id_for_backbone(name: Optional[str]) -> Optional[str]:
    """A hub id or an existing directory is returned unchanged; a converted
    local directory name that does not exist here maps to its hub id."""
    if not name or os.path.isdir(name):
        return name
    return BACKBONE_HF_IDS.get(os.path.basename(str(name).rstrip("/")), name)


def materialize_backbone_config(ckpt: dict, args: dict) -> bool:
    """Write an embedded ``backbone_config`` to a temporary directory and
    point ``args["backbone_model"]`` at it. Returns True when the backbone
    can be built without any download."""
    cfg = ckpt.get("backbone_config")
    if not cfg:
        return False
    d = tempfile.mkdtemp(prefix="ra_backbone_")
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    args["backbone_model"] = d
    return True


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_state(model: RelSGG, sd: dict, strict: bool = False) -> None:
    """Load ``sd`` into ``model``: obsolete keys are dropped, vocabulary-sized
    buffers are resized first. With ``strict`` any other mismatch raises, which
    is what a release must do."""
    sd = {k: v for k, v in sd.items() if k not in OBSOLETE_KEYS}
    for name in VOCAB_SIZED:
        t = sd.get(name)
        if t is None:
            continue
        mod = model
        *path, leaf = name.split(".")
        for p in path:
            mod = getattr(mod, p)
        if getattr(mod, leaf).shape != t.shape:
            setattr(mod, leaf, torch.empty_like(t))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if k != "W_obj"]
    if strict and (missing or unexpected):
        raise RuntimeError(f"strict load failed: missing={missing[:6]} unexpected={unexpected[:6]}")
    if missing or unexpected:
        print(f"[checkpoint] missing={missing[:4]} unexpected={unexpected[:4]}")


def build_model_from_ckpt(ckpt: dict, weights: str = "ema", strict: bool = False) -> RelSGG:
    """A ``RelSGG`` with the checkpoint's config and weights (``weights``:
    ``"ema"`` when the checkpoint has them, else the raw weights)."""
    a = dict(ckpt["args"] if isinstance(ckpt["args"], dict) else vars(ckpt["args"]))
    offline = materialize_backbone_config(ckpt, a)
    if not offline:
        a["backbone_model"] = hub_id_for_backbone(a.get("backbone_model"))
    model = RelSGG(config_from_args(a, backbone_pretrained=not offline))
    sd = ckpt["ema_model"] if (weights == "ema" and ckpt.get("ema_model")) else ckpt["model"]
    load_state(model, sd, strict=strict)
    return model
