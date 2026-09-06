"""Rebuild a ``RelSGG`` model from a training or released checkpoint.

``build_model_from_ckpt`` is the loader every evaluation, export and probe
script shares. It lived in the zero-shot evaluator for most of the project's
history; it is here so that nothing outside ``relsgg`` has to be imported to
load weights.

Backbone resolution, in order:

1. ``ckpt["backbone_config"]`` (present in every released checkpoint, written
   by ``release/strip_checkpoint.py``): the tower is built from that config and
   its weights come from the checkpoint. No download, no Hugging Face login.
2. ``args["backbone_model"]`` names a directory that exists: used as is.
3. ``args["backbone_model"]`` names a local converted directory that does not
   exist here (``checkpoints/hf/vits16plus_lvd1689m`` on the training
   machine): mapped to the matching ``facebook/dinov3-*`` hub id, which is
   gated and needs ``huggingface-cli login``.
"""
from __future__ import annotations

import dataclasses
import json
import os
import tempfile

import torch

from relsgg.model import RelSGG, RelSGGConfig

#: Local converted-backbone directory names (``checkpoints/hf/<name>``) and the
#: hub repositories they were converted from.
BACKBONE_HF_IDS = {
    "vits16_lvd1689m": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vits16plus_lvd1689m": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "vitb16_lvd1689m": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "vitl16_lvd1689m": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "convnext_tiny_lvd1689m": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "convnext_small_lvd1689m": "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "convnext_base_lvd1689m": "facebook/dinov3-convnext-base-pretrain-lvd1689m",
    "convnext_large_lvd1689m": "facebook/dinov3-convnext-large-pretrain-lvd1689m",
}


def hub_id_for_backbone(name):
    """Return ``name`` unchanged when it is a hub id or an existing directory;
    otherwise map a local converted-directory name to its hub id."""
    if not name or os.path.isdir(name):
        return name
    base = os.path.basename(str(name).rstrip("/"))
    return BACKBONE_HF_IDS.get(base, name)


def materialize_backbone_config(ckpt: dict, args: dict) -> bool:
    """If ``ckpt`` embeds ``backbone_config``, write it to a temporary
    directory and point ``args["backbone_model"]`` at it. Returns True when the
    backbone can be built without any download."""
    cfg = ckpt.get("backbone_config")
    if not cfg:
        return False
    d = tempfile.mkdtemp(prefix="ra_backbone_")
    with open(os.path.join(d, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    args["backbone_model"] = d
    return True


def pad_geo_checkpoint(ckpt: dict, target_dim: int) -> None:
    """Zero-pad the geometry-encoder input columns of a pre-2026-08 checkpoint
    (15 features) to the current width in place. ``build_model_from_ckpt``
    applies the same shim, so calling this first is harmless."""
    for sdk in ("model", "ema_model"):
        sd = ckpt.get(sdk)
        if not sd:
            continue
        for wk in ("geo_encoder.mlp.0.weight", "sampler.geo_scorer.0.weight"):
            w = sd.get(wk)
            if w is not None and w.shape[1] < target_dim:
                pad = torch.zeros(w.shape[0], target_dim - w.shape[1], dtype=w.dtype)
                sd[wk] = torch.cat([w, pad], dim=1)


def load_checkpoint(path: str) -> dict:
    """``torch.load`` with the settings every loader in the repo uses."""
    return torch.load(path, map_location="cpu", weights_only=False)


def build_model_from_ckpt(ckpt: dict, weights: str) -> RelSGG:
    a = ckpt["args"]
    if not isinstance(a, dict):
        a = vars(a)
    a = dict(a)
    # Released checkpoints embed the backbone's HF config, so the tower is
    # built from config and every weight comes from the checkpoint itself:
    # no hub download, no gated-repo login. Training-time checkpoints that
    # name a local converted directory fall back to the matching hub id.
    offline = materialize_backbone_config(ckpt, a)
    if not offline:
        a["backbone_model"] = hub_id_for_backbone(a.get("backbone_model"))
    # Geometry-width migration shim (15 -> 19 features, 2026-08-01): zero-pad
    # the new region-feature columns of pre-migration checkpoints. Bit-exact
    # by construction — relsgg/geometry.py zero-inits exactly these columns so
    # a padded old checkpoint reproduces the box-only model (see its
    # docstring); without this every eval of an old checkpoint crashes on a
    # geo_encoder/geo_scorer size mismatch.
    from relsgg.geometry import RelGeomEncoder as _RGE
    for _sdk in ("model", "ema_model"):
        _sd = ckpt.get(_sdk)
        if not _sd:
            continue
        for _wk in ("geo_encoder.mlp.0.weight", "sampler.geo_scorer.0.weight"):
            _w = _sd.get(_wk)
            if _w is not None and _w.shape[1] < _RGE.NUM_GEO:
                _pad = torch.zeros(_w.shape[0], _RGE.NUM_GEO - _w.shape[1],
                                   dtype=_w.dtype)
                _sd[_wk] = torch.cat([_w, _pad], dim=1)
    cfg = RelSGGConfig(
        backbone_type=a["backbone_type"],
        **({"backbone_model": a["backbone_model"]} if a.get("backbone_model") else {}),
        backbone_pretrained=not offline,
        lora_rank=a["lora_rank"],
        lora_layers=a.get("lora_layers"),
        d_model=a["d_model"],
        text_dim=a["text_dim"],
        geo_budget=a["geo_budget"],
        final_budget=a["final_budget"],
        n_self_layers=a["n_self_layers"],
        n_cross_layers=a["n_cross_layers"],
        logit_scale_init=a["logit_scale_init"],
        logit_bias_init=a.get("logit_bias_init") or 0.0,
        sampler_type=a.get("sampler_type", "cascade"),
        rel_neg_weight=a.get("rel_neg_weight", 0.3),
        swap_include=a.get("swap_include", True),
        proj_layers=a.get("proj_layers", 1),
        compose_query=(a.get("loss_type") == "batch_infonce"),
        dual_spatial_head=a.get("dual_spatial_head", False),
        beta_relatedness=a.get("beta_relatedness", False),
        fast_bilinear_head=a.get("fast_bilinear", False),
        lambda_fast=a.get("lambda_fast", 0.5),
        use_rel_interaction=a.get("use_rel_interaction", True),
        n_dep_layers=a.get("n_dep_layers", 2),
        n_gnd_layers=a.get("n_gnd_layers", 1),
        # Without this a --deformable_points checkpoint would load with its
        # trained read SILENTLY DROPPED (strict=False) — the beta_mlp bug
        # class. Old checkpoints lack the key -> 0 -> module absent, as before.
        deformable_points=a.get("deformable_points", 0),
        deformable_heads=a.get("deformable_heads", 1),
        deformable_nulls=a.get("deformable_nulls", 0),
        deformable_clamp=a.get("deformable_clamp", False),
        # Multi-level read: without these the builder makes a 1-level module and
        # the state_dict load fails on offset_mlp/weight_mlp shapes (the same
        # class of breakage scene_pe/pe_num_freqs caused before they were
        # threaded here). This builder is shared by the WHOLE eval suite.
        ms_depth_levels=a.get("ms_depth_levels", 0),
        ms_pool_level=a.get("ms_pool_level", False),
        ms_deconv_level=a.get("ms_deconv_level", False),
        deformable_v2=a.get("deformable_v2", False),
        deformable_ring=a.get("deformable_ring", None),
        deformable_gain=a.get("deformable_gain", None),
        deformable_border=a.get("deformable_border", None),
        # 2026-08-04 fix-stack flags: the first four change module shapes or
        # the forward pass, so dropping them here either crashes the load
        # (scene_pe/pe_num_freqs) or silently evaluates a different network
        # (norm_taps/geo_squash). Old checkpoints lack the keys -> defaults.
        scene_pe=a.get("scene_pe", False),
        pool_role_queries=a.get("pool_role_queries", False),
        pe_num_freqs=a.get("pe_num_freqs", 64),
        pe_max_octave=a.get("pe_max_octave", None),
        norm_taps=a.get("norm_taps", False),
        geo_squash=a.get("geo_squash", False),
        mode_gated=a.get("mode_gated", False),
        region_adjacency=a.get("region_adjacency", False),
        contact_field=a.get("contact_field", False),
        mask_adapter_dim=a.get("mask_adapter_dim", 0),
    )
    # DRIFT GUARD — same one as relsgg/api.py:_cfg_from_args, and added for the
    # same reason: this block is a SECOND hand-enumeration of the config that
    # train.py's build_model() also enumerates, so every new training flag has
    # to be added twice and a miss here rebuilds a differently-shaped model.
    # It has already cost two failures — `pe_num_freqs`/`deformable_nulls` in
    # api.py ([[relsgg-api-config-drift]]) and `stage_s2d` here, which crashed
    # the whole OVS eval of a finished 2.4-hour arm on
    # `stage_proj.0.weight: [768, 1536] vs [768, 96]`.
    # So: copy through ANY remaining RelSGGConfig field the checkpoint's args
    # carry under the SAME NAME, leaving the explicit block above as the
    # authority wherever it set something away from the dataclass default.
    for f in dataclasses.fields(RelSGGConfig):
        if f.name in ("backbone_pretrained", "backbone_model", "compose_query"):
            continue
        if f.name not in a or a[f.name] is None:
            continue
        if f.default is dataclasses.MISSING:
            continue
        cur = getattr(cfg, f.name)
        if cur != f.default or cur == a[f.name]:
            continue
        setattr(cfg, f.name, a[f.name])
    model = RelSGG(cfg)

    sd = ckpt["ema_model"] if (weights == "ema" and "ema_model" in ckpt) \
        else ckpt["model"]
    if "vocab_head.W_param" in sd:
        # --learn_W fine-tunes store the trainable matrix; the eval model keeps
        # the buffer form, so hand it the normalised rows under the buffer name.
        sd = dict(sd)
        sd["vocab_head.W"] = torch.nn.functional.normalize(
            sd.pop("vocab_head.W_param").float(), dim=-1)
    # v34+: the trainable spatialness gate is created on demand — build it
    # before loading when the checkpoint carries its weights, so routing at
    # vocab-swap time uses the trained MLP, not the probe.
    if any(k.startswith("vocab_head.gate_mlp.") for k in sd):
        model.vocab_head.build_gate_mlp()
    # Same on-demand pattern for the per-predicate relatedness weight: without
    # this, a --beta_relatedness checkpoint loads with its trained beta_mlp
    # silently DROPPED (strict=False) and decode falls back to the uniform
    # additive pair term — the arm evaluates as if beta had never trained.
    if any(k.startswith("vocab_head.beta_mlp.") for k in sd):
        model.vocab_head.build_beta_mlp()
    # Freshly-built buffers (W, alpha, W_obj, ...) are empty until a
    # vocabulary is installed; resize them so load_state_dict accepts the
    # checkpoint shapes. They are replaced again by encode_vocabulary below.
    for name, t in sd.items():
        mod = model
        *path, leaf = name.split(".")
        for p in path:
            mod = getattr(mod, p, None)
            if mod is None:
                break
        if mod is None:
            continue
        cur = getattr(mod, leaf, None)
        if (leaf in dict(mod.named_buffers(recurse=False))
                and isinstance(cur, torch.Tensor) and cur.shape != t.shape):
            setattr(mod, leaf, torch.empty_like(t))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    unexpected = [k for k in unexpected if "ontology" not in k]
    print(f"loaded {weights} weights  (missing={missing}, unexpected={unexpected})")
    return model
