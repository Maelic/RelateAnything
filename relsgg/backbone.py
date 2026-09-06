"""ViT backbone wrapper with optional LoRA fine-tuning.

Supports multiple backbone families (DINOv2, DINOv3, PE-Core, EUPE).
Exposes dense patch features as a spatial grid [B, H/patch, W/patch, d_model].

Multi-layer extraction: intermediate hidden states are combined with a learned
scalar combiner so the backbone can develop relation-specific representations
across depths, not just at the final layer.

LoRA fine-tuning (via PEFT) is applied to Q/K/V projections of every ViT
block by default (lora_rank=8). Set lora_rank=0 for full fine-tuning, or
freeze_backbone=True to disable all gradient flow.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.v2 import Normalize


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

# Relative layer indices to extract (counting from the end of the ViT stack).
# These are anchored relative to total depth so they work across ViT-B/L/H.
_LAYER_OFFSETS = [-6, -3, -1]

# HuggingFace model identifiers for each backbone type
_DEFAULT_MODELS: dict[str, str] = {
    "dinov3": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov2": "facebook/dinov2-base",
    "pe_core": "facebook/pe-core-l14-224",
    "eupe": "facebook/eupe-vitb16",
    # training/convert_dinov3_convnext.py writes the same layout from Meta's
    # raw torchhub state dicts for machines without hub access.
    "dinov3_convnext": "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
}

# ConvNeXt emits ONE feature map per stage at strides 4/8/16/32 with different
# channel counts, so the ViT path's "weighted sum of same-shape hidden states"
# does not apply. All four stages are fused (offsets index the 4 stage maps;
# hidden_states[0] is the input tensor, hence -4..-1 rather than 0..3).
_CONVNEXT_STAGE_OFFSETS = [-4, -3, -2, -1]

# Target PEFT modules for Q/K/V LoRA injection. HF naming differs per family:
# DINOv2-style ViTs expose `query`/`key`/`value`; DINOv3 exposes
# `q_proj`/`k_proj`/`v_proj`. PEFT matches by module-name suffix and is happy
# as long as at least one listed name exists, so the union covers both.
_LORA_TARGET_MODULES = ["query", "key", "value", "q_proj", "k_proj", "v_proj"]


class RelationInteractionBlock(nn.Module):
    """SGG-aware post-transformer relation refinement.

    Runs directly on top of :class:`RelationTransformer` output in
    ``d_model`` space.  Two explicit stages with a deliberate ordering:

    **Stage 1 — Dependency** (``n_dep`` self-attention layers, pairs → pairs):
        Pairs attend to each other in the already-meaningful ``d_model``
        representation space produced by the RelationTransformer.  This is
        the right point for SGG-specific reasoning:

        - *Co-occurrence*: ``(man, ride, horse)`` reinforces ``(saddle, on, horse)``.
        - *Transitivity*: A above B, B above C → A likely above C.
        - *Disambiguation*: competing predicates for the same pair are resolved
          by seeing what the surrounding pairs settled on.

    **Stage 2 — Grounding** (``n_gnd`` joint self-attention layers, pairs + scene):
        After dependency consensus, the query representations and the scene
        patch tokens are concatenated into a single sequence
        ``[queries ‖ scene]`` and passed through a standard
        :class:`~torch.nn.TransformerEncoderLayer`.  The query positions are
        then extracted back.  This is the *query injection* mechanism from
        SL-HOI: unlike one-directional cross-attention, queries attend to
        every scene patch *and* scene patches update conditioned on the query
        context — improving visual grounding without a frozen dino.txt head.

    Args:
        d_model:   Pair feature dimension (output of RelationTransformer).
        scene_dim: Backbone patch token dimension.  If not equal to
                   ``d_model``, a linear projection is inserted automatically.
        n_dep:     Self-attention layers for inter-pair dependency.  Default 2.
        n_gnd:     Cross-attention layers for image grounding.  Default 1.
        n_heads:   Attention heads (must divide ``d_model``).
        ffn_ratio: FFN hidden-dim multiplier.  Default 2.0 (lighter than ViT).
        dropout:   Attention/FFN dropout for both stages.  Default 0.1, which
                   was previously hardcoded here (and so unsweepable) — the
                   default keeps prior runs bit-identical.
    """

    def __init__(
        self,
        d_model: int,
        scene_dim: Optional[int] = None,
        n_dep: int = 2,
        n_gnd: int = 1,
        n_heads: int = 8,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
        scene_pe: bool = False,
    ) -> None:
        super().__init__()
        ffn_dim = int(d_model * ffn_ratio)

        # Lightweight scene projection when backbone_dim ≠ d_model
        if scene_dim is not None and scene_dim != d_model:
            self.scene_proj: nn.Module = nn.Linear(scene_dim, d_model, bias=False)
            nn.init.xavier_uniform_(self.scene_proj.weight)  # type: ignore[union-attr]
        else:
            self.scene_proj = nn.Identity()

        # Gated absolute PE on the grounding-stage scene tokens (zero-init;
        # see geometry.ScenePosEnc).
        self.scene_pe = None
        if scene_pe:
            from .geometry import ScenePosEnc
            self.scene_pe = ScenePosEnc(d_model)

        # Stage 1: dedicated inter-pair dependency layers
        self.dep_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_dep)
        ])

        # Stage 2: joint query-injection grounding layers.
        # The full sequence [queries ‖ scene] is passed through a self-attention
        # block so queries and scene patches attend to each other bidirectionally.
        self.gnd_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_gnd)
        ])

    def forward(
        self,
        queries: torch.Tensor,
        image_tokens: torch.Tensor,
        query_padding_mask: Optional[torch.Tensor] = None,
        grid_hw: "Optional[tuple]" = None,
    ) -> torch.Tensor:
        """
        Args:
            queries:           [B, K, d_model]  pair representations.
            image_tokens:      [B, N, scene_dim]  backbone patch tokens.
            query_padding_mask:[B, K] bool — True marks padding.
            grid_hw:           (h, w) of the patch grid — required when the
                               block was built with scene_pe=True.
        Returns:
            [B, K, d_model]
        """
        x = queries
        mem = self.scene_proj(image_tokens)  # [B, N, d_model]
        if self.scene_pe is not None:
            h, w = grid_hw
            mem = mem + self.scene_pe(h, w, mem.device, mem.dtype)

        for layer in self.dep_layers:
            x = layer(x, src_key_padding_mask=query_padding_mask)

        # Stage 2: query injection — build [queries ‖ scene], run joint self-attn,
        # then extract only the query positions back.
        n_q = x.shape[1]
        for layer in self.gnd_layers:
            joint = torch.cat([x, mem], dim=1)  # [B, K+N, d_model]
            if query_padding_mask is not None:
                # Scene tokens are never padding; append a False block.
                scene_no_pad = torch.zeros(
                    x.shape[0], mem.shape[1], dtype=torch.bool, device=x.device
                )
                joint_mask = torch.cat([query_padding_mask, scene_no_pad], dim=1)
            else:
                joint_mask = None
            joint = layer(joint, src_key_padding_mask=joint_mask)
            x = joint[:, :n_q]  # re-extract refined query representations

        return x


class RelAnythingBackbone(nn.Module):
    """Multi-layer ViT feature extractor with optional LoRA fine-tuning.

    Args:
        backbone_type: One of ``"dinov3"``, ``"dinov2"``, ``"pe_core"``,
                       ``"eupe"``.
        model_name:    Override the default HuggingFace identifier for the
                       chosen backbone type.
        patch_size:    Patch size of the ViT (default 16).
        lora_rank:     LoRA rank applied to Q/K/V projections.
                       ``0`` = full fine-tuning of the backbone;
                       ``-1`` = backbone frozen (no gradients at all).
        layer_offsets: Which hidden states to fuse, relative to the last layer.
                       Defaults to ``[-6, -3, -1]``.
    """

    def __init__(
        self,
        backbone_type: Literal["dinov3", "dinov2", "pe_core", "eupe",
                               "dinov3_convnext"] = "dinov3",
        model_name: Optional[str] = None,
        patch_size: int = 16,
        lora_rank: int = 8,
        lora_layers: Optional[int] = None,
        layer_offsets: List[int] = _LAYER_OFFSETS,
        pretrained: bool = True,
        drop_path: float = 0.0,
        norm_taps: bool = False,
        stage_s2d: bool = False,
        stage_weight_init: str = "",
    ):
        super().__init__()
        from transformers import AutoModel

        self.patch_size = patch_size
        self.is_convnext = backbone_type == "dinov3_convnext"
        self.layer_offsets = list(_CONVNEXT_STAGE_OFFSETS if self.is_convnext
                                  else layer_offsets)
        self.lora_rank = lora_rank
        # Parameter-free per-tap LayerNorm before the softmax-weighted fusion.
        # ViT activation norms GROW with depth, so without it the learned
        # weights conflate "how important is this layer" with "how large are
        # its activations" (ELMo normalizes layers before mixing for the same
        # reason). False = every prior run bit-identical.
        #
        # IT NOW APPLIES TO THE CONVNEXT PATH TOO. The old comment here claimed
        # the fresh 1x1 convs "absorb per-stage scale on their own". MEASURED,
        # that is false in both directions ([[relsgg-convnext-fusion-flaw]]):
        #   * raw ConvNeXt stage outputs differ 13.2x in RMS (0.158 / 0.156 /
        #     2.086 / 1.248 for strides 4/8/16/32), and the convs learn NEARLY
        #     EQUAL weight norms (16.2/17.3/18.3/22.1), so they do not
        #     compensate — the fine stages end up contributing 4.0% and 4.8%
        #     against stride-16+32's 91%. The hierarchy is discarded.
        #   * because each stage has its OWN conv, `w_i * Conv_i` is the same
        #     function for any w_i, so `layer_weights` is reparameterization-
        #     degenerate and never moved off uniform init in ANY of 8 arms,
        #     while the ViT combiner learned 0.21/0.28/0.51.
        # Normalizing each projected stage fixes both at once: it equalises the
        # summands AND removes the convs' scale freedom, which is what makes
        # `layer_weights` a real, readable level selector on this path.
        self.norm_taps = bool(norm_taps)

        # LEVER 2: how a stage FINER than the target grid is brought down to it.
        # False = adaptive_avg_pool2d (a fixed 4x4 / 2x2 area blur, the original
        # behaviour). True = space-to-depth: pixel_unshuffle folds the r x r
        # neighbourhood into channels and the stage's 1x1 conv learns the
        # reduction, which is exactly a stride-r kernel-r convolution and so is
        # LOSSLESS where the blur is not. Motivation is measured: pooling keeps
        # 87-95% of RMS but only 55% of the stride-4 map's SPATIAL VARIANCE (69%
        # for stride-8), and spatial variance is the whole reason to read a fine
        # stage at all. Costs one 1x1 conv at C*r^2 -> d_model on the 28x28 grid
        # (~0.9 GFLOP at stride 4), so it is affordable to leave on.
        self.stage_s2d = bool(stage_s2d)

        # DIAGNOSTIC ONLY, never set during training: index of one tap to
        # mean-ablate in extract(). -1 = off, and off is bit-identical to the
        # code before this hook existed. Replacing a tap by its per-image mean
        # token removes its SPATIAL information while leaving its magnitude (and
        # so its share of the fused sum) intact, which is what separates "this
        # tap is load-bearing" from "this tap merely carries scale". Zeroing the
        # tap instead would confound the two.
        self.tap_ablate = -1

        if self.is_convnext and lora_rank > 0:
            # PEFT matches Q/K/V projection names, which a pure-conv stack does
            # not have — the assert below would fire anyway, but this says why.
            raise ValueError(
                "dinov3_convnext has no attention projections for LoRA to "
                "target; use --lora_rank 0 (full fine-tune) or -1 (frozen).")

        hf_name = model_name or _DEFAULT_MODELS[backbone_type]
        # Stochastic depth: DINOv3's HF config carries drop_path_rate natively
        # (0.0 default; only active in train mode). Passed as an override so
        # 0.0 stays bit-identical to every prior run. Other backbone families
        # may not define the key — only forward it when non-zero.
        _cfg_over = {"drop_path_rate": drop_path} if drop_path > 0.0 else {}
        if pretrained:
            base_model = AutoModel.from_pretrained(hf_name, **_cfg_over)
        else:
            # Deployment: build the architecture from config only (weights are
            # supplied by the RelSGG checkpoint's own state_dict), so a portable
            # package needs just the tiny config.json — not the 0.34 GB HF
            # weights nor any network access.
            from transformers import AutoConfig
            base_model = AutoModel.from_config(
                AutoConfig.from_pretrained(hf_name, **_cfg_over))

        if lora_rank == -1:
            # Fully frozen — no parameter updates
            base_model.requires_grad_(False)
            self.model = base_model
        elif lora_rank == 0:
            # Full fine-tuning — all parameters trainable
            self.model = base_model
        else:
            # LoRA applied to Q/K/V projections. NO silent fallback: the old
            # code fell back to FULL fine-tuning when the target names didn't
            # match — and DINOv3's names (q_proj/k_proj/v_proj) did NOT match
            # the original list, so `--lora_rank N` would have silently trained
            # all 85.7M backbone params. Fail loudly instead.
            from peft import LoraConfig, get_peft_model

            n_blocks = base_model.config.num_hidden_layers
            extra = {}
            if lora_layers is not None:
                # Restrict LoRA to the LAST `lora_layers` blocks. The feature
                # taps read layers [-6,-3,-1], so adapting only the tail
                # halves the backward cost through the backbone.
                extra = dict(
                    layers_to_transform=list(range(n_blocks - lora_layers,
                                                   n_blocks)),
                    layers_pattern="layer",
                )
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank * 2,
                target_modules=_LORA_TARGET_MODULES,
                lora_dropout=0.0,
                bias="none",
                **extra,
            )
            self.model = get_peft_model(base_model, lora_cfg)
            n_tr = sum(p.numel() for p in self.model.parameters()
                       if p.requires_grad)
            assert n_tr > 0, "LoRA matched no target modules"
            print(f"[backbone] LoRA r={lora_rank} on "
                  f"{'last ' + str(lora_layers) if lora_layers else 'all'} "
                  f"blocks: {n_tr / 1e6:.2f}M trainable")

        # Freeze the params the FEATURE TAPS bypass, so they read as
        # frozen-by-design instead of DEAD in the in-training gradient audit
        # (mirrors the ConvNeXt branch's layer_norm handling below): the taps
        # read hidden_states, which are PRE-norm, so the final norm only feeds
        # last_hidden_state (never read); mask_token is only used when
        # bool_masked_pos is passed (never). Neither ever received a gradient
        # — AdamW skips grad-None params (no update, no decay) — so freezing
        # is bit-identical to every prior run, including full-FT ones.
        if not self.is_convnext:
            for _attr in ("norm", "layernorm"):
                _mod = getattr(base_model, _attr, None)
                if _mod is not None:
                    _mod.requires_grad_(False)
            _emb = getattr(base_model, "embeddings", None)
            _mt = getattr(_emb, "mask_token", None) if _emb is not None else None
            if _mt is not None:
                _mt.requires_grad_(False)

        cfg = base_model.config
        # ConvNeXt configs carry `hidden_sizes` (per stage) and no `hidden_size`.
        # The deepest stage sets d_model so the rest of the model is unchanged
        # (tiny/small 768 = ViT-B's width; base 1024; large 1536).
        self.d_model: int = (cfg.hidden_sizes[-1] if self.is_convnext
                             else cfg.hidden_size)
        self._normalize = Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

        # Learned scalar combiner for multi-layer fusion
        n_layers = len(self.layer_offsets)
        self.layer_weights = nn.Parameter(torch.zeros(n_layers))

        if self.is_convnext:
            # ConvNeXt stage i has stride 4*2^i (4/8/16/32), and the output
            # contract is the patch grid, i.e. stride `patch_size`. So the
            # resample ratio per stage is fixed and resolution-INDEPENDENT:
            #   r_i = patch_size / (4 * 2^i)   ->  4, 2, 1, 0.5 at patch_size 16
            # r > 1 downsamples (pool or space-to-depth), r < 1 upsamples.
            self.stage_ratios = [self.patch_size / (4 * 2 ** i)
                                 for i in range(len(cfg.hidden_sizes))]
            # 1x1 projections bring every stage to d_model so the SAME learned
            # softmax combiner can fuse them. Kept separate from the resampling
            # so cost stays low: resample first (to the patch grid), project
            # second — projecting the stride-4 map at full 112x112 would cost
            # 16x more compute for identical output (both ops are linear, so
            # pool-then-project == project-then-pool).
            #
            # Under space-to-depth the r x r neighbourhood moves INTO channels
            # before the projection, so the conv's fan-in grows by r^2 and the
            # conv itself becomes the learned reduction. Nothing else changes:
            # the output is still [B, d_model, h, w].
            in_ch = [int(c * (r ** 2)) if (self.stage_s2d and r > 1) else c
                     for c, r in zip(cfg.hidden_sizes, self.stage_ratios)]
            self.stage_proj = nn.ModuleList(
                [nn.Conv2d(c, self.d_model, kernel_size=1) for c in in_ch])
            # Post-projection per-stage LayerNorm (lever 1). Built only when
            # requested so an un-normalized arm has no extra parameters and
            # loads old checkpoints unchanged. Affine, because after the
            # normalization the ONLY way a stage can express "I matter more"
            # is `layer_weights` — an affine scale gives the fusion back a
            # per-channel degree of freedom without restoring the whole-stage
            # scale degeneracy that `layer_weights` needs broken.
            self.stage_norm = (
                nn.ModuleList([nn.LayerNorm(self.d_model)
                               for _ in cfg.hidden_sizes])
                if self.norm_taps else None)
            # Optional non-uniform init for the stage combiner. "measured" sets
            # softmax(layer_weights) to the contributions the ORIGINAL design
            # actually produced (4.0/4.8/34.5/56.7%), so turning lever 1 on
            # starts as a near-no-op and can only add — the same convention as
            # the zero-init gates elsewhere in this stack. "" keeps the uniform
            # zero init, which is the honest default when we want to SEE where
            # the level axis lands.
            if stage_weight_init == "measured":
                shares = torch.tensor([0.040, 0.048, 0.345, 0.567][
                    :len(cfg.hidden_sizes)], dtype=torch.float32)
                with torch.no_grad():
                    self.layer_weights.copy_(shares.log())
            elif stage_weight_init:
                raise ValueError(
                    f"stage_weight_init must be '' or 'measured', "
                    f"got {stage_weight_init!r}")
            # We read the per-stage maps, never `last_hidden_state`, so the
            # model's final norm is off our graph and would be reported DEAD by
            # audit_param_health.py — the exact false positive that audit is
            # meant to make impossible to ignore. Freeze it so it reads as
            # FROZEN-by-design instead, which is what it actually is.
            if getattr(base_model, "layer_norm", None) is not None:
                base_model.layer_norm.requires_grad_(False)

    def _resolve_layer_indices(self, total_layers: int) -> List[int]:
        return [total_layers + off if off < 0 else off for off in self.layer_offsets]

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize float [0, 1] images to ImageNet stats.

        Args:
            images: [B, 3, H, W] float32 in [0, 1].
        Returns:
            [B, 3, H, W] normalized.
        """
        return self._normalize(images)

    def extract(self, images: torch.Tensor, return_taps: bool = False):
        """Extract dense patch features from normalized images.

        Runs the backbone with ``output_hidden_states=True``, extracts the
        selected intermediate layers, and fuses them with a softmax-normalized
        learned scalar weighting.

        Args:
            images: [B, 3, H, W] float32, ImageNet-normalized.
                H and W must be divisible by patch_size.
            return_taps: also return the individual (pre-fusion) layer taps, so
                a consumer can treat depth as its own axis instead of accepting
                the fused sum. RAW taps are returned — NOT norm_taps-normalized
                — because the multi-level scene builder applies its own
                per-level LayerNorm before its per-level projection, and
                double-normalizing would hide the depth scale entirely.
                MEASURED reason this option exists: the fused sum is only
                nominally multi-layer. The learned combiner is near-uniform
                (.358/.330/.312) but mean patch norms at the taps are
                52.9/192.4/668.0, so the effective magnitude share is
                6.5%/21.8%/71.7% — the "3-layer fusion" is ~72% last layer.
                Exposing the taps lets a reader choose a depth per sample
                rather than inherit that imbalance.
        Returns:
            F: [B, H//patch_size, W//patch_size, d_model] fused patch features,
            or ``(F, taps)`` with taps a list of [B, h, w, d_model] (one per
            entry of ``layer_offsets``, shallowest first) when return_taps.
        """
        B, _, H, W = images.shape
        h = H // self.patch_size
        w = W // self.patch_size

        if self.is_convnext:
            if return_taps:
                raise NotImplementedError(
                    "return_taps is ViT-path only: the ConvNeXt branch already "
                    "re-projects each stage through its own 1x1 conv (which "
                    "absorbs per-stage scale), so its stages are available "
                    "without this hook.")
            return self._extract_convnext(images, h, w)

        outputs = self.model(pixel_values=images, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple of [B, 1+h*w, d]

        total = len(hidden_states)
        indices = self._resolve_layer_indices(total)

        weights = F.softmax(self.layer_weights, dim=0)  # [n_layers]

        # Weighted sum over selected layers.
        # Strip CLS and any register tokens by taking only the last h*w tokens —
        # patch tokens are always last in DINO-style models (CLS + registers first).
        n_patch = h * w
        raw_taps = [hidden_states[idx][:, -n_patch:, :] for idx in indices]
        taps = raw_taps
        if self.norm_taps:
            # normalized_shape must be a compile-time constant: under ONNX
            # tracing t.shape[-1:] is a symbolic value, which the layer_norm
            # symbolic rejects. d_model is the same number as a plain int.
            taps = [F.layer_norm(t, (self.d_model,)) for t in taps]
        if self.tap_ablate >= 0:
            # Diagnostic: spatially flatten one tap (see __init__). Per image,
            # so batch composition cannot leak between samples.
            ai = self.tap_ablate
            assert ai < len(taps), f"tap_ablate={ai} but only {len(taps)} taps"
            taps = list(taps)
            taps[ai] = taps[ai].mean(dim=1, keepdim=True).expand_as(taps[ai])
        fused = sum(weights[i] * t for i, t in enumerate(taps))  # [B, h*w, d]

        fused = fused.reshape(B, h, w, self.d_model)
        if return_taps:
            return fused, [t.reshape(B, h, w, self.d_model) for t in raw_taps]
        return fused

    def _extract_convnext(self, images: torch.Tensor,
                          h: int, w: int) -> torch.Tensor:
        """Fuse ConvNeXt's four stage maps onto the stride-16 grid.

        Returns the same contract as the ViT path — ``[B, h, w, d_model]`` —
        so nothing downstream needs to know which backbone family produced it.

        Resampling is direction-dependent on purpose: ``adaptive_avg_pool2d``
        for the fine stages (stride 4/8), which is a proper area average rather
        than the point-sampling that bilinear downsampling degenerates into,
        and bilinear for the coarse stage (stride 32) being upsampled.
        """
        outputs = self.model(pixel_values=images, output_hidden_states=True)
        # hidden_states = (pixel_values, stage0, stage1, stage2, stage3); the
        # input tensor is element 0, so negative offsets index the stage maps.
        hidden_states = outputs.hidden_states
        indices = self._resolve_layer_indices(len(hidden_states))
        weights = F.softmax(self.layer_weights, dim=0)

        fused = None
        for i, idx in enumerate(indices):
            f = hidden_states[idx]                       # [B, C_i, h_i, w_i]
            if f.shape[-2:] != (h, w):
                if f.shape[-1] > w:
                    # DOWN to the patch grid. Space-to-depth keeps every value
                    # and lets stage_proj learn the reduction; area-average is
                    # a fixed blur that discards 45% of the stride-4 map's
                    # spatial variance. pixel_unshuffle needs an exact integer
                    # ratio, so fall back to pooling if the grid is not a clean
                    # multiple (odd input sizes under multi-scale).
                    r = f.shape[-1] // w
                    if (self.stage_s2d and r > 1
                            and f.shape[-1] == w * r and f.shape[-2] == h * r):
                        f = F.pixel_unshuffle(f, r)      # [B, C*r^2, h, w]
                    else:
                        f = F.adaptive_avg_pool2d(f, (h, w))
                else:
                    f = F.interpolate(f, size=(h, w), mode="bilinear",
                                      align_corners=False)
            f = self.stage_proj[i](f)                    # [B, d_model, h, w]
            if self.stage_norm is not None:
                # LayerNorm over CHANNELS at each position (channels-last is the
                # ConvNeXt convention), so the normalization is spatial-structure
                # preserving — it equalises stage SCALE without flattening the
                # spatial variance we went to space-to-depth to keep.
                f = self.stage_norm[i](f.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            fused = f * weights[i] if fused is None else fused + f * weights[i]

        return fused.permute(0, 2, 3, 1).contiguous()


# Backward-compatible alias so existing checkpoints and imports still work
class DINOv3Backbone(RelAnythingBackbone):
    DEFAULT_MODEL = _DEFAULT_MODELS["dinov3"]

    def __init__(self, model_name: str = DEFAULT_MODEL, patch_size: int = 16):
        super().__init__(
            backbone_type="dinov3",
            model_name=model_name,
            patch_size=patch_size,
            lora_rank=8,
        )
