"""RelAnything — label-free re-parametrizable relation head.

Top-level model that composes all modules:

    DINOv3Backbone  →  dense patch features F
    masked_avg_pool →  per-object features v_i
    GeoEncoder      →  geometry features g_ij
    pair_proj       →  fused pair representation
    RelationTransformer  →  context-aware pair repr r
    VocabHead       →  cosine scores against predicate vocabulary

Inputs:  image tensor + bounding boxes (NO object class labels).
Outputs: scored (subject, object, predicate) triplets.

Public API
----------
    model = RelSGG()
    model.encode_vocabulary(["above", "behind", "next to"])
    model.reparameterize()          # fuse text embeddings → static matrix

    # Training (DDP-compatible)
    outputs = model(images, boxes, box_counts, targets)
    outputs["loss"].backward()

    # Inference
    triplets = model.predict(images, boxes)
    # → [[{"subject": 0, "object": 2, "predicate": "above", "score": 0.87}, ...], ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import RelAnythingBackbone, RelationInteractionBlock
from .geometry import BoxPromptEncoder, RelGeomEncoder
from .loss import RelSGGLoss, RelSpatialGroundingLoss
from .loss_synonym import (BatchLocalInfoNCE, MultiPositiveInfoNCE,
                           PredicateOntology, SynonymAwareRelLoss,
                           build_slot_targets, swap_direction_hinge)
from .roi import SoftSpatialPool, box_coverage_weights, masked_avg_pool, union_box
from .sampler import CascadePairSampler, RelatednessPairSampler
from .deformable import DeformableRelRead, MultiLevelScene
from .transformer import RelationTransformer, depth_scaled_residual_init_
from .vocab import VocabHead


@dataclass
class RelSGGConfig:
    # Backbone
    backbone_type: str = "dinov3"
    backbone_model: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    patch_size: int = 16
    lora_rank: int = 8  # 0 = full fine-tune, -1 = frozen
    lora_layers: Optional[int] = None  # LoRA only on the LAST N blocks
                                       # (None = all); taps read [-6,-3,-1]
    backbone_pretrained: bool = True  # False = build arch from config only
                                      # (deploy: weights come from checkpoint)

    # Relation Interaction Block (SGG-aware post-transformer refinement)
    use_rel_interaction: bool = True
    n_dep_layers: int = 2   # self-attn layers for inter-pair dependency
    n_gnd_layers: int = 1   # cross-attn layers for image grounding

    # Model dimensions
    d_model: int = 512

    # Pair sampler
    sampler_type: str = "cascade"   # "cascade" (P1) | "relatedness" (P2, D4)
    geo_budget: int = 400
    beta_relatedness: bool = False  # per-predicate relatedness weight
    lambda_sigmoid: float = 0.0     # per-cell sigmoid (SPML) auxiliary
    final_budget: int = 128
    rel_neg_weight: float = 0.3     # relatedness-BCE negative weight (PU-aware) — now
                                    # a FLOOR when neg_rate_table is supplied
    neg_rate_table: str = ""        # pair_opportunity.npz: per-category-pair interaction
                                    # rate, used to price the PU hedge per pair
    swap_include: bool = True       # force-include swapped GT pairs (ARO supervision)

    # Relation transformer
    n_self_layers: int = 2
    n_cross_layers: int = 2
    n_heads: int = 8
    ffn_ratio: float = 2.0
    dropout: float = 0.1            # attention/FFN dropout in BOTH the relation
                                    # transformer and the interaction block. Was
                                    # previously hardcoded 0.1 in each (and so
                                    # unsweepable); 0.1 keeps every prior run
                                    # bit-identical. Raise to regularize — v34
                                    # overfits the 472K pack from ~epoch 8.
    deformable_points: int = 0      # >0 enables the box-anchored deformable
                                    # scene read after the rel_transformer:
                                    # this many sampled points per anchor
                                    # (4 anchors: sub/obj/union/contact).
                                    # Additive behind a zero-init gate, so 0
                                    # vs untrained->0 gate is bit-identical.
    deformable_heads: int = 1       # sampling heads; each reads d_model/heads
                                    # channels at its OWN points, so H heads =
                                    # H x the locations at identical memory
                                    # traffic. 1 = v1 arm.
    deformable_nulls: int = 0       # V3: learnable non-image slots per
                                    # (head, anchor) competing in the sampling
                                    # softmax. Zero-init => exact zero reads =
                                    # clean per-pair attenuation, which v1 had
                                    # to fake by flinging points off-frame.
    deformable_clamp: bool = False  # V3: clamp sampled positions to the image.
                                    # Safe once nulls exist; reading outside
                                    # the BOX stays unrestricted.
    deformable_v2: bool = False     # compound shorthand for the v2 arm: ring
                                    # init + per-pair gain gate + border
                                    # padding, all three at once. Kept so v2
                                    # checkpoints rebuild exactly; prefer the
                                    # three explicit flags below for new arms,
                                    # since the v2 bundle MEASURED -6.8% A6 and
                                    # the blame is not separable inside it.
    deformable_ring: Optional[bool] = None    # distinct (angle, radius) per
                                    # (head, point) instead of all points on
                                    # the anchor center. Without it identical
                                    # points get identical gradients and can
                                    # only split through numerical noise, so
                                    # this is REQUIRED for heads to specialize.
    deformable_gain: Optional[bool] = None    # per-pair sigmoid scalar on the
                                    # read. Redundant once null slots exist
                                    # (they attenuate per head AND anchor);
                                    # keep off unless ablating.
    deformable_border: Optional[bool] = None  # grid_sample padding_mode
                                    # "border". A no-op under deformable_clamp
                                    # (no sample leaves the frame) and the
                                    # prime suspect for v2's A6 loss: it
                                    # replaced a clean null read with edge
                                    # pixels, i.e. with noise.
                                    # None on all three = follow deformable_v2.
    ms_depth_levels: int = 0        # MULTI-LEVEL deformable read. >0 exposes
                                    # this many backbone taps to the deformable
                                    # read as SEPARATE feature levels (shallow
                                    # first), each LayerNorm'd + given its own
                                    # 1x1 projection, instead of the single
                                    # fused map. Rationale is MEASURED: the
                                    # fused map's combiner looks uniform
                                    # (.358/.330/.312) but tap norms are
                                    # 52.9/192.4/668.0, so it is ~72% last
                                    # layer — the depth axis exists but is
                                    # being averaged away. 0 = single fused
                                    # level = bit-identical to every prior run.
    ms_pool_level: bool = False     # add a stride-32 average-pooled level.
                                    # Zero parameters (pooled AFTER projection)
                                    # and the only level that changes the
                                    # spatial support rather than the depth.
    ms_deconv_level: bool = False   # add a ViTDet-style stride-8 level from a
                                    # 2x2 transposed conv. HONEST LABEL: a
                                    # plain ViT/16 has no sub-16 representation,
                                    # so this is learned sharpening, not
                                    # recovered detail (a bilinear version
                                    # would be exactly redundant with
                                    # grid_sample's own interpolation). 1.05M
                                    # params. Off by default; ablatable.
    depth_scaled_init: bool = False # GPT-2-style 1/sqrt(N) shrink of every
                                    # residual out-projection in the from-scratch
                                    # rel_transformer + rel_interaction stacks
                                    # (N = residual additions on the query path).
                                    # Init-only: checkpoints override it, so eval
                                    # reconstruction never needs the flag.
                                    # False = every prior run bit-identical.
    drop_path: float = 0.0          # stochastic depth rate handed to the HF
                                    # backbone config (DINOv3 supports
                                    # drop_path_rate natively; inactive in eval
                                    # mode). 0.0 = prior runs bit-identical.
    norm_taps: bool = False         # LayerNorm on each backbone tap before the
                                    # softmax-weighted fusion — decouples layer
                                    # importance from activation scale. False =
                                    # prior runs bit-identical. Parameter-free
                                    # on the ViT path; on the CONVNEXT path it
                                    # is affine and post-projection, and it is
                                    # also what makes layer_weights non-
                                    # degenerate there ([[relsgg-convnext-fusion-flaw]]).
    stage_s2d: bool = False         # CONVNEXT ONLY. Bring stages finer than the
                                    # patch grid down by space-to-depth
                                    # (pixel_unshuffle + the stage's own 1x1
                                    # conv = a learned stride-r conv, lossless)
                                    # instead of adaptive_avg_pool2d, a fixed
                                    # blur that drops 45% of the stride-4 map's
                                    # spatial variance. False = prior runs
                                    # bit-identical. No-op on the ViT path.
    stage_weight_init: str = ""     # CONVNEXT ONLY. "" = uniform (zero) init.
                                    # "measured" = init softmax(layer_weights)
                                    # to the shares the un-normalized design
                                    # actually produced (.040/.048/.345/.567),
                                    # so enabling norm_taps starts as a
                                    # near-no-op instead of a step change.
    pe_num_freqs: int = 64          # Fourier bands per coordinate in every box
                                    # positional encoder (corner tokens + pool
                                    # queries). Legacy 64 with the doubling
                                    # ladder puts bands 20-63 past float32's
                                    # precision cliff — pure noise. Recommended
                                    # new-run setting: 16 with pe_max_octave 7.
                                    # NOTE: changes proj input width, so old
                                    # checkpoints need the legacy value.
    pe_max_octave: Optional[float] = None  # top frequency = 2**this, geometric
                                    # ladder. None = legacy 2**arange(n) —
                                    # prior runs bit-identical.
    geo_squash: bool = False        # 10*tanh(x/10) instead of clamp(-10,10) on
                                    # the 19 geometry features (encoder AND
                                    # samplers): the clamp is gradient-dead at
                                    # the rails, which small/distant boxes hit
                                    # routinely. False = prior runs
                                    # bit-identical.
    geo_pu: bool = False            # PU-price the geometry pre-scorer's BCE
                                    # negatives with the opportunity table
                                    # (same hedge the relatedness head gets).
                                    # False = prior runs bit-identical.
    scene_pe: bool = False          # gated absolute Fourier PE on the scene
                                    # keys at ALL THREE cross-attn sites
                                    # (rel_transformer, interaction grounding,
                                    # SoftSpatialPool). Zero-init gates: False
                                    # AND fresh-True both start bit-identical
                                    # to prior runs. See geometry.ScenePosEnc.
    pool_role_queries: bool = False # per-role query biases in SoftSpatialPool
                                    # (object/union/contact), zero-init.
    mode_gated: bool = False        # gate every mask-sensitive parameter on a
                                    # per-image mode bit (1 mask / 0 box) so a
                                    # box image runs the box-only network
                                    # EXACTLY; masks get a zero-init adapter.
                                    # See roi.SoftSpatialPool / geometry.
    region_adjacency: bool = False  # 5th mode-gated adapter column: region
                                    # BOUNDARY ADJACENCY. r_contact is an
                                    # INTERSECTION, which is identically zero
                                    # for masks that partition the image
                                    # (panoptic/PSG: 0% of test pairs overlap
                                    # at all, vs 19% for SAM) -- degenerate
                                    # exactly where contact predicates live.
                                    # This is overlap after a one-cell
                                    # dilation. Adapter-only (region_delta
                                    # gains a 5th input), so NUM_GEO stays 19
                                    # and every checkpoint loads unchanged.
    contact_field: bool = False     # mask-derived coverage bias for the CONTACT
                                    # half of the merged pool. Today that half
                                    # gets a FLAT raster (1-1e-4 => zero bias),
                                    # so it is the one place the network is
                                    # identical for boxes and masks BY
                                    # CONSTRUCTION. This replaces it with the
                                    # proximity field min(dil(cov_s),dil(cov_o)):
                                    # high on the interface and in a narrow gap,
                                    # ~0 for far-apart pairs -- and an all-zero
                                    # field is a CONSTANT bias, which softmax
                                    # normalises away, so those pairs fall back
                                    # to exactly today's behaviour. Mode-gated
                                    # in SoftSpatialPool, so a box image is
                                    # still bit-identical.
    mask_adapter_dim: int = 0       # capacity knob for the mode-gated adapter:
                                    # a zero-init residual MLP on the fused pair
                                    # input, added as m * MLP(pair_input). The
                                    # frozen-trunk retrofit otherwise has only
                                    # 1,416 trainable parameters (8 cov_lambda +
                                    # 384 mode_embed + 1024 region_delta), which
                                    # is the ceiling the adjacency experiment ran
                                    # into. 64 gives ~1.5e5. Needs mode_gated.
    box_token_dropout: float = 0.0  # train-time probability (per step) of
                                    # dropping the box-corner tokens from the
                                    # rel_transformer cross-attn memory, forcing
                                    # gradient through the scene pathway. The
                                    # box tokens are a geometry shortcut the
                                    # cross-attn can collapse onto — measured at
                                    # scene mass EXACTLY 0.000 under LoRA
                                    # (runs/analysis/relation_attn_stats.json);
                                    # standard modality-dropout remedy, sweet
                                    # spot 0.3-0.5 in the multimodal literature.
                                    # 0.0 = every prior run bit-identical.

    # Vocabulary / text encoder
    text_model: str = "openai/clip-vit-base-patch32"
    text_dim: Optional[int] = None  # None = same as d_model (CLIP @ 512); 2048 for dino.txt
    logit_scale_init: float = 1.0 / 0.07
    logit_bias_init: float = 0.0    # SigLIP-style; -10 for sigmoid multi-label training
    infonce_temp: float = 0.07
    proj_layers: int = 1            # visual→text projection depth (>=2 = MLP)
    compose_query: bool = False     # q = proj(r) + P_s(v_sub) + P_o(v_obj) (RLIP-flavored)
    tucker_query: str = ""          # "r12,r3" -> MUTAN-style MULTIPLICATIVE
                                    # pair term added into the composed query
                                    # (see TuckerQuery). "" = every prior run
                                    # bit-identical. Requires compose_query.
    dual_spatial_head: bool = False # two-expert query (semantic/spatial) mixed by
                                    # a text-side spatialness gate alpha_p
    fast_bilinear_head: bool = False  # RAM-style separable aux head: q = LN(P_s v_i
                                      # + P_o v_j) vs W — all-pairs × all-preds
                                      # scoring is two matmuls at export
    lambda_fast: float = 0.5          # weight of the fast head's aux InfoNCE
    lambda_swap: float = 0.0          # cross-slot direction hinge (v3.2); the
                                      # swap probe measured SwapAcc≈0.5 without it
    swap_margin: float = 0.05         # hinge margin on cosine scale

    # Loss weights
    lambda_cls: float = 1.0
    lambda_focal: float = 0.5
    lambda_geo: float = 1.0         # geometry pre-scorer BCE. Was 0.1 applied
                                    # to the SUM (geo + rel) — one knob
                                    # coupling a throwaway recall filter to the
                                    # relatedness head (half the deployed
                                    # score) at 10% weight. Now split;
                                    # 0.1/0.1 reproduces the <=v45 weighting.
    lambda_rel: float = 1.0         # relatedness-head BCE (pair existence —
                                    # the deployed sigmoid(rel) term). Full
                                    # citizen by default (ratified 2026-08-04).
    lambda_infonce: float = 0.5
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0

    # PCSG loss weights and temperatures
    lambda_izsa: float = 0.5    # Interaction Zone Spatial Alignment
    lambda_czsc: float = 0.2    # Cross-Zone Supervised Contrastive
    lambda_aa: float = 0.1      # Attention Anchoring (Phase 2, 0 = disabled)
    zone_temp: float = 0.07     # IZSA InfoNCE temperature
    czsc_temp: float = 0.1      # CZSC SupCon temperature

    # Background suppression (train-time only). Every predicate-head loss on
    # the batch path filters through has_gt, so the ~100 non-GT slots the
    # sampler selects contribute ZERO gradient to the pair transformer and
    # vocab head — "this pair has no relation" was delegated entirely to the
    # relatedness BCE. Nothing ever compared scores ACROSS pairs, which is the
    # by-construction cause of the near-chance true/false judgment
    # ([[relsgg-spatialsense-probe]], ECE 0.92 pre-Platt, "never stops
    # emitting"). This term supervises the FUSED score (the deployed one) on
    # valid non-GT slots, gently:
    #   - TOP-K only, not V-wide per-column BCE: gradient self-focuses on the
    #     currently-competitive columns (almost always head classes), so tail
    #     columns — which get little positive gradient — are structurally
    #     protected from the v37/v42 head-collapse failure mode.
    #   - PU-weighted per slot via the opportunity table (floor = the
    #     sampler's hedge): an unannotated plausible pair is pushed softly,
    #     an implausible one firmly.
    # 0.0 = prior runs bit-identical.
    lambda_bg: float = 0.0
    bg_topk: int = 5
    # How the per-slot "no relation holds here" penalty aggregates over the
    # vocabulary. "topk" = mean softplus over bg_topk columns (the original;
    # k=5 was never justified). "lse" = softplus(logsumexp_v z_v), which is
    # exactly the NLL of the null hypothesis under an AT-MOST-ONE model — the
    # same model deployment uses (graph-constrained argmax emits one predicate
    # per pair). Its gradient is sigma(LSE) * softmax(z)_v, so the argmax
    # column absorbs nearly all of it and a column 6 nats down gets ~0.2%:
    # stronger tail protection than a rank-k cliff, and continuous.
    # NOT the independent-Bernoulli sum(softplus(z_v)): at V~19K that puts
    # ~19000*sigma(-5)=127 of gradient mass on background columns against ~1
    # on the offending one, and collapses every column.
    bg_agg: str = "topk"

    # Role-conditioned object-semantics aux loss (compose path). The legacy
    # loss fed BOTH sub_text_proj and obj_text_proj the SAME (feature, label)
    # sets with the SAME objective — actively regressing them to one function,
    # while the compositional query needs them asymmetric. True: train
    # sub_text_proj only on boxes appearing as GT SUBJECTS and obj_text_proj
    # only on GT OBJECTS — the role-conditioned input distributions differ
    # systematically (subjects skew agentive), so the projections acquire
    # genuinely different priors. False = prior runs bit-identical.
    role_obj_loss: bool = False

    # Compositional feature augmentation (CFA-analogue, train-time only).
    # CFA (Li et al., ICCV'23) mixes ROI triplet features across samples that
    # share a predicate; we have no ROI stage, so the analogue operates on the
    # four pooled components of ``pair_input``:
    #   "entity" — mix v_sub / v_obj   (decouples the predicate from the
    #              appearance of the entities that instantiate it)
    #   "zone"   — mix v_union / v_contact (decouples it from the interaction
    #              zone's appearance)
    #   "both"   — both of the above, same partner and same lambda
    # Partners are drawn from slots ANYWHERE in the batch sharing the slot's
    # canonical predicate group, so the mix is label-preserving by
    # construction and no loss change is needed. geo_feat is never mixed: it
    # is a function of the boxes, which the augmentation does not touch.
    cfa_mode: str = "off"       # off | entity | zone | both
    cfa_prob: float = 0.0       # per-slot probability of being augmented
    cfa_alpha: float = 1.0      # lambda ~ Beta(a, a); a=1 is Uniform(0,1)
    # Partner selection. "group" = CFA proper (partner shares the canonical
    # predicate group, mix is label-preserving). "random" = MECHANISM CONTROL:
    # identical perturbation magnitude and identical lambda distribution, but
    # the partner is drawn without regard to predicate, so the mix no longer
    # pulls a slot toward its own class centroid. The group-vs-random contrast
    # is the single-variable test of whether same-predicate matching is what
    # produces the gain, or whether any feature perturbation would do.
    cfa_partner: str = "group"  # group | random

    # Inference
    predict_threshold: float = 0.3
    predict_topk_per_pair: int = 1


class TuckerQuery(nn.Module):
    """MUTAN-style multiplicative pair interaction, emitted as a TEXT-SPACE
    QUERY (Ben-younes et al., ICCV 2017 — Tucker fusion, mode-3 factor cut).

    WHY IT EXISTS. The pair fusion is concat->Linear — exactly the g_theta of
    Santoro's Relation Networks, whose known weakness is representing PRODUCTS
    of its inputs at practical width; the only multiplicative path in the whole
    head is attention's QK. Relations are second-order functions of (subject,
    object), and the additive version of this exact channel (compose_query's
    sub/obj terms) measured a 0.0% variance share with its gates driven to
    ~0.03 — so this arm asks whether the model declined the SIGNAL or only its
    additive form.

        q_tucker = P( (A v_sub) x_G (B v_obj) )        # [B, K, text_dim]

    OPEN-VOCAB CONTRACT. Classic MUTAN's third mode is a LEARNED per-predicate
    factor C[n_rels, r3] — fatal here (19,103 learned rows, unseen predicates
    unscorable). We cut the decomposition after mode 2 and emit a query; the
    frozen text bank W plays the mode-3 factor as W @ P^T inside the usual
    <q, W_v> cosine. Vocabulary stays parameter-free.

    r3 is theory-motivated, not guessed: the predicate bank's measured
    effective rank is 43.7 ([[relsgg-score-attribution]]), so a ~44-d
    bottleneck matches the output manifold the head actually regresses onto.

    P is ZERO-INIT: the term is exactly 0 at init, so step 0 is bit-identical
    to the baseline (single-mechanism diff; epoch-1 history must match ctl
    within seed noise). Gradient bootstraps in one step (dL/dP != 0 while
    A/B/G wait one optimizer step) — the standard zero-init-residual pattern.
    Deliberately NOT a scalar gate: this model drove compose_gate 0.1 -> 0.03,
    and a 1-parameter gate can strangle the pathway before it learns anything.

    fp32 island: a trilinear product under bf16 autocast is noisy and the
    tensors are tiny, so the whole forward runs with autocast disabled.

    NOTE for probe_attribution.py: this adds a FOURTH summand inside
    compose_norm. The probe reconstructs the query from (ctx, sub, obj) hooks
    and asserts the residual — on a tucker checkpoint that assertion FAILS
    LOUDLY until a tucker hook is added there. Loud is the design.
    """

    def __init__(self, d_in: int, text_dim: int, r12: int = 96, r3: int = 48):
        super().__init__()
        self.A = nn.Linear(d_in, r12, bias=False)
        self.B = nn.Linear(d_in, r12, bias=False)
        nn.init.xavier_uniform_(self.A.weight)
        nn.init.xavier_uniform_(self.B.weight)
        self.norm_a = nn.LayerNorm(r12)   # trilinear forms are scale-fragile
        self.norm_b = nn.LayerNorm(r12)
        self.G = nn.Parameter(torch.randn(r12, r12, r3) / (r12 * r3) ** 0.5)
        self.P = nn.Linear(r3, text_dim, bias=False)
        nn.init.zeros_(self.P.weight)

    def forward(self, v_sub: torch.Tensor, v_obj: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=v_sub.device.type, enabled=False):
            a = self.norm_a(self.A(v_sub.float()))            # [B, K, r12]
            b = self.norm_b(self.B(v_obj.float()))            # [B, K, r12]
            ag = torch.einsum("bki,ijr->bkjr", a, self.G)     # [B, K, r12, r3]
            g = torch.einsum("bkj,bkjr->bkr", b, ag)          # [B, K, r3]
            out = self.P(g)                                   # [B, K, text_dim]
        return out.to(v_sub.dtype)



def _box_cov(boxes: torch.Tensor, g: int) -> torch.Tensor:
    """Analytic area-coverage of normalized cxcywh boxes on a g x g grid.

    The vectorised twin of datagen.build_mask_rasters.box_raster (and
    data.relation_dataset._box_raster), so a box rasterised here is bit-for-bit
    the region the loader would have handed us had the mask been dropped.

    Args:  boxes [..., N, 4] normalized cxcywh.
    Returns: [..., N, g, g] cell-coverage fractions in [0, 1].
    """
    e = torch.arange(g + 1, device=boxes.device, dtype=boxes.dtype) / g
    lo, hi = e[:-1], e[1:]
    cx, cy, w, h = boxes.unbind(-1)
    x1, x2 = (cx - w / 2).unsqueeze(-1), (cx + w / 2).unsqueeze(-1)
    y1, y2 = (cy - h / 2).unsqueeze(-1), (cy + h / 2).unsqueeze(-1)
    ix = (torch.minimum(x2, hi) - torch.maximum(x1, lo)).clamp(min=0)
    iy = (torch.minimum(y2, hi) - torch.maximum(y1, lo)).clamp(min=0)
    return iy.unsqueeze(-1) * ix.unsqueeze(-2) * (g * g)


class RelSGG(nn.Module):
    """Label-free re-parametrizable relation prediction head.

    Args:
        config: :class:`RelSGGConfig` instance with all hyperparameters.
    """

    def __init__(self, config: RelSGGConfig = RelSGGConfig()):
        super().__init__()
        self.config = config

        self.backbone = RelAnythingBackbone(
            backbone_type=config.backbone_type,
            model_name=config.backbone_model,
            patch_size=config.patch_size,
            lora_rank=config.lora_rank,
            lora_layers=config.lora_layers,
            pretrained=config.backbone_pretrained,
            drop_path=config.drop_path,
            norm_taps=config.norm_taps,
            stage_s2d=config.stage_s2d,
            stage_weight_init=config.stage_weight_init,
        )
        backbone_dim = self.backbone.d_model

        self.box_prompt_encoder = BoxPromptEncoder(
            d_model=config.d_model, num_freqs=config.pe_num_freqs,
            max_octave=config.pe_max_octave)
        self.geo_encoder = RelGeomEncoder(d_model=config.d_model,
                                          squash=config.geo_squash,
                                          mode_gated=config.mode_gated,
            region_adjacency=getattr(config, "region_adjacency", False))
        self.spatial_pool = SoftSpatialPool(
            d_model=backbone_dim, num_freqs=config.pe_num_freqs,
            max_octave=config.pe_max_octave, scene_pe=config.scene_pe,
            role_queries=config.pool_role_queries,
            mode_gated=config.mode_gated)
        if config.sampler_type == "relatedness":
            neg_rate = neg_trusted = None
            n_cats = 0
            # pair_opportunity.npz weights statistical negatives DURING
            # TRAINING (it needs entity labels, which inference never has).
            # Released checkpoints strip the path; an old checkpoint that still
            # names a file this machine lacks loads without it.
            import os as _os
            if getattr(config, "neg_rate_table", "") and not _os.path.exists(config.neg_rate_table):
                print(f"[sampler] neg_rate_table {config.neg_rate_table} not found: "
                      "skipped (training-only; inference is unaffected)")
            if getattr(config, "neg_rate_table", "") and _os.path.exists(config.neg_rate_table):
                import numpy as _np
                _z = _np.load(config.neg_rate_table, allow_pickle=False)
                n_cats = int(_z["num_cats"])
                # float16: 17 MB at C=2884, and the rate needs no more precision than
                # that — it is a weight, not a score.
                neg_rate = torch.from_numpy(_z["rate"].astype(_np.float16))
                neg_trusted = torch.from_numpy(
                    _z["opportunities"] >= int(_z["min_support"]))
                print(f"[sampler] statistical negatives: {int(neg_trusted.sum()):,} "
                      f"trusted category pairs, median weight "
                      f"{float(1 - _np.median(_z['rate'][neg_trusted.numpy()])):.2f} "
                      f"(floor {config.rel_neg_weight})")
            self.sampler = RelatednessPairSampler(
                geo_budget=config.geo_budget,
                final_budget=config.final_budget,
                feat_dim=backbone_dim,
                neg_weight=config.rel_neg_weight,
                swap_include=config.swap_include,
                neg_rate=neg_rate, neg_trusted=neg_trusted, num_cats=n_cats,
                geo_squash=config.geo_squash,
                geo_pu=config.geo_pu,
            )
        else:
            self.sampler = CascadePairSampler(
                geo_budget=config.geo_budget,
                final_budget=config.final_budget,
                geo_squash=config.geo_squash,
            )

        # Fuse [v_sub ; v_obj ; v_union ; v_contact ; geo_feat] → d_model.
        # v_contact pools the intersection (overlapping pairs) or the gap
        # region between the boxes (disjoint pairs) — the thin interface zone
        # where contact relations live, which union-box pooling dilutes.
        pair_in = backbone_dim * 4 + config.d_model
        self.pair_proj = nn.Linear(pair_in, config.d_model)
        nn.init.xavier_uniform_(self.pair_proj.weight)
        nn.init.zeros_(self.pair_proj.bias)

        # Mode-gated capacity: zero-init second layer means it contributes
        # nothing at init, and the mode bit means a BOX image gets exactly zero
        # from it however it trains -- the same guarantee the rest of the
        # adapter carries (training/verify_mode_gate.py).
        self.mask_adapter = None
        if getattr(config, "mask_adapter_dim", 0) > 0 and config.mode_gated:
            self.mask_adapter = nn.Sequential(
                nn.Linear(pair_in, config.mask_adapter_dim),
                nn.GELU(),
                nn.Linear(config.mask_adapter_dim, config.d_model),
            )
            nn.init.zeros_(self.mask_adapter[2].weight)
            nn.init.zeros_(self.mask_adapter[2].bias)

        self.rel_transformer = RelationTransformer(
            d_model=config.d_model,
            backbone_dim=backbone_dim,
            n_self=config.n_self_layers,
            n_cross=config.n_cross_layers,
            n_heads=config.n_heads,
            ffn_ratio=config.ffn_ratio,
            dropout=config.dropout,
            scene_pe=config.scene_pe,
        )

        self.vocab_head = VocabHead(
            d_model=config.d_model,
            text_dim=config.text_dim,
            text_model_name=config.text_model,
            logit_scale_init=config.logit_scale_init,
            infonce_temp=config.infonce_temp,
            logit_bias_init=config.logit_bias_init,
            proj_layers=config.proj_layers,
        )

        # How predicate and relatedness logits become one score, shared with
        # eval and the deployed host (relsgg/scoring.py). Identity by default;
        # install a fitted one with `set_score_contract` so that `predict`
        # returns the same numbers the product shows. Not a Parameter and not
        # in the state_dict — the calibration is fitted AFTER training, on a
        # val split, and travels next to the checkpoint as calibration.json.
        from relsgg.scoring import ScoreContract
        self.score_contract = ScoreContract()
        self._score_mode: Optional[str] = None   # None = resolve_score_mode()

        if config.compose_query:
            # Compositional query (label-free): subject/object VISUAL semantics
            # are projected into the text space and summed with the pair
            # context. The same projections feed the object-semantics aux loss
            # (alignment to category-name text embeddings, train-time only).
            _t_dim = config.text_dim if config.text_dim is not None else config.d_model
            self.sub_text_proj = nn.Linear(backbone_dim, _t_dim, bias=False)
            self.obj_text_proj = nn.Linear(backbone_dim, _t_dim, bias=False)
            nn.init.xavier_uniform_(self.sub_text_proj.weight)
            nn.init.xavier_uniform_(self.obj_text_proj.weight)
            self.compose_norm = nn.LayerNorm(_t_dim)
            # gates start small: query begins ≈ proj(r), composition grows in
            self.compose_gate = nn.Parameter(torch.tensor([0.1, 0.1]))
            self.register_buffer("W_obj", torch.empty(0))  # category-name texts
            if config.tucker_query:
                _r12, _r3 = (int(x) for x in config.tucker_query.split(","))
                self.tucker_query = TuckerQuery(backbone_dim, _t_dim, _r12, _r3)

        if config.tucker_query and not config.compose_query:
            # The tucker term lives inside the composed-query sum (and its
            # LayerNorm); without compose_query there is no q to add it to and
            # the flag would silently train a baseline wearing a tucker name.
            raise ValueError("tucker_query requires compose_query "
                             "(loss_type batch_infonce or --compose_query 1)")

        if config.beta_relatedness:
            self.vocab_head.build_beta_mlp()
        if config.dual_spatial_head:
            # Spatial expert query: fed by the pair context PLUS a direct
            # geometry path ([r ; geo_feat]) so direction learning (above vs
            # below — invisible to the text space) no longer competes with
            # semantic representation learning inside one shared projection.
            # Routing between the experts is per-predicate, from the text
            # side (VocabHead.alpha) — object identity stays irrelevant here.
            _t_dim = config.text_dim if config.text_dim is not None else config.d_model
            _hidden = max(config.d_model * 2, _t_dim // 2)
            self.spa_proj = nn.Sequential(
                nn.Linear(config.d_model * 2, _hidden), nn.GELU(),
                nn.LayerNorm(_hidden), nn.Linear(_hidden, _t_dim, bias=False),
            )
            for m in self.spa_proj:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if config.fast_bilinear_head:
            # Separable pair scorer (Relate-Anything-Model style, lifted to
            # the open text space): q_ij = LN(P_s v_i + P_o v_j) scored
            # against the shared frozen W. No pair transformer, no geometry —
            # direction is carried entirely by the P_s/P_o asymmetry, and
            # all-pairs × all-predicates scoring factorizes into two N×V
            # matmuls + a broadcast add (the fixed-shape ONNX export path).
            _t_dim = config.text_dim if config.text_dim is not None else config.d_model
            self.fast_sub = nn.Linear(backbone_dim, _t_dim, bias=False)
            self.fast_obj = nn.Linear(backbone_dim, _t_dim, bias=False)
            nn.init.xavier_uniform_(self.fast_sub.weight)
            nn.init.xavier_uniform_(self.fast_obj.weight)
            self.fast_norm = nn.LayerNorm(_t_dim)

        self.criterion = RelSGGLoss(
            lambda_cls=config.lambda_cls,
            lambda_focal=config.lambda_focal,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
        )

        # PCSG: zone projection head (backbone_dim → text_dim) + grounding
        # loss — only on the non-batch_infonce paths (compose_query is set iff
        # batch_infonce); on the batch path this was 1.6M params of dead
        # weight riding along in every checkpoint.
        if not config.compose_query:
            _text_dim = config.text_dim if config.text_dim is not None else config.d_model
            self.zone_proj = nn.Linear(backbone_dim, _text_dim)
            nn.init.xavier_uniform_(self.zone_proj.weight)
            nn.init.zeros_(self.zone_proj.bias)

            self.spatial_criterion = RelSpatialGroundingLoss(
                zone_temp=config.zone_temp,
                czsc_temp=config.czsc_temp,
            )

        if config.deformable_points > 0:
            # Each sub-flag falls back to the v2 bundle only when left unset,
            # so old checkpoints rebuild identically while new arms can pick
            # ring init WITHOUT dragging in border padding.
            _v2 = config.deformable_v2
            _pick = lambda x: _v2 if x is None else x  # noqa: E731
            _n_lv = 1
            if (config.ms_depth_levels > 0 or config.ms_pool_level
                    or config.ms_deconv_level):
                if config.ms_depth_levels > len(self.backbone.layer_offsets):
                    raise ValueError(
                        f"ms_depth_levels={config.ms_depth_levels} exceeds the "
                        f"{len(self.backbone.layer_offsets)} backbone taps")
                self.ms_scene = MultiLevelScene(
                    backbone_dim=backbone_dim, d_model=config.d_model,
                    n_depth_levels=config.ms_depth_levels,
                    pool_level=config.ms_pool_level,
                    deconv_level=config.ms_deconv_level)
                _n_lv = self.ms_scene.n_levels
            self.deformable_read = DeformableRelRead(
                d_model=config.d_model, n_points=config.deformable_points,
                heads=config.deformable_heads,
                ring_init=_pick(config.deformable_ring),
                gain_gate=_pick(config.deformable_gain),
                border_pad=_pick(config.deformable_border),
                null_slots=config.deformable_nulls,
                clamp_to_image=config.deformable_clamp,
                n_levels=_n_lv)

        if config.use_rel_interaction:
            self.rel_interaction = RelationInteractionBlock(
                d_model=config.d_model,
                scene_dim=backbone_dim,
                n_dep=config.n_dep_layers,
                n_gnd=config.n_gnd_layers,
                n_heads=config.n_heads,
                ffn_ratio=config.ffn_ratio,
                dropout=config.dropout,
                scene_pe=config.scene_pe,
            )

        if config.depth_scaled_init:
            n_res = depth_scaled_residual_init_(
                [self.rel_transformer, getattr(self, "rel_interaction", None)])
            print(f"[init] depth-scaled residual init over {n_res} branches "
                  f"(scale {n_res ** -0.5:.3f})")

    # ------------------------------------------------------------------
    # Vocabulary API — delegates to VocabHead
    # ------------------------------------------------------------------

    def encode_vocabulary(self, pred_names: List[str]) -> None:
        """Encode predicate names with the text encoder.

        Call this once before inference / after loading a checkpoint.
        """
        self.vocab_head.encode_vocabulary(pred_names)

    def set_score_mode(self, mode: Optional[str]) -> None:
        """Pin the decode formula: "sigmoid", "softmax", or None to resolve."""
        if mode not in (None, "sigmoid", "softmax"):
            raise ValueError(f"score_mode must be sigmoid/softmax/None, got {mode}")
        self._score_mode = mode

    def resolve_score_mode(self) -> str:
        """Which formula `predict` scores with.

        THIS USED TO BE INFERRED FROM TRAINING STATE, and that was a bug. The
        test was `hasattr(self, "batch_infonce") or hasattr(self,
        "synonym_criterion")` — attributes installed by `install_synonym_losses`,
        which the TRAINING entrypoint calls and nothing else does. So a model
        rebuilt for inference (benchmark/eval_zeroshot.build_model_from_ckpt and
        every consumer of it) failed the test and silently fell through to the
        legacy `softmax(pred) * sigmoid(rel)` — the exact five-way divergence
        relsgg/scoring.py was written to end. Measured on a batch_infonce
        checkpoint over 19,103 predicates: the same relations score 0.013 /
        0.008 / 0.008 under the legacy branch and 0.990 / 0.933 / 0.930 under
        the contract, and the two orderings agree on only 7.1 of the top 12
        triplets per image (worst image: 1 of 12). Rank metrics were unaffected
        because relsgg/evaluator.py takes score_mode as an explicit argument;
        `predict` is the API surface that guessed.

        `config.compose_query` is set iff `loss_type == "batch_infonce"`, is
        persisted in the checkpoint, and is therefore the marker that survives a
        rebuild. It is ORed with the old attribute test rather than replacing
        it, so no model that got sigmoid before loses it. The vwide/synonym
        path has no equivalent config marker; rather than guess one, pin it
        with `set_score_mode("sigmoid")` at the call site.
        """
        if getattr(self, "_score_mode", None) is not None:
            return self._score_mode
        if (getattr(self.config, "compose_query", False)
                or hasattr(self, "synonym_criterion")
                or hasattr(self, "batch_infonce")):
            return "sigmoid"
        return "softmax"

    def set_score_contract(self, contract) -> None:
        """Install the deployment score contract (relsgg.scoring.ScoreContract).

        Monotone, so `predict`'s RANKING is untouched; what changes is the
        number attached to each triplet, and therefore what a threshold means.
        """
        self.score_contract = contract

    def reparameterize(self) -> None:
        """Fuse predicate text embeddings into a static weight matrix.

        After this call the text encoder is removed; the forward pass has
        zero language-model overhead.  Swap the vocabulary at any time by
        calling ``encode_vocabulary()`` again, followed by ``reparameterize()``.
        """
        self.vocab_head.reparameterize()

    def set_ontology(
        self,
        ontology: PredicateOntology,
        neg_weight: float = 0.3,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        mode: str = "batch",
        n_neg: int = 256,
        lambda_obj: float = 0.3,
        hard_lo: float = 0.5,
        cooc_path: Optional[str] = None,
        soft_neg_weight: float = 0.3,
        min_support: int = 30,
        pos_agg: str = "lse",
        pos_member_weight: float = 0.0,
    ) -> None:
        """Install synonym-aware training losses.

        mode="batch" (default): batch-local InfoNCE with sampled hard
            negatives (GLIP/YOLO-World-style) — the full vocabulary is used
            only at inference; nothing supervises a V-wide distribution.
            Pairs with compose_query + the object-semantics aux loss.
        mode="vwide": earlier masked-sigmoid formulation (kept for ablation).
        Call before ``.to(device)`` / DDP wrapping. The legacy CE path stays
        available by simply not calling this at all.
        """
        self.ontology = ontology
        self.register_buffer("canon_group_of", ontology.group_of,
                             persistent=False)
        if mode == "batch":
            self.batch_infonce = BatchLocalInfoNCE(
                ontology, temp=self.config.infonce_temp, n_neg=n_neg,
                hard_lo=hard_lo, cooc_path=cooc_path,
                soft_neg_weight=soft_neg_weight, min_support=min_support,
                pos_agg=pos_agg, pos_member_weight=pos_member_weight,
            )
            self.lambda_obj = lambda_obj
            # The batch-local InfoNCE recomputes cosines at a FIXED temperature
            # and never touches logit_scale/logit_bias — they get gradient only
            # from sigmoid-family terms (lambda_sigmoid / lambda_bg), which
            # is how they trained as dead weight for ten versions
            # ([[relsgg-untrained-output-head]]). Division of labor: InfoNCE
            # trains direction (scale-invariant ranking), sigmoid terms train
            # the deployed affine calibration. When no sigmoid term is active,
            # freeze the affine EXPLICITLY — frozen-by-design and loud, never
            # silently half-trained; calibration then comes from the post-hoc
            # score contract (Platt).
            _trains_affine = (self.config.lambda_sigmoid > 0.0
                              or getattr(self.config, "lambda_bg", 0.0) > 0.0)
            if not _trains_affine:
                self.vocab_head.logit_scale.requires_grad_(False)
                self.vocab_head.logit_bias.requires_grad_(False)
                print("[vocab_head] logit_scale/logit_bias FROZEN by design: "
                      "no sigmoid-family loss active on the batch path "
                      "(lambda_sigmoid=0, lambda_bg=0) — calibrate post-hoc "
                      "or enable one to train the affine.")
        else:
            self.synonym_criterion = SynonymAwareRelLoss(
                ontology, neg_weight=neg_weight,
                focal_gamma=focal_gamma, focal_alpha=focal_alpha,
            )
            self.mp_infonce = MultiPositiveInfoNCE(
                ontology, temp=self.config.infonce_temp
            )

    @torch.no_grad()
    def set_object_vocabulary(self, names: List[str], W_obj: torch.Tensor) -> None:
        """Install category-name text embeddings for the compositional aux
        loss (train-time only; inference never needs labels)."""
        self.obj_names = list(names)
        self.W_obj = F.normalize(W_obj.float(), dim=-1).to(
            self.vocab_head.logit_scale.device).clone()

    def _object_text_loss(
        self,
        v_obj: torch.Tensor,          # [B, N, backbone_dim]
        box_counts: torch.Tensor,     # [B]
        targets: List[dict],
    ) -> torch.Tensor:
        """InfoNCE aligning per-box subject/object projections to the
        category-name text space (497 classes — small, no sampling needed).

        With ``config.role_obj_loss`` the two projections get ROLE-DISJOINT
        training sets (GT subjects vs GT objects) instead of identical ones —
        see the config comment.
        """
        if self.W_obj.numel() == 0:
            return v_obj.new_zeros(())

        if self.config.role_obj_loss:
            feats_by_role: list = [[], []]     # 0 = subject, 1 = object
            labels_by_role: list = [[], []]
            for b, t in enumerate(targets):
                el = t.get("entity_labels")
                rels = t.get("relations")
                if el is None or rels is None or len(rels) == 0:
                    continue
                el = el.to(v_obj.device)
                n = min(int(box_counts[b]), el.shape[0])
                for role, sink_f, sink_l in ((0, feats_by_role[0], labels_by_role[0]),
                                             (1, feats_by_role[1], labels_by_role[1])):
                    ids = rels[:, role].unique()
                    ids = ids[ids < n]
                    ids = ids[el[ids] >= 0]
                    if ids.numel():
                        sink_f.append(v_obj[b, ids])
                        sink_l.append(el[ids])
            loss = v_obj.new_zeros(())
            n_terms = 0
            for proj, fl, ll in ((self.sub_text_proj, feats_by_role[0], labels_by_role[0]),
                                 (self.obj_text_proj, feats_by_role[1], labels_by_role[1])):
                if not fl:
                    continue
                f = torch.cat(fl)
                lab = torch.cat(ll).to(f.device)
                q = F.normalize(proj(f), dim=-1)
                loss = loss + F.cross_entropy(
                    q @ self.W_obj.T / self.config.infonce_temp, lab)
                n_terms += 1
            return loss / max(n_terms, 1)

        feats, labels = [], []
        for b, t in enumerate(targets):
            el = t.get("entity_labels")
            if el is None:
                continue
            n = min(int(box_counts[b]), el.shape[0])
            keep = el[:n] >= 0
            if keep.any():
                feats.append(v_obj[b, :n][keep])
                labels.append(el[:n][keep])
        if not feats:
            return v_obj.new_zeros(())
        feats = torch.cat(feats)
        labels = torch.cat(labels).to(feats.device)
        loss = feats.new_zeros(())
        for proj in (self.sub_text_proj, self.obj_text_proj):
            q = F.normalize(proj(feats), dim=-1)
            logits = q @ self.W_obj.T / self.config.infonce_temp
            loss = loss + F.cross_entropy(logits, labels)
        return loss * 0.5

    # ------------------------------------------------------------------
    # Compositional feature augmentation (train-time only)
    # ------------------------------------------------------------------

    def _cfa_partners(
        self,
        group: torch.Tensor,   # [B, K] canonical group id per slot
        ok: torch.Tensor,      # [B, K] bool — slot has a GT predicate
    ) -> Optional[tuple]:
        """Pair every eligible slot with a random OTHER batch slot.

        With ``cfa_partner="group"`` (CFA proper) the partner shares the
        slot's canonical predicate group, so the mix is label-preserving.
        With ``cfa_partner="random"`` the partner is predicate-blind — the
        mechanism control.

        Returns ``(src, dst)`` index pairs into the flattened [B*K] slot axis,
        or None when nothing is eligible. Groups with a single member are
        dropped rather than self-matched, so ``cfa_prob`` is not silently
        diluted by no-op mixes.

        IMPORTANT (matched-arm requirement): the ``src`` set is computed the
        SAME way in both modes — singleton groups are dropped even in
        "random", where nothing would force it. Without that, the random arm
        would augment strictly more slots than the group arm and the contrast
        would confound "which partner" with "how much augmentation".
        """
        idx = (ok.reshape(-1)).nonzero(as_tuple=True)[0]          # [M]
        if idx.numel() < 2:
            return None

        g = group.reshape(-1)[idx]
        order = torch.argsort(g)
        _, counts = torch.unique_consecutive(g[order], return_counts=True)
        starts = torch.cumsum(counts, 0) - counts
        start_per = torch.repeat_interleave(starts, counts)        # [M]
        count_per = torch.repeat_interleave(counts, counts)        # [M]
        # Draw uniformly from the group EXCLUDING the slot's own position:
        # sample in [0, n-2] then skip over self.
        pos = torch.arange(g.shape[0], device=g.device) - start_per
        r = (torch.rand(g.shape[0], device=g.device)
             * (count_per - 1).clamp(min=1)).long()
        r = torch.minimum(r, (count_per - 2).clamp(min=0))
        r = r + (r >= pos).long()
        # Drop singleton groups BEFORE gathering: for count_per == 1 the
        # self-skip above yields start_per + 1, which points past the group —
        # and off the end of `order` entirely when the singleton sorts last
        # (device-side "index out of bounds" assert). Singletons are the
        # common case on a vocabulary with thousands of canonical groups.
        keep = count_per > 1
        if not bool(keep.any()):
            return None
        src = idx[order[keep]]

        if self.config.cfa_partner == "random":
            # Same src set, same count — only the partner criterion differs.
            # Draw partners from the full eligible pool and resolve any
            # self-pair by taking the next slot round; a residual self-pair is
            # a harmless no-op mix, never a crash.
            pick = torch.randint(idx.numel(), (src.numel(),), device=idx.device)
            dst = idx[pick]
            clash = dst == src
            if bool(clash.any()):
                dst = torch.where(clash, idx[(pick + 1) % idx.numel()], dst)
            return (src, dst) if src.numel() else None

        dst = idx[order[(start_per + r)[keep]]]
        return (src, dst) if src.numel() else None

    def _cfa_mix(
        self,
        feats: tuple,          # (v_sub, v_obj, v_union, v_contact), each [B,K,d]
        group: torch.Tensor,
        ok: torch.Tensor,
    ) -> tuple:
        """Blend the selected pooled components with a same-predicate partner.

        One lambda and one partner per slot across every blended component, so
        a mixed triplet stays internally coherent (subject and object come
        from the same partner pair, not two different ones).
        """
        pairs = self._cfa_partners(group, ok)
        if pairs is None:
            return feats
        src, dst = pairs
        dev, n = src.device, src.shape[0]

        conc = torch.full((1,), float(self.config.cfa_alpha), device=dev)
        lam = torch.distributions.Beta(conc, conc).sample((n,)).reshape(n)
        # lambda = 1 leaves the slot untouched, so the probability gate is
        # just a where() and costs no branching.
        lam = torch.where(
            torch.rand(n, device=dev) < self.config.cfa_prob,
            lam, torch.ones_like(lam),
        ).unsqueeze(-1)

        mode = self.config.cfa_mode
        blend_entity = mode in ("entity", "both")
        blend_zone = mode in ("zone", "both")
        selected = (blend_entity, blend_entity, blend_zone, blend_zone)

        out = []
        for x, do in zip(feats, selected):
            if not do:
                out.append(x)
                continue
            flat = x.reshape(-1, x.shape[-1])
            mixed = flat.clone()
            # index the ORIGINAL flat tensor for the partner term so the mix
            # never reads an already-updated row
            mixed[src] = lam * flat[src] + (1.0 - lam) * flat[dst]
            out.append(mixed.reshape(x.shape))
        return tuple(out)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        box_counts: Optional[torch.Tensor] = None,
        targets: Optional[List[dict]] = None,
        force_attn: bool = False,
        cov: Optional[torch.Tensor] = None,
        fill: Optional[torch.Tensor] = None,
        precomputed_features: Optional[torch.Tensor] = None,
        mode: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Run the full pipeline.

        Args:
            images:     [B, 3, H, W] float32 in [0, 1].
                        Will be preprocessed (ImageNet normalisation) internally.
            boxes:      [B, max_N, 4] normalized cxcywh, zero-padded.
            box_counts: [B] actual number of valid boxes per image.
                        Defaults to max_N for all images if None.
            targets:    List[dict] at training time, each dict containing:
                          ``relations``: LongTensor [R, 3] of
                          (sub_idx, obj_idx, pred_label).
                        Pass None at inference.
            force_attn: Return cross-attention weights over the patch grid
                        in ``out["attn_weights"]`` regardless of the AA loss
                        being enabled (``lambda_aa``) — for visualization
                        only, no training-time cost when left False.
        Returns:
            Dict with keys:
              ``logits``      [B, K, V]
              ``sub_idx``     [B, K]
              ``obj_idx``     [B, K]
              ``valid_mask``  [B, K]
              ``pred_labels`` [B, K]
              ``loss``        scalar (only if targets is not None)
              ``loss_dict``   dict of component losses (only if targets)
        """
        B, max_N, _ = boxes.shape

        # 0. Region rasters. `cov` is [B, N, g, g] coverage in [0,1] — a BOX
        # rasterizes to its rectangle and a MASK to its shape, so there is one
        # representation and no modality branch. The collate_fn stashes them on
        # the targets list so the 4-tuple loader contract stays unchanged.
        if cov is None:
            cov = getattr(targets, "cov", None)
        if fill is None:
            fill = getattr(targets, "fill", None)
        # Mode bit [B] (1 = this image's rasters are masks, 0 = boxes). Only a
        # mode_gated model consumes it; legacy mask arms must not see it, or
        # their dropped images would silently lose the lambda bias they trained with.
        if not getattr(self.config, "mode_gated", False) or cov is None:
            mode = None
        else:
            if mode is None:
                mode = getattr(targets, "mode", None)
            if mode is None:
                # rasters with no bit (eval --rasters): mask mode
                mode = torch.ones(boxes.shape[0], device=boxes.device)
            mode = mode.to(boxes.device, dtype=boxes.dtype)
        if cov is None and getattr(self, "expect_region", False):
            raise RuntimeError(
                "region rasters were requested (--rasters) but none reached the "
                "model, so this would train in BOX mode while reporting a mask "
                "run. Pass them as explicit tensor kwargs: riding on `targets` "
                "fails under DDP, whose scatter rebuilds any list as a plain "
                "list and drops the TargetList attributes (jobs 6922420/6922421 "
                "silently trained maskless this way). See "
                "relsgg.train_engine._region_kwargs.")

        r_iou = r_contact = None
        r_adj = None
        cov_flat = None
        if cov is not None:
            cov = cov.to(boxes.device, dtype=boxes.dtype)
            if cov.dtype == torch.uint8:
                cov = cov.float() / 255.0
            cov_flat = cov.flatten(2)                                # [B, N, g*g]
            area = cov_flat.sum(-1)                                  # [B, N]
            inter = torch.einsum("bnc,bmc->bnm", cov_flat, cov_flat)  # [B, N, N]
            a_s, a_o = area.unsqueeze(2), area.unsqueeze(1)
            r_iou = inter / (a_s + a_o - inter).clamp(min=1e-6)
            r_contact = inter / torch.minimum(a_s, a_o).clamp(min=1e-6)
            if getattr(self.config, "region_adjacency", False):
                # Dilate by ONE cell (max-pool 3x3, stride 1) and re-intersect:
                # for masks that partition the image the plain intersection is
                # exactly 0 for every pair, so this band is the only place a
                # "these two regions touch" signal can come from. Symmetrised
                # over both directions to match geometry.box_adjacency.
                g_cells = cov.shape[-1]

                def _adj(c):
                    """Symmetric one-cell-dilated overlap / min(area)."""
                    flat = c.flatten(2)
                    ar = flat.sum(-1)
                    d = F.max_pool2d(c.flatten(0, 1).unsqueeze(1), 3,
                                     stride=1, padding=1).view(c.shape).flatten(2)
                    it = torch.einsum("bnc,bmc->bnm", d, flat)
                    it = 0.5 * (it + it.transpose(1, 2))
                    return it / torch.minimum(ar.unsqueeze(2),
                                              ar.unsqueeze(1)).clamp(min=1e-6)

                # BOTH sides from the same rasteriser. An analytic box reference
                # is off by ~0.4 on box regions, because max-pooling fractional
                # edge coverage grows the support by up to two cells while the
                # analytic box grows by exactly one -- the column would then
                # encode that artefact rather than mask-vs-box. Rasterising the
                # boxes here makes the delta vanish identically for a box.
                r_adj = _adj(cov) - _adj(_box_cov(boxes, g_cells))
            if fill is not None:
                fill = fill.to(boxes.device, dtype=boxes.dtype)

        # 1. Extract dense scene features (with gradient for backbone fine-tuning)
        # precomputed_features lets a deployment run the backbone CONCURRENTLY
        # with an external detector (the backbone does not depend on `boxes`,
        # so the two are independent branches — see deploy/pipeline.py). None =
        # every prior call path bit-identical.
        _taps = None
        if precomputed_features is not None:
            F_map = precomputed_features
            if getattr(self, "ms_scene", None) is not None \
                    and self.ms_scene.n_depth_levels > 0:
                # The feature cache stores the FUSED map only, so the depth
                # levels it would need are unrecoverable. Fail loudly rather
                # than silently degrade to a fused-only read (the
                # silently-trained-maskless class of bug).
                raise ValueError(
                    "precomputed_features cannot serve ms_depth_levels>0: the "
                    "cache holds the fused map, not the per-tap features. "
                    "Cache taps too, or run this arm without the cache.")
        else:
            images_norm = self.backbone.preprocess(images)
            if getattr(self, "ms_scene", None) is not None \
                    and self.ms_scene.n_depth_levels > 0:
                F_map, _taps = self.backbone.extract(images_norm,
                                                     return_taps=True)
            else:
                F_map = self.backbone.extract(images_norm)  # [B,h,w,d_backbone]

        # Coverage at the patch grid, for the pooling shape-gate.
        cov_grid = cov_grid_dil = None
        if cov is not None:
            h_f, w_f = F_map.shape[1], F_map.shape[2]
            cov_grid = F.adaptive_avg_pool2d(
                cov.reshape(B * cov.shape[1], 1, cov.shape[2], cov.shape[3]),
                (h_f, w_f),
            ).reshape(B, cov.shape[1], h_f * w_f)
            if getattr(self.config, "contact_field", False):
                cov_grid_dil = F.max_pool2d(
                    cov_grid.view(B * cov.shape[1], 1, h_f, w_f), 3,
                    stride=1, padding=1).reshape(B, cov.shape[1], h_f * w_f)

        # 2. Per-object features via coordinate-aware soft pooling: [B, max_N, d_backbone]
        v_obj = self.spatial_pool(F_map, boxes, cov=cov_grid, mode=mode)

        # 3. Sample K pairs per image
        # Entity categories for the sampler's statistically-weighted negatives. Training
        # only — `targets` is None at inference, so the label-free contract is intact.
        el_sampler = None
        if targets is not None and getattr(self.sampler, "neg_rate", None) is not None:
            el_sampler = boxes.new_full((boxes.shape[0], boxes.shape[1]), -1,
                                        dtype=torch.long)
            for b, t in enumerate(targets):
                el = t.get("entity_labels")
                if el is not None and el.numel():
                    n = min(el.shape[0], boxes.shape[1])
                    el_sampler[b, :n] = el[:n].to(el_sampler.device)
        sampler_out = self.sampler(
            boxes=boxes,
            obj_feats=v_obj,
            box_counts=box_counts,
            targets=targets,
            entity_labels=el_sampler,
        )
        if len(sampler_out) == 7:   # RelatednessPairSampler
            (sub_idx, obj_idx, valid_mask, pred_labels,
             geo_loss, rel_loss, rel_logits) = sampler_out
        else:                       # CascadePairSampler
            sub_idx, obj_idx, valid_mask, pred_labels, geo_loss = sampler_out
            rel_loss = boxes.new_zeros(())
            rel_logits = None
        K = sub_idx.shape[1]

        # 4. Gather pair features
        B_idx = torch.arange(B, device=boxes.device).unsqueeze(1).expand(B, K)

        sub_boxes_k = boxes[B_idx, sub_idx]   # [B, K, 4]
        obj_boxes_k = boxes[B_idx, obj_idx]   # [B, K, 4]
        v_sub = v_obj[B_idx, sub_idx]          # [B, K, d_backbone]
        v_obj_k = v_obj[B_idx, obj_idx]        # [B, K, d_backbone]

        # DIAGNOSTIC HOOK — training/e3_appearance_probe.py. Replaces the
        # subject/object APPEARANCE while leaving the boxes, the sampled pair
        # set, and every geometry feature untouched, which is what makes the
        # measurement causal: intervening earlier (e.g. on spatial_pool's
        # output) also perturbs the sampler and changes WHICH pairs exist, so
        # the two runs would no longer be comparable pair-by-pair.
        # `None` in every normal run — training and eval never set it.
        _swap = getattr(self, "_probe_swap", None)
        if _swap is not None:
            v_sub, v_obj_k = _swap(v_obj, B_idx, sub_idx, obj_idx, v_sub, v_obj_k)

        union_boxes_k = union_box(sub_boxes_k, obj_boxes_k)              # [B, K, 4]

        # The union of two regions is the elementwise max of their coverage —
        # the same operation for boxes and masks. The CONTACT zone below stays
        # box-derived: it is a geometric construct (the interface between two
        # extents), not a region either object actually occupies.
        union_cov = None
        region_k = None
        region_adj_k = None
        if cov_grid is not None:
            union_cov = torch.maximum(cov_grid[B_idx, sub_idx],
                                      cov_grid[B_idx, obj_idx])
            f = fill if fill is not None else torch.ones_like(boxes[..., 0])
            region_k = (f[B_idx, sub_idx], f[B_idx, obj_idx],
                        r_iou[B_idx, sub_idx, obj_idx],
                        r_contact[B_idx, sub_idx, obj_idx])
            if r_adj is not None:
                region_adj_k = r_adj[B_idx, sub_idx, obj_idx]

        # NOTE: the union pool is deliberately NOT issued here — it is merged
        # with the contact pool below into a single cross-attention call (5b').
        geo_feat = self.geo_encoder(
            sub_boxes_k, obj_boxes_k, region_k,
            mode=None if mode is None else mode.view(B, 1, 1),
            region_adj=region_adj_k)                            # [B, K, d_model]

        # 5. Build box corner tokens for extended cross-attention memory
        # BoxPromptEncoder expects xyxy; boxes are cxcywh — convert
        def _cxcywh_to_xyxy(b: torch.Tensor) -> torch.Tensor:
            cx, cy, w, h = b.unbind(-1)
            return torch.stack([cx - w * 0.5, cy - h * 0.5,
                                cx + w * 0.5, cy + h * 0.5], -1).clamp(0, 1)

        # Both members in ONE conversion (clamp is elementwise, so a single
        # clamp on the stacked tensor is bit-identical to four on the parts).
        _pair_xyxy = _cxcywh_to_xyxy(torch.stack([sub_boxes_k, obj_boxes_k], 0))
        sub_xyxy, obj_xyxy = _pair_xyxy[0], _pair_xyxy[1]
        box_tokens = self.box_prompt_encoder.encode_pairs(sub_xyxy, obj_xyxy)  # [B, K, 4, d_model]

        # 5b. Contact-region token: elementwise min/max of the "inner" corner
        # coordinates gives the intersection box when the pair overlaps and
        # the gap region between facing edges when it doesn't — either way,
        # the interface zone. SoftSpatialPool is attention-based, so
        # degenerate/zero-area boxes are safe (no area division).
        inner_x1 = torch.maximum(sub_xyxy[..., 0], obj_xyxy[..., 0])
        inner_y1 = torch.maximum(sub_xyxy[..., 1], obj_xyxy[..., 1])
        inner_x2 = torch.minimum(sub_xyxy[..., 2], obj_xyxy[..., 2])
        inner_y2 = torch.minimum(sub_xyxy[..., 3], obj_xyxy[..., 3])
        # Sorting the interval endpoints is unnecessary: the midpoint of
        # {min, max} is (a+b)/2 whichever way round they are, and the extent is
        # |a-b|. Bit-identical (IEEE addition is commutative and fl(a-b) is the
        # exact negation of fl(b-a)), four fewer kernels on a dispatch-bound
        # path, and it says what the box IS instead of how it is sorted.
        contact_boxes_k = torch.stack(
            [(inner_x1 + inner_x2) * 0.5, (inner_y1 + inner_y2) * 0.5,
             (inner_x2 - inner_x1).abs(), (inner_y2 - inner_y1).abs()], -1)
        # 5b'. MERGED POOL: union + contact in ONE cross-attention call.
        # SoftSpatialPool is a cross-attn of N box queries against the SAME
        # h*w patch tokens, so stacking the two query sets along N is exactly
        # the two separate calls — same math, one dispatch sequence instead of
        # two. Measured motivation: spatial_pool was the single most expensive
        # module at bs1 (6.7 ms of CPU dispatch, 3 calls/frame) on a model that
        # is dispatch-bound, not compute-bound ([[relsgg-inference-launch-bound]]).
        _pool_boxes = torch.cat([union_boxes_k, contact_boxes_k], dim=1)  # [B,2K,4]
        _pool_cov = None
        if union_cov is not None:
            # The contact half must reproduce the cov=None branch EXACTLY.
            # The bias is cov_lambda * log(cov.clamp(0,1) + 1e-4), so cov =
            # 1 - 1e-4 gives log(1.0) = 0 — bit-identical to no bias at all.
            if cov_grid_dil is not None:
                # Proximity field: high only where BOTH regions are within one
                # patch cell. Far-apart pairs give an all-zero field == constant
                # bias == no bias (softmax is shift-invariant), so they keep the
                # flat behaviour without a special case.
                _second = torch.minimum(cov_grid_dil[B_idx, sub_idx],
                                        cov_grid_dil[B_idx, obj_idx])
            else:
                _second = union_cov.new_full(union_cov.shape, 1.0 - 1e-4)
            _pool_cov = torch.cat([union_cov, _second], dim=1)
        _pool_roles = None
        if self.config.pool_role_queries:
            _pool_roles = torch.cat([
                torch.full((K,), 1, dtype=torch.long, device=boxes.device),
                torch.full((K,), 2, dtype=torch.long, device=boxes.device)])
        _v_pool = self.spatial_pool(F_map, _pool_boxes, cov=_pool_cov,
                                    roles=_pool_roles, mode=mode)
        v_union, v_contact = _v_pool[:, :K], _v_pool[:, K:]   # each [B,K,d_backbone]

        # 5c. CFA-analogue augmentation: recombine pooled components across
        # same-predicate slots. Train-only, label-preserving, shape-preserving
        # (so --static_shapes is unaffected). NOTE the architecture-specific
        # caveat: rel_transformer cross-attends the UNMIXED F_map downstream,
        # so the scene context of the original slot survives the mix — this is
        # weaker augmentation than CFA gets on a pure-ROI pipeline.
        if (self.training and targets is not None
                and self.config.cfa_mode != "off"
                and self.config.cfa_prob > 0.0
                and hasattr(self, "canon_group_of")):
            _grp = self.canon_group_of[pred_labels.clamp(min=0)]
            v_sub, v_obj_k, v_union, v_contact = self._cfa_mix(
                (v_sub, v_obj_k, v_union, v_contact),
                _grp, (pred_labels >= 0) & valid_mask,
            )

        # 6. Fuse into pair representation
        pair_input = torch.cat([v_sub, v_obj_k, v_union, v_contact, geo_feat], dim=-1)  # [B, K, 4*d_b+d]
        pair_feat = self.pair_proj(pair_input)                                 # [B, K, d_model]
        if self.mask_adapter is not None and mode is not None:
            pair_feat = pair_feat + mode.view(B, 1, 1).to(pair_feat.dtype) * \
                self.mask_adapter(pair_input)

        # 7. Relation transformer (self-attn + cross-attn to scene + box tokens)
        padding_mask = ~valid_mask  # [B, K]
        _return_attn = force_attn or (targets is not None and self.config.lambda_aa > 0)
        # Box-token (modality) dropout, PER SAMPLE (was per-step, whole-batch:
        # one Bernoulli per step is a high-variance regularizer and every
        # gradient step saw only one modality regime). Each image now loses
        # its box tokens independently with prob p via the cross-attn
        # key-padding mask; inference always keeps them.
        _bt_drop = None
        if self.training and self.config.box_token_dropout > 0.0:
            _bt_drop = (torch.rand(B, device=boxes.device)
                        < self.config.box_token_dropout)
        _transformer_out = self.rel_transformer(
            pair_feat, F_map,
            box_tokens=box_tokens,
            pair_padding_mask=padding_mask,
            return_attn=_return_attn,
            box_token_drop=_bt_drop,
        )
        if _return_attn:
            r, _attn_weights = _transformer_out   # [B,K,d], [B,K,h*w]
        else:
            r = _transformer_out
            _attn_weights = None

        # 7.4. Box-anchored deformable scene read (additive, zero-init gated).
        # Placed BEFORE the interaction block so dependency/grounding refine
        # deformable-informed queries. Scene is projected with the
        # rel_transformer's own scene_proj: same d_model space the queries
        # already cross-attend to, and no second projection to learn.
        if hasattr(self, "deformable_read"):
            _scene_d = self.rel_transformer.scene_proj(F_map)     # [B,h,w,d]
            _scene_d = _scene_d.permute(0, 3, 1, 2).contiguous()  # [B,d,h,w]
            _anchors = torch.stack(
                [sub_boxes_k, obj_boxes_k, union_boxes_k, contact_boxes_k],
                dim=2)                                            # [B,K,4,4]
            # Multi-level: hand the read a LIST of levels instead of one map.
            # The single-level path is untouched, so an arm differs by exactly
            # one mechanism. `_taps` is None unless depth levels were asked for.
            _scene_in = _scene_d
            if getattr(self, "ms_scene", None) is not None:
                _scene_in = self.ms_scene(_taps or [], fused_proj=_scene_d)
            r = self.deformable_read(r, _scene_in, _anchors)

        # 7.5. Relation Interaction Block: dependency + grounding refinement
        if hasattr(self, "rel_interaction"):
            scene_flat = F_map.reshape(B, F_map.shape[1] * F_map.shape[2], F_map.shape[3])
            r = self.rel_interaction(r, scene_flat, query_padding_mask=padding_mask,
                                     grid_hw=(F_map.shape[1], F_map.shape[2]))

        # 8. Compose the text-space query and score against the vocabulary.
        # Compositional path (config.compose_query): subject/object visual
        # semantics are projected into text space and summed with the pair
        # context — the query is explicitly compositional while the text side
        # stays frozen/cacheable. Gates start at 0.1 so training begins from
        # the pure pair-context query.
        if self.config.compose_query:
            g = self.compose_gate
            q = (self.vocab_head.proj(r)
                 + g[0] * self.sub_text_proj(v_sub)
                 + g[1] * self.obj_text_proj(v_obj_k))
            if hasattr(self, "tucker_query"):
                # Multiplicative pair term, INSIDE compose_norm so the additive
                # attribution decomposition stays exact (LayerNorm's mean
                # distributes over the summands). Exactly 0 at init (P zeros).
                q = q + self.tucker_query(v_sub, v_obj_k)
            q = self.compose_norm(q)                     # [B, K, text_dim]
        else:
            q = None

        q_spa = (self.spa_proj(torch.cat([r, geo_feat], dim=-1))
                 if self.config.dual_spatial_head else None)  # [B, K, text_dim]

        if q_spa is not None:
            q_sem = q if q is not None else self.vocab_head.proj(r)
            logits = self.vocab_head.score_query_dual(q_sem, q_spa)  # [B, K, V]
        elif q is not None:
            logits = self.vocab_head.score_query(q)      # [B, K, V]
        else:
            logits = self.vocab_head(r)                  # [B, K, V]

        q_fast = (self.fast_norm(self.fast_sub(v_sub) + self.fast_obj(v_obj_k))
                  if self.config.fast_bilinear_head else None)  # [B, K, text_dim]

        out = {
            "logits": logits,
            "sub_idx": sub_idx,
            "obj_idx": obj_idx,
            "valid_mask": valid_mask,
            "pred_labels": pred_labels,
            "pair_features": r,  # [B, K, d_model] — raw transformer output for embedding analysis
        }
        if force_attn and _attn_weights is not None:
            out["attn_weights"] = _attn_weights  # [B, K, h*w] — visualization only
        if rel_logits is not None:
            beta = (self.vocab_head.current_beta()
                    if self.config.beta_relatedness else None)
            if beta is not None:
                # PER-PREDICATE relatedness fusion. The relatedness logit is
                # annotation propensity, which correlates with contact — fusing
                # it with one global weight helps `on` and destroys `in front
                # of` ([[relsgg-relatedness-contact-prior]]). beta is read off
                # the text embedding, so each predicate learns how much
                # pair-existence should inform it.
                #
                # Fusing HERE (not in the evaluator) is what gives beta a
                # gradient: a per-pair constant cancels in the within-pair
                # softmax, but beta_p * rel varies ACROSS predicates and so
                # survives it. `pair_logits` is then deliberately NOT exported —
                # every evaluator adds it itself, and it is already inside
                # `logits`; exporting both would double-count it.
                # clamp: padded slots carry rel_logit = -inf, and beta is a
                # sigmoid that can in principle underflow to exactly 0 —
                # 0 * -inf = NaN. -1e4 still gives sigmoid == 0.0 in fp32, so
                # the score contract's "padded slots score exactly 0" holds.
                logits = logits + beta.view(
                    *(1,) * (logits.dim() - 1), -1) \
                    * rel_logits.clamp(min=-1e4).unsqueeze(-1)
                out["logits"] = logits
                out["beta_fused"] = True
            else:
                # Pair-existence logits: triplet score = sigmoid(rel) * sigmoid(pred)
                out["pair_logits"] = rel_logits
        if q_fast is not None and not self.training:
            # Eval/inference only: the V-wide matmul is wasted work at train
            # time (the fast head trains on the batch-local contrast set).
            # Gated on eval mode, not targets, so probes can pass targets
            # (for GT/swap slot force-include) and still read fast logits.
            out["logits_fast"] = self.vocab_head.score_query(q_fast)

        # 9. Loss (training only)
        if targets is not None:
            if hasattr(self, "batch_infonce"):
                # Batch-local contrastive (GLIP/YOLO-World style): the model
                # never sees a V-wide training signal; ranking over the full
                # vocabulary is an inference-time cosine matmul only.
                # Targets are the slot's FULL multi-hot from raw relations —
                # pred_labels keeps only one predicate per pair, which made
                # a pair's other true predicates negatives of each other.
                # slot_w carries rel_weights (geometric-source downweight).
                V = self.vocab_head.W.shape[0]
                gt_hot, slot_w = build_slot_targets(
                    sub_idx, obj_idx, valid_mask, targets, V)
                has_gt = gt_hot.any(-1) & valid_mask
                hot = gt_hot[has_gt]                          # [M, V]
                w_gt = slot_w[has_gt]                         # [M]
                q_all = q if q is not None else self.vocab_head.proj(r)
                spa_gt = q_spa[has_gt] if q_spa is not None else None
                # current_alpha(): live (differentiable) gate_mlp routing when
                # the trainable gate is installed, baked buffer otherwise.
                alpha = (self.vocab_head.current_alpha()
                         if q_spa is not None else None)
                # Per-slot (sub_cat, obj_cat) for cooc hard/soft negatives;
                # -1 where entity labels are missing (treated as all-soft).
                pair_cats = None
                if getattr(self.batch_infonce, "cooc_bits", None) is not None:
                    el_pad = boxes.new_full((B, max_N), -1, dtype=torch.long)
                    for b, t in enumerate(targets):
                        el = t.get("entity_labels")
                        if el is not None and el.numel():
                            n = min(el.shape[0], max_N)
                            el_pad[b, :n] = el[:n].to(el_pad.device)
                    pair_cats = torch.stack(
                        [el_pad[B_idx, sub_idx], el_pad[B_idx, obj_idx]],
                        dim=-1)[has_gt]                       # [M, 2]
                # SOURCE-AWARE NEGATIVE MASKING (--restrict_neg_sources): per-
                # source [n_src, V] allow matrix set by train.py; anchors from a
                # restricted source are contrasted only against that source's
                # own vocabulary. Off (None) unless the flag is set — every
                # other run is byte-identical. Plain attribute, not a buffer:
                # nothing to checkpoint, nothing for EMA/deploy to carry.
                col_allow = None
                src_img = None
                _allow = getattr(self, "source_col_allow", None)
                if _allow is not None and all("src" in t for t in targets):
                    _allow = _allow.to(boxes.device)
                    src_img = torch.as_tensor(
                        [int(t["src"]) for t in targets], device=boxes.device)
                    col_allow = _allow[src_img[B_idx][has_gt]]   # [M, V] bool
                nce_loss = self.batch_infonce(
                    q_all[has_gt], hot, self.vocab_head.W,
                    feats_spa=spa_gt, alpha=alpha, weights=w_gt,
                    pair_cats=pair_cats, col_allow=col_allow,
                )
                fast_loss = (self.batch_infonce(
                                 q_fast[has_gt], hot, self.vocab_head.W,
                                 weights=w_gt, col_allow=col_allow)
                             if q_fast is not None
                             else nce_loss.new_zeros(()))
                obj_loss = (self._object_text_loss(v_obj, box_counts, targets)
                            if self.config.compose_query
                            else nce_loss.new_zeros(()))
                swap_loss = nce_loss.new_zeros(())
                if self.config.lambda_swap > 0:
                    # v3.2: force score(s,o,g) > score(o,s,g) + margin across
                    # the two force-included directed slots — the constraint
                    # the batch-local InfoNCE lacks (probe: SwapAcc ~0.5,
                    # InvTop 0.9). Applied to every trained head.
                    heads = [(q_all, q_spa)]
                    if q_fast is not None:
                        heads.append((q_fast, None))
                    _sym = getattr(self.ontology, "sym", None)
                    swap_loss = swap_direction_hinge(
                        heads, alpha, self.vocab_head.W,
                        sub_idx, obj_idx, valid_mask, targets,
                        self.batch_infonce.inverse_mask,
                        margin=self.config.swap_margin,
                        sym=(_sym.to(self.vocab_head.W.device)
                             if _sym is not None else None),
                    )
                # PER-CELL SIGMOID AUXILIARY (SPML / SigLIP-flavoured).
                # InfoNCE normalises over predicates WITHIN a pair, so it never
                # trains cross-pair comparability — yet that is exactly what
                # per-predicate AUC and any true/false judgment need, and it is
                # where we sit near chance ([[relsgg-spatialsense-probe]]).
                # Removing the partition also stops synonyms competing for one
                # unit of mass, the A3 failure mode.
                #
                # Negatives are balanced PER SLOT rather than by a chosen
                # constant: each slot's negative columns share exactly the mass
                # of its positives, so the objective is self-normalising and no
                # weak-negative hyperparameter is introduced
                # ([[no-handset-cosine-thresholds]]).
                sig_loss = nce_loss.new_zeros(())
                if self.config.lambda_sigmoid > 0.0:
                    lg_gt = logits[has_gt]                      # [M, V] fused
                    tgt = hot.to(lg_gt.dtype)
                    pos_mass = tgt.sum(-1, keepdim=True)        # [M, 1]
                    # Restricted anchors: only allowed columns count as (and
                    # are weighted as) negatives; self-normalisation intact.
                    allow_f = (col_allow.to(tgt.dtype) if col_allow is not None
                               else torch.ones_like(tgt))
                    n_neg = ((tgt <= 0) & (allow_f > 0)).sum(
                        -1, keepdim=True).clamp(min=1)
                    w = torch.where(tgt > 0, torch.ones_like(tgt),
                                    pos_mass / n_neg * allow_f)
                    bce = F.binary_cross_entropy_with_logits(
                        lg_gt, tgt.clamp(0, 1), weight=w, reduction="none")
                    sig_loss = bce.sum(-1).mean()
                # BACKGROUND SUPPRESSION on valid non-GT slots — see the
                # RelSGGConfig.lambda_bg comment for the full rationale
                # (top-k = tail-safe, PU-weighted per slot, fused logits so
                # the relatedness pathway can absorb most of it). No .any()
                # host sync: an empty slot set reduces to a clean zero.
                bg_loss = nce_loss.new_zeros(())
                if self.config.lambda_bg > 0.0:
                    neg_slots = valid_mask & ~has_gt            # [B, K]
                    lg_neg = logits[neg_slots]                  # [M', V] fused
                    if src_img is not None:
                        # Restricted-source images may only declare their OWN
                        # vocabulary as background on unlabelled pairs.
                        lg_neg = lg_neg.masked_fill(
                            ~_allow[src_img[B_idx][neg_slots]],
                            torch.finfo(lg_neg.dtype).min)
                    if self.config.bg_agg == "lse":
                        # float32: logsumexp over V~19K columns in bf16 loses
                        # the small-logit mass that distinguishes slots.
                        per_slot = F.softplus(
                            torch.logsumexp(lg_neg.float(), dim=-1))
                    else:
                        kk = min(self.config.bg_topk, lg_neg.shape[-1])
                        top = lg_neg.topk(kk, dim=-1).values    # [M', kk]
                        per_slot = F.softplus(top).mean(-1)     # BCE, target 0
                    _floor = getattr(self.sampler, "neg_weight", 0.3)
                    if (el_sampler is not None
                            and getattr(self.sampler, "neg_rate", None) is not None):
                        w_slot = self.sampler._pu_neg_weight(
                            el_sampler[B_idx, sub_idx][neg_slots],
                            el_sampler[B_idx, obj_idx][neg_slots], per_slot)
                    else:
                        w_slot = torch.full_like(per_slot, _floor)
                    bg_loss = ((per_slot * w_slot).sum()
                               / w_slot.sum().clamp(min=1e-6))
                total_loss = (nce_loss
                              + self.config.lambda_fast * fast_loss
                              + self.lambda_obj * obj_loss
                              + self.config.lambda_swap * swap_loss
                              + self.config.lambda_sigmoid * sig_loss
                              + self.config.lambda_bg * bg_loss
                              + self.config.lambda_geo * geo_loss
                              + self.config.lambda_rel * rel_loss)
                out["loss"] = total_loss
                out["loss_dict"] = {
                    "loss_nce": nce_loss.detach(),
                    "loss_obj": obj_loss.detach(),
                    "loss_geo": geo_loss.detach(),
                    "loss_rel": rel_loss.detach(),
                    "loss_sig": sig_loss.detach(),
                    "loss_total": total_loss.detach(),
                }
                if self.config.lambda_bg > 0.0:
                    out["loss_dict"]["loss_bg"] = bg_loss.detach()
                # UN-detached term handles for gradient telemetry (train.py
                # --grad_telemetry): references into the live graph, zero cost
                # when unused. Values are pre-lambda; the engine reports both
                # raw and lambda-scaled norms.
                out["loss_terms"] = {
                    "nce": nce_loss, "fast": fast_loss, "obj": obj_loss,
                    "swap": swap_loss, "sig": sig_loss, "bg": bg_loss,
                    "geo": geo_loss, "rel": rel_loss,
                }
                out["loss_lambdas"] = {
                    "nce": 1.0, "fast": self.config.lambda_fast,
                    "obj": getattr(self, "lambda_obj", 0.0),
                    "swap": self.config.lambda_swap,
                    "sig": self.config.lambda_sigmoid,
                    "bg": self.config.lambda_bg,
                    "geo": self.config.lambda_geo,
                    "rel": self.config.lambda_rel,
                }
                if pair_cats is not None:
                    out["loss_dict"]["soft_neg_frac"] = \
                        self.batch_infonce.last_soft_frac.detach()
                if q_fast is not None:
                    out["loss_dict"]["loss_fast"] = fast_loss.detach()
                if self.config.lambda_swap > 0:
                    out["loss_dict"]["loss_swap"] = swap_loss.detach()
                return out

            # PCSG — zone fingerprints (pre-transformer, backbone-derived):
            # coverage-weighted pooling — deliberately not the learned
            # SoftSpatialPool, so the fingerprint is a function of backbone
            # features only. Computed here (not eagerly) because the
            # batch_infonce path above never uses it.
            v_zone = masked_avg_pool(F_map, union_boxes_k)   # [B, K, d_backbone]
            z = F.normalize(self.zone_proj(v_zone), dim=-1)  # [B, K, t_dim]

            if hasattr(self, "synonym_criterion"):
                # Synonym-aware path (plan D2): masked sigmoid multi-label
                # built from raw relations (keeps ALL predicates of
                # multi-predicate pairs), multi-positive InfoNCE alignment,
                # IZSA with the same masking, CZSC over canonical groups.
                rel_losses = self.synonym_criterion(
                    logits, valid_mask, sub_idx, obj_idx, targets
                )
                has_gt = (pred_labels >= 0) & valid_mask
                labels_gt = pred_labels[has_gt]
                infonce_loss = self.mp_infonce(
                    self.vocab_head.proj(r[has_gt]), self.vocab_head.W, labels_gt
                )
                z_gt = z[has_gt]
                pcsg_losses = {
                    "loss_izsa": self.mp_infonce(z_gt, self.vocab_head.W, labels_gt),
                    "loss_czsc": self.spatial_criterion._czsc(
                        z_gt, self.canon_group_of[labels_gt]
                    ),
                }
            else:
                rel_losses = self.criterion(logits, valid_mask, pred_labels)
                infonce_loss = self.vocab_head.compute_infonce_loss(r, pred_labels, valid_mask)

                # Compute zone masks for AA (Phase 2) only when lambda_aa > 0
                _zone_masks: Optional[torch.Tensor] = None
                if _attn_weights is not None:
                    _h, _w = F_map.shape[1], F_map.shape[2]
                    # Coverage weights of the union box over the patch grid [B, K, h*w]
                    _zone_masks = box_coverage_weights(union_boxes_k, _h, _w)

                pcsg_losses = self.spatial_criterion(
                    z=z,
                    W=self.vocab_head.W,
                    pred_labels=pred_labels,
                    valid_mask=valid_mask,
                    attn_weights=_attn_weights,
                    zone_masks=_zone_masks,
                )

            total_loss = (
                rel_losses["loss"]
                + self.config.lambda_geo * geo_loss
                + self.config.lambda_rel * rel_loss
                + self.config.lambda_infonce * infonce_loss
                + self.config.lambda_izsa * pcsg_losses["loss_izsa"]
                + self.config.lambda_czsc * pcsg_losses["loss_czsc"]
                + self.config.lambda_aa * pcsg_losses.get("loss_aa", z.new_zeros(1).squeeze())
            )
            out["loss"] = total_loss
            out["loss_dict"] = {
                **{k: v for k, v in rel_losses.items() if k != "loss"},
                "loss_geo": geo_loss.detach(),
                "loss_rel": rel_loss.detach(),
                "loss_infonce": infonce_loss.detach(),
                "loss_izsa": pcsg_losses["loss_izsa"].detach(),
                "loss_czsc": pcsg_losses["loss_czsc"].detach(),
                **({"loss_aa": pcsg_losses["loss_aa"].detach()} if "loss_aa" in pcsg_losses else {}),
                "loss_total": total_loss.detach(),
            }

        return out

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def predict(
        self,
        images: torch.Tensor,
        boxes: torch.Tensor,
        box_counts: Optional[torch.Tensor] = None,
        threshold: Optional[float] = None,
        topk_per_pair: Optional[int] = None,
        **region_kwargs,
    ) -> List[List[dict]]:
        """Predict relation triplets for a batch of images.

        Args:
            images:         [B, 3, H, W] float32 in [0, 1].
            boxes:          [B, max_N, 4] normalized cxcywh.
            box_counts:     [B] actual box counts. If None, uses max_N.
            threshold:      Minimum score to include a triplet.
            topk_per_pair:  How many top predicates to return per pair.
        Returns:
            List[List[dict]], one list per image, each dict having:
              ``subject``     (int) index into input boxes,
              ``object``      (int) index into input boxes,
              ``predicate``   (str) predicate name,
              ``predicate_id``(int) predicate index into vocabulary,
              ``score``       (float).
        """
        threshold = threshold if threshold is not None else self.config.predict_threshold
        topk_per_pair = topk_per_pair if topk_per_pair is not None else self.config.predict_topk_per_pair

        # region_kwargs (cov / fill / mode) reach the forward unchanged, so the
        # mode-gated mask path is usable from predict() and not only from the
        # training loop. Without this the deployed API is box-only by omission:
        # a caller with masks has no way to hand them over, and the model would
        # silently score box regions while the caller believed otherwise.
        out = self.forward(images, boxes, box_counts=box_counts, targets=None,
                           **region_kwargs)

        logits = out["logits"]         # [B, K, V]
        sub_idx = out["sub_idx"]       # [B, K]
        obj_idx = out["obj_idx"]       # [B, K]
        valid_mask = out["valid_mask"] # [B, K]

        B = images.shape[0]
        pred_names = self.vocab_head.pred_names
        results: List[List[dict]] = []

        # Sigmoid scoring in synonym/batch modes: softmax over a
        # synonym-sharing 10K vocabulary splits probability mass across every
        # synonym of the same relation, deflating all of them (plan D2).
        use_sigmoid = self.resolve_score_mode() == "sigmoid"
        pair_logits = out.get("pair_logits")

        for b in range(B):
            valid_k = valid_mask[b]  # [K]
            if use_sigmoid:
                # THE shared contract (relsgg/scoring.py) — the same object
                # relsgg/evaluator.py, deploy/pipeline.py and the ONNX host
                # use. Unrelated pairs carry a large negative rel_logit that
                # suppresses every predicate; padded slots have rel_logit=-inf
                # → score exactly 0.
                scores = self.score_contract.scores(
                    logits[b, valid_k].float(),
                    None if pair_logits is None
                    else pair_logits[b, valid_k].float())      # [K', V]
            else:
                scores = logits[b, valid_k].softmax(dim=-1)
                if pair_logits is not None:
                    scores = scores * pair_logits[b, valid_k].sigmoid().unsqueeze(-1)
            sub_b = sub_idx[b, valid_k]
            obj_b = obj_idx[b, valid_k]

            # Rank on the DEVICE, transfer once. The per-pair version of this
            # ran one topk kernel per pair and two host syncs per accepted
            # triplet (~400 syncs/frame at K=128) — on a path that is dispatch
            # bound, that is the whole cost. Same output, including tie order:
            # topk rows flatten row-major (pair, then rank) and the sort is
            # stable, which is exactly what list.sort() gave.
            kk = min(topk_per_pair, scores.shape[-1])
            top_scores, top_preds = scores.topk(kk, dim=-1)      # [K', kk]
            keep = top_scores >= threshold
            pair_of_hit = keep.nonzero(as_tuple=True)[0]         # [M] pair row
            sc, order = top_scores[keep].sort(descending=True, stable=True)
            sc_l = sc.tolist()
            pred_l = top_preds[keep][order].tolist()
            sub_l = sub_b[pair_of_hit[order]].tolist()
            obj_l = obj_b[pair_of_hit[order]].tolist()

            results.append([
                {
                    "subject": s,
                    "object": o,
                    "predicate": pred_names[p] if pred_names else str(p),
                    "predicate_id": p,
                    "score": sc_l[i],
                }
                for i, (s, o, p) in enumerate(zip(sub_l, obj_l, pred_l))
            ])

        return results
