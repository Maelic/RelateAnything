"""Box-anchored deformable scene read (DAB-DETR-flavored, pair-query version).

One additive stage after the RelationTransformer: each pair query predicts
sampling offsets around its four natural anchors (sub / obj / union / contact
box centers), reads the scene feature map at those points with bilinear
grid_sample, and adds the aggregated evidence back to the query through a
zero-init gate.

Design decisions, each tied to a measured finding:
  - ADDITIVE + zero-init gate (LayerScale at 0): training starts bit-exact at
    the ungated baseline, so the arm differs by one mechanism and the gate's
    own gradient decides whether the read is worth using (the beta-arm lesson:
    never let a fresh module shock a running system).
  - Offsets are UNBOUNDED, in units of the anchor's half-extent (DAB-style
    modulation): anchors are an initialization prior, not a constraint —
    measured evidence for `parked on`/`hanging from` lives OUTSIDE the boxes
    (road surface, attachment hook), and clamping would forbid exactly the
    reads that broke the SpatialSense band.
  - Per-query independent (no inter-pair mixing): a padded slot's read can
    only corrupt itself, so the padding-leak class of bug (cross-attn memory
    once included other pairs' padding) is structurally impossible here.
  - Scale floor 0.05: the contact anchor is zero-area for non-overlapping
    pairs; the floor gives every anchor a reachable neighborhood.

V2 (2026-08-02) — three fixes for defects MEASURED on the v1 arm
(runs/analysis/deform_vis/, 200 PSG-val images). All are opt-in; the
defaults reproduce the v1 module bit-exactly.

  `heads > 1` — MULTI-HEAD SAMPLING. v1 read 16 locations x 512 channels.
    Splitting the channels into H heads with their own points gives H x more
    locations at IDENTICAL memory traffic (H*A*P locations x d/H channels =
    A*P x d channel-samples either way) and lets heads specialize. Standard
    deformable-DETR layout, which we did not have.

  `ring_init` — SYMMETRY BREAKING. v1 zero-initialized both the weight AND
    bias of offset_mlp, so all P points of an anchor started at the SAME
    position with equal attention weight; identical points receive identical
    gradients, so they could only differentiate through numerical noise —
    the measured cause of v1's weak per-point specialization (only the union
    anchor developed query-dependence at all). Deformable-DETR's remedy:
    initialize each (head, point) at a distinct angle/radius. Radius grows
    with the point index (0.35, 0.70, 1.05, ... half-extents) so every anchor
    starts with both interior and just-exterior probes.

  `gain_gate` + `border_pad` — REMOVE THE NULL-READ HACK. Softmax weights sum
    to 1, so the read's MAGNITUDE cannot vary per pair, while gamma is a
    single global vector; v1 therefore had no per-pair gain control at all.
    It invented one: fling the union points 3.7-4.0 half-extents away, off
    the image, where zeros-padding returns zero vectors, and park 4-8% of
    the softmax mass there (more for spatial pairs) to attenuate the read.
    That spends 4 of 16 points on a volume knob. The fix is to supply the
    knob honestly (a per-pair sigmoid scalar) and to stop paying for it in
    free zeros (border padding returns real edge features off-image).

V3 (2026-08-03) — the principled fix for the "vanishing union point".

MEASURED PROBLEM (v1, runs/analysis/deform_vis/): the union points fly 3.7-4.0
half-extents away, 97-98% of them OUTSIDE the anchor and visibly off-IMAGE,
where zeros-padding returns zero vectors. The model parks 4-8% of the softmax
mass there (more for spatial pairs) to ATTENUATE the read. It is not a bug —
it is the only per-pair gain control the module has, since softmax weights sum
to 1 and gamma is a single global vector. But it costs 4 of 16 points.

WHY V2's FIX BACKFIRED (OVS-F1 0.3316 vs 0.3394, A6 -6.8%): border padding
removed the free zeros, so off-image samples started returning REAL EDGE
PIXELS — i.e. it replaced a clean null with NOISE — while the gain gate only
offered one scalar for the whole read, coarser than the per-anchor attenuation
v1 had discovered.

V3 gives the model what it actually wants, cleanly:
  `null_slots > 0` — extra learnable slots that compete in the SAME softmax but
    are not read from the image. Zero-initialised, so a null slot returns
    exactly the zero vector: identical attenuation to v1's off-image trick,
    at zero cost in sampling points, with no noise. They are Parameters, so
    the model can also learn a non-zero "default context" if that is better.
    Per (head, anchor) granularity matches the mechanism v1 discovered.
    Their softmax logits start at `null_logit_bias` = -2.0, which reproduces
    the 4-8% null mass v1 CONVERGED to; a 0 bias would instead hand them
    S/(P+S) = 33% of the read at init.
  `clamp_to_image` — sampled positions clamped to [0,1]. Once attenuation has
    its own slot, leaving the frame has no purpose, and every point stays on
    real pixels. NOTE this clamps to the IMAGE, not to the anchor box: reading
    outside the box (road surface for `parked on`, the hook above a towel) is
    exactly the evidence-seeking behaviour that broke the SpatialSense band,
    and it remains fully available.

Masks slot in later by swapping the anchor list for mask-derived reference
points (centroid + principal axes) — the module consumes (center, scale)
pairs and never needs to know their source.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLevelScene(nn.Module):
    """Builds the L feature levels the deformable read samples from.

    WHY THE LEVELS ARE MOSTLY A *DEPTH* AXIS, NOT A SCALE AXIS.
    Deformable-DETR / ViT-Adapter get their levels from a hierarchical backbone
    (strides 8/16/32 with genuinely different spatial sampling). A plain ViT/16
    has no such hierarchy: EVERY hidden state is stride 16 (28x28 at 448), so a
    "stride-8 level" synthesized from ViT tokens carries no information the
    stride-16 map lacks. Worse, a BILINEAR stride-8 level would be exactly
    redundant — grid_sample already bilinearly interpolates, so sampling an
    upsampled map at position p returns (up to edge effects) what sampling the
    original at p returns. ViTDet's deconv pyramid works for detection because
    it widens the HEAD's receptive field, not because it recovers detail.

    What IS a real information axis for us is DEPTH, and we measured that it is
    currently being thrown away: the fused map's combiner looks uniform
    (.358/.330/.312) but tap norms are 52.9/192.4/668.0, so state 12 supplies
    ~72% of the fused magnitude. Handing the reader the taps as SEPARATE
    levels lets a pair query choose a depth per sampled point instead of
    inheriting that imbalance.

    Construction, and the cost of each piece (d=512, 28x28 grid at 448):
      depth levels  one per selected backbone tap. Each is LayerNorm'd
                    (parameter-free) and then projected 768->512 by its OWN
                    1x1. The LayerNorm is the `norm_taps` rationale applied
                    where it belongs: without it each projection would have to
                    spend capacity undoing a 13x scale gap between tap 7 and
                    tap 12 before it could encode content. Separate (not
                    shared) projections because a matrix fitted to the
                    late-dominated fused map is the wrong basis for an early
                    tap — this mirrors the ConvNeXt branch, which already gives
                    every stage its own 1x1 for exactly this reason.
                    Cost: 0.39M params and ~0.62 GFLOPs per level.
      pool level    average-pool 2x2 of the DEEPEST projected level -> stride
                    32. Genuine context aggregation, ZERO parameters, and
                    pooling after projection rather than before makes it free
                    rather than another 0.39M.
      deconv level  ViTDet-style 2x2 stride-2 transposed conv on the deepest
                    projected level -> stride 8. Kept because ViTDet measures
                    it helps detection heads, but labelled honestly: it is
                    learned SHARPENING, not recovered detail. 1.05M params,
                    ~1.64 GFLOPs. Off by default.

    Every level gets a learnable level embedding added (Deformable DETR: a
    randomly-initialised, jointly-trained scale-level embedding is what lets a
    query tell which level a sample came from, since the sampled vectors
    otherwise arrive indistinguishable).

    NOTE ON READ COST: the deformable read is SPARSE, so level map SIZE costs
    nothing at read time — only the construction above is paid. This is why
    multi-level is ~2% here while ViT-Adapter (dense cross-attention against
    its pyramid) pays ~10%.
    """

    def __init__(
        self,
        backbone_dim: int,
        d_model: int,
        n_depth_levels: int = 0,
        pool_level: bool = False,
        deconv_level: bool = False,
    ):
        super().__init__()
        if n_depth_levels <= 0 and not (pool_level or deconv_level):
            raise ValueError("MultiLevelScene needs at least one level")
        self.n_depth_levels = n_depth_levels
        self.pool_level = pool_level
        self.deconv_level = deconv_level

        self.proj = nn.ModuleList(
            [nn.Linear(backbone_dim, d_model) for _ in range(n_depth_levels)])
        for p in self.proj:
            nn.init.xavier_uniform_(p.weight)
            nn.init.zeros_(p.bias)

        if deconv_level:
            self.deconv = nn.ConvTranspose2d(d_model, d_model,
                                             kernel_size=2, stride=2)
            nn.init.xavier_uniform_(self.deconv.weight)
            nn.init.zeros_(self.deconv.bias)

        self.n_levels = n_depth_levels + int(pool_level) + int(deconv_level)
        # Deformable-DETR initialises the scale-level embedding randomly (it is
        # an identity tag, not a prior); std 0.02 matches the other learnable
        # tokens in this codebase (base_query, register-style params).
        self.level_embed = nn.Parameter(torch.empty(self.n_levels, d_model))
        nn.init.normal_(self.level_embed, std=0.02)

    def forward(
        self,
        taps: list[torch.Tensor],        # each [B, h, w, d_backbone]
        fused_proj: torch.Tensor | None = None,   # [B, d, h, w], already in d_model
    ) -> list[torch.Tensor]:
        """Returns L tensors [B, d_model, h_l, w_l], ORDERED FINEST-FIRST
        (deconv, then depth levels shallow->deep, then pool)."""
        levels: list[torch.Tensor] = []
        depth: list[torch.Tensor] = []
        for i in range(self.n_depth_levels):
            x = taps[i]
            # LayerNorm over the channel dim, parameter-free: strips the
            # depth-dependent magnitude so the projection encodes content.
            x = F.layer_norm(x, x.shape[-1:])
            x = self.proj[i](x)                       # [B,h,w,d_model]
            depth.append(x.permute(0, 3, 1, 2).contiguous())

        # Fall back to the already-projected fused map when no depth level was
        # requested (deconv/pool-only configurations, i.e. the ViTDet arm).
        base = depth[-1] if depth else fused_proj
        if base is None:
            raise ValueError("no depth levels and no fused_proj supplied")

        if self.deconv_level:
            levels.append(self.deconv(base))          # stride 8
        levels.extend(depth)                          # stride 16
        if self.pool_level:
            levels.append(F.avg_pool2d(base, kernel_size=2))  # stride 32

        return [lv + self.level_embed[i].view(1, -1, 1, 1)
                for i, lv in enumerate(levels)]


class DeformableRelRead(nn.Module):
    """Args:
        d_model:    pair-query / projected-scene dimension.
        n_points:   sampled points per anchor PER HEAD (A=4 anchors fixed).
        heads:      sampling heads; each reads d_model/heads channels at its
                    own points. 1 = v1 behavior.
        ring_init:  distinct angle/radius per (head, point) instead of all
                    points at the anchor center. Breaks the init symmetry.
        gain_gate:  per-pair sigmoid scalar on the read (legitimate magnitude
                    control, replacing the off-image null trick).
        border_pad: grid_sample padding_mode="border" instead of "zeros", so
                    off-image samples cannot return free zero vectors.
        null_slots: V3. Learnable non-image slots per (head, anchor) competing
                    in the softmax; zero-init => exact zero-vector reads, i.e.
                    clean per-pair attenuation without spending sample points.
        clamp_to_image: V3. Clamp sampled positions to [0,1] so no point can
                    leave the frame (harmless once null_slots exist; reading
                    outside the BOX is still unrestricted).
    """

    N_ANCHORS = 4  # sub, obj, union, contact

    def __init__(
        self,
        d_model: int,
        n_points: int = 4,
        heads: int = 1,
        ring_init: bool = False,
        gain_gate: bool = False,
        border_pad: bool = False,
        null_slots: int = 0,
        clamp_to_image: bool = False,
        null_logit_bias: float = -2.0,
        n_levels: int = 1,
    ):
        super().__init__()
        if d_model % heads:
            raise ValueError(f"d_model {d_model} not divisible by heads {heads}")
        self.n_points = n_points
        self.heads = heads
        self.border_pad = border_pad
        self.null_slots = null_slots
        self.clamp_to_image = clamp_to_image
        # n_levels > 1: each (head, anchor, point) is replicated across L
        # feature levels (see MultiLevelScene). Sampling POSITIONS are shared
        # across levels — offsets are expressed in anchor half-extent units, a
        # resolution-independent frame, so "the same place at a different
        # level" is exactly what we want and matches Deformable DETR, which
        # also initialises every level with the same offset pattern.
        self.n_levels = n_levels
        # Diagnostics: when .capture is True, forward() stashes the sampled
        # positions / weights / anchors of its last call (visualization only;
        # never set during training). Same pattern as force_attn elsewhere.
        self.capture = False
        self.last_pos: torch.Tensor | None = None      # [B,K,H,A,L,P,2] in [0,1]
        self.last_w: torch.Tensor | None = None        # [B,K,H,A*L*P] image mass
        self.last_null_w: torch.Tensor | None = None   # [B,K,H,A] null mass
        self.last_anchors: torch.Tensor | None = None  # [B,K,A,4] cxcywh
        self.last_level_w: torch.Tensor | None = None  # [B,K,H,A,L] per-level mass
        A, P, H, L = self.N_ANCHORS, n_points, heads, n_levels

        self.norm = nn.LayerNorm(d_model)
        self.offset_mlp = nn.Linear(d_model, H * A * L * P * 2)
        # +null_slots logits per (head, anchor): they compete in the same
        # softmax as the sampled points but are read from a Parameter, not
        # from the image. The softmax spans anchors AND levels AND points
        # jointly, matching Deformable DETR's constraint (sum over levels and
        # points = 1) — at L=1 this is bit-identical to the pre-multi-level
        # module, so a multi-level arm differs by exactly one mechanism.
        self.weight_mlp = nn.Linear(d_model, H * A * (L * P + null_slots))
        if null_slots:
            self.null_vec = nn.Parameter(torch.zeros(H, A, null_slots, d_model // H))
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.offset_mlp.weight)
        nn.init.zeros_(self.weight_mlp.weight)
        nn.init.zeros_(self.weight_mlp.bias)
        if n_levels > 1:
            # Zero logits across levels => every level starts with equal mass
            # (1/L of each anchor's budget). Deformable DETR does the same
            # (attention weights initialised uniform at 1/(LK)); an unequal
            # start would pre-decide the level question we are running the arm
            # to answer. Level collapse is the known failure mode, so
            # last_level_w is captured for instrumentation.
            pass
        if null_slots:
            # All-zero logits would hand the nulls S/(P+S) of every anchor's
            # mass — 33% of the read zeroed at init for S=2,P=4. The v1 model
            # CONVERGED to 4-8% null mass, so start there instead: a -2.0 logit
            # puts nulls at exp(-2)=0.135 relative weight, i.e. 6.3% for
            # S=2,P=4. The prior comes from the measurement, not from taste.
            #
            # LEVEL CORRECTION (+ln L): the null slots compete against L*P image
            # samples, not P, so a fixed -2.0 would silently dilute the null
            # mass by ~L (6.3% -> 1.7% at L=4) and the multi-level arm would
            # differ from the baseline in TWO ways — levels AND attenuation
            # strength. Adding ln(L) makes the init null fraction exactly
            # invariant to L, so the arm isolates the level mechanism.
            _bias = null_logit_bias + math.log(L)
            with torch.no_grad():
                b = self.weight_mlp.bias.view(H, A, L * P + null_slots)
                b[..., L * P:] = _bias
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.gate_lin = nn.Linear(d_model, 1) if gain_gate else None
        if self.gate_lin is not None:
            nn.init.zeros_(self.gate_lin.weight)
            nn.init.zeros_(self.gate_lin.bias)   # sigmoid(0) = 0.5 at init
        self.gamma = nn.Parameter(torch.zeros(d_model))

        if ring_init:
            # Distinct (angle, radius) per (head, point); anchors share the
            # pattern because they already sit at different image locations.
            # Each head owns an angular SECTOR (deformable-DETR gives heads
            # different directions) and fans its P points across that sector
            # while the radius grows — deformable-DETR puts all of a head's
            # points on one ray, which leaves them collinear and correlated.
            # LEVELS also share the pattern: offsets are in half-extent units,
            # so the same (angle, radius) is the same PLACE at every level, and
            # the levels differ in which representation is read there — which
            # is the whole point of the level axis. (Deformable DETR likewise
            # replicates its init grid across levels.)
            bias = torch.zeros(H, A, L, P, 2)
            for h in range(H):
                for p in range(P):
                    theta = 2.0 * math.pi * (h + p / max(P, 1)) / H
                    r = 0.35 * (p + 1)
                    bias[h, :, :, p, 0] = math.cos(theta) * r
                    bias[h, :, :, p, 1] = math.sin(theta) * r
            with torch.no_grad():
                self.offset_mlp.bias.copy_(bias.reshape(-1))
        else:
            nn.init.zeros_(self.offset_mlp.bias)

    def forward(
        self,
        queries: torch.Tensor,        # [B, K, d]
        scene: "torch.Tensor | list[torch.Tensor]",  # [B,d,h,w] or L of those
        anchors_cxcywh: torch.Tensor, # [B, K, A, 4]  normalized [0, 1]
    ) -> torch.Tensor:
        B, K, d = queries.shape
        A, P, H, L = self.N_ANCHORS, self.n_points, self.heads, self.n_levels
        dh = d // H
        levels = [scene] if torch.is_tensor(scene) else list(scene)
        if len(levels) != L:
            raise ValueError(f"expected {L} feature levels, got {len(levels)}")

        S = self.null_slots
        qn = self.norm(queries)
        off = self.offset_mlp(qn).view(B, K, H, A, L, P, 2)
        centers = anchors_cxcywh[..., :2].view(B, K, 1, A, 1, 1, 2)
        half = (anchors_cxcywh[..., 2:].clamp_min(0.05)
                * 0.5).view(B, K, 1, A, 1, 1, 2)
        pos = centers + off * half                              # [B,K,H,A,L,P,2]
        if self.clamp_to_image:
            pos = pos.clamp(0.0, 1.0)

        # Per-head sampling: split the channels, give each head its own grid.
        # One grid_sample per LEVEL (the maps have different h,w so they cannot
        # share a call); each returns [B*H, dh, K, A*P] and they are stacked on
        # a level axis. Sampling cost is O(points), independent of map size —
        # which is why adding levels is cheap here (see MultiLevelScene).
        per_level = []
        for li, lv in enumerate(levels):
            lh, lw = lv.shape[-2:]
            scene_h = lv.view(B, H, dh, lh, lw).reshape(B * H, dh, lh, lw)
            # Slice (not int-index) the level axis: torch.onnx.export lowers
            # `pos[..., li]` to Gather(axis=4), which the OpenVINO Intel GPU
            # plugin refuses ("Unsupported gather axis: 4"). Slice+squeeze
            # traces to Slice+Squeeze instead, which the GPU plugin accepts,
            # and is numerically identical (n_levels=1 in deployed configs,
            # so this loop runs once either way).
            grid = (pos[:, :, :, :, li:li + 1].squeeze(4)
                    .permute(0, 2, 1, 3, 4, 5)
                    .reshape(B * H, K, A * P, 2) * 2.0 - 1.0)      # x,y order
            per_level.append(F.grid_sample(
                scene_h, grid, mode="bilinear", align_corners=False,
                padding_mode="border" if self.border_pad else "zeros",
            ).view(B, H, dh, K, A, P))
        # [B,H,dh,K,A,L,P] — level axis sits INSIDE the anchor, so the softmax
        # layout below stays (anchor-major, then level, then point).
        vals = torch.stack(per_level, dim=5)

        w = F.softmax(self.weight_mlp(qn).view(B, K, H, A * (L * P + S)), dim=-1)
        if self.capture:
            self.last_pos = pos.detach()
            # report only the IMAGE-sampled mass so visualisations stay
            # comparable across v1/v2/v3; null mass is exposed separately.
            _wv = w.view(B, K, H, A, L * P + S)
            self.last_w = _wv[..., :L * P].reshape(B, K, H, A * L * P).detach()
            self.last_null_w = (_wv[..., L * P:].sum(-1).detach() if S else None)
            self.last_anchors = anchors_cxcywh.detach()
            # per-level mass, the instrument for the known failure mode
            # (level collapse — all mass on one level).
            self.last_level_w = (_wv[..., :L * P]
                                 .view(B, K, H, A, L, P).sum(-1).detach())
        vals = vals.reshape(B, H, dh, K, A, L * P)
        if S:
            # per-anchor, append the null slots after that anchor's L*P samples
            nulls = (self.null_vec.view(1, H, A, S, dh)
                     .permute(0, 1, 4, 2, 3)               # [1,H,dh,A,S]
                     .unsqueeze(3).expand(B, H, dh, K, A, S))
            vals = torch.cat([vals, nulls], dim=-1)
        vals = vals.reshape(B * H, dh, K, A * (L * P + S))
        wh = w.permute(0, 2, 1, 3).reshape(B * H, K, A * (L * P + S))
        read = torch.einsum("bdkp,bkp->bkd", vals, wh)             # [B*H,K,dh]
        read = read.view(B, H, K, dh).permute(0, 2, 1, 3).reshape(B, K, d)

        out = self.out_proj(read)
        if self.gate_lin is not None:
            out = out * torch.sigmoid(self.gate_lin(qn))           # [B,K,1]
        return queries + self.gamma * out
