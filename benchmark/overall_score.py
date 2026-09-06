"""OVS — one balance-enforcing scalar over the OV-SGG axes.

WHY THIS EXISTS, GIVEN SPEC.md ONCE EXCLUDED IT
------------------------------------------------
SPEC.md rejected "a single aggregate score" because averaging raw numbers across
sources with different vocabularies invites the corpus-match gaming the benchmark
exists to expose. That objection is against an ARITHMETIC MEAN OF RAW METRICS,
and it stands. This is a different object, built so the objection does not apply:

  1. Every cell is CHANCE-CORRECTED before it is combined, so a metric whose
     floor is 0.5 (SpatialSense AUC) cannot outweigh one whose floor is 1/V
     (recall over a V-class vocabulary). Raw averages get this catastrophically
     wrong: AUC 0.66 and mR@50 0.22 are not "0.44 on average", they are
     0.32 and 0.20 above their respective floors.
  2. Axes are combined with a HARMONIC mean, which is minimised by imbalance.
     Being excellent on one axis and useless on another cannot produce a good
     score — the precedent is generalised zero-shot learning, where the harmonic
     mean of seen/unseen accuracy replaced the arithmetic mean precisely because
     the latter rewarded models that ignored unseen classes.
  3. It NEVER replaces the vector. `aggregate.py` remains the headline; this is a
     summary of it, and the per-axis components are printed with every score.

So corpus match now BUYS LESS, not more: a model that matches VG150's annotation
style gains on one A1 cell out of four and nothing on A2/A4/A6, and the harmonic
mean drags it back toward its weakest axis.

DEFINITION
----------
Per cell:  norm = clip((x - chance) / (1 - chance), 0, 1)
Per axis:  arithmetic mean of its cells (same capability, different sources)
Overall:   OVS = harmonic mean of the COMPOSITE axes (A1, A2, A4, A6)

A3 is measured, reported, and out of the composite: the baseline cannot be run
on it at all -- its predicate vocabulary is a caption capped at 512 word pieces
-- so a composite containing A3 exists for one of the two models being compared
and the head-to-head cell is a dash.

A5 costs one judge run per arm and has been run for the released tower only, so
the ladder in release_gate.py is scored WITHOUT it and prints its axis set. Do
not compare an OVS over four axes with an OVS over five.
Reported beside it: OVS_arith (arithmetic mean of axes), the WEAKEST axis, and
`balance` = OVS / OVS_arith in (0, 1] — 1.0 exactly when all axes are equal, so
it reads directly as "how specialised is this model".

CHANCE LEVELS (derived, never hand-set — cf. [[no-handset-cosine-thresholds]])
  A1 recall     1/V, V = the benchmark's own vocabulary size (a uniform-random
                predicate under the graph constraint)
  A2 fAP        the dataset's positive prevalence (AP of a random scorer)
  A3 open-vocab 1/|deployed vocab| ~ 5e-5, taken as 0   (reported, not composed)
  A4 wR@50      1/V of the source, after dividing by the measured pair-recall
                ceiling of the shared detector
  A5 bits       0, and the cell is already a share of the annotation's own
                information, so no further correction applies
  A6 AUC        0.5

CAVEAT THAT MUST TRAVEL WITH THE NUMBER: OVS is only comparable between models
scored on the SAME AXIS SET. Adding an axis changes every score. The axis set is
printed in the output and stored in the json.

    python benchmark/overall_score.py --out runs/benchmark/ovs.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ARMS = [
    ("proxy50k_v41cfg", "P v41"),
    ("proxy50k_v42cfg", "P v42"),
    ("proxy50k_v42cfg_wv2", "P wv2"),
    ("proxy50k_v42cfg_wv2_sw", "P wv2+sw"),
    ("proxy50k_v42cfg_singlehead_wv2", "P wv2+1head"),
    ("proxy50k_v42cfg_wv2_lora12", "P wv2+lora12"),
    ("proxy50k_v42cfg_singlehead_wv2_lora12", "P wv2+1h+l12"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg", "P wv2+l12+mix"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25", "P +sig0.25"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig1.0", "P +sig1.0"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_beta", "P +beta"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_lr4e-4_ep12", "P 4e-4/12ep"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_r0_lr4e-4_ep12", "P ft5e-5 12ep"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_r0_lr4e-4_ep12",
     "P ALLFIXES"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_btd0.3_r0_lr4e-4_ep12", "P btd-only"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.1_btd0.3_r0_lr4e-4_ep12",
     "P sig0.1+btd"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_llrd0.7_bwd0.0_dp0.1"
     "_dsi_lsi10.6_lbi-0.7_r0_lr4e-4_ep12", "P ARM-A init"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_def4_r0_lr4e-4_ep12",
     "P +deform4"),
    ("sched_lr4e-4_ep8_r0_sig0.25_btd0.3_def4", "FULL RECIPE v1"),
    ("proxy50k_v42cfg_wv2_lora12_mixvgoi_sig0.25_btd0.3_def4_r0_lr4e-4_ep12",
     "P +OI 25%"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_def4h8v2_r0_lr4e-4_ep12",
     "P deform v2"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_r0_lr4e-4_ep12_wise0.9",
     "P ALLFX w0.90"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_r0_lr4e-4_ep12_wise0.8",
     "P ALLFX w0.80"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_r0_lr4e-4_ep12_wise0.65",
     "P ALLFX w0.65"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_r0_lr4e-4_ep12_wise0.5",
     "P ALLFX w0.50"),
    ("sched_lr4e-4_ep8", "FULL 4e-4/8ep"),
    ("sched_lr2e-4_ep5_r0_blr1e-5", "FULL ft blr1e-5"),
    ("sched_lr2e-4_ep8_r0_blr1e-5", "FULL ft 8ep b1e-5"),
    ("sched_lr2e-4_ep8_r0_blr5e-5", "FULL ft 8ep b5e-5"),
    ("sched_lr2e-4_ep8", "FULL LoRA 8ep"),
    ("v41_lout_lora_5ep", "FULL v41"),
    ("v42_softsup_lora_5ep", "FULL v42"),
    ("v43_full_5ep", "FULL v43"),
    ("v44_full_5ep", "FULL v44"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_r0_lr4e-4_ep12", "P v3 def4h8"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_d768_r0_lr4e-4_ep12", "P v3 h8 d768"),
    ("proxy50k_v42cfg_wv2_lora12_mixvg_sig0.25_btd0.3_def8h8v3n2_r0_lr4e-4_ep12", "P v3 def8h8"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_r0_lr4e-4_ep12", "P W512"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05", "P W512+FIX"),
    # ntaps lineage, proxy pack, 12 epochs — the multi-scale confirmation pair.
    # Hardware-matched to each other (both A100:2); this is the pair the
    # multi-scale decision rests on.
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_r0_lr4e-4"
     "_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_full12", "P ntaps ctl12"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_r0_lr4e-4"
     "_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5_full12", "P ntaps+ms12"),
    # E3 bg_agg pair — the ONLY difference between these two rows is the
    # background-penalty aggregate (top-5 vs logsumexp over the 19K columns).
    # 6 epochs, so they are comparable to EACH OTHER and to nothing else in this
    # table: every other proxy row is 12 epochs.
    ("e3_bgagg_topk_ep6", "E3 bg topk6"),
    ("e3_bgagg_lse_ep6", "E3 bg lse6"),
    # FULL pack (503,754 img/ep), 12 epochs, current recipe + multi-scale, on the
    # two small DINOv3 towers. These differ from each other in backbone_model and
    # NOTHING else. They are NOT recipe-matched to any ViT-B row above: the only
    # full-pack ViT-B point is "FULL RECIPE v1" (8 epochs, pre-W512 recipe), so
    # every S-vs-B read here is confounded by recipe AND schedule, not just width.
    ("full_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_vits16_r0"
     "_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5", "F ViT-S/16"),
    ("full_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_vits16plus_r0"
     "_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5", "F ViT-S/16+"),
    # The ViT-B arm of the SAME three-way. vitb16 is the lineage default and so
    # takes no directory suffix, which is why this name is the bare recipe.
    # Recipe-, data- and schedule-matched to the two rows above; the only
    # unavoidable difference is A100fat vs A100 (the 0.5-1.5 ladder peaks
    # ~39.7 GB and OOMs a 40 GB card), same compute die.
    ("full_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_r0"
     "_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5", "F ViT-B/16"),
    # ConvNeXt-T fusion 3-way, PROXY pack, blr 1e-5, 12 ep, A40:2 (jobs
    # 6927332/34/36). RANKING ONLY — proxy levels are not reportable, and these
    # are A40 while every ViT proxy arm above is A100, so do NOT read them
    # against the ViT rows. Compare the three to EACH OTHER.
    # CAVEAT: in these runs layer_weights and stage_norm were starved at
    # backbone_lr, so the LEVEL AXIS is unmeasured; the normalization itself was
    # active (LayerNorm affine inits to identity) so the recall contrast stands
    # ([[relsgg-convnext-fusion-flaw]]).
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_convnext_tiny_r0_blr1e-5_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ms0.5-1.5", "CNX ctl"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_convnext_tiny_r0_blr1e-5_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_ms0.5-1.5", "CNX norm"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_convnext_tiny_r0_blr1e-5_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_s2d_ms0.5-1.5", "CNX norm+s2d"),
    # ---- The FAMILY COMPARISON bracket: 3 towers x 3 backbone_lr, PROXY pack,
    # 12 ep, A40:2 x bs32 x ACCUM=2 = global batch 128. All nine are recipe-,
    # data-, schedule- AND hardware-matched to each other, which is what makes
    # this the first legitimate ConvNeXt-vs-ViT read: every earlier ConvNeXt row
    # above sat at an inherited LR with a starved fusion combiner.
    # These are the first ConvNeXt arms where layer_weights / stage_norm /
    # stage_proj get args.lr instead of backbone_lr, so the LEVEL AXIS is
    # measured here and nowhere above ([[relsgg-convnext-fusion-flaw]]).
    # PROXY LEVELS ARE NOT REPORTABLE — ranking only.
    # NOTE the 5e-5 ViT arms carry NO blr suffix: 5e-5 is the runner default.
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_convnext_tiny_r0_blr3e-6_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_s2d_ms0.5-1.5_flr", "CNX-T blr3e-6"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_convnext_tiny_r0_blr1e-5_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_s2d_ms0.5-1.5_flr", "CNX-T blr1e-5"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_convnext_tiny_r0_blr3e-5_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_s2d_ms0.5-1.5_flr", "CNX-T blr3e-5"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_vits16_r0_blr1e-5_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_ms0.5-1.5", "ViT-S blr1e-5"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_vits16_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_ms0.5-1.5", "ViT-S blr5e-5"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_vits16_r0_blr1e-4_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_ms0.5-1.5", "ViT-S blr1e-4"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_vits16plus_r0_blr1e-5_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_ms0.5-1.5", "ViT-S+ blr1e-5"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_vits16plus_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_ms0.5-1.5", "ViT-S+ blr5e-5"),
    ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
     "_vits16plus_r0_blr1e-4_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05"
     "_ntaps_ms0.5-1.5", "ViT-S+ blr1e-4"),
]

# ---- EVAL-RESOLUTION SWEEP over the three full-scale towers, and the
# EARLY-STOPPING cells. Each is a SUBDIRECTORY of its tower's run dir written by
# job_res_sweep.sh, which is why these arm strings contain a slash — the arm is
# joined onto runs/train/ verbatim, so a subdirectory needs no support code.
# The 448px rows are the canonical "F ViT-*" arms above (same checkpoint, same
# resolution) and are NOT duplicated here.
# Resolution is a ZERO-TRAINING lever and 672 is IN-DISTRIBUTION: these towers
# trained on MULTISCALE 0.5-1.5 N=7 = rungs [224 304 368 448 528 592 672].
# 560 is NOT itself a rung (528 and 592 are), so it interpolates between two
# seen scales rather than extrapolating.
_FULL = ("full_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
         "{tower}_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5")
for _tw, _lab in (("_vits16", "ViT-S"), ("_vits16plus", "ViT-S+"), ("", "ViT-B")):
    for _r in (560, 672):
        ARMS.append((f"{_FULL.format(tower=_tw)}/res{_r}", f"{_lab} @{_r}"))
# checkpoint_best = the peak dev(PSG-val) mR@50 epoch: ViT-B ep6, ViT-S+ ep7
# (ViT-S never peaked — its best IS its last, i.e. it is undertrained at 12 ep).
# Scored at 448 so the ONLY difference from the canonical row is the epoch. This
# is the direct test of whether early stopping recovers ViT-B's lost OOD
# transfer, which the in-training dev curve cannot answer
# ([[relsgg-final-not-best-epoch]]).
ARMS.append((f"{_FULL.format(tower='')}/ep6", "ViT-B @ep6"))
ARMS.append((f"{_FULL.format(tower='_vits16plus')}/ep7", "ViT-S+ @ep7"))

# ---- SHIP RUN 2026-08-24 (train 6959720 / eval 6959721): ViT-S+ full recipe +
# HICO-DET TRAIN at relation share 0.10 (MIX=vgraw_hico, per-image
# 0.590/0.051/0.359, ~5.75 HICO passes/epoch) + TUCKER 96x48. TWO variables vs
# the canonical ViT-S+ row, by user decision; the proxy ladder has each alone
# (HICO-sup 0.10 / TUCKER 96x48). HICO cells are HOI-SUPERVISED for this arm —
# read A1/A2 with them excluded ([[relsgg-hoi-tail-and-loop-pilot]]).
ARMS.append(("full_v42cfg_wv2-512_lora12_mixvghico0.10_sig0.25_btd0.3_def4h8v3n2"
             "_tqk96x48_vits16plus_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps"
             "_ms0.5-1.5", "SHIP S+ hico+tqk"))
# Low-share single-variable arm (train 6959924 / eval 6959925): HICO share 0.05,
# NO Tucker -> 21% HICO images, ~3.4 passes/epoch (vs 36% / 5.75 in the SHIP
# run). Asks whether the A6 tax + HICO memorisation are the image-fraction /
# recycling and how much of the HICO gain survives below the proxy's 0.10.
ARMS.append(("full_v42cfg_wv2-512_lora12_mixvghico0.05_sig0.25_btd0.3_def4h8v3n2"
             "_vits16plus_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps"
             "_ms0.5-1.5", "HICO0.05 S+"))
# Ship candidate v2 (2026-08-26): HICO 0.05 + source-aware negative masking.
# Proxy read: negmask refunded ~60% of HICO's projective-spatial tax (A6 0.374
# -> 0.398 vs ctl 0.414-0.433) with HICO F1 intact (0.326 -> 0.323).
ARMS.append(("full_v42cfg_wv2-512_lora12_mixvghico0.05_sig0.25_btd0.3_negmask_def4h8v3n2_vits16plus_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5", "HICO0.05 negmask S+"))
# ---- RELEASE FAMILY (2026-08-26): the negmask recipe on all three towers, HICO
# train pack V2 (duplicate boxes merged). S+ has a backbone_lr 1e-4 hedge arm.
_REL = ("full_v42cfg_wv2-512_lora12_mixvghico0.05_sig0.25_btd0.3_negmask_def4h8v3n2"
        "{tower}_r0{blr}_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5_hicov2")
ARMS.append((_REL.format(tower="_vits16", blr=""), "REL ViT-S negmask V2"))
ARMS.append((_REL.format(tower="_vits16plus", blr=""), "REL ViT-S+ negmask V2"))
ARMS.append((_REL.format(tower="_vits16plus", blr="_blr1e-4"), "REL ViT-S+ negmask V2 blr1e-4"))
ARMS.append((_REL.format(tower="", blr=""), "REL ViT-B negmask V2"))
# Source-aware negative masking on the HICO-0.10 proxy (control = "HICO-sup
# 0.10"): HICO anchors contrast only against HICO's own 116 verbs. If the
# projective-spatial tax is the silent-negative pressure, A6 returns to the
# REG ctl band (0.41-0.43) with the HICO gain intact.
ARMS.append(("proxy50k_v42cfg_wv2-512_lora12_mixvghico0.10_sig0.25_btd0.3_negmask"
             "_def4h8v3n2_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5_hico",
             "HICO0.10 negmask"))
# Shared-init soups of the SHIP run (weight w) with the canonical ViT-S+ (1-w):
# same seed init, same recipe except mix + Tucker; Tucker P scaled by w. The
# untested variant in [[relsgg-model-soup-negative]]. Asks whether the HICO gain
# and the A6/vg150 tax interpolate (one basin) or the soup collapses.
for _w in ("0.25", "0.5", "0.75"):
    ARMS.append(("full_v42cfg_wv2-512_lora12_mixvghico0.10_sig0.25_btd0.3_def4h8v3n2"
                 "_tqk96x48_vits16plus_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps"
                 f"_ms0.5-1.5_soup{_w}", f"SOUP w={_w}"))

# ---- BACKBONE-REGULARIZATION sweep, single-variable on ViT-B, proxy, A40:2.
# The backbone was the one component with NO regularization (drop_path 0.0,
# LLRD off, WD 1e-4); dropout 0.2 only ever touched the head. Control is run on
# the SAME hardware rather than reusing `P ntaps+ms12` (that arm is A100).
# PROXY OVERSTATES REGULARIZATION: ctl's train-val gap here is +0.52 against
# full-scale ViT-B's +0.16, so read direction and ranking, not magnitude.
_REG = ("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2{knob}"
        "_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5_reg")
for _k, _l in (("", "REG ctl"), ("_dp0.1", "REG dp0.1"), ("_dp0.2", "REG dp0.2"),
               ("_llrd0.8", "REG llrd0.8"), ("_bwd0.01", "REG bwd0.01")):
    ARMS.append((_REG.format(knob=_k), _l))

# ---- TUCKER multiplicative query (job 6947602): MUTAN-style pair term added
# to the composed query, mode-3 factor = the frozen text bank (open vocab kept
# parameter-free), r3=48 ~ the bank's measured effective rank 43.7. Same
# recipe/hardware as REG ctl, which is its control. Motivation and kill
# criterion: [[relsgg-score-attribution]] — the ADDITIVE version of this
# channel carries a 0.0% variance share, so this arm asks whether the model
# declined the signal or only its additive form.
ARMS.append(("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
             "_tqk96x48_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps"
             "_ms0.5-1.5_tqk", "TUCKER 96x48"))
# Seed-43 replicas of BOTH arms (jobs 6952174/6952176) — the 2-seed protocol
# that settled CFA. tucker-vs-ctl must hold at BOTH seeds to be a win; the
# ctl@s43-vs-ctl@s42 delta is the direct read of composite seed noise.
ARMS.append(("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
             "_r0_lr4e-4_ep12_s43_newopt_spe_gsq_pe16_bg0.05_ntaps"
             "_ms0.5-1.5_reg", "REG ctl s43"))
ARMS.append(("proxy50k_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2"
             "_tqk96x48_r0_lr4e-4_ep12_s43_newopt_spe_gsq_pe16_bg0.05_ntaps"
             "_ms0.5-1.5_tqk", "TUCKER s43"))
# ---- HICO-DET TRAIN as a verb-supervision source (jobs 6958854 / 6958856):
# the CEILING for long-tail human actions — what perfect verb supervision buys
# on HICO and costs on every other axis. HICO cells are NOT zero-shot for these
# two arms (report as HOI-supervised). Control = REG ctl. Two relation shares
# give the dose-response; hico passes/epoch 1.12 and 1.46.
for _hs in ("0.10", "0.15"):
    ARMS.append(("proxy50k_v42cfg_wv2-512_lora12_mixvghico" + _hs +
                 "_sig0.25_btd0.3_def4h8v3n2_r0_lr4e-4_ep12_newopt_spe_gsq_pe16"
                 "_bg0.05_ntaps_ms0.5-1.5_hico", "HICO-sup " + _hs))

# A1 sources -> vocabulary size (chance = 1/V under the graph constraint)
A1 = {"vg150": 50, "psg": 56, "indoorvg": 37, "hicodet": 116}
A3 = ["vg150", "psg", "indoorvg"]

# A4 DEPLOYMENT. Detection-mode wR@50 on the shared open-vocabulary detector,
# divided by the MEASURED pair-recall ceiling before it is chance-corrected. The
# raw number is bounded by the detector, which no relation model controls and
# which bounds every model identically; dividing by the ceiling turns it into
# "share of the recoverable pairs recovered", which is the model's part. The
# ceiling is imported rather than repeated -- aggregate.py measured it.
A4_SOURCE = "psg"
A4_FILES = ("detbox/detbox_psg_yoloworld_gc.json",
            "zeroshot_detbox_psg_yoloworld_gc.json")
A4_METRIC = "wR@50"   # support-weighted; the same cell tab:sixaxes reports
# One copy of the ceiling: aggregate.py holds the measured value.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.aggregate import CEILING as _CEILING  # noqa: E402
CEILING = _CEILING[("A4", "psg/test")]

# A5 GRAPH QUALITY. Eq. (4) of the report -- surprisal of the relations a judge
# accepted -- divided by the same quantity on the images' own annotations, at
# matched graph depth. Written by a5b_annotation_reference.py.
#
# This axis was OUT of the composite while it was a pairwise win rate, for a
# reason that no longer holds: a win rate needs an opponent, so the two models'
# values summed to 1 and the cell described the pair rather than either model,
# and a model losing the head-to-head scored 0 and collapsed the harmonic mean.
# The share of the annotation's information is a property of one model, has a
# floor of 0 (a system whose claims the judge rejects earns no bits) and so
# needs no further chance correction, and is computable without a baseline.
A5_FILE = os.path.join("runs", "benchmark", "a5b",
                       "annotation_reference_top10_matched.json")
A5_KEY = {"ours": "RelateAnything-pack", "base": "OvSGTR"}

# The composite spans these five. A3 stays out: the baseline cannot be run on it
# at all (its vocabulary is one caption capped at 512 tokens), so a composite
# containing A3 exists for one of the two models being compared and the
# head-to-head cell is a dash.
COMPOSITE_AXES = ("A1 transfer", "A2 precision", "A4 detector",
                  "A5 graph quality", "A6 spatial")
# A2 prevalence: positives / labelled cells, from the packs' own annotations.
A2_PREVALENCE = {"haystack": 1.0 / (1.0 + 8.1),   # 8.1:1 neg:pos, SPEC.md sec.3
                 "hicodet": 18954.0 / 309895.0}     # pack v1 (fallback)
# HICO pack V2 (2026-08-26, duplicate boxes merged) has 18,846 positives over
# 100,449 labelled cells; v1 had 18,954 over 309,895. The chance level therefore
# depends on WHICH pack a run was evaluated on, and the fAP json records
# n_cells — so prevalence is resolved per run from that, never from a constant.
HICO_PREVALENCE_BY_CELLS = {309895: 18954.0 / 309895.0, 100449: 18846.0 / 100449.0}


USE_F1 = False


def f1_or(metrics, mr_key, r_key, use_f1):
    """A1/A3 cell value: mR@K alone, or F1@K = 2*R*mR/(R+mR).

    WHY F1 IS AN OPTION HERE. A1 and A3 are the two recall axes, and each is
    gameable in one direction on mR alone: A1 rewards tail-boosting that
    collapses the head (v42 took PSG mR +19% while `on` fell 81%), and A3
    rewards head collapse, because generic predicates sit inside nearly every
    accepted synonym set (SPEC.md §2). F1 weights the SMALLER of R and mR, so
    neither trick pays. A2 (fAP) and A6 (AUC) are NOT recall pairs and are
    left alone — forcing F1 onto them would be meaningless.

    ORDER OF OPERATIONS: this returns the RAW F1, which the caller then
    chance-corrects once. F1(norm(R), norm(mR)) != norm(F1(R, mR)) because a
    harmonic mean does not commute with an affine map; correcting once keeps
    the underlying quantity identical to SGG-Benchmark's definition, hence
    citable. Both R and mR share the same 1/V floor (a uniform-random
    predicate scores 1/V expected recall on EVERY class, so micro and macro
    coincide at chance), so there is no floor-mismatch reason to prefer the
    other order.
    """
    mr = metrics.get(mr_key)
    if not use_f1:
        return mr
    r = metrics.get(r_key)
    if r is None or mr is None:
        return None
    return 0.0 if (r + mr) <= 0 else 2.0 * r * mr / (r + mr)


def norm(x, chance):
    if x is None:
        return None
    return max(0.0, min(1.0, (x - chance) / (1.0 - chance)))


def parse_fap(logs, section):
    out, arm = {}, None
    hdr = re.compile(r"#+ +(\S+) — " + section)
    kv = re.compile(r"(\w[\w@]*): +([\d.]+)")
    for log in logs:
        if not Path(log).exists():
            continue
        for line in Path(log).read_text().splitlines():
            m = hdr.search(line)
            if m:
                arm = m.group(1)
                continue
            if arm and "mfAP:" in line and "coverage:" in line:
                out[arm] = {k: float(v) for k, v in kv.findall(line)}
                arm = None
    return out


def harmonic(vals):
    vals = [v for v in vals if v is not None]
    if not vals or min(vals) <= 0:
        return 0.0
    return len(vals) / sum(1.0 / v for v in vals)


def main() -> None:
    # fAP is read from the job LOGS, not the run dirs: the two eval_haystack
    # invocations shared an output basename before job_spec_cells.sh split them,
    # so some arms' A2 json on disk is actually their HICO fAP. Globbed and
    # sorted so every new spec_cells run is picked up without editing this list;
    # sorted == job-id order == chronological, and parse_fap lets later files
    # win, so a re-run of an arm supersedes its earlier entry.
    p = argparse.ArgumentParser()
    p.add_argument("--spec_log", nargs="*", default=[],
                   help="optional stdout logs of benchmark/eval_haystack.py; the "
                        "per-run haystack_sigmoid.json is read first")
    p.add_argument("--hico_log", nargs="*", default=[],
                   help="optional stdout logs of benchmark/eval_hico_map.py; the "
                        "per-run hico_fap/haystack_sigmoid.json is read first")
    p.add_argument("--latency", default="runs/benchmark/latency.json",
                   help="from benchmark/latency.py. Reported BESIDE "
                        "OVS, never inside it — see the LATENCY note below.")
    p.add_argument("--out", default="runs/benchmark/ovs.json")
    p.add_argument("--head_to_head", default=None,
                   help="also score the released tower against the baseline on "
                        "the cells both have, e.g. runs/benchmark/ovs_head2head.json")
    p.add_argument("--metric", choices=["mR", "F1"], default="mR",
                   help="cell metric for the two RECALL axes (A1, A3). "
                        "'F1' = harmonic mean of R@K and mR@K per SGG-Benchmark, "
                        "which neither tail-boosting nor head-collapse can game. "
                        "NOTE OVS values are NOT comparable across this choice — "
                        "it changes the axis definition, so recompute every arm.")
    a = p.parse_args()
    global USE_F1
    USE_F1 = (a.metric == "F1")

    # LATENCY IS A COST, NOT AN AXIS. It is deliberately kept out of the
    # harmonic mean: any cost term is optimised by doing less work, so folding
    # speed into a capability composite would let "fast and useless" outrank
    # "slow and correct". It is printed as its own column, plus an explicit
    # efficiency view (OVS per 100 ms) for anyone who wants the ratio — stated
    # rather than smuggled into the score.
    lat = {}
    if os.path.exists(a.latency):
        lj = json.load(open(a.latency))
        lat = {k: v for k, v in lj.get("runs", {}).items()}
        lat_meta = f"{lj.get('device', '?')}, bs1, budget {lj.get('eval_budget')}"
    else:
        lat_meta = ""

    vg_fap = parse_fap(a.spec_log, r"A2 Haystack fAP")
    hi_fap = parse_fap(a.hico_log, r"HICO A2")

    rows = []
    for arm, label in ARMS:
        d = Path("runs/train") / arm
        if not d.exists():
            continue
        cells, axes = {}, {}

        a1 = []
        for src, V in A1.items():
            f = d / f"zeroshot_{src}_test_gc.json"
            if f.exists():
                x = f1_or(json.load(open(f))["metrics"], "mR@50", "R@50", USE_F1)
                if x is None:
                    continue
                n = norm(x, 1.0 / V)
                cells[f"A1/{src}"] = (x, n)
                a1.append(n)
        if a1:
            axes["A1 transfer"] = sum(a1) / len(a1)

        a2 = []
        for src, prev in A2_PREVALENCE.items():
            # PREFER THE JSON, fall back to the log. eval_haystack.py writes
            # metrics.mfAP into haystack_sigmoid.json next to the checkpoint, and
            # that value is identical to the one it prints (verified: 0.75846 in
            # the json vs "mfAP: 0.7585" in the log for the ViT-B full run). The
            # log parser needs a "<arm> — A2 Haystack fAP" banner, so ANY caller
            # that prints a different banner silently loses A2 and its arm drops
            # to INCOMPLETE — which is exactly what happened to the resolution
            # sweep cells. Reading the file the eval actually produced removes
            # that coupling between a metric and a log format.
            jf = d / ("haystack_sigmoid.json" if src == "haystack"
                      else "hico_fap/haystack_sigmoid.json")
            m = None
            if jf.exists():
                m = json.load(open(jf)).get("metrics")
            if not m:
                m = (vg_fap if src == "haystack" else hi_fap).get(arm)
            if m:
                x = m.get("mfAP_sup5", m.get("mfAP"))
                if src == "hicodet":
                    nc = int(m.get("n_cells", 0) or 0)
                    if nc in HICO_PREVALENCE_BY_CELLS:
                        prev = HICO_PREVALENCE_BY_CELLS[nc]
                    elif nc:
                        print(f"!! {arm}: HICO fAP over {nc} cells — unknown pack, "
                              f"using v1 prevalence")
                n = norm(x, prev)
                cells[f"A2/{src}"] = (x, n)
                a2.append(n)
        if a2:
            axes["A2 precision"] = sum(a2) / len(a2)

        # A3 PROVENANCE GUARD. --tau_eval defaulted to 0.955, a threshold
        # calibrated in student_v1. Every arm from v38 on evaluates in
        # student_v2, where 0.955 accepts 0.6% of true synonyms instead of
        # ~64% — so the A3 cell measured near-exact string match for the v2
        # arms while the v1 arms kept a working matcher, and the two were
        # being ranked against each other. An A3 number is admitted only if it
        # was produced AFTER its space's calibration was fitted; anything older
        # is dropped, which makes that arm INCOMPLETE rather than silently
        # comparable. The eight deleted-checkpoint proxy arms can never be
        # re-measured, so they lose A3 permanently — correct, since the metric
        # they were scored under no longer exists.
        cal_mtime = max((os.path.getmtime(p) for p in
                         Path("runs/benchmark").glob("tau_calibration_*.json")),
                        default=0.0)
        a3, a3_stale = [], []
        for src in A3:
            f = d / f"zeroshot_{src}_test_gc_ov.json"
            if not f.exists():
                continue
            j = json.load(open(f))
            fresh = ("tau_eval" in j) or (os.path.getmtime(f) >= cal_mtime)
            x = f1_or(j["metrics"], "SoftmR@50", "SoftR@50", USE_F1)
            if x is None:
                continue
            n = norm(x, 0.0)
            cells[f"A3/{src}"] = (x, n)
            (a3 if fresh else a3_stale).append(n)
        if a3 and not a3_stale:
            axes["A3 open-vocab"] = sum(a3) / len(a3)
        elif a3_stale:
            cells["A3/STALE_TAU"] = (float("nan"), float("nan"))

        for name in A4_FILES:
            f = d / name
            if not f.exists():
                continue
            j = json.load(open(f))
            # "lenient" is the box-matching mode tab:sixaxes reports; a baseline
            # record written by the interchange scorer carries "metrics".
            blk = j.get("lenient") or j.get("metrics") or {}
            x = blk.get(A4_METRIC)
            if x is None:
                continue
            n = norm(x / CEILING, 1.0 / A1[A4_SOURCE])
            cells[f"A4/{A4_SOURCE}"] = (x, n)
            axes["A4 detector"] = n
            break

        f = d / "spatialsense.json"
        if f.exists():
            # MACRO (mean of per-predicate AUC), not the pooled AUC over all
            # 2,758 cells. SpatialSense's predicate mix is dominated by `on`
            # (807) and `behind` (406), which are the two we already handle, so
            # the pooled figure is largely a re-measurement of them: it read the
            # v43->v44 gain as +0.018 where the macro is +0.064, a 3.5x
            # understatement, and it RANKS wv2_lora12 above v43 (.684 vs .659)
            # where the macro puts them the other way (.655 vs .667). Every
            # other axis here is already macro; this makes A6 consistent.
            ss = json.load(open(f))
            pp = ss.get("per_predicate") or {}
            x = (sum(v["AUC"] for v in pp.values()) / len(pp)) if pp else ss["AUC"]
            n = norm(x, 0.5)
            cells["A6/spatialsense"] = (x, n)
            cells["A6/spatialsense_pooled"] = (ss["AUC"], norm(ss["AUC"], 0.5))
            axes["A6 spatial"] = n

        if not axes:
            continue
        # Every summary statistic is over the COMPOSITE axes. A3 stays in
        # `axes` because it is measured and reported; it does not enter here.
        comp = {k: axes[k] for k in COMPOSITE_AXES if k in axes}
        if not comp:
            continue
        vals = list(comp.values())
        ovs = harmonic(vals)
        arith = sum(vals) / len(vals)
        weakest = min(comp, key=comp.get)
        row = {"arm": arm, "label": label, "cells": cells, "axes": axes,
               "composite_axes": list(comp), "OVS": ovs, "OVS_arith": arith,
               "balance": (ovs / arith) if arith else 0.0,
               "weakest_axis": weakest, "weakest_value": comp[weakest]}
        L = lat.get(arm)
        if L:
            o = L.get("open_eval_bs1", {})
            c = L.get("closed_eval_bs1", {})
            row["latency"] = {
                "open_ms_mean": o.get("mean"), "open_ms_min": o.get("min"),
                "open_ms_max": o.get("max"), "open_ms_p95": o.get("p95"),
                "closed_ms_mean": c.get("mean"),
                "img_s_batch32": L.get("open_batch32_img_s"),
                "n_params_M": L.get("n_params_M"),
                "stages_ms": L.get("open_stages_ms")}
            if o.get("mean"):
                row["OVS_per_100ms"] = ovs / (o["mean"] / 100.0)
        rows.append(row)

    axis_names = sorted({k for r in rows for k in r["axes"]})
    for r in rows:
        r["missing_axes"] = [n for n in COMPOSITE_AXES if n not in r["axes"]]
        r["complete"] = not r["missing_axes"]
    print(f"composite: {list(COMPOSITE_AXES)}   (OVS is comparable only within "
          f"this set)\nalso measured, not in the composite: "
          f"{[n for n in axis_names if n not in COMPOSITE_AXES]}\n")
    w = max(len(r["label"]) for r in rows)

    def lat_cell(r):
        L = r.get("latency")
        if not L or L.get("open_ms_mean") is None:
            return f"{'--':>22s}"
        return (f"{L['open_ms_mean']:7.1f} "
                f"{'[%.1f-%.1f]' % (L['open_ms_min'], L['open_ms_max']):>14s}")

    def show(rs):
        for r in sorted(rs, key=lambda x: -x["OVS"]):
            print(f"{r['label']:{w}s} {r['OVS']:7.4f} {r['OVS_arith']:7.4f} "
                  f"{r['balance']:6.3f} "
                  + "".join(f"{r['axes'][n]:7.3f}" if n in r["axes"] else f"{'--':>7s}"
                            for n in axis_names)
                  + lat_cell(r)
                  + f"   {r['weakest_axis']} ({r['weakest_value']:.3f})")

    hdr = (f"{'arm':{w}s} {'OVS':>7s} {'arith':>7s} {'bal':>6s} "
           + "".join(f"{n.split()[0]:>7s}" for n in axis_names)
           + f"{'ms/img':>8s}{'[min-max]':>15s}   weakest")
    print(hdr)
    show([r for r in rows if r["complete"]])
    part = [r for r in rows if not r["complete"]]
    if part:
        # Scored on a SUBSET of the axes, so their OVS is not comparable to the
        # rows above (a harmonic mean over fewer axes cannot be penalised by the
        # axis that is absent). Printed separately rather than interleaved.
        print(f"\n-- INCOMPLETE (fewer axes; NOT comparable to the block above) --")
        show(part)

    timed = [r for r in rows if r.get("latency")]
    if timed:
        print(f"\n-- LATENCY ({lat_meta}) — a COST, reported beside OVS and "
              f"never inside its harmonic mean --")
        print(f"{'arm':{w}s} {'ms/img':>8s} {'p95':>7s} {'closed':>8s} "
              f"{'img/s@32':>9s} {'OVS/100ms':>10s}   stages (ms)")
        for r in sorted(timed, key=lambda x: x["latency"]["open_ms_mean"]):
            L = r["latency"]
            st = L.get("stages_ms") or {}
            print(f"{r['label']:{w}s} {L['open_ms_mean']:8.1f} "
                  f"{L['open_ms_p95']:7.1f} {L['closed_ms_mean']:8.1f} "
                  f"{L['img_s_batch32']:9.1f} {r.get('OVS_per_100ms', 0):10.4f}   "
                  + " ".join(f"{k} {v:.1f}" for k, v in
                             sorted(st.items(), key=lambda x: -x[1])))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"axis_set": list(COMPOSITE_AXES), "measured_axes": axis_names,
               "rows": rows,
               "definition": "norm=(x-chance)/(1-chance) per cell; arithmetic "
                             "within axis; HARMONIC across the composite axes",
               "chance": {"A1": "1/V per benchmark vocab", "A2": A2_PREVALENCE,
                          "A3": 0.0, "A4": f"1/{A1[A4_SOURCE]} after / ceiling "
                                           f"{CEILING}", "A6": 0.5}},
              open(a.out, "w"), indent=2)
    print(f"\nsaved → {a.out}")

    if a.head_to_head:
        h = head_to_head(USE_F1)
        Path(a.head_to_head).parent.mkdir(parents=True, exist_ok=True)
        json.dump(h, open(a.head_to_head, "w"), indent=2)
        print("\nhead-to-head (" + h["metric"] + ", cells both models have):")
        for n, m in h["models"].items():
            print(f"  {n:<16} OVS {100*m['OVS']:5.1f}   "
                  + " ".join(f"{k.split()[0]} {v:.3f}" for k, v in m["axes"].items())
                  + (f"   MISSING {m['missing_axes']}" if m["missing_axes"] else ""))
        print(f"saved → {a.head_to_head}")



# ========================================================= head-to-head ====
# The composite row of the paper's six-axis table. `main()` above scores OUR
# arms; this scores our released tower and the baseline side by side, on the
# CELLS BOTH MODELS HAVE. Two deliberate differences from the per-arm loop:
#
#   * A2 uses the haystack cell alone. Our arms average haystack with a HICO-DET
#     fAP cell; the baseline was never run on HICO fAP, and averaging a two-cell
#     axis against a one-cell axis compares different quantities. Restricting to
#     the common cell RAISES our A2 (0.608 -> 0.692), so it is stated here
#     rather than left implicit.
#   * A1's HICO-DET cell comes from the ZERO-SHOT tower for us, because the
#     released tower saw HICO-DET train and the baseline did not. This matches
#     the A1 rows of the same table; the per-arm loop above scores each arm on
#     its own HICO number, which is the right thing for arm selection and the
#     wrong thing for a head-to-head.
OURS_RUN = "runs/train/full_v42cfg_wv2-512_lora12_mixvghico0.05_sig0.25_btd0.3_negmask_def4h8v3n2_vits16plus_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5_hicov2"
OURS_ZS_RUN = "runs/train/full_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_vits16plus_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5"
BASE_DIR = "runs/ovsgtr"


def _metrics(path, block="metrics"):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    return j.get(block) or j.get("lenient") or j.get("metrics")


def _macro_auc(path):
    if not os.path.exists(path):
        return None
    j = json.load(open(path))
    pp = j.get("per_predicate") or {}
    return (sum(v["AUC"] for v in pp.values()) / len(pp)) if pp else j.get("AUC")


def head_to_head(use_f1: bool) -> dict:
    def a1(src, ours):
        if ours:
            run = OURS_ZS_RUN if src == "hicodet" else OURS_RUN
            m = _metrics(f"{run}/zeroshot_{src}_test_gc.json")
        else:
            m = _metrics(f"{BASE_DIR}/ovdr_mega_{src}_test_gtbox_gc.json")
        return f1_or(m, "mR@50", "R@50", use_f1) if m else None

    def a2(ours):
        m = (_metrics(f"{OURS_RUN}/haystack_sigmoid.json") if ours else
             _metrics(f"{BASE_DIR}/ovdr_mega_haystack_test_gtbox_haystack.json"))
        return m.get("mfAP_sup5", m.get("mfAP")) if m else None

    def a4(ours):
        m = (_metrics(f"{OURS_RUN}/zeroshot_detbox_psg_yoloworld_gc.json", "lenient")
             if ours else _metrics("runs/sgdet/ovdr_mega_psg_test_yoloworld_gc.json"))
        return m.get(A4_METRIC) if m else None

    def a5(ours):
        j = json.load(open(A5_FILE)) if os.path.exists(A5_FILE) else None
        if not j:
            return None
        v = j["per_system"].get(A5_KEY["ours" if ours else "base"], {})
        # Can exceed 1 at deployed depth (the annotation is sparse); the matched
        # depth used here does not, and the composite clips anyway.
        return v.get("share_of_annotation")

    def a6(ours):
        return _macro_auc(f"{OURS_RUN}/spatialsense.json" if ours else
                          f"{BASE_DIR}/ovdr_mega_spatialsense_test.json")

    out = {"axis_set": list(COMPOSITE_AXES), "metric": "F1" if use_f1 else "mR",
           "a2_cells": ["haystack"], "models": {}}
    for name, ours in (("RelateAnything", True), ("OvSGTR", False)):
        cells, axes = {}, {}
        vals = [(f"A1/{s}", a1(s, ours), 1.0 / V) for s, V in A1.items()]
        got = [(k, x, c) for k, x, c in vals if x is not None]
        if got:
            for k, x, c in got:
                cells[k] = (x, norm(x, c))
            axes["A1 transfer"] = sum(cells[k][1] for k, _, _ in got) / len(got)
        for key, axis, x, chance in (
                ("A2/haystack", "A2 precision", a2(ours), A2_PREVALENCE["haystack"]),
                ("A5/psg_matched", "A5 graph quality", a5(ours), 0.0),
                ("A6/spatialsense", "A6 spatial", a6(ours), 0.5)):
            if x is not None:
                cells[key] = (x, norm(x, chance))
                axes[axis] = cells[key][1]
        x = a4(ours)
        if x is not None:
            cells[f"A4/{A4_SOURCE}"] = (x, norm(x / CEILING, 1.0 / A1[A4_SOURCE]))
            axes["A4 detector"] = cells[f"A4/{A4_SOURCE}"][1]
        comp = {k: axes[k] for k in COMPOSITE_AXES if k in axes}
        out["models"][name] = {
            "cells": cells, "axes": axes, "OVS": harmonic(list(comp.values())),
            "missing_axes": [k for k in COMPOSITE_AXES if k not in axes],
            "weakest_axis": min(comp, key=comp.get) if comp else None}
    return out


if __name__ == "__main__":
    main()
