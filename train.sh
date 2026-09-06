#!/usr/bin/env bash
# Interactive / srun training launch: the CURRENT release recipe, single node.
#
#   BACKBONE=facebook/dinov3-vits16-pretrain-lvd1689m ./train.sh
#   NPROC=1 EPOCHS=1 ./train.sh                                    # smoke run
#
# Data roots are PACKED datasets (training/pack_megasg.py), not raw COCO; see
# docs/installation.md for where to download them. Every flag below is
# explained in docs/training.md; the parts that matter are ranked there by
# measured effect. The exact arguments of each released model are in
# training/configs/.
#
# The backbone is a gated Hugging Face repository: `huggingface-cli login`
# once, or point BACKBONE at a local conversion (training/convert_dinov3_local.py).
# On an offline compute node, warm the cache first and export
# HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export WANDB_MODE="${WANDB_MODE:-offline}"

BACKBONE="${BACKBONE:-facebook/dinov3-vits16plus-pretrain-lvd1689m}"
NPROC="${NPROC:-4}"
EPOCHS="${EPOCHS:-12}"
RUN="${RUN:-runs/train/relsgg_$(basename "$BACKBONE")}"

torchrun --nproc_per_node "$NPROC" train.py \
    --data_roots runs/packed/megasg_clean runs/packed/vg_raw runs/packed/hicodet \
    --mix_fractions 0.7274 0.063 0.2096 \
    --restrict_neg_sources hicodet \
    --exclude_ids runs/datamix/indoorvg_holdout.json \
    --backbone_type dinov3 --backbone_model "$BACKBONE" \
    --lora_rank 0 --lora_layers 12 --backbone_lr 5e-5 \
    --lr 4e-4 --epochs "$EPOCHS" --batch_size 32 --weight_decay 1e-4 \
    --warmup_epochs 1 --warmup_steps 500 --clip_grad 1.0 --min_lr_factor 0.01 \
    --img_size 448 --multi_scale 0.5,1.5 --multi_scale_n 7 --max_objects 40 \
    --d_model 512 --n_self_layers 2 --n_cross_layers 2 \
    --use_rel_interaction --n_dep_layers 2 --n_gnd_layers 1 \
    --sampler_type relatedness --geo_budget 400 --final_budget 128 \
    --text_student runs/packed/text_student_v2_512/student.pt --text_dim 512 \
    --dual_spatial_head --gate_mlp --proj_layers 2 \
    --deformable_points 4 --deformable_heads 8 --deformable_nulls 2 \
    --deformable_ring --deformable_clamp \
    --norm_taps --scene_pe --geo_squash --pe_num_freqs 16 --pe_max_octave 7 \
    --lambda_sigmoid 0.25 --lambda_bg 0.05 --lambda_swap 0.5 \
    --lambda_infonce 0.5 --lambda_czsc 0.05 --logit_scale_init 5.0 \
    --box_token_dropout 0.3 --dropout 0.2 --augment 0.3 \
    --cfa_mode entity --cfa_prob 0.5 \
    --soft_supervision runs/packed/datamix_v22/text_space/soft_supervision.npz \
    --neg_rate_table runs/packed/datamix_v22/pair_opportunity.npz \
    --dev_root runs/packed/psg --dev_split val --dev_metric mR@50 --dev_select \
    --amp --num_workers 16 --eval_budget 400 --ema_decay 0.9998 \
    --output_dir "$RUN"
