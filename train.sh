#!/usr/bin/env bash
# Train a released model on one node. Every architecture and optimisation flag
# defaults to the released recipe (relsgg/config.py), so this sets only the
# data, the backbone and the artifacts.
#
#   BACKBONE=facebook/dinov3-vits16-pretrain-lvd1689m ./train.sh
#   NPROC=1 EPOCHS=1 ./train.sh                                  # smoke run
#
# Data roots are PACKED datasets (training/pack_megasg.py), not raw COCO; see
# docs/installation.md for where to download them, and docs/training.md for
# what the recipe does and why.
#
# The backbone is a gated Hugging Face repository: run `huggingface-cli login`
# once, or point BACKBONE at a local conversion
# (training/convert_dinov3_local.py). On an offline node, warm the cache first
# and export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1.
set -euo pipefail

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

BACKBONE="${BACKBONE:-facebook/dinov3-vits16plus-pretrain-lvd1689m}"
NPROC="${NPROC:-4}"
EPOCHS="${EPOCHS:-12}"
TS="${TS:-runs/packed/datamix_v22/text_space}"
RUN="${RUN:-runs/train/relsgg_$(basename "$BACKBONE")}"

torchrun --nproc_per_node "$NPROC" train.py \
    --data_roots runs/packed/megasg_clean runs/packed/vg_raw runs/packed/hicodet \
    --mix_fractions 0.7274 0.063 0.2096 \
    --restrict_neg_sources hicodet \
    --exclude_ids runs/datamix/indoorvg_holdout.json \
    --val_root runs/packed/megasg \
    --dev_root runs/packed/psg --dev_metric mR@50 \
    --backbone_model "$BACKBONE" \
    --pred_embeds "$TS/pred_embeds_studentv2_512_photo.npz" \
    --ontology_meta "$TS/union_meta.json" \
    --soft_supervision "$TS/soft_supervision.npz" \
    --neg_rate_table runs/packed/datamix_v22/pair_opportunity.npz \
    --text_student runs/packed/text_student_v2_512/student.pt \
    --epochs "$EPOCHS" --batch_size 32 --num_workers 16 \
    --output_dir "$RUN"
