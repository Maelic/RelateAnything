#!/usr/bin/env bash
# Fresh-environment install for RelSGG.
#
#./install.sh                      # venv + editable install + HF cache warmup
#   SKIP_HF_WARMUP=1./install.sh     # offline machine: skip the downloads
#
# The dino.txt teacher checkpoint (2.25 GB, gated by Meta) is needed only to
# retrain the distilled text student. Released models ship their own student,
# so its absence is a note here, never a failure.
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-.venv}
HF_HOME_DIR=${HF_HOME_DIR:-$PWD/.hf_cache}
CHECKPOINT=${CHECKPOINT:-checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e.

mkdir -p "$HF_HOME_DIR"
export HF_HOME="$HF_HOME_DIR"

if [[ "${SKIP_HF_WARMUP:-0}" != "1" ]]; then
    python - <<'PY'
# Pre-download the tokenizer + ViT-B backbone so training/eval can run with
# HF_HUB_OFFLINE=1 afterwards. The dinov3 repo is gated: fetching it needs
# `huggingface-cli login` with an account that accepted Meta's terms.
from transformers import AutoModel, AutoTokenizer

AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
try:
    AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
except Exception as e:  # gated repo without login — install still succeeds
    print(f"[warn] could not fetch the DINOv3 backbone ({type(e).__name__}). "
          "Accept Meta's terms + `huggingface-cli login`, or point "
          "--backbone_model at a local converted copy (checkpoints/hf/...).")
PY
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[note] dino.txt teacher not present ($CHECKPOINT)."
    echo "       Only needed to retrain the text student."
fi

cat <<EOF
Environment ready.
Virtualenv: $VENV_DIR
HF cache:   $HF_HOME_DIR

Only the core package is installed. Optional extras, none of them required to
train or run the model:
  pip install -e ".[deploy]"      ONNX runtime + the webcam demo
  pip install -e ".[dev]"         pytest, black, isort  -> then: pytest tests/ -q
  pip install -e ".[monitoring]"  wandb + matplotlib    -> needed for --wandb
  pip install -e ".[detector]"    ultralytics (AGPL-3.0), to rebuild demo detectors
  pip install -e ".[release]"     huggingface_hub, for release/hf_upload.py
  pip install -e ".[datagen]"     the vLLM annotation pipeline (heavy, GPU only)

Docs: docs/README.md
EOF
