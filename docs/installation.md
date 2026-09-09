# Installation

## Quick path

```bash
git clone https://github.com/Maelic/RelateAnything
cd RelateAnything
./install.sh
```

`install.sh` creates `.venv`, installs the package editable, and warms the
Hugging Face cache into `./.hf_cache` so later runs can go fully offline. It is
idempotent — re-running it is safe.

Knobs:

```bash
PYTHON_BIN=python3.13./install.sh   # pick the interpreter
VENV_DIR=/scratch/env./install.sh   # put the venv elsewhere
SKIP_HF_WARMUP=1./install.sh        # no network at install time
```

## Extras

The base install is enough to train and evaluate. Everything else is opt-in:

```bash
pip install -e ".[deploy]"      # onnxruntime + onnx + full opencv — the demo
pip install -e ".[detector]"    # ultralytics (AGPL-3.0) — rebuild demo detectors
pip install -e ".[datagen]"     # vLLM annotation pipeline — heavy, GPU only
pip install -e ".[monitoring]"  # wandb + matplotlib
pip install -e ".[hub]"         # huggingface_hub — download weights and packs
pip install -e ".[gradio]"      # the server-side Gradio demo (needs [detector] too)
pip install -e ".[release]"     # huggingface_hub — publishing
pip install -e ".[dev]"         # pytest, black, isort
```

`datagen` is deliberately *not* a core dependency: it pulls `autoawq`, which
compiles, and nothing in training or inference needs it.

## Requirements

Python **≥ 3.12** (numpy 2.5 requires it); the released checkpoints were trained and verified on 3.13.
Dependencies are pinned with `~=` in [`pyproject.toml`](../pyproject.toml)
against that verified environment — loosening a pin is fine for
experimentation, but the pins document what is *known* to reproduce.

A note on `pycocotools`: under plain `pip` this resolves to the upstream
package. The project uses a fork, which only `uv` picks up automatically. With
plain pip:

```bash
pip install git+https://github.com/Maelic/pycocotools
```

## Get the weights

Three models, one repository each on the Hugging Face Hub, all under `maelic/`:
`relsgg-vits16`, `relsgg-vits16plus` (the recommended default) and
`relsgg-vitb16`. Each carries `model.pth` (torch, EMA weights, backbone config
embedded), `text_student.pt` (the distilled predicate text encoder),
`predicate_embeddings.npz` (the training vocabulary already encoded with that
student) and its calibrated `predicate_bank.npz`. `relsgg-vits16plus` also
carries the ONNX graph the torch-free demo runs on.

```bash
pip install -e ".[hub]"
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download("maelic/relsgg-vits16plus"))     # -> local directory with model.pth
PY
```

**Running a released checkpoint needs no gated download**: `model.pth` embeds
the backbone configuration and all weights. The gated DINOv3 repositories are
only needed to *train*.

## Get the data

| what | where | size |
|---|---|---|
| RA-4M annotations, image manifest, training packs | [`maelic/RA-4M`](https://huggingface.co/datasets/maelic/RA-4M) | about 0.4 GB |
| evaluation packs, negatives, text student, recipe artifacts | [`maelic/OV-SGG-Bench`](https://huggingface.co/datasets/maelic/OV-SGG-Bench) | about 0.2 GB |
| images | the source datasets (MegaSG via `JosephZ/mega_1m`, VG150, PSG, IndoorVG, HICO-DET, Haystack, SpatialSense) | not redistributed |

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("maelic/OV-SGG-Bench", repo_type="dataset", local_dir="runs/ovsgg_bench")
snapshot_download("maelic/RA-4M", repo_type="dataset", local_dir="runs/ra4m")
PY
mkdir -p runs/packed runs/datamix
ln -s../ovsgg_bench/packs/*../ra4m/packs/* runs/packed/
ln -s../ovsgg_bench/datamix/* runs/datamix/
ln -s ovsgg_bench/text_student_v2_512 runs/packed/text_student_v2_512
ln -s ovsgg_bench/datamix_v22 runs/packed/datamix_v22
```

After that every command in the docs works with its default paths. Packs
record image roots as `datasets/<name>/<split>`; point `RA_DATASETS` at the
directory that holds your copies of the source images.

## The gated backbone (training only)

DINOv3 weights live in gated Hugging Face repositories
(`facebook/dinov3-vits16-pretrain-lvd1689m`, `-vits16plus-`, `-vitb16-`). Two ways
through:

**Accept the terms and log in** (needed once):

```bash
huggingface-cli login          # account that accepted Meta's DINOv3 terms
```

**Or convert a local copy** of Meta's raw checkpoints and point
`--backbone_model` at the resulting directory:

```bash
python training/convert_dinov3_local.py --help       # ViT towers
python training/convert_dinov3_convnext.py --help    # ConvNeXt towers
```

If neither is available, `install.sh` still succeeds; it prints a warning and
you get the failure later, at model construction, with a clearer message.

## Running offline

Once the cache is warm, nothing needs the network:

```bash
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## On a cluster

Compute nodes are usually offline. Warm the cache on a login node, then export
the offline flags in the job:

```bash
source.venv/bin/activate
export HF_HOME="$PWD/.hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
export RA_DATASETS=/path/to/datasets
```

Two conventions worth copying:

- **Log to shared storage, not `/tmp`.** On most clusters `/tmp` is node-local,
  so a job that logs there silently discards its output.
- **Resolve run directories before submitting.** Assert that every arm of a
  sweep resolves to a distinct, non-pre-existing output directory, and that a
  retyped recipe reproduces a reference run's name character for character.
  Both guards exist because both failures happened.

## Verify the install

```bash
pytest tests/ -q
```

These are CPU-only and need no weights and no network — the same suite CI runs.
They cover the install path, imports, evaluator math, score parity between the
training and deploy paths, and edge decoding.
