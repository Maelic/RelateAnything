# RA-4M data generation

The pipeline that produced RA-4M
([`maelic/RA-4M`](https://huggingface.co/datasets/maelic/RA-4M)):

- train — 474,413 images, 4,282,531 relations (9.03 rels/img), 10,102 unique
  predicates, 48.5% spatial, 104.3 GPU-h
- val — 24,964 images, 226,203 relations (9.06 rels/img), 2,222 unique
  predicates, 48.4% spatial, 5.6 GPU-h

Both are synthetic visual-relationship annotations of MegaSG
(`$RA_DATASETS/MEGASG`) produced by
`google/gemma-4-26B-A4B-it` (MoE, 26B total / 4B active) via vLLM, for
training an open-vocabulary scene-graph model. The COCO scene-graph export (`export_coco_sgg.py`) is derived from the JSONL,
not a separate source of truth.

## Architecture: two-layer generation

1. **Semantic pass** — open-vocabulary actions/interactions, spatial predicates
   banned. Base prompt `datagen/prompts/iter_20.txt` (coverage-oriented; a
   stricter verify-before-assert prompt was tried and rejected — it bought
   precision by collapsing predicate diversity, and the deterministic gates
   below recover the same precision without the cost).
2. **Grow round** — one additive pass with `datagen/prompts/sgg26b_grow_v1.txt`
   to pad sparse images (mainly fixes body-part-heavy images; part junk it
   introduces is cleaned by the part gate).
3. **Spatial round** — open-vocabulary LLM spatial pass with
   `datagen/prompts/sgg26b_spatial_v3.txt`, geometry-verified
   (`verify_spatial_v2` in `sgg_vllm_generate.py`: direction fixed by
   role-swap, contact must touch, containment needs ⊂0.5, near needs
   ≤0.5×larger-box-diagonal) plus a salience-ranked geometric top-up
   (`--geo_backstop`, `sgg_geometric_spatial.py`) for images still below the
   spatial-density floor. An earlier deterministic-only geometric spatial
   layer (no LLM) was faster but monotone (6 forms, 58% left-right, weak
   depth) — the LLM+verify layer replaced it as the primary source, geometry
   is now only the backstop.

Deterministic cleanup applied to every relation (`sgg_canon.py` +
`sgg_postprocess.py`):

- `--contact_gate` — drops contact/containment relations (e.g. "wearing")
  between boxes that don't actually touch.
- `--part_gate` — drops body-part-as-subject junk and duplicate-person-box
  relations (the main failure mode the grow round introduces).
- `--canonicalize --keep_spatial_synonyms` — drops vague/scaffold predicates
  ("positioned near" → dropped, not collapsed to "near") but **keeps surface-
  form synonyms** ("atop"/"on top of"/"resting on" all survive as distinct
  strings). Canonical forms are used internally only, for dedup and gating —
  never emitted. This is intentional: the training goal is open-vocabulary
  latent coverage, so lexical variety is signal, not noise, even though it
  costs precision. See the same drop rule in `--keep_spatial_synonyms` when
  reusing this pipeline for a different target.

Engine settings: fp8 quantization on A100-40GB, `--gpu_mem_util 0.95
--chunk 256 --async_scheduling` (measured 1.16× throughput over the naive
config, ~0.79-0.80 s/img sustained at 475K scale).

## Running it

Everything runs from the repository root with `RA_DATASETS` pointing at the
directory that holds `MEGASG/{train,val}` (images + `_annotations.coco.json`
from [JosephZ/mega_1m](https://huggingface.co/datasets/JosephZ/mega_1m); the
image set is listed in `release/ra4m/`). The generator needs its own
environment (`datagen/requirements-vllm.txt`: vLLM pins a torch version that
must not clobber the training venv).

**Production flags.** One process annotates one shard; run N shards on N GPUs
in parallel (a job array on a cluster, or N processes). These are the exact
defaults the corpus was generated with:

```bash
python datagen/sgg_vllm_generate.py \
    --name megasg_26b_F2_train --outdir runs/vllm_generate \
    --model 26b --strategy grow --split train --overlay point_only \
    --num_shards 8 --shard_index $SHARD \
    --max_new_tokens 1024 --chunk 256 --gpu_mem_util 0.95 --async_scheduling \
    --quant fp8 \
    --grow_rounds 3 --prompt_file datagen/prompts/iter_20.txt --grow_prompt datagen/prompts/sgg26b_grow_v1.txt \
    --spatial_source llm_v2 --spatial_prompt datagen/prompts/sgg26b_spatial_v3.txt \
    --geo_backstop 5 --max_spatial_per_subject 3 \
    --canonicalize --keep_spatial_synonyms --contact_gate --part_gate --postprocess
```

`--num_shards N --shard_index i` gives shard `i` a disjoint, balanced slice of
the split; a shard resumes from its own JSONL when restarted. Use
`--exclude_run <dir>` (repeatable) to annotate only images not covered by
earlier runs (safer than `--skip`, which is a raw index and shifts when a
filter changes).
For a preview, add `--limit 2500`. The val split is the same command with
`--split val`. Check the flag names against `python datagen/sgg_vllm_generate.py --help`
if a version has renamed one.

**Assemble** after every shard has finished (verifies coverage, merges,
deduplicates resumed shards, writes the `.stats.json`):

```bash
python datagen/assemble_dataset.py runs/vllm_generate/megasg_26b_F2_train \
    --out runs/vllm_generate/megasg_sgg_train_full.jsonl
python datagen/analyze_sgg_run.py --run runs/vllm_generate/megasg_sgg_train_full.jsonl
python datagen/export_coco_sgg.py --split train          # COCO scene-graph export
python training/pack_megasg.py --dataset megasg          # -> runs/packed/megasg
```

**Masks** (optional sidecars for the mask-input experiments and the figure
panels): `sam3_masks.py` segments a COCO json with box prompts
(`--backend sam2` is ungated), sharded the same way; `sam3_masks.py --merge`
joins the shards. For the `.npy` packs, `pack_mask_manifest.py build` writes a
union manifest over every pack (one encode per unique image), the same
segmenter runs over it, and `pack_mask_manifest.py scatter` writes the
per-pack rasters that `build_mask_rasters.py` and `check_pack_masks.py`
consume. `psg_gt_masks.py` derives PSG ground-truth masks from the COCO
panoptic PNGs without a segmenter.

## Pipeline scripts kept in this repo

Production generator and its runtime dependencies:
- `sgg_vllm_generate.py` — vLLM batch generator (entry point)
- `creative_prompt_agent.py`, `prompt_eval.py`, `annotate_mask_som.py` —
  prompting/parsing/annotation-rendering helpers it imports
- `sgg_canon.py`, `sgg_geometric_spatial.py`, `sgg_postprocess.py` — the
  canonicalization, geometric-spatial, and cleanup logic described above

Assembly and QA tooling used to build and validate the final files:
- `assemble_dataset.py` — merges sharded runs into one final JSONL + stats,
  verifying shard coverage and deduplicating resumed shards
- `analyze_sgg_run.py` (+ `audit_contact_geometry.py`) — predicate/degree/
  spatial-layer statistics and the geometry-consistency audit
- `apply_gates_offline.py` — replay contact/part gates on an existing run
  without a GPU, for calibrating new gate rules before baking them into the
  live generator
- `export_coco_sgg.py` — converts the final flat JSONL into MegaSG's own
  COCO scene-graph schema, reusing object boxes/ids verbatim

Mask sidecars: `sam3_masks.py`, `pack_mask_manifest.py`, `build_mask_rasters.py`,
`check_pack_masks.py`, `psg_gt_masks.py`. `download_openimages_ext.py` fetches
the Open Images V6 relationship subset used in one ablation.

The exploratory work behind this recipe (per-strategy prompt variants and
early annotator comparisons) is not part of the release. The corpus itself is
published as [`maelic/RA-4M`](https://huggingface.co/datasets/maelic/RA-4M).
