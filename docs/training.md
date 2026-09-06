# Training

## What the released models were trained with

One recipe, three backbones. The backbone is the only variable across the
family. Full-scale runs are 503,754 images/epoch × 12 epochs on 4× A100
(global batch 128), roughly 6 hours each.

```bash
torchrun --nproc_per_node 4 train.py \
  --data_roots runs/packed/megasg_clean runs/packed/vg_raw runs/packed/hicodet \
  --mix_fractions 0.7274 0.063 0.2096 \
  --restrict_neg_sources hicodet \
  --backbone_model facebook/dinov3-vits16plus-pretrain-lvd1689m \
  --val_root runs/packed/megasg \
  --dev_root runs/packed/psg --dev_metric mR@50 \
  --pred_embeds runs/packed/datamix_v22/text_space/pred_embeds_studentv2_512_photo.npz \
  --ontology_meta runs/packed/datamix_v22/text_space/union_meta.json \
  --soft_supervision runs/packed/datamix_v22/text_space/soft_supervision.npz \
  --neg_rate_table runs/packed/datamix_v22/pair_opportunity.npz \
  --text_student runs/packed/text_student_v2_512/student.pt \
  --exclude_ids runs/datamix/indoorvg_holdout.json \
  --output_dir runs/train/<name>
```

Every architecture and optimisation flag defaults to the released recipe
([`relsgg/config.py`](../relsgg/config.py)), so the command above sets only the
data, the backbone and the artifacts. [`train.sh`](../train.sh) is the same
command with the backbone as a variable, and the resolved arguments of every
released run are in [`training/configs/`](../training/configs/), one JSON per
model.

Inputs, and where they come from (see [installation](installation.md#get-the-data)):

| input | source |
|---|---|
| `runs/packed/megasg_clean`, `vg_raw`, `hicodet` | training packs in the `maelic/RA-4M` dataset repository, or rebuilt with `training/pack_megasg.py` |
| `runs/packed/psg` (checkpoint selection) | `maelic/OV-SGG-Bench` |
| `runs/packed/text_student_v2_512/student.pt` | `maelic/OV-SGG-Bench`; to retrain it, `training/distill/` |
| `runs/packed/datamix_v22/{text_space/soft_supervision.npz, pair_opportunity.npz}` | `maelic/OV-SGG-Bench`; rebuilt by `training/build_soft_supervision.py` and `training/build_pair_opportunity.py` |
| `runs/datamix/indoorvg_holdout.json` | `maelic/OV-SGG-Bench`; rebuilt by `training/build_indoorvg_holdout.py` |
| the backbone | the gated `facebook/dinov3-*` repositories (`huggingface-cli login`) or a local conversion |
| images | `RA_DATASETS` must hold the MegaSG, Visual Genome and HICO-DET images the packs name |

## The parts of the recipe that actually matter

Ranked by measured effect, not by how interesting they sound.

| ingredient | flag | why |
|---|---|---|
| Backbone adaptation | full fine-tune at `--backbone_lr 5e-5` | adapting the backbone is *the* lever; a frozen backbone is far behind |
| Distilled text student | `--text_student ...` | the predicate space; a 512-d student beat 768-d on all four bars |
| Source-aware negatives | `--restrict_neg_sources hicodet` | best OVS of any arm. Stops InfoNCE pushing `on`/`above` down for verb-only HICO pairs |
| Deformable read | `--deformable_points 4 --deformable_heads 8 --deformable_nulls 2` | wins on every axis and refunds the sigmoid-aux tax |
| Sigmoid aux | `--lambda_sigmoid 0.25` | the first training change to move spatial reasoning (+0.041 AUC) |
| Multi-scale | `--multi_scale 0.5,1.5 --multi_scale_n 7` | +3.7 % OVS-F1, 9/11 cells. Mechanism is **regularization**, not resolution |
| Tap normalization | `--norm_taps` | free; decouples tap importance from activation scale |
| Feature augmentation | `--cfa_mode entity --cfa_prob 0.5` | +10.1 % zero-shot mean recall. Same-predicate partner matching is essential — random partners *hurt* |
| Background suppression | `--lambda_bg 0.05` | supervises the fused score on non-GT pairs; top-k only, so the tail is protected |
| Learning rate | `--lr 4e-4 --epochs 12` | +18.8 % over the short-ladder recipe. Proxy ladders were undertrained |

## Checkpoint selection

```
--dev_root runs/packed/psg --dev_split val --dev_metric mR@50 --dev_select
```

Selection reads a **zero-shot out-of-domain** metric — PSG validation mean
recall — not in-domain RA-4M validation. Selecting in-domain rewards corpus
fit, which is the opposite of what this model is for.

**Ship `checkpoint_last`, not `checkpoint_best`.** The final epoch predicts
out-of-domain behaviour better (rank correlation 0.771 vs 0.657); best-epoch
selection rewards curve noise. All released models ship the final epoch. This
inverts real conclusions — a full-fine-tune arm that "won by 8 %" on best-epoch
scoring *lost* when re-scored on final.

## Multi-GPU

```bash
torchrun --nproc_per_node 4 train.py...
```

Standard DDP, single node. Two constraints:

- **The loss is batch-local InfoNCE.** Per-rank batch size changes the contrast
  set, so raising it to "fill" a bigger GPU makes runs incomparable to the
  lineage. Keep global batch at 128 and change `--grad_accum` if you must.
- **Right-size the request.** Taking every GPU on a node takes all of its CPUs
  anyway, so ask for them and use them (`--num_workers 16` per rank). A run
  peaking at 25 GB on an 80 GB GPU is the profile cluster efficiency watchdogs
  cancel.

## Running an ablation

Use the 50K proxy pack: about 2.5 GPU-hours per arm instead of 24. Validate it
for your question first — a proxy is only trustworthy for effects that show up
on it at full scale too.

```bash
python training/build_proxy_pack.py --out runs/packed/megasg_proxy50k
```

Three rules learned the hard way, worth enforcing in whatever launcher you
write:

1. **Exported environment leaks between arms.** An arm that must *not* have a
   knob needs it unset explicitly, or it silently inherits it.
2. **Never retype a recipe.** Assert that the recipe reproduces a reference
   run's directory name character-for-character; a hand-copied recipe once
   dropped 20+ arguments silently.
3. **Resolve output directories before submitting.** Two arms resolving to one
   directory destroys both.

And gate arms on `SoftmR@50`, not `dev_mR@50` — arm ranking is stable from
about epoch 2 either way, but the level of a metric is not.

## Monitoring

```bash
--wandb --wandb_project relanything     # WANDB_MODE=offline on a cluster
```

`history.json` in the run directory carries per-epoch metrics with no external
service. Watch the train/val InfoNCE gap: the 472K-relation pack starts
overfitting around epoch 8, and `--drop_path` cuts that gap 8.8× while moving
accuracy by ≈ 0 — the overfitting and the accuracy are decoupled, so do not
chase the gap for its own sake.

> The per-class and SoftR splits printed **during training** are unconstrained
> and computed over the full 19K vocabulary. They are diagnostics. Spatial
> predicates can read near zero there while the real, graph-constrained value
> is 10–200× higher. Never quote them. See [pitfalls](pitfalls.md).
