# Pitfalls

Every entry here has produced a plausible, wrong result at least once in this
project. They are ordered by how much damage they did.

## Metrics

### The graph constraint inflates R@K by 12–19 points

Without `--graph_constraint`, a pair contributes its entire predicate column to
the ranked list instead of one predicate. An earlier round of this project's
results was unconstrained, and every comparison drawn from it was invalid.
Check both sides of any comparison.

### In-training diagnostics are not results

The per-class and `SoftR` splits printed in the training log are **unconstrained
and computed over the full 19K vocabulary**. Spatial predicates routinely read
near zero there while the real, graph-constrained value is 10–200× higher.
They are for watching a curve. Never quote them.

### Select on the final epoch, not the best one

Dev-set final-epoch performance predicts out-of-domain behaviour better than
best-epoch (rank correlation 0.771 vs 0.657); best-epoch selection rewards curve
noise. This is not a small effect: a full-fine-tune arm that "won by 8 %" on
best-epoch scoring **lost** when re-scored on the final epoch, which inverted
the conclusion about whether the lineage was mis-tuned.

### Never judge scaling on in-domain validation

The half-data ablation showed a large in-domain gain and only +2.5–5.2 % R@50
on transfer. Read in-domain only, the obvious conclusion was "annotate 500K more
images". That would have been wrong.

### Exact-string mean recall on a 19K vocabulary is misleading

A predicate reading 0.006 usually means surface-form fragmentation across
synonyms, not a missing capability. When we chased `right of` at 0.006, direction
turned out to be fine — swap accuracy 0.97. Check the synonym group before
diagnosing a capability gap.

### The composite is not a selection metric

OVS is a chance-corrected harmonic mean over six axes. Selecting on it optimizes
the aggregate rather than the deficiency. Only select on it when the run
explicitly targets the currently weakest axis.

## Data

### `mix_fractions` is per-image; the loss is per-relation

Images from different sources carry very different relation counts. A nominal
50 % source contributed **16 %** of the actual supervision. Convert to relation
share before reasoning about how much a source teaches.

### Evaluation images leak across corpora under different id spellings

IndoorVG holdout images appear in both `vg_raw` and `megasg` under two id
conventions, and the datamix registry does not catch it. Run
`training/audit_image_overlap.py` and pass `--exclude_ids`.

### Vocabulary overlap is a separate leak, and a bigger one

Zero image overlap does not make a benchmark neutral. See
[evaluation](evaluation.md).

### No horizontal flips

The vocabulary contains `to the left of` and `to the right of`. A flip silently
falsifies the label. The augmentation is deliberately absent — do not add it.

## Model and scoring

### The text space must match the checkpoint

The head's `W` lives in the space of the text encoder the checkpoint was trained
with — the distilled 512-d student that ships beside every released model.
Encoding a vocabulary with any other encoder raises no error and produces
meaningless cosines. Let
`from_checkpoint` read it from the checkpoint's own args.

### Thresholds do not transfer between checkpoints

Score scales are checkpoint-specific because the output head is rank-trained.
The release tooling writes **NaN** into an uncalibrated bank row rather than
borrowing a number from another model. Do not fill it in by hand.

### Raw scores are uncalibrated by construction

The head is trained against a balanced prior; a real frame is 0.2–4 % positive.
Raw scores therefore pile into `[0.9, 1.0)` and a threshold means nothing until a
Platt `(a, b)` is installed. It is monotone, so no ranking metric changes.

### Relatedness raises recall and lowers truthfulness

`pair_logit` is partly a model of which pairs a human bothered to annotate.
Dropping it at evaluation raises macro AUC on adjudicated negatives by 0.068
(projective predicates by 0.11–0.14) while lowering recall on every
annotation-derived benchmark. Pick `pair_weight` for the question you are asking;
do not assume the default is right for it.

### Modules built post-hoc must exist before loading

`gate_mlp` and `beta_mlp` are constructed at train time by flags. Load a
checkpoint without building them first and their weights land silently in the
"unexpected keys" bucket — a quiet accuracy regression, not an error.
`from_checkpoint` handles this; hand-rolled loaders must too.

## Deployment

### `torch.compile` with `reduce-overhead` is silently wrong

CUDA graphs give a real 3.45× speedup and produce incorrect results in this
pipeline. Batch-size-1 inference is CPU-dispatch bound (~1362 kernels), so
resolution changes and plain compilation buy nothing anyway.

### Deployment and evaluation must run one score function

The score was independently re-implemented in five places and they disagreed —
evaluation scored `sigmoid(pred + rel)`, the product scored
`sigmoid(pred) · sigmoid(rel)`. Those rank pairs differently, so no reported
number came from the formula a user actually ran. Worse, the benchmarks
*preferred* the wrong one. [`relsgg/scoring.py`](../relsgg/scoring.py) is now the
single definition, and `tests/test_score_parity.py` proves it rather than
asserting it.

## Experiment hygiene

### `sbatch --export=ALL` leaks environment between arms

An arm that must *not* have a knob inherits it from the submitting shell.
Strip per-arm with `env -u`.

### Never retype a recipe

A hand-copied recipe silently dropped 20+ arguments. Assert that the recipe
reproduces a reference run's directory name character-for-character.

### Resolve output directories before submitting

Two arms resolving to one directory destroys both. Resolve and assert
distinctness first.

### Model soup across seeds destroys the model

`--seed` re-randomizes head initialization, so different-seed runs land in
mismatched basins and averaging them halved A1. Shared-init souping is a
different question and remains untested.

