# What a source knows: the contrastive objective

Read this before adding a dataset, a pseudo-label source, or a loss term.
The code is [`relsgg/training/losses.py`](../relsgg/training/losses.py).

## The problem in one sentence

A pair's label set is never complete, and **the incompleteness has structure by
source**. A contrastive loss is a claim about what the data knows, and every
loss-side regression measured on this project was that claim being wrong
somewhere:

| what was treated as a negative | what it actually was | cost |
|---|---|---|
| a synonym of the annotation (`riding` against `riding on`) | a positive | tail collapse |
| a co-occurring predicate (`holding` on a `looking at` pair) | unknown, and often true | rare-class recall |
| `on` or `above` on a HICO `riding` pair | true, but outside what HICO can annotate | projective SpatialSense AUC −0.05, independent of the source's share |
| a pair's other annotated predicates, under single-label targets | positives | tail recall |

Each fix was local. This note is the general rule, so the next source does not
need a new one.

## Vocabulary

For an anchor `a` (a directed pair in an image) from source `s`:

- `L_a` — its annotated predicates (multi-hot).
- `C_s` — the **label space of the source**: the columns the annotator was
  choosing among. HICO-DET: 117 verbs. PSG: 56. RA-4M and Visual Genome: open
  text, effectively anything. A pseudo-label source: exactly the columns the
  labeller was asked about.
- `e_s` — **exhaustiveness within `C_s`**: the probability that a true predicate
  in `C_s` was annotated. HICO annotates every verb of a pair, so it is near 1;
  a Visual Genome annotator writes one relation and moves on, so it is low.
- `N_a` — explicitly negative cells (HICO's "no interaction", Haystack's
  adjudicated negatives).
- `inv(L_a)` — spatial inverses of the annotation. These are true negatives by
  structure, the one case where a negative is certain without annotation.
- `k(v | L_a, cats_a)` — the co-annotation kernel, the probability that `v` is
  also true given what was written and the two object categories, fitted on the
  pairs that carry more than one label.
- `syn(v | L_a)` — the synonym kernel, the probability that `v` paraphrases the
  annotation given their text cosine. Fitted, not thresholded.

Everything below is a function of those six objects and nothing else.

## The status of a column, for one anchor

For a column `v` in the batch's contrast set, exactly one of:

| status | condition | denominator weight | positive weight |
|---|---|---|---|
| positive | `syn(v \| L_a)` high | 0 | `syn` |
| structural negative | `v ∈ inv(L_a)` | 1 | 0 |
| explicit negative | `v ∈ N_a` | 1 | 0 |
| in-vocabulary negative | `v ∈ C_s \ L_a` | `e_s · (1 − k(v \| L_a, cats))` | 0 |
| unknown | `v ∉ C_s` | 0 (ignored) | 0 |

The shipped loss implements rows 1 to 3 exactly, row 4 with `e_s = 1`, and row
5 for the sources named by `--restrict_neg_sources`. The remaining gap is that
`e_s` is 1 for every source, so an unannotated in-vocabulary column of a
sparsely annotated corpus is pushed down at full weight.

The rule generalises without new hyperparameters: a source contributes
negatives only inside its own label space, scaled by how exhaustive it is
there, discounted by how often the column co-occurs with what was annotated.

## The cross-pair term is not optional

The contrastive loss normalises over columns *within* a pair. It never trains
"is this pair's score for `on` above that pair's score for `on`" — and
cross-pair comparability is exactly what per-predicate AUC, a precision test
and any true-or-false judgement measure. The per-cell sigmoid auxiliary
(`--lambda_sigmoid`) is that term, and it obeys the same status table: a cell
counts as a labelled negative only under rows 2 to 4, and an unknown cell gets
a positive-unlabelled prior weight rather than a hard zero. Structurally this
is OWL-ViT's federated loss and the "positive label is all you need" position:
train on what is known, do not invent negatives.

## What a new source has to declare

- its label space `C_s`, or "open";
- how exhaustive it is within that space, with the estimator that produced the
  number;
- its negatives sidecar, if it adjudicated any;
- its provenance: human, model-proposed, or model-proposed and verified.

`--restrict_neg_sources` is the interim form of the label-space field. A
pseudo-label source is then a first-class citizen: its label space is the
candidate set it was asked about, its exhaustiveness is the verifier's recall,
and its per-cell confidence rides on the positive weight.

## Where this sits in the literature

- **Partial label spaces across datasets.** Detic's multi-dataset detection
  penalises only within each dataset's label space; UniDet and ScaleDet unify
  label spaces with hard and semantically soft assignment. The status table is
  that idea at the predicate level, plus the exhaustiveness term detectors do
  not need — boxes are exhaustive per image, relations are not.
- **Positive-unlabelled contrast.** puNCE and puCL reweight unlabelled columns
  as soft positive-negative mixtures; SPML treats unobserved labels as unknown.
  Row 5 and the prior above are those.
- **Negatives in scene-graph generation.** Graphical Contrastive Losses use
  typed hard negatives; Lang3DSG draws negatives from the label set and always
  includes "no relation"; ReLIC-SGG treats unannotated relations as latent
  variables to complete rather than as negatives; contradiction sets are this
  project's inverse mask, generalised to antonyms through the text student.
