# Quickstart

## Loading a model

```python
from relsgg import RelateAnything

model = RelateAnything.from_pretrained("maelic/relsgg-vits16plus", device="cuda")
```

That downloads the model and loads it. `from_checkpoint` does the same for a
file you already have:

```python
model = RelateAnything.from_checkpoint(
    "model.pth",
    predicates=["holding", "looking at", "leaning against"],   # optional
    device="cuda",
)
```

Both encode the predicate vocabulary once with the text student that ships
beside `model.pth`, then run vision only. A released model also embeds its
backbone's configuration, so nothing is downloaded from the gated DINOv3
repositories, and its calibration is installed when `calibration.json` sits
next to the weights. Without `predicates` the vocabulary is the one in
[`relsgg/vocabulary.py`](../relsgg/vocabulary.py).

To answer from the whole training vocabulary instead — 19,103 strings for the
released towers — ask for it:

```python
model = RelateAnything.from_pretrained("maelic/relsgg-vits16plus", full_vocabulary=True)
```

The strings come from the checkpoint, and their embeddings from the
`predicate_embeddings.npz` published beside it, which is what the text student
produces for them; encoding that many strings takes about a minute and a half
on a CPU otherwise. Expect free-text answers the benchmark vocabularies do not
contain (`wearing clothing`, `riding on`), which is the point of the axis and
what exact-string recall penalises.

## Predicting

```python
triplets = model.predict(
    image,                  # PIL.Image, or HWC numpy (OpenCV BGR is fine)
    boxes_xyxy,             # [N, 4] float pixels, ORIGINAL image frame
    box_labels=names,       # optional, display only — never fed to the model
    box_scores=confs,       # optional detector confidences
    topk=20,
    max_boxes=60,
)

for t in triplets:
    print(t)                # (person) --riding [0.91]--> (horse)
    t.subject_idx, t.subject_box, t.predicate, t.score, t.object_idx, t.object_box
```

Two things to know about the arguments:

- **`box_labels` is cosmetic.** The model never receives object class labels.
  Passing them changes nothing about the prediction; they only make the
  `__repr__` readable.
- **`box_scores` changes the ranking**, not the scores. When given, triplets
  are ranked by `conf(sub) · conf(obj) · pred_score` — the SGDet convention,
  which suppresses pairs built on low-confidence boxes. Leave it out for
  ground-truth boxes.

`max_boxes` caps how many boxes reach the head (top-scoring first if scores are
given). Cost is quadratic in box count before the sampler prunes, so this is
the knob that keeps a crowded frame bounded.

## Two graphs from one pass

```python
graphs = model.predict(image, boxes_xyxy, decompose=True)
graphs["spatial"]     # layout: on, behind, to the left of,...
graphs["semantic"]    # interaction: holding, riding, looking at,...
```

One forward pass. The vocabulary columns are partitioned by predicate type; the
other type is masked to `-inf` and each stream is ranked independently, one
argmax edge per pair. **A pair can appear in both graphs** — holding a layout
relation and an interaction simultaneously — and that coexistence is the point
of the feature, not a bug.

How a predicate gets its type is a hybrid rule with measured reasons behind it,
documented in [`relsgg/decompose.py`](../relsgg/decompose.py): the training
corpus's flag when the string is known, the checkpoint's own spatialness gate
otherwise. Neither alone is sufficient — the corpus flag is
provenance-contaminated (`on` reads 0.985 spatial, its synonym `resting on`
reads 0.001) and the gate under-routes predicates it has never seen.

## Changing the vocabulary

```python
model.set_vocabulary(["about to collide with", "reflected in", "queuing behind"])
```

Any string at all: the text student's token table is the full CLIP byte-pair
vocabulary, so every subword has a row. This costs one pass of the text encoder
and re-fuses the head; inference afterwards costs the same as before.

Two consequences that are easy to forget:

- **Thresholds do not survive a vocabulary change.** Calibrated per-predicate
  thresholds are fitted per predicate *and* per checkpoint. New strings have
  none.
- **Synonyms compete.** The project never collapses synonyms — `riding` and
  `riding on` are separate columns and will split the ranking between them. If
  you supply both, expect both to appear at lower individual scores.

## Scores, thresholds and calibration

The one score definition is [`relsgg/scoring.py`](../relsgg/scoring.py), shared
by the evaluator, the torch API and the ONNX path:

```
score = sigmoid(a · (pred_logit + w · pair_logit) + b)
```

- `w` (**pair_weight**) is the relatedness fusion. `1.0` is the trained
  default. `0.0` drops the relatedness term — which *lowers* recall on
  annotation-derived benchmarks and *raises* accuracy on adjudicated negatives,
  because relatedness is partly a prior over which pairs a human bothered to
  annotate. Choose it according to which of those you care about.
- `(a, b)` is the deployment calibration:

```python
model.set_calibration(a, b)     # monotone for a > 0
```

Raw head scores pile into `[0.9, 1.0)` — the output head is trained against a
balanced prior while a real frame is 0.2–4 % positive — so an uncalibrated
threshold is a knob connected to nothing. A two-parameter Platt fit moves
expected calibration error from 0.176 to 0.004 when it is fitted on annotation
hits and scored against them. What the fit is made against decides what the
number means: the map the release ships is fitted on **adjudicated negatives**,
so its score estimates the probability that a person would call the relation
true, and it is conservative in the useful direction — 0.77 reported precision
where the adjudicated precision is 0.90. Because the fit is monotone, **every
ranking metric is bit-identical**; only the meaning of a threshold changes.

Fit one yourself with `benchmark/eval_deploy_metrics.py --fit_platt`.

## Masks instead of boxes

```python
triplets = model.predict(image, boxes_xyxy, masks=masks)   # masks: [N, H, W] bool
```

A mask is rasterised to the same coverage grid a box is, so the two are one
code path: the geometry features read region overlap instead of box overlap,
and a mask that fills its box reproduces the box result exactly. The model is
trained on boxes and accepts masks at inference — training on masks measured
worse under the box contract, while masks given at evaluation help.

## The text space

The head's predicate matrix lives in the space of the text encoder that trained
it. Encoding a vocabulary with a different encoder raises no error and produces
meaningless cosines, so the loaders read the right one out of the checkpoint
and find it next to `model.pth`. Pass `text_student=` only to override that
deliberately.

## Where to go next

- Boxes from a real detector, and what that costs: [evaluation](evaluation.md)
- Shipping this to a laptop: [deployment](deployment.md)
- Numbers that look right and are not: [pitfalls](pitfalls.md)
