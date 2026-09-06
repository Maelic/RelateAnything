# Quickstart

## Loading a model

There are two constructors, and the difference matters.

### `from_deploy` — a self-contained bundle

```python
from relsgg.api import RelateAnything

model = RelateAnything.from_deploy("relateanything_deploy.pt", device="cuda")
```

The predicate vocabulary is already baked into the head and the calibration is
already installed, so **no text encoder is needed** and inference is pure
vision. This is what you ship. Produced by
[`deploy/prepare_deploy_ckpt.py`](../deploy/prepare_deploy_ckpt.py).

### `from_checkpoint` — a training checkpoint

```python
from huggingface_hub import snapshot_download

d = snapshot_download("maelic/relsgg-vits16plus")      # model.pth + text_student.pt
model = RelateAnything.from_checkpoint(
    f"{d}/model.pth",
    predicates=["holding", "looking at", "leaning against"],
    device="cuda",
    weights="ema",          # "ema" (default, what ships) or "model"
)
```

Re-parameterizes on the fly, so it needs the text encoder the checkpoint was
trained with. It is read from the checkpoint's own `args` and found next to
`model.pth` as `text_student.pt` — you normally pass nothing. Released
checkpoints embed the backbone configuration, so no gated download happens. See [text space](#the-text-space-trap) if you are loading an old
checkpoint.

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
graphs["spatial"]     # layout: on, behind, to the left of, ...
graphs["semantic"]    # interaction: holding, riding, looking at, ...
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

Any string the CLIP BPE tokenizer can encode. This costs one text-encoder pass
and re-fuses the head; inference afterwards is unchanged in cost. It needs the
text encoder, so it works on a `from_checkpoint` model and on a
`from_deploy` bundle only if one was packed with it.

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
expected calibration error from ~0.92 to ~0.009 and transfers out of domain.
Because it is monotone, **every ranking metric is bit-identical**; only the
meaning of a threshold changes. Release bundles ship with the fit installed.

Fit one yourself with `benchmark/eval_deploy_metrics.py --fit_platt`.

## The text space trap

The head's predicate matrix `W` lives in the space of the text encoder the
checkpoint was trained with. Current checkpoints use a distilled 512-d student;
pre-v34 checkpoints used raw dino.txt at 2048-d. Mixing them does not raise an
error — it silently produces meaningless cosines. `from_checkpoint` reads the
right one out of the checkpoint's args, so let it.

## Where to go next

- Boxes from a real detector, and what that costs: [evaluation](evaluation.md#detector-boxes)
- Shipping this to a laptop: [deployment](deployment.md)
- Numbers that look right and are not: [pitfalls](pitfalls.md)
