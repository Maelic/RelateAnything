# RelateAnything documentation

Start here. Each page is self-contained; the order below is the order that
makes sense on a first read.

## Using the model

| page | read it when |
|---|---|
| [Installation](installation.md) | setting up an environment, hitting gated DINOv3 weights, or running offline / on a cluster |
| [Quickstart](quickstart.md) | you have a checkpoint and want triplets out of it |
| [Deployment](deployment.md) | shipping to ONNX / OpenVINO / a laptop, or picking thresholds |

## Understanding the model

| page | read it when |
|---|---|
| [Architecture](architecture.md) | you want to know what happens between pixels and triplets |
| [Data](data.md) | you are adding a dataset, building a pack, or wondering what RA-4M contains |
| [Evaluation](evaluation.md) | you are reporting a number, or comparing against another method |
| [Pitfalls](pitfalls.md) | **read before you trust a number.** Every entry here has produced a plausible wrong result at least once |

## Changing the model

| page | read it when |
|---|---|
| [Training](training.md) | reproducing the release, or running an ablation |
| [Contributing](../CONTRIBUTING.md) | opening a PR |
| [The objective](objective.md) | adding a dataset or a loss term: what a source knows, and what it may not be used to contradict |

## The short version

RelateAnything is a **box-conditioned, open-vocabulary relation head**. It takes
an image and boxes — from any detector or from ground truth — and returns ranked
`(subject, predicate, object)` triplets. It never receives object class labels,
and its predicate vocabulary is a runtime input rather than a trained weight
matrix.

The project has two halves, and they are separable:

1. **A model.** Three released sizes, one recipe. See [architecture](architecture.md)
   and [training](training.md).
2. **A claim about measurement.** Scene-graph recall largely measures how well a
   model's training vocabulary overlaps a benchmark's, so we built a protocol
   that cannot be won that way. See [evaluation](evaluation.md) and
   [`benchmark/SPEC.md`](../benchmark/SPEC.md).

If you only read one page beyond this index, read [pitfalls](pitfalls.md).
