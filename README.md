<div align="center">

# RelateAnything

**Real-time open-vocabulary relation prediction from any inputs.**<br>
53 M parameters · 20 ms per frame on an A40 · no object labels · predicate vocabulary given at inference.

[![arXiv](https://img.shields.io/badge/arXiv-2609.12552-b31b1b.svg)](https://arxiv.org/abs/2609.12552)
[![models](https://img.shields.io/badge/%F0%9F%A4%97%20models-relsgg--*-yellow.svg)](https://huggingface.co/collections/maelic/relateanything)
[![dataset](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-RA--4M-yellow.svg)](https://huggingface.co/datasets/maelic/RA-4M)
[![demo](https://img.shields.io/badge/demo-in%20your%20browser-brightgreen.svg)](https://maelic.github.io/RelateAnythingProject/demo/)
[![ci](https://github.com/Maelic/RelateAnything/actions/workflows/ci.yml/badge.svg)](https://github.com/Maelic/RelateAnything/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![license](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2609.12552) ·
[Project page](https://maelic.github.io/RelateAnythingProject) ·
[Browser demo](https://maelic.github.io/RelateAnythingProject/demo/) ·
[Models](https://huggingface.co/collections/maelic/relateanything) ·
[RA-4M](https://huggingface.co/datasets/maelic/RA-4M) ·
[OV-SGG-Bench](https://huggingface.co/datasets/maelic/OV-SGG-Bench) ·
[Docs](docs/)

</div>

https://github.com/user-attachments/assets/76544f31-53ac-407b-9445-157e83946938

<div align="center">

<sub>Five Creative Commons clips, six seconds each: instance masks from a 497-class segmenter, and the same relation model over every one of them. Boxes are tracked with a Kalman filter and each relation is held by a second filter on calibrated log-odds, so an edge survives the frames where one of its endpoints goes undetected. Rebuild it with <a href="deploy/render_video.py"><code>deploy/render_video.py</code></a>. Clip credits: <a href="assets/reel_video/credits.json"><code>assets/reel_video/credits.json</code></a>.</sub>

</div>

---

Give the model an image and a set of regions, from any detector, any
segmenter, or ground truth. It returns ranked relations between pairs of them:

```python
from relsgg import RelateAnything

model = RelateAnything.from_pretrained("maelic/relsgg-vits16plus")
model.predict(image, boxes)                       # [Triplet(sub=3, "riding", obj=7, 0.91), ...]
model.set_vocabulary(["about to collide with", "reflected in"])   # any strings, no retraining
```

Three properties make that work, and they are the point of the project:

- **The predicate vocabulary is supplied at inference.** Any phrase is a valid
  predicate. Changing the vocabulary costs one pass of a small text encoder;
  inference afterwards is pure vision, with no language model in the loop, and
  the model cannot output a relation outside the list you gave it.
- **Object class labels are never an input.** Pixels and regions in, relations
  out. Swap the detector, or use a class-agnostic segmenter, without touching
  the relation model.
- **One forward pass, one or two graphs.** Ask for a single ranked list, or for a
  **spatial** graph and a **semantic** graph at once, since a pair can hold a
  layout relation and an interaction at the same time.

The model runs at 20 ms per frame end to end on an A40 and as one ONNX graph on a
laptop CPU or [in the browser](https://maelic.github.io/RelateAnythingProject/demo/).

## Install

```bash
git clone https://github.com/Maelic/RelateAnything
cd RelateAnything
pip install -e ".[hub]"          # torch, transformers, huggingface_hub; Python 3.12+
```

Optional: `pip install -e ".[deploy]"` for ONNX Runtime and the laptop demo,
`".[dev]"` for the tests. Cluster and offline setups:
[docs/installation.md](docs/installation.md).

## Quickstart

```python
from relsgg import RelateAnything

model = RelateAnything.from_pretrained("maelic/relsgg-vits16plus", device="cuda")

triplets = model.predict(image, boxes_xyxy, topk=20)   # PIL or HWC array; boxes [N, 4] in pixels
for t in triplets:
    print(t)                                           # (person) --riding [0.91]--> (horse)

model.set_vocabulary(["tethered to", "grazing beside", "casting a shadow on"])
graphs = model.predict(image, boxes_xyxy, masks=masks, decompose=True)
graphs["spatial"], graphs["semantic"]                  # two graphs, one forward pass

# or answer from the whole training vocabulary, 19,103 strings, encoded once
# and shipped beside the weights
model = RelateAnything.from_pretrained("maelic/relsgg-vits16plus", full_vocabulary=True)
```

A released model carries its own backbone configuration and text encoder, so
nothing else is downloaded and no gated login is needed. Masks are optional;
boxes alone are the contract. Box sources, calibration, thresholds and
batching: [docs/quickstart.md](docs/quickstart.md).

## Models

Three checkpoints, one recipe; the backbone is the only variable. All three
train on RA-4M + raw Visual Genome + a 5 % share of HICO-DET. Every number is
generated from measured evaluation files by
[`release/make_model_cards.py`](release/make_model_cards.py).

| model | backbone | params | A40, batch 1 | img/s, batch 32 | OVS-F1 | HICO F1 |
|---|---|---|---|---|---|---|
| [`relsgg-vits16`](https://huggingface.co/maelic/relsgg-vits16) | DINOv3 ViT-S/16 | 46.1 M | 19.5 ms | 201 | 34.6 | 36.4 |
| [`relsgg-vits16plus`](https://huggingface.co/maelic/relsgg-vits16plus) ⭐ | DINOv3 ViT-S/16+ | 53.2 M | 20.0 ms | 188 | 37.2 | 37.1 |
| [`relsgg-vitb16`](https://huggingface.co/maelic/relsgg-vitb16) | DINOv3 ViT-B/16 | 113.8 M | 19.3 ms | 130 | 37.2 | 37.5 |

⭐ `relsgg-vits16plus` is the recommended default: it matches the ViT-B model's
composite at half the parameters, and the advantage ViT-B holds on individual
axes does not survive the deployment operating point. Latency is the relation
head alone, bf16, over the full 19,103-string vocabulary; at batch size 1 the
family is dispatch-bound, so it is flat across a 2.5× range of FLOPs and the
argument for the small tower is throughput. The OVS-F1 column is the
chance-corrected harmonic mean over **A1, A2, A4 and A6** — the axis set used
for the model ladder, since A5 costs one judge run per arm. It is therefore not
the five-axis composite of the [headline table](#results): that one adds A5 and
reads 40.1 for the released tower against 11.8 for the baseline. A composite is
comparable only between models scored on the same axis set. Each model card
lists the full per-benchmark numbers, deployment thresholds and provenance.

## Results

Every number below is **cross-dataset**: the tower is evaluated on benchmarks
that contributed no training image. The one exception is the HICO-DET row of
the released model, which takes a 5 % relation share of that training split and
is reported beside the zero-shot tower — the same recipe with no HICO-DET,
trained as an evaluation control and not published; its arguments are in
[`training/configs/`](training/configs/). The baseline is
[OvSGTR](https://github.com/gpt4vision/OvSGTR) pre-trained on MegaSG — the
corpus whose images we re-annotate, which makes it the closest control for
supervision quality — run through **our** evaluator on the same images and the
same vocabulary. It receives ground-truth object labels throughout; we never do.
It is also the only system we can run on every axis the composite spans, which
is why it carries the composite. On transfer the stronger baseline is
ROBIN-3B, a scene-graph model built on a 3B vision-language model: it leads
OvSGTR on F1@50 on all three benchmarks both were run on (19.6 / 27.9 / 22.7
against 16.5 / 13.5 / 20.2) and still trails us on both metrics everywhere.

**The six axes** (`relsgg-vits16plus`, one evaluator for both models):

| axis | measure | OvSGTR | RelateAnything |
|---|---|---|---|
| A1 transfer | VG150 F1@50 (mR@50), triplet mass 13 % | 16.5 (10.4) | **36.9 (28.2)** |
| | PSG F1@50 (mR@50), triplet mass 11 % | 13.5 (8.8) | **34.7 (30.6)** |
| | IndoorVG F1@50 (mR@50), triplet mass 7 % | 20.2 (12.8) | **37.8 (29.5)** |
| | HICO-DET F1@50 (mR@50), zero-shot tower | 7.9 (4.5) | **18.7 (12.7)** |
| A2 precision | Haystack mean fAP (rare fAP) | 52.1 (44.6) | **72.6 (70.7)** |
| A3 open vocabulary | mR@50 over 19,103 strings, VG150 / PSG / IndoorVG | not runnable | **34.5 / 28.3 / 34.6** |
| A4 deployment | PSG on a shared detector, wR@50 (mR@50) | 4.0 (6.2) | **20.0 (20.5)** |
| A5 graph quality | true bits per image (share of the annotation's) | 13.4 (0.48) | **18.6 (0.67)** |
| A6 spatial | SpatialSense macro AUC (pooled) | 59.1 (61.7) | **69.0 (67.5)** |
| **OVS composite** | harmonic mean of the chance-corrected axes | **11.8** | **40.1** |

A3 is reported but excluded from the composite: it cannot be run on OvSGTR,
whose vocabulary arrives as one caption capped at about 150 strings. Systems
that answer in free text can be scored there by construction, and against
ROBIN-3B on PSG the ordering depends on the matcher — it leads on exact strings
(20.0 mR@50 against our 13.0) and we lead under every synonym-tolerant one
(31.3 against 25.1), while micro recall never turns over. A single row there is
a choice of scorer rather than a measurement of a model, so the report prints
the band. What survives the matcher is which pairs a system proposes at all:
99.7 % of annotated pairs for us, 45.6–77.4 % for ROBIN, 23.1–35.5 % for
prompted general multimodal models.
A5 credits each relation a vision-language judge accepts with its surprisal
under the PSG training marginal, so a graph of five hundred `on` edges scores
nothing; the same judge returns two verdicts that favour the baseline, and both
are reported in the paper.

**End-to-end cost**, batch 1, median latency, eager PyTorch for both, whole
system including the detector:

| system | params | boxes/img | A40 | A100 | H100 | FPS (A40) |
|---|---|---|---|---|---|---|
| OvSGTR Swin-T | 177 M | 98 | 194.0 ms | 179.9 ms | 128.1 ms | 5.1 |
| OvSGTR Swin-B | 237 M | 98 | 228.5 ms | 195.1 ms | 134.3 ms | 4.3 |
| **RelateAnything + YOLO-World** | 231 M | 20 | **25.0 ms** | 35.0 ms | 25.6 ms | **40.0** |

7.8× end to end on an A40 while carrying more parameters than the Swin-T
baseline, because batch-1 cost is dispatch-bound rather than FLOP-bound. With
`torch.compile` the released tower reaches 20 ms per frame (49 FPS) on an A40;
on eight CPU threads through OpenVINO it reaches 7 FPS.

**Transfer on four test sets the model never trained on**
(`relsgg-vits16plus`, ground-truth boxes, graph-constrained):

| benchmark | R@50 | mR@50 | F1@50 | rare |
|---|---|---|---|---|
| VG150 | 0.533 | 0.282 | 0.369 | 0.427 |
| PSG | 0.401 | 0.306 | 0.347 | 0.239 |
| IndoorVG | 0.527 | 0.295 | 0.378 | 0.216 |
| HICO-DET | 0.452 | 0.314 | 0.371 | 0.239 |

Mean recall is 2.3–3.5× the baseline's and rare-bucket recall 5–21×, the ratio
being undefined on VG150 where the baseline scores exactly 0.0.

**Open vocabulary, no reparameterization**: all 19,103 training predicates stay
deployed and the model is never told the benchmark's label set; a prediction
counts when a synonym matcher accepts it at a calibrated threshold:

| benchmark | SoftR@50 | SoftmR@50 | SoftF1@50 |
|---|---|---|---|
| VG150 | 0.560 | 0.345 | 0.427 |
| PSG | 0.305 | 0.283 | 0.294 |
| IndoorVG | 0.533 | 0.346 | 0.419 |

## Why a new benchmark

Scene-graph benchmarks share their predicate vocabulary with the corpora models
train on. VG150's test split uses the same 50 predicate strings as its training
split, so a VG150-trained model faces no vocabulary novelty at all, and micro
recall tracks that agreement and nothing else. A counting baseline over
ground-truth object-category pairs, using no pixels, beats a trained model on
the most reported metric while losing to it by a wide margin per predicate:

| benchmark | edges | freq, micro | ours, micro | freq, macro | ours, macro |
|---|---|---|---|---|---|
| VG150 | 152,535 | **68.4** | 57.7 (−15.6 %) | 18.9 | **35.1** (+85.7 %) |
| PSG | 13,623 | **50.9** | 43.3 (−15.0 %) | 20.7 | **31.6** (+52.4 %) |
| IndoorVG | 29,175 | **67.9** | 57.3 (−15.5 %) | 29.8 | **38.4** (+28.9 %) |

The lookup table receives oracle object categories we never see, and the join
finds our model correct where it is wrong on 6.9–12.8 % of edges, so this is
not an argument that pixels are unnecessary. It is the narrower one: a metric a
pixel-free table can win does not measure relation understanding, and it is the
metric that orders leaderboards.

The second prior is what a benchmark shares with the corpus a model trained
on — and it is not the vocabulary. A predicate string is not an annotation:
`on` between a person and a horse and `on` between a book and a table are
different acts, so two corpora can agree on the string while never agreeing on
the pair it is asserted of. *Shared triplet mass* is the share of a training
corpus's relation instances whose ⟨subject category, predicate, object
category⟩ triple the benchmark also annotates:

| training corpus | matched on | VG150 | PSG | IndoorVG | Haystack |
|---|---|---|---|---|---|
| the released mixture (19,103 predicates) | predicate string | 53.4 % | 38.7 % | 49.5 % | 38.7 % |
| | both object categories | 44.4 % | 36.7 % | 26.8 % | 30.7 % |
| | **the whole triple** | **12.8 %** | **10.7 %** | **6.6 %** | **4.9 %** |
| VG150 train (typical baseline, 50 predicates) | predicate string | **100.0 %** | 57.3 % | **95.7 %** | 57.3 % |
| | both object categories | 100.0 % | 22.1 % | 12.3 % | 7.1 % |
| | **the whole triple** | **90.9 %** | 8.6 % | 10.4 % | 0.6 % |

The confound is concentrated in-domain and is very large there: on VG150 the
baseline's fine-tuning corpus reproduces 90.9 % of its relation mass as triples
the benchmark also annotates, against our 12.8 %. Micro recall tracks this
statistic and the tail metrics do not: an arm trained with a larger share of
raw Visual Genome reached 54.3 R@50 on VG150, the best zero-shot figure we are
aware of, while being the worst model we trained on every tail metric.

### OV-SGG-Bench

So this repository ships a protocol as well as a model. Six axes, chosen so
that no single one can be won by matching a benchmark's prior:

| axis | question | source |
|---|---|---|
| A1 transfer | generalises across annotation styles? | VG150, PSG, IndoorVG, HICO-DET, ground-truth boxes, four sources of differing shared triplet mass |
| A2 precision | hallucinates rare predicates? | Haystack's explicit negatives: 2,870 positives against 23,174 adjudicated negatives |
| A3 open vocabulary | means the right relation, without the label set? | all 19,103 strings, synonym matcher at a calibrated threshold |
| A4 deployment | survives a real detector? | SGDet on a shared open-vocabulary detector, against its measured pair-recall ceiling |
| A5 graph quality | is the graph true *and* informative? | a vision-language judge, one relation at a time, no ground truth in the prompt; each accepted relation credited with its surprisal |
| A6 spatial | understands space, or co-occurrence? | SpatialSense adversarial true/false pairs |

The composite over A1, A2, A4, A5 and A6 is chance-corrected and combined by a
harmonic mean, so a weak axis cannot be averaged away; withholding each axis in
turn leaves the ordering of the two systems unchanged, at ratios between 2.0
and 3.7×. Never select a model on it.

The protocol: [`benchmark/SPEC.md`](benchmark/SPEC.md). The argument and the
traps in scene-graph metrics: [docs/evaluation.md](docs/evaluation.md). The
evaluation packs, negatives and calibration files:
[`maelic/OV-SGG-Bench`](https://huggingface.co/datasets/maelic/OV-SGG-Bench).

## RA-4M

The training corpus: **474,413 images, 4,282,531 relations, 10,102 distinct
free-text predicates**, generated by a vision-language model against drawn,
numbered boxes and filtered by a deterministic geometric check. Images are
MegaSG's ([JosephZ/mega_1m](https://huggingface.co/datasets/JosephZ/mega_1m))
and are referenced by identifier only; the annotations are ours. Synonyms are
never collapsed, since surface-form diversity is part of the label space.

Download: [`maelic/RA-4M`](https://huggingface.co/datasets/maelic/RA-4M).
Pipeline: [`datagen/`](datagen/). Format and packs: [docs/data.md](docs/data.md).

## Run the demo

**In your browser**, fully client-side (ONNX Runtime Web, WebGPU or WASM), no
install: **<https://maelic.github.io/RelateAnythingProject/demo/>**.

**On your laptop**, CPU only, no torch: a detector, the relation head and the
decode with `numpy` and `onnxruntime`:

```bash
pip install -r deploy/dist/requirements.txt
python deploy/demo_webcam.py --dist deploy/dist/relsgg-vits16plus                    # webcam
python deploy/demo_webcam.py --dist deploy/dist/relsgg-vits16plus --image photo.jpg  # one image
python deploy/demo_webcam.py --dist deploy/dist/relsgg-vits16plus --decompose        # two graphs
```

The relation graph comes with the model repository. Detector weights are not
redistributed (AGPL upstream); [`deploy/README.md`](deploy/README.md) gives the
two-command local rebuild, and the ONNX and OpenVINO export recipe.

## How it works

<div align="center"><img src="assets/pipeline.svg" alt="Architecture: a DINOv3 backbone reads the image once; boxes become query tokens; a relation transformer scores pairs against a predicate matrix produced by a text encoder from the vocabulary supplied at inference" width="820"></div>

A frozen-config DINOv3 backbone reads the image once. Each region becomes a
token from its box (or mask) geometry and pooled features; a pair sampler keeps
the pairs worth scoring; a relation transformer attends over pairs and patches;
and a vocabulary head scores each pair against a matrix of predicate
embeddings. That matrix is produced from your strings by a distilled text
encoder, so it can be replaced at any time. Module by module:
[docs/architecture.md](docs/architecture.md).

## Reproduce the paper

| what | where |
|---|---|
| train a released model | [`train.sh`](train.sh), the recipe in [docs/training.md](docs/training.md), the exact arguments in [`training/configs/`](training/configs/) |
| build the packs, the vocabulary, the text student | [`training/`](training/) |
| evaluate on the six axes, run the OvSGTR baseline through the same scorer | [`benchmark/`](benchmark/) |
| generate RA-4M | [`datagen/`](datagen/) |
| the appendix probes (attribution, counterfactuals, text space, priors) | [`research/`](research/) |
| latency, FLOPs and deployment cost tables | `deploy/bench_*.py`, `benchmark/latency.py`, `benchmark/model_cost.py` |

Before you trust a number, read [docs/pitfalls.md](docs/pitfalls.md): every
entry there has produced a plausible wrong result at least once.

## Repository map

| path | what |
|---|---|
| [`relsgg/`](relsgg/) | the package: `api.py` and `config.py` at the top, then `model/` (backbone, geometry, pooling, sampler, transformer, deformable read, vocabulary head), `text/` (the predicate encoder), `data/` (packs and mixtures), `training/` (objective, loop), `eval/` (evaluators) |
| [`train.py`](train.py), [`train.sh`](train.sh) | training entry point and the released recipe |
| [`training/`](training/) | pack builders, converters, vocabulary and soft-supervision builders, text-student distillation, released configs |
| [`benchmark/`](benchmark/) | OV-SGG-Bench: the specification, every scorer and entry point, the baseline adapter |
| [`deploy/`](deploy/) | ONNX and OpenVINO export, bundles, calibration, the laptop and Gradio demos, cost benchmarks |
| [`datagen/`](datagen/) | the RA-4M generation pipeline and its prompts |
| [`research/`](research/) | probes behind the paper's analysis sections |
| [`release/`](release/) | checkpoint stripping, model and dataset cards, Hugging Face upload |
| [`docs/`](docs/) | installation, quickstart, architecture, data, training, evaluation, deployment, the objective, pitfalls |
| [`tests/`](tests/) | CPU-only tests, run by CI on Python 3.12 and 3.13 |

## Documentation

| page | read it when |
|---|---|
| [Installation](docs/installation.md) | setting up, downloading weights and packs, running offline or on a cluster |
| [Quickstart](docs/quickstart.md) | you have a checkpoint and want triplets out of it |
| [Architecture](docs/architecture.md) | you want to know what happens between pixels and triplets |
| [Data](docs/data.md) | RA-4M, the pack format, mixtures, adding a source |
| [Training](docs/training.md) | reproducing a released model or running an ablation |
| [Evaluation](docs/evaluation.md) | reporting a number or comparing against another method |
| [Deployment](docs/deployment.md) | exporting to ONNX or OpenVINO, picking thresholds |
| [Pitfalls](docs/pitfalls.md) | before you trust a number |
| [The objective](docs/objective.md) | adding a dataset or a loss term |
| [Contributing](CONTRIBUTING.md) | opening a pull request |

## License

- **Code**: [Apache-2.0](LICENSE).
- **Weights**: derivatives of Meta DINOv3, distributed under the
  [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- **RA-4M annotations**: generated by Gemma, distributed with the
  [Gemma Terms of Use](https://ai.google.dev/gemma/terms) notice. Images are not
  redistributed.
- **Demo detectors**: derive from ultralytics (AGPL-3.0) and are rebuilt locally,
  never shipped in a release artifact.

Every upstream credit and the exact terms: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@article{neau2026relateanything,
  title   = {RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs},
  author  = {Neau, Ma\"elic},
  journal = {arXiv preprint arXiv:2609.12552},
  eprint  = {2609.12552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url     = {https://arxiv.org/abs/2609.12552},
  year    = {2026}
}
```
