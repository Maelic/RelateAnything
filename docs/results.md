# Paper results and benchmark context

This page collects the detailed model comparisons, evaluation results and
benchmark rationale. For installation and inference, start with the
[project README](../README.md).

## Model comparison

Three checkpoints, one recipe; the backbone is the only variable. All three
train on RA-4M + raw Visual Genome + a 5 % share of HICO-DET. Every number is
generated from measured evaluation files by
[`release/make_model_cards.py`](../release/make_model_cards.py).

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
the five-axis composite of the [headline table](#evaluation-results): that one adds A5 and
reads 40.1 for the released tower against 11.8 for the baseline. A composite is
comparable only between models scored on the same axis set. Each model card
lists the full per-benchmark numbers, deployment thresholds and provenance.

## Evaluation results

Every number below is **cross-dataset**: the tower is evaluated on benchmarks
that contributed no training image. The one exception is the HICO-DET row of
the released model, which takes a 5 % relation share of that training split and
is reported beside the zero-shot tower — the same recipe with no HICO-DET,
trained as an evaluation control and not published; its arguments are in
[`training/configs/`](../training/configs/). The baseline is
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

**Transfer with ground-truth boxes (released model)**
(`relsgg-vits16plus`, graph-constrained). The HICO-DET row includes the
5 % HICO-DET training share described above; it is not the zero-shot control:

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

The protocol: [`benchmark/SPEC.md`](../benchmark/SPEC.md). The argument and the
traps in scene-graph metrics: [docs/evaluation.md](evaluation.md). The
evaluation packs, negatives and calibration files:
[`maelic/OV-SGG-Bench`](https://huggingface.co/datasets/maelic/OV-SGG-Bench).

