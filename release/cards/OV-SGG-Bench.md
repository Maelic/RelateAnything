---
license: other
license_name: mixed-upstream-licenses
task_categories:
  - object-detection
language:
  - en
tags:
  - scene-graph-generation
  - visual-relationship-detection
  - open-vocabulary
  - benchmark
  - evaluation
pretty_name: OV-SGG-Bench
size_categories:
  - 10K<n<100K
---

# OV-SGG-Bench: evaluation packs for open-vocabulary relation prediction

The six evaluation sources of **OV-SGG-Bench**, the protocol of
**RelateAnything** ([code](https://github.com/Maelic/RelateAnything) · paper:
*RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any
Inputs*), packed in the memmap format the evaluators read, plus the
negative-set and cell files the scorers need and the training-side artifacts
the shipped recipe names. **No images are redistributed**: every pack stores
boxes, labels and file names, and the evaluator reads pixels from your copy of
the source dataset (set `RA_DATASETS`).

The protocol has six axes, chosen so that no single one can be won by matching
a benchmark's annotation prior. The definition, the sources, and the reasons
are in
[`benchmark/SPEC.md`](https://github.com/Maelic/RelateAnything/blob/main/benchmark/SPEC.md).

| pack | images | role | axis |
|---|---|---|---|
| `vg150` (test) | 26,404 | in-domain control (100 % vocabulary overlap with VG150-trained baselines) | A1, A3 |
| `psg` (test) | 2,179 | primary transfer, COCO images, 56 predicates | A1, A3, A4, A5 |
| `indoorvg` (test) | 4,403 | domain shift | A1, A3 |
| `hicodet` (test) | 9,490 | 116 external verbs; federated negatives (its `train` split, 33,415 images, is a training source) | A1, A2 |
| `haystack` | 11,368 | explicit negatives on SA-1B images (no image overlap with any training set) | A2 |
| `spatialsense`, `spatialsense_test`, `spatialsense_valid` | 1,920 test images, 2,758 balanced true/false triples | adversarial spatial understanding | A6 |

## Files

| path | contents |
|---|---|
| `packs/<name>/<split>/` | `meta.json`, `file_names.json`, `img_meta.npy`, `boxes.npy`, `box_cats.npy`, `rels.npy` |
| `datamix/haystack_negatives.json`, `datamix/hicodet_negatives*.json` | explicit negative cells for the A2 precision axis |
| `datamix/spatialsense_{test,valid}_cells.json` | the balanced true/false cells for A6 |
| `datamix/indoorvg_holdout.json` | image ids excluded from training (leakage audit) |
| `datamix/registry.json`, `vg2coco.json`, `psg2coco.json` | id registry used by the converters |
| `detect/yoloworld_ov_psg_test.npz` | the shared YOLO-World detections on PSG test that A4 and A5 score every system on |
| `text_student_v2_512/student.pt` | the distilled predicate text encoder every released checkpoint was trained with |
| `datamix_v22/pair_opportunity.npz`, `datamix_v22/text_space/*` | the soft-supervision and synonym artifacts the training recipe names (`train.sh`) |

## Use

Two commands from a clone of [RelateAnything](https://github.com/Maelic/RelateAnything):

```bash
pip install -e ".[train,hub]"
python benchmark/setup_data.py        # this dataset, the test images (~15 GB), every pack relocated and checked
python benchmark/run_benchmark.py --checkpoint maelic/relsgg-vits16plus
```

`setup_data.py --axes a1 a4 a6` downloads only the images those axes read. The images come
from their sources (VG150, PSG and IndoorVG from `maelic/*-coco-format`, HICO-DET from
`maelic/HICO-DET-coco-format`, Haystack from `maelic/Haystack-coco-format` and SpatialSense
from `maelic/SpatialSense-coco-format`, mirrors of the authors' releases under their original
licenses); none are redistributed in this dataset. Each pack's `meta.json` records `img_dir` as `datasets/<source>/<split>`,
which `setup_data.py` re-roots at `$RA_DATASETS`.

`run_benchmark.py` takes a Hugging Face model id or any checkpoint of the recipe and prints
OVS-F1 over A1, A2, A4 and A6. A5, the graph-quality axis, needs a VLM judge (vLLM,
`Qwen/Qwen3.6-27B`) and runs only with `--judge`; it adds the fifth axis of the paper's
composite. The protocol and the composite are defined in
[`benchmark/SPEC.md`](https://github.com/Maelic/RelateAnything/blob/main/benchmark/SPEC.md)
and [`docs/evaluation.md`](https://github.com/Maelic/RelateAnything/blob/main/docs/evaluation.md).

## Licenses

Each pack retains the license of its source: Visual Genome / VG150 and
IndoorVG (CC BY 4.0), PSG (COCO images, CC BY 4.0), HICO-DET (research use),
Haystack (SA-1B research license), SpatialSense (CC BY 4.0). The text student
and the `datamix_v22` artifacts are derivatives of Meta DINOv3 (dino.txt) and
follow the [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).

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
