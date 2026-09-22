<div align="center">

# RelateAnything

**Real-time open-vocabulary relation prediction from any inputs.**

[![arXiv](https://img.shields.io/badge/arXiv-2609.12552-b31b1b.svg)](https://arxiv.org/abs/2609.12552)
[![models](https://img.shields.io/badge/%F0%9F%A4%97%20models-relsgg--*-yellow.svg)](https://huggingface.co/collections/maelic/relateanything)
[![demo](https://img.shields.io/badge/demo-try%20in%20your%20browser-brightgreen.svg)](https://maelic.github.io/RelateAnythingProject/demo/)
[![ci](https://github.com/Maelic/RelateAnything/actions/workflows/ci.yml/badge.svg)](https://github.com/Maelic/RelateAnything/actions/workflows/ci.yml)
[![license](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSE)

[Paper](https://arxiv.org/abs/2609.12552) ·
[Project page](https://maelic.github.io/RelateAnythingProject) ·
[Models](https://huggingface.co/collections/maelic/relateanything) ·
[Training data](https://huggingface.co/datasets/maelic/RA-4M) ·
[Documentation](docs/)

</div>

Give RelateAnything **an image, object regions, and the relations you want to
look for**. It returns scored `(subject, relation, object)` triplets, such as
`person → riding → horse`. The recommended model has 53 M parameters.

- **Use your own regions.** Boxes can come from a detector, a segmenter, or a
  person. Masks are optional. Object class labels are never fed to the model.
- **Choose the relation vocabulary.** Supply phrases at inference time without
  retraining. A small text encoder embeds them once; no language model runs
  per frame.
- **Get one graph or two.** Return a single ranked list, or separate spatial
  and semantic graphs from the same forward pass.

https://github.com/user-attachments/assets/76544f31-53ac-407b-9445-157e83946938

<p align="center"><sub>The same relation model across five scenes, with tracked regions and relations. <a href="assets/reel_video/credits.json">Clip credits</a> · <a href="deploy/README.md#8-the-video-reel-render_videopy">Reproduce the video</a></sub></p>

**[Try the browser demo](https://maelic.github.io/RelateAnythingProject/demo/)** —
no installation; inference runs on your device using WebGPU or WASM.

[Quickstart](#quickstart) · [Models](#models) · [Deployment](#deployment) ·
[Performance](#performance) · [Results](#results) · [How it works](#how-it-works)

## Quickstart

Use **Python 3.12+**. Install a CUDA-enabled PyTorch build for NVIDIA GPU
inference, or use `device="cpu"` in the example below.

```bash
git clone https://github.com/Maelic/RelateAnything.git
cd RelateAnything
pip install -e ".[hub]"
```

Run this from the repository root. The sample photo is included; the two
example boxes identify the person and the horse.

```python
import numpy as np
from PIL import Image
from relsgg import RelateAnything

model = RelateAnything.from_pretrained("maelic/relsgg-vits16plus", device="cuda")
image = Image.open("assets/reel/images/horse.jpg").convert("RGB")
boxes = np.array([[470, 130, 650, 630], [90, 310, 1010, 875]], dtype=np.float32)

# Boxes are [x1, y1, x2, y2] in original-image pixels.
# Labels are optional and only make the printed triplets easier to read.
for triplet in model.predict(image, boxes, box_labels=["person", "horse"], topk=5):
    print(triplet)

# Change what relations to look for, without retraining.
model.set_vocabulary(["riding", "carrying a rider", "beside", "in front of"])
graphs = model.predict(image, boxes, box_labels=["person", "horse"], decompose=True)
print(graphs["spatial"])
print(graphs["semantic"])
```

Released checkpoints include the backbone configuration and text encoder;
**no gated DINOv3 login is needed for inference**. Replace the sample image and
boxes with your own inputs. You can also pass masks or request the full
19,103-predicate training vocabulary with `full_vocabulary=True` when loading.

[API guide: inputs, vocabularies, masks, and scores](docs/quickstart.md) ·
[Installation and offline use](docs/installation.md)

## Models

Start with **`relsgg-vits16plus`**. It matches the larger model's four-axis
composite score at roughly half the parameters. All three checkpoints use the
same training recipe; their vision backbones differ.

| Checkpoint | Backbone | Parameters |
|---|---|---:|
| [`relsgg-vits16`](https://huggingface.co/maelic/relsgg-vits16) | DINOv3 ViT-S/16 | 46.1 M |
| **[`relsgg-vits16plus`](https://huggingface.co/maelic/relsgg-vits16plus)** | **DINOv3 ViT-S/16+** | **53.2 M** |
| [`relsgg-vitb16`](https://huggingface.co/maelic/relsgg-vitb16) | DINOv3 ViT-B/16 | 113.8 M |

Each linked model card includes evaluation results, calibration and provenance.
The [full model comparison](docs/results.md#model-comparison) explains the
composite and reports throughput.

## Deployment

- **PyTorch:** use the Python API above on CPU or CUDA.
- **NVIDIA GPU / TensorRT 10:** build a local FP32 engine from the released
  ONNX graph. Vocabulary selection, calibration and two-graph decoding stay
  available. [Setup and examples](deploy/README.md#tensorrt-nvidia-gpu).
- **Laptop / ONNX or OpenVINO:** run the local image or webcam demo.
  [Setup](deploy/README.md) · [OpenVINO](docs/deployment.md#laptop-cpu-openvino).
- **Browser:** [try the demo](https://maelic.github.io/RelateAnythingProject/demo/)
  without installing Python.

After downloading the relation graph and building the local detector as
explained in the [deployment guide](deploy/README.md):

```bash
# CPU image demo; omit --image to use a webcam.
python deploy/demo_webcam.py --dist deploy/dist/relsgg-vits16plus --image photo.jpg

# NVIDIA GPU, after building both TensorRT engines.
python deploy/demo_webcam.py --dist deploy/dist/relsgg-vits16plus \
  --backend tensorrt --device cuda --image photo.jpg --decompose
```

The relation model consumes regions; it does not detect objects itself.
Demo detectors are rebuilt locally and have separate licenses. The exported
ONNX/TensorRT graph scores boxes; native mask inputs are supported by the
PyTorch API. [Deployment contracts and limitations](docs/deployment.md).

## Performance

### RTX 3080 Laptop GPU

<!-- BEGIN GENERATED LATENCY -->
Warm median latency on an **RTX 3080 Laptop GPU**
with an 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz. All three released checkpoints use the same inputs,
vocabularies and host-side decoding.

Observed GPU power limit: **55.00 W**.
**Thermal limiting was recorded during this run.** The detailed report retains
the earlier pass to show how the laptop's operating state affects latency.

**35 predicates — default deployment vocabulary**

| Backend | ViT-S | ViT-S+ | ViT-B |
|---|---:|---:|---:|
| PyTorch eager FP32 | 33.2 ms | 39.2 ms | 76.2 ms |
| PyTorch eager BF16 | **23.9 ms** | **21.7 ms** | **28.3 ms** |
| ONNX CUDA FP32 | 36.7 ms | 43.0 ms | 83.8 ms |
| TensorRT FP32 | 27.1 ms | 31.3 ms | 69.5 ms |

**243 predicates — complete bundled predicate bank**

| Backend | ViT-S | ViT-S+ | ViT-B |
|---|---:|---:|---:|
| PyTorch eager FP32 | 33.4 ms | 38.9 ms | 75.3 ms |
| PyTorch eager BF16 | **23.9 ms** | **22.3 ms** | **28.4 ms** |
| ONNX CUDA FP32 | 37.1 ms | 43.0 ms | 82.0 ms |
| TensorRT FP32 | 27.3 ms | 31.2 ms | 69.3 ms |

At 35 predicates, TensorRT reduces median latency versus eager PyTorch FP32
by **18% (ViT-S)**, **20% (ViT-S+)**, **9% (ViT-B)**.

Batch 1, 448 × 448 input, 20 regions padded to 32, 128 candidate pairs.
Includes preprocessing, CPU/GPU transfers and decoding; **excludes the
detector, model loading and engine building**. Each configuration uses
180 timed calls across 3 shuffled rounds after warmup, over 6 images
with generated boxes.
FP32 runs have TF32 disabled; `torch.compile` was not timed. Bold marks the
lowest median in each column; BF16 accuracy was not evaluated here.

[p95 latency, validation and reproduction](docs/benchmarks/README.md) ·
[Raw samples](docs/benchmarks/rtx3080-laptop-family.json) ·
[TensorRT setup](deploy/README.md#tensorrt-nvidia-gpu). These are local
deployment measurements, not a dataset-wide accuracy evaluation.
<!-- END GENERATED LATENCY -->

### End-to-end pipeline in the paper

With **YOLO-World detection included**, the compiled PyTorch pipeline reaches
**20.3 ms/frame (49 FPS) on an A40**. This uses a different GPU and includes
detection, so it is separate from the laptop relation-only measurements above.

[Pipeline measurements and deployment choices](docs/deployment.md#realtime-pipeline) ·
[Full comparison with OvSGTR](docs/results.md#evaluation-results)

## Results

The paper evaluates transfer, precision, open vocabulary, detection-mode
prediction, graph quality and spatial reasoning. Selected transfer results for
`relsgg-vits16plus`, using ground-truth boxes and one relation per ordered pair:

| Test set | OvSGTR F1@50 | RelateAnything F1@50 |
|---|---:|---:|
| VG150 | 16.5 | **36.9** |
| PSG | 13.5 | **34.7** |
| IndoorVG | 20.2 | **37.8** |

Both models use the same evaluator and vocabulary. OvSGTR receives ground-truth
object labels; RelateAnything does not. ROBIN-3B is a stronger transfer baseline
than OvSGTR and is also compared in the [paper](https://arxiv.org/abs/2609.12552).

The five-axis OV-SGG-Bench composite is **40.1 vs. 11.8** for OvSGTR.
It covers transfer, precision, deployment, graph quality and spatial reasoning;
open-vocabulary performance is reported separately because OvSGTR cannot run
that protocol. HICO-DET results distinguish the released model, which uses a
5% HICO-DET training share, from a separate zero-shot control.

[Detailed results and benchmark rationale](docs/results.md) ·
[Evaluation protocol](benchmark/SPEC.md) · [Scoring guide](docs/evaluation.md)

## How it works

<a href="assets/architecture.png"><img src="assets/architecture.png" alt="Figure 2: RelateAnything's visual path, text embedding bank, predicate-conditioned scoring gate, and training objective" width="1000"></a>

*Figure 2 from the [paper](https://arxiv.org/abs/2609.12552). Click to enlarge.*

A DINOv3 backbone reads the image once. Regions become visual tokens; a pair
sampler selects candidate relations; a transformer refines them using the
scene. A vocabulary head scores each pair against embeddings of your predicate
strings, mixing spatial and semantic features. Swapping that embedding bank
changes the vocabulary without retraining the vision model.

[Architecture guide](docs/architecture.md)

## Data and evaluation packs

- **[RA-4M](https://huggingface.co/datasets/maelic/RA-4M):** 474,413 images
  referenced by ID, 4,282,531 relation annotations and 10,102 distinct free-text
  predicates. Images are not redistributed. [Format and generation](docs/data.md).
- **[OV-SGG-Bench](https://huggingface.co/datasets/maelic/OV-SGG-Bench):** evaluation
  packs, explicit negatives and calibration files for six complementary axes.
  [Protocol](benchmark/SPEC.md).

## Reproduce and contribute

- **Train:** [training guide](docs/training.md), [released configs](training/configs/)
  and [`train.sh`](train.sh).
- **Evaluate:** [evaluation guide](docs/evaluation.md) and [`benchmark/`](benchmark/).
- **Generate annotations:** [`datagen/`](datagen/). **Explore the paper's probes:**
  [`research/`](research/).
- **Contribute:** [contributor guide](CONTRIBUTING.md). Bug reports, deployment
  feedback and examples of new uses are welcome in
  [GitHub issues](https://github.com/Maelic/RelateAnything/issues). Include the
  model, runtime, hardware and a small reproducer when reporting a problem.

The [documentation index](docs/) covers the API, architecture, training, data
and deployment. Read [evaluation pitfalls](docs/pitfalls.md) before comparing
results or changing the scoring path.

## License

- **Code:** [Apache-2.0](LICENSE).
- **Model weights:** derivatives of Meta DINOv3; see the model cards and
  [DINOv3 license](https://ai.meta.com/resources/models-and-libraries/dinov3-license/).
- **RA-4M annotations:** carry the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
  notice. Source images are not redistributed.
- **Demo detectors:** ultralytics-derived components use AGPL-3.0 and are
  excluded from the relation model's release artifacts.

[Full third-party notices and credits](THIRD_PARTY_NOTICES.md)

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
