---
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
tags:
  - scene-graph-generation
  - open-vocabulary
  - visual-relationship-detection
library_name: relsgg
model-index:
  - name: relsgg-vits16
    results:
      - task:
          type: scene-graph-generation
        dataset:
          type: vg150
          name: Visual Genome 150 (test)
        metrics:
          - type: F1@50
            value: 0.3591
            name: 'F1@50 (vg150 test, graph-constrained)'
---
# relsgg-vits16

Open-vocabulary relation prediction from any boxes or masks. Give the model an
image and regions from any source (a detector, a segmenter, ground truth); it
returns ranked relations over a predicate vocabulary supplied at inference,
and optionally two graphs (spatial + semantic) from the same forward pass.
Object class labels are never an input.

Part of **RelateAnything** ([code](https://github.com/Maelic/RelateAnything) · paper: *RelateAnything: Real-Time
Open-Vocabulary Relation Prediction From Any Inputs*). Trained on
[RA-4M](https://huggingface.co/datasets/maelic/RA-4M); evaluated with
[OV-SGG-Bench](https://huggingface.co/datasets/maelic/OV-SGG-Bench).

## Use it

```bash
pip install git+https://github.com/Maelic/RelateAnything
hf download maelic/relsgg-vits16          # optional; the API fetches on first use
```

```python
from relsgg import RelateAnything

# Regions come from any detector, any segmenter, or your own annotation.
# Object class labels are never an input.
model = RelateAnything.from_pretrained("maelic/relsgg-vits16", device="cuda")
for t in model.predict(image, boxes_xyxy, topk=20):    # PIL/ndarray, boxes [N, 4] in pixels
    print(t)                                           # (person) --riding [0.67]--> (horse)

# Masks instead of boxes: pass the [N, H, W] binary masks beside their extents.
triplets = model.predict(image, boxes_xyxy, masks=masks, topk=20)

# The vocabulary is an input. Any strings, at any time, without retraining.
model.set_vocabulary(["about to collide with", "reflected in"])

# Or answer from the whole training vocabulary, 19,103 strings, read from the weights.
model = RelateAnything.from_pretrained("maelic/relsgg-vits16", full_vocabulary=True, device="cuda")

# Two graphs from one forward pass.
graphs = model.predict(image, boxes_xyxy, decompose=True)   # {"spatial": [...], "semantic": [...]}
```

Every vocabulary is encoded once by the text student shipped beside the
weights, and the head is reparameterized onto it; scoring afterwards is vision
only. `full_vocabulary=True` reads `predicate_embeddings.npz` instead of
encoding, which turns a minute and a half of CPU work into a download.
`model.pth` embeds the backbone configuration, so running these weights needs
no gated DINOv3 login.

Files: `model.pth` (torch, EMA weights), `text_student.pt` + tokenizer,
`predicate_embeddings.npz` (the training vocabulary, encoded), `predicate_bank.npz`, `thresholds.json`, `calibration.json`, `README.md`.

**Every number below is generated from measured eval artifacts
(`release/make_model_cards.py`); none is hand-typed.**

## Closed-vocabulary transfer (reparameterized, TEST, graph-constrained)

| source | R@50 | mR@50 | F1@50 |
|---|---|---|---|
| vg150 | 0.525 | 0.273 | 0.359 |
| psg | 0.396 | 0.298 | 0.340 |
| indoorvg | 0.519 | 0.276 | 0.360 |
| hicodet | 0.453 | 0.305 | 0.365 |

## Open-vocabulary, NO reparameterization (all 19,103 predicates deployed)

Synonym-matched at the calibrated tau (see provenance). This is the honest
"the model never saw your label set" protocol.

| source | SoftR@50 | SoftmR@50 | SoftF1@50 |
|---|---|---|---|
| vg150 | 0.551 | 0.333 | 0.415 |
| psg | 0.303 | 0.268 | 0.284 |
| indoorvg | 0.528 | 0.315 | 0.395 |

## Spatial reasoning (SpatialSense, adversarial true/false; chance = 0.5)

Macro AUC over predicates: **0.6757**

## Two-graph decomposition (spatial / semantic, type-stratified protocol)

| source | spatial R@50 / mR@50 | semantic R@50 / mR@50 |
|---|---|---|
| vg150 | 0.627 / 0.298 | 0.494 / 0.294 |
| psg | 0.601 / 0.534 | 0.404 / 0.316 |
| indoorvg | 0.606 / 0.335 | 0.406 / 0.276 |

## Deployment thresholds (per-predicate best-F1, measured on THIS checkpoint)

Score scales are checkpoint-specific (the output head is rank-trained), so
these thresholds transfer to no other model. Regime: gt
boxes, pair_weight=0, 5000
val images. Top predicates by support:

| predicate | threshold | best F1 | GT support |
|---|---|---|---|
| behind | 0.890 | 0.329 | 3599 |
| in front of | 0.865 | 0.337 | 3576 |
| wearing | 0.985 | 0.678 | 3417 |
| to the right of | 0.860 | 0.377 | 3198 |
| to the left of | 0.860 | 0.373 | 3098 |
| resting on | 0.975 | 0.577 | 2164 |
| on | 0.935 | 0.455 | 2042 |
| holding | 0.975 | 0.457 | 1553 |
| beside | 0.975 | 0.196 | 1404 |
| next to | 0.945 | 0.240 | 1352 |
| above | 0.895 | 0.333 | 1279 |
| below | 0.890 | 0.321 | 1239 |
| part of | 0.905 | 0.471 | 1134 |
| supporting | 0.985 | 0.191 | 947 |
| looking at | 0.965 | 0.254 | 872 |

## Provenance

| | |
|---|---|
| run | `relsgg-vits16` |
| git | `e9ea42aed60f766f12ad19d51709129c50110a3b` |
| backbone | facebook/dinov3-vits16-pretrain-lvd1689m |
| text student | `runs/packed/text_student_v2_512/student.pt` sha256 `e0317830b68ea51e...` |
| ONNX opset / parity | 17 / max|Δ| 1.36e-05 |
| torch / transformers | 2.13.0+cu130 / 5.14.1 |
| training mixture | megasg_clean + vg_raw + hicodet, per-image 0.727/0.063/0.210; source-aware negatives: ['hicodet'] |

## License and data notices

Weights are a derivative of Meta **DINOv3** pretrained weights and are
distributed under the DINOv3 license. Training annotations (RA-4M) were
generated by `gemma-4-26B` and carry the Gemma Terms of Use notice; images
are referenced by identifier only (Objects365/COCO/OpenImages). The `vg_raw`
subset derives from Visual Genome (CC BY 4.0). Predicate synonyms are
deliberately never collapsed — surface-form diversity is part of the label
space. Full notices: [THIRD_PARTY_NOTICES.md](https://github.com/Maelic/RelateAnything/blob/main/THIRD_PARTY_NOTICES.md)
in the code repository.

## Citation

```bibtex
@article{neau2026relateanything,
  title   = {RelateAnything: Real-Time Open-Vocabulary Relation Prediction From Any Inputs},
  author  = {Neau, Ma"elic},
  journal = {arXiv preprint arXiv:2609.12552},
  eprint  = {2609.12552},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url     = {https://arxiv.org/abs/2609.12552},
  year    = {2026}
}
```
