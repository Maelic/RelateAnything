---
license: other
license_name: dinov3-license
license_link: https://ai.meta.com/resources/models-and-libraries/dinov3-license/
tags:
  - scene-graph-generation
  - open-vocabulary
  - visual-relationship-detection
  - onnx
library_name: relsgg
model-index:
  - name: relsgg-vitb16
    results:
      - task:
          type: scene-graph-generation
        dataset:
          type: vg150
          name: Visual Genome 150 (test)
        metrics:
          - type: F1@50
            value: 0.3746
            name: 'F1@50 (vg150 test, graph-constrained)'
---
# relsgg-vitb16

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
```

```python
from huggingface_hub import snapshot_download
from relsgg.api import RelateAnything

d = snapshot_download("maelic/relsgg-vitb16")
model = RelateAnything.from_checkpoint(
    f"{d}/model.pth", predicates=["holding", "riding", "next to"], device="cuda")
triplets = model.predict(image, boxes_xyxy, topk=20)      # image: PIL / ndarray, boxes: [N, 4] pixels
model.set_vocabulary(["about to collide with", "reflected in"])   # any strings, no retraining
graphs = model.predict(image, boxes_xyxy, decompose=True)          # {"spatial": [...], "semantic": [...]}
```

`model.pth` embeds the backbone config, so nothing else is downloaded: no
gated DINOv3 login is needed to run it. `text_student.pt` (the distilled
predicate text encoder, with its CLIP tokenizer files) sits next to it and is
found automatically.

Files: `model.pth` (torch, EMA weights), `text_student.pt`, `relateanything.onnx` + OpenVINO fp16 IR, `predicate_bank.npz`, `thresholds.json`, `calibration.json` (the laptop bundle, see `deploy/README.md`), `README.md`.

**Every number below is generated from measured eval artifacts
(`release/make_model_cards.py`); none is hand-typed.**

## Closed-vocabulary transfer (reparameterized, TEST, graph-constrained)

| source | R@50 | mR@50 | F1@50 |
|---|---|---|---|
| vg150 | 0.531 | 0.289 | 0.375 |
| psg | 0.412 | 0.302 | 0.349 |
| indoorvg | 0.538 | 0.302 | 0.387 |
| hicodet | 0.468 | 0.313 | 0.375 |

## Open-vocabulary, NO reparameterization (all 19,103 predicates deployed)

Synonym-matched at the calibrated tau (see provenance). This is the honest
"the model never saw your label set" protocol.

| source | SoftR@50 | SoftmR@50 | SoftF1@50 |
|---|---|---|---|
| vg150 | 0.566 | 0.345 | 0.429 |
| psg | 0.324 | 0.299 | 0.311 |
| indoorvg | 0.545 | 0.357 | 0.432 |

## Spatial reasoning (SpatialSense, adversarial true/false; chance = 0.5)

Macro AUC over predicates: **0.6860**

## Two-graph decomposition (spatial / semantic, type-stratified protocol)

| source | spatial R@50 / mR@50 | semantic R@50 / mR@50 |
|---|---|---|
| vg150 | 0.636 / 0.314 | 0.494 / 0.312 |
| psg | 0.608 / 0.541 | 0.420 / 0.327 |
| indoorvg | 0.617 / 0.359 | 0.429 / 0.312 |

## Deployment thresholds (per-predicate best-F1, measured on THIS checkpoint)

Score scales are checkpoint-specific (the output head is rank-trained), so
these thresholds transfer to no other model. Regime: gt
boxes, pair_weight=0, 5000
val images. Top predicates by support:

| predicate | threshold | best F1 | GT support |
|---|---|---|---|
| behind | 0.895 | 0.337 | 3598 |
| in front of | 0.860 | 0.343 | 3580 |
| wearing | 0.985 | 0.684 | 3417 |
| to the right of | 0.860 | 0.385 | 3196 |
| to the left of | 0.870 | 0.370 | 3102 |
| resting on | 0.975 | 0.582 | 2166 |
| on | 0.925 | 0.460 | 2043 |
| holding | 0.980 | 0.469 | 1552 |
| beside | 0.980 | 0.194 | 1402 |
| next to | 0.935 | 0.231 | 1352 |
| above | 0.895 | 0.332 | 1283 |
| below | 0.895 | 0.328 | 1241 |
| part of | 0.905 | 0.495 | 1135 |
| supporting | 0.985 | 0.194 | 945 |
| looking at | 0.965 | 0.271 | 872 |

## Provenance

| | |
|---|---|
| run | `full_v42cfg_wv2-512_lora12_mixvghico0.05_sig0.25_btd0.3_negmask_def4h8v3n2_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5_hicov2` |
| git | `e9ea42aed60f766f12ad19d51709129c50110a3b` |
| backbone | dinov3 (facebook/dinov3-vitb16-pretrain-lvd1689m) |
| LoRA merged | False (pre-merge rank None) |
| text student | `runs/packed/text_student_v2_512/student.pt` sha256 `e0317830b68ea51e...` |
| ONNX opset / parity | 17 / max|Δ| 2.96e-05 |
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
  journal = {arXiv preprint},
  year    = {2026}
}
```
