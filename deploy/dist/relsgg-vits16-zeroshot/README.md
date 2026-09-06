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
  - name: relsgg-vits16-zeroshot
    results:
      - task:
          type: scene-graph-generation
        dataset:
          type: vg150
          name: Visual Genome 150 (test)
        metrics:
          - type: F1@50
            value: 0.3631
            name: 'F1@50 (vg150 test, graph-constrained)'
---
# relsgg-vits16-zeroshot

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

d = snapshot_download("maelic/relsgg-vits16-zeroshot")
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

Files: `model.pth` (torch, EMA weights), `text_student.pt`, `README.md`.

**Every number below is generated from measured eval artifacts
(`release/make_model_cards.py`); none is hand-typed.**

## Closed-vocabulary transfer (reparameterized, TEST, graph-constrained)

| source | R@50 | mR@50 | F1@50 |
|---|---|---|---|
| vg150 | 0.517 | 0.280 | 0.363 |
| psg | 0.392 | 0.289 | 0.333 |
| indoorvg | 0.501 | 0.287 | 0.365 |
| hicodet | 0.341 | 0.127 | 0.185 |

## Open-vocabulary, NO reparameterization (all 19,103 predicates deployed)

Synonym-matched at the calibrated tau (see provenance). This is the honest
"the model never saw your label set" protocol.

| source | SoftR@50 | SoftmR@50 | SoftF1@50 |
|---|---|---|---|
| vg150 | 0.541 | 0.340 | 0.418 |
| psg | 0.298 | 0.267 | 0.281 |
| indoorvg | 0.504 | 0.326 | 0.396 |

## Spatial reasoning (SpatialSense, adversarial true/false; chance = 0.5)

Macro AUC over predicates: **0.6937**

## Two-graph decomposition (spatial / semantic, type-stratified protocol)

| source | spatial R@50 / mR@50 | semantic R@50 / mR@50 |
|---|---|---|
| vg150 | 0.627 / 0.302 | 0.484 / 0.299 |
| psg | 0.609 / 0.547 | 0.407 / 0.310 |
| indoorvg | 0.609 / 0.346 | 0.408 / 0.296 |

## Deployment thresholds (per-predicate best-F1, measured on THIS checkpoint)

Score scales are checkpoint-specific (the output head is rank-trained), so
these thresholds transfer to no other model. Regime: gt
boxes, pair_weight=0, 5000
val images. Top predicates by support:

| predicate | threshold | best F1 | GT support |
|---|---|---|---|
| behind | 0.895 | 0.336 | 3601 |
| in front of | 0.870 | 0.342 | 3575 |
| wearing | 0.980 | 0.689 | 3417 |
| to the right of | 0.880 | 0.386 | 3197 |
| to the left of | 0.870 | 0.376 | 3094 |
| resting on | 0.975 | 0.572 | 2167 |
| on | 0.925 | 0.453 | 2043 |
| holding | 0.975 | 0.455 | 1552 |
| beside | 0.980 | 0.201 | 1405 |
| next to | 0.940 | 0.245 | 1352 |
| above | 0.900 | 0.327 | 1282 |
| below | 0.890 | 0.324 | 1239 |
| part of | 0.885 | 0.477 | 1134 |
| supporting | 0.980 | 0.194 | 944 |
| looking at | 0.970 | 0.256 | 873 |

## Provenance

| | |
|---|---|
| run | `full_v42cfg_wv2-512_lora12_mixvg_sig0.25_btd0.3_def4h8v3n2_vits16_r0_lr4e-4_ep12_newopt_spe_gsq_pe16_bg0.05_ntaps_ms0.5-1.5` |
| git | `06e7afdf0a8b0d9d5086879d6b6abd3848a0b5ab` |
| backbone | dinov3 (facebook/dinov3-vits16-pretrain-lvd1689m) |
| LoRA merged | False (pre-merge rank None) |
| text student | `runs/packed/text_student_v2_512/student.pt` sha256 `e0317830b68ea51e...` |
| ONNX opset / parity | 17 / max|Δ| 1.12e-05 |
| torch / transformers | 2.13.0+cu130 / 5.14.1 |
| training mixture | megasg_clean + vg_raw, per-image 0.919/0.081 |

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
