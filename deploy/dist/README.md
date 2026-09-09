# RelateAnything — laptop demo (ONNX)

Live scene graphs from your webcam. An open-vocabulary detector finds objects,
the relation head predicts open-vocabulary relations between them, and you get
an annotated video feed with a latency readout — one merged graph, or two
(spatial + semantic) with `--decompose` / the `g` key.

No torch, no ultralytics, no network. Just onnxruntime, opencv and numpy.

This directory holds one subfolder PER MODEL
(`relsgg-vits16/`, `relsgg-vits16plus/`, `relsgg-vitb16/`), each with its
provenance json, its `predicate_bank.npz` (vocabulary + calibrated
per-predicate thresholds + spatial/semantic types) and `thresholds.json`.
The ONNX graph the demo runs on is exported for `relsgg-vits16plus`; for the
other two, build it with `python deploy/build_release.py --only <model_id>`.
The detector lives in `detector-local/` — it derives from AGPL ultralytics and
is rebuilt locally, never redistributed.

## Run it

```bash
pip install -r requirements.txt

# --dist is a DIRECTORY, not a model name.
python../demo_webcam.py --dist relsgg-vits16plus                    # webcam
python../demo_webcam.py --dist relsgg-vits16plus --image photo.jpg  # one image
python../demo_webcam.py --dist relsgg-vits16plus --decompose        # two graphs
python../demo_webcam.py --dist relsgg-vits16plus --bench            # latency
```

If you moved this directory somewhere else, point the demo at it explicitly:
`python demo_webcam.py --dist /path/to/dist`.

### Live keys

| key | does |
|---|---|
| `q` / `ESC` | quit |
| `[` / `]` | lower / raise the score threshold |
| `+` / `-` | show more / fewer relations |
| `p` | cycle predicate preset: all → interaction → spatial |
| `b` | toggle detector-confidence weighting of the ranking |
| `h` | toggle the HUD |
| `s` | save the current frame |

## What's in here

| file | what |
|---|---|
| `detector.onnx` | YOLO-World v2 (small), re-parameterized to MEGASG's 497 object categories — the same label space the relation model was trained on. 52 MB. |
| `relateanything.onnx` | The relation head with its DINOv3 backbone baked in (ViT-S/16, S/16+ or B/16 depending on the bundle). |
| `predicate_bank.npz` | Predicate embeddings + spatialness routing weights + calibrated per-predicate thresholds. Lets you swap the predicate vocabulary at runtime without a text encoder. |
| `calibration.json` | The two-parameter Platt fit `(a, b)`. Without it a threshold means nothing — see `docs/quickstart.md`. |

## Dynamic thresholds and dynamic vocabulary

Both are real, and both are why the graph outputs **raw scores** rather than
finished triplets. Which bundle to pick: `relsgg-vits16plus` is the recommended
default, `relsgg-vits16` is the CPU pick (the backbone dominates CPU time), and
`relsgg-vitb16` is the accuracy ceiling.

**Threshold.** `relateanything.onnx` emits `pred_score [K,V]` and
`pair_score [K]`; all filtering happens host-side in `deploy/postprocess.py`.
Decoding costs ~0.3 ms against a ~560 ms frame, so every knob is free to change
per frame — global threshold, **per-predicate** thresholds, pair-existence
weight, top-k:

```python
from deploy.postprocess import ThresholdConfig, decode
cfg = ThresholdConfig(threshold=0.40,
                      per_predicate={"part of": 0.6, "beside": 0.7})
cfg.threshold = 0.55        # next frame — no re-export, no reload
```

The threshold is in **relation-confidence** units (`pred * pair`), the same
number shown on screen. Detector confidence affects ranking only — folding it
into the threshold would divide every score by ~10 and make the knob
meaningless.

**Vocabulary.** `W` and `alpha` are graph *inputs*, so the predicate set is
swappable at runtime — that is what the `p` key does. Any of the 243 predicates
in the bank can be activated:

```python
pipe.rel.set_predicates(["hugging", "feeding", "chasing", "repairing"])
print(pipe.rel.available_predicates())     # the full bank
```

Predicates outside the bank need re-encoding with the distilled text encoder on
the training box: `python deploy/build_predicate_bank.py`.

## Performance

Measured with the **ONNX** runtime on a Xeon 6338 at 8 threads, 640px detection
+ 448px relations, 10 boxes, 35 predicates, on the ViT-B/16 bundle — the slowest
of the three. A laptop CPU lands in the same ballpark. For a materially faster
CPU path, export the OpenVINO IR (`deploy/export_openvino.py`): fp16 measures
137 ms / 7.3 FPS at 0.955 top-1 agreement with fp32 ONNX.

| stage | ms | share |
|---|---:|---:|
| detector | 172 | 31% |
| relation head | 386 | 69% |
| decode | 0.3 | 0.1% |
| **total** | **558** | **1.8 FPS** |

The relation head dominates, and ~82% of *it* is the frozen DINOv3 backbone,
whose cost does not depend on how many boxes or predicates you use. So:

- **Fewer boxes or fewer predicates will not speed this up.** Don't bother.
- **A GPU will.** Install an `onnxruntime-*` GPU package (see
  `requirements.txt`) and pass `--providers CUDAExecutionProvider
  CPUExecutionProvider`. This is by far the biggest win.
- Lowering `--det_conf` costs nothing; raising `--max_boxes` past ~16 costs
  little, since the pair budget is fixed at 128 internally.

## Accuracy, honestly

A webcam has no ground truth, so what you see is qualitative. For real numbers
use the VG150/PSG evaluators in `training/` on the training box.

Two things to expect:

- **Object errors propagate.** The relation head is only as good as the boxes.
  A phone detected as a "toothbrush" yields "person holding toothbrush" — the
  *relation* is right, the noun is not.
- **Predicates were chosen from measured per-predicate quality** (recall and
  per-predicate PR curves), not intuition. The weakest common spatials — `near` (best-F1
  0.066) and `next to` (0.039) — are deliberately excluded; `beside` covers
  that meaning. Synonyms like `on` / `resting on` / `on top of` are kept
  deliberately, since the model was trained in a synonym-rich space.
