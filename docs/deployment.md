# Deployment

Three targets, in increasing order of constraint: a GPU with torch, a laptop CPU
with ONNX or OpenVINO, and a browser.

Full mechanics live in [`deploy/README.md`](../deploy/README.md); this page is
the contract and the decisions.

## Building a release bundle

One driver runs the whole chain — ONNX export with a parity check,
per-checkpoint threshold calibration, then the predicate bank:

```bash
python deploy/build_release.py --only relsgg-vits16plus
# -> deploy/dist/relsgg-vits16plus/
#      relateanything.onnx      the graph
#      relateanything.json      provenance
#      predicate_bank.npz       vocabulary + spatialness + thresholds
#      thresholds.json          per-predicate calibrated thresholds
#      calibration.json         the Platt (a, b)
```

Models are declared in
[`deploy/release_manifest.json`](../deploy/release_manifest.json) — the single
source of truth every release script iterates. The individual steps stay
runnable by hand (`export_onnx.py`, `calibrate_thresholds.py`,
`build_predicate_bank.py`); the driver only sequences them.

Three invariants the tooling **enforces** rather than trusts:

- **Text space.** The bank encoder is resolved from the checkpoint's own args
  and cross-checked against the training embeddings (minimum cosine ≥ 0.999 on
  shared strings) before a bank is written.
- **Thresholds are per-checkpoint.** Score scales do not transfer between
  models, so an uncalibrated bank row is written as **NaN**, loudly — never a
  borrowed number.
- **Provenance.** `relateanything.json` records run name, git SHA, backbone,
  text-student sha256, merge state, and the measured ONNX parity delta.

## What ships, and what does not

| artifact | ships | why |
|---|---|---|
| ONNX graph + provenance | yes | the deployment path |
| predicate bank, thresholds, calibration | yes | two floats and a vocabulary; useless without them |
| torch checkpoint (stripped) | yes | ~261 MB; EMA weights only |
| optimizer state, non-EMA weights | no | that is why a raw `.pth` is 1.95 GB |
| the 19K predicate `W` bank | no | ship the **text encoder** instead — smaller, and it makes any vocabulary work |
| detector weights | **no** | they derive from `ultralytics` (AGPL-3.0). Rebuilt locally, in two commands |

## Runtime: dynamic thresholds and dynamic vocabulary

Both are real, and they are why the graph emits **raw scores** rather than
finished triplets.

The ONNX graph outputs `pred_score [K, V]` and `pair_score [K]`. All filtering
happens host-side in `deploy/postprocess.py`, and decoding costs ~0.3 ms against
a frame in the hundreds of milliseconds — so every knob is free to change per
frame:

```python
from deploy.postprocess import ThresholdConfig, decode

cfg = ThresholdConfig(threshold=0.40,
                      per_predicate={"part of": 0.6, "beside": 0.7})
cfg.threshold = 0.55        # next frame; no re-export, no reload
```

`W` and `alpha` are graph **inputs**, so the predicate set is swappable at
runtime without a text encoder on the device — any subset of the bank can be
activated live.

The shipped bank holds **243 curated predicates**, not the full 19,103 training
vocabulary: the selection is by measured per-predicate quality, and it keeps the
bundle small. Strings outside it need re-encoding with the distilled text
encoder on the training box (`deploy/build_predicate_bank.py`) — which is why
releases ship the text encoder rather than the full `W`.

The threshold is in relation-confidence units (`pred · pair`), the same number
shown on screen. Detector confidence affects ranking only; folding it into the
threshold would divide every score by ~10 and make the knob meaningless.

## The operating point

The shipped default is **τ = 0.56** on a count-matched basis. The calibration is
deliberately conservative: measured precision at that point is 0.889 against an
estimated 0.779.

Worth knowing before you interpret a demo: **78 % of deployed edges sit on
non-ground-truth boxes.** The relation model is being asked about objects no
benchmark ever annotated, which is the intended use and also why benchmark
precision understates what you see on screen.

## Laptop CPU (OpenVINO)

```bash
python deploy/export_openvino.py --dist deploy/dist/relsgg-vits16plus \
    --int8 --calib-dir <images>
```

Measured on a node CPU, 8 threads, PSG validation images. Agreement is semantic
top-1 / top-20 Jaccard against the fp32 ONNX:

| variant | total ms | FPS | top-1 agreement | top-20 Jaccard |
|---|---|---|---|---|
| fp16 | 137 | 7.3 | 0.955 | 0.971 |
| w4 | 126 | 8.0 | 0.727 | 0.771 |

**fp16 is the default.** `w4` is a download-size lever, not a speed lever, and
its agreement is materially worse — use it only when bundle size dominates.

Int8 quantization deliberately **excludes** the `W`/`alpha` subgraph and
everything downstream: it is microseconds of compute, and quantizing unit-norm
embedding activations spends precision exactly where the model keeps its
meaning. Calibration runs real images through the real preprocessing, with the
relation head calibrated on detector boxes — the distribution it will actually
see — not on synthetic tensors.

## Realtime pipeline

The deployment path is **sequential and compiled**: `torch.compile` brings the
released tower, its detector and the decode to **20.3 ms per frame (49 FPS) on
an A40**, 24.4 ms on an A100 and 18.1 ms on an H100, against 30.5 / 43.1 / 32.1
ms eager. Compilation is the largest single gain we measured, 1.5–1.8× on every
GPU.

Running the backbone concurrently with the detector is *not* the win it appears
to be. Since the relation backbone reads only pixels it can overlap the
detector, but measured, that recovers +1.9 ms of the 8.6 ms available on an A40
and **costs 29 ms on an H100**, consistently across five configurations, two
thread settings and three detectors. The sequential path is what ships.

Batch-1 inference is CPU-dispatch bound, which is also why the three towers of
the family cost 19.3–20.0 ms on an A40 across a 2.5× range of FLOPs, and why
the A100 — a Zen3 host — is the *slowest* of the three GPUs at batch 1 while
delivering twice the A40's batched throughput. At batch 1 a latency does not
identify an operating point unless the host CPU is named.

Two things measured *not* to work, so you do not retry them:

- **`torch.compile`** is for demos and products only. `mode="default"` is
  stream-stable at 2.4×; `mode="reduce-overhead"` (CUDA graphs) adds about
  13 % with no error beyond inductor's own (an earlier "silently wrong" reading
  was a measurement bug). Either mode moves evaluation metrics by 40× the
  noise floor, so never benchmark a compiled model.
- **Resolution reduction** does nothing at batch size 1. The bottleneck is
  dispatch, not compute.

## Browser

The same graphs run fully client-side on ONNX Runtime Web (WebGPU/WASM) at
<https://maelic.github.io/RelateAnythingProject/demo/>. Model files are regenerated by
`deploy/web/export_web_models.py`.

The measured cost model is dominated by **detector choice and vocabulary size**.
Masks, `maxDet` and pipelining are dead ends there.

## Keeping the deploy path honest

The deploy path has silently diverged from the training recipe more than once —
a dropped `gate_mlp`, a stale text space, missing config fields. Two guards now
exist and should stay:

- a **config drift guard** in `relsgg/api.py` that fails loudly when a
  checkpoint's args contain a field the loader does not round-trip;
- `tests/test_score_parity.py`, which proves the evaluator and the ONNX/numpy
  path compute the same arithmetic — including a test that the two candidate
  formulas genuinely differ, so the parity test cannot pass vacuously.
