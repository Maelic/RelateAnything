# RelateAnything live demo — YOLOE-11m + open-vocabulary relations

Live scene graphs from a webcam, with **both** vocabularies editable at runtime.

## Run it

```bash
source.venv/bin/activate
python deploy/gradio_app.py                 # GPU if present, else CPU
python deploy/gradio_app.py --device cpu    # force CPU
python deploy/gradio_app.py --share         # public link (customer demo)
```

Open the printed URL. The browser only ships webcam frames — the model runs
server-side, so a laptop can drive a remote GPU.

On a **remote GPU box** forward the port:

```bash
ssh -L 7860:<node>:7860 <user>@<login-host>
```

### Which detector checkpoint

| flag | behaviour |
|---|---|
| `--det checkpoints/detectors/yoloe-11m-seg-pf.pt` (default) | **prompt-free** — detects anything out of the box, no class list needed |
| `--det checkpoints/detectors/yoloe-11m-seg.pt` | **text-prompt** — only detects the classes you type. Ships with an EMPTY vocabulary, so it finds nothing until classes are set |

## Using the demo

* **Webcam** tab streams live; **Image / upload** tab is better on CPU.
* **Object classes** — any comma-separated words (`forklift, pallet, hi-vis vest`).
  Re-parameterizes YOLOE through its text encoder. Leave empty on the
  prompt-free checkpoint.
* **Predicates** — any comma-separated relations (`lifting, blocking, reaching for`).
  Re-parameterized through the relation checkpoint's own text encoder; after
  that inference is pure vision. **Neither box requires retraining.**
* Sliders: detector confidence, max triplets, relation score threshold.
* **Graph decode** (radio): `merged` is one ranked graph. `both` / `spatial` /
  `semantic` use the **two-graph decode** — the *same* forward pass ranked
  separately inside the spatial (layout, blue) and semantic (content, green)
  predicate columns, then cut at top-K per stream. No second forward pass —
  it re-ranks scores already computed, measured at +3 ms (35.1 → 38.2) — and
  it beat a single merged graph of twice the budget on 6/6 benchmark cells.
  Requires a `--dual_spatial_head` checkpoint — the shipped full-recipe model
  has one (`dual_spatial_head: true`, `gate_mlp: true`); the HUD warns if a
  checkpoint does not. The split is derived per-vocabulary from the corpus type
  map, so editing the predicate box re-derives it (22/22 typed from the map,
  zero guessed, on the default list).
* **Spatial stream: drop relatedness prior** (checkbox, default **off**).
  Relatedness is an annotation-propensity ≈ contact signal. Dropping it is the
  measured optimum for *judging spatial truth* (+0.068 macro AUC on
  SpatialSense, projective predicates +0.11–0.14) — but on raw detector output
  relatedness is also what suppresses duplicate boxes, so dropping it saturates
  scores at 1.00 and surfaces junk (`motorcycle –on→ motorcycle`). Kept:
  `person –on→ motorcycle 0.57`, `boot –on→ motorcycle 0.48`. The demo defaults
  to readability; `ParallelScenePipeline` defaults to the measured optimum.
* The HUD shows per-stage latency so you can see the pipeline working.

## Why it is fast (measured, A40, bf16)

Batch-1 inference is **CPU-dispatch bound**: 1,362 kernel launches per frame,
~30 ms of CPU dispatch against ~9 ms of GPU work. Proof — latency is identical
at 448/392/336 px. So the usual levers (smaller backbone, lower resolution) buy
nothing at batch 1; three different things do:

1. **Detector ∥ backbone.** The DINOv3 backbone does not depend on the boxes,
   so the two are independent branches of the same frame, joined through
   `model.forward(precomputed_features=...)`.
   **Measured caveat:** doing this with a CUDA stream *alone* buys nothing
   (54.0 → 54.3 ms) — both branches are CPU-dispatch bound and a second stream
   does not add a second CPU thread. Real overlap needs thread-level
   parallelism; `--no_overlap` A/Bs it.
2. **Right-sized static shapes** (`--max_objects 16 --final_budget 64`, trained
   values 40/128). **Measured: no gain** (35.5 ms at 16/64, 35.1 at 10/40, vs
   35.1 at 40/128). Predicted to be the biggest lever since pairs go as N²; it
   is flat for the same reason resolution is flat — the cost is kernel COUNT,
   not kernel size. Kept as a knob for memory-constrained devices only.
3. **Trimmed predicate vocabulary.** Typing 20 predicates instead of 19K shrinks
   the head and the decode.

Not used for reported numbers: `torch.compile`. `mode="default"` is 2.4×
faster and stream-stable; `mode="reduce-overhead"` (CUDA graphs) adds about
13 % on top with no error beyond inductor's own. An earlier reading of
"reduce-overhead" as silently wrong was a tie-breaking measurement bug. Either
mode perturbs evaluation metrics by about 40× the noise floor, so compiled
models are for the demo and products, never for a table.

## Measured end-to-end (A40, 640×427 frame, 16 objects, masks on)

| detector | det | backbone | relations | total | FPS |
|---|---|---|---|---|---|
| text-prompt, sequential | 12.5 | (in rel) | 25.9 | 40.5 ms | 24.7 |
| text-prompt, **thread overlap** | 16.3 | 16.7 | 16.3 | **35.1 ms** | **28.5** |
| prompt-free (4.5k classes) | 25.6 | ~10 | 16.4 | 54.0 ms | 18.5 |

The text-prompt checkpoint is **2× faster** than prompt-free because its
classification head carries 10 columns instead of ~4,585 — so naming your
classes is itself an optimisation, not just a UX choice.

Sample output (unedited): `person –riding→ motorcycle 0.58`,
`person –wearing→ helmet 0.57`, `boot –on→ motorcycle 0.48`,
`saddlebag –on→ motorcycle 0.46`.

## CPU

Everything runs, just slower (the ONNX bundle measured ~1.8 FPS on a laptop,
82% of it backbone). Use the Image tab, or lower `--max_objects`. True CPU
real-time needs a smaller backbone, which is a training project, not a flag.

## Files

| file | role |
|---|---|
| `deploy/pipeline.py` | `ParallelScenePipeline` — overlap, static shapes, both re-parameterization APIs |
| `deploy/gradio_app.py` | the UI |
| `relsgg/api.py` | `RelateAnything.from_checkpoint` / `set_vocabulary` |
