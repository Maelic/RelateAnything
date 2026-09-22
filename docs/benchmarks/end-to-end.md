# Detector-to-relations latency

The end-to-end mode of `deploy/bench_gpu_backends.py` measures an image through
detection, relation prediction and triplet decoding. This report and its
[raw record (gzip JSON)](rtx3080-laptop-e2e.json.gz) are separate from the
[relation-only comparison](README.md).

## Measurements

<!-- BEGIN GENERATED E2E LATENCY -->
Observed enforced GPU power limits: **90.00 W**. No thermal slowdown counter increase was recorded during the measured blocks (warmup included).

All latency cells show **median / p95 in milliseconds**, recomputed from raw calls.
Total percentiles include orchestration and are measured directly, not summed from stage percentiles.

### YOLO26m

0 of 4,320 calls skipped relation inference. Active-frame and overall totals are identical.

| Relation model | Relation backend | Predicates | Detector | Relations | Decode | Total |
|---|---|---:|---:|---:|---:|---:|
| ViT-S | PyTorch eager FP32 | 35 | 18.59 / 19.76 | 21.77 / 22.92 | 0.24 / 0.34 | 40.73 / 42.73 |
| ViT-S | PyTorch eager BF16 | 35 | 17.59 / 18.83 | 21.05 / 24.84 | 0.24 / 0.35 | 38.91 / 43.61 |
| ViT-S | ONNX CUDA FP32 | 35 | 18.99 / 19.92 | 23.00 / 24.01 | 0.25 / 0.33 | 42.36 / 43.96 |
| ViT-S | TensorRT FP32 | 35 | 19.13 / 19.76 | 15.22 / 15.77 | 0.24 / 0.30 | 34.67 / 35.54 |
| ViT-S | PyTorch eager FP32 | 243 | 18.41 / 18.95 | 21.63 / 22.44 | 0.39 / 0.45 | 40.54 / 41.65 |
| ViT-S | PyTorch eager BF16 | 243 | 17.51 / 19.13 | 21.14 / 27.06 | 0.39 / 0.59 | 39.08 / 46.22 |
| ViT-S | ONNX CUDA FP32 | 243 | 18.79 / 19.26 | 23.09 / 23.62 | 0.40 / 0.46 | 42.41 / 43.31 |
| ViT-S | TensorRT FP32 | 243 | 19.20 / 19.92 | 15.40 / 16.04 | 0.39 / 0.52 | 35.14 / 36.09 |
| ViT-S+ | PyTorch eager FP32 | 35 | 18.53 / 19.50 | 23.76 / 24.79 | 0.22 / 0.31 | 42.58 / 44.72 |
| ViT-S+ | PyTorch eager BF16 | 35 | 17.16 / 18.06 | 20.96 / 22.59 | 0.23 / 0.30 | 38.38 / 40.52 |
| ViT-S+ | ONNX CUDA FP32 | 35 | 19.33 / 20.04 | 25.99 / 26.78 | 0.28 / 0.35 | 45.64 / 47.08 |
| ViT-S+ | TensorRT FP32 | 35 | 19.61 / 20.29 | 16.56 / 17.31 | 0.26 / 0.33 | 36.50 / 37.71 |
| ViT-S+ | PyTorch eager FP32 | 243 | 18.45 / 19.27 | 23.75 / 24.47 | 0.37 / 0.48 | 42.66 / 44.17 |
| ViT-S+ | PyTorch eager BF16 | 243 | 17.15 / 18.16 | 21.03 / 23.56 | 0.37 / 0.50 | 38.54 / 41.97 |
| ViT-S+ | ONNX CUDA FP32 | 243 | 18.92 / 20.00 | 25.73 / 26.62 | 0.39 / 0.49 | 45.13 / 47.00 |
| ViT-S+ | TensorRT FP32 | 243 | 19.25 / 20.00 | 16.57 / 17.28 | 0.38 / 0.47 | 36.25 / 37.53 |
| ViT-B | PyTorch eager FP32 | 35 | 21.01 / 21.71 | 41.05 / 41.85 | 0.27 / 0.33 | 62.47 / 63.67 |
| ViT-B | PyTorch eager BF16 | 35 | 18.05 / 18.56 | 22.77 / 24.33 | 0.26 / 0.32 | 41.15 / 43.00 |
| ViT-B | ONNX CUDA FP32 | 35 | 21.63 / 22.37 | 43.66 / 44.52 | 0.27 / 0.34 | 65.65 / 66.91 |
| ViT-B | TensorRT FP32 | 35 | 22.29 / 23.64 | 31.96 / 32.53 | 0.26 / 0.33 | 54.54 / 56.29 |
| ViT-B | PyTorch eager FP32 | 243 | 20.84 / 21.85 | 41.01 / 41.98 | 0.43 / 0.52 | 62.34 / 64.12 |
| ViT-B | PyTorch eager BF16 | 243 | 17.87 / 18.63 | 23.01 / 25.01 | 0.41 / 0.50 | 41.51 / 44.08 |
| ViT-B | ONNX CUDA FP32 | 243 | 21.58 / 22.30 | 43.67 / 44.45 | 0.44 / 0.52 | 65.80 / 66.85 |
| ViT-B | TensorRT FP32 | 243 | 22.28 / 23.57 | 32.03 / 32.59 | 0.43 / 0.55 | 54.83 / 56.39 |

### YOLO-World v2-m

0 of 4,320 calls skipped relation inference. Active-frame and overall totals are identical.

| Relation model | Relation backend | Predicates | Detector | Relations | Decode | Total |
|---|---|---:|---:|---:|---:|---:|
| ViT-S | PyTorch eager FP32 | 35 | 18.05 / 18.71 | 22.01 / 22.81 | 0.26 / 0.35 | 40.40 / 41.57 |
| ViT-S | PyTorch eager BF16 | 35 | 17.72 / 19.09 | 22.42 / 26.95 | 0.26 / 0.35 | 40.50 / 46.67 |
| ViT-S | ONNX CUDA FP32 | 35 | 18.36 / 18.97 | 23.22 / 23.86 | 0.27 / 0.34 | 41.94 / 43.09 |
| ViT-S | TensorRT FP32 | 35 | 18.67 / 19.28 | 15.46 / 15.89 | 0.27 / 0.34 | 34.44 / 35.37 |
| ViT-S | PyTorch eager FP32 | 243 | 17.93 / 18.63 | 21.92 / 22.63 | 0.43 / 0.54 | 40.48 / 41.36 |
| ViT-S | PyTorch eager BF16 | 243 | 17.50 / 18.23 | 22.27 / 23.67 | 0.42 / 0.52 | 40.24 / 41.92 |
| ViT-S | ONNX CUDA FP32 | 243 | 18.43 / 19.12 | 23.32 / 24.02 | 0.45 / 0.60 | 42.34 / 43.40 |
| ViT-S | TensorRT FP32 | 243 | 18.65 / 19.13 | 15.59 / 16.07 | 0.44 / 0.53 | 34.73 / 35.68 |
| ViT-S+ | PyTorch eager FP32 | 35 | 18.21 / 19.14 | 24.14 / 25.08 | 0.27 / 0.32 | 42.69 / 44.27 |
| ViT-S+ | PyTorch eager BF16 | 35 | 17.56 / 18.30 | 22.91 / 24.49 | 0.26 / 0.32 | 40.80 / 42.57 |
| ViT-S+ | ONNX CUDA FP32 | 35 | 18.60 / 19.46 | 26.09 / 27.11 | 0.27 / 0.36 | 45.04 / 46.89 |
| ViT-S+ | TensorRT FP32 | 35 | 19.04 / 20.35 | 16.73 / 17.67 | 0.27 / 0.45 | 36.07 / 38.18 |
| ViT-S+ | PyTorch eager FP32 | 243 | 18.22 / 18.87 | 24.16 / 25.02 | 0.43 / 0.53 | 42.91 / 44.34 |
| ViT-S+ | PyTorch eager BF16 | 243 | 17.66 / 18.52 | 22.96 / 25.74 | 0.43 / 0.53 | 41.17 / 44.55 |
| ViT-S+ | ONNX CUDA FP32 | 243 | 18.71 / 19.74 | 26.19 / 26.98 | 0.46 / 0.58 | 45.55 / 47.14 |
| ViT-S+ | TensorRT FP32 | 243 | 18.84 / 19.50 | 16.87 / 17.36 | 0.43 / 0.52 | 36.22 / 37.17 |
| ViT-B | PyTorch eager FP32 | 35 | 20.09 / 21.31 | 41.19 / 42.14 | 0.27 / 0.38 | 61.72 / 63.36 |
| ViT-B | PyTorch eager BF16 | 35 | 17.34 / 18.26 | 22.51 / 25.08 | 0.25 / 0.31 | 40.14 / 43.38 |
| ViT-B | ONNX CUDA FP32 | 35 | 20.93 / 22.52 | 43.88 / 44.78 | 0.28 / 0.40 | 65.17 / 67.24 |
| ViT-B | TensorRT FP32 | 35 | 21.46 / 22.68 | 32.12 / 32.67 | 0.27 / 0.35 | 53.95 / 55.49 |
| ViT-B | PyTorch eager FP32 | 243 | 20.38 / 21.84 | 41.41 / 42.55 | 0.45 / 0.68 | 62.47 / 64.49 |
| ViT-B | PyTorch eager BF16 | 243 | 17.43 / 18.46 | 22.92 / 25.85 | 0.43 / 0.61 | 40.75 / 44.80 |
| ViT-B | ONNX CUDA FP32 | 243 | 20.68 / 21.56 | 43.83 / 44.67 | 0.45 / 0.59 | 65.09 / 66.50 |
| ViT-B | TensorRT FP32 | 243 | 21.30 / 22.42 | 32.17 / 32.75 | 0.42 / 0.48 | 54.05 / 55.27 |

### YOLOE-26m

0 of 4,320 calls skipped relation inference. Active-frame and overall totals are identical.

| Relation model | Relation backend | Predicates | Detector | Relations | Decode | Total |
|---|---|---:|---:|---:|---:|---:|
| ViT-S | PyTorch eager FP32 | 35 | 25.80 / 28.07 | 22.73 / 24.86 | 0.26 / 0.42 | 48.77 / 53.14 |
| ViT-S | PyTorch eager BF16 | 35 | 24.82 / 26.02 | 20.76 / 24.09 | 0.24 / 0.30 | 45.84 / 49.84 |
| ViT-S | ONNX CUDA FP32 | 35 | 25.77 / 26.88 | 23.65 / 24.75 | 0.27 / 0.37 | 49.91 / 51.61 |
| ViT-S | TensorRT FP32 | 35 | 26.25 / 27.74 | 15.53 / 16.59 | 0.26 / 0.37 | 42.09 / 44.41 |
| ViT-S | PyTorch eager FP32 | 243 | 25.27 / 26.21 | 22.36 / 23.79 | 0.40 / 0.47 | 48.17 / 50.06 |
| ViT-S | PyTorch eager BF16 | 243 | 24.85 / 26.33 | 21.01 / 24.68 | 0.40 / 0.48 | 46.33 / 51.39 |
| ViT-S | ONNX CUDA FP32 | 243 | 25.84 / 27.37 | 23.96 / 25.22 | 0.44 / 0.59 | 50.16 / 52.88 |
| ViT-S | TensorRT FP32 | 243 | 26.12 / 27.59 | 15.62 / 16.89 | 0.41 / 0.52 | 42.27 / 44.96 |
| ViT-S+ | PyTorch eager FP32 | 35 | 25.45 / 26.09 | 24.33 / 25.42 | 0.25 / 0.31 | 50.11 / 51.10 |
| ViT-S+ | PyTorch eager BF16 | 35 | 24.72 / 25.83 | 21.17 / 23.27 | 0.24 / 0.31 | 46.17 / 48.93 |
| ViT-S+ | ONNX CUDA FP32 | 35 | 25.94 / 26.64 | 26.28 / 27.43 | 0.26 / 0.32 | 52.64 / 53.63 |
| ViT-S+ | TensorRT FP32 | 35 | 26.46 / 27.07 | 16.69 / 17.18 | 0.25 / 0.30 | 43.48 / 44.25 |
| ViT-S+ | PyTorch eager FP32 | 243 | 25.49 / 26.46 | 24.35 / 25.49 | 0.40 / 0.48 | 50.43 / 52.12 |
| ViT-S+ | PyTorch eager BF16 | 243 | 24.63 / 25.36 | 21.06 / 22.48 | 0.39 / 0.45 | 46.25 / 47.77 |
| ViT-S+ | ONNX CUDA FP32 | 243 | 25.84 / 26.46 | 26.34 / 27.50 | 0.41 / 0.49 | 52.77 / 53.72 |
| ViT-S+ | TensorRT FP32 | 243 | 26.44 / 27.07 | 16.86 / 17.38 | 0.40 / 0.47 | 43.79 / 44.71 |
| ViT-B | PyTorch eager FP32 | 35 | 26.85 / 27.68 | 41.06 / 41.45 | 0.18 / 0.24 | 68.08 / 69.32 |
| ViT-B | PyTorch eager BF16 | 35 | 23.65 / 24.93 | 22.09 / 23.12 | 0.17 / 0.23 | 45.98 / 47.89 |
| ViT-B | ONNX CUDA FP32 | 35 | 27.47 / 28.17 | 43.42 / 44.61 | 0.18 / 0.27 | 71.16 / 75.00 |
| ViT-B | TensorRT FP32 | 35 | 28.27 / 30.11 | 31.65 / 32.22 | 0.19 / 0.29 | 60.15 / 62.23 |
| ViT-B | PyTorch eager FP32 | 243 | 26.89 / 28.20 | 41.09 / 41.77 | 0.31 / 0.43 | 68.35 / 69.94 |
| ViT-B | PyTorch eager BF16 | 243 | 23.62 / 24.54 | 22.01 / 23.25 | 0.30 / 0.37 | 46.09 / 48.15 |
| ViT-B | ONNX CUDA FP32 | 243 | 27.49 / 28.51 | 43.47 / 44.03 | 0.33 / 0.41 | 71.36 / 72.62 |
| ViT-B | TensorRT FP32 | 243 | 27.87 / 28.65 | 31.71 / 32.10 | 0.29 / 0.39 | 59.84 / 61.03 |

### Workload and validation

Detected / retained box-count ranges, across every timed call for each image:

| Image | YOLO26m | YOLO-World v2-m | YOLOE-26m |
|---|---:|---:|---:|
| bicycle.jpg | 2 / 2 | 2 / 2 | 3 / 3 |
| catlaptop.jpg | 3 / 3 | 3 / 3 | 3 / 3 |
| frisbee.jpg | 7 / 7 | 7 / 7 | 8 / 8 |
| horse.jpg | 6 / 6 | 4 / 4 | 5 / 5 |
| skateboard.jpg | 5 / 5 | 4 / 4 | 6 / 6 |
| tennis.jpg | 2 / 2 | 2 / 2 | 2 / 2 |

All 216 FP32 preflight comparisons passed with identical valid pair sets.
Largest absolute logit difference: 0.008907795.
Tolerances and per-image results are retained in the raw record. BF16 accuracy was not evaluated.

### GPU conditions

Temperature ranges before warmup / after timing, and thermal counters including warmup:

| Detector | Relation model | GPU temperature | Enforced power limit | Thermal counter increase | Cooling waits |
|---|---|---|---|---:|---:|
| YOLO26m | ViT-S | 70–70°C / 73–76°C | 90.00 W | 0.0 ms | 233.3 s |
| YOLO26m | ViT-S+ | 69–70°C / 75–77°C | 90.00 W | 0.0 ms | 446.2 s |
| YOLO26m | ViT-B | 69–70°C / 75–77°C | 90.00 W | 0.0 ms | 511.3 s |
| YOLO-World v2-m | ViT-S | 70–70°C / 76–77°C | 90.00 W | 0.0 ms | 535.7 s |
| YOLO-World v2-m | ViT-S+ | 70–70°C / 76–77°C | 90.00 W | 0.0 ms | 566.4 s |
| YOLO-World v2-m | ViT-B | 69–70°C / 76–78°C | 90.00 W | 0.0 ms | 643.8 s |
| YOLOE-26m | ViT-S | 69–70°C / 75–77°C | 90.00 W | 0.0 ms | 397.4 s |
| YOLOE-26m | ViT-S+ | 70–70°C / 75–77°C | 90.00 W | 0.0 ms | 360.8 s |
| YOLOE-26m | ViT-B | 69–70°C / 75–77°C | 90.00 W | 0.0 ms | 331.8 s |

Software: PyTorch 2.14.0+cu130, CUDA 13.0, TensorRT 10.16.1.11, ONNX Runtime 1.30.0, Ultralytics 8.4.159.
CPU: 11th Gen Intel(R) Core(TM) i9-11950H @ 2.60GHz. PyTorch/ONNX CPU threads: 4; OpenCV threads: 1.
<!-- END GENERATED E2E LATENCY -->

## Comparison

The default sweep uses medium detector checkpoints and a shared COCO-80 object
vocabulary:

- **YOLO26m:** `yolo26m.pt`, the standard COCO-trained detector.
- **YOLO-World v2-m:** `yolov8m-worldv2.pt`, with its COCO class vocabulary.
- **YOLOE-26m:** `yoloe-26m-seg.pt`, with the same COCO class names encoded once
  before benchmarking. Its native segmentation computation remains included in
  detector time; only its boxes enter the relation model.

Each detector is paired with ViT-S, ViT-S+ and ViT-B, using both the default
deployment predicates and the complete bundled predicate bank. The four
**relation** backends are eager PyTorch FP32, eager PyTorch BF16, ONNX Runtime
CUDA FP32 and TensorRT FP32. Detection stays on **PyTorch FP32 in every arm**.
Thus a TensorRT row measures a PyTorch detector followed by TensorRT relations;
it does not describe a pipeline with both models converted to TensorRT.

Medium variants are a starting point, not equal-compute or equal-accuracy models.
Matching their class names does not match their training data or detection
quality. This experiment measures latency, not detection or scene-graph accuracy.

## What is timed

The input is a decoded BGR image in host memory. Every timed call reruns the
detector and records synchronized wall time for:

1. **Detection:** native image preprocessing, transfer, model execution and
   postprocessing, plus copying boxes/scores/classes to CPU and sorting by
   confidence. Ultralytics handles each family's output format. YOLO26 and YOLOE
   use its default NMS path (`nms=None`); this run does not select the optional
   NMS-free head. Class-aware NMS is requested for all three detectors.
2. **Relations:** capping the detections, relation preprocessing and transfers,
   backbone/head execution, and copying raw outputs to CPU.
3. **Decoding:** shared score calibration and triplet selection, with object
   boxes, confidence scores and labels supplied to the decoder.

Total latency is measured around the entire call, including synchronization
and orchestration. Total median/p95 comes from total samples, not the sum of
stage percentiles. Disk/video decoding, display, model loading, prompt encoding,
engine construction and warmup are excluded. Stages run serially; reciprocal
latency is not a measurement of pipelined video throughput.

Detector inputs use square 640-pixel letterboxing (`rect=False`), confidence
0.25, IoU 0.6 and at most 100 detections. The relation stage uses the 20 most
confident boxes, its bundle's 448-pixel input and padding/region contract, and
the same calibration/decoder settings as the relation-only benchmark. These
limits are CLI options. This comparison uses box inputs for all relation
backends, including when YOLOE produces masks.

Actual box counts can differ between detectors. Each sample records detected
and used counts, image index, round, stage times, total time, output triplet
count and whether relations were skipped. Frames with fewer than two retained
boxes skip relation inference and stay in the overall timing distribution.
`active_pipeline_latency` separately summarizes totals for frames that did run
relations; it is null when no measured call did so. Always report skipped
counts and box counts alongside latency. An input set with no active relation
frames fails the preflight.

The default input set is the six repository photos, useful for a repeatable
local comparison. Use `--image-dir /path/to/photos` for a broader, representative
workload; direct JPG/JPEG/PNG children are loaded in sorted order and hashed.
Choose `--iterations` as a multiple of the image count so every image appears
equally often in each configuration. All detector/checkpoint runs must use the
same input set and sampling settings.

## Prepare without GPU timing

Use the CUDA/TensorRT/ONNX Runtime environment from the
[relation benchmark setup](README.md#reproduce). Add the detector dependencies;
the adapter was checked with Ultralytics 8.4.159 and requires at least that
version. Detector code and weights retain their upstream licenses; see
[third-party notices](../../THIRD_PARTY_NOTICES.md).

```bash
pip install "ultralytics==8.4.159"
pip install "git+https://github.com/ultralytics/CLIP.git@a13192f8cb767260d7dfd98c843b0716593169e7"

# Downloads official weights and encodes any needed prompts on CPU.
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m deploy.e2e --size m --out checkpoints/detectors/e2e
```

Preparation writes a local `*-coco80.pt` checkpoint and a JSON sidecar with its
source/checkpoint hashes, class names and package version. YOLOE also downloads
the text encoder during this step. No text encoding or weight downloads should
be needed in a timed run. The adapter rejects changed weights, a mismatched
detector family or a different class vocabulary. These local detector artifacts
are ignored by Git and are not distributed with RelateAnything.

Prepare and validate one relation bundle per checkpoint using the export steps
in [Reproduce](README.md#reproduce), placing them under
`runs/benchmark/gpu/<model_id>/`. Engine building and its GPU parity checks are
separate from the timing run; reuse already validated engines on the same GPU
and software stack. This run uses fresh exports for all three checkpoints,
including the [sparse-pair ONNX CUDA correction](README.md#sparse-pair-onnx-cuda-correction).

## Preview and run the sweep

From the repository root, this loop **only prints plans**. `--dry-run` does not
load weights, require GPU libraries or create result files. Once the GPU and
bundles are ready, set `preview=()` to execute the same sweep sequentially.

```bash
preview=(--dry-run)
for detector in yolo26 yolo-world yoloe; do
  case "$detector" in
    yolo26) weights=yolo26m-coco80.pt ;;
    yolo-world) weights=yolov8m-worldv2-coco80.pt ;;
    yoloe) weights=yoloe-26m-seg-coco80.pt ;;
  esac
  for model_id in relsgg-vits16 relsgg-vits16plus relsgg-vitb16; do
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/bench_gpu_backends.py \
      --checkpoint "checkpoints/$model_id/model.pth" \
      --bundle "runs/benchmark/gpu/$model_id" \
      --detector "$detector" \
      --detector-weights "checkpoints/detectors/e2e/$weights" \
      --image-dir assets/reel/images \
      --rounds 3 --warmup 20 --iterations 60 \
      --cooldown-temperature 70 \
      --out "runs/benchmark/e2e/$detector-$model_id.json" \
      "${preview[@]}" || break 2
  done
done
```

Each invocation measures four relation backends at two vocabulary sizes. Across
the nine detector/checkpoint pairings, this is 72 configurations and 12,960 timed
calls with the settings above. Backend/vocabulary order is shuffled each round;
the outer detector/checkpoint order is sequential and appears in the records.
The recorded run checks YOLO26/ViT-S first, then runs the relation-only sweep,
then the remaining detector/checkpoint pairings. No GPU jobs overlap.

The cooling option waits before each warmup block for the requested temperature
and clear thermal-slowdown flags, failing after three minutes if they cannot be
reached. It does **not** guarantee stable clocks or power during the subsequent
block. Inspect the recorded temperature, enforced power limit and thermal
counters before publishing a comparison; cooling alone did not restore the
earlier laptop performance in the [previous run](README.md#conditions-and-interpretation).
Run no competing GPU workloads and retain failed/thermally limited runs as
separate evidence. Use a new output directory for each rerun.

The recorded collection ran GPU jobs sequentially, but host CPU activity was
not isolated. A background filesystem check was observed during the run;
timestamps and the execution order are included in `collection_context` in
both raw source files. The cooled measurements use fresh exports, so changes
from the earlier run must not be attributed to temperature alone.

Before timing, all FP32 relation backends are compared on the detector's actual
boxes for every input image and both vocabularies. The existing engine-hash and
export-parity requirements remain in force. BF16 is a latency baseline, without
a claim of accuracy equivalence. JSON records include both preparation and
runtime Ultralytics versions, detector settings/hashes, all latency samples,
per-round medians, relation parity checks and GPU telemetry. `complete` becomes
true only after every configuration finishes.

Publish these results separately from the relation-only table. The existing
README generator rejects detector-inclusive records so the two scopes cannot
be mixed accidentally. After collecting the nine complete records, assemble the
end-to-end source and regenerate its tables:

```bash
python - <<'PY'
import gzip
import json
from pathlib import Path

records = [json.loads(Path(f"runs/benchmark/e2e/{detector}-{model}.json").read_text())
           for detector in ("yolo26", "yolo-world", "yoloe")
           for model in ("relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16")]
Path("docs/benchmarks/rtx3080-laptop-e2e.json.gz").write_bytes(
    gzip.compress(json.dumps({"records": records}).encode(), mtime=0))
PY
python release/update_readme_e2e.py
python release/update_readme_e2e.py --check
```

CI regenerates the expected tables from raw samples and rejects missing
configurations, inconsistent detector settings, unequal image coverage,
incorrect stage totals, or incomplete FP32 parity evidence.

Upstream model definitions and supported checkpoints:
[YOLO26](https://docs.ultralytics.com/models/yolo26/),
[YOLO-World](https://docs.ultralytics.com/models/yolo-world/),
[YOLOE](https://docs.ultralytics.com/models/yoloe/).
