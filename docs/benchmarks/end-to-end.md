# Detector-to-relations latency

The end-to-end mode of `deploy/bench_gpu_backends.py` measures an image through
detection, relation prediction and triplet decoding. GPU measurements for this
mode are pending; the existing README numbers measure the relation stage only.

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
and software stack. If GPU work is deferred, defer those exports too.

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
      --cooldown-temperature 65 \
      --out "runs/benchmark/e2e/$detector-$model_id.json" \
      "${preview[@]}" || break 2
  done
done
```

Each invocation measures four relation backends at two vocabulary sizes. Across
the nine detector/checkpoint pairings, this is 72 configurations and 12,960 timed
calls with the settings above. Backend/vocabulary order is shuffled each round;
the outer detector/checkpoint order is sequential and appears in the records.

The cooling option waits before each warmup block for the requested temperature
and clear thermal-slowdown flags, failing after three minutes if they cannot be
reached. It does **not** guarantee stable clocks or power during the subsequent
block. Inspect the recorded temperature, enforced power limit and thermal
counters before publishing a comparison; cooling alone did not restore the
earlier laptop performance in the [previous run](README.md#conditions-and-interpretation).
Run no competing GPU workloads and retain failed/thermally limited runs as
separate evidence. Use a new output directory for each rerun.

Before timing, all FP32 relation backends are compared on the detector's actual
boxes for every input image and both vocabularies. The existing engine-hash and
export-parity requirements remain in force. BF16 is a latency baseline, without
a claim of accuracy equivalence. JSON records include both preparation and
runtime Ultralytics versions, detector settings/hashes, all latency samples,
per-round medians, relation parity checks and GPU telemetry. `complete` becomes
true only after every configuration finishes.

Publish these results separately from the relation-only table. The existing
README generator rejects detector-inclusive records so the two scopes cannot
be mixed accidentally.

Upstream model definitions and supported checkpoints:
[YOLO26](https://docs.ultralytics.com/models/yolo26/),
[YOLO-World](https://docs.ultralytics.com/models/yolo-world/),
[YOLOE](https://docs.ultralytics.com/models/yoloe/).
