# GPU latency across the released checkpoints

This benchmark compares ViT-S, ViT-S+ and ViT-B on the same RTX 3080 Laptop
GPU. It measures a warm, batch-one relation prediction, including image/box
preprocessing, CPU/GPU transfers and shared triplet decoding. Detection,
model loading, text encoding and engine building are outside the timed region.

For detector-inclusive timing with COCO YOLO26, YOLO-World and YOLOE, see the
[end-to-end comparison](end-to-end.md).

The [README](../../README.md#performance) highlights the recommended model's latency.
This page retains the detailed results and measurement conditions, including
an earlier pass affected by thermal limiting. Raw JSON, logs and engine files
are local experiment artifacts under `runs/`; they are not versioned.

## Measurements

<!-- BEGIN GENERATED LATENCY -->
Latency in milliseconds; each cell is **median / p95**.

| Checkpoint | Backend | 35 predicates | 243 predicates |
|---|---|---:|---:|
| ViT-S | PyTorch eager FP32 | 21.57 / 23.50 | 21.68 / 22.60 |
| ViT-S | PyTorch eager BF16 | 20.89 / 24.72 | 21.72 / 24.96 |
| ViT-S | ONNX CUDA FP32 | 23.11 / 24.82 | 23.40 / 25.13 |
| ViT-S | TensorRT FP32 | 15.61 / 16.33 | 15.82 / 17.07 |
| ViT-S+ | PyTorch eager FP32 | 24.25 / 25.19 | 24.53 / 27.91 |
| ViT-S+ | PyTorch eager BF16 | 23.39 / 26.07 | 23.79 / 26.71 |
| ViT-S+ | ONNX CUDA FP32 | 26.44 / 27.85 | 26.63 / 27.38 |
| ViT-S+ | TensorRT FP32 | 17.24 / 17.82 | 17.55 / 18.33 |
| ViT-B | PyTorch eager FP32 | 43.46 / 44.17 | 43.52 / 44.23 |
| ViT-B | PyTorch eager BF16 | 23.59 / 25.88 | 24.14 / 25.82 |
| ViT-B | ONNX CUDA FP32 | 46.04 / 46.63 | 46.20 / 46.75 |
| ViT-B | TensorRT FP32 | 34.73 / 35.46 | 34.89 / 35.63 |

TensorRT compared with eager PyTorch at 35 predicates:

| Checkpoint | Change vs. FP32 | Change vs. BF16 |
|---|---:|---:|
| ViT-S | 5.97 ms (27.7%) faster | 5.28 ms (25.3%) faster |
| ViT-S+ | 7.01 ms (28.9%) faster | 6.15 ms (26.3%) faster |
| ViT-B | 8.73 ms (20.1%) faster | 11.14 ms (47.2%) slower |

Engine validation against FP32 ONNX Runtime on CPU:

| Checkpoint | Cases | Largest absolute logit difference |
|---|---:|---:|
| ViT-S | 105 | 0.000314236 |
| ViT-S+ | 105 | 0.000193357 |
| ViT-B | 105 | 0.000058413 |

Before timing, the benchmark also checks each FP32 GPU backend against TensorRT
on every timing image and both vocabulary sizes (identical valid pair sets).

| Checkpoint | PyTorch FP32: largest logit difference | ONNX CUDA FP32: largest logit difference |
|---|---:|---:|
| ViT-S | 0.006183028 | 0.000636578 |
| ViT-S+ | 0.005682230 | 0.000095606 |
| ViT-B | 0.004575253 | 0.000055313 |

Thermal observations across the measurement blocks (warmup included):

| Checkpoint | GPU temperature before / after blocks | Observed power limit | Thermal slowdown counter increase |
|---|---|---|---:|
| ViT-S | 70–70°C / 73–76°C | 90.00 W | 0.0 ms |
| ViT-S+ | 69–70°C / 73–76°C | 90.00 W | 0.0 ms |
| ViT-B | 70–70°C / 76–77°C | 90.00 W | 0.0 ms |

Software: PyTorch 2.14.0+cu130, CUDA 13.0, TensorRT 10.16.1.11, ONNX Runtime 1.30.0.
<!-- END GENERATED LATENCY -->

### Initial pass

This pass ran ViT-S, ViT-S+ and ViT-B in that order, with the 35-predicate
configurations before the 243-predicate configurations. Thermal conditions
changed during the run; differences from the repeat must not be attributed
to the model or vocabulary alone. Values are **median / p95**, in milliseconds.

| Checkpoint | Backend | 35 predicates | 243 predicates |
|---|---|---:|---:|
| ViT-S | PyTorch eager FP32 | 22.69 / 23.54 | 23.14 / 24.02 |
| ViT-S | PyTorch eager BF16 | 20.48 / 21.72 | 21.49 / 25.09 |
| ViT-S | ONNX CUDA FP32 | 24.81 / 25.83 | 25.33 / 26.39 |
| ViT-S | TensorRT FP32 | 16.54 / 17.21 | 16.82 / 17.49 |
| ViT-S+ | PyTorch eager FP32 | 25.77 / 26.66 | 29.15 / 31.20 |
| ViT-S+ | PyTorch eager BF16 | 22.08 / 23.83 | 23.73 / 25.09 |
| ViT-S+ | ONNX CUDA FP32 | 28.65 / 29.74 | 33.43 / 35.56 |
| ViT-S+ | TensorRT FP32 | 18.57 / 19.41 | 20.57 / 21.87 |
| ViT-B | PyTorch eager FP32 | 53.53 / 58.85 | 59.36 / 60.24 |
| ViT-B | PyTorch eager BF16 | 25.45 / 26.36 | 27.81 / 32.02 |
| ViT-B | ONNX CUDA FP32 | 59.19 / 60.64 | 60.25 / 61.42 |
| ViT-B | TensorRT FP32 | 41.62 / 44.07 | 44.00 / 44.74 |

## Conditions and interpretation

- **Inputs:** the six repository photos in `assets/reel/images/`, resized to
  448 × 448. Each has 20 generated boxes, padded to 32, with a 128-pair budget.
  The box seed is fixed and the inputs are identical for all checkpoints.
- **Vocabulary:** 35 default deployment predicates or all 243 bundled
  predicates. All backends receive the same checkpoint-specific embeddings
  and routing weights. This does not time the 19,103-predicate training bank.
- **Sampling:** three rounds per configuration, each with 20 warmup calls and
  60 measured calls. All eight backend/vocabulary combinations are shuffled
  together each round. Checkpoints run sequentially; no engines are built
  during timing. Wall-clock measurements
  synchronize CUDA before and after each call. p95 uses linear interpolation
  across the 180 recorded calls; it is not a confidence interval.
- **Host work:** four PyTorch/ONNX CPU threads and one OpenCV thread. Decoding
  uses the checkpoint's calibration, threshold 0.5 and top-k 20. Image loading
  is outside the timed region, while resizing and box preprocessing are inside.
  GPU runs are sequential, but host activity is not isolated: a background
  filesystem check was observed during collection. Observation timestamps and
  the execution order are retained in the records' `collection_context`.
- **Precision:** FP32 backends disable TF32. The PyTorch BF16 arm uses
  autocast. There is no `torch.compile`, CUDA-graph capture or reduced-precision
  TensorRT arm in this comparison.
- **Artifacts:** all three models are exported locally from the released EMA
  checkpoints with the repository's exporter and predicate banks, including
  the sparse-pair padding fix described below. All TensorRT engines are built
  on this GPU. Export versions and artifact hashes are in the record.
- **Hardware:** laptop clocks and power are not locked. Temperature, clocks, power and
  thermal-throttling counters are recorded before warmup and after timing.
  No clocks, fan settings or power limits are changed by the benchmark. The
  current rerun waits before each block for at most 70°C and clear thermal
  flags. An earlier cooling attempt did not restore performance, and a
  subsequent repeat remained limited to 55 W. The telemetry table reports
  any thermal limiting during the blocks, including warmup. Compare backends
  within a run; do not interpret differences between runs as model changes.
  These measurements describe this machine and workload; they do not establish
  performance on Jetson, other GPUs or an end-to-end detector pipeline.

Each TensorRT engine passes the exporter's numerical check against CPU ONNX:
one synthetic frame plus the six timing photos, five region counts (including
empty/singleton inputs), and three vocabulary sizes. Valid pair sets must match
exactly; logits use `atol=1e-3, rtol=1e-4`. Separate FP32 GPU checks on the timing
inputs use `atol=1e-2, rtol=1e-4`, with the measured differences reported above.
BF16 is a latency baseline here; its accuracy was not evaluated. These checks
are numerical comparisons, not a replacement for dataset-level evaluation.

### Sparse-pair ONNX CUDA correction

The end-to-end preflight exposed duplicate valid pairs from ONNX Runtime 1.30.0
CUDA when a detector supplied few boxes. The exported sampler masked unused
slots with the most-negative float, also used for padding in
[CUDA TopK](https://github.com/microsoft/onnxruntime/blob/v1.30.0/onnxruntime/core/providers/cuda/math/topk_impl.cuh#L50-L67).
PyTorch and TensorRT agreed on the failing input; the earlier CPU ONNX export
checks had also passed. The failed preflight contributed no published latency
samples. Its diagnostic output is kept locally. The sampler
now masks with half the dtype minimum; only the padding sentinel changes, not
model weights or real pair-selection scores. A CUDA regression test covers
empty, singleton, sparse and dense inputs. All graphs and engines in this rerun
are rebuilt with that fix.
Re-export downloaded graphs before measuring the sparse ONNX CUDA path; checking
only the earlier generated-box workload did not expose this failure.

## Reproduce

Use a separate environment with CUDA-enabled PyTorch and TensorRT 10. Follow
the [TensorRT setup](../../deploy/README.md#tensorrt-nvidia-gpu), then replace
the CPU ONNX Runtime package with the GPU package for this comparison:

```bash
pip install -e ".[tensorrt,hub]"
pip uninstall -y onnxruntime
pip install "onnxruntime-gpu==1.30.0"
```

From the repository root, run the following once per model, setting `model_id`
to `relsgg-vits16`, `relsgg-vits16plus` or `relsgg-vitb16`. Run one benchmark
process at a time and avoid competing GPU workloads.

```bash
model_id=relsgg-vits16
bundle_dir="runs/benchmark/gpu/$model_id"
mkdir -p "$bundle_dir"

hf download "maelic/$model_id" model.pth calibration.json \
  --local-dir "checkpoints/$model_id"
cp "deploy/dist/$model_id/predicate_bank.npz" "$bundle_dir/"

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/export_onnx.py \
  --checkpoint "checkpoints/$model_id/model.pth" \
  --vocab-npz "$bundle_dir/predicate_bank.npz" --vocab-mode input \
  --out "$bundle_dir/relateanything.onnx" --check

python deploy/export_tensorrt.py --onnx "$bundle_dir/relateanything.onnx" \
  --check-images assets/reel/images/*.jpg

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/bench_gpu_backends.py \
  --checkpoint "checkpoints/$model_id/model.pth" --bundle "$bundle_dir" \
  --cooldown-temperature 70 \
  --out "runs/benchmark/gpu/$model_id.json"
```

The benchmark refuses an ONNX session without the CUDA provider, an unchecked
TensorRT engine, mismatched ONNX/engine hashes, or different FP32 valid-pair
selections. It writes the raw
samples and marks a record complete only after all configurations finish.
The CLI exposes sampling counts and region count for additional experiments;
keep them identical when comparing models.

After collecting all three records, assemble a local source and regenerate
the current documentation tables. The historical initial-pass table is retained
separately from the generated block:

```bash
python - <<'PY'
import json
from pathlib import Path

models = ["relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16"]
records = [json.loads(Path(f"runs/benchmark/gpu/{model}.json").read_text())
           for model in models]
Path("runs/benchmark/gpu/family.json").write_text(
    json.dumps({"records": records}, indent=2) + "\n")
PY
python release/update_readme_latency.py --source runs/benchmark/gpu/family.json
python release/update_readme_latency.py --source runs/benchmark/gpu/family.json --check
```

The generator recomputes the displayed medians, p95s and savings from raw
samples and rejects incomplete or inconsistent records. Run `--check` against
your local source before publishing updated tables. CI tests the reporting
logic with synthetic data and does not require benchmark artifacts.
