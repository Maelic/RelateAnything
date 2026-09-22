# GPU latency across the released checkpoints

This benchmark compares ViT-S, ViT-S+ and ViT-B on the same RTX 3080 Laptop
GPU. It measures a warm, batch-one relation prediction, including image/box
preprocessing, CPU/GPU transfers and shared triplet decoding. Detection,
model loading, text encoding and engine building are outside the timed region.

The [README](../../README.md#performance) shows the median comparison.
The [complete record](rtx3080-laptop-family.json) includes every timed call,
per-round medians, software versions, input and artifact hashes, seeds and
numerical checks. The [earlier ViT-S+ run](rtx3080-laptop.json) is retained for
provenance. The [first family pass](rtx3080-laptop-family-first-pass.json) is
also retained: thermal conditions changed during that pass and the GPU
reported throttling during ViT-B. The headline table uses a repeat with shuffled
backend/vocabulary combinations and a recorded power limit for every block.

## Measurements

<!-- BEGIN GENERATED LATENCY -->
Latency in milliseconds; each cell is **median / p95**.

| Checkpoint | Backend | 35 predicates | 243 predicates |
|---|---|---:|---:|
| ViT-S | PyTorch eager FP32 | 33.17 / 34.22 | 33.38 / 35.37 |
| ViT-S | PyTorch eager BF16 | 23.91 / 25.43 | 23.93 / 27.38 |
| ViT-S | ONNX CUDA FP32 | 36.69 / 39.29 | 37.06 / 39.29 |
| ViT-S | TensorRT FP32 | 27.07 / 30.34 | 27.27 / 30.84 |
| ViT-S+ | PyTorch eager FP32 | 39.19 / 40.69 | 38.86 / 40.43 |
| ViT-S+ | PyTorch eager BF16 | 21.72 / 22.72 | 22.34 / 24.32 |
| ViT-S+ | ONNX CUDA FP32 | 42.97 / 45.52 | 43.00 / 46.42 |
| ViT-S+ | TensorRT FP32 | 31.35 / 34.19 | 31.21 / 34.97 |
| ViT-B | PyTorch eager FP32 | 76.19 / 79.54 | 75.26 / 79.14 |
| ViT-B | PyTorch eager BF16 | 28.32 / 29.48 | 28.44 / 29.76 |
| ViT-B | ONNX CUDA FP32 | 83.83 / 87.20 | 82.02 / 88.30 |
| ViT-B | TensorRT FP32 | 69.45 / 73.07 | 69.27 / 73.58 |

TensorRT compared with eager PyTorch at 35 predicates:

| Checkpoint | Change vs. FP32 | Change vs. BF16 |
|---|---:|---:|
| ViT-S | 6.10 ms (18.4%) faster | 3.16 ms (13.2%) slower |
| ViT-S+ | 7.84 ms (20.0%) faster | 9.63 ms (44.4%) slower |
| ViT-B | 6.74 ms (8.8%) faster | 41.13 ms (145.3%) slower |

Engine validation against FP32 ONNX Runtime on CPU:

| Checkpoint | Cases | Largest absolute logit difference |
|---|---:|---:|
| ViT-S | 105 | 0.000313997 |
| ViT-S+ | 105 | 0.000234604 |
| ViT-B | 105 | 0.000061035 |

Before timing, the benchmark also checks each FP32 GPU backend against TensorRT
on every timing image and both vocabulary sizes (identical valid pair sets).

| Checkpoint | PyTorch FP32: largest logit difference | ONNX CUDA FP32: largest logit difference |
|---|---:|---:|
| ViT-S | 0.006182075 | 0.000636101 |
| ViT-S+ | 0.005680799 | 0.000077248 |
| ViT-B | 0.004575253 | 0.000055313 |

Thermal observations across the measurement blocks (warmup included):

| Checkpoint | GPU temperature before / after blocks | Observed power limit | Thermal slowdown counter increase |
|---|---|---|---:|
| ViT-S | 81–82°C / 81–82°C | 55.00 W | 59080.6 ms |
| ViT-S+ | 79–83°C / 79–83°C | 55.00 W | 65978.7 ms |
| ViT-B | 71–79°C / 72–79°C | 55.00 W | 124059.4 ms |

Software: PyTorch 2.14.0+cu130, CUDA 13.0, TensorRT 10.16.1.11, ONNX Runtime 1.30.0.

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
<!-- END GENERATED LATENCY -->

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
- **Precision:** FP32 backends disable TF32. The PyTorch BF16 arm uses
  autocast. There is no `torch.compile`, CUDA-graph capture or reduced-precision
  TensorRT arm in this comparison.
- **Artifacts:** ViT-S+ uses its published ONNX graph. ViT-S and ViT-B are
  exported locally from the released EMA checkpoints with the repository's
  exporter and predicate banks. All TensorRT engines are built on this GPU.
  Export versions and artifact hashes are in the record.
- **Hardware:** laptop clocks and power are not locked. Temperature, clocks, power and
  thermal-throttling counters are recorded before warmup and after timing.
  The laptop was in its **balanced** power profile. The repeat observed a
  reduced enforced GPU power limit. An [attempted repeat with
  cooling intervals](rtx3080-laptop-cooling-attempt.json) did not restore the
  earlier performance, so the final
  comparison runs all three models in the observed limited state without
  cooling waits. The telemetry table reports
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

if [ "$model_id" = relsgg-vits16plus ]; then
  hf download "maelic/$model_id" relateanything.onnx relateanything.json \
    --local-dir "$bundle_dir"
else
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/export_onnx.py \
    --checkpoint "checkpoints/$model_id/model.pth" \
    --vocab-npz "$bundle_dir/predicate_bank.npz" --vocab-mode input \
    --out "$bundle_dir/relateanything.onnx" --check
fi

python deploy/export_tensorrt.py --onnx "$bundle_dir/relateanything.onnx" \
  --check-images assets/reel/images/*.jpg

OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python deploy/bench_gpu_backends.py \
  --checkpoint "checkpoints/$model_id/model.pth" --bundle "$bundle_dir" \
  --out "runs/benchmark/gpu/$model_id.json"
```

The benchmark refuses an ONNX session without the CUDA provider, an unchecked
TensorRT engine, mismatched ONNX/engine hashes, or different FP32 valid-pair
selections. It writes the raw
samples and marks a record complete only after all configurations finish.
The CLI exposes sampling counts and region count for additional experiments;
keep them identical when comparing models.

After collecting all three records, assemble the checked-in source and
regenerate both documentation tables:

```bash
python - <<'PY'
import json
from pathlib import Path

models = ["relsgg-vits16", "relsgg-vits16plus", "relsgg-vitb16"]
records = [json.loads(Path(f"runs/benchmark/gpu/{model}.json").read_text())
           for model in models]
Path("docs/benchmarks/rtx3080-laptop-family.json").write_text(
    json.dumps({"records": records}, indent=2) + "\n")
PY
python release/update_readme_latency.py
python release/update_readme_latency.py --check
```

The generator recomputes the displayed medians, p95s and savings from raw
samples. CI checks both pages and rejects incomplete or inconsistent records.
