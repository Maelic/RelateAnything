"""Generate README medians and detailed GPU latency tables from raw samples.

Run without arguments to update both pages; --check fails if either is stale.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/benchmarks/rtx3080-laptop-family.json"
START = "<!-- BEGIN GENERATED LATENCY -->"
END = "<!-- END GENERATED LATENCY -->"
MODELS = {
    "maelic/relsgg-vits16": "ViT-S",
    "maelic/relsgg-vits16plus": "ViT-S+",
    "maelic/relsgg-vitb16": "ViT-B",
}
ARMS = ["PyTorch eager FP32", "PyTorch eager BF16", "ONNX CUDA FP32", "TensorRT FP32"]


def load_records():
    records = json.loads(SOURCE.read_text())["records"]
    if len(records) != len(MODELS) or {r["checkpoint"] for r in records} != set(MODELS):
        raise ValueError(
            "Expected exactly one completed record per released checkpoint"
        )
    records.sort(key=lambda r: list(MODELS).index(r["checkpoint"]))
    shared = [
        "gpu",
        "cpu",
        "torch",
        "cuda",
        "tensorrt",
        "onnxruntime",
        "batch",
        "image_size",
        "valid_boxes",
        "padded_boxes",
        "pair_budget",
        "images",
        "threads",
        "opencv_threads",
        "rounds",
        "warmup_per_round",
        "timed_iterations_per_round",
        "tf32",
        "torch_compile",
        "input_images",
        "box_seed",
        "arm_shuffle_seeds",
        "ort_cuda_options",
        "cooldown_temperature_c",
        "timing",
    ]
    for record in records:
        if not record["complete"]:
            raise ValueError("Cannot publish an incomplete benchmark")
        if record.get("detector") is not None:
            raise ValueError(
                "End-to-end detector records cannot populate relation-only tables"
            )
        if any(record[key] != records[0][key] for key in shared):
            raise ValueError("Benchmark conditions differ between checkpoints")
        rows = {(r["backend"], r["predicates"]): r for r in record["rows"]}
        if len(rows) != len(record["rows"]) or set(rows) != {
            (a, v) for a in ARMS for v in (35, 243)
        }:
            raise ValueError("Expected four backends at both vocabulary sizes")
        for row in rows.values():
            samples = row["samples_ms"]
            expected = record["rounds"] * record["timed_iterations_per_round"]
            if (
                len(samples) != row["n"]
                or row["n"] != expected
                or not all(math.isfinite(x) and x > 0 for x in samples)
            ):
                raise ValueError("Invalid latency samples")
    return records


def stats(record, backend, predicates):
    row = next(
        r
        for r in record["rows"]
        if r["backend"] == backend and r["predicates"] == predicates
    )
    values = sorted(row["samples_ms"])
    # Linear interpolation, matching numpy.percentile(..., 95).
    index = (len(values) - 1) * 0.95
    lo, hi = math.floor(index), math.ceil(index)
    p95 = values[lo] + (values[hi] - values[lo]) * (index - lo)
    return statistics.median(values), p95


def thermal_delta(block):
    total = 0
    for kind in ("sw_thermal_slowdown", "hw_thermal_slowdown"):
        key = "clocks_event_reasons_counters." + kind
        before = int(block["gpu_before_warmup"][key].split()[0])
        after = int(block["gpu_after_timing"][key].split()[0])
        if after < before:
            raise ValueError("Thermal counter reset during a measurement block")
        total += after - before
    return total


def render_summary(records):
    data = records[0]
    lines = [
        f"Warm median latency on an **{data['gpu'].removeprefix('NVIDIA GeForce ')}**",
        f"with an {data['cpu']}. All three released checkpoints use the same inputs,",
        "vocabularies and host-side decoding.",
        "",
    ]
    limits = sorted(
        {
            block[phase]["enforced.power.limit"]
            for record in records
            for block in record["measurement_blocks"]
            for phase in ("gpu_before_warmup", "gpu_after_timing")
        }
    )
    limited = any(
        thermal_delta(block) > 0
        for record in records
        for block in record["measurement_blocks"]
    )
    lines += [
        f"Observed GPU power limit: **{', '.join(limits)}**.",
        (
            "**Thermal limiting was recorded during this run.** The detailed report retains"
            if limited
            else "No thermal slowdown counter increase was recorded. The detailed report retains"
        ),
        "the earlier pass to show how the laptop's operating state affects latency.",
        "",
    ]
    for count, label in [
        (35, "default deployment vocabulary"),
        (243, "complete bundled predicate bank"),
    ]:
        lines += [
            f"**{count} predicates — {label}**",
            "",
            "| Backend | ViT-S | ViT-S+ | ViT-B |",
            "|---|---:|---:|---:|",
        ]
        for arm in ARMS:
            values = [arm]
            for record in records:
                value = stats(record, arm, count)[0]
                cell = f"{value:.1f} ms"
                if value == min(stats(record, a, count)[0] for a in ARMS):
                    cell = f"**{cell}**"
                values.append(cell)
            lines.append("| " + " | ".join(values) + " |")
        lines.append("")
    savings = [
        100
        * (1 - stats(r, "TensorRT FP32", 35)[0] / stats(r, "PyTorch eager FP32", 35)[0])
        for r in records
    ]
    lines += [
        "At 35 predicates, TensorRT reduces median latency versus eager PyTorch FP32",
        "by "
        + ", ".join(
            f"**{value:.0f}% ({MODELS[r['checkpoint']]})**"
            for r, value in zip(records, savings)
        )
        + ".",
        "",
        f"Batch {data['batch']}, {data['image_size']} × {data['image_size']} input, {data['valid_boxes']} regions padded to {data['padded_boxes']}, {data['pair_budget']} candidate pairs.",
        "Includes preprocessing, CPU/GPU transfers and decoding; **excludes the",
        "detector, model loading and engine building**. Each configuration uses",
        f"{data['rounds'] * data['timed_iterations_per_round']} timed calls across {data['rounds']} shuffled rounds after warmup, over {data['images']} images",
        "with generated boxes.",
        "FP32 runs have TF32 disabled; `torch.compile` was not timed. Bold marks the",
        "lowest median in each column; BF16 accuracy was not evaluated here.",
        "",
        "[p95 latency, validation and reproduction](docs/benchmarks/README.md) ·",
        "[Raw samples](docs/benchmarks/rtx3080-laptop-family.json) ·",
        "[TensorRT setup](deploy/README.md#tensorrt-nvidia-gpu). These are local",
        "deployment measurements, not a dataset-wide accuracy evaluation.",
    ]
    return "\n".join(lines)


def render_details(records):
    lines = [
        "Latency in milliseconds; each cell is **median / p95**.",
        "",
        "| Checkpoint | Backend | 35 predicates | 243 predicates |",
        "|---|---|---:|---:|",
    ]
    for record in records:
        for arm in ARMS:
            cells = [MODELS[record["checkpoint"]], arm]
            cells += [
                " / ".join(f"{value:.2f}" for value in stats(record, arm, count))
                for count in (35, 243)
            ]
            lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "TensorRT compared with eager PyTorch at 35 predicates:",
        "",
        "| Checkpoint | Change vs. FP32 | Change vs. BF16 |",
        "|---|---:|---:|",
    ]
    for record in records:
        trt = stats(record, "TensorRT FP32", 35)[0]
        cells = [MODELS[record["checkpoint"]]]
        for arm in ARMS[:2]:
            baseline = stats(record, arm, 35)[0]
            direction = "faster" if trt < baseline else "slower"
            cells.append(
                f"{abs(baseline - trt):.2f} ms ({abs(100 * (1 - trt / baseline)):.1f}%) {direction}"
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Engine validation against FP32 ONNX Runtime on CPU:",
        "",
        "| Checkpoint | Cases | Largest absolute logit difference |",
        "|---|---:|---:|",
    ]
    for record in records:
        report = record["engine_validation"]
        lines.append(
            f"| {MODELS[record['checkpoint']]} | {report['cases']} | {report['max_abs_logit_delta']:.9f} |"
        )
    lines += [
        "",
        "Before timing, the benchmark also checks each FP32 GPU backend against TensorRT",
        "on every timing image and both vocabulary sizes (identical valid pair sets).",
        "",
        "| Checkpoint | PyTorch FP32: largest logit difference | ONNX CUDA FP32: largest logit difference |",
        "|---|---:|---:|",
    ]
    for record in records:
        cells = [MODELS[record["checkpoint"]]]
        for arm in ["PyTorch eager FP32", "ONNX CUDA FP32"]:
            checks = [
                r for r in record["fp32_cross_backend_checks"] if r["backend"] == arm
            ]
            if len(checks) != record["images"] * 2 or not all(
                r["same_valid_pairs"] for r in checks
            ):
                raise ValueError("Incomplete FP32 cross-backend validation")
            cells.append(f"{max(r['max_logit_delta'] for r in checks):.9f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Thermal observations across the measurement blocks (warmup included):",
        "",
        "| Checkpoint | GPU temperature before / after blocks | Observed power limit | Thermal slowdown counter increase |",
        "|---|---|---|---:|",
    ]
    for record in records:
        blocks = record["measurement_blocks"]
        if len(blocks) != record["rounds"] * len(ARMS) * 2:
            raise ValueError("Missing measurement-block telemetry")
        temps = []
        for phase in ("gpu_before_warmup", "gpu_after_timing"):
            values = [int(block[phase]["temperature.gpu"]) for block in blocks]
            temps.append(f"{min(values)}–{max(values)}°C")
        thermal_us = 0
        power_limits = set()
        for block in blocks:
            for phase in ("gpu_before_warmup", "gpu_after_timing"):
                power_limits.add(block[phase]["enforced.power.limit"])
            thermal_us += thermal_delta(block)
        lines.append(
            f"| {MODELS[record['checkpoint']]} | {' / '.join(temps)} | {', '.join(sorted(power_limits))} | {thermal_us / 1000:.1f} ms |"
        )
    data = records[0]
    lines += [
        "",
        f"Software: PyTorch {data['torch']}, CUDA {data['cuda']}, "
        f"TensorRT {data['tensorrt']}, ONNX Runtime {data['onnxruntime']}.",
    ]
    initial = json.loads(
        (ROOT / "docs/benchmarks/rtx3080-laptop-family-first-pass.json").read_text()
    )["records"]
    lines += [
        "",
        "### Initial pass",
        "",
        "This pass ran ViT-S, ViT-S+ and ViT-B in that order, with the 35-predicate",
        "configurations before the 243-predicate configurations. Thermal conditions",
        "changed during the run; differences from the repeat must not be attributed",
        "to the model or vocabulary alone. Values are **median / p95**, in milliseconds.",
        "",
        "| Checkpoint | Backend | 35 predicates | 243 predicates |",
        "|---|---|---:|---:|",
    ]
    for record in initial:
        for arm in ARMS:
            cells = [MODELS[record["checkpoint"]], arm]
            cells += [
                " / ".join(f"{value:.2f}" for value in stats(record, arm, count))
                for count in (35, 243)
            ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_block(path, body, check):
    original = path.read_text()
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError(f"Expected exactly one latency block in {path}")
    before, remainder = original.split(START)
    _, after = remainder.split(END)
    updated = before + START + "\n" + body + "\n" + END + after
    if check:
        if updated != original:
            raise SystemExit(
                f"{path.relative_to(ROOT)} is stale; run python release/update_readme_latency.py"
            )
        print(f"{path.relative_to(ROOT)} matches the measured samples")
    else:
        path.write_text(updated)
        print(f"Updated {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = load_records()
    update_block(ROOT / "README.md", render_summary(records), args.check)
    update_block(
        ROOT / "docs/benchmarks/README.md", render_details(records), args.check
    )


if __name__ == "__main__":
    main()
