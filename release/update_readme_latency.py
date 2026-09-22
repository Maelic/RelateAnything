"""Render the README latency comparison from the recorded per-call measurements.

Run without arguments to update README.md; --check fails if the block is stale.
The benchmark record includes its hardware, versions, scope and raw samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/benchmarks/rtx3080-laptop.json"
START = "<!-- BEGIN GENERATED LATENCY -->"
END = "<!-- END GENERATED LATENCY -->"


def render(data):
    rows = {(r["backend"], r["predicates"]): r for r in data["rows"]}
    if len(rows) != len(data["rows"]):
        raise ValueError("Duplicate backend/vocabulary measurements")

    def median(backend, predicates=35):
        row = rows[backend, predicates]
        samples = row["samples_ms"]
        if len(samples) != row["n"] or not all(x > 0 for x in samples):
            raise ValueError(f"Invalid samples for {backend}")
        return statistics.median(samples)

    order = [
        "PyTorch eager FP32",
        "PyTorch eager BF16",
        "ONNX CUDA FP32",
        "TensorRT FP32",
    ]
    lines = [
        f"Warm median latency for `{data['checkpoint'].split('/')[-1]}` on an",
        f"**{data['gpu'].removeprefix('NVIDIA GeForce ')}** with an {data['cpu']}.",
        "The columns use the default deployment vocabulary and the complete bundled",
        "predicate bank, respectively.",
        "",
        "| Backend | 35 predicates | 243 predicates |",
        "|---|---:|---:|",
    ]
    for backend in order:
        values = [
            backend,
            f"{median(backend, 35):.1f} ms",
            f"{median(backend, 243):.1f} ms",
        ]
        if backend == "TensorRT FP32":
            values = [f"**{v}**" for v in values]
        lines.append("| " + " | ".join(values) + " |")

    trt = median("TensorRT FP32")
    fp32 = median("PyTorch eager FP32")
    bf16 = median("PyTorch eager BF16")
    lines.extend(
        [
            "",
            f"At 35 predicates, TensorRT saves **{fp32 - trt:.1f} ms ({100 * (1 - trt / fp32):.0f}%)** compared with",
            f"eager PyTorch FP32, and **{bf16 - trt:.1f} ms ({100 * (1 - trt / bf16):.1f}%)** compared with eager PyTorch BF16.",
            "",
            f"Batch {data['batch']}, {data['image_size']} × {data['image_size']} input, {data['valid_boxes']} regions padded to {data['padded_boxes']}, {data['pair_budget']} candidate pairs.",
            "Includes preprocessing, CPU/GPU transfers and decoding; **excludes the",
            "detector, model loading and engine building**. Each configuration uses",
            f"{data['rounds'] * data['timed_iterations_per_round']} timed calls across {data['rounds']} shuffled rounds after warmup, over {data['images']} images",
            f"with generated boxes. FP32 runs have TF32 disabled; `torch.compile` was not timed.",
            "",
            f"[Versions, method and raw samples](docs/benchmarks/rtx3080-laptop.json) ·",
            "[TensorRT setup](deploy/README.md#tensorrt-nvidia-gpu). These are local",
            "deployment measurements, not a dataset-wide accuracy evaluation.",
        ]
    )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "README.md"
    original = path.read_text()
    if original.count(START) != 1 or original.count(END) != 1:
        raise ValueError("Expected exactly one latency block in README.md")
    before, remainder = original.split(START)
    _, after = remainder.split(END)
    updated = (
        before
        + START
        + "\n"
        + render(json.loads(SOURCE.read_text()))
        + "\n"
        + END
        + after
    )
    if args.check:
        if updated != original:
            raise SystemExit(
                "README latency block is stale; run python release/update_readme_latency.py"
            )
        print("README latency block matches the measured samples")
    else:
        path.write_text(updated)
        print("Updated README latency block")


if __name__ == "__main__":
    main()
