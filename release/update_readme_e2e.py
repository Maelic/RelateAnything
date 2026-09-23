"""Generate detector-inclusive latency tables from checked, complete raw records."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from release.update_readme_latency import ARMS, MODELS, thermal_delta, update_block

START = "<!-- BEGIN GENERATED E2E LATENCY -->"
END = "<!-- END GENERATED E2E LATENCY -->"
DETECTORS = {"yolo26": "YOLO26m", "yolo-world": "YOLO-World v2-m", "yoloe": "YOLOE-26m"}
VOCABS = (35, 243)


def load_records(source):
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt") as stream:
        records = json.load(stream)["records"]
    expected = {(detector, model) for detector in DETECTORS for model in MODELS}
    keys = [(r["detector"]["family"], r["checkpoint"]) for r in records]
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Expected every detector/checkpoint combination exactly once")
    records.sort(
        key=lambda r: (
            list(DETECTORS).index(r["detector"]["family"]),
            list(MODELS).index(r["checkpoint"]),
        )
    )
    shared = (
        "gpu",
        "cpu",
        "torch",
        "cuda",
        "tensorrt",
        "onnxruntime",
        "ultralytics",
        "batch",
        "image_size",
        "max_relation_boxes",
        "padded_boxes",
        "pair_budget",
        "input_images",
        "images",
        "threads",
        "opencv_threads",
        "rounds",
        "warmup_per_round",
        "timed_iterations_per_round",
        "tf32",
        "torch_compile",
        "arm_shuffle_seeds",
        "ort_cuda_options",
        "cooldown_temperature_c",
        "timing",
        "stage_timing",
    )
    detector_metadata = {}
    for record in records:
        if not record["complete"]:
            raise ValueError("Cannot publish an incomplete end-to-end benchmark")
        if any(record[key] != records[0][key] for key in shared):
            raise ValueError("End-to-end benchmark conditions differ")
        detector = record["detector"]
        if any(
            detector[key] != records[0]["detector"][key]
            for key in (
                "classes",
                "predict",
                "precision",
                "runtime",
                "size",
                "masks_used_by_relation",
            )
        ):
            raise ValueError(
                "Detector vocabulary or inference settings differ between families"
            )
        if len(detector["classes"]) != 80 or detector["masks_used_by_relation"]:
            raise ValueError(
                "Expected COCO-80 detectors supplying boxes to the relation model"
            )
        if (
            detector["runtime"] != "pytorch"
            or detector["precision"] != "fp32"
            or detector["size"] != "m"
        ):
            raise ValueError("Expected medium PyTorch FP32 detectors")
        family = detector["family"]
        if detector != detector_metadata.setdefault(family, detector):
            raise ValueError("Detector configuration changed between checkpoints")
        configurations = {(arm, vocab) for arm in ARMS for vocab in VOCABS}
        rows = {(row["backend"], row["predicates"]): row for row in record["rows"]}
        if len(rows) != len(record["rows"]) or set(rows) != configurations:
            raise ValueError("Expected four relation backends at both vocabulary sizes")
        expected_n = record["rounds"] * record["timed_iterations_per_round"]
        if record["timed_iterations_per_round"] % record["images"]:
            raise ValueError("Every image must be timed equally")
        for row in rows.values():
            samples = row["pipeline_samples"]
            if len(samples) != expected_n or row["n"] != expected_n:
                raise ValueError("Missing pipeline samples")
            if [s["total_ms"] for s in samples] != row["samples_ms"]:
                raise ValueError("Pipeline and overall totals disagree")
            coverage = Counter((s["round"], s["image_index"]) for s in samples)
            if coverage != Counter(
                {
                    (r, i): record["timed_iterations_per_round"] // record["images"]
                    for r in range(1, record["rounds"] + 1)
                    for i in range(record["images"])
                }
            ):
                raise ValueError("Unequal per-image or per-round coverage")
            for sample in samples:
                stages = [
                    sample[key + "_ms"] for key in ("detector", "relation", "decode")
                ]
                if (
                    not all(math.isfinite(x) and x >= 0 for x in stages)
                    or not math.isfinite(sample["total_ms"])
                    or sample["total_ms"] <= 0
                    or sum(stages) > sample["total_ms"] + 1e-6
                ):
                    raise ValueError("Invalid stage timing")
                used = sample["used_boxes"]
                if (
                    not 0
                    <= used
                    <= min(sample["detected_boxes"], record["max_relation_boxes"])
                ):
                    raise ValueError("Invalid detected/used box counts")
                if sample["relation_skipped"] != (used < 2):
                    raise ValueError("Inconsistent skipped relation flag")
                if sample["relation_skipped"] and (
                    sample["relation_ms"] or sample["decode_ms"] or sample["triplets"]
                ):
                    raise ValueError("Skipped relations contain measured relation work")
        blocks = record["measurement_blocks"]
        if Counter(
            (b["backend"], b["predicates"], b["round"]) for b in blocks
        ) != Counter(
            {
                (arm, vocab, r): 1
                for arm, vocab in configurations
                for r in range(1, record["rounds"] + 1)
            }
        ):
            raise ValueError("Missing measurement-block telemetry")
        for block in blocks:
            thermal_delta(block)
        checks = record["fp32_cross_backend_checks"]
        expected_checks = {
            (arm, vocab, Path(image["path"]).name)
            for arm in (ARMS[0], ARMS[2])
            for vocab in VOCABS
            for image in record["input_images"]
        }
        if (
            len(checks) != len(expected_checks)
            or {(c["backend"], c["predicates"], c["image"]) for c in checks}
            != expected_checks
        ):
            raise ValueError("Incomplete FP32 parity checks")
        if not all(
            c["same_valid_pairs"] and math.isfinite(c["max_logit_delta"])
            for c in checks
        ):
            raise ValueError("Failed FP32 parity checks")
    return records


def samples(record, arm, vocab):
    return next(
        row["pipeline_samples"]
        for row in record["rows"]
        if row["backend"] == arm and row["predicates"] == vocab
    )


def stats(values):
    values = sorted(values)
    if not values:
        return None
    index = (len(values) - 1) * 0.95
    lo, hi = math.floor(index), math.ceil(index)
    return statistics.median(values), values[lo] + (values[hi] - values[lo]) * (
        index - lo
    )


def cell(values):
    result = stats(values)
    return "—" if result is None else " / ".join(f"{value:.2f}" for value in result)


def all_samples(record):
    return [sample for row in record["rows"] for sample in row["pipeline_samples"]]


def thermal_note(records):
    blocks = [block for record in records for block in record["measurement_blocks"]]
    limits = sorted(
        {
            b[phase]["enforced.power.limit"]
            for b in blocks
            for phase in ("gpu_before_warmup", "gpu_after_timing")
        }
    )
    limited = sum(thermal_delta(b) for b in blocks) > 0
    return f"Observed enforced GPU power limits: **{', '.join(limits)}**. " + (
        "**Thermal slowdown was recorded**; inspect the per-run telemetry below."
        if limited
        else "No thermal slowdown counter increase was recorded during the measured blocks (warmup included)."
    )


def render_overview(records):
    first = records[0]
    lines = [
        f"Warm **end-to-end median latency** on the {first['gpu'].removeprefix('NVIDIA GeForce ')}:",
        "detection, relation preprocessing/inference and triplet decoding, including transfers.",
        "Detectors run **PyTorch FP32 throughout**; columns select the **relation backend**.",
        "YOLOE includes segmentation computation, with boxes passed to RelateAnything.",
        "",
        "| Detector | Relation model | PyTorch FP32 | PyTorch BF16 | ONNX CUDA FP32 | TensorRT FP32 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        cells = [DETECTORS[record["detector"]["family"]], MODELS[record["checkpoint"]]]
        medians = [
            stats([s["total_ms"] for s in samples(record, arm, VOCABS[0])])[0]
            for arm in ARMS
        ]
        cells += [
            f"**{value:.1f} ms**" if value == min(medians) else f"{value:.1f} ms"
            for value in medians
        ]
        lines.append("| " + " | ".join(cells) + " |")
    count = sum(row["n"] for record in records for row in record["rows"])
    boxes = [s["used_boxes"] for record in records for s in all_samples(record)]
    skipped = sum(
        s["relation_skipped"] for record in records for s in all_samples(record)
    )
    lines += [
        "",
        f"Shown: {VOCABS[0]} predicates, batch {first['batch']}, {first['images']} photos, COCO-80 objects.",
        f"The full sweep contains **{count:,} timed calls** across {len(records) * len(ARMS) * len(VOCABS)} configurations, including {VOCABS[1]} predicates.",
        f"Retained detections ranged from {min(boxes)} to {max(boxes)} per image; {skipped} calls skipped relation inference.",
        *(
            [
                f"Blocks start at ≤{first['cooldown_temperature_c']}°C; cooling waits are excluded from latency."
            ]
            if first.get("cooldown_temperature_c") is not None
            else []
        ),
        "Image loading, display, prompt encoding and model/engine setup are excluded. This is serial latency, not pipelined video throughput.",
        "FP32 runs have TF32 disabled; BF16 accuracy was not evaluated.",
        "",
        thermal_note(records),
    ]
    return "\n".join(lines)


def render_summary(records):
    record = next(
        r
        for r in records
        if r["checkpoint"] == "maelic/relsgg-vits16plus"
        and r["detector"]["family"] == "yolo26"
    )
    latency = stats(
        [s["total_ms"] for s in samples(record, "TensorRT FP32", VOCABS[0])]
    )[0]
    return "\n".join(
        [
            f"With **{DETECTORS['yolo26']} detection included**, the full pipeline takes **{latency:.1f} ms per image**.",
            "Detection uses PyTorch FP32; ViT-S+ relations use TensorRT FP32.",
            "",
            f"*Warm median latency, batch {record['batch']}, {VOCABS[0]} relation types, {record['images']} sample images; loading and setup excluded.*",
            "",
            "[All models and runtimes](docs/benchmarks/README.md) ·",
            "[YOLO26, YOLO-World and YOLOE comparison](docs/benchmarks/end-to-end.md) ·",
            "[Paper pipeline benchmarks](docs/deployment.md#realtime-pipeline)",
        ]
    )


def render_details(records):
    lines = [
        render_overview(records),
        "",
        "### Stage breakdowns",
        "",
        "All latency cells show **median / p95 in milliseconds**, recomputed from raw calls.",
        "Total percentiles include orchestration and are measured directly, not summed from stage percentiles.",
        "",
    ]
    for family, label in DETECTORS.items():
        family_samples = [
            s
            for r in records
            if r["detector"]["family"] == family
            for s in all_samples(r)
        ]
        skipped = sum(s["relation_skipped"] for s in family_samples)
        columns = "| Relation model | Relation backend | Predicates | Detector | Relations | Decode | Total |"
        separator = "|---|---|---:|---:|---:|---:|---:|"
        if skipped:
            columns += " Skipped calls | Active-frame total |"
            separator += "---:|---:|"
        lines += [
            f"#### {label}",
            "",
            f"{skipped} of {len(family_samples):,} calls skipped relation inference."
            + (
                " Active-frame totals exclude those calls."
                if skipped
                else " Active-frame and overall totals are identical."
            ),
            "",
            columns,
            separator,
        ]
        for record in records:
            if record["detector"]["family"] != family:
                continue
            for vocab in VOCABS:
                for arm in ARMS:
                    measured = samples(record, arm, vocab)
                    cells = [MODELS[record["checkpoint"]], arm, str(vocab)]
                    cells += [
                        cell([s[stage + "_ms"] for s in measured])
                        for stage in ("detector", "relation", "decode", "total")
                    ]
                    if skipped:
                        cells += [
                            str(sum(s["relation_skipped"] for s in measured)),
                            cell(
                                [
                                    s["total_ms"]
                                    for s in measured
                                    if not s["relation_skipped"]
                                ]
                            ),
                        ]
                    lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "### Workload and validation",
        "",
        "Detected / retained box-count ranges, across every timed call for each image:",
        "",
        "| Image | YOLO26m | YOLO-World v2-m | YOLOE-26m |",
        "|---|---:|---:|---:|",
    ]
    for i, image in enumerate(records[0]["input_images"]):
        cells = [Path(image["path"]).name]
        for family in DETECTORS:
            measured = [
                s
                for r in records
                if r["detector"]["family"] == family
                for s in all_samples(r)
                if s["image_index"] == i
            ]
            ranges = []
            for key in ("detected_boxes", "used_boxes"):
                values = [s[key] for s in measured]
                ranges.append(
                    str(min(values))
                    if min(values) == max(values)
                    else f"{min(values)}–{max(values)}"
                )
            cells.append(" / ".join(ranges))
        lines.append("| " + " | ".join(cells) + " |")
    checks = [c for r in records for c in r["fp32_cross_backend_checks"]]
    lines += [
        "",
        f"All {len(checks)} FP32 preflight comparisons passed with identical valid pair sets.",
        f"Largest absolute logit difference: {max(c['max_logit_delta'] for c in checks):.9f}.",
        "Tolerances and per-image results are saved in local result files. BF16 accuracy was not evaluated.",
        "",
        "### GPU conditions",
        "",
        "Temperature ranges before warmup / after timing, and thermal counters including warmup:",
        "",
        "| Detector | Relation model | GPU temperature | Enforced power limit | Thermal counter increase | Cooling waits |",
        "|---|---|---|---|---:|---:|",
    ]
    for record in records:
        blocks = record["measurement_blocks"]
        temps = []
        for phase in ("gpu_before_warmup", "gpu_after_timing"):
            values = [int(b[phase]["temperature.gpu"]) for b in blocks]
            temps.append(f"{min(values)}–{max(values)}°C")
        limits = sorted(
            {
                b[phase]["enforced.power.limit"]
                for b in blocks
                for phase in ("gpu_before_warmup", "gpu_after_timing")
            }
        )
        lines.append(
            f"| {DETECTORS[record['detector']['family']]} | {MODELS[record['checkpoint']]} | {' / '.join(temps)} | {', '.join(limits)} | {sum(thermal_delta(b) for b in blocks) / 1000:.1f} ms | {sum(b['cooling_seconds'] for b in blocks):.1f} s |"
        )
    first = records[0]
    lines += [
        "",
        f"Software: PyTorch {first['torch']}, CUDA {first['cuda']}, TensorRT {first['tensorrt']}, ONNX Runtime {first['onnxruntime']}, Ultralytics {first['ultralytics']}.",
        f"CPU: {first['cpu']}. PyTorch/ONNX CPU threads: {first['threads']}; OpenCV threads: {first['opencv_threads']}.",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="local benchmark records JSON"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    records = load_records(args.source)
    update_block(ROOT / "README.md", render_summary(records), args.check, START, END)
    update_block(
        ROOT / "docs/benchmarks/end-to-end.md",
        render_details(records),
        args.check,
        START,
        END,
    )


if __name__ == "__main__":
    main()
