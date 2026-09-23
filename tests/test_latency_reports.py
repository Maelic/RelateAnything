"""Publishing guards and sample-derived statistics for latency reports."""

import json
from copy import deepcopy

import pytest

from release import update_readme_e2e as report


@pytest.fixture
def benchmark_data():
    """Small synthetic records: report tests never depend on a GPU run artifact."""
    base = dict.fromkeys(
        ("gpu", "cpu", "torch", "cuda", "tensorrt", "onnxruntime", "ultralytics"),
        "test",
    )
    base.update(
        complete=True,
        batch=1,
        image_size=448,
        max_relation_boxes=20,
        padded_boxes=32,
        pair_budget=128,
        input_images=[{"path": "scene.jpg"}],
        images=1,
        threads=1,
        opencv_threads=1,
        rounds=1,
        warmup_per_round=1,
        timed_iterations_per_round=1,
        tf32=False,
        torch_compile=False,
        arm_shuffle_seeds=[42],
        ort_cuda_options={},
        cooldown_temperature_c=None,
        timing="test",
        stage_timing="test",
        rows=[],
        measurement_blocks=[],
        fp32_cross_backend_checks=[],
    )
    gpu = {
        "clocks_event_reasons_counters.sw_thermal_slowdown": "0 us",
        "clocks_event_reasons_counters.hw_thermal_slowdown": "0 us",
        "enforced.power.limit": "90 W",
        "temperature.gpu": "60",
    }
    for arm in report.ARMS:
        for vocab in report.VOCABS:
            base["rows"].append(
                {
                    "backend": arm,
                    "predicates": vocab,
                    "n": 1,
                    "samples_ms": [10.0],
                    "median_ms": 10.0,
                    "pipeline_samples": [
                        {
                            "detector_ms": 3.0,
                            "relation_ms": 5.0,
                            "decode_ms": 1.0,
                            "total_ms": 10.0,
                            "round": 1,
                            "image_index": 0,
                            "detected_boxes": 2,
                            "used_boxes": 2,
                            "relation_skipped": False,
                            "triplets": 1,
                        }
                    ],
                }
            )
            base["measurement_blocks"].append(
                {
                    "backend": arm,
                    "predicates": vocab,
                    "round": 1,
                    "gpu_before_warmup": deepcopy(gpu),
                    "gpu_after_timing": deepcopy(gpu),
                    "cooling_seconds": 0,
                }
            )
            if arm in (report.ARMS[0], report.ARMS[2]):
                base["fp32_cross_backend_checks"].append(
                    {
                        "backend": arm,
                        "predicates": vocab,
                        "image": "scene.jpg",
                        "same_valid_pairs": True,
                        "max_logit_delta": 0.0,
                    }
                )
    records = []
    for family in report.DETECTORS:
        for checkpoint in report.MODELS:
            record = deepcopy(base)
            record.update(
                checkpoint=checkpoint,
                detector={
                    "family": family,
                    "classes": [f"object-{i}" for i in range(80)],
                    "predict": {"conf": 0.25},
                    "precision": "fp32",
                    "runtime": "pytorch",
                    "size": "m",
                    "masks_used_by_relation": False,
                },
            )
            records.append(record)
    return {"records": records}


def test_statistics_use_raw_calls_and_interpolated_p95():
    assert report.stats([100, 1, 50, 10]) == pytest.approx((30, 92.5))
    assert report.cell([]) == "—"


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "missing_pairing",
        "totals",
        "skipped",
        "coverage",
        "detector",
        "detector_settings",
        "parity",
    ],
)
def test_end_to_end_publishing_rejects_inconsistent_evidence(
    tmp_path, benchmark_data, fault
):
    data = benchmark_data
    record = data["records"][0]
    row = record["rows"][0]
    if fault == "incomplete":
        record["complete"] = False
    elif fault == "missing_pairing":
        data["records"].pop()
    elif fault == "totals":
        row["samples_ms"][0] += 1
    elif fault == "skipped":
        row["pipeline_samples"][0]["relation_skipped"] = True
    elif fault == "coverage":
        row["pipeline_samples"][0]["image_index"] = 100
    elif fault == "detector":
        record["detector"]["precision"] = "fp16"
    elif fault == "detector_settings":
        for other in data["records"]:
            if other["detector"]["family"] == "yoloe":
                other["detector"]["predict"]["conf"] = 0.4
    else:
        record["fp32_cross_backend_checks"].pop()
    path = tmp_path / "records.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        report.load_records(path)


def test_report_ignores_stored_medians(tmp_path, benchmark_data):
    path = tmp_path / "records.json"
    path.write_text(json.dumps(benchmark_data))
    records = report.load_records(path)
    renderers = (report.render_summary, report.render_overview, report.render_details)
    expected = [render(records) for render in renderers]
    for record in records:
        for row in record["rows"]:
            row["median_ms"] = -1
            row["stage_latency"] = {}
    assert [render(records) for render in renderers] == expected
