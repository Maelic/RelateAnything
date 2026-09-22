"""Publishing guards and sample-derived statistics for latency reports."""

import gzip
import json

import pytest

from release import update_readme_e2e as report


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
    tmp_path, monkeypatch, fault
):
    with gzip.open(report.SOURCE, "rt") as stream:
        data = json.load(stream)
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
    monkeypatch.setattr(report, "SOURCE", path)
    with pytest.raises(ValueError):
        report.load_records()


def test_report_ignores_stored_medians():
    records = report.load_records()
    expected = report.render_summary(records)
    for record in records:
        for row in record["rows"]:
            row["median_ms"] = -1
            row["stage_latency"] = {}
    assert report.render_summary(records) == expected
