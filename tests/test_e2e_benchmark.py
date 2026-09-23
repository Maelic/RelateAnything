"""CPU checks for full-pipeline timing; no detector weights or GPU dependencies."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from deploy import e2e
from deploy.bench_gpu_backends import sha256


def test_plan_needs_no_models_or_gpu_packages():
    code = """
import runpy, sys
sys.modules.update(torch=None, tensorrt=None, onnxruntime=None, ultralytics=None)
sys.argv = ['bench', '--checkpoint', 'missing/model.pth', '--bundle', 'missing',
            '--out', 'missing/results.json', '--detector', 'yolo26',
            '--detector-weights', 'missing.pt', '--dry-run']
runpy.run_module('deploy.bench_gpu_backends', run_name='__main__')
"""
    plan = json.loads(subprocess.check_output([sys.executable, "-c", code], text=True))
    assert plan["detector"] == "yolo26"
    assert plan["detector_runtime"] == "PyTorch FP32"
    assert len(plan["relation_backends"]) == 4
    assert plan["timed_calls_per_configuration"] == 180


@pytest.mark.parametrize("count", [0, 1, 2, 4])
def test_pipeline_times_detection_on_every_call_and_skips_only_small_inputs(count):
    boxes = np.arange(count * 4).reshape(count, 4)
    scores = np.linspace(1, 0, count)
    labels = [str(i) for i in range(count)]
    events = []

    def detector(frame):
        assert frame == "image"
        events.append("detect")
        return boxes, scores, labels

    def relation(frame, selected):
        events.append("relate")
        np.testing.assert_array_equal(selected, boxes[:2])
        return "raw logits"

    def decode(raw, selected, selected_scores, selected_labels):
        events.append("decode")
        assert raw == "raw logits"
        np.testing.assert_array_equal(selected, boxes[:2])
        np.testing.assert_array_equal(selected_scores, scores[:2])
        assert selected_labels == labels[:2]
        return ["triplet"]

    for _ in range(2):
        ticks = iter([0.0, 0.01, 0.03, 0.034])
        sample = e2e.pipeline_call(
            "image",
            detector,
            relation,
            decode,
            2,
            clock=lambda: next(ticks),
            synchronize=lambda: events.append("sync"),
        )
        assert sample["detector_ms"] == pytest.approx(10)
        assert sample["detected_boxes"] == count
        assert sample["used_boxes"] == min(count, 2)
        assert sample["relation_skipped"] == (count < 2)
        assert sample["relation_ms"] == pytest.approx(20 if count >= 2 else 0)
        assert sample["decode_ms"] == pytest.approx(4 if count >= 2 else 0)
        assert sample["triplets"] == (1 if count >= 2 else 0)
    expected = (
        ["detect", "sync", "relate", "sync", "decode"]
        if count >= 2
        else ["detect", "sync"]
    )
    assert events == expected * 2


def test_stage_summary_keeps_skipped_frames_and_uses_measured_active_totals():
    samples = [
        dict(
            detector_ms=10,
            relation_ms=0,
            decode_ms=0,
            total_ms=11,
            detected_boxes=1,
            used_boxes=1,
            relation_skipped=True,
        ),
        dict(
            detector_ms=20,
            relation_ms=30,
            decode_ms=2,
            total_ms=55,
            detected_boxes=5,
            used_boxes=3,
            relation_skipped=False,
        ),
        dict(
            detector_ms=40,
            relation_ms=10,
            decode_ms=1,
            total_ms=60,
            detected_boxes=3,
            used_boxes=3,
            relation_skipped=False,
        ),
    ]
    summary = e2e.summarize_pipeline(samples)
    assert summary["relation_skipped_calls"] == 1
    assert summary["active_pipeline_latency"]["median_ms"] == 57.5
    assert summary["active_pipeline_latency"]["p95_ms"] == 59.75
    assert summary["stage_latency"]["relation"]["median_ms"] == 10
    assert summary["box_counts"]["detected_boxes"] == dict(min=1, median=3, max=5)
    assert summary["pipeline_samples"] == samples
    assert e2e.summarize_pipeline(samples[:1])["active_pipeline_latency"] is None
    assert e2e.summarize_pipeline([])["active_pipeline_latency"] is None


@pytest.fixture
def fake_detector(tmp_path, monkeypatch):
    weights = tmp_path / "detector.pt"
    weights.write_bytes(b"local test weights")
    classes = ["person", "cat", "dog"]
    monkeypatch.setattr(e2e, "coco_classes", lambda: classes)

    class Boxes:
        xyxy = torch.arange(12).reshape(3, 4)
        conf = torch.tensor([0.2, 0.9, 0.9])
        cls = torch.tensor([0, 2, 1])

        def __len__(self):
            return 3

    class Model:
        names = dict(enumerate(classes))
        task = "detect"

        def to(self, device):
            assert device == "cpu"
            return self

        def predict(self, frame, **kwargs):
            self.options = kwargs
            return [SimpleNamespace(boxes=Boxes())]

    model = Model()
    monkeypatch.setattr(e2e, "load_model", lambda family, path: model)

    def create(family):
        weights.with_suffix(".json").write_text(
            json.dumps(
                dict(family=family, weights_sha256=sha256(weights), classes=classes)
            )
        )
        model.task = "segment" if family == "yoloe" else "detect"
        return weights, model

    return create


@pytest.mark.parametrize("family", list(e2e.FAMILIES))
def test_detectors_preserve_native_postprocess_then_sort_by_confidence(
    fake_detector, family
):
    weights, model = fake_detector(family)
    detector = e2e.CocoDetector(family, weights, device="cpu")
    boxes, scores, labels = detector(np.zeros((4, 4, 3), np.uint8))
    np.testing.assert_array_equal(boxes[:, 0], [4, 8, 0])
    np.testing.assert_allclose(scores, [0.9, 0.9, 0.2])
    assert labels == ["dog", "cat", "person"]
    assert model.options["quantize"] == 32
    assert model.options["rect"] is False
    assert model.options["agnostic_nms"] is False
    assert model.options["nms"] is None
    assert detector.metadata["masks_computed"] == (family == "yoloe")
    assert detector.metadata["masks_used_by_relation"] is False


def test_changed_weights_and_wrong_vocabulary_are_rejected(fake_detector):
    weights, model = fake_detector("yoloe")
    weights.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="metadata"):
        e2e.CocoDetector("yoloe", weights, device="cpu")
    weights, model = fake_detector("yoloe")
    model.names = ["arbitrary", "vocabulary"]
    with pytest.raises(ValueError, match="COCO-80"):
        e2e.CocoDetector("yoloe", weights, device="cpu")


def test_end_to_end_records_cannot_be_published_as_relation_only(tmp_path):
    from release import update_readme_latency as report

    data = {
        "records": [
            {"checkpoint": model, "complete": True, "detector": {"family": "yolo26"}}
            for model in report.MODELS
        ]
    }
    source = tmp_path / "records.json"
    source.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="relation-only"):
        report.load_records(source)
