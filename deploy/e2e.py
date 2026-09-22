"""Detector adapters and timing boundaries for the end-to-end GPU benchmark.

Ultralytics is optional and imported only when preparing/loading a detector.
All detectors run PyTorch FP32; relation backends are varied independently.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

FAMILIES = {
    "yolo26": "yolo26{size}.pt",
    "yolo-world": "yolov8{size}-worldv2.pt",
    "yoloe": "yoloe-26{size}-seg.pt",
}


def coco_classes():
    from ultralytics.utils import ROOT, YAML

    names = YAML.load(ROOT / "cfg/datasets/coco.yaml")["names"]
    return [names[i] for i in range(len(names))]


def load_model(family, weights):
    from ultralytics import YOLO, YOLOE, YOLOWorld, __version__
    from ultralytics.utils.checks import check_version

    check_version(__version__, ">=8.4.159", name="ultralytics", hard=True)

    return {"yolo26": YOLO, "yolo-world": YOLOWorld, "yoloe": YOLOE}[family](
        str(weights)
    )


def names_list(model):
    names = model.names
    return (
        [names[i] for i in range(len(names))]
        if isinstance(names, dict)
        else list(names)
    )


def prepare(family, size, destination):
    """Download official weights and prepare COCO prompts on CPU, outside timing."""
    import ultralytics

    from deploy.bench_gpu_backends import sha256

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / FAMILIES[family].format(size=size)
    model = load_model(family, source).to("cpu")
    classes = coco_classes()
    if family != "yolo26" and names_list(model) != classes:
        model.set_classes(classes)
    if names_list(model) != classes:
        raise ValueError("Detector class vocabulary is not COCO-80")
    target = destination / f"{source.stem}-coco80.pt"
    model.save(str(target))
    metadata = {
        "family": family,
        "size": size,
        "source": source.name,
        "source_sha256": sha256(source),
        "weights_sha256": sha256(target),
        "ultralytics": ultralytics.__version__,
        "classes": classes,
        "task": model.task,
        "runtime": "pytorch",
        "precision": "fp32",
    }
    target.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    return target


class CocoDetector:
    def __init__(
        self, family, weights, device="cuda", imgsz=640, conf=0.25, iou=0.6, max_det=100
    ):
        from deploy.bench_gpu_backends import sha256

        weights = Path(weights)
        if weights.suffix != ".pt" or not weights.is_file():
            raise ValueError(
                "Expected prepared local .pt weights; run python -m deploy.e2e first"
            )
        self.metadata = json.loads(weights.with_suffix(".json").read_text())
        if self.metadata["family"] != family or self.metadata[
            "weights_sha256"
        ] != sha256(weights):
            raise ValueError("Detector metadata does not match the weights/family")
        self.model = load_model(family, weights).to(device)
        self.classes = names_list(self.model)
        if self.classes != coco_classes() or self.classes != self.metadata["classes"]:
            raise ValueError("Prepared detector must use the same COCO-80 vocabulary")
        self.options = dict(
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            max_det=max_det,
            quantize=32,
            rect=False,
            verbose=False,
            agnostic_nms=False,
            nms=None,
            save=False,
        )
        self.metadata = {
            **self.metadata,
            "weights": str(weights),
            "predict": self.options,
            "masks_computed": self.model.task == "segment",
            "masks_used_by_relation": False,
        }

    def __call__(self, frame):
        result = self.model.predict(frame, **self.options)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        boxes = result.boxes.xyxy.float().cpu().numpy()
        scores = result.boxes.conf.float().cpu().numpy()
        classes = result.boxes.cls.long().cpu().numpy()
        # Some detector heads return anchor order; cap by confidence for all.
        order = np.argsort(-scores, kind="stable")
        return (
            boxes[order],
            scores[order],
            [self.classes[int(classes[i])] for i in order],
        )


def pipeline_call(
    frame,
    detector,
    relation,
    decoder,
    max_boxes,
    clock=time.perf_counter,
    synchronize=lambda: None,
):
    """Time the serial path. Model adapters must return CPU-ready outputs.

    `relation` includes relation preprocessing/transfers and model execution;
    `decoder` includes score calibration and triplet selection. GPU callers
    supply synchronization at stage boundaries and additionally synchronize
    around the entire call for total latency.
    """
    start = clock()
    boxes, scores, labels = detector(frame)
    synchronize()
    detector_end = clock()
    detected = len(boxes)
    boxes, scores, labels = boxes[:max_boxes], scores[:max_boxes], labels[:max_boxes]
    sample = {
        "detector_ms": (detector_end - start) * 1000,
        "relation_ms": 0.0,
        "decode_ms": 0.0,
        "detected_boxes": detected,
        "used_boxes": len(boxes),
        "relation_skipped": len(boxes) < 2,
        "triplets": 0,
    }
    if len(boxes) >= 2:
        raw = relation(frame, boxes)
        synchronize()
        relation_end = clock()
        triplets = decoder(raw, boxes, scores, labels)
        decode_end = clock()
        sample.update(
            relation_ms=(relation_end - detector_end) * 1000,
            decode_ms=(decode_end - relation_end) * 1000,
            triplets=len(triplets),
        )
    return sample


def summarize_pipeline(samples):
    """Summarize measured calls, preserving skipped frames and the raw evidence."""

    def stats(values):
        if not values:
            return None
        return {
            "n": len(values),
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "mean_ms": float(np.mean(values)),
        }

    return {
        "stage_latency": {
            stage: stats([s[stage + "_ms"] for s in samples])
            for stage in ("detector", "relation", "decode")
        },
        "active_pipeline_latency": stats(
            [s["total_ms"] for s in samples if not s["relation_skipped"]]
        ),
        "relation_skipped_calls": sum(s["relation_skipped"] for s in samples),
        "box_counts": (
            {
                kind: {
                    "min": min(s[kind] for s in samples),
                    "median": float(np.median([s[kind] for s in samples])),
                    "max": max(s[kind] for s in samples),
                }
                for kind in ("detected_boxes", "used_boxes")
            }
            if samples
            else {}
        ),
        "pipeline_samples": samples,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare COCO-80 detector weights on CPU; does not run a benchmark."
    )
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--size", choices=["s", "m", "l"], default="m")
    parser.add_argument("--out", type=Path, default=Path("checkpoints/detectors/e2e"))
    args = parser.parse_args()
    for family in args.families:
        print(prepare(family, args.size, args.out), flush=True)


if __name__ == "__main__":
    main()
