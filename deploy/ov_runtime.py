"""OpenVINO backend for the deploy pipeline: same models, Intel-native engine.

Subclasses the ONNX runtime classes and overrides ONLY the engine hooks
(`_build_engine` / `_run_engine`), so every convention that matters —
letterbox, NMS, the plain-square-resize relation preprocessing, the score
contract, the predicate bank — is inherited from deploy/runtime.py and cannot
drift between backends.

Why this backend exists: onnxruntime's CPU EP is the portable floor. OpenVINO
adds the Intel-specific ceilings — the iGPU ("GPU") and NPU on Core Ultra
laptops, plus a faster CPU path — and is what NNCF int8 quantization targets.

    from deploy.runtime import ScenePipeline
    pipe = ScenePipeline("deploy/dist/relsgg-vits16plus", backend="openvino",
                         device="GPU")          # or CPU / NPU / AUTO

Device notes:
  CPU   works with the dynamic-vocabulary graphs as-is.
  GPU   (the Intel iGPU) works with dynamic shapes; first compile per shape is
        slow (seconds) and cached — expect a one-time warmup.
  NPU   requires fully static shapes: export with
        deploy/export_openvino.py --static-vocab N to freeze V.
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.runtime import OnnxDetector, OnnxRelationHead

_CORE = None


def _core():
    global _CORE
    if _CORE is None:
        import openvino as ov
        _CORE = ov.Core()
    return _CORE


def pick_ir(dist_dir: str, stem: str,
            order: Sequence[str] = ("int8", "fp16")) -> str:
    """Best available IR for a model stem, by preference order."""
    for variant in order:
        p = os.path.join(dist_dir, f"{stem}_{variant}.xml")
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"no {stem}_{{{','.join(order)}}}.xml in {dist_dir} — build them "
        "with deploy/export_openvino.py")


def _compile(path: str, device: str, threads: int):
    core = _core()
    device = {"cuda": "GPU", "igpu": "GPU"}.get(device.lower(), device).upper()
    config = {"PERFORMANCE_HINT": "LATENCY"}
    if threads and device == "CPU":
        config["INFERENCE_NUM_THREADS"] = threads
    return core.compile_model(core.read_model(path), device, config)


class OVDetector(OnnxDetector):
    """YOLO-World v2 on OpenVINO. Same sidecar/meta conventions as the ONNX
    class — the IR's.xml sits next to a.json copied at conversion time."""

    def __init__(self, xml_path: str, device: str = "CPU", threads: int = 0,
                 providers: Optional[Sequence[str]] = None):
        self._device = device
        super().__init__(xml_path, threads=threads, providers=providers)

    def _build_engine(self, path, threads, providers):
        self.compiled = _compile(path, self._device, threads)
        self.req = self.compiled.create_infer_request()
        self.iname = self.compiled.inputs[0].get_any_name()

    def _run_engine(self, x):
        return self.req.infer({self.iname: x})[self.compiled.outputs[0]]


class OVRelationHead(OnnxRelationHead):
    """Relation head on OpenVINO; dynamic predicate vocabulary works exactly
    as on ONNX (W/alpha are graph inputs, rows sliced from the bank)."""

    def __init__(self, xml_path: str, bank_path: str = "", device: str = "CPU",
                 threads: int = 0, providers: Optional[Sequence[str]] = None):
        self._device = device
        super().__init__(xml_path, bank_path=bank_path, threads=threads,
                         providers=providers)

    def _build_engine(self, path, threads, providers) -> List[str]:
        self.compiled = _compile(path, self._device, threads)
        self.req = self.compiled.create_infer_request()
        self.input_names = [i.get_any_name() for i in self.compiled.inputs]
        return [o.get_any_name() for o in self.compiled.outputs]

    def _run_engine(self, feed):
        if "W" in feed and "W" not in self.input_names:
            # --static-vocab IR: the vocabulary is frozen in the graph. The
            # bank rows selected at runtime MUST match what was baked; the
            # sidecar carries the baked list and set_predicates still works
            # for slicing thresholds/is_spatial, but W/alpha are not inputs.
            feed = {k: v for k, v in feed.items() if k in self.input_names}
        results = self.req.infer(feed)
        return [np.asarray(results[o]) for o in self.compiled.outputs]
