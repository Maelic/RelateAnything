"""Torch-free ONNX runtime for the full pipeline: detector -> boxes -> relations.

Everything here is onnxruntime + numpy. No torch, no ultralytics, no
transformers — so the laptop install is ~60 MB of wheels instead of ~2.5 GB.
That means the two things ultralytics would normally do for us (letterbox
preprocessing and NMS) are reimplemented below; they are simple and stable, but
they must match ultralytics' conventions exactly or boxes land in the wrong
place, so the details are spelled out rather than compressed.

    from deploy.runtime import ScenePipeline
    pipe = ScenePipeline("deploy/dist")
    result = pipe(frame_bgr)          # -> Result(boxes, labels, scores, triplets)
"""
from __future__ import annotations

import warnings
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.postprocess import (ThresholdConfig, Triplet, decode,
                                decode_decomposed)


# ---------------------------------------------------------------------------
# detector pre/post — numpy reimplementation of ultralytics' conventions
# ---------------------------------------------------------------------------

def letterbox(img: np.ndarray, new: int = 640, color: int = 114):
    """Resize keeping aspect ratio, pad to a square with grey borders.

    Returns (padded_image, scale, pad_x, pad_y). Ultralytics centres the pad,
    so the inverse mapping is (xy - pad) / scale — applied in `undo_letterbox`.
    """
    h, w = img.shape[:2]
    r = min(new / h, new / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    import cv2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((new, new, 3), color, dtype=img.dtype)
    dh, dw = (new - nh) // 2, (new - nw) // 2
    out[dh:dh + nh, dw:dw + nw] = resized
    return out, r, dw, dh


def undo_letterbox(xyxy: np.ndarray, r: float, dw: int, dh: int,
                   W: int, H: int) -> np.ndarray:
    b = xyxy.copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - dw) / r
    b[:, [1, 3]] = (b[:, [1, 3]] - dh) / r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W - 1)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H - 1)
    return b


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.5,
        max_det: int = 300) -> List[int]:
    """Plain greedy NMS (class-agnostic; callers offset boxes per class)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


@dataclass
class DetectorConfig:
    conf: float = 0.25
    iou: float = 0.5
    max_det: int = 40
    """Cap on boxes handed to the relation head. The head's pair budget is
    fixed at K=128, so more boxes do not cost more there — but they do dilute
    which pairs win a slot."""


class OnnxDetector:
    """YOLO-World v2, raw head output [1, 4+C, A], class scores pre-sigmoided."""

    def __init__(self, onnx_path: str, threads: int = 0,
                 providers: Optional[Sequence[str]] = None):
        meta_path = os.path.splitext(onnx_path)[0] + ".json"
        meta = json.load(open(meta_path))
        self.classes: List[str] = meta["classes"]
        self.imgsz: int = meta["imgsz"]
        self._build_engine(onnx_path, threads, providers)

    # -- engine hooks: the only backend-specific code. deploy/ov_runtime.py
    # overrides these two; everything else (letterbox, NMS, conventions) is
    # shared so the backends cannot drift apart.
    def _build_engine(self, path: str, threads: int,
                      providers: Optional[Sequence[str]]) -> None:
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            path, so, providers=list(providers or ["CPUExecutionProvider"]))
        self.iname = self.sess.get_inputs()[0].name

    def _run_engine(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.iname: x})[0]

    def make_input(self, frame_bgr: np.ndarray):
        """Letterbox + BGR->RGB, HWC->CHW, [0,1]. Returns (x, r, dw, dh)."""
        pad, r, dw, dh = letterbox(frame_bgr, self.imgsz)
        x = pad[:,:,::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), r, dw, dh

    def __call__(self, frame_bgr: np.ndarray, cfg: DetectorConfig):
        H, W = frame_bgr.shape[:2]
        x, r, dw, dh = self.make_input(frame_bgr)
        out = self._run_engine(x)

        pred = out[0].T                              # [A, 4+C]
        boxes_cxcywh, scores = pred[:,:4], pred[:, 4:]
        cls = scores.argmax(1)
        conf = scores[np.arange(len(cls)), cls]
        m = conf >= cfg.conf
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        boxes_cxcywh, conf, cls = boxes_cxcywh[m], conf[m], cls[m]

        cx, cy, bw, bh = boxes_cxcywh.T
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)

        # class-aware NMS via a per-class coordinate offset
        offset = cls.astype(np.float32)[:, None] * (self.imgsz + 1)
        keep = nms(xyxy + offset, conf, cfg.iou, cfg.max_det)
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]

        xyxy = undo_letterbox(xyxy, r, dw, dh, W, H)
        labels = [self.classes[int(c)] for c in cls]
        return xyxy.astype(np.float32), conf.astype(np.float32), labels


# ---------------------------------------------------------------------------
# relation head
# ---------------------------------------------------------------------------

class OnnxRelationHead:
    """Re-parameterized relation head. Predicate vocabulary is a runtime input
    when the graph was exported with --vocab-mode input."""

    def __init__(self, onnx_path: str, bank_path: str = "", threads: int = 0,
                 providers: Optional[Sequence[str]] = None):
        meta = json.load(open(os.path.splitext(onnx_path)[0] + ".json"))
        self.img_size: int = meta["img_size"]
        self.max_boxes: int = meta["max_boxes"]
        self.vocab_mode: str = meta.get("vocab_mode", "baked")
        # v2 graphs emit RAW LOGITS ("pred_logits"/"pair_logits"); v1 emitted
        # already-sigmoided scores. Key off the graph's own output names rather
        # than the sidecar, so a mismatched pair of files cannot be read wrong.
        out_names = self._build_engine(onnx_path, threads, providers)
        self.emits_logits = "pred_logits" in out_names
        if not self.emits_logits:
            warnings.warn(
                f"{os.path.basename(onnx_path)} is a v1 export emitting "
                f"sigmoided scores ({out_names[:2]}). They will be converted "
                "back to logits, but the raw head saturates above 0.9999 where "
                "fp32 has ~2 significant digits left, so the recovered ranking "
                "is APPROXIMATE. Re-export with deploy/export_onnx.py.",
                RuntimeWarning, stacklevel=2)
        # The score contract ships with the artifact; the sidecar carries the
        # calibration the numbers were produced under.
        from relsgg.scoring import ScoreContract
        cal = meta.get("calibration") or {}
        self.contract = ScoreContract(calib_a=float(cal.get("a", 1.0)),
                                      calib_b=float(cal.get("b", 0.0)))

        self.bank = None
        if self.vocab_mode == "input":
            if not bank_path:
                raise ValueError("vocab-mode=input needs a predicate bank")
            z = np.load(bank_path, allow_pickle=True)
            self.bank = {
                "names": [str(x) for x in z["names"]],
                "W": z["W"].astype(np.float32),
                "alpha": z["alpha"].astype(np.float32),
                # new-schema arrays (build_predicate_bank >= release): the
                # two-graph type vector and calibrated thresholds. Old banks
                # lack them -> features unavailable, never silently wrong.
                "is_spatial": (z["is_spatial"].astype(bool)
                               if "is_spatial" in z.files else None),
                "thr": (z["thr"].astype(np.float32)
                        if "thr" in z.files else None),
            }
            self.bank_index = {n: i for i, n in enumerate(self.bank["names"])}
            self.set_predicates([str(x) for x in z["default"]])
        else:
            self.predicates = meta["predicates"]

    # -- engine hooks (see OnnxDetector): deploy/ov_runtime.py overrides ----
    def _build_engine(self, path: str, threads: int,
                      providers: Optional[Sequence[str]]) -> List[str]:
        """Create the inference engine; returns the graph's output names."""
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            path, so, providers=list(providers or ["CPUExecutionProvider"]))
        self.input_names = [i.name for i in self.sess.get_inputs()]
        return [o.name for o in self.sess.get_outputs()]

    def _run_engine(self, feed: dict):
        return self.sess.run(None, feed)

    # -- the dynamic vocabulary ------------------------------------------
    def set_predicates(self, names: Sequence[str]) -> "OnnxRelationHead":
        """Swap the predicate vocabulary. No re-export, no text encoder — rows
        are sliced out of the precomputed bank."""
        if self.bank is None:
            raise RuntimeError("graph was exported with a baked vocabulary")
        missing = [n for n in names if n not in self.bank_index]
        if missing:
            raise KeyError(
                f"not in the predicate bank: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''} — rebuild it with "
                f"deploy/build_predicate_bank.py on the training box")
        idx = np.array([self.bank_index[n] for n in names], np.int64)
        self.predicates = list(names)
        self._W = np.ascontiguousarray(self.bank["W"][idx])
        self._alpha = np.ascontiguousarray(self.bank["alpha"][idx])
        self.is_spatial = (self.bank["is_spatial"][idx]
                           if self.bank.get("is_spatial") is not None else None)
        self.thr = (self.bank["thr"][idx]
                    if self.bank.get("thr") is not None else None)
        return self

    def available_predicates(self) -> List[str]:
        return list(self.bank["names"]) if self.bank else list(self.predicates)

    # -- inference ---------------------------------------------------------
    def make_feed(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray) -> dict:
        """Preprocess one frame + boxes into the graph's feed dict. Split out
        of __call__ so quantization calibration (deploy/export_openvino.py)
        feeds the model EXACTLY what inference will."""
        import cv2
        H, W = frame_bgr.shape[:2]
        N = len(boxes_xyxy)
        # NOTE: a plain square resize, NOT letterbox — the head's geometry
        # features are normalized cxcywh, so subject/object/union boxes and the
        # image are distorted identically and the mapping stays consistent.
        # This matches relsgg.api.RelateAnything._to_chw (and training).
        img = cv2.resize(frame_bgr, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        x = img[:,:,::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        b = boxes_xyxy.astype(np.float32).copy()
        b[:, [0, 2]] /= max(W, 1); b[:, [1, 3]] /= max(H, 1)
        cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2
        bw = b[:, 2] - b[:, 0];       bh = b[:, 3] - b[:, 1]
        boxes = np.stack([cx, cy, bw, bh], -1).astype(np.float32)

        # the graph was built for a fixed N — zero-pad and declare the true
        # count, exactly as the training collate does
        Nmax = self.max_boxes
        padded = np.zeros((1, Nmax, 4), np.float32)
        n = min(N, Nmax)
        padded[0,:n] = boxes[:n]

        feed = {"image": np.ascontiguousarray(x), "boxes": padded,
                "box_counts": np.array([n], np.int64)}
        if self.vocab_mode == "input":
            feed["W"] = self._W
            feed["alpha"] = self._alpha
        return feed

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray):
        """Returns raw (pred_logits [K,V], pair_logits [K], sub, obj, valid)."""
        feed = self.make_feed(frame_bgr, boxes_xyxy)
        pred, pair, sub, obj, valid = self._run_engine(feed)
        if not self.emits_logits:
            # v1 artifact: invert the sigmoids so the host sees logits either
            # way. Lossy at the top of the range by construction — warned once
            # at load. eps keeps logit(0) and logit(1) finite.
            eps = np.float32(1e-7)
            lg = lambda p: np.log(np.clip(p, eps, 1 - eps)
                                  / (1 - np.clip(p, eps, 1 - eps)))
            pred, pair = lg(pred.astype(np.float32)), lg(pair.astype(np.float32))
        return pred[0], pair[0], sub[0], obj[0], valid[0]


# ---------------------------------------------------------------------------
# the whole pipeline
# ---------------------------------------------------------------------------

@dataclass
class Result:
    boxes: np.ndarray
    labels: List[str]
    scores: np.ndarray
    triplets: List[Triplet]
    det_ms: float = 0.0
    rel_ms: float = 0.0
    dec_ms: float = 0.0
    # decompose mode: {"spatial": [...], "semantic": [...]} from the same
    # forward pass; `triplets` stays the single merged stream.
    graphs: "Optional[dict]" = None

    @property
    def total_ms(self) -> float:
        return self.det_ms + self.rel_ms + self.dec_ms


class TorchDetector:
    """ultralytics YOLO-World / YOLOE. Torch backend only."""

    def __init__(self, weights: str, arch: str = "yolo-world", device: str = "cpu",
                 classes: Optional[Sequence[str]] = None):
        if arch == "yoloe":
            from ultralytics import YOLOE as _Y
        else:
            from ultralytics import YOLOWorld as _Y
        self.model = _Y(weights)
        if classes is not None:                      # not needed for baked weights
            if arch == "yoloe":
                self.model.set_classes(list(classes), self.model.get_text_pe(list(classes)))
            else:
                self.model.set_classes(list(classes))
        self.device = device
        self.classes = [self.model.names[i] for i in range(len(self.model.names))]
        self.imgsz = 640

    def __call__(self, frame_bgr: np.ndarray, cfg: DetectorConfig):
        r = self.model.predict(frame_bgr, conf=cfg.conf, iou=cfg.iou,
                               imgsz=self.imgsz, max_det=cfg.max_det,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        return xyxy, conf, [self.model.names[int(c)] for c in cls]


class TorchRelationHead:
    """Relation head from a released ``model.pth``, exposing the same raw-score
    contract as ``OnnxRelationHead`` so both backends share ``decode``, the
    thresholds and the demo's live keys."""

    def __init__(self, checkpoint: str, bank_path: str = "", device: str = "cpu"):
        import torch
        from relsgg.api import RelateAnything
        self._torch = torch
        self.ra = RelateAnything.from_checkpoint(checkpoint, device=device)
        self.model = self.ra.model
        self.device = device
        self.img_size = self.ra.img_size
        self.max_boxes = 32          # only a cap; the torch path needs no padding
        self.predicates = list(self.ra.predicates)
        self.vocab_mode = "bank" if bank_path else "baked"
        self.bank = None
        if bank_path:
            z = np.load(bank_path, allow_pickle=True)
            self.bank = {"names": [str(x) for x in z["names"]],
                         "W": z["W"].astype(np.float32),
                         "alpha": z["alpha"].astype(np.float32)}
            self.bank_index = {n: i for i, n in enumerate(self.bank["names"])}

    def set_predicates(self, names: Sequence[str]) -> "TorchRelationHead":
        """Swap the vocabulary by writing bank rows straight into the head's
        buffers — same effect as feeding W/alpha to the ONNX graph, and it needs
        no text encoder either."""
        if self.bank is None:
            raise RuntimeError("no predicate bank loaded; vocabulary is baked")
        missing = [n for n in names if n not in self.bank_index]
        if missing:
            raise KeyError(f"not in the predicate bank: {missing[:5]}")
        idx = [self.bank_index[n] for n in names]
        t = self._torch
        with t.no_grad():
            self.model.vocab_head.W = t.from_numpy(self.bank["W"][idx]).to(self.device)
            self.model.vocab_head.alpha = t.from_numpy(self.bank["alpha"][idx]).to(self.device)
        self.model.vocab_head.is_reparameterized = True
        self.predicates = list(names)
        return self

    def available_predicates(self) -> List[str]:
        return list(self.bank["names"]) if self.bank else list(self.predicates)

    def __call__(self, frame_bgr: np.ndarray, boxes_xyxy: np.ndarray):
        import cv2
        t = self._torch
        H, W = frame_bgr.shape[:2]
        img = cv2.resize(frame_bgr, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        x = img[:,:,::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        b = boxes_xyxy.astype(np.float32).copy()
        b[:, [0, 2]] /= max(W, 1); b[:, [1, 3]] /= max(H, 1)
        cx = (b[:, 0] + b[:, 2]) / 2; cy = (b[:, 1] + b[:, 3]) / 2
        bw = b[:, 2] - b[:, 0];       bh = b[:, 3] - b[:, 1]
        boxes = np.stack([cx, cy, bw, bh], -1).astype(np.float32)[None]
        n = boxes.shape[1]

        with t.no_grad():
            out = self.model(t.from_numpy(np.ascontiguousarray(x)).to(self.device),
                             t.from_numpy(boxes).to(self.device),
                             box_counts=t.tensor([n], device=self.device),
                             targets=None)
        # Logits, as the ONNX head emits them; the host applies the score
        # contract.
        pred = out["logits"][0].float().cpu().numpy()
        pair = (out["pair_logits"][0].float().cpu().numpy()
                if out.get("pair_logits") is not None
                else np.zeros(pred.shape[0], np.float32))
        return (pred, pair, out["sub_idx"][0].cpu().numpy(),
                out["obj_idx"][0].cpu().numpy(),
                out["valid_mask"][0].cpu().numpy().astype(bool))


def _find_detector(dist_dir: str) -> str:
    """detector.onnx in the bundle, else../detector-local/detector.onnx."""
    cands = [os.path.join(dist_dir, "detector.onnx"),
             os.path.join(os.path.dirname(os.path.abspath(dist_dir)), "detector-local", "detector.onnx")]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "no detector.onnx found in %s or %s. Detector weights are not redistributed; rebuild them "
        "with deploy/reparam_detector.py + deploy/export_detector_onnx.py (see deploy/README.md)."
        % (cands[0], os.path.dirname(cands[1])))


class ScenePipeline:
    """Detector + relation head + decoding, all ONNX/numpy."""

    def __init__(self, dist_dir: str = "deploy/dist", detector: str = "",
                 relation: str = "", bank: str = "", threads: int = 0,
                 providers: Optional[Sequence[str]] = None,
                 det_cfg: Optional[DetectorConfig] = None,
                 thr_cfg: Optional[ThresholdConfig] = None,
                 backend: str = "onnx", device: str = "cpu",
                 rel_device: str = "", det_arch: str = "yolo-world"):
        """backend="onnx" (default, torch-free), "openvino" (torch-free,
        `device` selects CPU/GPU/NPU for both models unless `rel_device` is
        given, which overrides it for the relation head only — the Intel GPU
        plugin's Gather kernel doesn't support the relation head's axis-4
        gather (RuntimeError: "Unsupported gather axis: 4"), so GPU detector
        + CPU relation head is the working combo on iGPU. files are OpenVINO IR.xml produced by
        deploy/export_openvino.py — int8 preferred, fp16 fallback), or "torch"
        (needs torch + ultralytics; `relation` is a released model.pth
        bundle and `detector` an ultralytics.pt)."""
        bank = bank or os.path.join(dist_dir, "predicate_bank.npz")
        if not os.path.exists(bank):
            bank = ""
        if backend == "torch":
            if not relation:
                raise ValueError("backend=torch needs --deploy (a.pt bundle)")
            self.det = TorchDetector(
                detector or "checkpoints/detectors/yolov8s-worldv2_megasg497.pt",
                arch=det_arch, device=device)
            self.rel = TorchRelationHead(relation, bank_path=bank, device=device)
        elif backend == "openvino":
            from deploy.ov_runtime import OVDetector, OVRelationHead, pick_ir
            # detector prefers fp16: int8 detection is measurably damaged
            # (box IoU 0.80, label agreement 0.62 vs fp32) and only built on
            # explicit request — never auto-picked.
            detector = detector or pick_ir(dist_dir, "detector",
                                           order=("fp16", "int8"))
            # fp16 first for the relation head too: its decoded output is
            # bit-identical to fp32 (measured top-20 Jaccard 1.000 on fixed
            # boxes), while backbone-int8 changes ~24% of top-20 edges. int8
            # is the explicit speed knob, not the silent default.
            relation = relation or pick_ir(dist_dir, "relateanything",
                                           order=("fp16", "int8"))
            self.det = OVDetector(detector, device=device, threads=threads)
            self.rel = OVRelationHead(relation, bank_path=bank,
                                      device=rel_device or device,
                                      threads=threads)
        else:
            # The detector is rebuilt locally (AGPL upstream, never shipped):
            # inside the bundle directory, or in the shared detector-local/
            # directory next to the bundles (deploy/README.md).
            detector = detector or _find_detector(dist_dir)
            relation = relation or os.path.join(dist_dir, "relateanything.onnx")
            self.det = OnnxDetector(detector, threads=threads, providers=providers)
            self.rel = OnnxRelationHead(relation, bank_path=bank, threads=threads,
                                        providers=providers)
        self.backend = backend
        self.det_cfg = det_cfg or DetectorConfig()
        self.thr_cfg = thr_cfg or ThresholdConfig()
        # The artifact's own calibration drives the threshold unless the caller
        # supplied one. Without it `threshold` is not a control: the raw head
        # puts ~97% of scores above 0.9, so the 0.40 default keeps everything.
        # Keyed on whether the CALLER set a calibration, not on whether they
        # passed a ThresholdConfig at all — demo_webcam.py always passes one
        # (to carry --threshold), and it would otherwise never receive the
        # artifact's calibration.
        head_contract = getattr(self.rel, "contract", None)
        if (head_contract is not None
                and (self.thr_cfg.calib_a, self.thr_cfg.calib_b) == (1.0, 0.0)):
            self.thr_cfg.calib_a = head_contract.calib_a
            self.thr_cfg.calib_b = head_contract.calib_b
        if (self.thr_cfg.calib_a, self.thr_cfg.calib_b) == (1.0, 0.0):
            warnings.warn(
                "relation head is UNCALIBRATED: ~97% of scores sit in "
                f"[0.9, 1.0), so ThresholdConfig.threshold="
                f"{self.thr_cfg.threshold} keeps nearly every prediction. Ship "
                "a calibration.json with the checkpoint (see "
                "benchmark/eval_deploy_metrics.py --fit_platt).",
                RuntimeWarning, stacklevel=2)
        # Calibrated per-predicate thresholds from the bank become the
        # per_predicate overrides — but a threshold the CALLER already set
        # wins (explicit beats calibrated). NaN = uncalibrated = global floor.
        head_thr = getattr(self.rel, "thr", None)
        if head_thr is not None:
            # calibrate_thresholds.py measures these on sigmoid(pred_logit)
            # ALONE. The score they are now compared against is
            # sigmoid(a*(pred + w*rel) + b), so with a calibration installed
            # they are in the wrong units and would silently gate on nothing.
            # Map each one through the SAME affine, which is exact:
            #   sigmoid(a*z + b) >= t'  <=>  sigmoid(z) >= t   for
            #   t' = sigmoid(a*logit(t) + b).
            # (The w-term mismatch is pre-existing and unchanged — see the
            # REGIME note in calibrate_thresholds.py.)
            a, b = self.thr_cfg.calib_a, self.thr_cfg.calib_b
            for n, v in zip(self.rel.predicates, head_thr):
                if not np.isfinite(v) or n in self.thr_cfg.per_predicate:
                    continue
                t = float(np.clip(v, 1e-6, 1 - 1e-6))
                z = np.log(t / (1 - t))
                self.thr_cfg.per_predicate[n] = float(1.0 / (1.0 + np.exp(-(a * z + b))))

    def __call__(self, frame_bgr: np.ndarray,
                 decompose: bool = False) -> Result:
        t0 = time.perf_counter()
        boxes, confs, labels = self.det(frame_bgr, self.det_cfg)
        t1 = time.perf_counter()
        if len(boxes) < 2:
            return Result(boxes, labels, confs, [], (t1 - t0) * 1e3, 0.0, 0.0)

        n = min(len(boxes), self.rel.max_boxes)
        boxes, confs, labels = boxes[:n], confs[:n], labels[:n]
        pred, pair, sub, obj, valid = self.rel(frame_bgr, boxes)
        t2 = time.perf_counter()

        triplets = decode(pred, pair, sub, obj, valid, self.rel.predicates,
                          self.thr_cfg, boxes_xyxy=boxes, box_scores=confs,
                          box_labels=labels)
        graphs = None
        if decompose:
            is_sp = getattr(self.rel, "is_spatial", None)
            if is_sp is not None:
                graphs = decode_decomposed(
                    pred, pair, sub, obj, valid, self.rel.predicates, is_sp,
                    self.thr_cfg, boxes_xyxy=boxes, box_scores=confs,
                    box_labels=labels)
        t3 = time.perf_counter()
        return Result(boxes, labels, confs, triplets,
                      (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
                      graphs=graphs)
