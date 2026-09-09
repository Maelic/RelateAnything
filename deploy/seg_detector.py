"""ONNX segmentation detectors: boxes, instance masks, and names or no names.

`deploy/runtime.py` already carries a box detector, but it reads the RAW YOLO
head (`[1, 4+C, A]`, argmax taken host-side) and returns no masks. The web
demo's exports are a different, smaller contract — max/argmax and NMS-free
filtering are folded into the graph, and the mask branch is a separate pair of
outputs:

    images     [1, 3, imgsz, imgsz]  letterboxed RGB, 0..1
    boxes      [1, 4, A]             cxcywh in LETTERBOXED pixels
    conf       [1, A]                already sigmoided
    cls        [1, A]                argmax class id
    mask_coef  [1, md, A]            md=32 prototype coefficients per anchor
    proto      [1, md, mh, mw]       mh=mw=imgsz/mask_stride (160 at 640)

This module decodes that layout, so the same three detectors the browser demo
offers can be driven from Python for figure and video rendering. It is a port
of `demo/js/pipeline.js` (`decodeDetections`, `decodeMasks`, `colourLabels`),
kept line-comparable on purpose: a figure that disagrees with the live demo is
worse than no figure. Measured against the demo's own stored output on
`horse.jpg`: same detection count, boxes within 1 px, confidences within 0.006
(the browser runs fp16 weights, this runs fp32). Colour NAMES can differ on an
instance or two, because `colour_labels` here votes over full-resolution
pixels where the browser votes at 160x160 prototype resolution; the boxes and
masks those names refer to are the same.

The detectors themselves are AGPL-3.0 ultralytics derivatives and are NOT
redistributed with this repository — pass `--models` a directory holding the
web demo's exports (see `deploy/README.md`).

Why three of them, and why this file exists at all: the relation model's input
contract is *regions*, and these three cover the ways a region can arrive.

  yoloworld-s-megasg497   boxes, 497 class names   (the laptop-bundle detector)
  yoloe-11s-pf            boxes + masks, 4,585 class names
  fastsam-s               boxes + masks, NO names at all — class-agnostic

The third is the interesting one. FastSAM never emits a category, so there is
nothing to name an instance with; `colour_labels` names each one by the colour
a person would call it, purely so a caption can refer to it. The relation model
never reads any of these names.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .runtime import letterbox, nms, undo_letterbox


@dataclass
class SegDetection:
    """One detector pass over one image."""

    boxes: np.ndarray          # [N, 4] xyxy in ORIGINAL image pixels
    scores: np.ndarray         # [N]
    labels: List[str]          # class names, or colour names, or [] if unnamed
    masks: Optional[np.ndarray]  # [N, H, W] bool, or None when masks are off
    index_map: Optional[np.ndarray]  # [H, W] uint8, 0 = background, i+1 = det i

    def __len__(self) -> int:
        return len(self.boxes)


class OnnxSegDetector:
    """A `boxes_conf_cls` export, with the optional mask branch decoded."""

    def __init__(self, onnx_path: str, classes: Optional[Sequence[str]] = None,
                 imgsz: int = 640, threads: int = 0,
                 providers: Optional[Sequence[str]] = None):
        self.imgsz = imgsz
        if classes is None:
            side = os.path.splitext(onnx_path)[0] + ".classes.json"
            classes = json.load(open(side)) if os.path.exists(side) else []
        self.classes: List[str] = list(classes)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(
            onnx_path, so, providers=list(providers or ["CPUExecutionProvider"]))
        self.iname = self.sess.get_inputs()[0].name
        self.onames = [o.name for o in self.sess.get_outputs()]

    # -- input ---------------------------------------------------------------

    def make_input(self, frame_bgr: np.ndarray):
        """Letterbox + BGR->RGB, HWC->CHW, [0,1]. Returns (x, r, dw, dh)."""
        pad, r, dw, dh = letterbox(frame_bgr, self.imgsz)
        x = pad[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(x), r, dw, dh

    # -- inference -----------------------------------------------------------

    def __call__(self, frame_bgr: np.ndarray, conf: float = 0.25,
                 iou: float = 0.5, max_det: int = 32,
                 masks: bool = True) -> SegDetection:
        H, W = frame_bgr.shape[:2]
        x, r, dw, dh = self.make_input(frame_bgr)

        # Fetching only the outputs we need lets onnxruntime prune the graph:
        # with masks off, the 160x160 ConvTranspose prototype head never runs,
        # which is most of a segmentation checkpoint's extra cost.
        want = ["boxes", "conf", "cls"] + (["mask_coef", "proto"] if masks else [])
        want = [n for n in want if n in self.onames]
        out = dict(zip(want, self.sess.run(want, {self.iname: x})))

        scores = out["conf"][0]
        keep_conf = scores >= conf
        empty = SegDetection(np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                             [], None, None)
        if not keep_conf.any():
            return empty

        anchors = np.nonzero(keep_conf)[0]
        cxcywh = out["boxes"][0][:, anchors].T          # [4, A] -> [n, 4]
        cls = out["cls"][0][anchors].astype(np.int64)
        scores = scores[anchors]

        cx, cy, bw, bh = cxcywh.T
        lb = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)

        # class-aware NMS via a per-class coordinate offset (ultralytics'
        # convention; a class-agnostic checkpoint has one class, so the offset
        # is constant and this degrades to plain NMS).
        offset = cls.astype(np.float32)[:, None] * (self.imgsz + 1)
        keep = nms(lb + offset, scores, iou, max_det)
        lb, scores, cls, anchors = lb[keep], scores[keep], cls[keep], anchors[keep]

        boxes = undo_letterbox(lb, r, dw, dh, W, H).astype(np.float32)
        labels = [self.classes[int(c)] if int(c) < len(self.classes) else str(int(c))
                  for c in cls]

        idx_proto = None
        if masks and "proto" in out and len(lb):
            idx_proto = decode_masks(out["mask_coef"][0], out["proto"][0], anchors,
                                     lb, self.imgsz)
        index_map, inst = None, None
        if idx_proto is not None:
            index_map = _proto_to_image(idx_proto, r, dw, dh, W, H, self.imgsz)
            inst = np.stack([index_map == i + 1 for i in range(len(lb))]) \
                if len(lb) else None

        return SegDetection(boxes, scores.astype(np.float32), labels, inst,
                            index_map)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------

def decode_masks(coef: np.ndarray, proto: np.ndarray, anchors: np.ndarray,
                 lb_boxes: np.ndarray, imgsz: int) -> np.ndarray:
    """Prototype coefficients -> one index map [mh, mw]: 0 = bg, n+1 = det n.

    `sum_k coef[k] * proto[k]` is the mask logit, so `> 0` is `sigmoid > 0.5`.
    Each mask is cropped to its own detection box scaled into prototype space,
    exactly as ultralytics' `process_mask(upsample=False)` does — an uncropped
    prototype mask fires on every instance of the class in the picture.

    Detections are painted LARGEST BOX FIRST, so a small object standing in
    front of a large one keeps its own pixels instead of being overwritten.
    """
    md, mh, mw = proto.shape
    out = np.zeros((mh, mw), np.uint8)
    logits = np.einsum("kn,khw->nhw", coef[:, anchors], proto)   # [n, mh, mw]

    area = (lb_boxes[:, 2] - lb_boxes[:, 0]).clip(0) * \
           (lb_boxes[:, 3] - lb_boxes[:, 1]).clip(0)
    for n in np.argsort(-area):
        b = lb_boxes[n]
        # crop_mask keeps column c when (c >= x1 and c < x2) on the UNROUNDED
        # scaled box, so both bounds ceil with a half-open range. Flooring the
        # low side leaks one column of the neighbouring object.
        x1 = max(0, int(np.ceil(b[0] * mw / imgsz)))
        x2 = min(mw, int(np.ceil(b[2] * mw / imgsz)))
        y1 = max(0, int(np.ceil(b[1] * mh / imgsz)))
        y2 = min(mh, int(np.ceil(b[3] * mh / imgsz)))
        if x2 <= x1 or y2 <= y1:
            continue
        sub = logits[n, y1:y2, x1:x2] > 0
        win = out[y1:y2, x1:x2]
        win[sub] = n + 1
    return out


def _proto_to_image(idx: np.ndarray, r: float, dw: int, dh: int,
                    W: int, H: int, imgsz: int) -> np.ndarray:
    """Index map in prototype space -> index map in original image pixels.

    Prototype space is the LETTERBOXED square shrunk by `mw / imgsz`, so the
    photo occupies the sub-rectangle the padding left behind. One nearest-
    neighbour affine warp undoes both the padding and the resize; nearest is
    required because the values are instance ids, not intensities.
    """
    mh, mw = idx.shape
    kx, ky = mw / imgsz, mh / imgsz
    sx, sy = dw * kx, dh * ky                       # photo origin in proto px
    # `letterbox` rounds the resized side before padding, so the extent has to
    # round the same way or every mask drifts by up to half a prototype cell.
    sw, sh = round(W * r) * kx, round(H * r) * ky   # photo extent in proto px
    M = np.array([[W / sw, 0.0, -sx * W / sw],
                  [0.0, H / sh, -sy * H / sh]], np.float32)
    return cv2.warpAffine(idx, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


# ---------------------------------------------------------------------------
# colour naming, for detectors that emit no class at all
# ---------------------------------------------------------------------------

COLOUR_WORDS = ["black", "white", "gray", "red", "orange", "brown", "beige",
                "yellow", "green", "teal", "blue", "purple", "pink"]
_NO_SHADE = {0, 1, 5, 6, 12}       # words that already carry their lightness


def colour_bins(rgb: np.ndarray) -> np.ndarray:
    """[..., 3] in 0..1 -> index into COLOUR_WORDS, vectorised.

    A plurality vote over these thirteen bins survives texture and shading
    where a mean RGB does not: the mean of a red-and-white striped shirt is
    pink, its plurality is red.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    v = mx
    s = np.where(mx <= 1e-6, 0.0, d / np.maximum(mx, 1e-6))

    dd = np.where(d <= 1e-6, 1.0, d)
    h = np.where(mx == r, 60 * (((g - b) / dd) % 6),
                 np.where(mx == g, 60 * ((b - r) / dd + 2),
                          60 * ((r - g) / dd + 4)))
    h = np.where(h < 0, h + 360, h)

    # np.select evaluates in order, which is what makes this equal to the JS
    # if/else chain: value and saturation decide before hue is consulted at all.
    warm = (h < 12) | (h >= 345)         # brown / pink / red
    return np.select(
        [v < 0.16,
         s < 0.12,
         warm,
         h < 45,
         h < 66,
         h < 160,
         h < 200,
         h < 258,
         h < 300],
        [0,
         np.where(v > 0.86, 1, 2),
         # dark unsaturated red is skin, wood, rust — not red
         np.where((v < 0.62) & (s < 0.62), 5,
                  np.where((s < 0.5) & (v > 0.72), 12, 3)),
         # pale warm tones (wood, skin, sand) read beige
         np.where((v < 0.55) & (s > 0.25), 5,
                  np.where((s < 0.35) | ((v > 0.7) & (s < 0.45)), 6, 4)),
         np.where(v < 0.45, 5, np.where(s < 0.30, 6, 7)),
         8, 9, 10, 11],
        default=12).astype(np.int64)


def colour_labels(frame_bgr: np.ndarray, det: SegDetection) -> List[str]:
    """Name every instance by the colour a person would call it.

    Pixels come from the instance mask; an instance whose mask is empty (or is
    entirely painted over by the ones in front of it) is read from the middle
    of its box instead, inset so the sample is the object rather than whatever
    the box corners caught.
    """
    n = len(det)
    if n == 0:
        return []
    rgb = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    bins = colour_bins(rgb)                                   # [H, W]
    H, W = bins.shape

    words: List[str] = []
    for i in range(n):
        m = det.masks[i] if det.masks is not None else None
        if m is not None and m.sum() >= 24:
            px, samples = bins[m], rgb[m]
        else:
            x1, y1, x2, y2 = det.boxes[i]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            hw, hh = max(1.0, (x2 - x1) * 0.3), max(1.0, (y2 - y1) * 0.3)
            xs = slice(max(0, int(cx - hw)), min(W, int(cx + hw) + 1))
            ys = slice(max(0, int(cy - hh)), min(H, int(cy + hh) + 1))
            px, samples = bins[ys, xs].ravel(), rgb[ys, xs].reshape(-1, 3)
        if px.size == 0:
            words.append("gray")
            continue

        hist = np.bincount(px, minlength=len(COLOUR_WORDS))
        best = int(hist.argmax())
        # Shade prefixes read the winning bin only: the mean over every pixel
        # would drag a dark object towards whatever bright thing shares its box.
        won = samples[px == best]
        mxs = won.max(-1)
        v = float(mxs.mean())
        sat = float(np.where(mxs <= 1e-6, 0.0,
                             (mxs - won.min(-1)) / np.maximum(mxs, 1e-6)).mean())

        w = COLOUR_WORDS[best]
        if best == 2:
            w = "light gray" if v > 0.6 else ("dark gray" if v < 0.32 else "gray")
        elif best not in _NO_SHADE:
            w = ("dark " + w) if v < 0.42 else \
                (("light " + w) if (v > 0.82 and sat < 0.6) else w)
        words.append(w)

    # Two objects the same colour need telling apart, so a repeated word is
    # numbered across every instance of it. The numbering runs left to right,
    # not by confidence, so "brown object 3" is findable in the picture.
    groups: dict[str, List[int]] = {}
    for i, w in enumerate(words):
        groups.setdefault(w, []).append(i)
    out = [""] * n
    for w, ids in groups.items():
        if len(ids) == 1:
            out[ids[0]] = w + " object"
            continue
        ids.sort(key=lambda i: det.boxes[i][0] + det.boxes[i][2])
        for k, i in enumerate(ids):
            out[i] = f"{w} object {k + 1}"
    return out
