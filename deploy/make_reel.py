"""Render the project reel: one model, three ways of handing it regions.

The reel exists to make ONE claim visible, the claim the README opens with:
the relation model reads *regions*, and nothing else. So each shot changes
only where the regions came from, and the graph is predicted the same way
every time.

  boxes    YOLO-World v2-S re-parameterised to MEGASG-497, boxes + class names
  masks    YOLOE-11s prompt-free, instance masks + names from 4,585 classes
  unnamed  FastSAM-s, instance masks and NO class names at all

The third shot type is the argument. FastSAM emits no category for anything it
segments, so there is no object name anywhere in the picture or in the model's
input, and the predicted relations are unchanged in kind.

WHAT THE VIDEO DOES NOT CLAIM. `relsgg.api.RelateAnything.predict` takes an
optional `masks=` argument that rasterises coverage and fill into the head, and
that is NOT what runs here: the exported ONNX graph (`deploy/export_onnx.py`)
takes `image` and `boxes` only, which is also what the browser demo runs. In
the mask shots the segmenter decides the regions and the masks are what you
see; the boxes around them are what the graph reads. Captions say so.

Detector weights are AGPL-3.0 ultralytics derivatives and are not redistributed
here — `--models` points at the web demo's export directory.

    python deploy/make_reel.py --models ../relate-anything/demo/models \
        --out assets/reel

Outputs `reel.mp4` (H.264, yuv420p, faststart — the web-playable combination)
and `hero.gif`, plus `poster.jpg` taken from the frame the reel holds longest.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.postprocess import Triplet, ThresholdConfig, decode      # noqa: E402
from deploy.runtime import DetectorConfig, OnnxDetector, OnnxRelationHead  # noqa: E402
from deploy.seg_detector import OnnxSegDetector, colour_labels       # noqa: E402

# ---------------------------------------------------------------------------
# look: the project page's palette. The page is light, but its image stages are
# dark (`.stage { background: #0a0d12 }`), and this is an image stage.
# ---------------------------------------------------------------------------

GROUND = (10, 13, 18)
INK = (245, 248, 251)
MUTED = (150, 162, 175)
RULE = (34, 41, 51)
SPATIAL = (183, 121, 31)        # --spatial, on the page and in the report
SEMANTIC = (47, 133, 90)        # --semantic
# Region colours: Okabe-Ito, which stays distinguishable for the common colour
# vision deficiencies and on a projector.
REGION = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
          (0, 114, 178), (213, 94, 0), (204, 121, 167), (145, 200, 120)]

W, H = 1280, 720
HEADER, FOOTER = 62, 58
FPS = 30

_FONTS = "/usr/share/fonts/truetype/dejavu"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(os.path.join(_FONTS, name), size)
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------------------
# shot list
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    image: str
    mode: str                    # "boxes" | "masks" | "unnamed"
    conf: float = 0.0            # 0 = the mode's default
    max_edges: int = 4
    note: str = ""

    # filled in by `run_shot`
    boxes: np.ndarray = field(default=None, repr=False)
    labels: List[str] = field(default_factory=list, repr=False)
    masks: Optional[np.ndarray] = field(default=None, repr=False)
    edges: List[Triplet] = field(default_factory=list, repr=False)
    frame: np.ndarray = field(default=None, repr=False)


MODES = {
    # id            detector file            conf   caption
    "boxes":   ("yoloworld-s-megasg497", 0.15,
                "boxes in", "YOLO-World v2-S · 497 classes"),
    "masks":   ("yoloe-11s-pf", 0.25,
                "masks in", "YOLOE-11s prompt-free · 4,585 classes"),
    "unnamed": ("fastsam-s", 0.40,
                "masks in, no class names", "FastSAM-s · class-agnostic"),
}


# ---------------------------------------------------------------------------
# inference
# ---------------------------------------------------------------------------

def build_detectors(models_dir: str, dist_dir: str, threads: int):
    """One session per mode. The box detector is the one this repo ships."""
    det = {}
    local = os.path.join(os.path.dirname(dist_dir), "detector-local", "detector.onnx")
    if os.path.exists(local):
        det["boxes"] = OnnxDetector(local, threads=threads)
    else:
        det["boxes"] = OnnxSegDetector(
            os.path.join(models_dir, "yoloworld-s-megasg497.onnx"), threads=threads)
    for mode in ("masks", "unnamed"):
        det[mode] = OnnxSegDetector(
            os.path.join(models_dir, MODES[mode][0] + ".onnx"), threads=threads)
    return det


def regions(shot: Shot, img_bgr: np.ndarray, detectors, max_boxes: int):
    """Run the shot's detector. Returns (boxes, labels, masks|None)."""
    conf = shot.conf or MODES[shot.mode][1]
    d = detectors[shot.mode]
    if shot.mode == "boxes":
        if isinstance(d, OnnxDetector):
            boxes, scores, labels = d(img_bgr, DetectorConfig(conf=conf, max_det=max_boxes))
            return boxes, labels, None
        r = d(img_bgr, conf=conf, max_det=max_boxes, masks=False)
        return r.boxes, r.labels, None
    r = d(img_bgr, conf=conf, max_det=max_boxes, masks=True)
    if shot.mode == "unnamed":
        # FastSAM names nothing. Colour words exist only so this docstring and
        # the console log can refer to an instance; the reel never draws them.
        return r.boxes, colour_labels(img_bgr, r), r.masks
    return r.boxes, r.labels, r.masks


def dedupe_regions(boxes: np.ndarray, labels: Sequence[str], masks, iou_thr=0.7):
    """Drop regions that are another region again under a different name.

    The detectors run class-AWARE NMS, so a 4,585-class vocabulary happily
    returns the same cat as `persian cat` and as `feline`, the same board as
    `skateboard`, `longboard` and `roller skates`. Both survive NMS because
    their classes differ, and the relation head then predicts the same relation
    once per synonym: `feline sitting on laptop` directly above `persian cat
    sitting on laptop`. A class-agnostic pass at IoU 0.7 collapses them, and is
    loose enough to leave genuinely nested regions (a person and their shirt
    overlap far less than that) alone.

    Detections arrive confidence-sorted, so keeping the first of each cluster
    keeps the detector's own preferred name.
    """
    keep = nms_agnostic(boxes, iou_thr)
    m = masks[keep] if masks is not None else None
    return boxes[keep], [labels[i] for i in keep], m


def nms_agnostic(boxes: np.ndarray, iou_thr: float) -> List[int]:
    area = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)
    keep: List[int] = []
    for i in range(len(boxes)):
        drop = False
        for j in keep:
            xx1, yy1 = max(boxes[i][0], boxes[j][0]), max(boxes[i][1], boxes[j][1])
            xx2, yy2 = min(boxes[i][2], boxes[j][2]), min(boxes[i][3], boxes[j][3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            if inter / (area[i] + area[j] - inter + 1e-9) > iou_thr:
                drop = True
                break
        if not drop:
            keep.append(i)
    return keep


def select_edges(trips: Sequence[Triplet], boxes: np.ndarray, shape: Tuple[int, int],
                 max_edges: int) -> List[Triplet]:
    """Pick the edges to draw, by a stated rule rather than by hand.

    Hand-picking is how a qualitative figure turns into an argument with
    pictures, so the model's ranking is taken in order and only three
    LEGIBILITY constraints remove anything:

      * one edge per unordered pair, so two arrows never overlap exactly;
      * one edge per PREDICATE, so a shot is not three `wearing`. Clothing
        relations are both frequent and confident, and without this a reel of
        six pictures shows four predicates. Dropping the second `wearing` in a
        shot costs a true relation and buys a different true relation; it does
        not promote anything the ranking put below the cut for any other shot;
      * both endpoints at least 0.08 % of the frame, or the arrowhead lands on
        something too small for a viewer to find.

    There is deliberately NO minimum length. An earlier version required the
    centres to be 7 % of the diagonal apart, which reads as a legibility rule
    and is not one: it threw away `person riding skateboard` — the endpoints
    nearly concentric, which is what riding something looks like — and kept
    `doorplate on telegraph pole` at the other end of the street. A rule that
    systematically discards the interaction relations and keeps the background
    furniture is a rule that chooses the result. Short edges are handled where
    the problem actually is, in `bezier`, which bows a short arc harder.
    """
    Himg, Wimg = shape
    area = (boxes[:, 2] - boxes[:, 0]).clip(0) * (boxes[:, 3] - boxes[:, 1]).clip(0)

    out: List[Triplet] = []
    seen_pair, per_pred = set(), {}
    for t in trips:
        s, o = t.subject_idx, t.object_idx
        if s >= len(boxes) or o >= len(boxes):
            continue
        key = (min(s, o), max(s, o))
        if key in seen_pair or t.predicate in per_pred:
            continue
        if min(area[s], area[o]) < 0.0008 * Wimg * Himg:
            continue
        out.append(t)
        seen_pair.add(key)
        per_pred[t.predicate] = 1
        if len(out) >= max_edges:
            break
    return out


def run_shot(shot: Shot, detectors, rel: OnnxRelationHead, thr: ThresholdConfig,
             max_boxes: int) -> Shot:
    img = cv2.imread(shot.image)
    if img is None:
        raise FileNotFoundError(shot.image)
    boxes, labels, masks = regions(shot, img, detectors, max_boxes)
    boxes, labels, masks = dedupe_regions(boxes, labels, masks)
    shot.boxes, shot.labels, shot.masks, shot.frame = boxes, labels, masks, img
    if len(boxes) < 2:
        return shot
    n = min(len(boxes), rel.max_boxes)
    boxes, labels = boxes[:n], labels[:n]
    if masks is not None:
        masks = masks[:n]
    shot.boxes, shot.labels, shot.masks = boxes, labels, masks

    pred, pair, sub, obj, valid = rel(img, boxes)
    trips = decode(pred, pair, sub, obj, valid, rel.predicates, thr,
                   boxes_xyxy=boxes, box_labels=labels)
    shot.edges = select_edges(trips, boxes, img.shape[:2], shot.max_edges)
    return shot


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def fit(img: np.ndarray, box_w: int, box_h: int):
    """Scale to fit, centred. Returns (resized, x0, y0, scale)."""
    h, w = img.shape[:2]
    s = min(box_w / w, box_h / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return r, (box_w - nw) // 2, (box_h - nh) // 2, s


def bezier(p0, p1, bulge: float = 0.18, min_bow: float = 46.0,
           n: int = 48) -> np.ndarray:
    """Quadratic arc from p0 to p1, bowed perpendicular to the chord.

    Straight arrows between region centres collide with the regions they
    connect and with each other; a consistent bow separates them and reads as
    a graph edge rather than a measurement line.

    The bow is a fraction of the chord but never less than `min_bow` pixels,
    which is what makes a short edge legible: `person riding skateboard` has
    nearly concentric endpoints, and a proportional bow would draw it as a
    smudge with the predicate chip sitting on top of both endpoints.

    A negative `bulge` bows the other way; the magnitude floor is applied to
    the absolute value, so the sign survives it.
    """
    p0, p1 = np.asarray(p0, np.float32), np.asarray(p1, np.float32)
    d = p1 - p0
    chord = float(np.hypot(*d)) or 1.0
    perp = np.array([-d[1], d[0]], np.float32) / chord
    bow = max(abs(chord * bulge), min_bow) * (1.0 if bulge >= 0 else -1.0)
    ctrl = (p0 + p1) / 2 + perp * bow
    t = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * ctrl + t ** 2 * p1


def draw_masks(canvas: np.ndarray, masks, ox: int, oy: int, size: Tuple[int, int],
               alpha: float, used: Sequence[int]) -> None:
    """Tint each instance and outline it.

    Only the regions an edge actually touches are tinted. FastSAM returns
    thirty-odd segments for a street scene, and tinting all of them turns the
    photograph into confetti with no graph visible on top of it.
    """
    if masks is None or alpha <= 0:
        return
    pw, ph = size
    roi = canvas[oy:oy + ph, ox:ox + pw]
    for i in used:
        mm = cv2.resize(masks[i].astype(np.uint8), (pw, ph),
                        interpolation=cv2.INTER_NEAREST)
        col = np.array(REGION[i % len(REGION)][::-1], np.float32)
        sel = mm.astype(bool)
        # A ground plane — a tennis court, a road, a lawn — is a region like any
        # other and gets its edges predicted like any other, but filling it
        # tints the entire photograph and the picture stops being a photograph.
        # Outline those; fill the rest.
        if sel.mean() < 0.45:
            roi[sel] = (roi[sel] * (1 - alpha) + col * alpha).astype(np.uint8)
        cont, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(roi, cont, -1, tuple(int(c) for c in col), 2, cv2.LINE_AA)


def draw_boxes(canvas: np.ndarray, boxes, ox, oy, s, alpha, used) -> None:
    """One overlay for every box, so the blend is applied once, not per box."""
    if alpha <= 0:
        return
    over = canvas.copy()
    for i in used:
        x1, y1, x2, y2 = (boxes[i] * s).astype(int)
        col = tuple(int(c) for c in REGION[i % len(REGION)][::-1])
        cv2.rectangle(over, (ox + x1, oy + y1), (ox + x2, oy + y2), (0, 0, 0), 5,
                      cv2.LINE_AA)
        cv2.rectangle(over, (ox + x1, oy + y1), (ox + x2, oy + y2), col, 3, cv2.LINE_AA)
    cv2.addWeighted(over, alpha, canvas, 1 - alpha, 0, canvas)


def draw_edge(canvas: np.ndarray, pts: np.ndarray, colour, progress: float) -> None:
    """Draw `pts` revealed to `progress`, with an arrowhead once complete.

    Every stroke gets a black halo underneath. These arcs cross photographs of
    unknown brightness, and a green line on sunlit grass is invisible without
    one.
    """
    seg = pts[:max(2, int(len(pts) * progress))].astype(np.int32)
    col = tuple(int(c) for c in colour[::-1])
    cv2.polylines(canvas, [seg], False, (0, 0, 0), 7, cv2.LINE_AA)
    cv2.polylines(canvas, [seg], False, col, 3, cv2.LINE_AA)
    if progress < 0.999:
        return
    d = (seg[-1] - seg[-5 if len(seg) >= 5 else 0]).astype(np.float32)
    d /= np.hypot(*d) or 1.0
    perp = np.array([-d[1], d[0]], np.float32)
    tip = seg[-1].astype(np.float32)
    head = np.array([tip, tip - d * 17 + perp * 9, tip - d * 17 - perp * 9], np.int32)
    cv2.fillConvexPoly(canvas, head, (0, 0, 0), cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, (head * 0.86 + tip * 0.14).astype(np.int32), col,
                       cv2.LINE_AA)


PAD = (9, 5)


def chip_box(draw: ImageDraw.ImageDraw, text: str, f) -> Tuple[float, float, float, float]:
    """(w, h, left_bearing, top_bearing) for the chip that would hold `text`."""
    l, t, r, b = draw.textbbox((0, 0), text, font=f)
    return r - l + 2 * PAD[0], b - t + 2 * PAD[1], l, t


def draw_chip(draw: ImageDraw.ImageDraw, rect, text: str, fg, bg, f,
              alpha: int = 255) -> None:
    x0, y0, w, h, l, t = rect
    draw.rounded_rectangle([x0, y0, x0 + w, y0 + h], radius=7, fill=bg + (alpha,))
    draw.text((x0 + PAD[0] - l, y0 + PAD[1] - t), text, font=f, fill=fg + (alpha,))


class Placer:
    """First-fit label placement against everything already placed.

    Chips that overlap are worse than chips slightly off their anchor: two
    predicates stacked on the same pixels are unreadable, whereas a predicate
    sitting 30 px along its own arc still clearly belongs to that arc. Ported
    in spirit from the demo's `placeLabel`.
    """

    def __init__(self, bounds: Tuple[int, int, int, int]):
        self.taken: List[Tuple[float, float, float, float]] = []
        self.bounds = bounds

    def _hits(self, x, y, w, h) -> bool:
        bx0, by0, bx1, by1 = self.bounds
        if x < bx0 or y < by0 or x + w > bx1 or y + h > by1:
            return True
        return any(not (x + w <= p[0] or p[0] + p[2] <= x or
                        y + h <= p[1] or p[1] + p[3] <= y) for p in self.taken)

    def place(self, cands: Sequence[Tuple[float, float]], w: float, h: float):
        """`cands` are centre points, best first. Returns the chosen top-left."""
        for cx, cy in cands:
            x, y = cx - w / 2, cy - h / 2
            if not self._hits(x, y, w, h):
                self.taken.append((x, y, w, h))
                return x, y
        x, y = cands[0][0] - w / 2, cands[0][1] - h / 2
        self.taken.append((x, y, w, h))
        return x, y


def render(shot: Shot, t: float, hud: str) -> np.ndarray:
    """One frame. `t` is seconds since the shot started."""
    canvas = np.full((H, W, 3), GROUND[::-1], np.uint8)
    stage_h = H - HEADER - FOOTER
    photo, ox, oy, s = fit(shot.frame, W - 80, stage_h - 16)
    oy += HEADER + 8
    ox += 40

    # ---- timeline (seconds) -------------------------------------------------
    t_photo, t_reg, t_edge, t_hold = 0.45, 0.75, 0.62, 1.15
    used = sorted({i for e in shot.edges for i in (e.subject_idx, e.object_idx)})

    fade = min(1.0, t / t_photo)
    ph, pw = photo.shape[:2]
    canvas[oy:oy + ph, ox:ox + pw] = (photo * fade).astype(np.uint8)

    reg = float(np.clip((t - t_photo) / t_reg, 0, 1))
    if shot.masks is not None:
        draw_masks(canvas, shot.masks, ox, oy, (pw, ph), 0.42 * reg, used)
    else:
        draw_boxes(canvas, shot.boxes, ox, oy, s, reg, used)

    # ---- edges, one at a time; chips are deferred to the text pass ----------
    edge_t0 = t_photo + t_reg
    arcs: List[Tuple[np.ndarray, str, Tuple[int, int, int]]] = []
    for k, e in enumerate(shot.edges):
        p = float(np.clip((t - edge_t0 - k * t_edge) / t_edge, 0, 1))
        if p <= 0:
            continue
        b1, b2 = shot.boxes[e.subject_idx] * s, shot.boxes[e.object_idx] * s
        c1 = (ox + (b1[0] + b1[2]) / 2, oy + (b1[1] + b1[3]) / 2)
        c2 = (ox + (b2[0] + b2[2]) / 2, oy + (b2[1] + b2[3]) / 2)
        col = SPATIAL if e.predicate in SPATIAL_PREDICATES else SEMANTIC
        # Alternate which side the arc bows to. Relations in these pictures are
        # mostly between nested regions (a person and what they are riding, or
        # wearing), so every arc starts from nearly the same point; bowing them
        # all the same way stacks them into one thick smear.
        pts = bezier(c1, c2, bulge=0.18 * (1 if k % 2 == 0 else -1))
        draw_edge(canvas, pts, col, p)
        if p >= 0.999:
            arcs.append((pts, f"{e.predicate}  {e.score:.2f}", col))

    # ---- text, once, on top of everything ----------------------------------
    out = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(out, "RGBA")
    f_lab, f_pred = font(17, True), font(19, True)
    place = Placer((8, HEADER + 2, W - 8, H - FOOTER - 2))

    # Region names go first: they are pinned to a box and have almost no room
    # to move, whereas a predicate chip can slide along its own arc. Never in
    # the unnamed mode — there are no names to draw there, which is the point.
    if shot.mode != "unnamed" and reg > 0.5:
        a = int(255 * min(1.0, (reg - 0.5) / 0.5))
        for i in used:
            x1, y1, x2, _ = shot.boxes[i] * s
            w, h, l, t_ = chip_box(d, shot.labels[i], f_lab)
            cx, top = ox + x1 + w / 2, oy + y1
            xy = place.place([(cx, top - h / 2 - 4), (cx, top + h / 2 + 4),
                              (ox + x2 - w / 2, top - h / 2 - 4),
                              (cx, top + h * 1.6)], w, h)
            draw_chip(d, (*xy, w, h, l, t_), shot.labels[i], (12, 15, 20),
                      REGION[i % len(REGION)], f_lab, alpha=a)

    for pts, text, col in arcs:
        w, h, l, t_ = chip_box(d, text, f_pred)
        # Slide along the arc first — a chip anywhere on its own arc still
        # clearly belongs to it. Only when the whole arc is occupied (which is
        # the common case for two short arcs between the same nested regions)
        # push outwards from its midpoint, which costs proximity but keeps the
        # two predicates readable.
        cands = [tuple(pts[int(len(pts) * f)]) for f in
                 (0.5, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74, 0.18, 0.82)]
        mid = pts[len(pts) // 2]
        for step in (1, 2, 3):
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0),
                           (-1, -1), (1, -1), (-1, 1), (1, 1)):
                cands.append((mid[0] + dx * step * (w * 0.62),
                              mid[1] + dy * step * (h * 1.5)))
        xy = place.place(cands, w, h)
        draw_chip(d, (*xy, w, h, l, t_), text, INK, col, f_pred)

    _chrome(d, shot, hud)
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


SPATIAL_PREDICATES = {
    "on", "in", "above", "below", "behind", "in front of", "beside", "near",
    "next to", "under", "inside", "on top of", "to the left of", "attached to",
    "to the right of", "surrounding", "contained in", "hanging from", "part of",
}


def _chrome(d: ImageDraw.ImageDraw, shot: Shot, hud: str) -> None:
    """Header caption, footer HUD, legend."""
    caption, detector = MODES[shot.mode][2], MODES[shot.mode][3]
    d.text((40, 20), caption, font=font(24, True), fill=INK + (255,))
    w = d.textlength(detector, font=font(16))
    d.text((W - 40 - w, 26), detector, font=font(16), fill=MUTED + (255,))
    d.line([(40, HEADER - 8), (W - 40, HEADER - 8)], fill=RULE + (255,), width=1)

    y = H - FOOTER + 16
    d.text((40, y), hud, font=font(16), fill=MUTED + (255,))
    # legend, right-aligned
    items = [("spatial", SPATIAL), ("semantic", SEMANTIC)]
    x = W - 40
    for name, col in reversed(items):
        tw = d.textlength(name, font=font(15))
        d.text((x - tw, y + 1), name, font=font(15), fill=MUTED + (255,))
        d.ellipse([x - tw - 18, y + 6, x - tw - 8, y + 16], fill=col + (255,))
        x -= tw + 34


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------

def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return "ffmpeg"


def encode_mp4(frames, path: str, fps: int = FPS, crf: int = 20) -> None:
    """H.264 / yuv420p / faststart — the combination every browser plays."""
    cmd = [ffmpeg_exe(), "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.PIPE)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(p.stderr.read().decode()[-2000:])


def encode_gif(mp4: str, path: str, width: int = 720, fps: int = 12) -> None:
    """Two-pass palettegen. A GIF quantised per-frame bands the photographs."""
    vf = f"fps={fps},scale={width}:-1:flags=lanczos"
    pal = path + ".png"
    subprocess.run([ffmpeg_exe(), "-y", "-i", mp4, "-vf", vf + ",palettegen=stats_mode=diff",
                    pal], check=True, capture_output=True)
    subprocess.run([ffmpeg_exe(), "-y", "-i", mp4, "-i", pal, "-lavfi",
                    vf + "[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3",
                    "-loop", "0", path], check=True, capture_output=True)
    os.remove(pal)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", required=True,
                    help="web demo model directory (yoloe-11s-pf.onnx, fastsam-s.onnx, ...)")
    ap.add_argument("--dist", default="deploy/dist/relsgg-vits16plus")
    ap.add_argument("--shots", default="assets/reel/shots.json")
    ap.add_argument("--out", default="assets/reel")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--max_boxes", type=int, default=24)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--gif_width", type=int, default=720)
    ap.add_argument("--no_gif", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    spec = json.load(open(args.shots))
    root = os.path.dirname(os.path.abspath(args.shots))
    shots = [Shot(image=os.path.join(root, s["image"]), mode=s["mode"],
                  conf=s.get("conf", 0.0), max_edges=s.get("max_edges", 4),
                  note=s.get("note", "")) for s in spec["shots"]]

    detectors = build_detectors(args.models, args.dist, args.threads)
    rel = OnnxRelationHead(os.path.join(args.dist, "relateanything.onnx"),
                           bank_path=os.path.join(args.dist, "predicate_bank.npz"),
                           threads=args.threads)
    thr = ThresholdConfig(threshold=args.threshold)

    frames: List[np.ndarray] = []
    finals: List[Tuple[int, int]] = []          # (edge count, last frame index)
    for i, shot in enumerate(shots):
        run_shot(shot, detectors, rel, thr, args.max_boxes)
        hud = (f"{len(shot.boxes)} regions · {len(shot.edges)} relations drawn"
               f" · {os.path.basename(shot.image)}")
        print(f"[{i+1}/{len(shots)}] {os.path.basename(shot.image):18s} {shot.mode:8s}"
              f" {len(shot.boxes):3d} regions, {len(shot.edges)} edges: "
              + ", ".join(f"{e.subject_label or e.subject_idx} {e.predicate} "
                          f"{e.object_label or e.object_idx}" for e in shot.edges))
        if len(shot.edges) == 0:
            continue
        dur = 0.45 + 0.75 + 0.62 * (len(shot.edges) + 1) + 1.15
        for n in range(int(dur * FPS)):
            frames.append(render(shot, n / FPS, hud))
        finals.append((len(shot.edges), len(frames) - 1))

    mp4 = os.path.join(args.out, "reel.mp4")
    encode_mp4(frames, mp4)
    print(f"wrote {mp4}  ({len(frames)} frames, {len(frames)/FPS:.1f}s, "
          f"{os.path.getsize(mp4)/1e6:.2f} MB)")

    # The poster is what a visitor sees before pressing play, so it should be a
    # finished graph rather than whatever frame a fixed offset lands on: take
    # the last frame of the shot that ended up with the most edges.
    poster = os.path.join(args.out, "poster.jpg")
    cv2.imwrite(poster, frames[max(finals)[1]], [cv2.IMWRITE_JPEG_QUALITY, 90])

    if not args.no_gif:
        gif = os.path.join(args.out, "hero.gif")
        encode_gif(mp4, gif, width=args.gif_width)
        print(f"wrote {gif}  ({os.path.getsize(gif)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
