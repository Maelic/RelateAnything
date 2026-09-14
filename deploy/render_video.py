#!/usr/bin/env python3
"""Offline video renderer for the demo reel -- the stabilised version of demo_webcam.

The relation head is per-frame and has no memory, so running it straight over
video and drawing what comes back looks broken: boxes jitter by a few pixels
every frame, the detector alternates between 'man' and 'person' for the same
region, and a relation that sits right on the threshold blinks on and off. None
of that is the model being wrong -- it is the model being asked a fresh question
30 times a second -- but on a project page it reads as instability.

Four fixes, all of them here and none of them in the model:

  1. IoU tracking          detections get persistent identities, so a subject
                           keeps its colour and its label across the clip
  2. EMA smoothing         box coordinates and relation scores are low-passed;
                           boxes also carry a velocity term, because an EMA
                           alone lags behind anything that moves
  3. hysteresis            a relation must clear `on_thr` for `on_frames`
                           before it appears and fall under `off_thr` for
                           `off_frames` before it leaves -- one threshold with
                           two sides is what makes edges blink
  4. alpha fades           entering and leaving edges ramp instead of popping

Two modes:

    python deploy/render_video.py --scan footage/          # rank clips, render nothing
    python deploy/render_video.py --render footage/clip.webm --out reel/clip.mp4

--scan is how you pick the footage: it samples frames through each clip, runs
the real pipeline, and ranks by how much confident *interaction* (non-spatial)
relation content is actually in there. Cheaper than watching 30 videos, and it
ranks by what the model sees rather than by what the thumbnail promises.

Inference runs at --stride (default 2) but every frame is rendered: the smoother
carries the overlay through the gaps, which is most of the reason this is
affordable on a CPU. The HUD deliberately shows no FPS -- these are offline
renders on whatever machine ran them, and a CPU number would undersell a model
the page times at 25 ms on an A40.
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field

import cv2
import imageio_ffmpeg
import numpy as np

# imageio warns once per opened stream that the scaled output size differs from
# the source -- which is the whole point of passing a scale filter.
logging.getLogger("imageio_ffmpeg").setLevel(logging.ERROR)
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.postprocess import ThresholdConfig, decode    # noqa: E402
from deploy.runtime import DetectorConfig, ScenePipeline  # noqa: E402

# Interaction predicates: what the reel is for. Spatial ones ('on', 'behind')
# are true but dull, and they crowd out the interesting edges in a top-k.
INTERACTION = [
    "riding", "holding", "carrying", "playing", "using", "wearing",
    "sitting on", "sitting at", "standing on", "looking at", "watching",
    "eating", "eating from", "drinking from", "cutting", "pushing", "pulling",
    "climbing", "talking to", "walking with", "leaning against", "reading",
    "playing with", "feeding", "petting", "hitting", "kicking", "driving",
    "steering", "pedaling", "lying on", "hugging", "embracing", "serving",
    "preparing", "operating", "typing on", "photographing", "pointing at",
    "shaking hands with", "holding hands with", "dancing with", "gripping",
    "reaching for", "swinging", "leading", "following", "walking on",
    "jumping over", "resting on", "hanging from", "attached to", "touching",
]
SPATIAL = ["on", "inside", "on top of", "in front of", "beside", "behind",
           "above", "below", "next to", "near", "under", "over"]

FONTS = ["/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]

# Distinct hues that survive video compression and read on both bright and dark
# footage. RGB here (PIL); the frame is converted once per render.
PALETTE = [(64, 156, 255), (255, 99, 88), (52, 199, 123), (255, 184, 48),
           (191, 122, 255), (0, 199, 208), (255, 133, 71), (145, 200, 60)]


def _font(size: int):
    for p in FONTS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[Na, Nb] pairwise IoU. Empty-safe, which the association loop relies on."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(ar_a[:, None] + ar_b[None, :] - inter, 1e-6)).astype(np.float32)


# --------------------------------------------------------------------------- video io
# Everything reads through ffmpeg rather than cv2.VideoCapture. A third of the
# Commons footage is AV1, which the OpenCV wheel's bundled decoder cannot open
# at all -- it returns an empty capture, so the clip looks corrupt rather than
# unsupported. The ffmpeg imageio ships has libdav1d and reads all of it.
def probe(path: str) -> dict:
    gen = imageio_ffmpeg.read_frames(path, pix_fmt="bgr24")
    try:
        meta = next(gen)
    finally:
        gen.close()
    w, h = meta["size"]
    return {"w": w, "h": h, "fps": float(meta.get("fps") or 25.0),
            "duration": float(meta.get("duration") or 0.0)}


def even(n: int) -> int:
    return int(n) // 2 * 2


def target_size(w: int, h: int, width: int, upscale: bool = False) -> tuple[int, int]:
    """Downscale to fit `width`, and upscale to it when a crop asked for it.

    Upscaling adds no information, but the detector letterboxes to 640px
    regardless: a 435px crop left at 435px hands it a subject a third the size
    it would get from the same crop scaled to 960 first. That is the difference
    between finding the rider and labelling his torso 'backpack'."""
    s = width / max(w, h)
    if not upscale:
        s = min(1.0, s)
    return even(w * s), even(h * s)


def _vf(W: int, H: int, pad: bool, crop=None, src=None) -> str:
    pre = ""
    if crop and src:
        # Normalised [x0, y0, x1, y1] of the source frame. Commons footage is
        # mostly wide establishing shots, and a relation drawn on a subject 80px
        # tall is legible to nobody -- framing is the difference between a demo
        # and a surveillance still.
        sw, sh = src
        x0, y0, x1, y1 = crop
        cw, ch = even((x1 - x0) * sw), even((y1 - y0) * sh)
        cx, cy = even(x0 * sw), even(y0 * sh)
        pre = f"crop={cw}:{ch}:{cx}:{cy},"
    if not pad:
        return pre + f"scale={W}:{H}"
    # Fit inside the canvas and pad the remainder. A reel mixes 16:9 and
    # vertical phone footage, and the concat demuxer will not join streams of
    # differing dimensions -- so every segment is forced onto one canvas.
    return (pre + f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black")


def read_frames(path: str, W: int, H: int, start: float = 0.0,
                duration: float = 0.0, pad: bool = False, fps: float = 0.0,
                crop=None, src=None):
    """Sequential BGR frames, scaled by ffmpeg. -ss before -i so the seek is
    a keyframe jump instead of a decode-and-discard of everything before it."""
    ip = ["-ss", f"{start:.3f}"] if start > 0 else []
    vf = _vf(W, H, pad, crop, src) + (f",fps={fps:.4f}" if fps else "")
    op = ["-vf", vf]
    if duration > 0:
        op += ["-t", f"{duration:.3f}"]
    gen = imageio_ffmpeg.read_frames(path, pix_fmt="bgr24",
                                     input_params=ip, output_params=op)
    next(gen)                                    # meta
    for buf in gen:
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3)


def grab(path: str, t: float, W: int, H: int) -> "np.ndarray | None":
    """One frame at t seconds. Used by --scan, which wants a spread of frames
    and would otherwise decode whole clips to look at ten of their frames."""
    try:
        gen = imageio_ffmpeg.read_frames(
            path, pix_fmt="bgr24", input_params=["-ss", f"{t:.3f}"],
            output_params=["-vf", f"scale={W}:{H}", "-frames:v", "1"])
        next(gen)
        buf = next(gen)
        gen.close()
        return np.frombuffer(buf, np.uint8).reshape(H, W, 3)
    except (StopIteration, RuntimeError, OSError):
        return None


# --------------------------------------------------------------------------- tracks
class KalmanBox:
    """Constant-velocity Kalman filter on (cx, cy, w, h).

    This replaces the EMA-plus-velocity smoother that was here before. An EMA
    has one knob and uses it for two jobs: crank it down and the box stops
    jittering but lags behind anything moving, crank it up and it tracks motion
    but shakes. A Kalman filter separates them — process noise says how much the
    object really moves, measurement noise says how much the detector's box
    wobbles — so a fast subject stays locked on while a stationary one stops
    trembling. It also predicts through frames with no detection, which is what
    carries a track across an occlusion and across the gaps left by --stride.
    """

    __slots__ = ("x", "P", "_q", "_r")

    def __init__(self, box: np.ndarray, q: float = 1.0, r: float = 10.0):
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        w, h = box[2] - box[0], box[3] - box[1]
        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], np.float64)
        # Position starts trusted, velocity does not: one observation says
        # nothing about how fast a thing is going.
        self.P = np.diag([10., 10., 10., 10., 1e3, 1e3, 1e3, 1e3])
        self._q, self._r = q, r

    def predict(self) -> None:
        self.x[:4] += self.x[4:]
        q = self._q
        self.P[:4, :4] += self.P[4:, 4:] + np.diag([q, q, q, q])
        self.P[4:, 4:] += np.diag([q * .01] * 4)

    def update(self, box: np.ndarray) -> None:
        z = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2,
                      box[2] - box[0], box[3] - box[1]], np.float64)
        y = z - self.x[:4]
        S = self.P[:4, :4] + np.diag([self._r] * 4)
        K = np.zeros((8, 4))
        Sinv = np.linalg.inv(S)
        K[:4] = self.P[:4, :4] @ Sinv
        K[4:] = self.P[4:, :4] @ Sinv
        self.x += K @ y
        self.x[2:4] = np.maximum(self.x[2:4], 2.0)
        self.P[:4, :4] -= K[:4] @ self.P[:4, :4]
        self.P[4:, 4:] -= K[4:] @ self.P[:4, 4:]

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], np.float32)


@dataclass
class Track:
    tid: int
    kf: KalmanBox
    conf: float
    votes: Counter = field(default_factory=Counter)
    hits: int = 0
    misses: int = 0
    alpha: float = 0.0
    mask: "np.ndarray | None" = None
    mask_seen: bool = False

    @property
    def box(self) -> np.ndarray:
        return self.kf.box

    @property
    def label(self) -> str:
        """Majority vote, not the last frame's answer. The 497-class detector
        flips between 'man' and 'person' on the same region from frame to frame
        and a label that changes every frame looks like a bug."""
        return self.votes.most_common(1)[0][0] if self.votes else "?"


class Tracker:
    """Two-stage IoU association over Kalman-filtered boxes.

    The two stages are ByteTrack's idea, reimplemented here rather than imported:
    match confident detections first, then give still-unmatched tracks a second
    chance against the leftovers the detector was unsure about. A subject that
    dips to 0.2 confidence for a few frames — turning away, motion blur — keeps
    its identity instead of dying and coming back as a new colour.

    Assignment is greedy on descending IoU. Hungarian would be the textbook
    answer, but with under 32 boxes behind an IoU gate the assignments are not
    contested, and greedy needs no solver dependency.
    """

    def __init__(self, iou_thr=0.3, low_iou_thr=0.5, merge_iou=0.75, max_age=12,
                 min_hits=2, high_conf=0.25, q=1.0, r=10.0, stride=1):
        self.iou_thr, self.low_iou_thr = iou_thr, low_iou_thr
        self.merge_iou, self.max_age, self.min_hits = merge_iou, max_age, min_hits
        self.high_conf, self.q, self.r = high_conf, q, r
        self.stride = max(1, int(stride))
        self.tracks: list[Track] = []
        self._next = 0

    # -- association ------------------------------------------------------
    def _match(self, cand, boxes, tracks, thr):
        """Greedy IoU matching. Returns [(det_idx, Track)] and what went unused."""
        if not cand or not tracks:
            return [], list(cand), list(tracks)
        M = iou_matrix(boxes[cand], np.stack([t.box for t in tracks]))
        pairs = sorted(((M[r_, c], r_, c) for r_ in range(M.shape[0])
                        for c in range(M.shape[1]) if M[r_, c] >= thr), reverse=True)
        used_d, used_t, out = set(), set(), []
        for _, r_, c in pairs:
            if r_ in used_d or c in used_t:
                continue
            used_d.add(r_); used_t.add(c)
            out.append((cand[r_], tracks[c]))
        return (out,
                [d for i, d in enumerate(cand) if i not in used_d],
                [t for i, t in enumerate(tracks) if i not in used_t])

    def observe(self, boxes, labels, confs, masks=None, frame_area=0.0,
                max_frac=1.0) -> dict[int, int]:
        """Associate detections to tracks. Returns detection index -> track id.

        Detections are NEVER compacted: a triplet addresses its endpoints by
        index into the detector's own output, so dropping a row and re-indexing
        would silently re-point every relation after it at the wrong object.
        Rejected rows are skipped by index instead.
        """
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        confs = np.asarray(confs, np.float32)
        ok = np.ones(len(boxes), bool)
        if frame_area > 0 and max_frac < 1.0 and len(boxes):
            area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    / frame_area)
            ok &= area <= max_frac
        # Cross-class duplicate suppression: the detector emits 'man' and
        # 'person' as separate classes over the same pixels and class-aware NMS
        # has no reason to merge them, so without this every relation is drawn
        # twice with two different subject names.
        keep, kept_boxes = [], []
        for i in np.argsort(-confs) if len(boxes) else []:
            if not ok[i]:
                continue
            if kept_boxes and iou_matrix(boxes[i:i + 1],
                                         np.array(kept_boxes)).max() > self.merge_iou:
                continue
            keep.append(int(i)); kept_boxes.append(boxes[i])

        hi = [d for d in keep if confs[d] >= self.high_conf]
        lo = [d for d in keep if confs[d] < self.high_conf]

        det2track: dict[int, int] = {}
        matched, un_hi, un_tracks = self._match(hi, boxes, self.tracks, self.iou_thr)
        # second stage: unmatched tracks against the low-confidence leftovers
        matched2, _, un_tracks = self._match(lo, boxes, un_tracks, self.low_iou_thr)

        for d, t in matched + matched2:
            t.kf.update(boxes[d])
            t.conf = float(confs[d]); t.votes[labels[d]] += 1
            t.hits += 1; t.misses = 0
            if masks is not None and d < len(masks):
                t.mask = masks[d]; t.mask_seen = True
            det2track[d] = t.tid

        for d in un_hi:                        # confident and unexplained: new track
            t = Track(self._next, KalmanBox(boxes[d], self.q, self.r), float(confs[d]))
            t.votes[labels[d]] += 1; t.hits = 1
            if masks is not None and d < len(masks):
                t.mask = masks[d]; t.mask_seen = True
            self.tracks.append(t); det2track[d] = t.tid
            self._next += 1

        for t in un_tracks:
            t.misses += 1
            t.mask = None                      # a stale mask is worse than none
        self.tracks = [t for t in self.tracks if t.misses <= self.max_age]
        return det2track

    def step(self, W: int, H: int, fade=0.18):
        """Per RENDERED frame. The Kalman predicts on every one, so between
        inferences the box keeps moving instead of freezing until the next."""
        for t in self.tracks:
            t.kf.predict()
            b = t.kf.box
            b[0] = min(max(b[0], 0), W - 2); b[1] = min(max(b[1], 0), H - 2)
            b[2] = min(max(b[2], b[0] + 2), W); b[3] = min(max(b[3], b[1] + 2), H)
            t.kf.x[0] = (b[0] + b[2]) / 2; t.kf.x[1] = (b[1] + b[3]) / 2
            t.kf.x[2] = b[2] - b[0]; t.kf.x[3] = b[3] - b[1]
            vis = t.hits >= self.min_hits and t.misses <= 2
            t.alpha = min(1.0, t.alpha + fade) if vis else max(0.0, t.alpha - fade)

    def by_id(self) -> dict[int, Track]:
        return {t.tid: t for t in self.tracks}


# --------------------------------------------------------------------------- edges
class RelationBelief:
    """A 1-D Kalman filter on the calibrated LOG-ODDS that one relation holds.

    Filtering the score would be wrong twice over. The score is a sigmoid, and
    averaging probabilities is not how independent evidence combines -- log-odds
    add, probabilities do not. Logits are also unbounded, so the Gaussian a
    Kalman filter assumes is at least defensible there.

    There is no velocity term. A box has one; a relation does not. Giving it one
    would extrapolate a relation "becoming more true", which means nothing.

    The state persists when nothing is observed: `predict` widens the belief
    without moving it. That is the entire point -- see EdgeBook.observe for why
    the old EMA could not express it.
    """

    __slots__ = ("x", "P")

    def __init__(self, z: float, P0: float = 4.0):
        self.x, self.P = float(z), float(P0)

    def predict(self, q: float = 0.05) -> None:
        self.P += q

    def update(self, z: float, r: float = 1.0, p_min: float = 0.04) -> None:
        K = self.P / (self.P + r)
        self.x += K * (z - self.x)
        self.P *= (1 - K)
        # Consecutive frames are nearly the same image, so their measurements
        # are strongly correlated -- but this filter treats them as independent
        # evidence and P collapses within a few frames, after which nothing can
        # move the belief. Flooring P keeps it able to change its mind.
        self.P = max(self.P, p_min)

    @property
    def prob(self) -> float:
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, self.x))))



@dataclass
class Edge:
    sub: int
    obj: int
    pred: str
    score: float
    spatial: bool
    on: bool = False
    on_run: int = 0
    off_run: int = 0
    alpha: float = 0.0
    seen: int = 0
    bel: "RelationBelief | None" = None
    shown: float = 0.0
    """What gets drawn. An edge on its way out keeps its last confident value
    rather than counting itself down: the EMA decay is a display artifact of
    letting go, and rendering it reads as the model losing confidence."""


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


class EdgeBook:
    """EMA + hysteresis over (subject track, object track, predicate)."""

    def __init__(self, score_ema=0.35, on_thr=0.50, off_thr=0.38,
                 on_frames=2, off_frames=6, fade=0.12, max_edges=6,
                 max_spatial=0, spatial_on_thr=None, belief=False,
                 kf_q=0.05, kf_r=1.0, p_max=8.0):
        self.score_ema, self.on_thr, self.off_thr = score_ema, on_thr, off_thr
        self.on_frames, self.off_frames = on_frames, off_frames
        self.fade, self.max_edges = fade, max_edges
        # Spatial predicates are easy and numerous -- almost everything is
        # 'near' almost everything -- so on a shared top-k they crowd out the
        # interactions that are the point. They get their own budget instead.
        self.max_spatial = max_spatial
        self.spatial_on_thr = (on_thr if spatial_on_thr is None else spatial_on_thr)
        self.belief, self.kf_q, self.kf_r, self.p_max = belief, kf_q, kf_r, p_max
        self.edges: dict[tuple, Edge] = {}

    def observe(self, triplets, det2track, is_spatial: dict, raw=None):
        """raw, when given, switches on the Kalman belief filter.

        The EMA path below decays every edge it did not see this frame. But an
        edge goes missing for two unrelated reasons: the model looked at the
        pair and scored it low (real negative evidence), or an endpoint was not
        detected at all (no measurement). Decaying both means a relation dies
        whenever an object flickers, which is the opposite of persistence. The
        belief path separates them -- update when the pair was measurable,
        predict when it was not -- which is the whole reason it exists.
        """
        if raw is not None and self.belief:
            return self._observe_belief(triplets, det2track, is_spatial, raw)
        fresh: dict[tuple, float] = {}
        for t in triplets:
            s, o = det2track.get(t.subject_idx), det2track.get(t.object_idx)
            if s is None or o is None or s == o:
                continue
            k = (s, o, t.predicate)
            # Two detection pairs can land on one track pair once duplicates
            # are merged; keep the stronger reading rather than the last one.
            fresh[k] = max(fresh.get(k, 0.0), float(t.score))
        for k, sc in fresh.items():
            e = self.edges.get(k)
            if e is None:
                e = Edge(k[0], k[1], k[2], sc, bool(is_spatial.get(k[2], False)))
                self.edges[k] = e
            else:
                e.score = (1 - self.score_ema) * e.score + self.score_ema * sc
            if e.score >= self.off_thr:
                e.shown = e.score
            e.seen += 1
        for k, e in self.edges.items():        # decay what was not re-observed
            if k not in fresh:
                e.score = (1 - self.score_ema) * e.score
        self._hysteresis()

    def _hysteresis(self):
        """Run counters advance per OBSERVATION, and BOTH score paths need this.

        It used to live inside the EMA loop, so the belief path returned before
        reaching it: every hypothesis was filtered correctly and none of them
        ever turned on.
        """
        for e in self.edges.values():
            on_thr = self.spatial_on_thr if e.spatial else self.on_thr
            if e.score >= on_thr:
                e.on_run += 1; e.off_run = 0
            elif e.score < self.off_thr:
                e.off_run += 1; e.on_run = 0
            else:
                e.on_run = e.off_run = 0       # the dead band: hold whatever we are
            if not e.on and e.on_run >= self.on_frames:
                e.on = True
            elif e.on and e.off_run >= self.off_frames:
                e.on = False

    def _observe_belief(self, triplets, det2track, is_spatial, raw):
        pred, pair, kof, contract, vidx = raw
        track2det = {t: d for d, t in det2track.items()}

        # Birth: promote whatever decode surfaced this frame. Filtering all
        # K x V hypotheses is not on (24 boxes is ~36k); a few dozen live ones
        # cost nothing.
        for t in triplets:
            s_, o_ = det2track.get(t.subject_idx), det2track.get(t.object_idx)
            v = vidx.get(t.predicate)
            if s_ is None or o_ is None or s_ == o_ or v is None:
                continue
            k = (s_, o_, t.predicate)
            if k not in self.edges:
                e = Edge(s_, o_, t.predicate, float(t.score),
                         bool(is_spatial.get(t.predicate, False)))
                e.bel = RelationBelief(_logit(float(t.score)))
                self.edges[k] = e

        dead = []
        for k, e in self.edges.items():
            v = vidx.get(e.pred)
            ds, do = track2det.get(e.sub), track2det.get(e.obj)
            kk = kof.get((ds, do)) if (ds is not None and do is not None) else None
            if kk is None or v is None:
                e.bel.predict(self.kf_q)          # unmeasurable: hold the belief
            else:
                # Straight off the raw matrices, NOT through decode: decode
                # applies top-k and a threshold, so a pair that scored low never
                # comes back and negative evidence can never be observed.
                z = float(contract.fuse(pred[kk], pair[kk])[v])
                e.bel.update(z, self.kf_r)
                e.seen += 1
            e.score = e.bel.prob
            if e.score >= self.off_thr:
                e.shown = e.score
            if e.bel.P > self.p_max:              # unobserved too long
                dead.append(k)
        for k in dead:
            self.edges.pop(k, None)
        self._hysteresis()

    def step(self, live_tracks: set[int]):
        dead = []
        for k, e in self.edges.items():
            if e.sub not in live_tracks or e.obj not in live_tracks:
                e.on, e.alpha = False, max(0.0, e.alpha - self.fade)
                if e.alpha <= 0:
                    dead.append(k)
                continue
            e.alpha = (min(1.0, e.alpha + self.fade) if e.on
                       else max(0.0, e.alpha - self.fade))
            if not e.on and e.alpha <= 0 and e.score < 0.05:
                dead.append(k)
        for k in dead:
            self.edges.pop(k, None)

    def visible(self) -> list[Edge]:
        vis = [e for e in self.edges.values() if e.alpha > 0.01]
        # One edge per (object, predicate). A class-agnostic segmenter splits a
        # person into several regions, and each of them then "plays" the same
        # guitar -- the same claim drawn three times. Keyed on the object rather
        # than the predicate alone, so "wearing hat" and "wearing shoes" both
        # survive: those are different claims.
        best: dict[tuple, Edge] = {}
        for e in sorted(vis, key=lambda e: -max(e.shown, e.score)):
            best.setdefault((e.obj, e.pred), e)
        # Letting a pair emit two predicates (--max_per_pair 2) is what gets
        # 'holding' alongside 'playing'. It also gets 'playing' alongside
        # 'playing with', and 'sitting on' alongside 'sitting at', which say the
        # same thing twice. Same pair + same leading word = keep the stronger.
        head: dict[tuple, Edge] = {}
        for e in sorted(best.values(), key=lambda e: -max(e.shown, e.score)):
            head.setdefault((e.sub, e.obj, e.pred.split()[0]), e)
        vis = list(head.values())
        vis.sort(key=lambda e: -max(e.shown, e.score))
        inter = [e for e in vis if not e.spatial]
        spat = [e for e in vis if e.spatial][:self.max_spatial]
        # Interactions get first call on the budget; spatial fills what is left.
        inter = inter[:max(0, self.max_edges - len(spat))]
        return inter + spat


# --------------------------------------------------------------------------- segmentation
class SegDetector:
    """ultralytics segmentation front-end, feeding the SAME ONNX relation head.

    Two kinds, and the difference is the point of the reel's last act:

      yoloe   YOLOE-11l re-parameterised to MEGASG-497 — masks WITH names, from
              the same 497-word vocabulary the relation model was trained
              against.
      fastsam FastSAM — class-agnostic. Its entire vocabulary is {0: 'object'},
              so when the reel drops labels there is genuinely no name anywhere,
              in the picture or in the input. The relation head never saw names
              in either case: it is handed boxes and pixels, never words.
    """

    def __init__(self, weights: str, kind: str = "yoloe", device: str = "cpu",
                 imgsz: int = 640):
        self.kind, self.device, self.imgsz = kind, device, imgsz
        if kind == "fastsam":
            from ultralytics import FastSAM
            self.model = FastSAM(weights)
            self.names = {0: "object"}
        else:
            from ultralytics import YOLOE
            self.model = YOLOE(weights)
            self.names = self.model.names
        self.classes = [self.names[i] for i in sorted(self.names)]

    def __call__(self, frame_bgr, conf=0.25, iou=0.5, max_det=24):
        kw = dict(imgsz=self.imgsz, device=self.device, retina_masks=True,
                  verbose=False)
        if self.kind == "fastsam":
            # FastSAM segments everything it can see, background included, and
            # a relation graph over 24 regions is unreadable. Keep the largest
            # few: the subject and what it is interacting with are never the
            # small ones.
            r = self.model.predict(frame_bgr, conf=conf, iou=0.9, **kw)[0]
        else:
            r = self.model.predict(frame_bgr, conf=conf, iou=iou,
                                   max_det=max_det, **kw)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [], None)
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        cf = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        labels = [self.names.get(int(c), "object") for c in cls]
        masks = None
        if r.masks is not None:
            masks = r.masks.data.cpu().numpy().astype(bool)
        if self.kind == "fastsam":
            area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            order = np.argsort(-area)[:max_det]
            xyxy, cf, cls = xyxy[order], cf[order], cls[order]
            labels = [labels[i] for i in order]
            if masks is not None:
                masks = masks[order]
        return xyxy, cf, labels, masks


def paint_masks(frame_bgr, tracks, alpha=0.38):
    """Composite track-coloured masks under the overlay, with a crisp edge.

    Done in numpy on the BGR frame rather than through PIL: a 1280x720 alpha
    composite per mask per frame is the one place in this renderer where the
    naive version actually costs something.
    """
    out = frame_bgr
    for t in tracks:
        if t.mask is None or t.alpha <= 0.01:
            continue
        m = t.mask
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(m.astype(np.uint8), (out.shape[1], out.shape[0]),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        r, g, b = PALETTE[t.tid % len(PALETTE)]
        col = np.array([b, g, r], np.float32)          # PALETTE is RGB, frame is BGR
        a = alpha * t.alpha
        out[m] = (out[m] * (1 - a) + col * a).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (int(b), int(g), int(r)), 2, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- drawing
def _overlaps(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _place(box, occupied, W, H):
    """Slide a chip vertically until it stops colliding with the chips already
    down. Without this, boxes that overlap -- which for a relation model is most
    of them, since subject and object are usually adjacent -- stack their labels
    into an unreadable pile and the later one wins on draw order alone."""
    w, h = box[2] - box[0], box[3] - box[1]
    x = min(max(box[0], 2), max(2, W - w - 2))
    # Six steps. Three was enough when a frame carried four edges; at eight
    # plus their object labels the chips run out of room and start stacking.
    # Sliding further than this stops labelling anything, so past it,
    # overlapping in the right place beats being legible in the wrong one.
    for step in range(7):
        for dy in ((0,) if step == 0 else (-(h + 5) * step, (h + 5) * step)):
            y = box[1] + dy
            if y < 2 or y + h > H - 2:
                continue
            cand = [x, y, x + w, y + h]
            if not any(_overlaps(cand, o) for o in occupied):
                return cand
    return [x, min(max(box[1], 2), H - h - 2), x + w,
            min(max(box[1], 2), H - h - 2) + h]


def _chip(dr, xy, text, font, fg, bg, alpha, occupied=None, W=10**5, H=10**5,
          pad=6, r=7):
    x, y = xy
    w = int(dr.textlength(text, font=font)); h = font.size + 2
    box = [x, y, x + w + 2 * pad, y + h + 2 * pad - 2]
    if occupied is not None:
        box = _place(box, occupied, W, H)
        occupied.append(box)
    dr.rounded_rectangle(box, radius=r, fill=bg + (int(230 * alpha),))
    dr.text((box[0] + pad, box[1] + pad - 2), text, font=font,
            fill=fg + (int(255 * alpha),))
    return box[2] - box[0], box[3] - box[1]


def _tag(img, text, font):
    """Burn a corner caption straight onto a rendered BGR frame."""
    im = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im)
    w = int(dr.textlength(text, font=font))
    dr.rectangle([8, 8, 8 + w + 16, 8 + font.size + 12], fill=(18, 18, 20))
    dr.text((16, 13), text, font=font, fill=(255, 255, 255))
    img[:, :, :] = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    return img


def render_frame(frame_bgr, tracker: Tracker, book: EdgeBook, fonts, credit="",
                 show_masks: bool = False, label_alpha: float = 1.0,
                 legend: str = ""):
    """How regions are drawn and whether they are named are separate choices.

    They used to be one `mode` string, which forced dropping names to also mean
    switching detector -- and switching detector resets every tracklet, so the
    colours all changed at the very moment the point was that nothing had
    changed except the names. `label_alpha` ramps instead, so the names can fade
    out mid-shot while one tracker runs straight through.
    """
    H, W = frame_bgr.shape[:2]
    show_labels = label_alpha > 0.02
    vis_now = book.visible()
    carrying = {e.sub for e in vis_now} | {e.obj for e in vis_now}
    if show_masks:
        frame_bgr = paint_masks(frame_bgr.copy(),
                                [t for t in tracker.tracks if t.tid in carrying])
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)
    f_lab, f_pred, f_small = fonts
    tracks = tracker.by_id()

    occupied: list = []
    drawn = carrying
    for t in tracker.tracks:
        if t.alpha <= 0.01 or t.tid not in drawn:
            continue                           # only regions that carry an edge
        c = PALETTE[t.tid % len(PALETTE)]
        x0, y0, x1, y1 = [int(v) for v in t.box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        a = int(235 * t.alpha)
        # Once a mask is drawn the box is redundant and fights it for attention;
        # keep it only where the mask is missing for this track.
        # A track that is coasting (no detection this frame) has had its mask
        # dropped, since a stale mask drifts off the object. Falling back to a
        # box would make a rectangle appear among the masks, which reads as a
        # rendering fault -- so a track that has ever carried a mask just fades
        # with its alpha. `mask_seen` keeps the fallback alive for a segmenter
        # that returns no masks at all, where the box is all there is.
        if not show_masks or (t.mask is None and not t.mask_seen):
            dr.rounded_rectangle([x0, y0, x1, y1], radius=6,
                                 outline=c + (a,), width=3)
        if show_labels:
            ly = y0 - f_lab.size - 12 if y0 - f_lab.size - 12 > 4 else y1 + 4
            _chip(dr, (x0, ly), t.label, f_lab, (255, 255, 255), c,
                  t.alpha * label_alpha, occupied, W, H)

    for i, e in enumerate(vis_now):
        ts, to = tracks.get(e.sub), tracks.get(e.obj)
        if ts is None or to is None:
            continue
        c = PALETTE[e.sub % len(PALETTE)]
        p0 = np.array([(ts.box[0] + ts.box[2]) / 2, (ts.box[1] + ts.box[3]) / 2])
        p1 = np.array([(to.box[0] + to.box[2]) / 2, (to.box[1] + to.box[3]) / 2])
        # Bow each edge off the straight line by a rank-dependent offset, so
        # the several relations that share one subject do not overprint.
        d = p1 - p0
        n = np.array([-d[1], d[0]])
        n = n / max(np.linalg.norm(n), 1e-6)
        bow = (18 + 22 * (i % 3)) * (1 if i % 2 == 0 else -1)
        ctrl = (p0 + p1) / 2 + n * bow
        pts = [tuple(((1 - s) ** 2 * p0 + 2 * (1 - s) * s * ctrl + s ** 2 * p1))
               for s in np.linspace(0, 1, 24)]
        a = int(225 * e.alpha)
        dr.line(pts, fill=c + (a,), width=3, joint="curve")
        # arrow head along the last segment of the curve
        v = np.array(pts[-1]) - np.array(pts[-3])
        v = v / max(np.linalg.norm(v), 1e-6)
        w_ = np.array([-v[1], v[0]])
        tip = np.array(pts[-1])
        dr.polygon([tuple(tip), tuple(tip - v * 16 + w_ * 7),
                    tuple(tip - v * 16 - w_ * 7)], fill=c + (a,))
        # Spread the labels along their curves by rank. Bowing the curves apart
        # separates the lines but not the labels: every edge out of one subject
        # still put its chip at its own midpoint, and those midpoints cluster.
        mx, my = pts[[6, 12, 17, 9, 14, 11, 16, 8][i % 8]]
        txt = f"{e.pred}  {max(e.shown, e.score):.2f}"
        w = int(dr.textlength(txt, font=f_pred))
        _chip(dr, (int(mx - w / 2), int(my - f_pred.size / 2 - 4)), txt,
              f_pred, (255, 255, 255), (22, 24, 28), e.alpha, occupied, W, H)

    if legend:
        # Bottom-left, opposite the credit. A chip rather than shadowed text:
        # this one names what the viewer is looking at and has to stay readable
        # over a mask as easily as over grass.
        lw = int(dr.textlength(legend, font=f_lab))
        pad, bh = 10, f_lab.size + 12
        dr.rounded_rectangle([16, H - bh - 16, 16 + lw + 2 * pad, H - 16],
                             radius=7, fill=(18, 18, 20, 215))
        dr.text((16 + pad, H - bh - 16 + 5), legend, font=f_lab,
                fill=(255, 255, 255, 245))
    if credit:
        w = int(dr.textlength(credit, font=f_small))
        cx, cy = W - w - 18, H - f_small.size - 14
        # Sand, snow and sky are all near-white, and 150-alpha white on them is
        # invisible. A shadow costs nothing and works on any footage.
        for ox, oy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            dr.text((cx + ox, cy + oy), credit, font=f_small, fill=(0, 0, 0, 130))
        dr.text((cx, cy), credit, font=f_small, fill=(255, 255, 255, 225))
    out = Image.alpha_composite(img, ov).convert("RGB")
    return cv2.cvtColor(np.array(out), cv2.COLOR_RGB2BGR)


# --------------------------------------------------------------------------- pipeline
def build_pipeline(args) -> ScenePipeline:
    pipe = ScenePipeline(
        args.dist, threads=args.threads, providers=args.providers,
        det_cfg=DetectorConfig(conf=args.det_conf, iou=args.det_iou,
                               max_det=args.max_boxes),
        thr_cfg=ThresholdConfig(threshold=0.0, topk=args.topk,
                                pair_weight=1.0,
                                max_per_pair=args.max_per_pair))
    # Both vocabularies are always scored. The page measures 19,103 strings at
    # under a millisecond, so restricting the bank buys nothing -- what needs
    # controlling is how many of each kind get DRAWN, and that is the EdgeBook's
    # job, per shot.
    want = INTERACTION + SPATIAL
    # Intersect with the bank rather than asserting against it: set_predicates
    # raises KeyError on the first unknown name, so one predicate that the
    # shipped bank happens not to carry would take the whole reel down.
    have = set(pipe.rel.available_predicates())
    keep = [p for p in want if p in have]
    if len(keep) != len(want):
        print(f"[render] not in this bank, dropped: {sorted(set(want) - have)}")
    try:
        pipe.rel.set_predicates(keep)
    except (RuntimeError, KeyError) as e:
        print(f"[render] vocabulary is fixed ({e}); using the baked predicates")
    return pipe


def spatial_map(pipe) -> dict:
    is_sp = getattr(pipe.rel, "is_spatial", None)
    if is_sp is None:
        return {p: p in SPATIAL for p in pipe.rel.predicates}
    return {p: bool(v) for p, v in zip(pipe.rel.predicates, np.asarray(is_sp).ravel())}


def scan(args) -> None:
    """Rank clips by how much confident interaction content the model finds."""
    pipe = build_pipeline(args)
    is_sp = spatial_map(pipe)
    clips = sorted(glob.glob(os.path.join(args.scan, "*.webm")) +
                   glob.glob(os.path.join(args.scan, "*.mp4")))
    rows = []
    for ci, path in enumerate(clips):
        name = os.path.basename(path)
        try:
            m = probe(path)
        except Exception as e:
            print(f"[{ci+1}/{len(clips)}] SKIP {name[:42]} ({e})")
            continue
        if m["duration"] <= 1.0:
            continue
        W, H = target_size(m["w"], m["h"], args.width)
        # Skip the first and last 8%: titles, fades and slates live there.
        times = np.linspace(m["duration"] * 0.08, m["duration"] * 0.92,
                            args.scan_frames)
        seen, per_frame, boxes_n = Counter(), [], []
        for t in times:
            fr = grab(path, float(t), W, H)
            if fr is None:
                continue
            r = pipe(fr)
            boxes_n.append(len(r.boxes))
            hits = [x for x in r.triplets
                    if x.score >= args.on_thr and not is_sp.get(x.predicate, False)]
            per_frame.append(len(hits))
            for x in hits:
                seen[f"{x.subject_label} {x.predicate} {x.object_label}"] += 1
        if not per_frame:
            print(f"[{ci+1}/{len(clips)}] SKIP {name[:42]} (no readable frames)")
            continue
        k = len(per_frame)
        # Persistence is the number that matters: a triplet the model finds in
        # most sampled frames will survive hysteresis, a one-frame flash won't.
        persist = sum(c for _, c in seen.most_common(4)) / max(4 * k, 1)
        density = float(np.mean(per_frame))
        crowd = float(np.mean(boxes_n))
        # A clip with 25 boxes makes an unreadable overlay however good it is.
        penalty = 1.0 if crowd <= args.ideal_boxes else args.ideal_boxes / crowd
        score = (0.65 * persist + 0.35 * min(density / 4.0, 1.0)) * penalty
        rows.append({"file": name, "score": round(score, 3),
                     "persist": round(persist, 2), "density": round(density, 1),
                     "boxes": round(crowd, 1), "duration": round(m["duration"], 1),
                     "top": [t for t, _ in seen.most_common(4)]})
        print(f"[{ci+1}/{len(clips)}] {score:.3f}  {name[:42]:42s}"
              f" persist {persist:.2f} density {density:4.1f} boxes {crowd:4.1f}",
              flush=True)
    rows.sort(key=lambda r: -r["score"])
    out = os.path.join(args.scan, "scan.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"\n=== ranked ({len(rows)} clips) -> {out} ===")
    for r in rows[:12]:
        print(f"  {r['score']:.3f}  {r['file'][:44]:44s}  {'; '.join(r['top'][:2])[:72]}")


def relate(pipe, frame_bgr, boxes, confs, labels):
    """Run the ONNX relation head over boxes from any detector.

    ScenePipeline.__call__ runs its own detector and then relates; when the
    boxes come from somewhere else — a segmentation model, a tracker, a human —
    only the second half is wanted. Same graph, same thresholds, same decode.
    """
    if len(boxes) < 2:
        return [], None
    n = min(len(boxes), pipe.rel.max_boxes)
    boxes, confs = np.asarray(boxes)[:n], np.asarray(confs)[:n]
    labels = list(labels)[:n]
    pred, pair, sub, obj, valid = pipe.rel(frame_bgr, boxes)
    trips = decode(pred, pair, sub, obj, valid, pipe.rel.predicates, pipe.thr_cfg,
                   boxes_xyxy=boxes, box_scores=confs, box_labels=labels)
    # Truncation keeps the prefix, so detection indices still agree with the
    # tracker's, which saw the untruncated array.
    kof = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(sub, obj))
           if valid[k]}
    vidx = {p_: i for i, p_ in enumerate(pipe.rel.predicates)}
    return trips, (pred, pair, kof, pipe.thr_cfg.contract(), vidx)


def render_one(pipe, is_sp, args, src, out, start, duration, credit,
               size=None, fps=None, pad=False, crop=None, mode="box",
               segdet=None, labels=True, nolabel_after=None,
               label_fade=0.6, on_thr=None, max_edges=None,
               max_spatial=None, legend="", legend_after=None,
               det_conf=None) -> tuple:
    """Render one segment. Returns (W, H, fps) so a reel can check they match."""
    m = probe(src)
    fps = float(fps or m["fps"])
    if crop:
        cw = (crop[2] - crop[0]) * m["w"]
        ch = (crop[3] - crop[1]) * m["h"]
    else:
        cw, ch = m["w"], m["h"]
    W, H = size if size else target_size(int(cw), int(ch), args.width,
                                        upscale=bool(crop))
    # The box-area test has to measure against the PICTURE, not the canvas. In
    # reel mode a segment is letterboxed onto a fixed 16:9 frame, so a box round
    # the whole picture covers only 0.75 of the canvas and slips under a 0.85
    # limit -- which is how a box labelled "pen" ended up framing a whole field.
    if pad:
        sc = min(W / cw, H / ch)
        content_area = (cw * sc) * (ch * sc)
    else:
        content_area = float(W * H)

    fonts = (_font(max(14, int(H * 0.022))), _font(max(13, int(H * 0.021))),
             _font(max(11, int(H * 0.016))))
    tracker = Tracker(iou_thr=args.iou_thr, merge_iou=args.merge_iou,
                      max_age=args.max_age, min_hits=args.min_hits,
                      high_conf=args.track_high, q=args.kf_q, r=args.kf_r,
                      stride=args.stride)
    on_thr = args.on_thr if on_thr is None else float(on_thr)
    mk_book = lambda belief: EdgeBook(score_ema=args.score_ema, on_thr=on_thr,
                    off_thr=min(args.off_thr, on_thr - 0.10),
                    on_frames=args.on_frames, off_frames=args.off_frames,
                    max_edges=args.max_edges if max_edges is None else int(max_edges),
                    max_spatial=(args.max_spatial if max_spatial is None
                                 else int(max_spatial)),
                    spatial_on_thr=args.spatial_on_thr, belief=belief,
                    kf_q=args.kf_rel_q, kf_r=args.kf_rel_r)
    # Same detections, same tracker, two books: any difference on screen is the
    # filter and nothing else.
    book = mk_book(not args.no_belief or args.compare_belief)
    book2 = mk_book(False) if args.compare_belief else None
    OUT_W = W * 2 if args.compare_belief else W

    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # H.264 + yuv420p + faststart, because the target is a <video> tag: VP9 in
    # webm and cv2's mp4v both have browsers where they are the wrong answer.
    cmd = [ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_W}x{H}",
           "-r", f"{fps:.4f}", "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "slow", "-crf", str(args.crf), "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    n_want = int(duration * fps) if duration else 10 ** 9
    i = 0
    for fr in read_frames(src, W, H, start, duration, pad=pad, fps=fps,
                          crop=crop, src=(m["w"], m["h"])):
        if i >= n_want:
            break
        if i % args.stride == 0:
            if segdet is None:
                r = pipe(fr)
                boxes, confs, labels, triplets = (r.boxes, r.scores, r.labels,
                                                  r.triplets)
                masks, raw = None, None
            else:
                # Segmentation front-end, SAME ONNX relation head: the detector
                # changes, the model under test does not.
                # A segmenter's confidences are not on the box detector's
                # scale, so the threshold is per-shot: one global value either
                # lets the segmenter's junk through or starves the box shots.
                boxes, confs, labels, masks = segdet(
                    fr, conf=args.det_conf if det_conf is None else det_conf,
                    iou=args.det_iou, max_det=args.max_boxes)
                triplets, raw = relate(pipe, fr, boxes, confs, labels)
            det2track = tracker.observe(boxes, labels, confs, masks=masks,
                                        frame_area=content_area,
                                        max_frac=args.max_box_frac)
            book.observe(triplets, det2track, is_sp, raw)
            if book2 is not None:
                book2.observe(triplets, det2track, is_sp, raw)
        tracker.step(W, H)
        book.step({t.tid for t in tracker.tracks})
        if book2 is not None:
            book2.step({t.tid for t in tracker.tracks})
        # Names fade on a ramp inside the shot; the tracker never restarts, so
        # every region keeps the colour it had while it still had a name.
        if not labels:
            la = 0.0
        elif nolabel_after is None:
            la = 1.0
        else:
            la = 1.0 - (i / fps - nolabel_after) / max(label_fade, 1e-3)
            la = float(min(1.0, max(0.0, la)))
        # The legend flips at the same instant the names start to go, so the
        # caption never describes a frame that has already changed.
        leg = (legend_after if (legend_after and nolabel_after is not None
                                and i / fps >= nolabel_after) else legend)
        left = render_frame(fr, tracker, book, fonts, credit, mode != "box", la,
                            leg)
        if book2 is None:
            proc.stdin.write(left.tobytes())
        else:
            right = render_frame(fr, tracker, book2, fonts, "", mode != "box", la, leg)
            _tag(left, "Kalman belief", fonts[2])
            _tag(right, "EMA + hysteresis (current)", fonts[2])
            proc.stdin.write(np.hstack([left, right]).tobytes())
        i += 1
        if i % 50 == 0:
            print(f"    {i} frames  ({len(tracker.tracks)} tracks, "
                  f"{len(book.visible())} edges)", flush=True)
    proc.stdin.close(); proc.wait()
    mb = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0
    print(f"  [seg] {i} frames -> {os.path.basename(out)} "
          f"({OUT_W}x{H} @ {fps:.2f}, {mb:.1f} MB)")
    return W, H, fps


def render(args) -> None:
    pipe = build_pipeline(args)
    render_one(pipe, spatial_map(pipe), args, args.render, args.out,
               args.start, args.duration, args.credit,
               crop=json.loads(args.crop) if args.crop else None,
               mode=args.mode, segdet=get_segdet(args, args.mode, args.segmenter),
               labels=not args.no_labels, nolabel_after=args.nolabel_after,
               label_fade=args.label_fade, legend=args.legend)


_SEGDETS: dict = {}


def get_segdet(args, mode: str, segmenter: str = "yoloe"):
    """One model per kind, loaded once and shared across every shot using it.

    Dropping names does NOT change the segmenter. FastSAM is the stronger claim
    on paper -- its whole vocabulary is {0: 'object'} -- but swapping models
    mid-reel restarts every tracklet, and the reel is making a point about the
    relations surviving, not about the tracker forgetting. Pass
    `"segmenter": "fastsam"` on a shot to use it anyway.
    """
    if mode == "box":
        return None
    kind = segmenter if segmenter in ("yoloe", "fastsam") else "yoloe"
    if kind not in _SEGDETS:
        w = args.fastsam if kind == "fastsam" else args.yoloe
        print(f"  [seg] loading {kind}: {w}")
        _SEGDETS[kind] = SegDetector(w, kind=kind, device=args.device)
    return _SEGDETS[kind]


def reel(args) -> None:
    """Render every segment in a spec onto one canvas and concatenate them.

    spec.json is a list of {file, start, duration, credit}. Keeping it a file
    rather than a pile of flags is what makes the reel reproducible: the shot
    list is the artifact, and re-rendering after a model change is one command.
    """
    with open(args.reel) as f:
        spec = json.load(f)
    root = os.path.dirname(os.path.abspath(args.reel))
    W = even(args.width)
    H = even(round(args.width * 9 / 16))
    tmp = os.path.join(os.path.dirname(os.path.abspath(args.out)) or ".", "_seg")
    os.makedirs(tmp, exist_ok=True)

    pipe = build_pipeline(args)
    is_sp = spatial_map(pipe)
    parts = []
    for i, sh in enumerate(spec):
        src = sh["file"] if os.path.isabs(sh["file"]) else os.path.join(root, sh["file"])
        if not os.path.exists(src):
            print(f"  [seg {i+1}] MISSING {src}"); continue
        out = os.path.join(tmp, f"seg{i:02d}.mp4")
        print(f"  [seg {i+1}/{len(spec)}] {os.path.basename(src)[:44]} "
              f"@{sh.get('start',0)}s +{sh.get('duration',0)}s")
        mode = sh.get("mode", "box")
        render_one(pipe, is_sp, args, src, out, float(sh.get("start", 0)),
                   float(sh.get("duration", 0)), sh.get("credit", ""),
                   size=(W, H), fps=args.fps, pad=True, crop=sh.get("crop"),
                   mode=mode,
                   segdet=get_segdet(args, mode, sh.get("segmenter", "yoloe")),
                   labels=sh.get("labels", True),
                   nolabel_after=sh.get("nolabel_after"),
                   label_fade=args.label_fade,
                   on_thr=sh.get("on_thr"), max_edges=sh.get("max_edges"),
                   max_spatial=sh.get("max_spatial"),
                   legend=sh.get("legend", ""),
                   legend_after=sh.get("legend_after"),
                   det_conf=sh.get("det_conf"))
        parts.append(out)

    if not parts:
        sys.exit("no segments rendered")
    lst = os.path.join(tmp, "concat.txt")
    with open(lst, "w") as f:
        for p_ in parts:
            f.write(f"file '{os.path.abspath(p_)}'\n")
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # Re-encode rather than -c copy: the segments were produced by separate
    # encoder runs, and stream copy across them leaves some players stuck on
    # the first segment's timestamps.
    subprocess.run([ff, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(args.crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", args.out],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    poster = os.path.splitext(args.out)[0] + "_poster.jpg"
    subprocess.run([ff, "-y", "-ss", str(args.poster_at), "-i", args.out,
                    "-frames:v", "1", "-q:v", "3", poster],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    mb = os.path.getsize(args.out) / 1e6
    print(f"\n[reel] {len(parts)} segments -> {args.out} ({W}x{H}, {mb:.1f} MB)")
    print(f"[reel] poster -> {poster}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "dist", "relsgg-vits16plus"))
    ap.add_argument("--scan", default="", help="directory of clips to rank")
    ap.add_argument("--render", default="", help="one clip to render")
    ap.add_argument("--reel", default="", help="shot-list JSON to render and concat")
    ap.add_argument("--fps", type=float, default=30.0, help="reel output fps")
    ap.add_argument("--poster_at", type=float, default=2.0)
    ap.add_argument("--out", default="reel.mp4")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0, help="0 = whole clip")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--credit", default="", help="attribution burned bottom-right")
    ap.add_argument("--crop", default="", help="normalised [x0,y0,x1,y1] of the source")
    ap.add_argument("--mode", default="box", choices=["box", "mask"],
                    help="how regions are drawn; naming is --no_labels")
    ap.add_argument("--segmenter", default="yoloe", choices=["yoloe", "fastsam"])
    ap.add_argument("--no_labels", action="store_true",
                    help="draw regions and relations but never name an object")
    ap.add_argument("--nolabel_after", type=float, default=None,
                    help="seconds into the shot at which names fade out, one "
                         "tracker running straight through")
    ap.add_argument("--legend", default="", help="caption burned bottom-left")
    ap.add_argument("--label_fade", type=float, default=0.6,
                    help="seconds the name fade takes")
    ap.add_argument("--yoloe", default="checkpoints/detectors/yoloe-11l-megasg497.pt",
                    help="segmentation weights for --mode mask")
    ap.add_argument("--fastsam", default="checkpoints/detectors/FastSAM-s.pt",
                    help="class-agnostic weights for --segmenter fastsam")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stride", type=int, default=2, help="infer every Nth frame")
    ap.add_argument("--scan_frames", type=int, default=10)
    ap.add_argument("--ideal_boxes", type=float, default=8.0)
    # model
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--providers", nargs="*", default=None)
    ap.add_argument("--det_conf", type=float, default=0.20)
    ap.add_argument("--det_iou", type=float, default=0.5)
    ap.add_argument("--max_boxes", type=int, default=24)
    ap.add_argument("--topk", type=int, default=24)
    # smoothing
    ap.add_argument("--kf_q", type=float, default=1.0,
                    help="Kalman process noise: how much the object really moves")
    ap.add_argument("--kf_r", type=float, default=10.0,
                    help="Kalman measurement noise: how much the detector's box "
                         "wobbles. Raise for a calmer box, lower to track harder")
    ap.add_argument("--track_high", type=float, default=0.25,
                    help="confidence splitting the tracker's two matching stages")
    ap.add_argument("--score_ema", type=float, default=0.35)
    ap.add_argument("--iou_thr", type=float, default=0.3)
    ap.add_argument("--merge_iou", type=float, default=0.75)
    ap.add_argument("--max_age", type=int, default=8)
    ap.add_argument("--min_hits", type=int, default=2)
    ap.add_argument("--on_thr", type=float, default=0.45)
    ap.add_argument("--off_thr", type=float, default=0.38)
    ap.add_argument("--on_frames", type=int, default=2)
    ap.add_argument("--off_frames", type=int, default=6)
    ap.add_argument("--max_edges", type=int, default=8)
    ap.add_argument("--max_spatial", type=int, default=3,
                    help="how many of the drawn edges may be spatial "
                         "('near', 'behind'). 0 for interactions only")
    ap.add_argument("--spatial_on_thr", type=float, default=0.42,
                    help="spatial predicates are easy, so they are held to a "
                         "higher bar than the interactions")
    ap.add_argument("--no_belief", action="store_true",
                    help="fall back to the old EMA+hysteresis relation scoring. "
                         "The Kalman belief filter is the default: measured on "
                         "the bicycle clip it holds the same 7.91 edges on "
                         "screen with 39%% fewer blinks and runs 53%% longer")
    ap.add_argument("--compare_belief", action="store_true",
                    help="render both side by side off one inference pass")
    ap.add_argument("--kf_rel_q", type=float, default=0.05,
                    help="relation process noise: how fast a relation may "
                         "genuinely change. Small = sticky")
    ap.add_argument("--kf_rel_r", type=float, default=1.0,
                    help="relation measurement noise: how much one frame's "
                         "opinion is worth")
    ap.add_argument("--max_per_pair", type=int, default=2,
                    help="predicates one (subject, object) pair may emit. 2 "
                         "gets 'holding' AND 'playing' on the same guitar")
    ap.add_argument("--max_box_frac", type=float, default=0.85,
                    help="drop detections covering more than this fraction of "
                         "the frame; a relation whose object is the entire "
                         "picture tells the viewer nothing and buries the rest")
    args = ap.parse_args()

    if args.scan:
        scan(args)
    elif args.reel:
        reel(args)
    elif args.render:
        render(args)
    else:
        ap.error("need --scan DIR, --render CLIP or --reel SPEC.json")


if __name__ == "__main__":
    main()
