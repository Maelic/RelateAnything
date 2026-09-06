"""Parallel YOLOE + RelateAnything inference pipeline for batch-1 real time.

WHY THIS EXISTS (all numbers measured on the full-recipe checkpoint, A40, bf16):

  * batch-1 is CPU-DISPATCH bound, not compute bound: 1,362 kernel launches per
    image, ~30 ms of CPU dispatch against ~9 ms of GPU work. Proof: bs1 latency
    is identical at 448/392/336 px. So the classic moves (smaller backbone,
    lower resolution) buy nothing here — overlapping and shrinking the WORK
    GRAPH does.
  * `torch.compile`: "default" is 2.4x stream-stable (measured over 32 object
    counts); "reduce-overhead" (CUDA graphs) adds ~13 % on top with NO extra
    error beyond inductor's own — the earlier "silently wrong" reading was a
    tie-breaking measurement bug (see relsgg-inference-launch-bound). Either
    mode perturbs eval metrics ~40x the noise floor: demo/product only, never
    for reported numbers.

THE THREE OPTIMISATIONS IMPLEMENTED

 1. INTRA-FRAME OVERLAP ON A WORKER THREAD. The DINOv3 backbone does not
    depend on the boxes (`model.forward` uses `boxes` only from SoftSpatialPool
    onward), so the detector and the backbone are INDEPENDENT branches:

        frame ─┬─→ YOLOE-11m ────────→ boxes/masks ─┐
               └─→ DINOv3 backbone ──→ F_map ───────┴─→ SSP → … → triplets

    MEASURED, and the obvious implementation is the wrong one: a second CUDA
    STREAM alone buys nothing (22.6 ms sequential -> 22.7 ms), because both
    branches are CPU-dispatch bound and a stream does not add a CPU thread.
    Running the backbone on a worker THREAD, whose launches proceed while
    ultralytics holds the main thread, gives 22.6 -> 16.9 ms (-25%, i.e. 57%
    of the theoretical overlap). Joined via `precomputed_features`.

 2. RIGHT-SIZED STATIC SHAPES. With static shapes the head always pays for
    `max_objects` boxes and `final_budget` pairs no matter how many the
    detector returned — filtering to the top-k detections saves NOTHING unless
    the shapes shrink too. `PipelineConfig.max_objects/final_budget` shrink
    them to what a webcam scene actually contains. Candidate pairs go as N².

 3. CROSS-FRAME PIPELINING (optional, `pipelined=True`). A worker thread runs
    detector+backbone for frame N+1 while the main thread finishes the relation
    stage of frame N, so steady-state throughput becomes max(stage) instead of
    sum(stage). Costs one frame of latency. Works on CPU too, where there are
    no streams to overlap.

Both models stay re-parameterizable at runtime: `set_object_classes` (YOLOE
text prompts) and `set_predicates` (relation vocabulary via the checkpoint's
own text encoder).
"""

from __future__ import annotations

import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from contextlib import nullcontext as _nullctx
from queue import Empty, Queue
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class PipelineConfig:
    # detector
    det_weights: str = "checkpoints/detectors/yoloe-11m-seg.pt"
    det_conf: float = 0.25
    det_iou: float = 0.60
    det_imgsz: int = 640
    masks: bool = True
    # Applied at construction for text-prompt YOLOE, which otherwise has an
    # empty vocabulary and returns zero boxes. None = leave as-is (prompt-free).
    default_classes: Optional[Sequence[str]] = None
    # relation head — these are the STATIC shapes; shrinking them is the
    # single biggest compute reduction available for a webcam scene.
    ckpt: str = ("runs/train/sched_lr4e-4_ep8_r0_sig0.25_btd0.3_def4/"
                 "checkpoint_best.pth")
    img_size: int = 448
    # MEASURED, and the old values were the single largest recall loss in the
    # product.
    #
    # max_objects=16 DELETED 16.5% of PSG-test GT relations before the sampler
    # ever ran, and no eval could see it: data/relation_dataset.py drops
    # relations whose endpoints exceed the cap, so they leave `targets`
    # entirely and every recall metric silently loses its denominator. Box
    # coverage of GT pairs vs an untruncated run: 16 -> 0.835, 24 -> 0.951,
    # 32 -> 0.983, 40 -> 0.995.
    #
    # final_budget=64 then kept only 94.8% of what survived. At max_objects=32
    # there are 32*31 = 992 ordered pairs, so a budget of 992 is EXHAUSTIVE —
    # the sampler stops being a filter at all and its recall is exactly 1.0.
    # Affordable because bs1 is CPU-dispatch bound: a larger K grows kernel
    # SIZE, not kernel COUNT.
    #
    #   config                   latency         box    sampler   product
    #   old  mo=16 K=64          21.17 ms        0.835   0.948     0.792
    #   new  mo=32 K=992         23.08 ms 1.09x  0.983   1.000     0.983
    #
    # +9% latency (47.2 -> 43.3 FPS) for +24% relative on the structural
    # ceiling. Shrink these again only against that table, not by intuition.
    max_objects: int = 32      # trained at 40
    geo_budget: int = 992      # = max_objects*(max_objects-1): stage 1 is a
    final_budget: int = 992    # no-op, stage 2 is exhaustive
    # A release bundle's predicate_bank.npz (names + already-encoded W). When
    # set it overrides `predicates` and no text encoder is loaded.
    vocab_npz: str = ""
    # runtime
    device: str = "cuda"
    overlap: bool = True       # intra-frame detector ∥ backbone
    # The worker thread is what buys the overlap; the private CUDA stream is a
    # separate choice and was measured to buy nothing on its own. Kept as a knob
    # because a stream is not free either -- it adds cross-stream sync on the join.
    overlap_stream: bool = True
    pipelined: bool = False    # cross-frame worker thread
    amp: bool = True
    # torch.compile mode for the relation model ("" = eager).
    #
    # MEASURED (120 real frames, 32 distinct object counts):
    # p50 22.50 -> 9.43 ms, p99 23.02 -> 9.83 ms, 44 -> 106 FPS, and ZERO
    # stalls — the worst frame is 1.1x the median, same as eager, so dynamo
    # does not re-guard on a varying detection count. Costs ~66 s once at
    # startup, which `warmup()` pays up front instead of on frame 1.
    #
    # NOT FREE, and not for benchmarks: inductor changes reduction order, which
    # moves eval metrics by ~40x the measured noise floor — in
    # BOTH directions, largest on low-support tail buckets. Ship it for a live
    # demo; never use it to produce a reported number.
    compile: str = ""
    # Deployment calibration (a, b) for sigmoid(a * (pred + rel) + b), as
    # fitted by benchmark/eval_deploy_metrics.py --fit_platt on a val split.
    # None leaves the raw head, whose scores all sit in [0.9, 1.0) — a
    # score_thr against those is not a meaningful control (see the warning in
    # __init__ and RelateAnything.set_calibration).
    calibration: Optional[Tuple[float, float]] = None


@dataclass
class Timing:
    det: float = 0.0
    backbone: float = 0.0
    relation: float = 0.0
    total: float = 0.0
    fps: float = 0.0
    overlap_saved: float = 0.0


@dataclass
class SceneResult:
    boxes_xyxy: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))
    labels: List[str] = field(default_factory=list)
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    masks: Optional[np.ndarray] = None            # [N, H, W] bool, if requested
    triplets: List[tuple] = field(default_factory=list)  # (s_i, pred, o_i, score)
    # Two-graph decode (dual_spatial_head checkpoints): the SAME forward pass
    # ranked twice, once inside the spatial predicate columns and once inside
    # the semantic ones. Measured to beat a single merged graph of twice the
    # budget on 6/6 benchmark cells.
    triplets_spatial: List[tuple] = field(default_factory=list)
    triplets_semantic: List[tuple] = field(default_factory=list)
    timing: Timing = field(default_factory=Timing)
    frame: Optional[np.ndarray] = None


class ParallelScenePipeline:
    def __init__(self, cfg: PipelineConfig,
                 predicates: Optional[Sequence[str]] = None):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if (cfg.device != "cuda" or torch.cuda.is_available())
            else "cpu")
        self.is_cuda = self.device.type == "cuda"

        # YOLOE for the seg/prompt-free demo checkpoints, but the deployment
        # family also benchmarks YOLO-World and plain closed-set detectors, and
        # YOLOE() refuses those. ultralytics' generic YOLO() dispatches on the
        # checkpoint, so it is the right loader unless the file is a YOLOE one.
        if "yoloe" in os.path.basename(cfg.det_weights).lower():
            from ultralytics import YOLOE as _Det
        else:
            from ultralytics import YOLO as _Det
        self.det = _Det(cfg.det_weights)
        self.det.to(str(self.device))
        # A text-prompt YOLOE checkpoint ships with NO usable vocabulary (its
        # `names` are the placeholder "0".."79"), so it detects nothing until
        # set_classes is called. The "-pf" prompt-free checkpoint is the one
        # that works out of the box — use it when no classes are supplied.
        self.prompt_free = "-pf" in cfg.det_weights
        self._det_classes: List[str] = list(self.det.names.values()) \
            if isinstance(self.det.names, dict) else list(self.det.names)
        if not self.prompt_free and cfg.default_classes:
            self.set_object_classes(cfg.default_classes)

        from relsgg.api import RelateAnything
        # A release bundle's predicate_bank.npz carries the vocabulary ALREADY
        # encoded, so pointing at one skips the text encoder entirely -- which
        # is what a deployment host does, and the only route when the text
        # student the checkpoint names is not on the machine.
        bank_E = None
        if cfg.vocab_npz:
            import numpy as _np
            z = _np.load(cfg.vocab_npz, allow_pickle=True)
            predicates = [str(q) for q in z["names"]]
            bank_E = z["W"]
        preds = list(predicates) if predicates else _default_predicates()
        self.ra = RelateAnything.from_checkpoint(
            cfg.ckpt, preds, device=str(self.device), weights="ema",
            embeddings=bank_E)
        # Prefer an explicit config, then the checkpoint's own calibration.json,
        # then identity-with-a-warning. Same resolution order as the ONNX host.
        from relsgg.scoring import ScoreContract
        if cfg.calibration:
            self.ra.set_calibration(*cfg.calibration)
        else:
            self.ra.contract = ScoreContract.for_checkpoint(cfg.ckpt)
        self._contract = self.ra.contract
        if not self._contract.is_calibrated:
            warnings.warn(
                "no calibration installed: the raw head puts ~97% of scores "
                "in [0.9, 1.0), so score_thr is close to a no-op (on Haystack, "
                "0.05 -> 0.90 changed precision 0.0020 -> 0.0021 and dropped "
                "2.4% of predictions). Fit one with eval_deploy_metrics.py "
                "--fit_platt and pass PipelineConfig(calibration=(a, b)).",
                RuntimeWarning, stacklevel=2)
        self.model = self.ra.model.eval()
        self._apply_static_shapes()
        # Compile AFTER the static shapes are set, so dynamo guards on the
        # budgets this pipeline actually runs. `backbone.extract` is called
        # directly by the worker thread and so is compiled separately.
        self._compiled = None
        if cfg.compile:
            self._compiled = torch.compile(self.model, mode=cfg.compile,
                                           dynamic=False)

        # Dedicated stream for the box-independent backbone branch.
        self.s_bb = torch.cuda.Stream(device=self.device) if (
            self.is_cuda and cfg.overlap and cfg.overlap_stream) else None

        self._is_spatial: Optional[np.ndarray] = None
        self._q_in: "Queue" = Queue(maxsize=1)
        self._q_out: "Queue" = Queue(maxsize=1)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()

    # ---------------------------------------------------------------- config
    def _apply_static_shapes(self) -> None:
        c = self.model.config
        c.geo_budget = self.cfg.geo_budget
        c.final_budget = self.cfg.final_budget
        if hasattr(self.model, "sampler"):
            for attr, val in (("geo_budget", self.cfg.geo_budget),
                              ("final_budget", self.cfg.final_budget)):
                if hasattr(self.model.sampler, attr):
                    setattr(self.model.sampler, attr, val)

    @torch.no_grad()
    def warmup(self, n_boxes: int = 4) -> float:
        """Run one synthetic frame through the relation path; returns seconds.

        With `compile` on, the first call pays ~66 s of inductor compilation.
        Paying it here keeps it out of the frame loop, where it would look like
        the demo had hung. Cheap and harmless when compile is off (it still
        warms the allocator and cuDNN autotuning).

        This deliberately calls the relation model directly rather than going
        through __call__: on a synthetic frame the detector finds <2 objects,
        __call__ returns early, and the path we need to compile never runs.
        """
        t0 = time.perf_counter()
        x = self._prep_image(np.zeros((480, 640, 3), dtype=np.uint8))
        bt = torch.zeros(1, self.cfg.max_objects, 4, device=self.device)
        bt[0,:n_boxes, 0] = torch.linspace(0.2, 0.8, n_boxes, device=self.device)
        bt[0,:n_boxes, 1] = 0.5
        bt[0,:n_boxes, 2:] = 0.2
        cnt = torch.tensor([n_boxes], device=self.device)
        m = self._compiled if self._compiled is not None else self.model
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.cfg.amp and self.is_cuda):
            # Match the configured branch: overlap feeds precomputed features,
            # and the two differ as separate dynamo graphs.
            F = (self.model.backbone.extract(self.model.backbone.preprocess(x))
                 if self.cfg.overlap else None)
            m(x, bt, box_counts=cnt, targets=None, precomputed_features=F)
        if self.is_cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    def set_object_classes(self, names: Sequence[str]) -> "ParallelScenePipeline":
        """Re-parameterize YOLOE to an arbitrary open-vocabulary class list."""
        names = [n.strip() for n in names if n.strip()]
        if not names:
            raise ValueError("empty class list")
        self.det.set_classes(names, self.det.get_text_pe(names))
        self.det.to(str(self.device))
        self._det_classes = list(names)
        return self

    def set_predicates(self, names: Sequence[str]) -> "ParallelScenePipeline":
        """Re-parameterize the relation head to an arbitrary predicate list.
        Uses the checkpoint's own text encoder; inference stays pure-vision."""
        names = [n.strip() for n in names if n.strip()]
        if not names:
            raise ValueError("empty predicate list")
        self.ra.set_vocabulary(names)
        self._is_spatial = None          # split follows the vocabulary
        return self

    @property
    def has_dual_head(self) -> bool:
        """True when the checkpoint was trained with --dual_spatial_head, i.e.
        the spatial columns are served by their own query (`spa_proj`). The
        two-graph split still *works* without it, but it is then only a
        re-ranking of one head's scores, not two specialised experts."""
        return bool(getattr(self.model.config, "dual_spatial_head", False))

    @property
    def is_spatial(self) -> np.ndarray:
        """[V] bool over the CURRENT predicate vocabulary. Hybrid corpus-map /
        gate-alpha rule — see RelateAnything._type_vector."""
        if self._is_spatial is None:
            self._is_spatial = np.asarray(self.ra._type_vector(), bool)
        return self._is_spatial

    @property
    def object_classes(self) -> List[str]:
        return list(self._det_classes)

    @property
    def predicates(self) -> List[str]:
        return list(self.ra.predicates)

    # ------------------------------------------------------------- inference
    def _detect(self, frame_bgr: np.ndarray):
        c = self.cfg
        r = self.det.predict(frame_bgr, conf=c.det_conf, iou=c.det_iou,
                             imgsz=c.det_imgsz, max_det=c.max_objects,
                             device=str(self.device), retina_masks=c.masks,
                             verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [], None)
        xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
        conf = r.boxes.conf.cpu().numpy().astype(np.float32)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        names = [self.det.names[int(k)] if isinstance(self.det.names, dict)
                 else self._det_classes[int(k)] for k in cls]
        masks = None
        if c.masks and getattr(r, "masks", None) is not None:
            masks = r.masks.data.cpu().numpy().astype(bool)
        return xyxy, conf, names, masks

    def _prep_image(self, frame_bgr: np.ndarray) -> torch.Tensor:
        import cv2
        s = self.cfg.img_size
        img = cv2.resize(frame_bgr, (s, s), interpolation=cv2.INTER_LINEAR)
        x = img[:,:,::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return torch.from_numpy(np.ascontiguousarray(x)).to(self.device)

    @torch.no_grad()
    def __call__(self, frame_bgr: np.ndarray, top_k: int = 12,
                 score_thr: float = 0.30, decompose: bool = False,
                 spatial_drop_pair: bool = True) -> SceneResult:
        t0 = time.perf_counter()
        x = self._prep_image(frame_bgr)

        # --- branch A: backbone on a WORKER THREAD -------------------------
        # A CUDA stream alone is not enough: both branches are CPU-DISPATCH
        # bound, and a second stream does not give a second CPU thread —
        # measured 22.7 ms vs 22.6 ms sequential, i.e. nothing. A worker thread
        # (whose kernel launches proceed while ultralytics holds the main
        # thread) recovers 57% of the theoretical overlap: 22.6 -> 16.9 ms.
        F_map = None
        t_bb0 = time.perf_counter()
        holder: dict = {}

        def _backbone_worker():
            try:
                ctx = (torch.cuda.stream(self.s_bb) if self.s_bb is not None
                       else _nullctx())
                with ctx:
                    with torch.amp.autocast(self.device.type,
                                            dtype=torch.bfloat16,
                                            enabled=self.cfg.amp and self.is_cuda):
                        holder["F"] = self.model.backbone.extract(
                            self.model.backbone.preprocess(x))
                if self.s_bb is not None:
                    self.s_bb.synchronize()
            except Exception as e:            # never kill the frame loop
                holder["err"] = e

        th = None
        if self.cfg.overlap:
            th = threading.Thread(target=_backbone_worker, daemon=True)
            th.start()

        # --- branch B: detector, runs on the main thread concurrently ------
        t_d0 = time.perf_counter()
        xyxy, conf, names, masks = self._detect(frame_bgr)
        t_det = time.perf_counter() - t_d0

        if th is not None:
            th.join()
            if "err" in holder:
                raise holder["err"]
            F_map = holder.get("F")
        t_bb = time.perf_counter() - t_bb0

        res = SceneResult(boxes_xyxy=xyxy, labels=names, scores=conf,
                          masks=masks, frame=frame_bgr)
        if len(xyxy) < 2:
            res.timing = Timing(det=t_det * 1e3, backbone=t_bb * 1e3,
                                total=(time.perf_counter() - t0) * 1e3)
            res.timing.fps = 1e3 / max(res.timing.total, 1e-6)
            return res

        # --- relation stage ------------------------------------------------
        t_r0 = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        n = min(len(xyxy), self.cfg.max_objects)
        b = xyxy[:n].astype(np.float32).copy()
        b[:, [0, 2]] /= max(W, 1)
        b[:, [1, 3]] /= max(H, 1)
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        boxes = np.stack([cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], -1)
        # Pad to max_objects so the box tensor's shape is the same on every
        # frame; box_counts drives the validity mask, so padding changes no
        # output, and a fixed shape is what torch.compile and CUDA graphs need.
        padded = np.zeros((self.cfg.max_objects, 4), dtype=np.float32)
        padded[:n] = boxes.astype(np.float32)
        bt = torch.from_numpy(padded[None]).to(self.device)
        cnt = torch.tensor([n], device=self.device)

        _relmodel = self._compiled if self._compiled is not None else self.model
        with torch.amp.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.cfg.amp and self.is_cuda):
            out = _relmodel(x, bt, box_counts=cnt, targets=None,
                            precomputed_features=F_map)
        logits = out["logits"][0].float()
        pair = out.get("pair_logits")
        # One score contract for evaluation and deployment (relsgg/scoring.py).
        # The head's affine is trained against balanced positives and negatives
        # while a frame carries few true pairs, so raw scores crowd into
        # [0.9, 1.0) and a threshold means little there. The
        # calibration is the missing fit; it is monotone, so ranking is
        # bit-identical.
        score = self._contract.scores(
            logits, None if pair is None else pair[0].float())
        valid = out["valid_mask"][0]
        sub = out["sub_idx"][0]
        obj = out["obj_idx"][0]

        s_flat = score[valid]
        preds = self.ra.predicates
        if s_flat.numel():
            best, arg = s_flat.max(-1)
            order = best.argsort(descending=True)[:top_k]
            vi = valid.nonzero(as_tuple=True)[0]
            for j in order.tolist():
                sc = float(best[j])
                if sc < score_thr:
                    break
                k = int(vi[j])
                res.triplets.append((int(sub[k]), preds[int(arg[j])],
                                     int(obj[k]), sc))

        # Two-graph decode. Free: it re-ranks the scores we already computed,
        # no second forward pass. Only meaningful on a dual_spatial_head
        # checkpoint, where the spatial columns are served by their own query.
        if decompose and self.has_dual_head and s_flat.numel():
            from relsgg.decompose import split_ranked
            # `spatial_drop_pair` drops the relatedness (pair) prior from the
            # spatial stream. Relatedness is an annotation-propensity ~ contact
            # signal: dropping it HELPS spatial truth-judgment (+0.068 macro
            # AUC on SpatialSense, projective predicates +0.11..0.14 —
            #) but HURTS recall, and on
            # raw detector output it is also what suppresses duplicate boxes.
            # Measured on a live frame: with it kept, `person -on-> motorcycle
            # 0.57`; dropped, scores saturate at 1.00 and junk pairs surface
            # (`motorcycle -on-> motorcycle`). Hence default True here (the
            # measured optimum for judging truth) and False in the demo UI,
            # where readability wins. The two defaults disagree on purpose.
            sc_sem = score[valid].detach().cpu().numpy()
            sc_spa = (torch.sigmoid(logits)[valid].detach().cpu().numpy()
                      if (spatial_drop_pair and pair is not None) else sc_sem)
            sub_v = sub[valid].detach().cpu().numpy()
            obj_v = obj[valid].detach().cpu().numpy()
            ones = np.ones(len(sub_v), bool)
            is_sp = self.is_spatial
            spa = split_ranked(sc_spa, sub_v, obj_v, ones, is_sp,
                               topk=top_k)["spatial"]
            sem = split_ranked(sc_sem, sub_v, obj_v, ones, is_sp,
                               topk=top_k)["semantic"]
            res.triplets_spatial = [(s, preds[p], o, v) for s, o, p, v in spa
                                    if v >= score_thr]
            res.triplets_semantic = [(s, preds[p], o, v) for s, o, p, v in sem
                                     if v >= score_thr]
        t_rel = time.perf_counter() - t_r0

        total = (time.perf_counter() - t0) * 1e3
        res.timing = Timing(det=t_det * 1e3, backbone=t_bb * 1e3,
                            relation=t_rel * 1e3, total=total,
                            fps=1e3 / max(total, 1e-6),
                            overlap_saved=max(0.0, (t_det + t_bb) * 1e3
                                               - t_bb * 1e3 - t_det * 1e3))
        return res

    # ----------------------------------------------------- cross-frame mode
    def start(self) -> None:
        if self._worker or not self.cfg.pipelined:
            return
        self._stop.clear()

        def loop():
            while not self._stop.is_set():
                try:
                    item = self._q_in.get(timeout=0.1)
                except Empty:
                    continue
                if item is None:
                    break
                frame, kw = item
                r = self(frame, **kw)
                if self._q_out.full():
                    try:
                        self._q_out.get_nowait()
                    except Empty:
                        pass
                self._q_out.put(r)

        self._worker = threading.Thread(target=loop, daemon=True)
        self._worker.start()

    def submit(self, frame_bgr: np.ndarray, **kw) -> Optional[SceneResult]:
        """Non-blocking: push a frame, return the most recent finished result."""
        if not self._q_in.full():
            self._q_in.put((frame_bgr, kw))
        try:
            return self._q_out.get_nowait()
        except Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._worker:
            self._q_in.put(None)
            self._worker.join(timeout=2.0)
            self._worker = None


def _default_predicates() -> List[str]:
    return [
        "on", "in", "next to", "above", "under", "behind", "in front of",
        "holding", "wearing", "riding", "sitting on", "standing on",
        "looking at", "carrying", "using", "eating", "attached to",
        "hanging from", "leaning against", "part of", "near", "on top of",
    ]
