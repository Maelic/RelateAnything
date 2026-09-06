"""RelateAnything live demo — ONNX end to end.

    python deploy/demo_webcam.py                      # webcam
    python deploy/demo_webcam.py --image photo.jpg    # one image, no camera
    python deploy/demo_webcam.py --video clip.mp4
    python deploy/demo_webcam.py --bench              # latency breakdown, no camera

Needs only onnxruntime + opencv + numpy (see deploy/dist/requirements.txt).
Everything is loaded from a distribution directory (default `deploy/dist`)
containing detector.onnx, relateanything.onnx and predicate_bank.npz.

Live keys
    q / ESC   quit                          s   save frame
    + / -     more / fewer triplets     [ / ]   lower / raise score threshold
    p         cycle predicate preset (all / interaction / spatial)
    b         toggle detector-confidence weighting
    h         toggle the HUD
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deploy.postprocess import ThresholdConfig            # noqa: E402
from deploy.runtime import DetectorConfig, ScenePipeline  # noqa: E402

# Predicate presets you can cycle live. "interaction" is where the interesting
# output lives; "spatial" is the geometric frame; "all" is both.
INTERACTION = [
    "wearing", "riding", "playing", "sitting on", "sitting at", "holding",
    "sitting in", "looking at", "using", "watching", "standing on", "carrying",
    "talking to", "smiling at", "standing beside", "walking past",
    "posing with", "leaning against", "part of", "resting on", "covering",
    "inside", "contained in", "hanging from", "surrounding", "attached to",
]
SPATIAL = ["on", "on top of", "in front of", "behind", "beside",
           "to the left of", "to the right of", "above", "below"]
PRESETS = {"all": INTERACTION + SPATIAL, "interaction": INTERACTION,
           "spatial": SPATIAL}
PRESET_ORDER = ["all", "interaction", "spatial"]

_PALETTE = [(66, 133, 244), (219, 68, 55), (15, 157, 88), (244, 160, 0),
            (171, 71, 188), (0, 172, 193), (255, 112, 67), (124, 179, 66)]


def draw(frame, res, show_hud=True, thr=0.0, preset="all"):
    out = frame.copy()
    for i, (b, lab) in enumerate(zip(res.boxes, res.labels)):
        c = _PALETTE[i % len(_PALETTE)]
        x0, y0, x1, y1 = b.astype(int)
        cv2.rectangle(out, (x0, y0), (x1, y1), c, 2)
        # keep the label clear of the HUD bar (top 46px): drop it below the box
        ly = y0 - 5 if y0 - 5 >= 58 else min(y1 + 14, out.shape[0] - 4)
        cv2.putText(out, f"{i}:{lab}", (x0, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    # Labels are staggered along their arrow (t in [0.25, 0.75] by rank) because
    # triplets routinely share endpoints — placing every label at the midpoint
    # stacks them into an unreadable pile.
    streams = ([("spatial", (60, 160, 255), res.graphs["spatial"]),
                ("semantic", (240, 240, 240), res.graphs["semantic"])]
               if getattr(res, "graphs", None) else
               [("all", (240, 240, 240), res.triplets)])
    trips = [(c, t) for _, c, ts in streams for t in ts]
    n = max(len(trips), 1)
    for i, (edge_color, t) in enumerate(trips):
        sb, ob = t.subject_box, t.object_box
        p0 = (int((sb[0] + sb[2]) / 2), int((sb[1] + sb[3]) / 2))
        p1 = (int((ob[0] + ob[2]) / 2), int((ob[1] + ob[3]) / 2))
        cv2.arrowedLine(out, p0, p1, edge_color, 2, cv2.LINE_AA, tipLength=0.03)
        f = 0.25 + 0.5 * (i / n)
        lx = int(p0[0] + (p1[0] - p0[0]) * f)
        ly = int(p0[1] + (p1[1] - p0[1]) * f)
        txt = f"{t.predicate} {t.score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        lx = max(2, min(lx, out.shape[1] - tw - 4))
        ly = max(th + 4, min(ly, out.shape[0] - 4))
        cv2.rectangle(out, (lx - 2, ly - th - 3), (lx + tw + 2, ly + 3),
                      (30, 30, 30), -1)
        cv2.putText(out, txt, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if not show_hud:
        return out
    fps = 1000.0 / max(res.total_ms, 1e-3)
    lines = [
        f"det {res.det_ms:5.1f}  rel {res.rel_ms:5.1f}  dec {res.dec_ms:4.2f} ms"
        f"   {fps:4.1f} FPS",
        f"boxes {len(res.boxes):2d}   triplets {len(res.triplets):2d}"
        f"   thr {thr:.2f}   [{preset}]",
    ]
    # solid bar behind the HUD so it never fights with box labels
    cv2.rectangle(out, (0, 0), (out.shape[1], 46), (24, 24, 24), -1)
    for i, ln in enumerate(lines):
        cv2.putText(out, ln, (8, 18 + i * 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 128), 1, cv2.LINE_AA)
    return out


def bench(pipe, args) -> None:
    """Latency breakdown. Synthetic frame unless --image is given."""
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    if args.image and os.path.exists(args.image):
        frame = cv2.imread(args.image)
    print("[bench] warmup x2...")
    for _ in range(2):
        r = pipe(frame)
    det, rel, dec = [], [], []
    for _ in range(args.bench_iters):
        r = pipe(frame)
        det.append(r.det_ms); rel.append(r.rel_ms); dec.append(r.dec_ms)
    d, e, c = np.median(det), np.median(rel), np.median(dec)
    tot = d + e + c
    print(f"[bench] n={args.bench_iters}  threads={args.threads or 'auto'}  "
          f"boxes={len(r.boxes)}  predicates={len(pipe.rel.predicates)}")
    print(f"  detector    {d:7.1f} ms  ({d/tot*100:4.1f}%)")
    print(f"  relation    {e:7.1f} ms  ({e/tot*100:4.1f}%)")
    print(f"  decode      {c:7.2f} ms  ({c/tot*100:4.1f}%)")
    print(f"  TOTAL       {tot:7.1f} ms  -> {1000/tot:.2f} FPS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "dist"), help="directory with the ONNX artifacts")
    ap.add_argument("--backend", default="onnx", choices=["onnx", "torch"],
                    help="onnx (default, torch-free) or torch (needs torch+ultralytics)")
    ap.add_argument("--checkpoint", default="",
                    help="backend=torch: a released model.pth")
    ap.add_argument("--det_weights", default="",
                    help="backend=torch: ultralytics detector.pt")
    ap.add_argument("--det_arch", default="yolo-world", choices=["yolo-world", "yoloe"])
    ap.add_argument("--device", default="cpu", help="backend=torch: cpu or cuda")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--image", default="")
    ap.add_argument("--video", default="")
    ap.add_argument("--threads", type=int, default=0, help="0 = onnxruntime default")
    ap.add_argument("--providers", nargs="*", default=None,
                    help="e.g. CUDAExecutionProvider CPUExecutionProvider")
    ap.add_argument("--det_conf", type=float, default=0.25)
    ap.add_argument("--det_iou", type=float, default=0.5)
    ap.add_argument("--max_boxes", type=int, default=32,
                    help="boxes handed to the relation head. At 16, 16.5%% of "
                         "the annotated relations of PSG test have an endpoint "
                         "that never reaches the sampler; 32 keeps 98.3%% of "
                         "them for about 9%% more latency.")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=None,
                    help="score floor. On a calibrated head it reads as a "
                         "target precision: 0.5 shows relations the model "
                         "would be right about half the time, and it means the "
                         "same on any calibrated model. Uncalibrated scores "
                         "crowd into [0.9, 1.0), where a threshold means "
                         "little. Default: 0.5 calibrated, 0.3 otherwise.")
    ap.add_argument("--pair_weight", type=float, default=1.0)
    ap.add_argument("--preset", default="all", choices=PRESET_ORDER)
    ap.add_argument("--predicates", nargs="*", default=None,
                    help="explicit predicate list (overrides --preset)")
    ap.add_argument("--decompose", action="store_true",
                    help="render TWO graphs from the same pass: spatial "
                         "(orange edges) + semantic (white). Needs a bank "
                         "with the is_spatial vector (release banks have it); "
                         "toggle live with the g key.")
    ap.add_argument("--save", default="")
    ap.add_argument("--no_window", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bench_iters", type=int, default=10)
    args = ap.parse_args()

    print(f"[demo] backend={args.backend}"
          + (f"  dist={args.dist}" if args.backend == "onnx"
             else f"  deploy={args.checkpoint}  device={args.device}"))
    pipe = ScenePipeline(
        args.dist, threads=args.threads, providers=args.providers,
        backend=args.backend, device=args.device, det_arch=args.det_arch,
        relation=args.checkpoint, detector=args.det_weights,
        det_cfg=DetectorConfig(conf=args.det_conf, iou=args.det_iou,
                               max_det=args.max_boxes),
        thr_cfg=ThresholdConfig(threshold=0.0, topk=args.topk,
                                pair_weight=args.pair_weight))
    # Resolve the default only AFTER the pipeline has told us whether the
    # artifact carries a calibration — the two regimes are ~15x apart in score
    # and one default cannot serve both.
    if args.threshold is None:
        cal = (pipe.thr_cfg.calib_a, pipe.thr_cfg.calib_b) != (1.0, 0.0)
        # A calibrated threshold is a precision target, so 0.5 means the same
        # thing under ANY correct calibration. Do NOT hardcode a number read
        # off one fit's score histogram: the shipped Haystack-adjudicated fit
        # puts emitted scores near 0.47 while the PSG-val fit puts them below
        # 0.1, and a constant tuned for one blanks the screen on the other.
        args.threshold = 0.5 if cal else 0.30
        print(f"[demo] {'calibrated' if cal else 'UNCALIBRATED'} head "
              f"-> threshold {args.threshold}"
              + ("  (= target precision)" if cal else "  (score is not a probability)"))
    pipe.thr_cfg.threshold = args.threshold
    preset = args.preset
    want = args.predicates if args.predicates else PRESETS[preset]
    try:
        pipe.rel.set_predicates(want)
    except RuntimeError as e:      # baked vocabulary, no bank — keep what's baked
        print(f"[demo] vocabulary is fixed ({e}); using the baked predicates")
    print(f"[demo] detector: {len(pipe.det.classes)} classes @ {pipe.det.imgsz}px")
    print(f"[demo] predicates: {len(pipe.rel.predicates)} active "
          f"({len(pipe.rel.available_predicates())} available)")
    if args.backend == "onnx":
        print(f"[demo] providers: {pipe.rel.sess.get_providers()}")

    if args.bench:
        bench(pipe, args)
        return

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"could not read {args.image}")
        res = pipe(frame, decompose=args.decompose)
        if args.decompose and res.graphs is None:
            print("[demo] WARNING --decompose ignored: bank has no is_spatial "
                  "vector (rebuild with deploy/build_predicate_bank.py)")
        print(f"[demo] det {res.det_ms:.1f} ms  rel {res.rel_ms:.1f} ms  "
              f"dec {res.dec_ms:.2f} ms  ({1000/max(res.total_ms,1e-3):.2f} FPS)")
        print(f"[demo] {len(res.boxes)} boxes, {len(res.triplets)} triplets:")
        if res.graphs is not None:
            for tag in ("spatial", "semantic"):
                print(f"  [{tag}]")
                for t in res.graphs[tag]:
                    print("   ", t)
        else:
            for t in res.triplets:
                print("   ", t)
        vis = draw(frame, res, thr=pipe.thr_cfg.threshold, preset=preset)
        out = args.save or (os.path.splitext(args.image)[0] + "_sg.jpg")
        cv2.imwrite(out, vis)
        print(f"[demo] wrote {out}")
        if not args.no_window:
            cv2.imshow("RelateAnything", vis); cv2.waitKey(0); cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        sys.exit("could not open camera/video")
    print("[demo] running — q/ESC quit, [ ] threshold, p preset, +/- triplets, "
          "g two-graph")
    show_hud = True
    decompose = args.decompose
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = pipe(frame, decompose=decompose)
        if args.no_window:
            continue
        cv2.imshow("RelateAnything",
                   draw(frame, res, show_hud, pipe.thr_cfg.threshold, preset))
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("s"):
            fn = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(fn, draw(frame, res, show_hud, pipe.thr_cfg.threshold, preset))
            print("saved", fn)
        elif key in (ord("+"), ord("=")):
            pipe.thr_cfg.topk += 2
        elif key == ord("-"):
            pipe.thr_cfg.topk = max(2, pipe.thr_cfg.topk - 2)
        elif key == ord("["):                    # the dynamic threshold, live
            pipe.thr_cfg.threshold = max(0.0, pipe.thr_cfg.threshold - 0.02)
        elif key == ord("]"):
            pipe.thr_cfg.threshold = min(1.0, pipe.thr_cfg.threshold + 0.02)
        elif key == ord("p"):                    # the dynamic vocabulary, live
            nxt = PRESET_ORDER[(PRESET_ORDER.index(preset) + 1) % len(PRESET_ORDER)]
            try:
                pipe.rel.set_predicates(PRESETS[nxt]); preset = nxt
            except RuntimeError:
                print("[demo] vocabulary is baked — no runtime swap available")
        elif key == ord("b"):
            pipe.thr_cfg.box_score_weight = not pipe.thr_cfg.box_score_weight
        elif key == ord("g"):                    # two-graph mode, live
            decompose = not decompose
            print(f"[demo] two-graph {'ON' if decompose else 'OFF'}")
        elif key == ord("h"):
            show_hud = not show_hud
    cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
