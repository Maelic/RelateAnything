"""RelateAnything live demo — YOLOE-11m (masks) + open-vocabulary relations.

    python deploy/gradio_app.py                    # auto GPU, opens in browser
    python deploy/gradio_app.py --device cpu       # no GPU needed
    python deploy/gradio_app.py --share            # public link (SME demo)
    python deploy/gradio_app.py --port 7860

Everything runs server-side; the browser only ships webcam frames, so the demo
works on a laptop with the model on a remote GPU. On CPU it still runs, just
slower — use the Image tab rather than the live webcam there.

BOTH models are re-parameterizable live, which is the point of the demo:
  * object classes  -> YOLOE text prompts (`set_classes` + `get_text_pe`)
  * predicates      -> relation head vocabulary, encoded by the checkpoint's
                       own text encoder; inference stays pure-vision afterwards
Type any words into either box, hit Apply, and the pipeline is re-targeted
without touching the weights.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr                                        # noqa: E402
from deploy.pipeline import (ParallelScenePipeline,        # noqa: E402
                             PipelineConfig, _default_predicates)

DEFAULT_CLASSES = [
    "person", "face", "hand", "laptop", "keyboard", "mouse", "monitor", "cup",
    "bottle", "phone", "book", "chair", "desk", "backpack", "hat", "glasses",
    "plant", "door", "window", "picture", "lamp", "bag",
]

# Distinct hues per object instance, fixed by index so a box keeps its colour
# across frames (colour follows the entity, never its rank).
PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (138, 99, 210),
           (214, 168, 42), (52, 187, 235), (200, 62, 120), (120, 160, 60)]

PIPE: ParallelScenePipeline | None = None


def _color(i: int):
    return PALETTE[i % len(PALETTE)]


# Two-graph edge colours: layout (spatial) vs content (semantic). Fixed by
# meaning, not by rank — a predicate keeps its colour across frames.
SPATIAL_EDGE = (120, 200, 255)    # BGR: warm-cyan  → "where things are"
SEMANTIC_EDGE = (140, 255, 170)   # BGR: green      → "what things do"
MERGED_EDGE = (255, 255, 255)


def _edges_for(res, mode: str):
    """(list_of_triplets, colour) pairs for the requested graph mode."""
    if mode == "merged":
        return [(res.triplets, MERGED_EDGE)]
    if mode == "spatial":
        return [(res.triplets_spatial, SPATIAL_EDGE)]
    if mode == "semantic":
        return [(res.triplets_semantic, SEMANTIC_EDGE)]
    return [(res.triplets_spatial, SPATIAL_EDGE),
            (res.triplets_semantic, SEMANTIC_EDGE)]     # both


def render(res, show_masks: bool, show_labels: bool,
           mode: str = "merged") -> np.ndarray:
    """Draw masks, boxes and relation arrows onto the frame (BGR in, RGB out)."""
    img = res.frame.copy()
    H, W = img.shape[:2]

    if show_masks and res.masks is not None and len(res.masks):
        overlay = img.copy()
        for i, m in enumerate(res.masks[:len(res.boxes_xyxy)]):
            if m.shape[:2] != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[m] = _color(i)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    centers = []
    for i, bb in enumerate(res.boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bb]
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), _color(i), 2)
        if show_labels:
            lab = res.labels[i] if i < len(res.labels) else f"obj{i}"
            sc = float(res.scores[i]) if i < len(res.scores) else 0.0
            txt = f"{i}:{lab} {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                          _color(i), -1)
            cv2.putText(img, txt, (x1 + 2, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # relation arrows, thickness ~ score; colour ~ graph (layout vs content).
    # In "both" mode the two streams are drawn with a small perpendicular
    # offset so a pair carrying one edge of each type stays readable.
    streams = _edges_for(res, mode)
    for si, (trips, edge_col) in enumerate(streams):
        off = 0 if len(streams) == 1 else (6 if si == 0 else -6)
        for s, pred, o, sc in trips:
            if s >= len(centers) or o >= len(centers):
                continue
            p1, p2 = centers[s], centers[o]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            n = max(1.0, (dx * dx + dy * dy) ** 0.5)
            ox, oy = int(-dy / n * off), int(dx / n * off)
            q1, q2 = (p1[0] + ox, p1[1] + oy), (p2[0] + ox, p2[1] + oy)
            cv2.arrowedLine(img, q1, q2, edge_col, max(1, int(1 + 3 * sc)),
                            cv2.LINE_AA, tipLength=0.03)
            mid = ((q1[0] + q2[0]) // 2, (q1[1] + q2[1]) // 2)
            (tw, th), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (mid[0] - 2, mid[1] - th - 4),
                          (mid[0] + tw + 4, mid[1] + 3), (30, 30, 30), -1)
            cv2.putText(img, pred, (mid[0] + 1, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, edge_col, 1, cv2.LINE_AA)

    t = res.timing
    hud = (f"{t.fps:5.1f} FPS | det {t.det:5.1f} | backbone {t.backbone:5.1f} "
           f"| rel {t.relation:5.1f} | total {t.total:5.1f} ms")
    cv2.rectangle(img, (0, 0), (W, 24), (20, 20, 20), -1)
    cv2.putText(img, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 230, 160), 1, cv2.LINE_AA)
    return img[:,:,::-1]


def infer(frame_rgb, conf, top_k, score_thr, show_masks, show_labels, mode,
          spatial_raw):
    if frame_rgb is None or PIPE is None:
        return None, "", ""
    PIPE.cfg.det_conf = float(conf)
    res = PIPE(frame_rgb[:,:,::-1].copy(), top_k=int(top_k),
               score_thr=float(score_thr), decompose=(mode != "merged"),
               spatial_drop_pair=bool(spatial_raw))

    def _rows(trips):
        return "\n".join(
            f"| {res.labels[s] if s < len(res.labels) else s} | **{p}** | "
            f"{res.labels[o] if o < len(res.labels) else o} | {sc:.2f} |"
            for s, p, o, sc in trips) or "| – | – | – | – |"

    hdr = "| subject | predicate | object | score |\n|---|---|---|---|\n"
    if mode == "merged":
        table = hdr + _rows(res.triplets)
    elif mode == "spatial":
        table = "**layout graph (spatial)**\n\n" + hdr + _rows(res.triplets_spatial)
    elif mode == "semantic":
        table = "**content graph (semantic)**\n\n" + hdr + _rows(res.triplets_semantic)
    else:
        table = ("**layout graph (spatial)**\n\n" + hdr + _rows(res.triplets_spatial)
                 + "\n\n**content graph (semantic)**\n\n" + hdr
                 + _rows(res.triplets_semantic))
    t = res.timing
    stats = (f"**{t.fps:.1f} FPS**  ·  detector {t.det:.1f} ms  ·  "
             f"backbone {t.backbone:.1f} ms  ·  relations {t.relation:.1f} ms  "
             f"·  **total {t.total:.1f} ms**  ·  {len(res.boxes_xyxy)} objects")
    if mode != "merged" and not PIPE.has_dual_head:
        stats += "  ·  ⚠ checkpoint has no dual head — split is a re-ranking only"
    return render(res, show_masks, show_labels, mode), table, stats


def apply_vocab(classes_txt, preds_txt):
    if PIPE is None:
        return "pipeline not ready"
    msgs = []
    try:
        cls = [c.strip() for c in classes_txt.replace("\n", ",").split(",") if c.strip()]
        if cls:
            PIPE.set_object_classes(cls)
            msgs.append(f"detector → {len(cls)} classes")
    except Exception as e:
        msgs.append(f"detector FAILED: {e}")
    try:
        prs = [p.strip() for p in preds_txt.replace("\n", ",").split(",") if p.strip()]
        if prs:
            PIPE.set_predicates(prs)
            msgs.append(f"relations → {len(prs)} predicates")
    except Exception as e:
        msgs.append(f"relations FAILED: {e}")
    return " · ".join(msgs)


def build_ui(device_note: str):
    with gr.Blocks(title="RelateAnything — live scene graphs",
                   theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# RelateAnything · live open-vocabulary scene graphs\n"
            "YOLOE-11m (masks) feeds boxes to an open-vocabulary relation "
            f"model. **Both vocabularies are editable live.** {device_note}")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Webcam"):
                    cam = gr.Image(sources=["webcam"], streaming=True,
                                   type="numpy", label="camera",
                                   height=380)
                with gr.Tab("Image / upload"):
                    still = gr.Image(sources=["upload", "clipboard"],
                                     type="numpy", label="image", height=380)
                out = gr.Image(label="scene graph", height=440)
                stats = gr.Markdown()
            with gr.Column(scale=2):
                gr.Markdown("### Vocabularies — type anything, then Apply")
                classes_txt = gr.Textbox(
                    label="object classes (YOLOE prompts) — leave empty on the "
                          "prompt-free checkpoint to detect anything",
                    lines=4, value="")
                preds_txt = gr.Textbox(
                    label="predicates (relation head)", lines=4,
                    value=", ".join(_default_predicates()))
                apply_btn = gr.Button("Apply vocabularies", variant="primary")
                vocab_msg = gr.Markdown()
                gr.Markdown("### Knobs")
                conf = gr.Slider(0.05, 0.9, 0.25, step=0.05,
                                 label="detector confidence")
                top_k = gr.Slider(1, 30, 12, step=1, label="max triplets")
                score_thr = gr.Slider(0.0, 0.95, 0.30, step=0.05,
                                      label="relation score threshold")
                show_masks = gr.Checkbox(True, label="show masks")
                show_labels = gr.Checkbox(True, label="show labels")
                mode = gr.Radio(
                    ["merged", "both", "spatial", "semantic"], value="merged",
                    label="graph decode",
                    info="merged = one ranked graph. The others use the "
                         "two-graph decode: the same forward pass ranked "
                         "separately inside the spatial (layout, blue) and "
                         "semantic (content, green) predicate columns — "
                         "measured to beat a single graph of twice the budget "
                         "on 6/6 benchmark cells.")
                spatial_raw = gr.Checkbox(
                    False, label="spatial stream: drop relatedness prior",
                    info="ON matches the measured spatial-truth optimum "
                         "(+0.068 macro AUC on SpatialSense), but relatedness "
                         "is also what suppresses junk pairs from overlapping "
                         "detections — with it OFF the spatial scores saturate "
                         "near 1.00 and duplicate boxes surface. OFF is the "
                         "readable default for a demo.")
                table = gr.Markdown()

        apply_btn.click(apply_vocab, [classes_txt, preds_txt], vocab_msg)
        args_in = [conf, top_k, score_thr, show_masks, show_labels, mode,
                   spatial_raw]
        cam.stream(infer, [cam] + args_in, [out, table, stats],
                   stream_every=0.12, concurrency_limit=1, show_progress="hidden")
        still.change(infer, [still] + args_in, [out, table, stats])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PipelineConfig.ckpt)
    ap.add_argument("--det", default="checkpoints/detectors/yoloe-11m-seg-pf.pt",
                    help="'-pf' = prompt-free YOLOE (works with no class list); "
                         "yoloe-11m-seg.pt = text-prompt (needs classes set)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_objects", type=int, default=16)
    ap.add_argument("--final_budget", type=int, default=64)
    ap.add_argument("--no_overlap", action="store_true",
                    help="disable detector‖backbone overlap (A/B the pipeline)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    import torch
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if dev != a.device:
        print("[demo] CUDA unavailable — falling back to CPU")

    global PIPE
    prompt_free = "-pf" in a.det
    cfg = PipelineConfig(ckpt=a.ckpt, det_weights=a.det, device=dev,
                         max_objects=a.max_objects,
                         final_budget=a.final_budget,
                         overlap=(dev == "cuda" and not a.no_overlap),
                         default_classes=None if prompt_free else DEFAULT_CLASSES)
    print(f"[demo] loading pipeline on {dev} …")
    PIPE = ParallelScenePipeline(cfg)
    note = (f"Running on **{dev.upper()}**"
            + (" with detector‖backbone overlap." if cfg.overlap else "."))
    print(f"[demo] ready — {note}")
    build_ui(note).queue(max_size=4).launch(
        server_name="0.0.0.0", server_port=a.port, share=a.share,
        show_error=True)


if __name__ == "__main__":
    main()
