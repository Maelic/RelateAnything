"""Render open-vocabulary scene graphs on MEGASG val images.

Pipeline: raw (pretrained, not MegaSG-finetuned) YOLO-World v2 OR YOLOE
detects objects in the image with an open-vocab prompt list -> every
detected box is handed (no labels used by the head, boxes only) to the
trained relation checkpoint, which scores its own sampled candidate pairs
over its native 10,102-predicate vocabulary (no reparameterization: exactly
what the model learned at train time, most "any relation" reading of the
checkpoint) -> edges are kept above a predicate-confidence threshold, capped
for legibility, and drawn on the image.

--det_vocab controls the DETECTOR's prompt vocabulary (independent from the
relation model's vocab, which is never reparameterized):
    megasg  the 497 categories MEGASG/the relation model was trained on
            (closed-set sanity check, IS a form of reparameterization to the
            training distribution)
    lvis    the 1203 LVIS categories bundled with ultralytics (1029 of them
            NOT in the MEGASG-497 set) -> true open-vocab test: can the
            pipeline find + relate objects outside the training vocabulary?

Usage:
    python deploy/render_val_graphs.py \\
        --checkpoint runs/train/full_v33a_50ep_v3/checkpoint_best.pth \\
        --data_root runs/packed/megasg --n_samples 10 \\
        --det_arch yolo-world --det_vocab lvis \\
        --det_conf 0.4 --rel_conf 0.5 \\
        --out_dir runs/analysis/val_graph_demo
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.api import RelateAnything          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

_DEFAULT_DET_WEIGHTS = {
    "yolo-world": "checkpoints/detectors/yolov8x-worldv2.pt",
    "yoloe": "checkpoints/detectors/yoloe-11l-seg.pt",
}


def load_lvis_vocab() -> list[str]:
    """1203 LVIS category names bundled with ultralytics, cleaned up: each
    LVIS entry is a '/'-joined synonym list (e.g. "aerosol_can/spray_can");
    take the first alias and de-underscore it. NOT the MEGASG/Objects365-497
    vocabulary the relation model trained on (1029 of 1203 are disjoint)."""
    import ultralytics
    import yaml
    path = os.path.join(os.path.dirname(ultralytics.__file__),
                        "cfg", "datasets", "lvis.yaml")
    names = yaml.safe_load(open(path))["names"]
    names = names.values() if isinstance(names, dict) else names
    out, seen = [], set()
    for n in names:
        clean = n.split("/")[0].replace("_", " ")
        if clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def load_vocab(name: str, categories_from_pack) -> list[str]:
    if name == "megasg":
        return categories_from_pack
    if name == "lvis":
        return load_lvis_vocab()
    raise ValueError(f"unknown --det_vocab {name!r}")


def load_detector(arch: str, weights: str, classes):
    if arch == "yolo-world":
        from ultralytics import YOLOWorld
        det = YOLOWorld(weights)
        det.set_classes(classes)
        return det
    if arch == "yoloe":
        from ultralytics import YOLOE
        det = YOLOE(weights)
        det.set_classes(classes, det.get_text_pe(classes))
        return det
    raise ValueError(f"unknown --det_arch {arch!r}")


def detect(det, img_path, conf, imgsz, max_det):
    r = det.predict(img_path, conf=conf, imgsz=imgsz, max_det=max_det,
                    agnostic_nms=True, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), []
    xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float32)
    cf = r.boxes.conf.cpu().numpy().astype(np.float32)
    cls = r.boxes.cls.cpu().numpy().astype(int)
    names = [det.names[c] for c in cls]
    return xyxy, cf, names


@torch.no_grad()
def predict_all(ra: RelateAnything, image, boxes_xyxy, box_labels, box_scores,
                max_boxes=60):
    """Like RelateAnything.predict() but returns EVERY valid candidate pair's
    top predicate + raw predicate-confidence (sigmoid) score, unfiltered and
    uncapped, so the caller can threshold/rank/inspect the distribution
    itself instead of only getting a fixed top-K."""
    boxes_xyxy = np.asarray(boxes_xyxy, np.float32).reshape(-1, 4)
    N = len(boxes_xyxy)
    if N < 2:
        return []
    box_scores = np.asarray(box_scores, np.float32).reshape(-1)
    if N > max_boxes:
        order = np.argsort(-box_scores)[:max_boxes]
        boxes_xyxy = boxes_xyxy[order]
        box_scores = box_scores[order]
        box_labels = [box_labels[i] for i in order]
        N = max_boxes

    img_t, W, H = ra._to_chw(image, ra.img_size)
    img_t = img_t.to(ra.device)
    b = boxes_xyxy.copy()
    b[:, [0, 2]] /= max(W, 1)
    b[:, [1, 3]] /= max(H, 1)
    cx = (b[:, 0] + b[:, 2]) / 2
    cy = (b[:, 1] + b[:, 3]) / 2
    bw = b[:, 2] - b[:, 0]
    bh = b[:, 3] - b[:, 1]
    boxes_t = torch.from_numpy(np.stack([cx, cy, bw, bh], -1).astype(np.float32))
    boxes_t = boxes_t.unsqueeze(0).to(ra.device)
    box_counts = torch.tensor([N], device=ra.device)

    out = ra.model(img_t, boxes_t, box_counts=box_counts, targets=None)
    logits = out["logits"][0].float()
    sub_idx = out["sub_idx"][0].cpu().numpy()
    obj_idx = out["obj_idx"][0].cpu().numpy()
    valid = out["valid_mask"][0].cpu().numpy().astype(bool)
    pair_gate = None
    if out.get("pair_logits") is not None:
        # The score contract: sigmoid(pred_logit + rel_logit), the same one
        # model.predict and the evaluators use.
        logits = logits + out["pair_logits"][0].float().unsqueeze(-1)
        pair_gate = torch.sigmoid(out["pair_logits"][0].float()).cpu().numpy()
    scores = torch.sigmoid(logits)
    best_s, best_p = scores.max(dim=-1)
    best_s = best_s.cpu().numpy()
    best_p = best_p.cpu().numpy()

    cand = []
    for k in range(len(sub_idx)):
        if not valid[k]:
            continue
        si, oi = int(sub_idx[k]), int(obj_idx[k])
        if si >= N or oi >= N or si == oi:
            continue
        cand.append({
            "sub_idx": si, "obj_idx": oi,
            "predicate": ra.predicates[int(best_p[k])],
            "pred_conf": float(best_s[k]),
            "pair_gate": float(pair_gate[k]) if pair_gate is not None else None,
            "sub_box": boxes_xyxy[si].tolist(), "obj_box": boxes_xyxy[oi].tolist(),
            "sub_label": box_labels[si], "obj_label": box_labels[oi],
            "sub_det_conf": float(box_scores[si]), "obj_det_conf": float(box_scores[oi]),
        })
    return cand


_PALETTE = ["#4285F4", "#DB4437", "#0F9D58", "#F4B400", "#AB47BC",
            "#00ACC1", "#FF7043", "#7CB342", "#5C6BC0", "#EC407A"]


def render_panel(ax, img_np, boxes_xyxy, labels, box_confs, edges, title):
    import matplotlib.patches as mpatches

    ax.imshow(img_np)
    for i, (b, lab, c) in enumerate(zip(boxes_xyxy, labels, box_confs)):
        color = _PALETTE[i % len(_PALETTE)]
        x0, y0, x1, y1 = b
        ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                        fill=False, edgecolor=color, linewidth=2))
        ax.text(x0, max(0, y0 - 4), f"{i}:{lab} {c:.2f}", fontsize=7, color="white",
               bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"))
    for e in edges:
        sb, ob = e["sub_box"], e["obj_box"]
        sc = ((sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2)
        oc = ((ob[0] + ob[2]) / 2, (ob[1] + ob[3]) / 2)
        ax.annotate("", xy=oc, xytext=sc,
                   arrowprops=dict(arrowstyle="->", color="white", lw=1.6,
                                  shrinkA=6, shrinkB=6))
        mid = ((sc[0] + oc[0]) / 2, (sc[1] + oc[1]) / 2)
        ax.text(mid[0], mid[1], f"{e['predicate']} ({e['pred_conf']:.2f})",
               fontsize=7.5, color="black", ha="center", va="center",
               bbox=dict(facecolor="white", alpha=0.85, pad=1.5, edgecolor="none"))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                   default="runs/train/full_v33a_50ep_v3/checkpoint_best.pth")
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--data_root", default="runs/packed/megasg")
    ap.add_argument("--split", default="val")
    ap.add_argument("--det_arch", default="yolo-world", choices=["yolo-world", "yoloe"])
    ap.add_argument("--det_weights", default="",
                   help="RAW (not MegaSG-finetuned) detector checkpoint; "
                        "default derived from --det_arch")
    ap.add_argument("--det_vocab", default="lvis", choices=["megasg", "lvis"],
                   help="detector prompt vocabulary (independent of the "
                        "relation model's native, never-reparameterized vocab)")
    ap.add_argument("--det_conf", type=float, default=0.4)
    ap.add_argument("--det_imgsz", type=int, default=640)
    ap.add_argument("--det_max", type=int, default=25, help="max detections/image")
    ap.add_argument("--rel_conf", type=float, default=0.5,
                   help="min predicate sigmoid confidence to draw an edge")
    ap.add_argument("--max_edges", type=int, default=12,
                   help="cap on edges drawn per image (after thresholding)")
    ap.add_argument("--max_boxes", type=int, default=20,
                   help="cap on boxes handed to the relation head")
    ap.add_argument("--n_samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=448, help="relation-model input res")
    ap.add_argument("--out_dir", default="runs/analysis/val_graph_demo")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "panels"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[render] device={device}")

    meta = json.load(open(os.path.join(args.data_root, args.split, "meta.json")))
    file_names = json.load(open(os.path.join(args.data_root, args.split, "file_names.json")))
    img_dir = meta["img_dir"]
    categories = meta["categories"]
    print(f"[render] {len(file_names)} images in {args.split}, {len(categories)} MEGASG object classes")

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(file_names), size=args.n_samples, replace=False)
    sample_paths = [os.path.join(img_dir, file_names[i]) for i in idx]

    det_weights = args.det_weights or _DEFAULT_DET_WEIGHTS[args.det_arch]
    det_vocab = load_vocab(args.det_vocab, categories)
    print(f"[render] loading detector: {args.det_arch} ({det_weights}), "
         f"vocab={args.det_vocab} ({len(det_vocab)} classes, no reparameterization)")
    det = load_detector(args.det_arch, det_weights, det_vocab)
    if device.type == "cuda":
        det.to(device)

    print(f"[render] loading relation checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    pred_names = ckpt["pred_names"]
    print(f"[render] native vocab: {len(pred_names)} predicates | ckpt epoch {ckpt.get('epoch')}")
    ra = RelateAnything(model, pred_names, img_size=args.img_size, device=device,
                        score_mode="sigmoid")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_dump = []
    fig_grid, axes_grid = plt.subplots(2, 5, figsize=(28, 12))
    axes_flat = axes_grid.flatten()

    for n, (i, path) in enumerate(zip(idx.tolist(), sample_paths)):
        img = Image.open(path).convert("RGB")
        img_np = np.asarray(img)
        boxes, confs, labels = detect(det, path, args.det_conf, args.det_imgsz, args.det_max)
        n_det = len(boxes)

        cand = predict_all(ra, img, boxes, labels, confs, max_boxes=args.max_boxes) if n_det >= 2 else []
        if cand:
            scores = np.array([c["pred_conf"] for c in cand])
            print(f"[{n}] {os.path.basename(path)}: {n_det} boxes, "
                 f"{len(cand)} candidate pairs, pred_conf min/mean/max = "
                 f"{scores.min():.3f}/{scores.mean():.3f}/{scores.max():.3f}")
        else:
            print(f"[{n}] {os.path.basename(path)}: {n_det} boxes, no candidate pairs")

        kept = [c for c in cand if c["pred_conf"] >= args.rel_conf]
        kept.sort(key=lambda c: -c["pred_conf"])
        kept = kept[:args.max_edges]

        title = f"{os.path.basename(path)}  |  {n_det} boxes, {len(kept)}/{len(cand)} edges kept"
        render_panel(axes_flat[n], img_np, boxes, labels, confs, kept, title)

        fig_single, ax_single = plt.subplots(figsize=(9, 9 * img_np.shape[0] / max(img_np.shape[1], 1)))
        render_panel(ax_single, img_np, boxes, labels, confs, kept, title)
        fig_single.tight_layout()
        fig_single.savefig(os.path.join(args.out_dir, "panels", f"sample_{n:02d}.png"), dpi=130)
        plt.close(fig_single)

        all_dump.append({
            "sample": n, "file_name": file_names[i], "path": path,
            "n_detections": n_det,
            "boxes": [{"box": b.tolist(), "label": l, "conf": float(c)}
                     for b, l, c in zip(boxes, labels, confs)],
            "n_candidate_pairs": len(cand),
            "edges_kept": kept,
        })

    for j in range(len(idx), len(axes_flat)):
        axes_flat[j].axis("off")
    fig_grid.tight_layout()
    fig_grid.savefig(os.path.join(args.out_dir, "grid.png"), dpi=110)
    plt.close(fig_grid)

    json.dump({
        "det_arch": args.det_arch, "det_weights": det_weights,
        "det_vocab": args.det_vocab, "det_vocab_size": len(det_vocab),
        "det_conf": args.det_conf, "rel_conf": args.rel_conf,
        "samples": all_dump,
    }, open(os.path.join(args.out_dir, "samples.json"), "w"),
              indent=2, default=float)
    print(f"\n[render] wrote grid.png + panels/sample_*.png + samples.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
