#!/usr/bin/env python3
"""
annotate_mask_som.py — Set-of-Mark mask prompting evaluation.

Compares mask-based visual grounding (SAM3 masks from O365) against the
standard bbox overlay, using the same Gemma-4 E2B pipeline from prompt_eval.

Workflow
--------
1.  Match MEGASG-val O365 images to the O365 mask DB (SAM3 masks)
2.  For each MEGASG object, find the best spatially-overlapping SAM3 mask
3.  Overlay masks in SoM style (coloured fill + ID number) on the image
4.  Run Gemma-4 inference, collect metrics, compare to bbox baseline

Usage
-----
  python datagen/annotate_mask_som.py --n_eval 200 --n_viz 20 \
      --outdir runs/mask_som_e2b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_util

import sys
sys.path.insert(0, str(Path(__file__).parent))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

from prompt_eval import (
    MODEL_CHOICES,
    FORBIDDEN_PREDS,
    _extract_json,
    _format_prompt,
    _object_list,
    _is_forbidden,
    _gen_cfg,
    load_model,
    compute_metrics,
    draw_viz_panel,
    print_summary,
    clip_score_image,
    annotate_image,
    _ACTIVE_MODEL_ID,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
MEGASG_DIR = os.environ.get(
    "MEGASG_DIR",
    str(DATASETS / "MEGASG"),
)
MASK_DB_PATH = os.environ.get(
    "MASK_DB_PATH",
    "db/mask_only/o365-train.sqlite",
)

# SoM colour palette — 20 visually-distinct colours (CSS-safe hex)
SOM_PALETTE = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]

DEFAULT_PROMPT_FILE = str(REPO_ROOT / "datagen/prompts/compound_no_overlap_v3.txt")


# ── Font loading ────────────────────────────────────────────────────────────────
def _get_fonts(size_big=32, size_small=14):
    try:
        big = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size_big
)
        small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size_small
)
    except OSError:
        big = small = ImageFont.load_default()
    return big, small


# ── Mask DB helpers ────────────────────────────────────────────────────────────
def open_mask_db(db_path: str = MASK_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def load_masks_for_image(
    conn: sqlite3.Connection, db_img_id: int, img_h: int, img_w: int
) -> list[dict]:
    """Load and decode all SAM3 masks for one image."""
    rows = conn.execute(
        "SELECT id, pos, lbl, rle, iou FROM masks WHERE img = ? ORDER BY pos",
        (db_img_id,),
).fetchall()

    cat_names = dict(
        conn.execute("SELECT id, name FROM categories").fetchall()
)

    masks = []
    for r in rows:
        rle_str = r["rle"].decode("utf-8") if isinstance(r["rle"], bytes) else r["rle"]
        rle = {"counts": rle_str, "size": [img_h, img_w]}
        binary = mask_util.decode(rle)  # (H, W) uint8
        masks.append({
            "mask_id": r["id"],
            "pos": r["pos"],
            "label": cat_names.get(r["lbl"], f"cls_{r['lbl']}"),
            "iou": r["iou"],
            "binary": binary,
        })
    return masks


def mask_to_bbox(binary: np.ndarray) -> list[int]:
    """Convert binary mask to [x1, y1, x2, y2]."""
    ys, xs = np.where(binary)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def compute_iou_mask_bbox(mask_binary: np.ndarray, bbox: list[int]) -> float:
    """IoU between a binary mask and a bbox [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    h, w = mask_binary.shape
    # Clamp bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0

    bbox_mask = np.zeros_like(mask_binary)
    bbox_mask[y1:y2, x1:x2] = 1

    inter = (mask_binary & bbox_mask).sum()
    union = (mask_binary | bbox_mask).sum()
    return float(inter) / max(float(union), 1.0)


def match_objects_to_masks(
    objects: list[dict], masks: list[dict], iou_threshold: float = 0.2
) -> list[dict | None]:
    """For each MEGASG object, find the best overlapping SAM3 mask.

    Returns a list parallel to `objects` — each entry is a mask dict or None.
    """
    matched: list[dict | None] = [None] * len(objects)
    used_mask_ids: set[int] = set()

    # Compute all IoUs
    scores = []
    for oi, obj in enumerate(objects):
        for mi, mask in enumerate(masks):
            iou = compute_iou_mask_bbox(mask["binary"], obj["bbox"])
            if iou >= iou_threshold:
                scores.append((iou, oi, mi))

    # Greedy assignment (highest IoU first)
    scores.sort(reverse=True)
    for iou, oi, mi in scores:
        if matched[oi] is not None:
            continue
        if masks[mi]["mask_id"] in used_mask_ids:
            continue
        matched[oi] = masks[mi]
        used_mask_ids.add(masks[mi]["mask_id"])

    return matched


# ── SoM mask overlay ──────────────────────────────────────────────────────────
def annotate_image_som_mask(
    pil: Image.Image,
    objects: list[dict],
    masks_matched: list[dict | None],
    alpha: float = 0.45,
) -> Image.Image:
    """Set-of-Mark overlay: coloured mask fill + bold ID number per object.

    Objects without a matched mask fall back to bbox overlay (like `som` mode).
    """
    img = pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    font_big, _ = _get_fonts(size_big=32)

    for i, (obj, mask_info) in enumerate(zip(objects, masks_matched)):
        hex_color = SOM_PALETTE[i % len(SOM_PALETTE)]
        # Parse hex to RGB
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        a = int(alpha * 255)
        idx_str = str(i + 1)

        if mask_info is not None:
            binary = mask_info["binary"]
            # Create a coloured mask overlay
            mask_rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
            mask_rgba[binary > 0] = [r, g, b, a]
            mask_img = Image.fromarray(mask_rgba, "RGBA")
            overlay = Image.alpha_composite(overlay, mask_img)

            # Draw contour for crisp boundary
            contours, _ = cv2.findContours(
                binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
            # Convert contours to polygon points for PIL
            draw_overlay = ImageDraw.Draw(overlay)
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    draw_overlay.line(pts + [pts[0]], fill=(r, g, b, 255), width=2)

            # Centroid of mask for number placement
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            # Fallback: bbox overlay
            x1, y1, x2, y2 = obj["bbox"]
            draw_overlay.rectangle([x1, y1, x2, y2], fill=(r, g, b, a))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw centred ID number
        bb = draw_overlay.textbbox((0, 0), idx_str, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw_overlay.text(
            (cx - tw // 2, cy - th // 2),
            idx_str,
            font=font_big,
            fill=(255, 255, 255, 255),
            stroke_width=3,
            stroke_fill=(0, 0, 0, 255),
)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


# ── Build matched sample list ─────────────────────────────────────────────────
def build_matched_samples(
    n: int,
    megasg_dir: str = MEGASG_DIR,
    mask_db_path: str = MASK_DB_PATH,
    split: str = "val",
    filter_to_rel_objects: bool = False,
    min_masks: int = 3,
) -> list[dict]:
    """Load MEGASG COCO samples and attach compatibility mask fields.

    This keeps downstream scripts working without relying on SAM3 DB masks.
    """
    import prompt_eval as _pe

    anno_path = f"{megasg_dir}/{split}/_annotations.coco.json"
    img_dir = f"{megasg_dir}/{split}"
    samples = _pe._load_coco_samples(
        n,
        anno_path=anno_path,
        img_dir=img_dir,
        filter_to_rel_objects=filter_to_rel_objects,
)
    for s in samples:
        s["matched_masks"] = [None] * len(s["objects"])
        s["n_masks_matched"] = 0
        s["n_masks_total"] = 0
    print(f"  Loaded {len(samples)} MEGASG images (SAM3 disabled; using COCO objects only)")
    return samples



# ── Single-image inference with mask overlay ──────────────────────────────────
@torch.inference_mode()
def run_one_mask(
    processor,
    model,
    sample: dict,
    *,
    prompt_template: str,
    token_budget: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float = 1.0,
    max_rels: int | None = None,
    max_objects: int | None = None,
    model_id: str = "google/gemma-4-E2B-it",
) -> tuple[float, dict | None, str]:
    """Run inference with SoM mask overlay."""
    objects = sample["objects"]
    matched_masks = sample["matched_masks"]

    if max_objects is not None and len(objects) > max_objects:
        areas = [
            (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1])
            for o in objects
        ]
        indices = sorted(range(len(objects)), key=lambda i: areas[i], reverse=True)[
:max_objects
        ]
        objects = [objects[i] for i in indices]
        matched_masks = [matched_masks[i] for i in indices]

    processor.image_processor.max_soft_tokens = token_budget

    gen_cfg = _gen_cfg(max_new_tokens, greedy, temperature, model_id=model_id)
    ann_img = annotate_image_som_mask(sample["pil_image"], objects, matched_masks)
    prompt = _format_prompt(prompt_template, objects, max_rels=max_rels)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
)
    inputs = processor(text=text, images=[ann_img], return_tensors="pt").to(
        model.device
)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**inputs, generation_config=gen_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    raw = processor.decode(out[0][input_len:], skip_special_tokens=True)
    sg = _extract_json(raw)
    return elapsed, sg, raw


# ── Visualization (with mask overlay) ─────────────────────────────────────────
def draw_mask_viz_panel(
    sample: dict,
    sg: dict | None,
    elapsed: float,
    idx: int,
    outdir: Path,
) -> Path:
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ann_img = annotate_image_som_mask(
        sample["pil_image"], sample["objects"], sample["matched_masks"]
)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    n_matched = sample["n_masks_matched"]
    n_total = len(sample["objects"])
    fig.suptitle(
        f"{sample['file_name']}  ({n_total} objects, {n_matched} masks matched) — SoM mask mode",
        fontsize=10,
)
    axes[0].imshow(ann_img)
    axes[0].axis("off")

    lines = [f"── Gemma-4 predictions ({elapsed:.1f}s) ──\n"]
    if sg:
        desc = sg.get("scene_description", "")
        if desc:
            lines.append(textwrap.fill(f'"{desc}"', 52))
            lines.append("")
        for r in sg.get("relations", []):
            sid = r.get("subject_id", "?")
            slbl = r.get("subject_label", "?")
            pred = r.get("predicate", "?")
            oid = r.get("object_id", "?")
            olbl = r.get("object_label", "?")
            flag = "  ⚠" if _is_forbidden(pred) else ""
            lines.append(f"• {sid}_{slbl} –[{pred}]→ {oid}_{olbl}{flag}")
        if sg.get("_truncated"):
            lines.append("\n(output truncated)")
    else:
        lines.append("(parse failed)")

    lines.append(f"\n── MEGASG ground truth ──\n")
    for s, p, o in sample.get("gt_relations", []):
        lines.append(f"• {s} –[{p}]→ {o}")

    axes[1].text(
        0.05,
        0.97,
        "\n".join(lines),
        transform=axes[1].transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)
    axes[1].axis("off")

    plt.tight_layout()
    fname = f"viz_{idx:02d}_{Path(sample['file_name']).stem}.png"
    out_path = outdir / fname
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="SoM mask-prompting evaluation on MEGASG"
)
    parser.add_argument("--prompt_file", type=str, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--token_budget", type=int, default=140)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--greedy", dest="greedy", action="store_true", default=True)
    parser.add_argument("--sampling", dest="greedy", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--n_viz", type=int, default=20)
    parser.add_argument("--outdir", type=str, default="runs/mask_som_e2b")
    parser.add_argument("--max_time_s", type=float, default=10.0)
    parser.add_argument("--skip_clip", action="store_true")
    parser.add_argument("--no_flash", action="store_true")
    parser.add_argument("--quant4", action="store_true")
    parser.add_argument(
        "--model",
        type=str,
        default="e2b",
        choices=list(MODEL_CHOICES.keys()),
)
    parser.add_argument("--max_rels", type=int, default=None)
    parser.add_argument("--max_objects", type=int, default=None)
    parser.add_argument(
        "--also_bbox",
        action="store_true",
        help="Also run bbox baseline on the same images for direct comparison",
)
    parser.add_argument(
        "--mask_alpha", type=float, default=0.45,
        help="Mask overlay opacity (0–1). Default 0.45.",
)
    parser.add_argument(
        "--mode", type=str, default="mask_som",
        choices=["mask_som", "point", "bbox"],
        help="Visual grounding mode. mask_som=SoM mask overlay, point=centroid circles+IDs, bbox=bbox overlay.",
)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    viz_dir = outdir / "viz"
    outdir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    prompt_template = Path(args.prompt_file).read_text()
    prompt_label = Path(args.prompt_file).stem

    model_id = MODEL_CHOICES[args.model]

    config = {
        "run_id": outdir.name,
        "prompt_label": prompt_label,
        "prompt_file": args.prompt_file,
        "token_budget": args.token_budget,
        "max_new_tokens": args.max_new_tokens,
        "greedy": args.greedy,
        "temperature": args.temperature if not args.greedy else None,
        "image_mode": args.mode,
        "mask_alpha": args.mask_alpha,
        "max_rels": args.max_rels,
        "max_objects": args.max_objects,
        "model_id": model_id,
        "n_eval": args.n_eval,
        "n_viz": args.n_viz,
        "max_time_s": args.max_time_s,
        "skip_clip": args.skip_clip,
        "also_bbox": args.also_bbox,
        "dataset": "megasg",
    }

    W = 60
    print("=" * W)
    print("  SoM Mask Prompting — Gemma 4 SGG Annotation Eval")
    print("=" * W)
    print(f"  Model: {model_id}{'  [4-bit NF4]' if args.quant4 else ''}")
    print(f"  Prompt: {prompt_label}")
    print(f"  Budget: {args.token_budget} tok  greedy={args.greedy}")
    print(f"  Mode: {args.mode}")
    print(f"  Mask alpha: {args.mask_alpha}")
    print(f"  N eval: {args.n_eval}  N viz: {args.n_viz}")
    print(f"  Also bbox: {args.also_bbox}")
    print(f"  Out: {outdir}")

    # ── 1. Load samples ─────────────────────────────────────────────────────
    n_load = max(args.n_eval, args.n_viz)
    print(f"\n[1] Building matched sample list ({n_load} images) …")
    samples = build_matched_samples(n_load)
    if len(samples) < args.n_eval:
        print(
            f"  WARNING: only {len(samples)} matched images found "
            f"(requested {args.n_eval})"
)
    print(f"  Got {len(samples)} images.")

    # Mask match stats
    n_full_match = sum(1 for s in samples if s["n_masks_matched"] == len(s["objects"]))
    avg_match = np.mean(
        [s["n_masks_matched"] / max(len(s["objects"]), 1) for s in samples]
)
    print(
        f"  Mask coverage: {avg_match:.0%} avg, "
        f"{n_full_match}/{len(samples)} fully matched"
)

    # ── 2. Load model ────────────────────────────────────────────────────────
    print(f"\n[2] Loading model …")

    import prompt_eval
    prompt_eval._ACTIVE_MODEL_ID = model_id

    processor, model = load_model(
        use_flash=not args.no_flash,
        model_id=model_id,
        quant4=args.quant4,
)

    # Save image list early so we can reuse the same set for other modes
    img_list_path = outdir / "image_list.json"
    with open(img_list_path, "w") as f:
        json.dump(
            [{"img_id": s["img_id"], "file_name": s["file_name"]} for s in samples],
            f, indent=2,
)
    print(f"  Image list    → {img_list_path}")

    # ── 3. Warm-up ───────────────────────────────────────────────────────────
    print("\n[3] Warm-up pass …")
    if args.mode == "mask_som":
        run_one_mask(
            processor,
            model,
            samples[0],
            prompt_template=prompt_template,
            token_budget=args.token_budget,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            max_rels=args.max_rels,
            max_objects=args.max_objects,
            model_id=model_id,
)
    else:
        from prompt_eval import run_one
        run_one(
            processor,
            model,
            samples[0],
            prompt_template=prompt_template,
            token_budget=args.token_budget,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            image_mode=args.mode,
            max_rels=args.max_rels,
            max_objects=args.max_objects,
)

    # ── 4. Inference ─────────────────────────────────────────────────────────
    eval_samples = samples[: args.n_eval]
    mode_label = args.mode.replace('_', '-')
    print(f"\n[4] Running {mode_label} inference on {len(eval_samples)} images …")
    mask_results = []
    n_errors = 0
    for i, sample in enumerate(eval_samples):
        try:
            if args.mode == "mask_som":
                elapsed, sg, raw = run_one_mask(
                    processor,
                    model,
                    sample,
                    prompt_template=prompt_template,
                    token_budget=args.token_budget,
                    max_new_tokens=args.max_new_tokens,
                    greedy=args.greedy,
                    temperature=args.temperature,
                    max_rels=args.max_rels,
                    max_objects=args.max_objects,
                    model_id=model_id,
)
            else:
                from prompt_eval import run_one
                elapsed, sg, raw = run_one(
                    processor,
                    model,
                    sample,
                    prompt_template=prompt_template,
                    token_budget=args.token_budget,
                    max_new_tokens=args.max_new_tokens,
                    greedy=args.greedy,
                    temperature=args.temperature,
                    image_mode=args.mode,
                    max_rels=args.max_rels,
                    max_objects=args.max_objects,
)
        except Exception as exc:
            print(
                f"  [{i+1:3d}/{len(eval_samples)}] {sample['file_name']:<40}"
                f"  ERROR: {exc}",
                flush=True,
)
            n_errors += 1
            mask_results.append({
                "elapsed": 0.0,
                "sg": None,
                "raw": f"ERROR: {exc}",
                "sample": sample,
            })
            continue

        n_rel = len(sg["relations"]) if sg else -1
        if args.mode == "mask_som":
            mask_cov = sample["n_masks_matched"]
            extra = f"  masks={mask_cov}/{len(sample['objects'])}"
        else:
            extra = f"  ({args.mode})"
        print(
            f"  [{i+1:3d}/{len(eval_samples)}] {sample['file_name']:<40}"
            f"  {elapsed:.1f}s  {n_rel} rels{extra}",
            flush=True,
)
        mask_results.append({
            "elapsed": elapsed,
            "sg": sg,
            "raw": raw,
            "sample": sample,
        })
    if n_errors:
        print(f"  ⚠ {n_errors} errors encountered (skipped)")

    # ── 5. Metrics ───────────────────────────────────────────────────────────
    print(f"\n[5] Computing metrics ({mode_label} mode) …")
    mask_metrics = compute_metrics(mask_results, args.max_time_s, args.skip_clip)
    mask_metrics["config"] = config
    mask_metrics["timestamp"] = datetime.now().isoformat()
    if args.mode == "mask_som":
        mask_metrics["mask_coverage"] = {
            "avg_match_rate": round(float(avg_match), 4),
            "n_fully_matched": n_full_match,
            "n_total": len(eval_samples),
        }
    mask_metrics["n_errors"] = n_errors

    # ── 6. Viz ───────────────────────────────────────────────────────────────
    print(f"\n[6] Saving {args.n_viz} viz panels …")
    for i, r in enumerate(mask_results[: args.n_viz]):
        try:
            if args.mode == "mask_som":
                p = draw_mask_viz_panel(r["sample"], r["sg"], r["elapsed"], i + 1, viz_dir)
            else:
                p = draw_viz_panel(
                    r["sample"], r["sg"], r["elapsed"], i + 1, viz_dir,
                    image_mode=args.mode,
)
            print(f"  → {p}")
        except Exception as exc:
            print(f"  ⚠ viz {i+1} failed: {exc}")

    # ── 7. (Optional) bbox baseline on same images ───────────────────────────
    if args.also_bbox:
        print(f"\n[7] Running bbox baseline on same {len(eval_samples)} images …")
        from prompt_eval import run_one

        bbox_results = []
        for i, sample in enumerate(eval_samples):
            try:
                elapsed, sg, raw = run_one(
                    processor,
                    model,
                    sample,
                    prompt_template=prompt_template,
                    token_budget=args.token_budget,
                    max_new_tokens=args.max_new_tokens,
                    greedy=args.greedy,
                    temperature=args.temperature,
                    image_mode="bbox",
                    max_rels=args.max_rels,
                    max_objects=args.max_objects,
)
            except Exception as exc:
                print(
                    f"  [{i+1:3d}/{len(eval_samples)}] {sample['file_name']:<40}"
                    f"  ERROR: {exc}",
                    flush=True,
)
                bbox_results.append({"elapsed": 0.0, "sg": None, "raw": f"ERROR: {exc}", "sample": sample})
                continue
            n_rel = len(sg["relations"]) if sg else -1
            print(
                f"  [{i+1:3d}/{len(eval_samples)}] {sample['file_name']:<40}"
                f"  {elapsed:.1f}s  {n_rel} rels  (bbox)",
                flush=True,
)
            bbox_results.append({
                "elapsed": elapsed,
                "sg": sg,
                "raw": raw,
                "sample": sample,
            })

        bbox_metrics = compute_metrics(bbox_results, args.max_time_s, args.skip_clip)
        bbox_metrics["config"] = {**config, "image_mode": "bbox"}
        bbox_metrics["timestamp"] = datetime.now().isoformat()

        # Save bbox results
        bbox_fitness_path = outdir / "fitness_bbox.json"
        with open(bbox_fitness_path, "w") as f:
            json.dump(bbox_metrics, f, indent=2)
        print(f"  Bbox fitness → {bbox_fitness_path}")

        bbox_pred_path = outdir / "predictions_bbox.json"
        with open(bbox_pred_path, "w") as f:
            json.dump(
                {
                    "config": {**config, "image_mode": "bbox"},
                    "predictions": [
                        {
                            "img_id": r["sample"]["img_id"],
                            "file_name": r["sample"]["file_name"],
                            "elapsed_s": round(r["elapsed"], 3),
                            "objects": r["sample"]["objects"],
                            "sg": r["sg"],
                            "gt_relations": r["sample"]["gt_relations"],
                        }
                        for r in bbox_results
                    ],
                },
                f,
                indent=2,
)

        # Print comparison
        print("\n" + "=" * W)
        print("  COMPARISON: MASK-SoM vs BBOX")
        print("=" * W)
        for label, m in [("Mask-SoM", mask_metrics), ("Bbox", bbox_metrics)]:
            pd = m["predicate_distribution"]
            q = m["quality"]
            c = m["clip"]
            print(f"\n  [{label}]")
            print(f"    Fitness: {m['fitness']:.4f}")
            print(f"    Entropy: {pd['entropy_nats']:.3f} nats ({pd['n_unique_predicates']} unique)")
            print(f"    Top-5 cov: {pd['top5_coverage']*100:.1f}%")
            print(f"    Forbidden: {q['forbidden_rate']*100:.1f}%")
            print(f"    Parse OK: {q['parse_ok_rate']*100:.1f}%")
            print(f"    Rels/img: {pd['rels_per_img_mean']:.1f}")
            print(f"    Speed: {m['speed']['median_s']:.2f}s")
            if not c["skipped"]:
                print(f"    CLIP score: {c['mean']:.4f}")
        print("=" * W)

    # ── 8. Save outputs ──────────────────────────────────────────────────────
    fitness_path = outdir / "fitness.json"
    with open(fitness_path, "w") as f:
        json.dump(mask_metrics, f, indent=2)
    print(f"\n  Fitness       → {fitness_path}")

    predictions_path = outdir / "predictions.json"
    preds_out = []
    for r in mask_results:
        entry = {
            "img_id": r["sample"]["img_id"],
            "file_name": r["sample"]["file_name"],
            "elapsed_s": round(r["elapsed"], 3),
            "objects": r["sample"]["objects"],
            "sg": r["sg"],
            "gt_relations": r["sample"]["gt_relations"],
        }
        if args.mode == "mask_som":
            entry["n_masks_matched"] = r["sample"]["n_masks_matched"]
            entry["n_masks_total"] = r["sample"]["n_masks_total"]
        preds_out.append(entry)
    with open(predictions_path, "w") as f:
        json.dump({"config": config, "predictions": preds_out}, f, indent=2)
    print(f"  Predictions   → {predictions_path}")

    # Predicate frequency
    all_preds = Counter()
    for r in mask_results:
        if r["sg"]:
            for rel in r["sg"].get("relations", []):
                p = rel.get("predicate", "").lower().strip()
                if p:
                    all_preds[p] += 1
    freq_path = outdir / "pred_freq.tsv"
    total_preds = max(sum(all_preds.values()), 1)
    with open(freq_path, "w") as f:
        f.write("predicate\tcount\tfraction\tforbidden\n")
        for pred, cnt in all_preds.most_common():
            f.write(f"{pred}\t{cnt}\t{cnt / total_preds:.4f}\t{int(_is_forbidden(pred))}\n")
    print(f"  Pred freq     → {freq_path}")

    print_summary(mask_metrics, config)


if __name__ == "__main__":
    main()
