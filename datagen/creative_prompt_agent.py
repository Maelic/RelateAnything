#!/usr/bin/env python3
"""
creative_prompt_agent.py — Interactive agent that optimises open-vocabulary
SGG prompts by iteratively experimenting with prompt text AND image overlays.

Design philosophy
-----------------
* Small budget: 10 images, ~10 s/image → one trial ≈ 100 s
* Model loaded ONCE, then the agent loops through "ideas"
* Each idea = (prompt_text, overlay_function)
* After each trial the agent renders a rich HTML report with:
  - side-by-side: annotated image vs. predicted relations
  - aggregate stats: unique predicates, entropy, forbidden%, diversity
* The agent is CREATIVE: it programmatically generates novel overlay
  variants (contour-only masks, variable transparency, mask+point combos,
  heatmap-style overlays, edge-glow, etc.) and prompt variants (open-vocab,
  encouraging rare predicates, few-shot exemplars, chain-of-thought, …).

Usage
-----
  python datagen/creative_prompt_agent.py [--n_images 10] [--rounds 12]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import textwrap
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).parent))

from prompt_eval import (
    MODEL_CHOICES,
    _extract_json,
    _format_prompt,
    _is_forbidden,
    _gen_cfg,
    load_model,
    annotate_image,
)
from annotate_mask_som import (
    build_matched_samples,
    SOM_PALETTE,
    _get_fonts,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "datagen" / "prompts"

# ═══════════════════════════════════════════════════════════════════════════════
# OVERLAY LIBRARY — each function: (pil_image, objects, matched_masks) -> Image
# ═══════════════════════════════════════════════════════════════════════════════

def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def overlay_mask_fill(pil, objects, masks, alpha=0.45):
    """Classic SoM: coloured mask fill + contour + centred ID."""
    img = pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    font_big, _ = _get_fonts(32)
    a = int(alpha * 255)

    for i, (obj, m) in enumerate(zip(objects, masks)):
        r, g, b = _hex_to_rgb(SOM_PALETTE[i % len(SOM_PALETTE)])
        idx = str(i + 1)
        if m is not None:
            binary = m["binary"]
            rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
            rgba[binary > 0] = [r, g, b, a]
            overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, "RGBA"))
            draw = ImageDraw.Draw(overlay)
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    draw.line(pts + [pts[0]], fill=(r, g, b, 255), width=2)
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            draw = ImageDraw.Draw(overlay)
            x1, y1, x2, y2 = obj["bbox"]
            draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, a))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bb = draw.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw // 2, cy - th // 2), idx, font=font_big,
                  fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


def overlay_contour_only(pil, objects, masks, thickness=3):
    """Mask contour only — no fill. Cleaner, less occlusion."""
    img = pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_big, _ = _get_fonts(32)

    for i, (obj, m) in enumerate(zip(objects, masks)):
        r, g, b = _hex_to_rgb(SOM_PALETTE[i % len(SOM_PALETTE)])
        idx = str(i + 1)
        if m is not None:
            binary = m["binary"]
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    draw.line(pts + [pts[0]], fill=(r, g, b, 255), width=thickness)
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            x1, y1, x2, y2 = obj["bbox"]
            for t in range(thickness):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(r, g, b, 255))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bb = draw.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw // 2, cy - th // 2), idx, font=font_big,
                  fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


def overlay_contour_thick_glow(pil, objects, masks, thickness=5):
    """Thick contour with Gaussian glow effect — high visibility, no fill."""
    img = pil.copy().convert("RGBA")
    glow_layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw_glow = ImageDraw.Draw(glow_layer)
    font_big, _ = _get_fonts(32)

    for i, (obj, m) in enumerate(zip(objects, masks)):
        r, g, b = _hex_to_rgb(SOM_PALETTE[i % len(SOM_PALETTE)])
        idx = str(i + 1)
        if m is not None:
            binary = m["binary"]
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    # Thick glow line
                    draw_glow.line(pts + [pts[0]], fill=(r, g, b, 180), width=thickness + 4)
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            x1, y1, x2, y2 = obj["bbox"]
            for t in range(thickness + 2):
                draw_glow.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(r, g, b, 120))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bb = draw_glow.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw_glow.text((cx - tw // 2, cy - th // 2), idx, font=font_big,
                       fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))

    # Blur the glow layer for soft edges
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=2))
    # Redraw crisp contours on top
    draw_sharp = ImageDraw.Draw(glow_layer)
    for i, (obj, m) in enumerate(zip(objects, masks)):
        r, g, b = _hex_to_rgb(SOM_PALETTE[i % len(SOM_PALETTE)])
        idx = str(i + 1)
        if m is not None:
            binary = m["binary"]
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    draw_sharp.line(pts + [pts[0]], fill=(r, g, b, 255), width=thickness)
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            x1, y1, x2, y2 = obj["bbox"]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bb = draw_sharp.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw_sharp.text((cx - tw // 2, cy - th // 2), idx, font=font_big,
                        fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))

    return Image.alpha_composite(img, glow_layer).convert("RGB")


def overlay_mask_light(pil, objects, masks):
    """Very light mask fill (15%) + thin contour. Minimal occlusion."""
    return overlay_mask_fill(pil, objects, masks, alpha=0.15)


def overlay_mask_heavy(pil, objects, masks):
    """Heavy mask fill (65%) — maximum region delineation."""
    return overlay_mask_fill(pil, objects, masks, alpha=0.65)


def overlay_contour_plus_point(pil, objects, masks, thickness=2):
    """Contour + coloured centroid point (no fill). Best of both worlds."""
    img = pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_big, _ = _get_fonts(28)

    for i, (obj, m) in enumerate(zip(objects, masks)):
        r, g, b = _hex_to_rgb(SOM_PALETTE[i % len(SOM_PALETTE)])
        idx = str(i + 1)
        if m is not None:
            binary = m["binary"]
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    draw.line(pts + [pts[0]], fill=(r, g, b, 255), width=thickness)
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            x1, y1, x2, y2 = obj["bbox"]
            for t in range(thickness):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(r, g, b, 255))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Draw filled centroid circle
        rr = 10
        draw.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
                     fill=(r, g, b, 255), outline=(255, 255, 255, 255), width=2)
        # Number next to point
        bb = draw.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx + rr + 4, cy - th // 2), idx, font=font_big,
                  fill=(r, g, b, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


def overlay_mask_desaturate_bg(pil, objects, masks, alpha=0.35):
    """Desaturate the background; only masked regions stay vivid."""
    import colorsys
    arr = np.array(pil)
    # Make grayscale background
    gray = np.mean(arr, axis=2, keepdims=True).astype(np.uint8)
    gray_bg = np.repeat(gray, 3, axis=2)

    # Build combined mask of all objects
    h, w = arr.shape[:2]
    any_mask = np.zeros((h, w), dtype=bool)
    for m in masks:
        if m is not None:
            any_mask |= (m["binary"] > 0)

    # Blend: inside mask = original, outside = gray
    result = np.where(any_mask[:,:, None], arr, gray_bg)
    pil_result = Image.fromarray(result)

    # Now overlay contours + IDs on top
    return overlay_contour_only(pil_result, objects, masks, thickness=3)


def overlay_point_only(pil, objects, masks):
    """Point prompting (no mask info). Baseline."""
    return annotate_image(pil, objects, mode="point")


def overlay_point_fixed(pil, objects, masks):
    """Point prompting with fixed marker size (radius=10 px)."""
    return annotate_image(pil, objects, mode="point_fixed")


def overlay_bbox_only(pil, objects, masks):
    """Standard bbox overlay. Baseline."""
    return annotate_image(pil, objects, mode="bbox")


def overlay_mask_edge_highlight(pil, objects, masks, dilation_px=6):
    """Highlight only the edge band of each mask (dilated - eroded). Emphasises boundaries."""
    img = pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_big, _ = _get_fonts(32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px, dilation_px))

    for i, (obj, m) in enumerate(zip(objects, masks)):
        r, g, b = _hex_to_rgb(SOM_PALETTE[i % len(SOM_PALETTE)])
        idx = str(i + 1)
        if m is not None:
            binary = m["binary"].astype(np.uint8)
            dilated = cv2.dilate(binary, kernel)
            eroded = cv2.erode(binary, kernel)
            edge_band = dilated - eroded
            rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
            rgba[edge_band > 0] = [r, g, b, 200]
            overlay = Image.alpha_composite(overlay, Image.fromarray(rgba, "RGBA"))
            draw = ImageDraw.Draw(overlay)
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
        else:
            x1, y1, x2, y2 = obj["bbox"]
            for t in range(3):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(r, g, b, 255))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bb = draw.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw // 2, cy - th // 2), idx, font=font_big,
                  fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


def overlay_contour_with_label(pil, objects, masks, thickness=2):
    """Contour + small label tag at mask top showing 'ID_classname'."""
    img = pil.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font_big, font_small = _get_fonts(28, 14)

    for i, (obj, m) in enumerate(zip(objects, masks)):
        r, g, b = _hex_to_rgb(SOM_PALETTE[i % len(SOM_PALETTE)])
        idx = str(i + 1)
        tag = f" {idx}_{obj['label']} "
        if m is not None:
            binary = m["binary"]
            contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if len(cnt) >= 3:
                    pts = [(int(p[0][0]), int(p[0][1])) for p in cnt]
                    draw.line(pts + [pts[0]], fill=(r, g, b, 255), width=thickness)
            ys, xs = np.where(binary)
            cx, cy = int(xs.mean()), int(ys.mean())
            top_y = int(ys.min())
            top_x = int(xs[ys == ys.min()].mean())
        else:
            x1, y1, x2, y2 = obj["bbox"]
            for t in range(thickness):
                draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=(r, g, b, 255))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            top_x, top_y = (x1 + x2) // 2, y1
        # Label tag at top of mask
        tb = draw.textbbox((0, 0), tag, font=font_small)
        tw2, th2 = tb[2] - tb[0], tb[3] - tb[1]
        ty = max(top_y - th2 - 2, 0)
        tx = max(top_x - tw2 // 2, 0)
        draw.rectangle([tx, ty, tx + tw2, ty + th2], fill=(r, g, b, 200))
        draw.text((tx, ty), tag, font=font_small, fill=(255, 255, 255, 255))
        # Centred number
        bb = draw.textbbox((0, 0), idx, font=font_big)
        tw, th = bb[2] - bb[0], bb[3] - bb[1]
        draw.text((cx - tw // 2, cy - th // 2), idx, font=font_big,
                  fill=(255, 255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

PROMPT_OPENVOC_CONCISE = """\
{n} objects highlighted with coloured overlays. Compound IDs:
{object_list}

Describe ALL visual relationships. You may use ANY predicate — be precise and creative.
Avoid vague terms (near, next to, with, has, beside). Prefer specific actions (gripping, \
leaning against, draped over) or spatial descriptions (above, behind, inside).

Output ONLY JSON:
```json
{{"scene_description": "<one sentence>", "relations": [{{"subject_id": <int>, "subject_label": "<str>", "predicate": "<str>", "object_id": <int>, "object_label": "<str>"}}]}}
```"""

PROMPT_OPENVOC_RICH = """\
{n} objects are marked in this image — each with a coloured region and an ID number.

Compound IDs:
{object_list}

Your task: identify ALL visual relationships between these objects.

VOCABULARY: You are NOT limited to any fixed list. Invent the most accurate predicate.
  · Physical actions: gripping, balancing on, draped over, tucked into, strapped to, \
leaning against, resting upon, hanging off, pinned to, wrapped around, propped against
  · Spatial: above, below, in front of, behind, to the left of, to the right of, \
inside, surrounding, overlapping with, adjacent to, flush with
  · Functional: protecting, supporting, containing, displaying, illuminating, \
reflecting, casting shadow on, framing, obscuring
  · Part-whole: part of, attached to, embedded in, extending from, branching off

BANNED predicates: near, next to, with, has, beside, and, on, same scene, overlapping.
For body parts → "part of".

Think about WHAT each object is DOING relative to others — not just WHERE it is.

Output ONLY this JSON:
```json
{{"scene_description": "<sentence>", "relations": [{{"subject_id": <int>, "subject_label": "<str>", "predicate": "<str>", "object_id": <int>, "object_label": "<str>"}}]}}
```"""

PROMPT_OPENVOC_COT = """\
{n} objects highlighted in this image:
{object_list}

STEP 1: Briefly describe what's happening in the scene (one sentence).
STEP 2: For each pair of nearby objects, ask: "Is there a physical interaction? \
A functional relationship? A containment/part-whole link? Or only spatial proximity?"
STEP 3: Choose the MOST SPECIFIC predicate for each relationship. You may use any \
English verb phrase. BANNED: near, next to, with, has, beside, on.

Output format (JSON only, no other text):
```json
{{"scene_description": "<from step 1>", "relations": [{{"subject_id": <int>, "subject_label": "<str>", "predicate": "<str>", "object_id": <int>, "object_label": "<str>"}}]}}
```"""

PROMPT_OPENVOC_DENSE = """\
{n} objects with coloured overlays:
{object_list}

List EVERY visual relationship you can identify — aim for at least {min_rels} relations.
Use highly specific predicates: prefer "perched on" over "on top of", "clutching" over \
"holding", "obscuring" over "in front of" when something blocks the view.

BANNED: near, next to, with, has, beside, and, on, same scene.
Body parts → "part of".

Output ONLY JSON:
```json
{{"scene_description": "<sentence>", "relations": [{{"subject_id": <int>, "subject_label": "<str>", "predicate": "<str>", "object_id": <int>, "object_label": "<str>"}}]}}
```"""

PROMPT_OPENVOC_CONTRASTIVE = """\
{n} objects marked in this image:
{object_list}

For each pair of objects that interact or overlap, describe their relationship using the \
MOST DISTINGUISHING predicate possible. Two different pairs should almost never share \
the same predicate — maximise variety.

Example predicates: mounted on, dangling from, wedged between, resting against, \
covering, parked beside, protruding from, casting shadow on, reflected in, \
worn on, planted in, suspended above, stacked under, anchored to.

BANNED: near, next to, with, has, beside, and, on, same scene.

Output ONLY JSON:
```json
{{"scene_description": "<sentence>", "relations": [{{"subject_id": <int>, "subject_label": "<str>", "predicate": "<str>", "object_id": <int>, "object_label": "<str>"}}]}}
```"""

PROMPT_OPENVOC_FEWSHOT = """\
{n} objects in this image:
{object_list}

Example output for a different scene with 3 objects (1_cat, 2_laptop, 3_table):
```json
{{"scene_description": "A cat sits on a laptop placed on a wooden table.", "relations": [{{"subject_id": 1, "subject_label": "cat", "predicate": "perched on", "object_id": 2, "object_label": "laptop"}}, {{"subject_id": 2, "subject_label": "laptop", "predicate": "resting on", "object_id": 3, "object_label": "table"}}, {{"subject_id": 1, "subject_label": "cat", "predicate": "looking at", "object_id": 3, "object_label": "table"}}]}}
```

Now do the same for this image. Use the MOST SPECIFIC predicates possible. Open vocabulary.
BANNED: near, next to, with, has, beside, and, on, same scene.

Output ONLY JSON (same format as example):"""

# ═══════════════════════════════════════════════════════════════════════════════
# IDEA REGISTRY — each idea = (name, overlay_fn, prompt_template)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_ideas() -> list[dict]:
    """Generate the full set of (overlay × prompt) ideas to try."""
    overlays = [
        ("mask_fill_45", overlay_mask_fill),
        ("contour_only", overlay_contour_only),
        ("contour_glow", overlay_contour_thick_glow),
        ("mask_light_15", overlay_mask_light),
        ("mask_heavy_65", overlay_mask_heavy),
        ("contour+point", overlay_contour_plus_point),
        ("desaturate_bg", overlay_mask_desaturate_bg),
        ("point_only", overlay_point_only),
        ("bbox_only", overlay_bbox_only),
        ("edge_highlight", overlay_mask_edge_highlight),
        ("contour+label", overlay_contour_with_label),
    ]

    prompts = [
        ("openvoc_concise", PROMPT_OPENVOC_CONCISE),
        ("openvoc_rich", PROMPT_OPENVOC_RICH),
        ("openvoc_cot", PROMPT_OPENVOC_COT),
        ("openvoc_dense", PROMPT_OPENVOC_DENSE),
        ("openvoc_contrastive", PROMPT_OPENVOC_CONTRASTIVE),
        ("openvoc_fewshot", PROMPT_OPENVOC_FEWSHOT),
    ]

    ideas = []
    # Core experiments: try all overlays with the concise prompt first
    for oname, ofn in overlays:
        ideas.append({
            "name": f"{oname}__concise",
            "overlay_fn": ofn,
            "prompt_template": PROMPT_OPENVOC_CONCISE,
            "overlay_name": oname,
            "prompt_name": "openvoc_concise",
        })

    # Then try all prompts with the best couple of overlays
    for pname, ptpl in prompts:
        if pname == "openvoc_concise":
            continue  # Already covered above
        for oname, ofn in [("contour_only", overlay_contour_only),
                            ("contour+point", overlay_contour_plus_point),
                            ("mask_fill_45", overlay_mask_fill)]:
            ideas.append({
                "name": f"{oname}__{pname}",
                "overlay_fn": ofn,
                "prompt_template": ptpl,
                "overlay_name": oname,
                "prompt_name": pname,
            })
    return ideas


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

@torch.inference_mode()
def run_one(processor, model, sample, *, overlay_fn, prompt_template, token_budget,
            max_new_tokens, greedy, temperature, model_id):
    """Run inference with a custom overlay + prompt."""
    objects = sample["objects"]
    masks = sample["matched_masks"]

    processor.image_processor.max_soft_tokens = token_budget
    gen_cfg = _gen_cfg(max_new_tokens, greedy, temperature, model_id=model_id)

    # Build annotated image using the overlay function
    ann_img = overlay_fn(sample["pil_image"], objects, masks)

    # Format prompt
    prompt = prompt_template.format(
        n=len(objects),
        object_list="\n".join(f"  {i+1}_{obj['label']}" for i, obj in enumerate(objects)),
        min_rels=max(len(objects), 4),
)

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
    inputs = processor(text=text, images=[ann_img], return_tensors="pt").to(model.device)
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
    return elapsed, sg, raw, ann_img


# ═══════════════════════════════════════════════════════════════════════════════
# QUICK METRICS  (lightweight, no CLIP)
# ═══════════════════════════════════════════════════════════════════════════════

def quick_metrics(results: list[dict]) -> dict:
    """Fast metrics from a small batch."""
    n = len(results)
    times = [r["elapsed"] for r in results]
    parse_ok = [r for r in results if r["sg"] is not None]

    pred_count = Counter()
    all_rels = []
    for r in parse_ok:
        rels = r["sg"].get("relations", [])
        all_rels.extend(rels)
        for rel in rels:
            p = rel.get("predicate", "").lower().strip()
            if p:
                pred_count[p] += 1

    total = len(all_rels)
    n_unique = len(pred_count)
    rels_per_img = total / max(n, 1)

    if total > 0:
        probs = [c / total for c in pred_count.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
    else:
        entropy = 0.0

    n_forbidden = sum(1 for r in all_rels if _is_forbidden(r.get("predicate", "")))
    forbidden_rate = n_forbidden / max(total, 1)

    top5 = pred_count.most_common(5)
    top5_cov = sum(c for _, c in top5) / max(total, 1)

    return {
        "n_images": n,
        "parse_ok": len(parse_ok),
        "total_rels": total,
        "rels_per_img": round(rels_per_img, 1),
        "n_unique": n_unique,
        "entropy_nats": round(entropy, 3),
        "forbidden_rate": round(forbidden_rate, 4),
        "n_forbidden": n_forbidden,
        "top5_coverage": round(top5_cov, 3),
        "top5": top5,
        "all_predicates": dict(pred_count.most_common()),
        "median_s": round(statistics.median(times), 2),
        "truncated": sum(1 for r in results if r.get("sg") and r["sg"].get("_truncated")),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HTML REPORT
# ═══════════════════════════════════════════════════════════════════════════════

def _img_to_data_uri(img: Image.Image, max_w=600) -> str:
    """Convert PIL image to base64 data URI for inline HTML."""
    import base64, io
    if img.width > max_w:
        ratio = max_w / img.width
        img = img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def render_html_report(
    all_trials: list[dict],
    outpath: Path,
    title: str = "Creative Prompt Agent — Report",
):
    """Render an interactive HTML report comparing all trials."""
    css = """
    body { font-family: system-ui, sans-serif; margin: 20px; background: #f5f5f5; }
    h1 { color: #333; }
.summary-table { border-collapse: collapse; margin: 20px 0; width: 100%; }
.summary-table th,.summary-table td { border: 1px solid #ccc; padding: 6px 10px; text-align: center; }
.summary-table th { background: #444; color: white; }
.summary-table tr:nth-child(even) { background: #eee; }
.summary-table tr:hover { background: #ddf; }
.best { background: #cfc !important; font-weight: bold; }
.trial { background: white; border-radius: 8px; margin: 16px 0; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.15); }
.trial h2 { margin-top: 0; }
.image-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.image-card { border: 1px solid #ddd; border-radius: 6px; padding: 8px; background: #fafafa; max-width: 620px; }
.image-card img { max-width: 100%; border-radius: 4px; }
.rels { font-size: 13px; margin-top: 6px; }
.rel { margin: 2px 0; }
.forbidden { color: red; font-weight: bold; }
.pred-list { font-size: 12px; color: #666; column-count: 3; }
    details { margin: 8px 0; }
    """

    # Sort trials by a composite score: entropy * (1 - forbidden_rate)
    ranked = sorted(
        all_trials,
        key=lambda t: t["metrics"]["entropy_nats"] * (1 - t["metrics"]["forbidden_rate"]),
        reverse=True,
)
    best_name = ranked[0]["name"] if ranked else ""

    rows_html = ""
    for i, trial in enumerate(ranked):
        m = trial["metrics"]
        cls = ' class="best"' if trial["name"] == best_name else ""
        rows_html += f"""<tr{cls}>
        <td>{i+1}</td>
        <td style="text-align:left">{trial['overlay_name']}</td>
        <td style="text-align:left">{trial['prompt_name']}</td>
        <td>{m['n_unique']}</td>
        <td>{m['entropy_nats']:.3f}</td>
        <td>{m['rels_per_img']}</td>
        <td>{m['forbidden_rate']*100:.1f}%</td>
        <td>{m['top5_coverage']*100:.1f}%</td>
        <td>{m['median_s']}s</td>
        <td>{m['parse_ok']}/{m['n_images']}</td>
        </tr>"""

    # Per-trial detail sections
    detail_html = ""
    for trial in ranked:
        m = trial["metrics"]
        cards = ""
        for r in trial["results"][:10]:
            img_uri = _img_to_data_uri(r["ann_img"])
            rels_html = ""
            if r["sg"]:
                for rel in r["sg"].get("relations", []):
                    p = rel.get("predicate", "")
                    cls = ' class="forbidden"' if _is_forbidden(p) else ""
                    rels_html += (
                        f'<div class="rel"{cls}>'
                        f'{rel.get("subject_id","?")}_{rel.get("subject_label","")} '
                        f'<b>→ {p} →</b> '
                        f'{rel.get("object_id","?")}_{rel.get("object_label","")}'
                        f'</div>'
)
                desc = r["sg"].get("scene_description", "")
            else:
                rels_html = "<em>Parse failed</em>"
                desc = ""
            cards += f"""<div class="image-card">
              <img src="{img_uri}" />
              <div class="rels"><em>{desc}</em>{rels_html}
                <div style="font-size:11px;color:#999">{r['elapsed']:.1f}s</div>
              </div></div>"""

        preds_html = ", ".join(f"{p} ({c})" for p, c in m["all_predicates"].items())
        detail_html += f"""
        <div class="trial" id="{trial['name']}">
          <h2>{'🥇 ' if trial['name'] == best_name else ''}{trial['name']}</h2>
          <p><b>Overlay:</b> {trial['overlay_name']} &nbsp;|&nbsp;
             <b>Prompt:</b> {trial['prompt_name']} &nbsp;|&nbsp;
             <b>Unique:</b> {m['n_unique']} &nbsp;|&nbsp;
             <b>Entropy:</b> {m['entropy_nats']:.3f} &nbsp;|&nbsp;
             <b>Forbidden:</b> {m['forbidden_rate']*100:.1f}%</p>
          <details><summary>All predicates ({len(m['all_predicates'])})</summary>
            <div class="pred-list">{preds_html}</div></details>
          <div class="image-grid">{cards}</div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>{css}</style></head><body>
<h1>{title}</h1>
<p>Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — {len(all_trials)} trials</p>

<h2>Summary Table (ranked by entropy × cleanness)</h2>
<table class="summary-table">
<tr><th>#</th><th>Overlay</th><th>Prompt</th><th>Unique</th><th>Entropy</th>
<th>Rels/img</th><th>Forbid%</th><th>Top5 Cov</th><th>Speed</th><th>Parse</th></tr>
{rows_html}
</table>

<h2>Per-Trial Details</h2>
{detail_html}
</body></html>"""

    outpath.write_text(html)
    return outpath


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Creative open-vocab prompt optimisation agent")
    parser.add_argument("--model", default="e2b", choices=list(MODEL_CHOICES.keys()))
    parser.add_argument("--n_images", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=0, help="Max trials (0 = all)")
    parser.add_argument("--token_budget", type=int, default=140)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--greedy", action="store_true", default=True)
    parser.add_argument("--sampling", dest="greedy", action="store_false")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--outdir", default="runs/creative_agent")
    parser.add_argument("--no_flash", action="store_true")
    parser.add_argument("--quant4", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_id = MODEL_CHOICES[args.model]

    ideas = _make_ideas()
    if args.rounds > 0:
        ideas = ideas[:args.rounds]

    W = 60
    print("=" * W)
    print("  Creative Prompt Optimisation Agent")
    print("=" * W)
    print(f"  Model: {model_id}")
    print(f"  Images: {args.n_images}")
    print(f"  Trials: {len(ideas)}")
    print(f"  Budget: {args.token_budget} tok, greedy={args.greedy}")
    est = args.n_images * len(ideas) * 5
    print(f"  Est time: ~{est/60:.0f} min")
    print(f"  Output: {outdir}")
    print("=" * W)

    # 1. Load samples
    print(f"\n[1] Loading {args.n_images} samples …")
    samples = build_matched_samples(args.n_images)
    print(f"  Got {len(samples)} images")

    with open(outdir / "image_list.json", "w") as f:
        json.dump([{"img_id": s["img_id"], "file_name": s["file_name"]} for s in samples], f, indent=2)

    # 2. Load model
    print(f"\n[2] Loading model …")
    import prompt_eval
    prompt_eval._ACTIVE_MODEL_ID = model_id
    processor, model = load_model(use_flash=not args.no_flash, model_id=model_id, quant4=args.quant4)

    # 3. Warm-up
    print("\n[3] Warm-up …")
    run_one(processor, model, samples[0],
            overlay_fn=overlay_point_only, prompt_template=PROMPT_OPENVOC_CONCISE,
            token_budget=args.token_budget, max_new_tokens=args.max_new_tokens,
            greedy=args.greedy, temperature=args.temperature, model_id=model_id)
    print("  Done.\n")

    # 4. Run trials
    all_trials = []
    t_start = time.perf_counter()

    for ti, idea in enumerate(ideas):
        print(f"\n{'─'*W}")
        print(f"  [{ti+1}/{len(ideas)}] {idea['name']}")
        print(f"  overlay={idea['overlay_name']}  prompt={idea['prompt_name']}")
        print(f"{'─'*W}")

        results = []
        for si, sample in enumerate(samples):
            try:
                elapsed, sg, raw, ann_img = run_one(
                    processor, model, sample,
                    overlay_fn=idea["overlay_fn"],
                    prompt_template=idea["prompt_template"],
                    token_budget=args.token_budget,
                    max_new_tokens=args.max_new_tokens,
                    greedy=args.greedy,
                    temperature=args.temperature,
                    model_id=model_id,
)
            except Exception as exc:
                print(f"    [{si+1}] ERROR: {exc}")
                results.append({"elapsed": 0, "sg": None, "raw": str(exc),
                                "ann_img": samples[si]["pil_image"], "sample": sample})
                continue

            n_rel = len(sg["relations"]) if sg else -1
            results.append({"elapsed": elapsed, "sg": sg, "raw": raw,
                            "ann_img": ann_img, "sample": sample})
            if si < 3 or si == len(samples) - 1:
                print(f"    [{si+1}/{len(samples)}] {elapsed:.1f}s  {n_rel} rels")

        metrics = quick_metrics(results)
        print(f"  → unique={metrics['n_unique']}  entropy={metrics['entropy_nats']:.3f}"
              f"  forbidden={metrics['forbidden_rate']*100:.1f}%  rels/img={metrics['rels_per_img']}")

        all_trials.append({
            "name": idea["name"],
            "overlay_name": idea["overlay_name"],
            "prompt_name": idea["prompt_name"],
            "metrics": metrics,
            "results": results,
        })

        # Save per-trial JSON (without images)
        trial_dir = outdir / idea["name"]
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_json = {
            "name": idea["name"],
            "overlay": idea["overlay_name"],
            "prompt": idea["prompt_name"],
            "metrics": {k: v for k, v in metrics.items() if k != "all_predicates"},
            "predictions": [
                {"img_id": r["sample"]["img_id"], "elapsed": round(r["elapsed"], 2),
                 "sg": r["sg"]}
                for r in results
            ],
        }
        with open(trial_dir / "trial.json", "w") as f:
            json.dump(trial_json, f, indent=2)

        # Save annotated images for this trial
        for vi, r in enumerate(results[:5]):
            try:
                r["ann_img"].save(trial_dir / f"annotated_{vi+1}.jpg", quality=90)
            except Exception:
                pass

    total_s = time.perf_counter() - t_start
    print(f"\n\n{'='*W}")
    print(f"  ALL DONE — {len(all_trials)} trials in {total_s/60:.1f} min")
    print(f"{'='*W}")

    # 5. Summary table
    ranked = sorted(all_trials,
                    key=lambda t: t["metrics"]["entropy_nats"] * (1 - t["metrics"]["forbidden_rate"]),
                    reverse=True)
    print(f"\n{'Overlay':<20} {'Prompt':<22} {'Uniq':>5} {'Entropy':>8} {'R/img':>6} "
          f"{'Forbid%':>8} {'Top5cov':>8}")
    print("─" * 80)
    for t in ranked:
        m = t["metrics"]
        print(f"{t['overlay_name']:<20} {t['prompt_name']:<22} {m['n_unique']:>5} "
              f"{m['entropy_nats']:>8.3f} {m['rels_per_img']:>6.1f} "
              f"{m['forbidden_rate']*100:>7.1f}% {m['top5_coverage']*100:>7.1f}%")

    # 6. HTML report
    report_path = outdir / "report.html"
    render_html_report(all_trials, report_path)
    print(f"\n  HTML report → {report_path}")

    # 7. JSON summary
    summary = {
        "model_id": model_id,
        "n_images": len(samples),
        "n_trials": len(all_trials),
        "total_time_s": round(total_s, 1),
        "timestamp": datetime.now().isoformat(),
        "ranking": [
            {
                "rank": i + 1,
                "name": t["name"],
                "overlay": t["overlay_name"],
                "prompt": t["prompt_name"],
                "unique": t["metrics"]["n_unique"],
                "entropy": t["metrics"]["entropy_nats"],
                "forbidden_rate": t["metrics"]["forbidden_rate"],
                "rels_per_img": t["metrics"]["rels_per_img"],
                "top5_coverage": t["metrics"]["top5_coverage"],
                "top5": t["metrics"]["top5"],
            }
            for i, t in enumerate(ranked)
        ],
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary   → {outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
