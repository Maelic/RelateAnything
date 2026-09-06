#!/usr/bin/env python3
"""
prompt_eval.py — Evaluate a Gemma 4 annotation prompt + generation config.

Metrics
-------
  speed: median s/image  (hard constraint: pass if < --max_time_s)
  pred_entropy: Shannon H of predicate distribution          (↑ better)
  top5_coverage: fraction covered by top-5 predicates         (↓ better)
  forbidden_rate: fraction of rels using forbidden predicates  (↓ better)
  parse_ok_rate: fraction of images with valid JSON output     (↑ better)
  clip_score: mean CLIP cosine-sim(triplet_text, image)    (↑ better)

Fitness (composite, [0..1])
---------------------------
  fitness = (0.35·H_norm + 0.35·CLIP_norm + 0.15·(1−top5_cov) + 0.15·(1−forbidden))
            × int(median_s < max_time_s)

  H_norm    = min(1, H / ln(50))                      — normalised Shannon entropy
  CLIP_norm = clamp((μ_clip − 0.18) / 0.17, 0, 1)    — empirical [0.18, 0.35] range

Outputs (written to --outdir)
------------------------------
  fitness.json   — all metrics + fitness score (read by prompt_optim_agent)
  viz/           — N viz panels (annotated image + predictions + GT)
  pred_freq.tsv  — predicate frequency table

Usage
-----
  python datagen/prompt_eval.py --n_eval 50 --n_viz 5 --outdir runs/eval_v1
  python datagen/prompt_eval.py --prompt_file datagen/prompts/strict_v2.txt \\
      --token_budget 280 --greedy --n_eval 50 --outdir runs/eval_v2
"""
from __future__ import annotations

import os

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
MODEL_ID    = "google/gemma-4-E4B-it"
MODEL_26B_ID = "google/gemma-4-26B-A4B-it" #google/gemma-4-26B-A4B-it
MODEL_CHOICES = {
    "e4b":  MODEL_ID,
    "26b":  MODEL_26B_ID,
    "e2b":  "google/gemma-4-E2B-it",
    "31b":  "google/gemma-4-31B-it",
}
# Dataset locations are site-specific: set RA_DATASETS (or the per-dataset
# vars below) to wherever the COCO-format roots live on this machine.
DATASETS_DIR = os.environ.get("RA_DATASETS", os.environ.get("DATASETS_DIR", "datasets"))
MEGASG_DIR  = os.environ.get("MEGASG_DIR", f"{DATASETS_DIR}/MEGASG")
ANNO_PATH   = f"{MEGASG_DIR}/val/_annotations.coco.json"
IMG_DIR     = f"{MEGASG_DIR}/val"

DATASET_ROOTS = {
    "megasg": f"{MEGASG_DIR}/val",
    "vg150":  os.environ.get("VG150_DIR", f"{DATASETS_DIR}/VG150/VG150_coco_format") + "/val",
    "psg":    os.environ.get("PSG_DIR", f"{DATASETS_DIR}/PSG/coco_format") + "/val",
}

FORBIDDEN_PREDS = frozenset({
    "near", "next to", "with", "has", "beside", "and", "same scene",
    "is near", "is next to",
})

# Image marking modes (search axis for the optimizer agent)
# bbox: current default — coloured border + light fill + centred number + label tag
# som: Set-of-Mark style — solid 55% opacity fill + large centred number; no outline
# point: centroid dot + number only; zero visual clutter
# bbox_no_label: bbox + number but no text label tag
# raw: pass raw unmodified image; rely on text-only object list
IMAGE_MODES = ("bbox", "som", "point", "point_fixed", "bbox_no_label", "raw")

MAX_TIME_S    = 10.0   # hard speed constraint
CLIP_LO       = 0.18   # empirical lower anchor for CLIP score normalisation
CLIP_HI       = 0.35   # empirical upper anchor for CLIP score normalisation
H_NORM_N      = 50     # normalise H against ln(50) ≈ 3.91 nats

# Fitness weights (must sum to 1.0)
W_ENTROPY    = 0.35
W_CLIP       = 0.35
W_DIVERSITY  = 0.15    # 1 − top5_coverage
W_CLEAN      = 0.15    # 1 − forbidden_rate


# ── Default prompt (matches gemma4_speed_benchmark.py) ────────────────────────
DEFAULT_PROMPT = """\
The image has {n} annotated objects, each marked with a large bold number inside a coloured bounding box.

Objects present (number → label):
{object_list}

Task: Identify ALL meaningful visual relations between these {n} objects.

Strict rules:
1. ONLY use object IDs 1–{n}. Never invent objects outside this list.
2. Be EXHAUSTIVE — examine every directed pair. Aim to describe the image fully.
3. Predicate priority (most specific that clearly applies):
   Actions: riding, holding, wearing, sitting on, carrying, eating, pulling,
               pushing, climbing, playing with, driving, using, touching, kicking,
               throwing, catching, looking at, walking toward, standing on, lying on
   Spatial: above, below, in front of, behind, to the left of, to the right of,
               inside, on top of, hanging from, attached to, leaning against, overlapping
   FORBIDDEN: near, next to, with, has, beside, and, same scene  (too vague)
4. Only output a single JSON code block — no other text.

```json
{{
  "scene_description": "<one sentence: who is doing what>",
  "relations": [
    {{"subject_id": <int 1-{n}>, "subject_label": "<str>", "predicate": "<str>", "object_id": <int 1-{n}>, "object_label": "<str>"}}
  ]
}}
```"""


# ── COCO loader ────────────────────────────────────────────────────────────────
def _load_coco_samples(
    n: int,
    anno_path: str = ANNO_PATH,
    img_dir: str = IMG_DIR,
    filter_to_rel_objects: bool = False,
) -> list[dict]:
    """Return up to n samples with ≥1 relation from COCO-format annotations.

    Each sample dict:
      img_id, file_name, pil_image, objects [{id, label, bbox}], gt_relations [(s, p, o)]

    filter_to_rel_objects: if True, drop objects not referenced by any GT relation.
      Dramatically reduces object count on VG150 (~38%) and PSG (~47%) where many
      annotated objects have no associated relation.
    """
    with open(anno_path) as f:
        data = json.load(f)

    cat_id2name  = {c["id"]: c["name"] for c in data["categories"]}
    pred_id2name = {c["id"]: c["name"] for c in data["rel_categories"]}

    ann_by_img: dict = defaultdict(list)
    ann_by_id:  dict = {}
    for a in data["annotations"]:
        ann_by_img[a["image_id"]].append(a)
        ann_by_id[a["id"]] = a

    rel_by_img: dict = defaultdict(list)
    for r in data["rel_annotations"]:
        rel_by_img[r["image_id"]].append(r)

    results = []
    for img_meta in data["images"]:
        if len(results) >= n:
            break
        img_id    = img_meta["id"]
        file_name = img_meta["file_name"]
        if not rel_by_img.get(img_id):
            continue
        img_path = Path(img_dir) / file_name
        if not img_path.exists():
            continue

        img_rels = rel_by_img[img_id]
        rel_ann_ids = (
            {r["subject_id"] for r in img_rels} | {r["object_id"] for r in img_rels}
            if filter_to_rel_objects else None
)

        objects = []
        for a in ann_by_img.get(img_id, []):
            if rel_ann_ids is not None and a["id"] not in rel_ann_ids:
                continue
            x, y, w, h = a["bbox"]
            objects.append({
                "id":    a["id"],
                "label": cat_id2name[a["category_id"]],
                "bbox":  [int(x), int(y), int(x + w), int(y + h)],
            })

        gt_relations = []
        for r in rel_by_img[img_id]:
            s = ann_by_id.get(r["subject_id"])
            o = ann_by_id.get(r["object_id"])
            if s and o:
                gt_relations.append((
                    cat_id2name[s["category_id"]],
                    pred_id2name[r["predicate_id"]],
                    cat_id2name[o["category_id"]],
))

        results.append({
            "img_id":       img_id,
            "file_name":    file_name,
            "pil_image":    Image.open(img_path).convert("RGB"),
            "objects":      objects,
            "gt_relations": gt_relations,
        })

    return results


# ── Image annotation — multi-mode dispatcher ──────────────────────────────────
_HEX = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def _get_fonts():
    from PIL import ImageFont
    try:
        big   = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        big = small = ImageFont.load_default()
    return big, small


def annotate_image(pil: Image.Image, objects: list, mode: str = "bbox") -> Image.Image:
    """
    Mark ground-truth object regions on the image so the model can ground IDs.

    Modes
    -----
    bbox          Current default: thick coloured border + 19% fill + large centred
                  number + label tag at top-left corner.
    som           Set-of-Mark (Yang et al. 2023): solid 55%-opacity coloured fill
                  + large centred number; no border, no label tag. The high-opacity
                  fill resembles the segmentation-mask overlay from the original SoM
                  paper (we approximate it from COCO bboxes instead of SAM masks).
    point         Centroid only: filled coloured circle (r≈10 px) + number label
                  just to the right. Zero bounding-box noise — useful to test whether
                  the model grounds on spatial position rather than region extent.
    point_fixed   Same as point, but uses a constant radius for every object in
                  every image (r=10 px). Useful for ablation against size-scaled
                  points.
    bbox_no_label Like bbox but no text label tag. Tests whether label information
                  in the prompt text alone is sufficient.
    raw           Unmodified image. Object list is text-only. Baseline for ablation.
    """
    if mode == "raw":
        return pil.copy()

    from PIL import ImageDraw
    font_big, font_label = _get_fonts()

    img  = pil.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = obj["bbox"]
        color = _HEX[i % len(_HEX)]
        idx   = str(i + 1)

        if mode == "som":
            # Solid fill at 55% opacity — emulates segmentation mask overlay
            draw.rectangle([x1, y1, x2, y2], fill=color + "8c")  # 0x8c ≈ 55%
            # Large bold number, centred
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bb = draw.textbbox((0, 0), idx, font=font_big)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((cx - tw // 2, cy - th // 2), idx,
                      font=font_big, fill="white", stroke_width=3, stroke_fill="black")

        elif mode == "point":
            # Filled circle at bbox centroid + number label to the right
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = max(8, min(14, (x2 - x1 + y2 - y1) // 20))
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + "ff",
                         outline="white", width=2)
            bb = draw.textbbox((0, 0), idx, font=font_big)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((cx + r + 4, cy - th // 2), idx,
                      font=font_big, fill=color, stroke_width=2, stroke_fill="white")

        elif mode == "point_fixed":
            # Same centroid marker as point mode, but with constant radius.
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            r = 10
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + "ff",
                         outline="white", width=2)
            bb = draw.textbbox((0, 0), idx, font=font_big)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((cx + r + 4, cy - th // 2), idx,
                      font=font_big, fill=color, stroke_width=2, stroke_fill="white")

        elif mode in ("bbox", "bbox_no_label"):
            # Thick coloured border
            for t in range(3):
                draw.rectangle([x1-t, y1-t, x2+t, y2+t], outline=color)
            # Light fill
            draw.rectangle([x1, y1, x2, y2], fill=color + "30")  # 19% opacity
            # Large centred number
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bb = draw.textbbox((0, 0), idx, font=font_big)
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
            draw.text((cx - tw // 2, cy - th // 2), idx,
                      font=font_big, fill="white", stroke_width=2, stroke_fill="black")
            # Label tag (bbox mode only) — compound key so model can unambiguously
            # map the visual anchor back to an object ID
            if mode == "bbox":
                tag = f" {idx}_{obj['label']} "
                tb  = draw.textbbox((0, 0), tag, font=font_label)
                tw2, th2 = tb[2] - tb[0], tb[3] - tb[1]
                ty = max(y1 - th2 - 2, 0)
                draw.rectangle([x1, ty, x1 + tw2, ty + th2], fill=color + "cc")
                draw.text((x1, ty), tag, font=font_label, fill="white")

    return img


# Keep old name as alias so imports from gemma4_sample_viz still work
_annotate_image = annotate_image


# ── JSON extraction (truncation-tolerant) ──────────────────────────────────────
def _extract_json(text: str) -> dict | None:
    for pat in [r"```json\s*\n(.*?)\n?\s*```", r"```\s*\n(\{.*?\})\s*```"]:
        m = re.search(pat, text, re.S)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                candidate = m.group(1)
                break
    else:
        m = re.search(r"(\{.*\})", text, re.S)
        candidate = m.group(1) if m else text
        if m:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

    rel_pat = re.compile(
        r'\{\s*"subject_id"\s*:\s*(\d+).*?"predicate"\s*:\s*"([^"]+)".*?"object_id"\s*:\s*(\d+).*?\}',
        re.S,
)
    rels = []
    for rm in rel_pat.finditer(candidate):
        try:
            rels.append(json.loads(candidate[rm.start(): rm.end()]))
        except json.JSONDecodeError:
            rels.append({
                "subject_id":    int(rm.group(1)), "subject_label": "",
                "predicate":     rm.group(2),
                "object_id":     int(rm.group(3)), "object_label": "",
            })
    if rels:
        return {"scene_description": "", "relations": rels, "_truncated": True}
    return None


# ── Prompt helpers ─────────────────────────────────────────────────────────────
def _object_list(objects: list) -> str:
    # Compound key «N_label» mirrors the label tag drawn on the image so the model
    # can unambiguously match what it sees to the numbered region.
    return "\n".join(f"  {i+1}_{obj['label']}" for i, obj in enumerate(objects))


def _format_prompt(template: str, objects: list, max_rels: int | None = None) -> str:
    prompt = template.format(n=len(objects), object_list=_object_list(objects))
    if max_rels is not None:
        prompt += f"\n\nIMPORTANT: Output at most {max_rels} relations. Prioritise the most specific and grounded ones."
    return prompt


# ── Model loading ──────────────────────────────────────────────────────────────
def load_model(
    use_flash: bool = False,
    use_compile: bool = False,
    model_id: str = MODEL_ID,
    quant4: bool = False,
    no_offload: bool = False,
    load_drafter: bool = False,
):
    """Load Gemma4 target model (and optionally the MTP drafter for speculative decoding).

    When load_drafter=True, also loads google/gemma-4-E4B-it-assistant (78.8M params)
    in bfloat16. The drafter is returned as the third element of the tuple.
    Pass it as assistant_model= to run_batch_raw() for ~2x speedup (requires greedy=True).

    Returns: (processor, model) or (processor, model, drafter) if load_drafter=True.
    """
    from transformers import Gemma4ForConditionalGeneration, Gemma4Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    # torch.compile is incompatible with flash_attention_2 custom ops
    # 4-bit quantization via bitsandbytes also requires sdpa (not flash)
    attn = "sdpa" if (use_compile or not use_flash or quant4) else "flash_attention_2"
    quant_str = " 4-bit NF4" if quant4 else " bfloat16"
    print(f"  Loading {model_id}  (attn={attn},{quant_str}) …")
    t0 = time.time()

    # device_map="auto" may offload layers to CPU/meta on tight VRAM; allow
    # forcing full-GPU placement for clean model-to-model comparisons.
    load_kwargs: dict = {
        "device_map": ({"": "cuda:0"} if no_offload else "auto"),
        "attn_implementation": attn,
    }
    if quant4:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
)
        # dtype is inferred from BitsAndBytesConfig; don't pass torch_dtype
        # Prevent accelerate from offloading layers to CPU/disk, which bitsandbytes
        # 4-bit does not support.  Use all available GPU memory (minus 2 GiB headroom)
        # and explicitly forbid CPU offloading.
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
            load_kwargs["max_memory"] = {0: f"{gpu_mem_gb - 2}GiB", "cpu": "0GiB"}
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16

    processor = Gemma4Processor.from_pretrained(model_id)
    model = Gemma4ForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
    model.eval()
    if use_compile:
        if quant4:
            print("  torch.compile skipped: incompatible with 4-bit quantization")
        else:
            on_cpu = any(p.device.type in ("meta", "cpu") for p in model.parameters())
            if on_cpu:
                print("  torch.compile skipped: CPU-offloaded layers detected (need full GPU fit)")
            else:
                print("  torch.compile(mode='reduce-overhead') …")
                torch._dynamo.config.capture_scalar_outputs = True
                model.model = torch.compile(model.model, mode="reduce-overhead", fullgraph=False)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    if not load_drafter:
        return processor, model

    # ── MTP drafter (speculative decoding) ────────────────────────────────────
    drafter_id = model_id + "-assistant"
    print(f"  Loading MTP drafter {drafter_id} (78.8M, bfloat16) …")
    t1 = time.time()
    from transformers import AutoModelForCausalLM
    drafter = AutoModelForCausalLM.from_pretrained(
        drafter_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
)
    drafter.eval()
    # Dynamic draft-token schedule: starts at 4, adjusts based on acceptance rate
    drafter.generation_config.num_assistant_tokens = 4
    drafter.generation_config.num_assistant_tokens_schedule = "heuristic"
    print(f"  Drafter loaded in {time.time() - t1:.1f}s")
    return processor, model, drafter


# ── Generation config cache ────────────────────────────────────────────────────
_GEN_CFG_CACHE: dict = {}


def _gen_cfg(max_new_tokens: int, greedy: bool, temperature: float = 1.0, model_id: str = MODEL_ID):
    from transformers import GenerationConfig

    key = (max_new_tokens, greedy, temperature, model_id)
    if key not in _GEN_CFG_CACHE:
        cfg = GenerationConfig.from_pretrained(model_id)
        cfg.max_new_tokens = max_new_tokens
        if greedy:
            cfg.do_sample   = False
            cfg.temperature = None
            cfg.top_p       = None
            cfg.top_k       = None
        else:
            cfg.do_sample   = True
            cfg.temperature = temperature
            cfg.top_p       = 0.95
            cfg.top_k       = 64
        _GEN_CFG_CACHE[key] = cfg
    return _GEN_CFG_CACHE[key]


_ACTIVE_MODEL_ID: str = MODEL_ID  # set at load time; used by _gen_cfg calls


# ── Single-image inference ─────────────────────────────────────────────────────
@torch.inference_mode()
def run_one(
    processor,
    model,
    sample: dict,
    *,
    prompt_template: str,
    token_budget: int,
    max_new_tokens: int,
    greedy: bool,
    temperature: float = 1.0,
    image_mode: str = "bbox",
    max_rels: int | None = None,
    max_objects: int | None = None,
) -> tuple[float, dict | None, str]:
    """Return (elapsed_s, scene_graph_or_None, raw_text)."""
    objects = sample["objects"]
    if max_objects is not None and len(objects) > max_objects:
        objects = sorted(
            objects,
            key=lambda o: (o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]),
            reverse=True,
)[:max_objects]
    processor.image_processor.max_soft_tokens = token_budget
    gen_cfg  = _gen_cfg(max_new_tokens, greedy, temperature, model_id=_ACTIVE_MODEL_ID)
    ann_img  = annotate_image(sample["pil_image"], objects, mode=image_mode)
    prompt   = _format_prompt(prompt_template, objects, max_rels=max_rels)

    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": prompt},
    ]}]
    text   = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
    inputs    = processor(text=text, images=[ann_img], return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0  = time.perf_counter()
    out = model.generate(**inputs, generation_config=gen_cfg)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    raw = processor.decode(out[0][input_len:], skip_special_tokens=True)
    sg  = _extract_json(raw)
    return elapsed, sg, raw


# ── CLIP scoring (CPU; lazy-loaded to avoid OOM alongside Gemma4) ──────────────
_CLIP_MODEL = None
_CLIP_PROC  = None


def _ensure_clip():
    global _CLIP_MODEL, _CLIP_PROC
    if _CLIP_MODEL is None:
        from transformers import CLIPModel, CLIPProcessor

        print("  Loading CLIP-B/32 on CPU for semantic scoring …", flush=True)
        _CLIP_PROC  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_MODEL = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _CLIP_MODEL.eval()
    return _CLIP_MODEL, _CLIP_PROC


@torch.no_grad()
def clip_score_image(image: Image.Image, relations: list[dict]) -> float:
    """Mean cosine-sim('{subj} {pred} {obj}', image) over all relations."""
    if not relations:
        return 0.0
    model, proc = _ensure_clip()
    texts = [
        f"a photo of {r.get('subject_label','')} {r.get('predicate','')} {r.get('object_label','')}"
        for r in relations
    ]
    img_inp  = proc(images=image, return_tensors="pt")
    img_feat = model.get_image_features(**img_inp)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)   # (1, D)

    txt_inp  = proc(text=texts, return_tensors="pt",
                    padding=True, truncation=True, max_length=77)
    txt_feat = model.get_text_features(**txt_inp)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)   # (T, D)

    sims = (txt_feat @ img_feat.T).squeeze(-1)   # (T,)
    return sims.mean().item()


# ── Metrics ────────────────────────────────────────────────────────────────────
def _is_forbidden(pred: str) -> bool:
    p = pred.lower().strip()
    return p in FORBIDDEN_PREDS


def compute_metrics(
    results: list[dict],   # [{elapsed, sg, raw, sample}]
    max_time_s: float = MAX_TIME_S,
    skip_clip: bool = False,
) -> dict:
    """Compute all quality metrics from a list of per-image inference results."""
    times = [r["elapsed"] for r in results]
    n     = len(results)

    # ── Speed ─────────────────────────────────────────────────────────────────
    med_s = statistics.median(times)
    p90_s = sorted(times)[max(int(0.9 * n) - 1, 0)]

    # ── Parse / truncation ────────────────────────────────────────────────────
    parse_ok       = [r for r in results if r["sg"] is not None]
    parse_ok_rate  = len(parse_ok) / n
    truncated      = [r for r in results if r["sg"] and r["sg"].get("_truncated")]
    truncated_rate = len(truncated) / n

    # ── Predicate distribution ─────────────────────────────────────────────────
    all_rels   = []
    pred_count = Counter()
    for r in parse_ok:
        rels = r["sg"].get("relations", [])
        all_rels.extend(rels)
        for rel in rels:
            p = rel.get("predicate", "").lower().strip()
            if p:
                pred_count[p] += 1

    total_rels = len(all_rels)
    n_unique   = len(pred_count)
    rels_per_img_mean   = total_rels / n
    rels_per_img_median = statistics.median(
        [len(r["sg"].get("relations", [])) if r["sg"] else 0 for r in results]
)

    if total_rels > 0:
        probs        = [c / total_rels for c in pred_count.values()]
        entropy_nats = -sum(p * math.log(p) for p in probs if p > 0)
    else:
        entropy_nats = 0.0
    entropy_score = min(1.0, entropy_nats / math.log(H_NORM_N))

    top5_count     = sum(c for _, c in pred_count.most_common(5))
    top10_count    = sum(c for _, c in pred_count.most_common(10))
    top5_coverage  = top5_count  / max(total_rels, 1)
    top10_coverage = top10_count / max(total_rels, 1)
    top5_list      = [(p, round(c / max(total_rels, 1), 4))
                      for p, c in pred_count.most_common(5)]

    n_forbidden    = sum(1 for r in all_rels if _is_forbidden(r.get("predicate", "")))
    forbidden_rate = n_forbidden / max(total_rels, 1)

    # ── CLIP scoring ───────────────────────────────────────────────────────────
    if skip_clip:
        clip_mean, clip_std, clip_norm = 0.0, 0.0, 0.0
        n_scored = 0
    else:
        print("  Computing CLIP scores …", flush=True)
        clip_scores = []
        for r in results:
            if r["sg"] and r["sg"].get("relations"):
                clip_scores.append(
                    clip_score_image(r["sample"]["pil_image"], r["sg"]["relations"])
)
        n_scored  = len(clip_scores)
        clip_mean = statistics.mean(clip_scores) if clip_scores else 0.0
        clip_std  = statistics.stdev(clip_scores) if len(clip_scores) > 1 else 0.0
        clip_norm = max(0.0, min(1.0, (clip_mean - CLIP_LO) / (CLIP_HI - CLIP_LO)))

    # ── Fitness ────────────────────────────────────────────────────────────────
    speed_ok       = int(med_s < max_time_s)
    entropy_term   = W_ENTROPY   * entropy_score
    clip_term      = W_CLIP      * clip_norm
    diversity_term = W_DIVERSITY * (1.0 - top5_coverage)
    clean_term     = W_CLEAN     * (1.0 - forbidden_rate)
    fitness        = speed_ok * (entropy_term + clip_term + diversity_term + clean_term)

    return {
        "speed": {
            "median_s":          round(med_s, 3),
            "mean_s":            round(statistics.mean(times), 3),
            "p90_s":             round(p90_s, 3),
            "passes_constraint": bool(speed_ok),
            "max_time_s":        max_time_s,
        },
        "predicate_distribution": {
            "entropy_nats":        round(entropy_nats, 4),
            "entropy_score":       round(entropy_score, 4),
            "n_unique_predicates": n_unique,
            "top5_coverage":       round(top5_coverage, 4),
            "top10_coverage":      round(top10_coverage, 4),
            "top5_list":           top5_list,
            "rels_per_img_mean":   round(rels_per_img_mean, 2),
            "rels_per_img_median": rels_per_img_median,
        },
        "clip": {
            "mean":       round(clip_mean, 4),
            "std":        round(clip_std, 4),
            "normalized": round(clip_norm, 4),
            "n_scored":   n_scored,
            "skipped":    skip_clip,
        },
        "quality": {
            "parse_ok_rate":   round(parse_ok_rate, 4),
            "forbidden_rate":  round(forbidden_rate, 4),
            "truncated_rate":  round(truncated_rate, 4),
            "n_total_rels":    total_rels,
            "n_forbidden_rels": n_forbidden,
        },
        "fitness": round(fitness, 4),
        "fitness_breakdown": {
            "entropy_term":    round(entropy_term, 4),
            "clip_term":       round(clip_term, 4),
            "diversity_term":  round(diversity_term, 4),
            "clean_term":      round(clean_term, 4),
            "speed_penalty":   not bool(speed_ok),
        },
    }


# ── Visualization ──────────────────────────────────────────────────────────────
def draw_viz_panel(
    sample: dict,
    sg: dict | None,
    elapsed: float,
    idx: int,
    outdir: Path,
    image_mode: str = "bbox",
    dataset_label: str = "MEGASG",
) -> Path:
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ann_img = annotate_image(sample["pil_image"], sample["objects"], mode=image_mode)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle(
        f"{sample['file_name']}  ({len(sample['objects'])} objects) — mode={image_mode}",
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
            sid  = r.get("subject_id", "?")
            slbl = r.get("subject_label", "?")
            pred = r.get("predicate", "?")
            oid  = r.get("object_id", "?")
            olbl = r.get("object_label", "?")
            flag = "  ⚠" if _is_forbidden(pred) else ""
            lines.append(f"• {sid}_{slbl} –[{pred}]→ {oid}_{olbl}{flag}")
        if sg.get("_truncated"):
            lines.append("\n(output truncated)")
    else:
        lines.append("(parse failed)")

    lines.append(f"\n── {dataset_label} ground truth ──\n")
    for s, p, o in sample.get("gt_relations", []):
        lines.append(f"• {s} –[{p}]→ {o}")

    axes[1].text(
        0.05, 0.97, "\n".join(lines),
        transform=axes[1].transAxes,
        fontsize=8, verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)
    axes[1].axis("off")

    plt.tight_layout()
    fname    = f"viz_{idx:02d}_{Path(sample['file_name']).stem}.png"
    out_path = outdir / fname
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ── Summary printer ────────────────────────────────────────────────────────────
def print_summary(metrics: dict, config: dict) -> None:
    W = 58
    print("\n" + "=" * W)
    print("  EVAL SUMMARY")
    print("=" * W)
    lbl = config.get("prompt_label", "(default)")
    print(f"  Prompt: {lbl}")
    print(
        f"  Budget: {config['token_budget']} tok  "
        f"greedy={config['greedy']}  "
        f"max_new={config['max_new_tokens']}"
)
    print("-" * W)
    s  = metrics["speed"]
    ok = "✓" if s["passes_constraint"] else "✗ FAIL"
    print(
        f"  Speed: {s['median_s']:.2f}s median  "
        f"{s['p90_s']:.2f}s p90   "
        f"[{ok} < {config['max_time_s']}s]"
)
    pd = metrics["predicate_distribution"]
    print(
        f"  Entropy: {pd['entropy_nats']:.3f} nats  "
        f"(score={pd['entropy_score']:.2f})  "
        f"{pd['n_unique_predicates']} unique"
)
    print(
        f"  Top-5 cov: {pd['top5_coverage']*100:.1f}%   "
        f"top-10: {pd['top10_coverage']*100:.1f}%"
)
    print(
        f"  Rels/img: {pd['rels_per_img_mean']:.1f} mean  "
        f"{pd['rels_per_img_median']:.0f} median"
)
    c = metrics["clip"]
    if c["skipped"]:
        print("  CLIP score: skipped (--skip_clip)")
    else:
        print(
            f"  CLIP score: {c['mean']:.4f} ± {c['std']:.4f}  "
            f"(norm={c['normalized']:.2f}, n={c['n_scored']})"
)
    q = metrics["quality"]
    print(
        f"  Parse OK: {q['parse_ok_rate']*100:.1f}%   "
        f"forbidden: {q['forbidden_rate']*100:.1f}%   "
        f"truncated: {q['truncated_rate']*100:.1f}%"
)
    print(f"  Top-5 preds: {pd['top5_list']}")
    print("-" * W)
    fb = metrics["fitness_breakdown"]
    print(
        f"  Fitness: \033[1m{metrics['fitness']:.4f}\033[0m"
        f"  (H={fb['entropy_term']:.3f}  "
        f"CLIP={fb['clip_term']:.3f}  "
        f"Div={fb['diversity_term']:.3f}  "
        f"Clean={fb['clean_term']:.3f})"
)
    if fb["speed_penalty"]:
        print("  ⚠  Speed constraint violated — fitness zeroed out")
    print("=" * W)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file",    type=str, default=None,
                        help="Prompt template file. Placeholders: {n}, {object_list}.")
    parser.add_argument("--token_budget",   type=int, default=280)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    # --greedy is the default; --sampling overrides it
    parser.add_argument("--greedy",   dest="greedy", action="store_true",  default=True)
    parser.add_argument("--sampling", dest="greedy", action="store_false")
    parser.add_argument("--temperature",    type=float, default=1.0)
    parser.add_argument("--n_eval",         type=int,   default=50,
                        help="Images to run inference on for metrics")
    parser.add_argument("--n_viz",          type=int,   default=5,
                        help="Images to save side-by-side panels for")
    parser.add_argument("--outdir",         type=str,   default="runs/prompt_eval")
    parser.add_argument("--max_time_s",     type=float, default=MAX_TIME_S)
    parser.add_argument("--skip_clip",      action="store_true",
                        help="Skip CLIP scoring (faster iteration)")
    parser.add_argument("--no_flash",       action="store_true")
    parser.add_argument("--compile",        action="store_true",
                        help="torch.compile model.model (reduce-overhead); requires full GPU fit")
    parser.add_argument("--quant4",         action="store_true",
                        help="Load model in 4-bit NF4 quantization via bitsandbytes (~5 GB for E4B, ~15.6 GB for 26B)")
    parser.add_argument("--model",          type=str, default="e4b",
                        choices=list(MODEL_CHOICES.keys()),
                        help="Model variant: e4b (default, 15 GB), 26b (MoE, 48 GB), e2b, 31b")
    parser.add_argument("--image_mode",     type=str, default="bbox",
                        choices=list(IMAGE_MODES),
                        help="How to mark objects on the image before passing to model")
    parser.add_argument("--max_rels",       type=int, default=None,
                        help="Hint model to output at most N relations (appended to prompt)")
    parser.add_argument("--max_objects",    type=int, default=None,
                        help="Truncate object list to top-K by bbox area before inference. "
                             "Reduces prompt + visual clutter for dense scenes without capping output.")
    parser.add_argument("--filter_objects", action="store_true", default=False,
                        help="Drop objects not referenced by any GT relation before loading. "
                             "Reduces object count by ~38%% on VG150 and ~47%% on PSG.")
    parser.add_argument("--run_id",         type=str, default=None)
    parser.add_argument("--dataset",        type=str, default="megasg",
                        choices=list(DATASET_ROOTS.keys()),
                        help="Evaluation dataset: megasg (default), vg150, psg")
    args = parser.parse_args()

    outdir  = Path(args.outdir)
    viz_dir = outdir / "viz"
    outdir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(exist_ok=True)

    if args.prompt_file:
        prompt_template = Path(args.prompt_file).read_text()
        prompt_label    = Path(args.prompt_file).stem
    else:
        prompt_template = DEFAULT_PROMPT
        prompt_label    = "default"

    model_id = MODEL_CHOICES[args.model]

    dataset_dir  = DATASET_ROOTS[args.dataset]
    anno_path    = f"{dataset_dir}/_annotations.coco.json"
    img_dir      = dataset_dir

    config = {
        "run_id":         args.run_id or outdir.name,
        "prompt_label":   prompt_label,
        "prompt_file":    args.prompt_file,
        "token_budget":   args.token_budget,
        "max_new_tokens": args.max_new_tokens,
        "greedy":         args.greedy,
        "temperature":    args.temperature if not args.greedy else None,
        "image_mode":     args.image_mode,
        "max_rels":       args.max_rels,
        "max_objects":    args.max_objects,
        "filter_objects": args.filter_objects,
        "compile":        args.compile,
        "quant4":         args.quant4,
        "model_id":       model_id,
        "n_eval":         args.n_eval,
        "n_viz":          args.n_viz,
        "max_time_s":     args.max_time_s,
        "skip_clip":      args.skip_clip,
        "dataset":        args.dataset,
    }

    print("=" * 60)
    print("  SGG Annotation Quality Eval — Gemma 4")
    print("=" * 60)
    print(f"  Model: {model_id}{'  [4-bit NF4]' if args.quant4 else ''}")
    print(f"  Prompt: {prompt_label}")
    print(f"  Budget: {args.token_budget} tok  greedy={args.greedy}"
          f"  max_new={args.max_new_tokens}")
    print(f"  Image mode: {args.image_mode}  max_rels={args.max_rels}  max_objects={args.max_objects}  filter_objects={args.filter_objects}")
    print(f"  N eval: {args.n_eval}  N viz: {args.n_viz}")
    print(f"  Max time: {args.max_time_s}s / image")
    print(f"  CLIP: {'skipped' if args.skip_clip else 'enabled'}")
    print(f"  Compile: {args.compile}")
    print(f"  Out: {outdir}")

    global _ACTIVE_MODEL_ID
    _ACTIVE_MODEL_ID = model_id

    print("\n[1] Loading model …")
    processor, model = load_model(
        use_flash=not args.no_flash,
        use_compile=args.compile,
        model_id=model_id,
        quant4=args.quant4,
)

    n_load = max(args.n_eval, args.n_viz)
    print(f"\n[2] Loading {n_load} images from {args.dataset} val …")
    samples = _load_coco_samples(n_load, anno_path=anno_path, img_dir=img_dir,
                                 filter_to_rel_objects=args.filter_objects)
    if not samples:
        print("ERROR: no samples loaded.")
        return
    print(f"  Got {len(samples)} images.")

    print("\n[3] Warm-up pass …")
    run_one(processor, model, samples[0],
            prompt_template=prompt_template,
            token_budget=args.token_budget,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            image_mode=args.image_mode,
            max_rels=args.max_rels,
            max_objects=args.max_objects)

    eval_samples = samples[:args.n_eval]
    print(f"\n[4] Running inference on {len(eval_samples)} images …")
    results = []
    for i, sample in enumerate(eval_samples):
        elapsed, sg, raw = run_one(
            processor, model, sample,
            prompt_template=prompt_template,
            token_budget=args.token_budget,
            max_new_tokens=args.max_new_tokens,
            greedy=args.greedy,
            temperature=args.temperature,
            image_mode=args.image_mode,
            max_rels=args.max_rels,
            max_objects=args.max_objects,
)
        n_rel = len(sg["relations"]) if sg else -1
        print(
            f"  [{i+1:3d}/{len(eval_samples)}] {sample['file_name']:<40}"
            f"  {elapsed:.1f}s  {n_rel} rels",
            flush=True,
)
        results.append({"elapsed": elapsed, "sg": sg, "raw": raw, "sample": sample})

    print("\n[5] Computing metrics …")
    metrics             = compute_metrics(results, args.max_time_s, args.skip_clip)
    metrics["config"]    = config
    metrics["timestamp"] = datetime.now().isoformat()

    print(f"\n[6] Saving {args.n_viz} visualization panels …")
    for i, r in enumerate(results[:args.n_viz]):
        p = draw_viz_panel(r["sample"], r["sg"], r["elapsed"], i + 1, viz_dir,
                           image_mode=args.image_mode,
                           dataset_label=args.dataset.upper())
        print(f"  → {p}")
    metrics["viz_dir"] = str(viz_dir)

    # Predicate frequency table
    all_preds = Counter()
    for r in results:
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
            f.write(f"{pred}\t{cnt}\t{cnt/total_preds:.4f}\t{int(_is_forbidden(pred))}\n")
    print(f"  Predicate freq table → {freq_path}")

    fitness_path = outdir / "fitness.json"
    with open(fitness_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Fitness report       → {fitness_path}")

    # Per-image predictions — used by clip_compare.py for downstream CLIP eval
    predictions_path = outdir / "predictions.json"
    with open(predictions_path, "w") as f:
        json.dump({
            "config": config,
            "predictions": [
                {
                    "img_id":       r["sample"]["img_id"],
                    "file_name":    r["sample"]["file_name"],
                    "elapsed_s":    round(r["elapsed"], 3),
                    "objects":      r["sample"]["objects"],
                    "sg":           r["sg"],
                    "gt_relations": r["sample"]["gt_relations"],
                }
                for r in results
            ],
        }, f, indent=2)
    print(f"  Per-image predictions → {predictions_path}")

    print_summary(metrics, config)


if __name__ == "__main__":
    main()
