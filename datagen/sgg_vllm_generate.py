#!/usr/bin/env python3
"""
sgg_vllm_generate.py — production-scale SGG annotation with vLLM.

Built for the 500K-image run. The binding constraint at scale is aggregate
decode throughput, which vLLM (continuous batching + paged attention + fused
MoE kernels) delivers — the HF `.generate()` path used by sgg_trial.py is
10–20× slower and, on the 26B-A4B MoE, suffers a further unbatched-expert
penalty (measured ~69 s/img → unusable at 500K).

Two strategies (pick with --strategy):

  single        one image+prompt pass → JSON triples (cheapest, proven).
  caption_first CF-1/CF-2: an Observer pass captions the image, then a
                TEXT-ONLY Parser pass converts the caption to cited triples.
                The Parser has no pixels, so it cannot hallucinate; and being
                text-only it is the cheapest thing vLLM does, so caption_first
                costs only ~1.3× single (not 2×).

Sharding for SLURM job arrays: --shard_index / --num_shards strides the image
list so each GPU process annotates a disjoint, balanced subset. Output is
streamed to a per-shard JSONL (bounded memory, crash-safe, resumable).

This script depends ONLY on tracked/unchanged modules (prompt_eval,
creative_prompt_agent) so it ships as a clean, self-contained unit; the
caption-first parsing mirrors datagen/sgg_caption_first.py.

Example (calibration, 1 GPU):
  python datagen/sgg_vllm_generate.py --name calib --model 26b --quant fp8 \
      --strategy caption_first --split val --limit 1000 --emit_trial_json

Example (one shard of a 500K array job):
  python datagen/sgg_vllm_generate.py --name megasg_500k --model 26b --quant fp8 \
      --strategy single --split train --num_shards 64 --shard_index $SLURM_ARRAY_TASK_ID
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Bound allocator fragmentation; harmless under vLLM, helps any HF fallback.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent))

# Tracked + unchanged modules only → keeps this a clean, isolated deliverable.
from prompt_eval import MODEL_CHOICES, MEGASG_DIR, _extract_json, _is_forbidden
from creative_prompt_agent import (
    overlay_point_only,
    overlay_point_fixed,
    overlay_bbox_only,
    overlay_contour_only,
)
from sgg_canon import canonicalize, canonicalize_spatial
from sgg_postprocess import cleanup as _postprocess_cleanup
from sgg_geometric_spatial import spatial_for_image, PARTS as BODY_PARTS

REPO_ROOT = Path(__file__).resolve().parents[1]

OVERLAYS = {
    "point_only":   overlay_point_only,
    "point_fixed":  overlay_point_fixed,
    "bbox_only":    overlay_bbox_only,
    "contour_only": overlay_contour_only,
}

# Default prompt per strategy (used unless --prompt_file is explicitly overridden).
STRATEGY_DEFAULT_PROMPT = {
    "single":           "datagen/prompts/iter_19.txt",
    "relchain":         "datagen/prompts/relchain_v1.txt",
    "subject_anchored": "datagen/prompts/subject_anchor_v2.txt",
    "pairwise":         "datagen/prompts/openvoc_cot_pairwise_v1.txt",
    "refine":           "datagen/prompts/sgg26b_coverage_v1.txt",   # turn-1 draft
    "anchor_refine":    "datagen/prompts/sgg26b_anchor_recall_v1.txt",  # per-anchor draft
    "grow":             "datagen/prompts/iter_19.txt",                   # round-1 base
}

# ── Prompt templating ─────────────────────────────────────────────────────────
# iter_19 (single) escapes its JSON braces as {{ }} for str.format(); the
# caption-first prompts use raw { } and must be filled by literal replacement.

def _fmt_single(tmpl: str, n: int, object_list: str) -> str:
    return tmpl.format(n=n, object_list=object_list)


def _fill(tmpl: str, **kw) -> str:
    for k, v in kw.items():
        tmpl = tmpl.replace("{" + k + "}", str(v))
    return tmpl


def _object_list(objects: list) -> str:
    return "\n".join(f"  {i+1}_{o['label']}" for i, o in enumerate(objects))


def _format_draft(rels: list) -> str:
    """Render a parsed relation list as a readable draft for the revise turn."""
    if not rels:
        return "(no relationships found)"
    return "\n".join(
        f"  {r['subject_label']} {r['subject_id']} -> {r['predicate']} -> "
        f"{r['object_label']} {r['object_id']}" for r in rels)


# The LLM spatial round is a CLOSED-set task: anything outside this whitelist is a
# leak (bare "on", between, action verbs) and gets dropped at parse time.
# Keys are the collapsed canonical forms so "on top of"→above passes the check.
SPATIAL_WHITELIST = frozenset({
    "above", "below", "in front of", "behind",
    "to the left of", "to the right of", "inside",
})

# Geometry-emitted directions are canonical (subject = left/upper/front). For
# training data we randomize each edge's direction so both surface forms appear.
GEO_FLIP = {"to the left of": "to the right of", "above": "below",
            "in front of": "behind"}

# ── llm_v2: OPEN-vocabulary spatial round, geometry-verified ───────────────────
# The LLM picks WHICH pairs define the layout and judges the axes boxes can't
# see; the boxes falsify what they can. Canonical class → verification:
#   lr / vert — box centers falsify the direction (fixable by a role swap)
#   depth     — boxes can't falsify front/behind → trust the model (it sees pixels)
#   contact   — boxes must touch      inside — subject box mostly within object's
#   near      — boxes touch or sit within half the larger box diagonal
V2_CLASS = {
    "to the left of": "lr", "to the right of": "lr",
    "above": "vert", "below": "vert",
    "in front of": "depth", "behind": "depth",
    "inside": "inside", "near": "near",
    "on": "contact", "sitting on": "contact", "standing on": "contact",
    "lying on": "contact", "leaning against": "contact", "leaning on": "contact",
    "hanging from": "contact", "hanging on": "contact", "attached to": "contact",
    "touching": "contact", "against": "contact",
    "resting on": "contact", "mounted on": "contact",
}
# Surface forms that collapse to "above" but CLAIM support: boxes must also touch,
# and a wrong direction can't be fixed by a role swap ("table atop cup" is junk).
V2_NEEDS_CONTACT = frozenset({"on top of", "atop", "on top"})
# Surface-form opposites — used to fix a wrong direction by swapping roles and to
# 50%-flip verified directional edges (the LLM has a strong left/above subject
# bias; measured 34% "to the left of" vs 10% "to the right of" in config B).
V2_OPPOSITE = {
    "to the left of": "to the right of", "left of": "right of",
    "on the left side of": "on the right side of",
    "on the left of": "on the right of",
    "above": "below", "over": "under", "beneath": "above", "underneath": "above",
    "in front of": "behind", "ahead of": "behind",
}
for _k, _v in list(V2_OPPOSITE.items()):       # reverse entries; first mapping wins
    V2_OPPOSITE.setdefault(_v, _k)             # ("behind" → "in front of", not "ahead of")

# Predicates that REQUIRE the two boxes to touch — a disjoint pair is a label-prior
# hallucination ("wearing" shoes lying on the ground). Applied post-canonicalization
# when --contact_gate is set. Attention preds are NOT gated (gaze crosses gaps).
CONTACT_GATE = frozenset({
    "wearing", "part of",                                     # containment
    "holding", "carrying", "riding", "sitting on", "lying on", "standing on",
    "resting on", "leaning against", "mounted on", "attached to", "covering",
    "hanging from", "eating from", "drinking from",
    "on", "on top of", "atop",                                # support family
})

# Person-instance labels — two of these with near-identical boxes are duplicate
# annotations of the SAME person ("woman"+"girl"), and relations between them
# ("girl holding woman") are artifacts of the duplication, not of the scene.
PERSON_CLASSES = frozenset({
    "person", "man", "woman", "girl", "boy", "child", "lady", "guy", "human",
})

# Body-part-as-SUBJECT is only ever meaningful as "part of" its owner —
# "human arm wearing suit" / "human face listening to woman" are junk.
PART_OK_PREDICATES = frozenset({"part of"})

# module flag set from --keep_spatial_synonyms in main()
COLLAPSE_SPATIAL = True


def _boxes_touch(b1, b2) -> bool:
    """True if two [x1,y1,x2,y2] boxes overlap at all."""
    return (min(b1[2], b2[2]) > max(b1[0], b2[0])
            and min(b1[3], b2[3]) > max(b1[1], b2[1]))


def _contain_frac_xyxy(inner, outer) -> float:
    """Fraction of inner [x1,y1,x2,y2] box that lies inside outer."""
    x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    ia = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (x2 - x1) * (y2 - y1) / ia if ia else 0.0


def _iou_xyxy(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter)


# ── Caption-first parsing/validation (mirrors sgg_caption_first.py) ────────────
_LEADING_COPULA = re.compile(r"^(is|are|was|were|be|being|been)\s+", re.IGNORECASE)
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[a-z0-9]+")
_NON_RELATIONAL = {
    "visible", "partially visible", "present", "seen", "exists",
    "in background", "in the background", "there",
}


def canon_pred(p: str) -> str:
    if not p:
        return ""
    s = str(p).strip().lower().replace("_", " ")
    s = _LEADING_COPULA.sub("", s)
    return _WS.sub(" ", s).strip(".,-")


def _words(s: str) -> set:
    return set(_TOKEN.findall(s.lower())) if s else set()


def _cited(evidence: str, caption: str, min_overlap: float = 0.5) -> bool:
    """CF-2 gate — evidence span must genuinely come from the caption."""
    ev = _words(evidence)
    if not ev:
        return False
    return len(ev & _words(caption)) / len(ev) >= min_overlap


def _validate(s, o, pred, n: int, seen: set):
    """Shared id/predicate validation → (ok, reason)."""
    try:
        s = int(s); o = int(o)
    except (TypeError, ValueError):
        return None, "bad-id"
    if not (1 <= s <= n and 1 <= o <= n):
        return None, "out-of-range"
    if s == o:
        return None, "reflexive"
    pred = canon_pred(pred)
    if not pred:
        return None, "empty-predicate"
    if (s, o, pred) in seen:
        return None, "duplicate"
    return (s, o, pred), ""


def _canon_or_drop(pred: str, drops: Counter):
    """Apply canonicalization; return (canon_pred, raw_pred) or (None, _) to drop."""
    c = canonicalize(pred, collapse_spatial=COLLAPSE_SPATIAL)
    if c is None:
        drops["vague-spatial"] += 1
        return None
    return c


def parse_caption_relations(parser_raw: str, objects: list, caption: str,
                            canon: bool = False):
    """caption_first: validate parser JSON with the CF-2 evidence gate."""
    n = len(objects)
    id2label = {i + 1: o["label"] for i, o in enumerate(objects)}
    sg = _extract_json(parser_raw)
    if not sg or "relations" not in sg:
        return [], Counter({"parse-fail": 1}), False
    kept, drops, seen = [], Counter(), set()
    for rel in sg.get("relations", []):
        v, reason = _validate(rel.get("subject_id"), rel.get("object_id"),
                              rel.get("predicate", ""), n, seen)
        if v is None:
            drops[reason] += 1
            continue
        s, o, pred = v
        if pred in _NON_RELATIONAL:
            drops["non-relational"] += 1
            continue
        evidence = str(rel.get("evidence", "")).strip()
        if not _cited(evidence, caption):
            drops["uncited"] += 1
            continue
        raw_pred = pred
        if canon:
            c = _canon_or_drop(pred, drops)
            if c is None:
                continue
            pred = c
            if (s, o, pred) in seen:          # canon may collapse two into one
                drops["duplicate"] += 1
                continue
        seen.add((s, o, pred))
        rec = {"subject_id": s, "subject_label": id2label[s],
               "predicate": pred, "object_id": o, "object_label": id2label[o],
               "evidence": evidence}
        if canon and raw_pred != pred:
            rec["predicate_raw"] = raw_pred
        kept.append(rec)
    return kept, drops, True


def parse_single_relations(raw: str, objects: list, canon: bool = False):
    """single: extract triples directly, validate ids, drop reflexive/dupes."""
    n = len(objects)
    id2label = {i + 1: o["label"] for i, o in enumerate(objects)}
    sg = _extract_json(raw)
    if not sg or "relations" not in sg:
        return [], Counter({"parse-fail": 1}), False
    kept, drops, seen = [], Counter(), set()
    for rel in sg.get("relations", []):
        v, reason = _validate(rel.get("subject_id"), rel.get("object_id"),
                              rel.get("predicate", ""), n, seen)
        if v is None:
            drops[reason] += 1
            continue
        s, o, pred = v
        raw_pred = pred
        if canon:
            c = _canon_or_drop(pred, drops)
            if c is None:
                continue
            pred = c
            if (s, o, pred) in seen:
                drops["duplicate"] += 1
                continue
        seen.add((s, o, pred))
        rec = {"subject_id": s, "subject_label": id2label[s],
               "predicate": pred, "object_id": o, "object_label": id2label[o]}
        if canon and raw_pred != pred:
            rec["predicate_raw"] = raw_pred
        if "kind" in rel:                       # dual-layer prompt (interaction|layout)
            rec["kind"] = rel["kind"]
        kept.append(rec)
    return kept, drops, True


# ── Extra strategies: relchain / subject_anchored / pairwise ───────────────────
# These mirror the sgg_trial.py modes (relchain_v1.txt, subject_anchor_v2.txt,
# openvoc_cot_pairwise_v1.txt) but run under vLLM. The single-pass strategies were
# capacity-limited on E4B; the 26B MoE motivates re-testing them.

def _min_rels(n: int, formula: str = "quarter") -> int:
    if formula == "half":
        return max(3, min(50, n * (n - 1) // 2))
    if formula == "half_20":
        return max(3, min(20, n * (n - 1) // 2))
    return max(3, min(20, round(n * (n - 1) / 4) + 1))


def pair_salience(b1, b2, W, H) -> float:
    """Proximity+IoU+area-ratio score to rank pairs (geo-top-K selection)."""
    cax, cay = (b1[0] + b1[2]) / 2, (b1[1] + b1[3]) / 2
    cbx, cby = (b2[0] + b2[2]) / 2, (b2[1] + b2[3]) / 2
    diag = math.hypot(W, H) or 1.0
    proximity = max(0.0, 1.0 - math.hypot(cax - cbx, cay - cby) / diag * 1.41)
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    aa = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
    ab = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
    union = aa + ab - inter
    iou = inter / union if union > 0 else 0.0
    area_ratio = min(aa, ab) / max(aa, ab) if max(aa, ab) > 0 else 0.0
    return 0.50 * proximity + 0.30 * iou + 0.20 * area_ratio


_PAIR_FONT = None


def _pair_font():
    global _PAIR_FONT
    if _PAIR_FONT is None:
        from PIL import ImageFont
        try:
            _PAIR_FONT = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        except Exception:                              # noqa: BLE001
            _PAIR_FONT = ImageFont.load_default()
    return _PAIR_FONT


def crop_and_annotate_pair(pil, s_obj, o_obj, padding_frac: float = 0.25):
    """Crop the union bbox (padded) and mark subject=red dot 1, object=blue dot 2."""
    from PIL import ImageDraw
    W, H = pil.size
    bs, bo = s_obj["bbox"], o_obj["bbox"]
    ux1, uy1 = min(bs[0], bo[0]), min(bs[1], bo[1])
    ux2, uy2 = max(bs[2], bo[2]), max(bs[3], bo[3])
    pad_x = max(10, int((ux2 - ux1) * padding_frac))
    pad_y = max(10, int((uy2 - uy1) * padding_frac))
    cx1, cy1 = max(0, ux1 - pad_x), max(0, uy1 - pad_y)
    cx2, cy2 = min(W, ux2 + pad_x), min(H, uy2 + pad_y)
    crop = pil.crop((cx1, cy1, cx2, cy2)).copy()
    draw = ImageDraw.Draw(crop, "RGBA")
    font = _pair_font()
    for k, obj in enumerate([s_obj, o_obj]):
        bx1, by1, bx2, by2 = obj["bbox"]
        pcx, pcy = (bx1 + bx2) // 2 - cx1, (by1 + by2) // 2 - cy1
        c = ["#e6194b", "#4363d8"][k]
        draw.ellipse([pcx - 10, pcy - 10, pcx + 10, pcy + 10],
                     fill=c + "ff", outline="white", width=2)
        draw.text((pcx + 14, pcy - 9), str(k + 1), font=font, fill=c,
                  stroke_width=2, stroke_fill="white")
    return crop.convert("RGB")


def parse_relchain(raw: str, objects: list, canon: bool = False):
    """relchain: like single, but preserves chain_id metadata."""
    n = len(objects)
    id2label = {i + 1: o["label"] for i, o in enumerate(objects)}
    sg = _extract_json(raw)
    if not sg or "relations" not in sg:
        return [], Counter({"parse-fail": 1}), False
    kept, drops, seen = [], Counter(), set()
    for rel in sg.get("relations", []):
        v, reason = _validate(rel.get("subject_id"), rel.get("object_id"),
                              rel.get("predicate", ""), n, seen)
        if v is None:
            drops[reason] += 1
            continue
        s, o, pred = v
        raw_pred = pred
        if canon:
            c = _canon_or_drop(pred, drops)
            if c is None:
                continue
            pred = c
            if (s, o, pred) in seen:
                drops["duplicate"] += 1
                continue
        seen.add((s, o, pred))
        rec = {"subject_id": s, "subject_label": id2label[s],
               "predicate": pred, "object_id": o, "object_label": id2label[o],
               "chain_id": rel.get("chain_id", 0)}
        if canon and raw_pred != pred:
            rec["predicate_raw"] = raw_pred
        kept.append(rec)
    return kept, drops, True


def parse_anchor_into(raw: str, objects: list, anchor_id: int, *,
                      canon: bool, conf_threshold: int,
                      kept: list, drops: Counter, seen: set):
    """subject_anchored: parse ONE anchor pass, append confident relations."""
    n = len(objects)
    id2label = {i + 1: o["label"] for i, o in enumerate(objects)}
    sg = _extract_json(raw)
    if not sg or "relations" not in sg:
        drops["parse-fail"] += 1
        return False
    for rel in sg.get("relations", []):
        conf = rel.get("confidence")
        if conf_threshold > 0 and (conf is None or conf < conf_threshold):
            drops["low-conf"] += 1
            continue
        v, reason = _validate(rel.get("subject_id", anchor_id),
                              rel.get("object_id"), rel.get("predicate", ""), n, seen)
        if v is None:
            drops[reason] += 1
            continue
        s, o, pred = v
        raw_pred = pred
        if canon:
            c = _canon_or_drop(pred, drops)
            if c is None:
                continue
            pred = c
        if (s, o, pred) in seen:
            drops["duplicate"] += 1
            continue
        seen.add((s, o, pred))
        rec = {"subject_id": s, "subject_label": id2label[s],
               "predicate": pred, "object_id": o, "object_label": id2label[o],
               "confidence": conf}
        if canon and raw_pred != pred:
            rec["predicate_raw"] = raw_pred
        kept.append(rec)
    return True


def parse_pair_into(raw: str, s_obj: dict, o_obj: dict, si: int, oi: int, *,
                    canon: bool, conf_threshold: int,
                    kept: list, drops: Counter, seen: set):
    """pairwise: parse ONE pair crop; map dot ids {1,2} → global ids {si+1,oi+1}."""
    sg = _extract_json(raw)
    if not sg or "relations" not in sg:
        drops["parse-fail"] += 1
        return False
    dot2glob = {1: si + 1, 2: oi + 1}
    dot2lbl = {1: s_obj["label"], 2: o_obj["label"]}
    for rel in sg.get("relations", []):
        conf = rel.get("confidence")
        if conf_threshold > 0 and (conf is None or conf < conf_threshold):
            drops["low-conf"] += 1
            continue
        m_s, m_o = rel.get("subject_id", 1), rel.get("object_id", 2)
        if m_s == m_o or m_s not in dot2glob or m_o not in dot2glob:
            drops["bad-id"] += 1
            continue
        pred = canon_pred(rel.get("predicate", ""))
        if not pred:
            drops["empty-predicate"] += 1
            continue
        raw_pred = pred
        if canon:
            c = canonicalize(pred)
            if c is None:
                drops["vague-spatial"] += 1
                continue
            pred = c
        g_s, g_o = dot2glob[m_s], dot2glob[m_o]
        if (g_s, g_o, pred) in seen:
            drops["duplicate"] += 1
            continue
        seen.add((g_s, g_o, pred))
        rec = {"subject_id": g_s, "subject_label": dot2lbl[m_s],
               "predicate": pred, "object_id": g_o, "object_label": dot2lbl[m_o],
               "confidence": conf}
        if canon and raw_pred != pred:
            rec["predicate_raw"] = raw_pred
        kept.append(rec)
    return True


# ── Geometric spatial layer + contact gate ─────────────────────────────────────

def append_geometric_spatial(rels: list, objects: list, img_id: int,
                             max_per_subject: int, limit: int = 0) -> list:
    """Append the deterministic box-derived spatial layer (sgg_geometric_spatial)
    to an image's relation list. Direction is randomized per edge (seeded on
    img_id) so both surface forms of each axis appear in the training data.

    Pairs that already carry ANY relation (semantic or LLM-spatial) are skipped —
    their layout is implied by the interaction, and box containment on
    interacting pairs is where the junk lives ("hat inside person"). Geometric
    "inside" is disabled outright: 2D containment ≠ 3D containment; those pairs
    fall through to the occlusion axis (in front of / behind), which is what the
    boxes actually show.

    limit>0 keeps only the top-N candidate edges ranked by pair_salience
    (proximity+IoU+size balance) instead of object loop order — used by the
    top-up mode, where geometry fills the gap the LLM round left."""
    geo_objs = [{"id": i + 1, "label": o["label"],
                 "bbox": [o["bbox"][0], o["bbox"][1],
                          o["bbox"][2] - o["bbox"][0], o["bbox"][3] - o["bbox"][1]]}
                for i, o in enumerate(objects)]
    covered = {(r["subject_id"], r["object_id"]) for r in rels}
    # duplicate person boxes (same person annotated twice) get no layout edge
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if (objects[i]["label"].lower() in PERSON_CLASSES
                    and objects[j]["label"].lower() in PERSON_CLASSES
                    and _iou_xyxy(objects[i]["bbox"], objects[j]["bbox"]) >= 0.7):
                covered.add((i + 1, j + 1))
    geo = spatial_for_image(geo_objs, drop_parts=True,
                            max_per_subject=max_per_subject,
                            emit_inside=False, skip_pairs=covered)
    if limit > 0 and len(geo) > limit:
        # rank by salience within the image; max box extent proxies the frame size
        W = max((o["bbox"][2] for o in objects), default=1)
        H = max((o["bbox"][3] for o in objects), default=1)
        geo.sort(key=lambda g: -pair_salience(objects[g["subject_id"] - 1]["bbox"],
                                              objects[g["object_id"] - 1]["bbox"],
                                              W, H))
        geo = geo[:limit]
    rng = random.Random(img_id)
    for g in geo:
        if g["predicate"] in GEO_FLIP and rng.random() < 0.5:
            g = {**g, "subject_id": g["object_id"], "subject_label": g["object_label"],
                 "object_id": g["subject_id"], "object_label": g["subject_label"],
                 "predicate": GEO_FLIP[g["predicate"]]}
        rels.append(g)
    return rels


def _box_gap(a, b) -> float:
    """Euclidean gap between two xyxy boxes (0 if they touch/overlap)."""
    dx = max(a[0] - b[2], b[0] - a[2], 0)
    dy = max(a[1] - b[3], b[1] - a[3], 0)
    return math.hypot(dx, dy)


def verify_spatial_v2(rels: list, objects: list, img_id: int,
                      drops: Counter) -> list:
    """Geometry-verify the open-vocabulary LLM spatial round (--spatial_source
    llm_v2). Selection and depth/contact judgment come from the model; the boxes
    falsify what they can:

      lr/vert  — wrong direction is FIXED by swapping roles (surface form kept);
                 support phrasings (atop/on top of) can't be role-swapped → drop
      contact  — boxes must touch          inside — subj ≥50% within obj box
      near     — boxes touch or gap ≤ 0.5 × the larger box diagonal
      depth    — trusted (boxes can't falsify what occludes what)

    Vocabulary outside V2_CLASS is dropped (the round is layout-only; actions
    belong to the semantic rounds). Body-part boxes get no layout edge (their
    position is implied by their owner). Verified directional edges are then
    50%-flipped (seeded on img_id) to balance the model's left/above bias."""
    out = []
    n = len(objects)
    rng = random.Random(img_id ^ 0x5F3759DF)   # decorrelated from the geo-layer rng
    for r in rels:
        s, o = r.get("subject_id"), r.get("object_id")
        if not (isinstance(s, int) and isinstance(o, int)
                and 1 <= s <= n and 1 <= o <= n):
            drops["spatial-bad-id"] += 1
            continue
        if (objects[s - 1]["label"].lower() in BODY_PARTS
                or objects[o - 1]["label"].lower() in BODY_PARTS):
            drops["spatial-part"] += 1
            continue
        raw = r.get("predicate", "")
        surface = canonicalize_spatial(raw, collapse_spatial=False)
        canon = canonicalize_spatial(raw, collapse_spatial=True)
        if surface is None or canon not in V2_CLASS:
            drops["spatial-vocab"] += 1
            continue
        cls = V2_CLASS[canon]
        sb, ob = objects[s - 1]["bbox"], objects[o - 1]["bbox"]

        if ((cls == "contact" or surface in V2_NEEDS_CONTACT)
                and not _boxes_touch(sb, ob)):
            drops["spatial-contact"] += 1
            continue
        if cls == "inside" and _contain_frac_xyxy(sb, ob) < 0.5:
            drops["spatial-inside"] += 1
            continue
        if cls == "near" and not _boxes_touch(sb, ob):
            diag = max(math.hypot(sb[2] - sb[0], sb[3] - sb[1]),
                       math.hypot(ob[2] - ob[0], ob[3] - ob[1]))
            if _box_gap(sb, ob) > 0.5 * diag:
                drops["spatial-near-far"] += 1
                continue

        if cls in ("lr", "vert"):
            if cls == "lr":
                subj_first = sb[0] + sb[2] < ob[0] + ob[2]     # center further left
            else:
                subj_first = sb[1] + sb[3] < ob[1] + ob[3]     # center higher up
            claim_first = canon in ("to the left of", "above")
            if subj_first != claim_first:
                if surface in V2_OPPOSITE:       # projective → swap roles, keep surface
                    drops["spatial-dir-fixed"] += 1
                    s, o = o, s
                else:                            # support phrasing → the claim is wrong
                    drops["spatial-dir"] += 1
                    continue

        # 50% direction balancing on the verified projective/depth axes
        if (cls in ("lr", "vert", "depth") and surface in V2_OPPOSITE
                and rng.random() < 0.5):
            s, o = o, s
            surface = V2_OPPOSITE[surface]

        rec = {**r, "subject_id": s, "subject_label": objects[s - 1]["label"],
               "object_id": o, "object_label": objects[o - 1]["label"],
               "predicate": surface}
        if surface != raw:
            rec.setdefault("predicate_raw", raw)
        out.append(rec)
    return out


def apply_part_gate(rels: list, objects: list, drops: Counter) -> list:
    """Deterministic filter for the body-part failure class (20-image review,
    2026-07-14): the grow round pads sparse images with relations on body-part
    boxes ("man wearing human nose", "man drinking from human face").

      1. body-part SUBJECT → only "part of" survives
      2. body-part OBJECT mostly inside the subject's own box → self-part
         relation ("man looking at [his own] human face") → drop, unless "part of"
      3. two person-class boxes with IoU ≥ 0.7 are duplicate annotations of one
         person → any relation between them is an artifact → drop

    Genuine cross-instance part relations survive: "glasses covering human face"
    (subject not a part), "person holding human hand" (hand not inside holder)."""
    kept = []
    n = len(objects)
    for r in rels:
        s, o = r.get("subject_id"), r.get("object_id")
        if not (isinstance(s, int) and isinstance(o, int)
                and 1 <= s <= n and 1 <= o <= n):
            kept.append(r)
            continue
        sl = objects[s - 1]["label"].lower()
        ol = objects[o - 1]["label"].lower()
        # duplicate-person rule applies to EVERY source, geometry included —
        # a layout edge between two boxes of the same person is still junk
        if (sl in PERSON_CLASSES and ol in PERSON_CLASSES
                and _iou_xyxy(objects[s - 1]["bbox"], objects[o - 1]["bbox"]) >= 0.7):
            drops["duplicate-person-box"] += 1
            continue
        if r.get("source") == "geometric":                 # geometry pre-filters parts
            kept.append(r)
            continue
        pred = canonicalize(r.get("predicate", "")) or ""
        if pred not in PART_OK_PREDICATES:
            if sl in BODY_PARTS:
                drops["part-subject"] += 1
                continue
            if ol in BODY_PARTS and _contain_frac_xyxy(
                    objects[o - 1]["bbox"], objects[s - 1]["bbox"]) >= 0.6:
                drops["self-part"] += 1
                continue
        kept.append(r)
    return kept


def apply_contact_gate(rels: list, objects: list, drops: Counter) -> list:
    """Drop contact/containment predicates whose boxes are fully disjoint —
    the geometric signature of a label-prior hallucination (audit_contact_geometry
    measures these; this enforces it). Predicates are matched on the collapsed
    canonical form so phrasing variants are gated too."""
    kept = []
    for r in rels:
        c = canonicalize(r.get("predicate", ""))
        if c in CONTACT_GATE:
            s, o = r.get("subject_id"), r.get("object_id")
            if (isinstance(s, int) and isinstance(o, int)
                    and 1 <= s <= len(objects) and 1 <= o <= len(objects)
                    and not _boxes_touch(objects[s - 1]["bbox"], objects[o - 1]["bbox"])):
                drops["contact-disjoint"] += 1
                continue
        kept.append(r)
    return kept


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(per_image_rels: list[list], n_images: int, parse_ok: int) -> dict:
    pred_count: Counter = Counter()
    total = 0
    for rels in per_image_rels:
        for r in rels:
            p = r.get("predicate", "").lower().strip()
            if p:
                pred_count[p] += 1
                total += 1
    if total:
        probs = [c / total for c in pred_count.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
    else:
        entropy = 0.0
    n_forbidden = sum(c for p, c in pred_count.items() if _is_forbidden(p))
    return {
        "n_images": n_images, "parse_ok": parse_ok, "total_rels": total,
        "rels_per_img": round(total / max(n_images, 1), 2),
        "n_unique": len(pred_count),
        "entropy_nats": round(entropy, 3),
        "forbidden_rate": round(n_forbidden / max(total, 1), 4),
        "top10": pred_count.most_common(10),
    }


# ── Annotation index (no pixels) + sharding ────────────────────────────────────

def load_anno_index(anno_path: Path, require_relations: bool, max_objects: int):
    """Return [{img_id, file_name, objects:[{id,label,bbox}]}] without loading images."""
    with open(anno_path) as f:
        data = json.load(f)
    cat = {c["id"]: c["name"] for c in data["categories"]}
    ann_by_img: dict = defaultdict(list)
    for a in data["annotations"]:
        ann_by_img[a["image_id"]].append(a)
    rel_imgs = {r["image_id"] for r in data.get("rel_annotations", [])}

    metas = []
    for img in data["images"]:
        iid = img["id"]
        if require_relations and iid not in rel_imgs:
            continue
        objs = []
        for a in ann_by_img.get(iid, []):
            x, y, w, h = a["bbox"]
            objs.append({"id": a["id"], "label": cat[a["category_id"]],
                         "bbox": [int(x), int(y), int(x + w), int(y + h)]})
        if len(objs) < 2:
            continue
        if max_objects and len(objs) > max_objects:
            objs = objs[:max_objects]
        metas.append({"img_id": iid, "file_name": img["file_name"], "objects": objs})
    return metas


# ── vLLM runner ─────────────────────────────────────────────────────────────────

class VLLMRunner:
    def __init__(self, model_id: str, *, dtype: str, quant: str | None,
                 tp: int, gpu_mem: float, max_model_len: int, seed: int,
                 spec_ngram: int = 0, spec_lookup: int = 4,
                 kv_dtype: str = "auto", async_scheduling: bool = False,
                 max_num_batched_tokens: int = 0):
        from vllm import LLM, SamplingParams      # lazy: not installed on dev box
        from transformers import AutoProcessor

        self.SamplingParams = SamplingParams
        self.thinking = False                       # set by main from --thinking
        self.processor = AutoProcessor.from_pretrained(model_id)
        kw = dict(model=model_id, dtype=dtype, tensor_parallel_size=tp,
                  gpu_memory_utilization=gpu_mem, max_model_len=max_model_len,
                  limit_mm_per_prompt={"image": 1}, seed=seed,
                  disable_log_stats=True)
        if quant:
            kw["quantization"] = quant
        if spec_ngram > 0:
            # exact under greedy decoding: proposals are verified by the target
            # model, so outputs are unchanged — pure decode speedup. Our JSON
            # output re-emits prompt tokens (object labels) and repeats its own
            # keys every relation → high n-gram acceptance.
            kw["speculative_config"] = {"method": "ngram",
                                        "num_speculative_tokens": spec_ngram,
                                        "prompt_lookup_max": spec_lookup,
                                        "prompt_lookup_min": 2}
        if kv_dtype and kv_dtype != "auto":
            kw["kv_cache_dtype"] = kv_dtype
        if async_scheduling:
            kw["async_scheduling"] = True
        if max_num_batched_tokens:
            kw["max_num_batched_tokens"] = max_num_batched_tokens
        print(f"  vLLM LLM(model={model_id}, dtype={dtype}, quant={quant}, "
              f"tp={tp}, gpu_mem={gpu_mem}, max_len={max_model_len}, "
              f"spec_ngram={spec_ngram}, kv={kv_dtype}, "
              f"async_sched={async_scheduling}) …", flush=True)
        self.llm = LLM(**kw)

    def _tmpl(self, prompt: str, with_image: bool) -> str:
        content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        try:
            return self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.thinking)
        except TypeError:                            # template doesn't accept enable_thinking
            return self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def _sp(self, max_tokens: int, temperature: float):
        if temperature and temperature > 0:
            return self.SamplingParams(temperature=temperature, top_p=0.95, top_k=64,
                                       max_tokens=max_tokens, skip_special_tokens=True)
        return self.SamplingParams(temperature=0.0, max_tokens=max_tokens,
                                   skip_special_tokens=True)

    def generate_mm(self, prompts: list[str], images: list, max_tokens: int, temperature: float):
        reqs = [{"prompt": self._tmpl(p, True), "multi_modal_data": {"image": img}}
                for p, img in zip(prompts, images)]
        outs = self.llm.generate(reqs, self._sp(max_tokens, temperature))
        return [o.outputs[0].text for o in outs]

    def generate_text(self, prompts: list[str], max_tokens: int):
        reqs = [{"prompt": self._tmpl(p, False)} for p in prompts]
        outs = self.llm.generate(reqs, self._sp(max_tokens, 0.0))
        return [o.outputs[0].text for o in outs]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--strategy",
                    choices=["single", "caption_first", "relchain",
                             "subject_anchored", "pairwise", "refine", "anchor_refine",
                             "grow"],
                    default="single")
    ap.add_argument("--model", default="26b", choices=sorted(MODEL_CHOICES))
    # data
    ap.add_argument("--megasg_dir", default=MEGASG_DIR)
    ap.add_argument("--split", default="train")
    ap.add_argument("--require_relations", action="store_true",
                    help="Only keep images that have GT relations (eval/calibration). "
                         "Off by default for production generation.")
    ap.add_argument("--max_objects", type=int, default=40)
    ap.add_argument("--canonicalize", action="store_true",
                    help="Collapse spatial-phrasing synonyms + drop vague-proximity "
                         "predicates (sgg_canon). Keeps raw in predicate_raw.")
    ap.add_argument("--keep_spatial_synonyms", action="store_true",
                    help="With --canonicalize: still drop vague/scaffold predicates, "
                         "but KEEP the model's spatial surface forms (atop, on top of, "
                         "underneath …) instead of collapsing them — preserves "
                         "open-vocabulary predicate diversity. Canonical forms are "
                         "still used internally for dedup/contradiction logic.")
    ap.add_argument("--spatial_source", default="llm",
                    choices=["llm", "llm_v2", "geometric"],
                    help="Where the spatial layout layer comes from. 'llm' = the "
                         "closed-set spatial prompt round (grow strategy). "
                         "'llm_v2' = OPEN-vocabulary spatial round (depth, contact, "
                         "proximity, projective + synonyms), geometry-verified per "
                         "edge (verify_spatial_v2) — pair selection by scene "
                         "salience instead of box enumeration. 'geometric' = "
                         "deterministic box-derived layer (sgg_geometric_spatial), "
                         "appended per image and tagged source=geometric; any "
                         "--spatial_prompt round is skipped.")
    ap.add_argument("--geo_backstop", type=int, default=0,
                    help="llm/llm_v2 spatial only: top up images with fewer than N "
                         "spatial relations using the most SALIENT remaining "
                         "geometric edges (pair_salience ranked, tagged "
                         "source=geometric) until the floor N is reached. 0 = off.")
    ap.add_argument("--contact_gate", action="store_true",
                    help="Drop contact/containment predicates (wearing, holding, "
                         "sitting on, on …) whose subject/object boxes are fully "
                         "disjoint — deterministic hallucination filter.")
    ap.add_argument("--part_gate", action="store_true",
                    help="Drop body-part junk: part-as-subject relations (except "
                         "'part of'), self-part relations ('man wearing human "
                         "nose'), and relations between duplicate person boxes.")
    ap.add_argument("--postprocess", action="store_true",
                    help="Deterministic cleanup of the unpooled union (sgg_postprocess): "
                         "drop inverses/contradictions, cap spatial. Preserves density.")
    ap.add_argument("--max_spatial_per_subject", type=int, default=1,
                    help="--postprocess: max directional-spatial edges per subject "
                         "(raise when 30-50%% spatial is acceptable).")
    ap.add_argument("--keep_spatial_with_action", action="store_true",
                    help="--postprocess: keep a spatial edge even if the pair has an action.")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--skip", type=int, default=0,
                    help="Skip the first N images of the filtered index BEFORE sharding "
                         "(chains disjoint incremental runs, e.g. a second batch after "
                         "an already-completed --limit run).")
    ap.add_argument("--limit", type=int, default=0, help="Cap images in this shard (0=all).")
    ap.add_argument("--exclude_run", action="append", default=[], type=Path,
                    help="Run directory (or several, repeat the flag) whose "
                         "shard_*.jsonl img_ids are excluded from the index before "
                         "--skip/sharding. Use to chain a disjoint incremental batch "
                         "onto already-completed runs WITHOUT relying on index "
                         "position staying stable (e.g. after changing "
                         "--require_relations, which changes the pool itself).")
    ap.add_argument("--chunk", type=int, default=512, help="Images loaded+submitted per vLLM call.")
    # prompts
    ap.add_argument("--prompt_file", default="datagen/prompts/iter_19.txt",
                    help="single → iter_19; relchain → relchain_v1; "
                         "subject_anchored → subject_anchor_v2; pairwise → openvoc_cot_pairwise_v1.")
    ap.add_argument("--observer_prompt", default="datagen/prompts/cf_observer.txt")
    ap.add_argument("--parser_prompt", default="datagen/prompts/cf_parser.txt")
    ap.add_argument("--revise_prompt", default="datagen/prompts/sgg26b_revise_v1.txt",
                    help="refine strategy: turn-2 revise prompt ({object_list},{draft}).")
    ap.add_argument("--grow_prompt", default="datagen/prompts/sgg26b_grow_v1.txt",
                    help="grow strategy: additive 'complete this graph' prompt ({object_list},{draft}).")
    ap.add_argument("--spatial_prompt", default=None,
                    help="grow strategy: if set, the LAST grow round uses this spatial-only "
                         "prompt ({object_list},{draft}) instead of --grow_prompt.")
    ap.add_argument("--grow_rounds", type=int, default=2,
                    help="grow strategy: total passes (1 base + N-1 additive completions).")
    ap.add_argument("--thinking", action="store_true",
                    help="Enable the model's thinking/reasoning mode (more tokens; "
                         "raise --max_new_tokens accordingly).")
    ap.add_argument("--overlay", default="point_only", choices=sorted(OVERLAYS))
    # extra-strategy knobs
    ap.add_argument("--conf_threshold", type=int, default=0,
                    help="subject_anchored/pairwise: drop relations below this confidence (1-5).")
    ap.add_argument("--geo_top_k", type=int, default=0,
                    help="pairwise: keep top-K pairs by geometric salience; "
                         "subject_anchored: use K largest objects as anchors (0=all).")
    ap.add_argument("--pair_frac", type=float, default=1.0,
                    help="pairwise: fraction of all object pairs to sample when geo_top_k=0.")
    ap.add_argument("--rels_formula", default="quarter",
                    choices=["quarter", "half", "half_20"],
                    help="relchain: min_rels = f(n_objects).")
    # generation
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--observer_max_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    # vLLM engine
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quant", default=None, help="e.g. fp8 (needed to fit 26B on 48GB).")
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    # speed knobs (output-preserving unless noted)
    ap.add_argument("--spec_ngram", type=int, default=0,
                    help="N-gram speculative decoding: propose N tokens per step "
                         "from prompt/output lookup. EXACT under greedy decoding "
                         "(target model verifies) — pure decode speedup. 0=off.")
    ap.add_argument("--spec_lookup", type=int, default=4,
                    help="Max n-gram length to match for --spec_ngram proposals.")
    ap.add_argument("--kv_dtype", default="auto",
                    help="KV cache dtype (e.g. fp8 → ~2x KV capacity = larger "
                         "decode batch). CAUTION: quantized KV can shift outputs "
                         "slightly — validate against a baseline before production.")
    ap.add_argument("--async_scheduling", action="store_true",
                    help="Overlap CPU scheduling with GPU execution (V1 engine).")
    ap.add_argument("--max_num_batched_tokens", type=int, default=0,
                    help="Chunked-prefill token budget per engine step (0=engine "
                         "default 8192). Larger = faster prefill, more activation "
                         "memory.")
    # output
    ap.add_argument("--outdir", default="runs/vllm_generate")
    ap.add_argument("--no_save_captions", action="store_true")
    ap.add_argument("--emit_trial_json", action="store_true",
                    help="Also write a trial.json (judge-compatible) — for calibration runs.")
    args = ap.parse_args()

    global COLLAPSE_SPATIAL
    COLLAPSE_SPATIAL = not args.keep_spatial_synonyms

    def _abs(p):
        p = Path(p)
        return p if p.is_absolute() else REPO_ROOT / p

    anno_path = Path(args.megasg_dir) / args.split / "_annotations.coco.json"
    img_dir = Path(args.megasg_dir) / args.split
    out_dir = _abs(args.outdir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"shard_{args.shard_index:04d}.jsonl"

    # Prompts. caption_first uses observer+parser; every other strategy uses a
    # single template (auto-selected per strategy unless --prompt_file overrides).
    main_tmpl = observer_tmpl = parser_tmpl = revise_tmpl = grow_tmpl = None
    if args.strategy == "caption_first":
        observer_tmpl = _abs(args.observer_prompt).read_text()
        parser_tmpl = _abs(args.parser_prompt).read_text()
    else:
        prompt_path = args.prompt_file
        if (args.prompt_file == STRATEGY_DEFAULT_PROMPT["single"]
                and args.strategy in STRATEGY_DEFAULT_PROMPT):
            prompt_path = STRATEGY_DEFAULT_PROMPT[args.strategy]   # default per strategy
        main_tmpl = _abs(prompt_path).read_text()
        print(f"  prompt: {prompt_path}  thinking={args.thinking}", flush=True)
        if args.strategy in ("refine", "anchor_refine"):
            revise_path = args.revise_prompt
            if (args.strategy == "anchor_refine"
                    and args.revise_prompt == "datagen/prompts/sgg26b_revise_v1.txt"):
                revise_path = "datagen/prompts/sgg26b_consolidate_v1.txt"   # default for anchor_refine
            revise_tmpl = _abs(revise_path).read_text()
            print(f"  revise: {revise_path}", flush=True)
        if args.strategy == "grow":
            grow_tmpl = _abs(args.grow_prompt).read_text()
            spatial_tmpl = _abs(args.spatial_prompt).read_text() if args.spatial_prompt else None
            if args.spatial_source == "geometric" and spatial_tmpl:
                print("  --spatial_source geometric: skipping the LLM spatial round "
                      f"({args.spatial_prompt})", flush=True)
                spatial_tmpl = None
            print(f"  grow: {args.grow_prompt}  rounds={args.grow_rounds}", flush=True)
            if spatial_tmpl:
                print(f"  spatial pass (last round): {args.spatial_prompt}", flush=True)
    if args.spatial_source == "geometric":
        print(f"  spatial layer: geometric (box-derived), "
              f"max/subject={args.max_spatial_per_subject}", flush=True)
    elif args.spatial_source == "llm_v2":
        print(f"  spatial layer: llm_v2 (open vocab, geometry-verified), "
              f"geo_backstop={args.geo_backstop}", flush=True)

    # Index + shard
    print(f"Indexing {anno_path} …", flush=True)
    metas = load_anno_index(anno_path, args.require_relations, args.max_objects)
    if args.exclude_run:
        exclude_ids = set()
        for d in args.exclude_run:
            d = _abs(d)
            n_before = len(exclude_ids)
            for shard in sorted(d.glob("shard_*.jsonl")):
                with open(shard) as f:
                    for line in f:
                        try:
                            exclude_ids.add(json.loads(line)["img_id"])
                        except Exception:                      # noqa: BLE001
                            pass
            print(f"  --exclude_run {d}: {len(exclude_ids) - n_before} img_ids", flush=True)
        n_before = len(metas)
        metas = [m for m in metas if m["img_id"] not in exclude_ids]
        print(f"  excluded {n_before - len(metas)} images already covered by "
              f"{len(args.exclude_run)} run(s) → {len(metas)} remain", flush=True)
    if args.skip:
        metas = metas[args.skip:]
    metas = metas[args.shard_index::args.num_shards]          # strided → balanced
    if args.limit:
        metas = metas[: args.limit]
    print(f"  shard {args.shard_index}/{args.num_shards}"
          f"{f' (skip {args.skip})' if args.skip else ''}: {len(metas)} images", flush=True)

    # Resume
    done_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["img_id"])
                except Exception:                              # noqa: BLE001
                    pass
        print(f"  resume: {len(done_ids)} already done → skipping", flush=True)
    metas = [m for m in metas if m["img_id"] not in done_ids]
    if not metas:
        print("  nothing to do.", flush=True)
        return

    # Engine
    print(f"Loading {args.model} via vLLM …", flush=True)
    model_id = MODEL_CHOICES[args.model]
    runner = VLLMRunner(model_id, dtype=args.dtype, quant=args.quant,
                        tp=args.tensor_parallel_size, gpu_mem=args.gpu_mem_util,
                        max_model_len=args.max_model_len, seed=args.seed,
                        spec_ngram=args.spec_ngram, spec_lookup=args.spec_lookup,
                        kv_dtype=args.kv_dtype,
                        async_scheduling=args.async_scheduling,
                        max_num_batched_tokens=args.max_num_batched_tokens)
    runner.thinking = args.thinking
    overlay_fn = OVERLAYS[args.overlay]

    from PIL import Image

    per_image_rels, drop_totals = [], Counter()
    parse_ok = 0
    t_start = time.time()
    n_total = len(metas)
    fout = open(jsonl_path, "a")

    area = lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    from concurrent.futures import ThreadPoolExecutor

    def _load_one(m):
        """(meta, raw_pil, overlay_img) or (meta, None, exception). PIL decode
        releases the GIL, so threading overlaps chunk image I/O with itself."""
        try:
            pil = Image.open(img_dir / m["file_name"]).convert("RGB")
            # pairwise crops from the raw image per-pair; others get the overlay
            img = (None if args.strategy == "pairwise"
                   else overlay_fn(pil, m["objects"], [None] * len(m["objects"])))
            return m, pil, img
        except Exception as exc:                               # noqa: BLE001
            return m, None, exc

    loader = ThreadPoolExecutor(max_workers=8)

    for c0 in range(0, n_total, args.chunk):
        chunk = metas[c0: c0 + args.chunk]
        imgs, raw_pils, valid = [], [], []
        for m, pil, img in loader.map(_load_one, chunk):       # order-preserving
            if pil is None:
                fout.write(json.dumps({"img_id": m["img_id"], "file_name": m["file_name"],
                                       "error": f"load:{img}", "relations": []}) + "\n")
                continue
            raw_pils.append(pil)
            imgs.append(img)
            valid.append(m)
        if not valid:
            continue

        t0 = time.time()
        captions = [None] * len(valid)
        if args.strategy == "single":
            prompts = [_fmt_single(main_tmpl, len(m["objects"]), _object_list(m["objects"]))
                       for m in valid]
            raws = runner.generate_mm(prompts, imgs, args.max_new_tokens, args.temperature)
            parsed = [parse_single_relations(r, m["objects"], args.canonicalize)
                      for r, m in zip(raws, valid)]

        elif args.strategy == "refine":
            # turn 1 — dense draft (coverage prompt); turn 2 — revise with the draft
            # embedded (drop wrong/redundant/contradictory, add missed). Two single-turn
            # multimodal calls (robust; no fragile multi-turn template).
            draft_prompts = [_fmt_single(main_tmpl, len(m["objects"]), _object_list(m["objects"]))
                             for m in valid]
            draft_raws = runner.generate_mm(draft_prompts, imgs, args.max_new_tokens, args.temperature)
            draft_parsed = [parse_single_relations(r, m["objects"], args.canonicalize)
                            for r, m in zip(draft_raws, valid)]
            revise_prompts = [revise_tmpl.format(object_list=_object_list(m["objects"]),
                                                 draft=_format_draft(dp[0]))
                              for m, dp in zip(valid, draft_parsed)]
            raws = runner.generate_mm(revise_prompts, imgs, args.max_new_tokens, args.temperature)
            parsed = [parse_single_relations(r, m["objects"], args.canonicalize)
                      for r, m in zip(raws, valid)]

        elif args.strategy == "grow":
            # round 1: clean diverse base; rounds 2..N: holistic ADDITIVE completion
            # (model sees image + current graph, adds only MISSING relations → escapes
            # the ~4.5 prior without per-anchor mis-attribution or synonym piles).
            base_prompts = [_fmt_single(main_tmpl, len(m["objects"]), _object_list(m["objects"]))
                            for m in valid]
            raws = runner.generate_mm(base_prompts, imgs, args.max_new_tokens, args.temperature)
            accum, oks = [], []
            for r, m in zip(raws, valid):
                rels, _d, ok = parse_single_relations(r, m["objects"], args.canonicalize)
                for rel in rels: rel["round"] = 1
                accum.append(rels); oks.append(ok)
            for rnd in range(2, args.grow_rounds + 1):
                is_spatial_round = spatial_tmpl is not None and rnd == args.grow_rounds
                tmpl = spatial_tmpl if is_spatial_round else grow_tmpl
                gprompts = [tmpl.format(object_list=_object_list(m["objects"]),
                                        draft=_format_draft(accum[vi]))
                            for vi, m in enumerate(valid)]
                graws = runner.generate_mm(gprompts, imgs, args.max_new_tokens, args.temperature)
                for vi, (g, m) in enumerate(zip(graws, valid)):
                    v2 = is_spatial_round and args.spatial_source == "llm_v2"
                    # llm_v2 parses raw: the generic canonicalizer would kill the
                    # proximity vocabulary; verify_spatial_v2 owns canon + checks
                    newr, _d, _ok = parse_single_relations(
                        g, m["objects"], False if v2 else args.canonicalize)
                    if v2:
                        newr = verify_spatial_v2(newr, m["objects"], m["img_id"],
                                                 drop_totals)
                    elif is_spatial_round:
                        # closed-set round: enforce the whitelist (collapsed form,
                        # so surface variants of an allowed axis still pass)
                        inlist, n0 = [], len(newr)
                        for r in newr:
                            if canonicalize(r["predicate"]) in SPATIAL_WHITELIST:
                                inlist.append(r)
                        drop_totals["spatial-whitelist"] += n0 - len(inlist)
                        newr = inlist
                    seen = {(r["subject_id"], r["object_id"], r["predicate"]) for r in accum[vi]}
                    for r in newr:
                        k = (r["subject_id"], r["object_id"], r["predicate"])
                        if k not in seen:
                            r["round"] = rnd
                            if is_spatial_round:
                                r["spatial"] = True
                            seen.add(k); accum[vi].append(r)
            parsed = [(accum[vi], Counter(), oks[vi]) for vi in range(len(valid))]

        elif args.strategy == "anchor_refine":
            # STAGE 1 — high-recall anchored draft (N open passes/object, aggregated).
            sub_prompts, sub_imgs, owner, anchors = [], [], [], []
            for vi, m in enumerate(valid):
                objs = m["objects"]
                for ai in range(len(objs)):
                    sub_prompts.append(main_tmpl.format(
                        n=len(objs), object_list=_object_list(objs),
                        anchor_id=ai + 1, anchor_label=objs[ai]["label"]))
                    sub_imgs.append(imgs[vi]); owner.append(vi); anchors.append(ai + 1)
            draft_raws = runner.generate_mm(sub_prompts, sub_imgs, args.max_new_tokens, args.temperature)
            draft = [([], set()) for _ in valid]            # (rels, seen) per image
            for out, vi, aid in zip(draft_raws, owner, anchors):
                kept, seen = draft[vi]
                parse_anchor_into(out, valid[vi]["objects"], aid, canon=args.canonicalize,
                                  conf_threshold=args.conf_threshold,
                                  kept=kept, drops=Counter(), seen=seen)
            # STAGE 2 — consolidate the full draft into a clean scene graph.
            revise_prompts = [revise_tmpl.format(object_list=_object_list(m["objects"]),
                                                 draft=_format_draft(draft[vi][0]))
                              for vi, m in enumerate(valid)]
            raws = runner.generate_mm(revise_prompts, imgs, args.max_new_tokens, args.temperature)
            parsed = [parse_single_relations(r, m["objects"], args.canonicalize)
                      for r, m in zip(raws, valid)]

        elif args.strategy == "relchain":
            prompts = [main_tmpl.format(n=len(m["objects"]), object_list=_object_list(m["objects"]),
                                        min_rels=_min_rels(len(m["objects"]), args.rels_formula))
                       for m in valid]
            raws = runner.generate_mm(prompts, imgs, args.max_new_tokens, args.temperature)
            parsed = [parse_relchain(r, m["objects"], args.canonicalize)
                      for r, m in zip(raws, valid)]

        elif args.strategy == "subject_anchored":
            sub_prompts, sub_imgs, owner, anchors = [], [], [], []
            for vi, m in enumerate(valid):
                objs = m["objects"]
                idxs = list(range(len(objs)))
                if args.geo_top_k > 0:
                    idxs = sorted(idxs, key=lambda i: -area(objs[i]["bbox"]))[:args.geo_top_k]
                for ai in idxs:
                    sub_prompts.append(main_tmpl.format(
                        n=len(objs), object_list=_object_list(objs),
                        anchor_id=ai + 1, anchor_label=objs[ai]["label"]))
                    sub_imgs.append(imgs[vi]); owner.append(vi); anchors.append(ai + 1)
            raws = runner.generate_mm(sub_prompts, sub_imgs, args.max_new_tokens, args.temperature)
            agg = [([], Counter(), set(), [False]) for _ in valid]
            for out, vi, aid in zip(raws, owner, anchors):
                kept, drops, seen, okref = agg[vi]
                ok = parse_anchor_into(out, valid[vi]["objects"], aid,
                                       canon=args.canonicalize, conf_threshold=args.conf_threshold,
                                       kept=kept, drops=drops, seen=seen)
                okref[0] = okref[0] or ok
            parsed = [(a[0], a[1], a[3][0]) for a in agg]

        elif args.strategy == "pairwise":
            sub_prompts, sub_imgs, owner, pmeta = [], [], [], []
            for vi, m in enumerate(valid):
                objs = m["objects"]; W, H = raw_pils[vi].size
                pairs = [(i, j) for i in range(len(objs)) for j in range(i + 1, len(objs))]
                if args.geo_top_k > 0:
                    pairs = sorted(pairs, key=lambda p: -pair_salience(
                        objs[p[0]]["bbox"], objs[p[1]]["bbox"], W, H))[:args.geo_top_k]
                elif args.pair_frac < 1.0 and pairs:
                    rng = random.Random(42 + vi)
                    pairs = rng.sample(pairs, max(1, math.ceil(len(pairs) * args.pair_frac)))
                for si, oi in pairs:
                    sub_prompts.append(main_tmpl.format(label_1=objs[si]["label"],
                                                        label_2=objs[oi]["label"]))
                    sub_imgs.append(crop_and_annotate_pair(raw_pils[vi], objs[si], objs[oi]))
                    owner.append(vi); pmeta.append((si, oi))
            raws = runner.generate_mm(sub_prompts, sub_imgs, args.max_new_tokens, args.temperature)
            agg = [([], Counter(), set(), [False]) for _ in valid]
            for out, vi, (si, oi) in zip(raws, owner, pmeta):
                kept, drops, seen, okref = agg[vi]
                ok = parse_pair_into(out, valid[vi]["objects"][si], valid[vi]["objects"][oi],
                                     si, oi, canon=args.canonicalize,
                                     conf_threshold=args.conf_threshold,
                                     kept=kept, drops=drops, seen=seen)
                okref[0] = okref[0] or ok
            parsed = [(a[0], a[1], a[3][0]) for a in agg]

        else:  # caption_first
            obs_prompts = [_fill(observer_tmpl, n=len(m["objects"]),
                                 object_list=_object_list(m["objects"])) for m in valid]
            captions = [c.strip() for c in
                        runner.generate_mm(obs_prompts, imgs, args.observer_max_tokens, args.temperature)]
            par_prompts = [_fill(parser_tmpl, object_list=_object_list(m["objects"]), caption=cap)
                           for m, cap in zip(valid, captions)]
            raws = runner.generate_text(par_prompts, args.max_new_tokens)
            parsed = [parse_caption_relations(r, m["objects"], cap, args.canonicalize)
                      for r, m, cap in zip(raws, valid, captions)]

        if args.part_gate:                            # body-part junk filter
            parsed = [(apply_part_gate(rels, m["objects"], drops), drops, ok)
                      for m, (rels, drops, ok) in zip(valid, parsed)]

        if args.contact_gate:                         # disjoint-box hallucination filter
            parsed = [(apply_contact_gate(rels, m["objects"], drops), drops, ok)
                      for m, (rels, drops, ok) in zip(valid, parsed)]

        if args.spatial_source == "geometric":        # deterministic layout layer
            parsed = [(append_geometric_spatial(rels, m["objects"], m["img_id"],
                                                args.max_spatial_per_subject),
                       drops, ok)
                      for m, (rels, drops, ok) in zip(valid, parsed)]
        elif args.geo_backstop:                       # top up sparse LLM layouts to the floor
            n_spa = lambda rels: sum(1 for r in rels
                                     if r.get("spatial") or r.get("source") == "geometric")
            parsed = [(rels if n_spa(rels) >= args.geo_backstop
                       else append_geometric_spatial(rels, m["objects"], m["img_id"],
                                                     max_per_subject=2,
                                                     limit=args.geo_backstop - n_spa(rels)),
                       drops, ok)
                      for m, (rels, drops, ok) in zip(valid, parsed)]

        if args.postprocess:                          # deterministic union cleanup
            parsed = [(_postprocess_cleanup(
                          rels, args.max_spatial_per_subject,
                          drop_spatial_with_action=not args.keep_spatial_with_action,
                          collapse_spatial=COLLAPSE_SPATIAL),
                       drops, ok) for rels, drops, ok in parsed]

        chunk_t = time.time() - t0
        per_img_t = chunk_t / len(valid)

        for m, cap, (rels, drops, ok) in zip(valid, captions, parsed):
            per_image_rels.append(rels)
            drop_totals.update(drops)
            parse_ok += int(ok)
            rec = {"img_id": m["img_id"], "file_name": m["file_name"],
                   "n_objects": len(m["objects"]), "elapsed": round(per_img_t, 3),
                   "relations": rels}
            if args.strategy == "caption_first" and not args.no_save_captions:
                rec["caption"] = cap
            fout.write(json.dumps(rec) + "\n")
        fout.flush()

        done = len(per_image_rels)
        wall = time.time() - t_start
        eta = wall / done * (n_total - done)
        rpm = sum(len(r) for r in per_image_rels[-len(valid):]) / len(valid)
        print(f"  [{done}/{n_total}] {per_img_t:.2f}s/img  rels/img {rpm:.1f}  "
              f"ETA {eta/60:.0f}m", flush=True)

    fout.close()
    metrics = compute_metrics(per_image_rels, len(per_image_rels), parse_ok)
    summary = {
        "name": args.name, "strategy": args.strategy, "model": args.model,
        "shard": [args.shard_index, args.num_shards], "overlay": args.overlay,
        "quant": args.quant, "split": args.split,
        "config": {"max_new_tokens": args.max_new_tokens,
                   "observer_max_tokens": args.observer_max_tokens,
                   "temperature": args.temperature, "chunk": args.chunk,
                   "canonicalize": args.canonicalize,
                   "keep_spatial_synonyms": args.keep_spatial_synonyms,
                   "spatial_source": args.spatial_source,
                   "geo_backstop": args.geo_backstop,
                   "contact_gate": args.contact_gate,
                   "part_gate": args.part_gate,
                   "conf_threshold": args.conf_threshold,
                   "geo_top_k": args.geo_top_k, "pair_frac": args.pair_frac,
                   "rels_formula": args.rels_formula},
        "metrics": metrics,
        "drops": dict(drop_totals),
        "timing": {"total_min": round((time.time() - t_start) / 60, 2),
                   "sec_per_img": round((time.time() - t_start) / max(len(per_image_rels), 1), 3)},
        "timestamp": datetime.now().isoformat(),
    }
    (out_dir / f"summary_shard_{args.shard_index:04d}.json").write_text(json.dumps(summary, indent=2))

    if args.emit_trial_json:
        preds = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:                              # noqa: BLE001
                    continue
                preds.append({"img_id": r["img_id"], "file_name": r["file_name"],
                              "elapsed": r.get("elapsed", 0),
                              "sg": {"relations": r.get("relations", [])}})
        (out_dir / "trial.json").write_text(json.dumps(
            {"name": args.name, "strategy": args.strategy, "model": args.model,
             "metrics": metrics, "predictions": preds}, indent=2))

    print(f"\n{'='*60}")
    print(f"  vLLM GENERATE — {args.name}  shard {args.shard_index}/{args.num_shards}")
    print(f"{'='*60}")
    print(f"  strategy/model: {args.strategy} / {args.model} (quant={args.quant})")
    print(f"  images: {len(per_image_rels)}  (parse_ok {parse_ok})")
    print(f"  sec/img: {summary['timing']['sec_per_img']}")
    print(f"  rels/img: {metrics['rels_per_img']}")
    print(f"  unique predicates: {metrics['n_unique']}")
    print(f"  entropy (nats): {metrics['entropy_nats']}")
    print(f"  forbidden rate: {metrics['forbidden_rate']*100:.1f}%")
    if drop_totals:
        print(f"  drops: {dict(drop_totals)}")
    print(f"  → {jsonl_path}")
    if args.emit_trial_json:
        print(f"  → {out_dir/'trial.json'}  (judge with datagen/llm_judge_v2.py)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
