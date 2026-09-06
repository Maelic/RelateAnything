#!/usr/bin/env python
"""export_coco_sgg.py — convert our flat per-image synthetic SGG jsonl into
the coco-style scene-graph schema (images/annotations/categories/
rel_categories/rel_annotations, global ids, predicate_id refs into
rel_categories) used by MegaSG's own GT json and targeted by
github.com/Maelic/SGG-Annotate + huggingface.co/datasets/maelic/PSG-coco-format.

Object side is reused verbatim from MegaSG's GT: our generator's relation
subject_id/object_id are LOCAL 1-based positions into the first --max_objects
GT annotations for that image (file order) — the same trimming rule as
load_boxes() in analyze_sgg_run.py and load_anno_index() in
sgg_vllm_generate.py. We resolve those local positions back to MegaSG's
GLOBAL annotations[].id and reuse the bbox/category_id/area records as-is,
so no boxes are invented and object-category ids stay in MegaSG's own space.

rel_categories is intentionally kept OPEN-VOCABULARY: one entry per unique
literal predicate string actually emitted (10k+ for train), NOT collapsed to
a closed set. Collapsing would destroy the synonym diversity this pipeline
exists to produce — see memory: keep-predicate-synonym-diversity ("canonical
forms for logic only"). predicate_id indexes this open rel_categories list;
predicate_raw/spatial/source/round are carried along as additive fields on
each rel_annotation for consumers that want the extra provenance.
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

MEGASG = DATASETS / "MEGASG"


def convert(split: str, jsonl_path: Path, out_path: Path, max_objects: int) -> None:
    print(f"Loading MegaSG {split} GT …", flush=True)
    gt = json.load(open(MEGASG / split / "_annotations.coco.json"))
    images_by_id = {im["id"]: im for im in gt["images"]}
    categories = gt["categories"]

    anns_by_img: dict = defaultdict(list)
    for a in gt["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    for iid in anns_by_img:
        anns_by_img[iid] = anns_by_img[iid][:max_objects]

    out_images, out_annotations, out_rels = [], [], []
    pred_to_id: dict = {}
    rel_categories: list = []
    next_rel_id = 1
    n_skipped_img = n_skipped_rel = 0

    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            iid = rec["img_id"]
            im = images_by_id.get(iid)
            objs = anns_by_img.get(iid, [])
            if im is None or len(objs) < rec["n_objects"]:
                n_skipped_img += 1
                continue
            out_images.append(im)
            local_to_global = {i + 1: a["id"] for i, a in enumerate(objs)}
            out_annotations.extend(objs)

            for r in rec.get("relations", []):
                s_g = local_to_global.get(r["subject_id"])
                o_g = local_to_global.get(r["object_id"])
                if s_g is None or o_g is None:
                    n_skipped_rel += 1
                    continue
                p = r["predicate"]
                pid = pred_to_id.get(p)
                if pid is None:
                    pid = len(rel_categories)
                    pred_to_id[p] = pid
                    rel_categories.append({"id": pid, "name": p})
                rel = {
                    "id": next_rel_id,
                    "image_id": iid,
                    "subject_id": s_g,
                    "object_id": o_g,
                    "predicate_id": pid,
                }
                if r.get("spatial"):
                    rel["spatial"] = True
                if r.get("source"):
                    rel["source"] = r["source"]
                if r.get("round") is not None:
                    rel["round"] = r["round"]
                if r.get("predicate_raw"):
                    rel["predicate_raw"] = r["predicate_raw"]
                out_rels.append(rel)
                next_rel_id += 1

    out = {
        "info": {
            "description": f"MegaSG {split} — open-vocabulary synthetic SGG "
                            "(gemma-4-26B-A4B-it, recipe F2/engine C2)",
            "images_source": str(MEGASG / split / "_annotations.coco.json"),
            "relations_source": str(jsonl_path),
        },
        "images": out_images,
        "annotations": out_annotations,
        "categories": categories,
        "rel_categories": rel_categories,
        "rel_annotations": out_rels,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f)

    print(f"[{split}] images={len(out_images)} (skipped {n_skipped_img}) "
          f"annotations={len(out_annotations)} "
          f"rel_annotations={len(out_rels)} (skipped {n_skipped_rel}) "
          f"rel_categories={len(rel_categories)}")
    print(f"  → {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max_objects", type=int, default=40)
    args = ap.parse_args()
    convert(args.split, args.jsonl, args.out, args.max_objects)
