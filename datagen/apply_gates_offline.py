#!/usr/bin/env python3
"""
apply_gates_offline.py — re-apply the deterministic gates to an EXISTING run's
jsonl (no GPU). Used to validate new gate rules (part gate, canon drops) on
already-generated relations before baking them into a production run.

Usage:
  python datagen/apply_gates_offline.py \
      --run runs/vllm_generate/calib_final_E_iter20_geo \
      --out runs/vllm_generate/calib_final_E_gated
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from sgg_canon import canonicalize, canonicalize_spatial
from sgg_vllm_generate import apply_part_gate, apply_contact_gate, load_anno_index
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

MEGASG = DATASETS / "MEGASG"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Indexing annotations …", flush=True)
    anno = MEGASG / args.split / "_annotations.coco.json"
    metas = {m["img_id"]: m for m in load_anno_index(anno, require_relations=True,
                                                     max_objects=40)}

    drops = Counter()
    n_before = n_after = n_img = 0
    with open(args.run / "shard_0000.jsonl") as f, \
         open(args.out / "shard_0000.jsonl", "w") as w:
        for line in f:
            rec = json.loads(line)
            m = metas.get(rec["img_id"])
            rels = rec.get("relations", [])
            n_before += len(rels)
            # canon drops (possessing & co); predicates are already surface-canon,
            # so re-checking for None is the only effect — never rewrite them.
            # Spatial-layer rels use the proximity-aware canonicalizer ("near"
            # from the verified llm_v2 round is signal, not filler).
            kept = []
            for r in rels:
                canon_fn = canonicalize_spatial if r.get("spatial") else canonicalize
                if (r.get("source") != "geometric"
                        and canon_fn(r.get("predicate", "")) is None):
                    drops["canon-drop"] += 1
                    continue
                kept.append(r)
            if m is not None:
                kept = apply_part_gate(kept, m["objects"], drops)
                kept = apply_contact_gate(kept, m["objects"], drops)
            rec["relations"] = kept
            n_after += len(kept)
            n_img += 1
            w.write(json.dumps(rec) + "\n")

    print(f"  {n_img} images: {n_before/max(n_img,1):.2f} → {n_after/max(n_img,1):.2f} rels/img")
    print(f"  drops: {dict(drops)}")
    print(f"  → {args.out}/shard_0000.jsonl")


if __name__ == "__main__":
    main()
