#!/usr/bin/env python3
"""
invert_spatial.py — Post-process a spatial trial.json to add symmetric inverse relations.

For each canonical spatial predicate, appends the inverse triplet (subject/object swapped).
Near-free data augmentation — no additional VLM inference required.

Inverse map:
  above           <-> below
  to the left of  <-> to the right of
  in front of     <-> behind
  on top of       <-> beneath
  (inside has no clean inverse and is skipped)

Writes a new trial.json into --outdir/<output_name>/ that siglip_viz.py and
vqascore_eval.py can score normally.

Usage:
  python training/invert_spatial.py \\
    training/runs/strategy_ablation/spatial_pairwise_geo4_50_bs2_e4b \\
    --outdir training/runs/strategy_ablation \\
    --name counterfactual_spatial_50_bs2_e4b
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

INVERSE = {
    "above":           "below",
    "below":           "above",
    "to the left of":  "to the right of",
    "to the right of": "to the left of",
    "in front of":     "behind",
    "behind":          "in front of",
    "on top of":       "beneath",
    "beneath":         "on top of",
}


def quick_metrics(predictions: list[dict]) -> dict:
    pred_count: Counter = Counter()
    total_rels = 0
    times = []
    parse_ok = 0
    for p in predictions:
        times.append(p.get("elapsed", 0.0))
        sg = p.get("sg")
        if sg:
            parse_ok += 1
            for rel in sg.get("relations", []):
                pred = rel.get("predicate", "").lower().strip()
                if pred:
                    pred_count[pred] += 1
                    total_rels += 1
    n = len(predictions)
    rels_per_img = total_rels / max(n, 1)
    n_unique = len(pred_count)
    if total_rels > 0:
        probs = [c / total_rels for c in pred_count.values()]
        entropy = -sum(p * math.log(p) for p in probs if p > 0)
    else:
        entropy = 0.0
    top5 = pred_count.most_common(5)
    top5_cov = sum(c for _, c in top5) / max(total_rels, 1)
    return {
        "n_images": n,
        "parse_ok": parse_ok,
        "total_rels": total_rels,
        "rels_per_img": round(rels_per_img, 1),
        "n_unique": n_unique,
        "entropy_nats": round(entropy, 3),
        "forbidden_rate": 0.0,
        "n_forbidden": 0,
        "top5_coverage": round(top5_cov, 3),
        "top5": list(top5),
        "all_predicates": dict(pred_count.most_common()),
        "median_s": round(statistics.median(times) if times else 0.0, 2),
        "truncated": 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Add inverse spatial relations to a trial")
    parser.add_argument("source_dir", help="Path to the source trial directory")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: same parent as source_dir)")
    parser.add_argument("--name", default=None,
                        help="Output trial name (default: <source_name>_inverted)")
    args = parser.parse_args()

    src = Path(args.source_dir)
    trial_path = src / "trial.json"
    if not trial_path.exists():
        print(f"ERROR: {trial_path} not found")
        return 1

    trial = json.load(open(trial_path))
    src_name = trial.get("name", src.name)
    out_name = args.name or f"{src_name}_inverted"
    out_parent = Path(args.outdir) if args.outdir else src.parent
    out_dir = out_parent / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_orig = 0
    n_added = 0
    new_predictions = []

    for pred_item in trial["predictions"]:
        sg = pred_item.get("sg")
        if not sg:
            new_predictions.append(pred_item)
            continue

        orig_rels = sg.get("relations", [])
        new_rels = list(orig_rels)
        seen = {
            (r["subject_id"], r.get("predicate", "").lower(), r["object_id"])
            for r in orig_rels
        }

        n_orig += len(orig_rels)
        for rel in orig_rels:
            pred = rel.get("predicate", "").lower().strip()
            inv_pred = INVERSE.get(pred)
            if inv_pred is None:
                continue
            inv_key = (rel["object_id"], inv_pred, rel["subject_id"])
            if inv_key in seen:
                continue  # already present
            seen.add(inv_key)
            n_added += 1
            new_rels.append({
                "subject_id":    rel["object_id"],
                "subject_label": rel.get("object_label", ""),
                "predicate":     inv_pred,
                "object_id":     rel["subject_id"],
                "object_label":  rel.get("subject_label", ""),
                "inverted":      True,
            })

        new_sg = dict(sg, relations=new_rels)
        new_predictions.append(dict(pred_item, sg=new_sg))

    metrics = quick_metrics(new_predictions)

    output = {
        "name":        out_name,
        "mode":        "counterfactual_spatial",
        "source":      str(src),
        "model":       trial.get("model"),
        "config":      trial.get("config", {}),
        "metrics":     metrics,
        "timestamp":   datetime.now().isoformat(),
        "predictions": new_predictions,
    }

    out_file = out_dir / "trial.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    # Also copy image_list.json so siglip_viz / vqascore_eval can find images
    src_imglist = src / "image_list.json"
    if src_imglist.exists():
        import shutil
        shutil.copy(src_imglist, out_dir / "image_list.json")

    print(f"Source:   {src_name}  ({n_orig} relations)")
    print(f"Inverted: {out_name}  ({n_orig + n_added} relations, +{n_added} added)")
    print(f"Saved → {out_dir}/")


if __name__ == "__main__":
    main()
