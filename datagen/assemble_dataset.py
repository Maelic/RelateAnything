#!/usr/bin/env python
"""Assemble sharded sgg_vllm_generate.py runs into one final dataset JSONL.

Verifies each run before merging:
  * every shard summary present and shard counts consistent (a missing summary
    means the shard never finished — resubmit that array index, resume skips
    done images);
  * coverage against the MegaSG annotation index (same inclusion rule as the
    generator: images with >=2 boxes, optional GT-relation filter) — missing
    img_ids are written to <out>.missing.json;
  * unparseable lines (torn writes from killed jobs) are skipped, duplicates
    (resume rewrites) are deduplicated last-wins per (split, img_id).

Emits one record per image {img_id, split, file_name, n_objects, relations}
plus <out>.stats.json (per-split + overall predicate stats, drop counters,
GPU-hours). Object boxes are NOT embedded — join on img_id/annotation ids with
the MegaSG COCO file, as everywhere else in the project.

Usage (full 500K production):
  python datagen/assemble_dataset.py \\
      runs/vllm_generate/megasg_26b_F2_full_train \\
      runs/vllm_generate/megasg_26b_F2_full_val \\
      --out runs/vllm_generate/megasg_sgg_500k.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MEGASG_DIR = str(DATASETS / "MEGASG")


def expected_img_ids(megasg_dir: str, split: str, require_relations: bool) -> set:
    """Image ids the generator would annotate — mirrors load_anno_index()
    inclusion in sgg_vllm_generate.py (>=2 boxes, optional GT-rel filter)."""
    with open(Path(megasg_dir) / split / "_annotations.coco.json") as f:
        data = json.load(f)
    n_obj: dict = defaultdict(int)
    for a in data["annotations"]:
        n_obj[a["image_id"]] += 1
    rel_imgs = {r["image_id"] for r in data.get("rel_annotations", [])}
    return {img["id"] for img in data["images"]
            if n_obj[img["id"]] >= 2
            and (not require_relations or img["id"] in rel_imgs)}


def is_spatial(rel: dict) -> bool:
    return bool(rel.get("spatial")) or rel.get("source") == "geometric"


def scan_run(run_dir: Path):
    """Read summaries + index shard files. Returns (split, num_shards,
    summaries, shard_paths, last_seen) where last_seen maps
    img_id -> (shard_path, line_no) of the record to keep (last wins)."""
    summaries = {}
    for sp in sorted(run_dir.glob("summary_shard_*.json")):
        s = json.loads(sp.read_text())
        summaries[s["shard"][0]] = s

    shard_paths = sorted(run_dir.glob("shard_*.jsonl"))
    if not shard_paths:
        sys.exit(f"ERROR: no shard_*.jsonl in {run_dir}")
    if not summaries:
        sys.exit(f"ERROR: no summary_shard_*.json in {run_dir} — no shard ever "
                 "finished; nothing to verify against.")

    splits = {s["split"] for s in summaries.values()}
    num_shards_set = {s["shard"][1] for s in summaries.values()}
    if len(splits) > 1 or len(num_shards_set) > 1:
        sys.exit(f"ERROR: inconsistent summaries in {run_dir}: "
                 f"splits={splits} num_shards={num_shards_set}")
    split, num_shards = splits.pop(), num_shards_set.pop()

    missing_summaries = sorted(set(range(num_shards)) - set(summaries))
    if missing_summaries:
        print(f"  WARNING: {run_dir.name}: shards without a summary (never "
              f"finished): {missing_summaries}")

    last_seen: dict = {}
    torn = dup = 0
    for path in shard_paths:
        with open(path) as f:
            for i, line in enumerate(f):
                try:
                    iid = json.loads(line)["img_id"]
                except Exception:                              # noqa: BLE001
                    torn += 1
                    continue
                if iid in last_seen:
                    dup += 1
                last_seen[iid] = (path, i)
    if torn or dup:
        print(f"  note: {run_dir.name}: skipped {torn} torn line(s), "
              f"deduplicated {dup} rewritten record(s) (last wins)")
    return split, num_shards, summaries, shard_paths, last_seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="run directories. Multiple dirs may share a split (e.g. "
                         "incremental --skip/--limit chunks of the same train split) "
                         "— their coverage and stats are merged, not overwritten. "
                         "On img_id collision across runs, the later argument wins.")
    ap.add_argument("--out", default=str(REPO_ROOT / "runs/vllm_generate/megasg_sgg_500k.jsonl"))
    ap.add_argument("--megasg_dir", default=MEGASG_DIR)
    ap.add_argument("--require_relations", action="store_true",
                    help="verify against GT-rel-filtered index (calibration runs "
                         "used it; the full production run does NOT)")
    ap.add_argument("--allow_partial", action="store_true",
                    help="assemble even with missing images (still reported)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Scan every run dir, then GROUP by split — multiple run dirs can be disjoint
    # chunks of the same split (e.g. incremental --skip/--limit batches), and their
    # coverage/stats must accumulate rather than the later one overwriting the first.
    by_split: dict = defaultdict(list)     # split -> [(run_dir, summaries, shard_paths, last_seen),...]
    for rd in args.runs:
        run_dir = Path(rd) if Path(rd).is_absolute() else REPO_ROOT / rd
        print(f"Scanning {run_dir} …")
        split, num_shards, summaries, shard_paths, last_seen = scan_run(run_dir)
        print(f"  split={split}  shards={len(shard_paths)}/{num_shards}  images={len(last_seen)}")
        by_split[split].append((run_dir, summaries, shard_paths, last_seen))

    # Winner map per split: (path, line_no) to keep for each img_id. Runs are
    # applied in --runs argument order, so a later run wins on img_id collision
    # (matches the intra-run resume "last wins" policy).
    winner: dict = {}                      # split -> {img_id: (path, line_no)}
    all_missing: dict = {}
    for split, contributors in by_split.items():
        w: dict = {}
        for _run_dir, _summaries, _shard_paths, last_seen in contributors:
            w.update(last_seen)
        winner[split] = w
        expected = expected_img_ids(args.megasg_dir, split, args.require_relations)
        missing = expected - set(w)
        extra = set(w) - expected
        n_runs = len(contributors)
        print(f"[{split}] {n_runs} run(s) merged  images={len(w)}/{len(expected)}  "
              f"missing={len(missing)}  unexpected={len(extra)}")
        if extra:
            print(f"  WARNING: {len(extra)} images outside the expected index "
                  "(different --require_relations at generation vs verification?)")
        if missing:
            all_missing[split] = sorted(missing)

    if all_missing:
        miss_path = out_path.with_suffix(".missing.json")
        miss_path.write_text(json.dumps(all_missing, indent=2))
        n = sum(len(v) for v in all_missing.values())
        print(f"\n{n} images missing → {miss_path}")
        if not args.allow_partial:
            sys.exit("ERROR: incomplete run(s). Resubmit the unfinished shard "
                     "indices (resume skips done images), or pass --allow_partial.")

    # Merge (second pass: emit only the winning (path, line) per img_id per split)
    stats = {}
    drop_totals: Counter = Counter()
    gpu_min = 0.0
    n_out = 0
    with open(out_path, "w") as fout:
        for split, contributors in by_split.items():
            w = winner[split]
            preds: Counter = Counter()
            n_img = n_rel = n_spa = n_zero = 0
            for _run_dir, summaries, shard_paths, _last_seen in contributors:
                for s in summaries.values():
                    drop_totals.update(s.get("drops", {}))
                    gpu_min += s["timing"]["total_min"]
                for path in shard_paths:
                    with open(path) as f:
                        for i, line in enumerate(f):
                            try:
                                r = json.loads(line)
                            except Exception:                  # noqa: BLE001
                                continue
                            if w.get(r["img_id"]) != (path, i):
                                continue                       # superseded elsewhere
                            rels = r.get("relations", [])
                            fout.write(json.dumps(
                                {"img_id": r["img_id"], "split": split,
                                 "file_name": r["file_name"],
                                 "n_objects": r.get("n_objects"),
                                 "relations": rels}) + "\n")
                            n_out += 1
                            n_img += 1
                            n_rel += len(rels)
                            n_zero += not rels
                            for x in rels:
                                preds[x["predicate"]] += 1
                                n_spa += is_spatial(x)
            H = -sum(p * math.log(p) for p in
                     (c / n_rel for c in preds.values()) if p > 0) if n_rel else 0.0
            stats[split] = {
                "run_dirs": [str(rd) for rd, *_ in contributors],
                "images": n_img, "relations": n_rel,
                "rels_per_img": round(n_rel / max(n_img, 1), 2),
                "zero_rel_images": n_zero,
                "unique_predicates": len(preds),
                "entropy_nats": round(H, 3),
                "pct_spatial": round(100 * n_spa / max(n_rel, 1), 1),
                "top20_predicates": preds.most_common(20),
                "missing_images": len(all_missing.get(split, [])),
            }

    stats_out = {
        "output": str(out_path), "total_images": n_out,
        "per_split": stats,
        "drops_from_summaries": dict(drop_totals),   # approximate under resume
        "gpu_hours_from_summaries": round(gpu_min / 60, 1),
    }
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats_out, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  ASSEMBLED → {out_path}  ({n_out} images)")
    for split, m in stats.items():
        print(f"  [{split}] {m['images']} imgs  {m['rels_per_img']} rels/img  "
              f"{m['unique_predicates']} uniq preds  H={m['entropy_nats']}  "
              f"{m['pct_spatial']}% spatial  zero-rel={m['zero_rel_images']}")
    print(f"  GPU-hours (summaries): {stats_out['gpu_hours_from_summaries']}")
    print(f"  stats → {stats_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
