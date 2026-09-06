#!/usr/bin/env python
"""pack_megasg.py — pack a COCO-SGG json (images/annotations/categories/
rel_categories/rel_annotations, global-id refs — the schema produced by
export_coco_sgg.py and used by VG150/PSG/IndoorVG *_coco_format) into flat
memmap arrays for fast training startup.

Why: the merged MEGASG train json is 957MB and takes minutes + ~8GB RAM to
parse per job; the packed form is memmapped in milliseconds. All id
resolution (global ann id → per-image local box position) and the
xywh-pixel → normalized-cxcywh conversion happen once, here, with asserts —
downstream code never re-derives them.

Output layout  <out>/<split>/:
    meta.json        dataset/split info, predicate + category vocabularies,
                     raw-predicate links (for synonym-map building), stats
    file_names.json  [N_img] image file names (aligned with img_meta rows)
    img_meta.npy     int64  [N_img, 7]  (img_id, width, height,
                                          box_start, n_boxes, rel_start, n_rels)
    boxes.npy        float32 [ΣN, 4]    normalized cxcywh, clamped to [0,1]
    box_cats.npy     int32   [ΣN]       contiguous category index (see meta)
    rels.npy         int32   [ΣR, 5]    (sub_local, obj_local, pred_id,
                                          flags, raw_pred_id or -1)

flags bitfield: bit0 = spatial, bit1 = source=="geometric", bits2+ = round.

Usage:
    python training/pack_megasg.py --preset megasg
    python training/pack_megasg.py --preset megasg --limit 10000 --out runs/packed/megasg_10k
    python training/pack_megasg.py --preset vg150
    python training/pack_megasg.py --train_ann A.json --val_ann B.json \
        --train_img_dir DIR --val_img_dir DIR --out runs/packed/custom
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).

PRESETS = {
    "megasg": {
        "ann": str(RUNS / "vllm_generate/coco_format/megasg_sgg_{split}_coco.json"),
        "img_dir": str(DATASETS / "MEGASG/{split}"),
        "out": str(RUNS / "packed/megasg"),
    },
    "vg150": {
        "ann": str(DATASETS / "VG150_coco_format/{split}/_annotations.coco.json"),
        "img_dir": str(DATASETS / "VG150_coco_format/{split}"),
        "out": str(RUNS / "packed/vg150"),
    },
    "psg": {
        "ann": str(DATASETS / "PSG_coco_format/{split}/_annotations.coco.json"),
        "img_dir": str(DATASETS / "PSG_coco_format/{split}"),
        "out": str(RUNS / "packed/psg"),
    },
    # Haystack has one split only, and its ann path carries no {split} token.
    # Pack it as "test" so min_rels lands at 0 — 9,151 of its images have ONLY
    # negative annotations (kept in a sidecar) and must not be dropped.
    "haystack": {
        "ann": str(RUNS / "datamix/haystack_test_coco.json"),
        "img_dir": str(DATASETS / "Haystack/images"),
        "out": str(RUNS / "packed/haystack"),
    },
    "indoorvg": {
        "ann": str(DATASETS / "IndoorVG_coco_format/{split}/_annotations.coco.json"),
        "img_dir": str(DATASETS / "IndoorVG_coco_format/{split}"),
        "out": str(RUNS / "packed/indoorvg"),
    },
}

FLAG_SPATIAL = 1
FLAG_GEOMETRIC = 2
ROUND_SHIFT = 2


def pack_split(
    name: str,
    split: str,
    ann_path: Path,
    img_dir: Path,
    out_dir: Path,
    max_objects: int,
    min_rels: int,
    limit: int | None,
    exclude_file_names: set | None = None,
) -> None:
    print(f"[{name}/{split}] loading {ann_path} …", flush=True)
    d = json.load(open(ann_path))
    if exclude_file_names:
        n0 = len(d["images"])
        d["images"] = [im for im in d["images"]
                       if im["file_name"] not in exclude_file_names]
        print(f"[{name}/{split}] excluded {n0 - len(d['images'])} images "
              f"(eval-overlap registry)")

    # ---- vocabularies -------------------------------------------------
    # rel_categories ids are contiguous 0-based in our exports but arbitrary
    # (unsorted, 1-based) in VG150/PSG — remap to file-order indices.
    rel_cats = d["rel_categories"]
    predicates = [rc["name"] for rc in rel_cats]
    rel_id_to_idx = {rc["id"]: i for i, rc in enumerate(rel_cats)}

    categories = d["categories"]  # ids may be non-contiguous (MEGASG starts at 1)
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}

    # raw-predicate side vocabulary (MEGASG only; absent elsewhere)
    raw_vocab: dict[str, int] = {}
    raw_links: Counter = Counter()  # (raw, canonical-surface) observed pairs

    # ---- group annotations per image (file order — this IS the local order)
    anns_by_img: dict = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    rels_by_img: dict = defaultdict(list)
    for r in d["rel_annotations"]:
        rels_by_img[r["image_id"]].append(r)

    images = d["images"]
    seen_ids = set()
    for im in images:
        assert im["id"] not in seen_ids, f"duplicate image id {im['id']}"
        seen_ids.add(im["id"])
    if limit is not None:
        images = images[:limit]

    # ---- pack ----------------------------------------------------------
    img_meta, file_names = [], []
    boxes_out, cats_out, rels_out = [], [], []
    n_drop_img_fewbox = n_drop_img_norel = 0
    n_drop_rel_range = n_drop_rel_self = 0
    n_degenerate_boxes = 0
    pred_counts: Counter = Counter()

    for im in images:
        iid = im["id"]
        anns = anns_by_img.get(iid, [])[:max_objects]
        if len(anns) < 2:
            n_drop_img_fewbox += 1
            continue

        gid_to_local = {a["id"]: i for i, a in enumerate(anns)}
        W, H = float(im["width"]), float(im["height"])
        assert W > 0 and H > 0, f"image {iid} has invalid size {W}x{H}"

        img_rels = []
        for r in rels_by_img.get(iid, []):
            s = gid_to_local.get(r["subject_id"])
            o = gid_to_local.get(r["object_id"])
            if s is None or o is None:          # beyond max_objects window
                n_drop_rel_range += 1
                continue
            if s == o:
                n_drop_rel_self += 1
                continue
            pid = rel_id_to_idx[r["predicate_id"]]
            flags = 0
            if r.get("spatial"):
                flags |= FLAG_SPATIAL
            if r.get("source") == "geometric":
                flags |= FLAG_GEOMETRIC
            flags |= int(r.get("round", 0)) << ROUND_SHIFT
            raw = r.get("predicate_raw")
            if raw:
                rid = raw_vocab.setdefault(raw, len(raw_vocab))
                raw_links[(raw, predicates[pid])] += 1
            else:
                rid = -1
            img_rels.append((s, o, pid, flags, rid))
            pred_counts[pid] += 1

        if len(img_rels) < min_rels:
            n_drop_img_norel += 1
            continue

        img_boxes = np.empty((len(anns), 4), dtype=np.float32)
        for i, a in enumerate(anns):
            x, y, w, h = a["bbox"]
            # COCO xywh with a negative extent (6 PSG-train boxes, 2026-08-27)
            # puts (x, y) at the far corner: the centre is still x + w/2, the
            # extent is |w|. Without this the geometry log-features go NaN.
            w, h = abs(w), abs(h)
            if w <= 1 or h <= 1:
                n_degenerate_boxes += 1
            # xywh pixels → normalized cxcywh; clamp (never drop: relations
            # reference local positions, dropping would shift indices)
            cx = np.clip((x + w * 0.5) / W, 0.0, 1.0)
            cy = np.clip((y + h * 0.5) / H, 0.0, 1.0)
            img_boxes[i] = (cx, cy, min(w / W, 1.0), min(h / H, 1.0))
            cats_out.append(cat_id_to_idx[a["category_id"]])

        img_meta.append((iid, im["width"], im["height"],
                         len(boxes_out), len(anns), len(rels_out), len(img_rels)))
        file_names.append(im["file_name"])
        boxes_out.extend(img_boxes)
        rels_out.extend(img_rels)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "img_meta.npy", np.asarray(img_meta, dtype=np.int64))
    np.save(out_dir / "boxes.npy", np.asarray(boxes_out, dtype=np.float32))
    np.save(out_dir / "box_cats.npy", np.asarray(cats_out, dtype=np.int32))
    np.save(out_dir / "rels.npy",
            np.asarray(rels_out, dtype=np.int32).reshape(-1, 5))
    json.dump(file_names, open(out_dir / "file_names.json", "w"))

    raw_predicates = [None] * len(raw_vocab)
    for s, i in raw_vocab.items():
        raw_predicates[i] = s

    meta = {
        "dataset": name,
        "split": split,
        "ann_source": str(ann_path),
        "img_dir": str(img_dir),
        "max_objects": max_objects,
        "min_rels": min_rels,
        "num_images": len(img_meta),
        "num_boxes": len(boxes_out),
        "num_rels": len(rels_out),
        "predicates": predicates,
        "predicate_counts": {predicates[p]: c for p, c in pred_counts.most_common()},
        "categories": [c["name"] for c in categories],
        "category_ids": [c["id"] for c in categories],
        "raw_predicates": raw_predicates,
        "raw_links": [{"raw": a, "predicate": b, "count": c}
                      for (a, b), c in raw_links.most_common()],
        "flags_legend": {"bit0": "spatial", "bit1": "source==geometric",
                         "bits2+": "round"},
        "drops": {
            "images_fewer_than_2_boxes": n_drop_img_fewbox,
            "images_below_min_rels": n_drop_img_norel,
            "rels_beyond_max_objects": n_drop_rel_range,
            "rels_self_loop": n_drop_rel_self,
            "degenerate_boxes_kept": n_degenerate_boxes,
        },
    }
    json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)

    print(f"[{name}/{split}] images={len(img_meta)} boxes={len(boxes_out)} "
          f"rels={len(rels_out)} predicates={len(predicates)} "
          f"drops={meta['drops']}")
    print(f"  → {out_dir}")


def verify_roundtrip(ann_path: Path, out_dir: Path, n_samples: int = 100) -> None:
    """Re-derive n_samples packed images from the source json and compare."""
    import random

    d = json.load(open(ann_path))
    anns_by_img = defaultdict(list)
    for a in d["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    images_by_id = {im["id"]: im for im in d["images"]}

    meta = json.load(open(out_dir / "meta.json"))
    img_meta = np.load(out_dir / "img_meta.npy")
    boxes = np.load(out_dir / "boxes.npy", mmap_mode="r")
    rels = np.load(out_dir / "rels.npy", mmap_mode="r")

    rng = random.Random(0)
    rows = rng.sample(range(len(img_meta)), min(n_samples, len(img_meta)))
    for row in rows:
        iid, W, H, b0, nb, r0, nr = img_meta[row]
        im = images_by_id[iid]
        anns = anns_by_img[iid][: meta["max_objects"]]
        assert nb == len(anns), f"img {iid}: box count {nb} != {len(anns)}"
        for i, a in enumerate(anns):
            x, y, w, h = a["bbox"]
            # COCO xywh with a negative extent (6 PSG-train boxes, 2026-08-27)
            # puts (x, y) at the far corner: the centre is still x + w/2, the
            # extent is |w|. Without this the geometry log-features go NaN.
            w, h = abs(w), abs(h)
            cx = np.clip((x + w * 0.5) / im["width"], 0, 1)
            cy = np.clip((y + h * 0.5) / im["height"], 0, 1)
            assert np.allclose(boxes[b0 + i],
                               [cx, cy, min(w / im["width"], 1), min(h / im["height"], 1)],
                               atol=1e-6), f"img {iid} box {i} mismatch"
        for k in range(nr):
            s, o, pid, flags, rid = rels[r0 + k]
            assert 0 <= s < nb and 0 <= o < nb and s != o
            assert 0 <= pid < len(meta["predicates"])
    print(f"round-trip OK on {len(rows)} sampled images")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--train_ann"), ap.add_argument("--val_ann")
    ap.add_argument("--train_img_dir"), ap.add_argument("--val_img_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--max_objects", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None,
                    help="Pack only the first N images (subset builds).")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--verify", type=int, default=100,
                    help="Round-trip-verify N sampled images per split (0 = skip).")
    ap.add_argument("--exclude_registry", default=None,
                    help="datamix registry.json; its megasg_train_exclude "
                         "file names are dropped from the TRAIN split "
                         "(eval-image overlap, see audit_image_overlap.py).")
    args = ap.parse_args()

    exclude = set()
    if args.exclude_registry:
        exclude = set(json.load(open(args.exclude_registry))
.get("megasg_train_exclude", []))
        print(f"exclusion registry: {len(exclude)} file names")

    if args.preset:
        p = PRESETS[args.preset]
        name = args.name or args.preset
        out_root = Path(args.out or p["out"])
        cfg = {s: (Path(p["ann"].format(split=s)), Path(p["img_dir"].format(split=s)))
               for s in args.splits}
    else:
        assert args.train_ann and args.out, "--preset or (--train_ann + --out) required"
        name = args.name or "custom"
        out_root = Path(args.out)
        cfg = {}
        if "train" in args.splits:
            cfg["train"] = (Path(args.train_ann), Path(args.train_img_dir))
        if "val" in args.splits and args.val_ann:
            cfg["val"] = (Path(args.val_ann), Path(args.val_img_dir))

    for split, (ann, img_dir) in cfg.items():
        # train images without relations teach nothing (GT slots drive every
        # loss term); keep them only in eval splits
        min_rels = 1 if split == "train" else 0
        out_dir = out_root / split
        pack_split(name, split, ann, img_dir, out_dir,
                   args.max_objects, min_rels, args.limit,
                   exclude_file_names=(exclude if split == "train" else None))
        if args.verify:
            verify_roundtrip(ann, out_dir, args.verify)


if __name__ == "__main__":
    main()
