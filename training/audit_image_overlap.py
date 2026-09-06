"""Content-level image-overlap audit: perceptual-hash every training image and
compare against the protected PSG val+test images.

The id-level dedup (build_datamix_registry.py) is exact where an id bridge
exists (VG<->COCO via image_data.json, PSG<->COCO via psg.json) but cannot see:
  * MEGASG (Objects365) photos that happen to be COCO/Flickr photos — no bridge
  * VG photos that are COCO photos but have coco_id=None in VG metadata
  * genuinely duplicated photos inside COCO itself (distinct coco ids)

This closes those gaps with a 64-bit dHash (difference hash) over 9x8 grayscale
thumbnails: exact photo re-encodes/resizes land at Hamming distance 0-6. We
report dist==0 as duplicates and dist<=8 as suspects for manual review.

Hashes are cached to runs/datamix/hashes/<set>.npz — reruns are instant.

Usage:
    python training/audit_image_overlap.py --sets psg_eval psg_train_used vg_used
    python training/audit_image_overlap.py --sets megasg_train   # the big one
    python training/audit_image_overlap.py --compare             # after hashing
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset and run roots come from relsgg.paths (RA_DATASETS / RA_RUNS).
MIX = DATAMIX
HDIR = MIX / "hashes"


UNREADABLE = 0xFFFFFFFFFFFFFFFF  # uint64-safe sentinel for corrupt images


def dhash64(path: str) -> int:
    """64-bit difference hash; UNREADABLE sentinel on corrupt image."""
    try:
        with Image.open(path) as im:
            im.draft("L", (72, 72))          # fast JPEG DCT-domain downscale
            g = im.convert("L").resize((9, 8), Image.BILINEAR)
        a = np.asarray(g, dtype=np.int16)
        bits = (a[:, 1:] > a[:,:-1]).flatten()
        h = 0
        for b in bits:
            h = (h << 1) | int(b)
        return h
    except Exception:
        return UNREADABLE


def mix_file_names(*jsons) -> set:
    out = set()
    for j in jsons:
        out |= {im["file_name"] for im in json.load(open(MIX / j))["images"]}
    return out


def set_paths(name: str) -> list:
    if name == "psg_eval":
        return [str(DATASETS / f"PSG_coco_format/{s}" / f)
                for s in ("val", "test")
                for f in sorted(os.listdir(DATASETS / f"PSG_coco_format/{s}"))
                if f.endswith(".jpg")]
    if name == "psg_train_used":
        fns = mix_file_names("svg_psg_train_coco.json", "asv2_train_coco.json")
        d = DATASETS / "PSG_coco_format/train"
        return [str(d / f) for f in sorted(fns)]
    if name == "psg_train_all":
        # every PSG train photo — target of the psg-image-free mix policy
        d = DATASETS / "PSG_coco_format/train"
        return [str(d / f) for f in sorted(os.listdir(d)) if f.endswith(".jpg")]
    if name == "spatialsense":
        d = DATASETS / "SpatialSense/images"
        # skip macOS AppleDouble junk ("._*") shipped inside the tarball
        return [str(p) for p in sorted(d.rglob("*.jpg"))
                if not p.name.startswith("._")]
    if name == "vg150_eval":
        return [str(DATASETS / f"VG150_coco_format/{s}" / f)
                for s in ("val", "test")
                for f in sorted(os.listdir(DATASETS / f"VG150_coco_format/{s}"))
                if f.endswith(".jpg")]
    if name == "vg_used":
        # union of whichever VG-image mix files currently exist
        fns = set()
        for j in ("svg_vg_train_coco.json", "gqa_train_coco.json",
                  "vg150_train_dedup_coco.json"):
            if (MIX / j).exists():
                fns |= mix_file_names(j)
        d = DATASETS / "VG150_coco_format/train"
        return [str(d / f) for f in sorted(fns)]
    if name == "megasg_train":
        d = DATASETS / "MEGASG/train"
        return [str(d / f) for f in sorted(os.listdir(d))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if name == "megasg_oi_ext":
        # new Open Images V6 extension (not part of original MEGASG) — combined
        # train/val/test download pool, flat list across the 3 subdirs
        root = DATASETS / "MEGASG_OI_ext"
        return [str(root / s / f) for s in ("train", "val", "test")
                for f in sorted(os.listdir(root / s)) if f.endswith(".jpg")]
    raise ValueError(name)


def build(name: str, workers: int) -> None:
    out = HDIR / f"{name}.npz"
    if out.exists():
        print(f"[{name}] cached ({out})")
        return
    paths = set_paths(name)
    print(f"[{name}] hashing {len(paths)} images with {workers} workers...")
    with Pool(workers) as p:
        hashes = p.map(dhash64, paths, chunksize=256)
    h = np.array(hashes, dtype=np.uint64)
    names = np.array([os.path.basename(x) for x in paths])
    n_bad = int((h == np.uint64(UNREADABLE)).sum())
    np.savez_compressed(out, hashes=h, names=names)
    print(f"[{name}] done -> {out}  (unreadable: {n_bad})")


def compare(max_report: int = 30) -> None:
    ev = np.load(HDIR / "psg_eval.npz")
    eh, en = ev["hashes"], ev["names"]
    print(f"protected PSG eval set: {len(eh)} images\n")
    for name in ("psg_train_used", "vg_used", "megasg_train"):
        f = HDIR / f"{name}.npz"
        if not f.exists():
            print(f"[{name}] not hashed yet — skipped")
            continue
        d = np.load(f)
        th, tn = d["hashes"], d["names"]
        # min Hamming distance of each train hash to any eval hash, chunked
        best = np.full(len(th), 64, dtype=np.uint8)
        arg = np.zeros(len(th), dtype=np.int32)
        for i in range(0, len(th), 8192):
            x = th[i:i + 8192, None] ^ eh[None,:]
            dist = np.bitwise_count(x).astype(np.uint8)
            best[i:i + 8192] = dist.min(axis=1)
            arg[i:i + 8192] = dist.argmin(axis=1)
        exact = np.where(best == 0)[0]
        near = np.where((best > 0) & (best <= 8))[0]
        print(f"[{name}] {len(th)} imgs | exact dupes (d=0): {len(exact)} | "
              f"suspects (1<=d<=8): {len(near)}")
        for i in list(exact)[:max_report]:
            print(f"    DUP  {tn[i]}  ==  {en[arg[i]]}")
        for i in list(near)[:max_report]:
            print(f"    d={best[i]}  {tn[i]}  ~  {en[arg[i]]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="*", default=[])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    HDIR.mkdir(parents=True, exist_ok=True)
    for s in args.sets:
        build(s, args.workers)
    if args.compare:
        compare()


if __name__ == "__main__":
    main()
