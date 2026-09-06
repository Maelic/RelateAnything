#!/usr/bin/env python3
"""Ground-truth PSG masks from COCO panoptic PNGs — no SAM inference.

PSG annotates relations on top of COCO's panoptic segmentations ("relations are
mask-to-mask", OpenPSG README), so its masks are real annotations, not
predictions. `psg.json` carries only `segments_info` = {id, category_id, area,
...}; the mask itself lives in the RGB-encoded panoptic PNG, where

    segment_id = R + 256*G + 256*256*B

VERIFIED before writing this: for `runs/packed/psg/val`, `segments_info[i]`
corresponds to pack box `i` — 19,038 boxes compared, max coordinate error
2.97e-08. The join below is therefore positional.

PNGs are streamed straight out of the COCO zips, so no image files are ever
written to disk. Output is one JSONL per pack split, keyed by pack image index:

    {"idx": 0, "image_id": "107899", "rles": [{...} | null,...]}

Usage
-----
  python datagen/psg_gt_masks.py --pack runs/packed/psg --splits train val test \
      --out runs/sam_masks/packs/psg --workers 16
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_util
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

# Dataset roots come from relsgg.paths (RA_DATASETS).
PSG_JSON = str(DATASETS / "PSG_coco_format/psg.json")
ZIP_DIR = str(DATASETS / "COCO_panoptic")

_ZIPS: dict[str, zipfile.ZipFile] = {}


def _init_worker(zip_dir: str) -> None:
    """Each worker needs its own ZipFile handle — they are not fork-safe."""
    for split in ("train", "val"):
        p = Path(zip_dir) / f"panoptic_{split}2017.zip"
        if p.exists():
            _ZIPS[f"panoptic_{split}2017"] = zipfile.ZipFile(p)


def _read_png(pan_name: str) -> np.ndarray | None:
    """pan_name like 'panoptic_train2017/000000417720.png' → HxW int32 id map."""
    root = pan_name.split("/")[0]
    z = _ZIPS.get(root)
    if z is None:
        return None
    for cand in (pan_name, pan_name.split("/")[-1]):
        try:
            raw = z.read(cand)
            break
        except KeyError:
            continue
    else:
        return None
    rgb = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint32)
    return (rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]).astype(np.int32)


def _encode(m: np.ndarray) -> dict:
    rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _job(task):
    """task = (idx, image_id, pan_name, [segment_id,...], n_boxes)"""
    idx, image_id, pan_name, seg_ids, n_boxes = task
    ids = _read_png(pan_name)
    if ids is None:
        return idx, image_id, None, "png_missing"

    rles, n_empty = [], 0
    for sid in seg_ids:
        m = ids == sid
        if not m.any():
            rles.append(None)
            n_empty += 1
        else:
            rles.append(_encode(m))
    # pad if the pack holds more boxes than segments (should not happen)
    while len(rles) < n_boxes:
        rles.append(None)
        n_empty += 1
    return idx, image_id, rles[:n_boxes], n_empty


def build_tasks(pack_split: Path, psg: dict):
    fn = json.load(open(pack_split / "file_names.json"))
    im = np.load(pack_split / "img_meta.npy")
    tasks, missing = [], 0
    for k in range(len(fn)):
        image_id = str(im[k][0])
        rec = psg.get(image_id)
        if rec is None:
            missing += 1
            continue
        n_boxes = int(im[k][4])
        seg_ids = [s["id"] for s in rec["segments_info"]][:n_boxes]
        tasks.append((k, image_id, rec["pan_seg_file_name"], seg_ids, n_boxes))
    return tasks, missing


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True, help="e.g. runs/packed/psg")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--out", required=True)
    p.add_argument("--zip_dir", default=ZIP_DIR)
    p.add_argument("--psg_json", default=PSG_JSON)
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    print("loading psg.json...", flush=True)
    psg = {r["image_id"]: r for r in json.load(open(args.psg_json))["data"]}
    print(f"  {len(psg)} PSG records", flush=True)

    for split in args.splits:
        ps = Path(args.pack) / split
        if not (ps / "img_meta.npy").exists():
            print(f"skip {split}: no pack", flush=True)
            continue

        tasks, missing = build_tasks(ps, psg)
        out_dir = Path(args.out) / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "masks.jsonl"

        print(f"\n{split}: {len(tasks)} images ({missing} not in psg.json)", flush=True)
        t0 = time.time()
        n_box = n_empty = n_fail = 0

        with mp.Pool(args.workers, initializer=_init_worker,
                     initargs=(args.zip_dir,)) as pool, out_path.open("w") as f:
            for i, (idx, image_id, rles, info) in enumerate(
                    pool.imap_unordered(_job, tasks, chunksize=64), 1):
                if rles is None:
                    n_fail += 1
                    continue
                n_box += len(rles)
                n_empty += int(info)
                f.write(json.dumps({"idx": idx, "image_id": image_id,
                                    "rles": rles}) + "\n")
                if i % 5000 == 0:
                    r = i / (time.time() - t0)
                    print(f"  {i}/{len(tasks)}  {r:.0f} img/s", flush=True)

        dt = time.time() - t0
        print(f"  DONE {split}: {n_box} masks in {dt/60:.1f} min "
              f"({len(tasks)/max(dt,1e-9):.0f} img/s)", flush=True)
        print(f"  empty segments: {n_empty} | png missing: {n_fail}", flush=True)
        print(f"  -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
