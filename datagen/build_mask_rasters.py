#!/usr/bin/env python3
"""Precompute low-res coverage rasters from mask sidecars.

Decoding RLEs inside `__getitem__` would starve the GPU — training is
compute-bound at ~121 img/s/A40, and 100 full-res RLE decodes per image does not
fit in that budget. So rasters are built once, offline, and mmap'd at train time
exactly like `boxes.npy`.

A "coverage raster" is the region's area fraction per cell of a fixed g x g grid,
quantized to uint8. This is the single representation the model consumes: a BOX
rasterizes to its (analytic, anti-aliased) rectangle, a MASK to its own shape.
Nothing downstream needs to know which it got — see relsgg/roi.py.

Output (aligned to the pack's ABSOLUTE box offsets, which is what img_meta
carries, so view-packs like megasg_proxy50k index the parent array correctly):

    cov.npy        uint8  [n_present, g, g]   area fraction * 255
    cov_index.npy  int32  [n_parent_boxes]    box offset -> row in cov.npy, -1 = none
    fill.npy       float16[n_present]         mask area / box area
    meta.json

Storing a compact array plus an index (rather than a dense parent-sized one)
matters for view packs: megasg_proxy50k references 276,790 of its parent's
2,595,560 box slots, so dense would be 2.7 GB of mostly zeros against 283 MB.

Usage
-----
  python datagen/build_mask_rasters.py --packs megasg_proxy50k:train \
      --masks_root runs/sam_masks/packs --out runs/sam_masks/rasters --res 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_mask_manifest import load_pack, pack_dir     # noqa: E402


def box_raster(x1: float, y1: float, x2: float, y2: float, g: int) -> np.ndarray:
    """Analytic area-coverage of an axis-aligned box over a g x g unit grid.

    This is the box branch of the rasterizer, and it is exact (not sampled), so
    a box and a mask of the same region produce commensurable rasters.
    """
    edges = np.arange(g + 1, dtype=np.float64) / g
    lo, hi = edges[:-1], edges[1:]
    ix = np.clip(np.minimum(x2, hi) - np.maximum(x1, lo), 0, None)
    iy = np.clip(np.minimum(y2, hi) - np.maximum(y1, lo), 0, None)
    return np.outer(iy, ix) * (g * g)          # cell area fraction in [0, 1]


def _bounds(n: int, g: int) -> np.ndarray:
    """Start row/col of each of the g output cells over an n-length axis."""
    return np.minimum((np.arange(g) * n) // g, n - 1)


def mask_raster(m: np.ndarray, g: int) -> np.ndarray:
    """Area-average a HxW binary mask down to g x g (exact box filter).

    Separable reduceat rather than scatter-add: this runs once per mask over
    2.3M masks, and np.add.at on an HxW index array is ~100x slower.
    """
    H, W = m.shape
    yb, xb = _bounds(H, g), _bounds(W, g)
    rows = np.add.reduceat(m, yb, axis=0)              # [g, W]
    acc = np.add.reduceat(rows, xb, axis=1)            # [g, g]
    ycnt = np.diff(np.append(yb, H)).astype(np.float64)
    xcnt = np.diff(np.append(xb, W)).astype(np.float64)
    return acc / np.maximum(np.outer(ycnt, xcnt), 1.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--packs", nargs="+", required=True)
    p.add_argument("--masks_root", default="runs/sam_masks/packs")
    p.add_argument("--out", default="runs/sam_masks/rasters")
    p.add_argument("--res", type=int, default=32)
    args = p.parse_args()
    g = args.res

    for spec in args.packs:
        ds, split = spec.split(":")
        sidecar = Path(args.masks_root) / ds / split / "masks.jsonl"
        if not sidecar.exists():
            print(f"skip {spec}: no {sidecar}")
            continue

        fn, im, bx, _ = load_pack(spec)
        n_parent = len(bx)
        cov_index = np.full(n_parent, -1, dtype=np.int32)

        rows, fills = [], []
        t0 = time.time()
        n_mask = n_boxfallback = 0

        with sidecar.open() as f:
            for line_no, line in enumerate(f):
                rec = json.loads(line)
                k = rec["idx"]
                boff, bcnt = int(im[k][3]), int(im[k][4])
                cx, cy, w, h = bx[boff:boff + bcnt].T

                for j in range(min(bcnt, len(rec["rles"]))):
                    rle = rec["rles"][j]
                    if rle is None:
                        # No mask: leave -1. The loader rasterizes the box
                        # instead, which is the correct region, not a fallback.
                        n_boxfallback += 1
                        continue
                    r = dict(rle)
                    if isinstance(r["counts"], str):
                        r["counts"] = r["counts"].encode()
                    m = mask_util.decode(r)
                    if m.ndim == 3:
                        m = m[..., 0]
                    if not m.any():
                        n_boxfallback += 1
                        continue
                    raster = mask_raster(m.astype(np.float64), g)
                    box_area = float(w[j]) * float(h[j]) * m.shape[0] * m.shape[1]
                    cov_index[boff + j] = len(rows)
                    rows.append(np.clip(raster * 255.0, 0, 255).astype(np.uint8))
                    fills.append(float(m.sum()) / max(box_area, 1.0))
                    n_mask += 1

                if (line_no + 1) % 5000 == 0:
                    r = (line_no + 1) / (time.time() - t0)
                    print(f"  {spec}: {line_no+1}/{len(fn)} imgs  {r:.0f} img/s",
                          flush=True)

        out_dir = Path(args.out) / ds / split
        out_dir.mkdir(parents=True, exist_ok=True)
        cov = (np.stack(rows) if rows
               else np.zeros((0, g, g), dtype=np.uint8))
        np.save(out_dir / "cov.npy", cov)
        np.save(out_dir / "cov_index.npy", cov_index)
        np.save(out_dir / "fill.npy", np.asarray(fills, dtype=np.float16))
        json.dump({"pack": spec, "res": g, "n_masks": n_mask,
                   "n_parent_boxes": int(n_parent),
                   "n_box_fallback": n_boxfallback,
                   "mean_fill": float(np.mean(fills)) if fills else 0.0},
                  open(out_dir / "meta.json", "w"), indent=2)

        mb = cov.nbytes / 1e6
        print(f"  {spec}: {n_mask} rasters ({mb:.0f} MB), "
              f"{n_boxfallback} box-fallback, mean fill "
              f"{np.mean(fills) if fills else 0:.3f}, "
              f"{time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
