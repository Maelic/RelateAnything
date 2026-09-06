#!/usr/bin/env python3
"""QA the per-pack mask sidecars produced by pack_mask_manifest.py scatter.

Checks the three things that can go wrong silently:

  1. COVERAGE    — every pack box got a mask (nulls should equal the known
                   sub-pixel count, nothing more).
  2. ALIGNMENT   — bbox(mask) sits inside the prompting box. A join bug that
                   handed back another object's mask shows up here as a
                   collapsed containment ratio, where coverage alone reads 100%.
  3. FILL RATIO  — mask area / box area. This is the quantity the whole exercise
                   is about: it measures how much of each box is background that
                   box-pooling currently averages in.

Compare against the MegaSG SAM run (99.1% inside box, fill 0.56) and PSG's
ground truth (fill 0.535).

Usage
-----
  python datagen/check_pack_masks.py --root runs/sam_masks/packs \
      --packs vg150:train vg_raw:train... --sample 2000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_mask_manifest import load_pack, to_xyxy   # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="scatter output root")
    p.add_argument("--packs", nargs="+", required=True)
    p.add_argument("--sample", type=int, default=2000, help="images per pack for geometry")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print(f"{'pack':<26} {'boxes':>9} {'masks':>9} {'cover':>7} "
          f"{'inside':>7} {'fill':>6} {'empty':>6}")
    print("-" * 76)

    tot_box = tot_mask = 0
    for spec in args.packs:
        ds, split = spec.split(":")
        path = Path(args.root) / ds / split / "masks.jsonl"
        if not path.exists():
            print(f"{spec:<26} MISSING {path}")
            continue

        fn, im, bx, _ = load_pack(spec)
        lines = path.read_text().splitlines()

        n_box = n_mask = 0
        for line in lines:
            rec = json.loads(line)
            n_box += len(rec["rles"])
            n_mask += sum(r is not None for r in rec["rles"])

        # geometry on a random sample — decoding every mask would be minutes
        rng = random.Random(args.seed)
        idxs = rng.sample(range(len(lines)), min(args.sample, len(lines)))
        inside, fills, n_empty_px = [], [], 0
        for k in idxs:
            rec = json.loads(lines[k])
            _, W, H, boff, bcnt, _, _ = (int(v) for v in im[k])
            xy = to_xyxy(bx[boff:boff + bcnt], W, H)
            for j, rle in enumerate(rec["rles"]):
                if rle is None:
                    continue
                r = dict(rle)
                r["counts"] = r["counts"].encode() if isinstance(r["counts"], str) \
                    else r["counts"]
                m = mask_util.decode(r).astype(bool)
                area = float(m.sum())
                if area == 0:
                    n_empty_px += 1
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in xy[j])
                x1, y1 = max(x1, 0), max(y1, 0)
                x2, y2 = min(x2, m.shape[1]), min(y2, m.shape[0])
                in_box = float(m[y1:y2, x1:x2].sum())
                inside.append(in_box / area)
                fills.append(in_box / max((x2 - x1) * (y2 - y1), 1))

        cov = 100 * n_mask / max(n_box, 1)
        print(f"{spec:<26} {n_box:>9,} {n_mask:>9,} {cov:>6.2f}% "
              f"{100*np.mean(inside):>6.1f}% {np.mean(fills):>6.3f} {n_empty_px:>6}")
        tot_box += n_box
        tot_mask += n_mask

    print("-" * 76)
    print(f"{'TOTAL':<26} {tot_box:>9,} {tot_mask:>9,} "
          f"{100*tot_mask/max(tot_box,1):>6.2f}%")


if __name__ == "__main__":
    main()
