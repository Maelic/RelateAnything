#!/usr/bin/env python3
"""MegaSG COCO masks -> pack sidecars.

The MegaSG SAM run wrote its masks back into the COCO
json, because MegaSG *has* a COCO json. The `.npy` packs do not, so training
needs the same sidecar shape the pack-based and PSG paths emit:

    {"idx": <pack image index>, "image_id":..., "rles": [rle|null,...]}

VERIFIED before writing this: `annotations` grouped by image_id keep pack box
order. On megasg_50k/val, 4,029 sampled boxes gave max unclipped coordinate
error 0.5 px (p99 1e-4), and pack box count == COCO annotation count for all
5,000 images. The join below is therefore positional, and `--check_px` asserts
it stays that way for every pack converted.

Usage
-----
  python datagen/megasg_coco_to_pack.py \
      --coco runs/sam_masks/megasg_sgg_train_coco_masks.json \
      --packs megasg:train megasg_proxy50k:train megasg_50k:train \
      --out runs/sam_masks/packs
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_mask_manifest import pack_dir            # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--coco", required=True, help="COCO json WITH segmentation filled")
    p.add_argument("--packs", nargs="+", required=True, help="ds:split...")
    p.add_argument("--out", required=True)
    p.add_argument("--check_px", type=float, default=1.0,
                   help="fail if any pack box disagrees with its COCO box by more")
    args = p.parse_args()

    print(f"loading {args.coco}...", flush=True)
    coco = json.load(open(args.coco))
    by_img: dict[int, list] = defaultdict(list)
    for a in coco["annotations"]:
        by_img[a["image_id"]].append(a)
    del coco
    print(f"  {len(by_img)} images", flush=True)

    for spec in args.packs:
        ds, split = spec.split(":")
        pd = pack_dir(spec)
        if not (pd / "img_meta.npy").exists():
            print(f"skip {spec}: no pack")
            continue

        im = np.load(pd / "img_meta.npy")
        bx = np.load(pd / "boxes.npy")
        out_dir = Path(args.out) / ds / split
        out_dir.mkdir(parents=True, exist_ok=True)

        n_box = n_hit = n_missimg = n_short = n_oob = 0
        worst = 0.0
        with (out_dir / "masks.jsonl").open("w") as f:
            for k in range(len(im)):
                iid, W, H, boff, bcnt = (int(v) for v in im[k][:5])
                anns = by_img.get(iid)
                if anns is None:
                    n_missimg += 1
                    f.write(json.dumps({"idx": k, "image_id": iid,
                                        "rles": [None] * bcnt}) + "\n")
                    n_box += bcnt
                    continue
                if len(anns) < bcnt:
                    n_short += 1

                cx, cy, w, h = bx[boff:boff + bcnt].T
                xy = np.stack([(cx - w / 2) * W, (cy - h / 2) * H,
                               (cx + w / 2) * W, (cy + h / 2) * H], axis=1)

                rles = []
                for j in range(bcnt):
                    if j >= len(anns):
                        rles.append(None)
                        n_box += 1
                        continue
                    a = anns[j]
                    x, y, ww, hh = a["bbox"]
                    # Boxes that run past the image edge are clipped by the pack
                    # builder, so they legitimately disagree with COCO by many
                    # px. Only in-bounds boxes can witness a mis-ordered join,
                    # which is what this check is actually for.
                    if x >= 0 and y >= 0 and x + ww <= W and y + hh <= H:
                        d = float(np.abs(xy[j] - np.array([x, y, x + ww, y + hh])).max())
                        worst = max(worst, d)
                    else:
                        n_oob += 1
                    seg = a.get("segmentation")
                    rles.append(seg if seg else None)
                    n_box += 1
                    n_hit += bool(seg)
                f.write(json.dumps({"idx": k, "image_id": iid, "rles": rles}) + "\n")

        pct = 100 * n_hit / max(n_box, 1)
        flag = "" if worst <= args.check_px else "   <-- JOIN SUSPECT"
        print(f"  {ds}/{split}: {n_hit}/{n_box} masks ({pct:.2f}%) | "
              f"max box err {worst:.4f} px{flag}")
        if n_missimg or n_short or n_oob:
            print(f"      images not in coco: {n_missimg} | fewer anns than boxes: "
                  f"{n_short} | out-of-bounds boxes skipped by check: {n_oob}")
        if worst > args.check_px:
            raise SystemExit(
                f"ABORT: {spec} box mismatch {worst:.3f} px > {args.check_px} px — "
                "the positional join does not hold for this pack.")


if __name__ == "__main__":
    main()
