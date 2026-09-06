#!/usr/bin/env python3
"""Union manifest + scatter for pack-based SAM mask generation.

The `.npy` packs have no `segmentation` field and no COCO json, so the
box-prompted SAM path needs a different join than MegaSG's. Two extra facts make
a plain per-pack loop wasteful:

  * several packs point at the SAME image root (vg_raw/train and vg150/train are
    both `VG150_coco_format/train`; 87% of vg_raw's images sit inside vg150), and
  * SAM's cost is dominated by the vision encoder, which runs ONCE per image
    regardless of how many boxes are prompted.

So we encode the union of images, not the concatenation of packs. A SAM mask is
a deterministic function of (image, box), which is what makes sharing masks
across packs exactly correct rather than an approximation.

Three steps:

  build    packs -> union.jsonl (unique images, deduped boxes) + refs/*.npz
  (gpu)    sam3_masks.py --union  -> masks_shard_*.jsonl keyed by union index
  scatter  union shards + refs    -> runs/sam_masks/packs/<ds>/<split>/masks.jsonl

The scatter output is byte-for-byte the same shape as `psg_gt_masks.py` emits —
`{"idx": <pack image index>, "image_id":..., "rles": [rle|null,...]}` in pack
box order — so downstream code never learns which masks came from GT and which
from SAM.

Usage
-----
  python datagen/pack_mask_manifest.py build \
      --packs vg150:train vg150:val vg150:test vg_raw:train... \
      --manifest runs/sam_masks/packs/_union

  python datagen/pack_mask_manifest.py scatter \
      --manifest runs/sam_masks/packs/_union \
      --shards runs/sam_masks/packs/_union/shards \
      --out runs/sam_masks/packs
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PACK_ROOT = Path("runs/packed")

# Box dedup granularity. Packs derived from the same source often carry the same
# box to within float32 round-trip; 0.1 px is far below anything SAM resolves.
BOX_QUANT = 10.0


def pack_dir(spec: str) -> Path:
    ds, split = spec.split(":")
    return PACK_ROOT / ds / split


def load_pack(spec: str):
    """-> (file_names, img_meta, boxes_norm, img_dir). Boxes are normalized cxcywh."""
    p = pack_dir(spec)
    fn = json.load(open(p / "file_names.json"))
    im = np.load(p / "img_meta.npy")
    bx = np.load(p / "boxes.npy")
    img_dir = json.load(open(p / "meta.json"))["img_dir"]
    return fn, im, bx, Path(img_dir)


def to_xyxy(boxes_norm: np.ndarray, W: int, H: int) -> np.ndarray:
    """Normalized cxcywh -> absolute xyxy, clipped to the image."""
    cx, cy, w, h = boxes_norm.T
    xy = np.stack([(cx - w / 2) * W, (cy - h / 2) * H,
                   (cx + w / 2) * W, (cy + h / 2) * H], axis=1)
    xy[:, 0::2] = xy[:, 0::2].clip(0, W)
    xy[:, 1::2] = xy[:, 1::2].clip(0, H)
    return xy


def build(args) -> None:
    manifest = Path(args.manifest)
    (manifest / "refs").mkdir(parents=True, exist_ok=True)

    # union image key -> union index; per-image box table
    key_to_u: dict[str, int] = {}
    u_path: list[str] = []
    u_wh: list[tuple[int, int]] = []
    u_boxes: list[list[list[float]]] = []
    u_boxkey: list[dict[tuple, int]] = []

    n_deg = 0
    for spec in args.packs:
        fn, im, bx, img_dir = load_pack(spec)
        n_img = len(fn)
        ref_u = np.full(n_img, -1, dtype=np.int64)
        # Proxy packs are VIEWS: they subset img_meta but keep the parent's full
        # boxes.npy, so box offsets index the parent array. Size by len(bx), not
        # by the subset's box count.
        ref_b = np.full(len(bx), -1, dtype=np.int64)

        for k in range(n_img):
            _, W, H, boff, bcnt, _, _ = (int(v) for v in im[k])
            path = str(img_dir / fn[k])
            u = key_to_u.get(path)
            if u is None:
                u = len(u_path)
                key_to_u[path] = u
                u_path.append(path)
                u_wh.append((W, H))
                u_boxes.append([])
                u_boxkey.append({})
            ref_u[k] = u

            if bcnt == 0:
                continue
            # Boxes are stored normalized, so a pack that recorded a different
            # (W,H) for the same file still lands on the same pixels here.
            xy = to_xyxy(bx[boff:boff + bcnt], *u_wh[u])
            for j in range(bcnt):
                x1, y1, x2, y2 = (float(v) for v in xy[j])
                if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                    n_deg += 1              # sub-pixel: SAM would return garbage
                    continue
                qk = (round(x1 * BOX_QUANT), round(y1 * BOX_QUANT),
                      round(x2 * BOX_QUANT), round(y2 * BOX_QUANT))
                b = u_boxkey[u].get(qk)
                if b is None:
                    b = len(u_boxes[u])
                    u_boxkey[u][qk] = b
                    u_boxes[u].append([x1, y1, x2, y2])
                ref_b[boff + j] = b

        np.savez_compressed(manifest / "refs" / f"{spec.replace(':', '__')}.npz",
                            u=ref_u, b=ref_b, image_id=im[:, 0].astype(np.int64))
        print(f"  {spec}: {n_img} imgs, {len(ref_b)} boxes "
              f"-> union now {len(u_path)} imgs", flush=True)

    with (manifest / "union.jsonl").open("w") as f:
        for u in range(len(u_path)):
            f.write(json.dumps({"u": u, "path": u_path[u],
                                "wh": list(u_wh[u]), "boxes": u_boxes[u]}) + "\n")

    n_union_box = sum(len(b) for b in u_boxes)
    n_pack_box = 0
    for spec in args.packs:
        r = np.load(manifest / "refs" / f"{spec.replace(':', '__')}.npz")
        n_pack_box += len(r["b"])

    json.dump({"packs": args.packs, "n_union_images": len(u_path),
               "n_union_boxes": n_union_box, "n_pack_boxes": n_pack_box,
               "n_degenerate": n_deg},
              open(manifest / "manifest_meta.json", "w"), indent=2)

    print(f"\nunion: {len(u_path)} images, {n_union_box} boxes")
    print(f"packs: {sum(len(json.load(open(pack_dir(s) / 'file_names.json'))) for s in args.packs)}"
          f" image-passes, {n_pack_box} boxes")
    print(f"saved:  {sum(len(json.load(open(pack_dir(s) / 'file_names.json'))) for s in args.packs) - len(u_path)}"
          f" image encodes, {n_pack_box - n_union_box - n_deg} box decodes")
    print(f"degenerate (sub-pixel) boxes dropped: {n_deg}")
    print(f"-> {manifest / 'union.jsonl'}")


def refs_only(args) -> None:
    """Build refs for extra packs against an ALREADY-SEGMENTED union.

    Subset packs (vg_raw_proxy15k ⊂ vg_raw, megasg_50k ⊂ megasg) carry images and
    boxes the union already covers, so they need no GPU — only the lookup table.
    Nothing is inserted: a box the union never saw resolves to null and is
    counted, so a genuinely-new pack reports a low hit rate instead of silently
    producing empty masks.
    """
    manifest = Path(args.manifest)
    meta = json.load(open(manifest / "manifest_meta.json"))

    key_to_u, u_boxkey = {}, {}
    for line in (manifest / "union.jsonl").open():
        r = json.loads(line)
        key_to_u[r["path"]] = r["u"]
        u_boxkey[r["u"]] = {
            (round(b[0] * BOX_QUANT), round(b[1] * BOX_QUANT),
             round(b[2] * BOX_QUANT), round(b[3] * BOX_QUANT)): i
            for i, b in enumerate(r["boxes"])}

    for spec in args.packs:
        fn, im, bx, img_dir = load_pack(spec)
        ref_u = np.full(len(fn), -1, dtype=np.int64)
        ref_b = np.full(len(bx), -1, dtype=np.int64)   # parent-sized; see build()
        n_img_miss = n_box_miss = n_box = 0

        for k in range(len(fn)):
            _, W, H, boff, bcnt, _, _ = (int(v) for v in im[k])
            u = key_to_u.get(str(img_dir / fn[k]))
            if u is None:
                n_img_miss += 1
                n_box += bcnt
                n_box_miss += bcnt
                continue
            ref_u[k] = u
            if bcnt == 0:
                continue
            xy = to_xyxy(bx[boff:boff + bcnt], W, H)
            for j in range(bcnt):
                x1, y1, x2, y2 = (float(v) for v in xy[j])
                n_box += 1
                b = u_boxkey[u].get((round(x1 * BOX_QUANT), round(y1 * BOX_QUANT),
                                     round(x2 * BOX_QUANT), round(y2 * BOX_QUANT)))
                if b is None:
                    n_box_miss += 1
                else:
                    ref_b[boff + j] = b

        np.savez_compressed(manifest / "refs" / f"{spec.replace(':', '__')}.npz",
                            u=ref_u, b=ref_b, image_id=im[:, 0].astype(np.int64))
        hit = 100 * (n_box - n_box_miss) / max(n_box, 1)
        print(f"  {spec}: {len(fn)} imgs ({n_img_miss} not in union), "
              f"{n_box - n_box_miss}/{n_box} boxes resolved ({hit:.2f}%)")
        if spec not in meta["packs"]:
            meta["packs"].append(spec)

    json.dump(meta, open(manifest / "manifest_meta.json", "w"), indent=2)


def scatter(args) -> None:
    manifest = Path(args.manifest)
    meta = json.load(open(manifest / "manifest_meta.json"))
    if args.packs:
        meta = {**meta, "packs": args.packs}

    # union index -> list of rles (aligned with union.jsonl box order)
    masks: dict[int, list] = {}
    shards = sorted(Path(args.shards).glob("masks_shard_*.jsonl"))
    if not shards:
        raise SystemExit(f"no shards under {args.shards}")
    for sp in shards:
        with sp.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue            # truncated final line from a killed job
                masks[rec["u"]] = rec["rles"]
    print(f"loaded {len(masks)}/{meta['n_union_images']} union images "
          f"from {len(shards)} shards")

    for spec in meta["packs"]:
        ds, split = spec.split(":")
        r = np.load(manifest / "refs" / f"{spec.replace(':', '__')}.npz")
        ref_u, ref_b, img_id = r["u"], r["b"], r["image_id"]
        im = np.load(pack_dir(spec) / "img_meta.npy")

        out_dir = Path(args.out) / ds / split
        out_dir.mkdir(parents=True, exist_ok=True)
        n_box = n_hit = 0
        with (out_dir / "masks.jsonl").open("w") as f:
            for k in range(len(ref_u)):
                boff, bcnt = int(im[k][3]), int(im[k][4])
                got = masks.get(int(ref_u[k]))
                rles = []
                for j in range(bcnt):
                    b = int(ref_b[boff + j])
                    rle = got[b] if (got is not None and 0 <= b < len(got)) else None
                    rles.append(rle)
                    n_box += 1
                    n_hit += rle is not None
                f.write(json.dumps({"idx": k, "image_id": int(img_id[k]),
                                    "rles": rles}) + "\n")
        pct = 100 * n_hit / max(n_box, 1)
        print(f"  {ds}/{split}: {n_hit}/{n_box} masks ({pct:.2f}%) "
              f"-> {out_dir / 'masks.jsonl'}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--packs", nargs="+", required=True, help="ds:split...")
    b.add_argument("--manifest", required=True)

    r = sub.add_parser("refs", help="lookup-only refs against a frozen union")
    r.add_argument("--packs", nargs="+", required=True)
    r.add_argument("--manifest", required=True)

    s = sub.add_parser("scatter")
    s.add_argument("--manifest", required=True)
    s.add_argument("--shards", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--packs", nargs="+", default=None,
                   help="restrict to these packs (default: all in the manifest)")

    args = p.parse_args()
    {"build": build, "refs": refs_only, "scatter": scatter}[args.cmd](args)


if __name__ == "__main__":
    main()
