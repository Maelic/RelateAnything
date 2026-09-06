#!/usr/bin/env python3
"""Box-prompted mask generation for MegaSG (SAM 3 / SAM 2.1).

MegaSG ships boxes only (`segmentation: []` in the COCO export). This script
fills that field by prompting a promptable segmenter with each *existing* box,
so masks come back 1:1 with annotation ids — entity ids, categories and every
`rel_annotation` stay valid. That is the whole reason we prompt with boxes
instead of running automatic mask generation: SAM's automatic mode emits ~100
part/subpart masks per image with no correspondence to our 5.6 entities.

Two backends, identical call surface:
  sam3  facebook/sam3            — GATED; needs an approved account AND a local
                                   token. `hf auth login` writes it to
                                   $HF_HOME/token, so HF_HOME must match at
                                   submit time or the job 401s despite access.
  sam2  facebook/sam2.1-hiera-large — ungated fallback

Measured on one A40 (100 val images): 6.44 img/s, 601/601 masks
non-empty, 99.1% of mask area inside the prompt box, mean SAM score 0.906.
Full MegaSG (499K images) is therefore ~21.5 GPU-h — ~1.4 h wall on a 16-way
array. Output costs ~1,367 B/mask: 3.7 GB train + 0.2 GB val.

Output is one JSONL shard per array task (never an in-place edit of the big
COCO json), so the job is resumable, parallel-safe and cheap to ship between
clusters. `--merge` folds the shards back into a COCO json at the end.

Usage
-----
  # benchmark throughput on 200 images before committing to the full run
  python datagen/sam3_masks.py --coco runs/vllm_generate/coco_format/megasg_sgg_val_coco.json \
      --images../DATASETS/MEGASG/val --out runs/sam_masks/val --limit 200

  # one shard of a 16-way array
  python datagen/sam3_masks.py --coco... --images... --out... \
      --shard $SLURM_ARRAY_TASK_ID --num_shards 16

  # fold shards into a COCO json with `segmentation` populated
  python datagen/sam3_masks.py --merge --coco <in.json> --out runs/sam_masks/val \
      --merged_coco runs/sam_masks/megasg_val_coco_masks.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_util

DEFAULT_MODELS = {
    "sam3": "facebook/sam3",
    "sam2": "facebook/sam2.1-hiera-large",
}

# NOT facebook/sam3.1: that repo publishes only `sam3.1_multiplex.pt` (no
# safetensors / pytorch_model.bin), so `from_pretrained` cannot load it. No loss
# here — 3.1's gains are video-inference speed, and box prompting runs through
# the `sam3_tracker` branch that both releases share.
SAM3_FALLBACK_IDS = ["facebook/sam3"]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def load_backend(backend: str, model_id: str, device: str, dtype: torch.dtype):
    """Return (model, processor). Both backends expose the same SAM interface:
    processor(images=..., input_boxes=...) → model(**inputs) →.pred_masks.

    For sam3 we try the requested id first, then the ids in SAM3_FALLBACK_IDS:
    `facebook/sam3.1` publishes only `sam3.1_multiplex.pt` (no safetensors), so
    it may not be loadable by `from_pretrained` while `facebook/sam3` is.
    """
    if backend == "sam3":
        from transformers import Sam3TrackerModel as Model, Sam3TrackerProcessor as Proc
        candidates = [model_id] + [m for m in SAM3_FALLBACK_IDS if m != model_id]
    elif backend == "sam2":
        from transformers import Sam2Model as Model, Sam2Processor as Proc
        candidates = [model_id]
    else:
        raise ValueError(f"unknown backend {backend!r}")

    errors = []
    for cand in candidates:
        try:
            processor = Proc.from_pretrained(cand)
            model = Model.from_pretrained(cand, dtype=dtype).to(device).eval()
            if cand != model_id:
                print(f"  note: fell back to {cand} ({model_id} did not load)", flush=True)
            return model, processor
        except Exception as e:
            errors.append(f"  {cand}: {type(e).__name__}: {e}")

    msg = "\n".join(errors)
    raise RuntimeError(
        f"could not load any {backend} checkpoint:\n{msg}\n\n"
        "If these are 401/403: the repo is gated — request access on the hub AND\n"
        "authenticate locally with `hf auth login` (or export HF_TOKEN=...).\n"
        "To start work immediately, use --backend sam2 (ungated)."
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def xywh_to_xyxy_clipped(bbox, width: int, height: int):
    """COCO xywh → xyxy clipped to the image. Returns None if degenerate.

    Degenerate boxes (sub-pixel after clipping) are dropped rather than fed to
    the model: SAM returns garbage for them, so a null RLE is the honest result.
    """
    x, y, w, h = (float(v) for v in bbox)
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]


def encode_rle(mask_bool: np.ndarray) -> dict:
    """Binary HxW mask → COCO RLE with a JSON-safe (str) counts field."""
    rle = mask_util.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

@torch.inference_mode()
def segment_image(model, processor, image: Image.Image, boxes_xyxy, device, dtype,
                  multimask: bool):
    """Prompt with every box of one image in a single forward.

    The vision encoder dominates cost and runs once per image regardless of box
    count, so per-image batching is already near-optimal.

    Returns a list of (rle | None, score) aligned with ``boxes_xyxy``.
    """
    inputs = processor(images=image, input_boxes=[boxes_xyxy], return_tensors="pt")
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    outputs = model(**inputs, multimask_output=multimask)

    masks = processor.post_process_masks(
        outputs.pred_masks.float().cpu(), inputs["original_sizes"]
)[0]                                    # [n_boxes, n_masks, H, W] or [n_boxes, H, W]
    scores = outputs.iou_scores.float().cpu()

    if masks.ndim == 3:                     # no mask dimension → add one
        masks = masks.unsqueeze(1)
    while scores.ndim > 2:                  # [1, n_boxes, n_masks] → [n_boxes, n_masks]
        scores = scores[0]
    if scores.ndim == 1:
        scores = scores.unsqueeze(-1)

    results = []
    for i in range(masks.shape[0]):
        # multimask=True returns ambiguity candidates; keep the highest-IoU one.
        best = int(torch.argmax(scores[i])) if masks.shape[1] > 1 else 0
        m = masks[i, best].numpy().astype(bool)
        results.append((encode_rle(m) if m.any() else None, float(scores[i, best])))
    return results


# ---------------------------------------------------------------------------
# Sharding / resume
# ---------------------------------------------------------------------------

def load_done_ids(path: Path, key: str = "image_id") -> set:
    """Ids already written, so a requeued job resumes instead of redoing."""
    if not path.exists():
        return set()
    done = set()
    with path.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)[key])
            except Exception:
                continue          # truncated final line from a killed job
    return done


def check(args) -> None:
    """Load the model and segment one synthetic box. Validates auth, weights and
    tensor shapes in ~1 min, so a 16-way array never fails on task 0."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_id = args.model_id or DEFAULT_MODELS[args.backend]

    from huggingface_hub import get_token
    print(f"backend={args.backend}  model={model_id}  device={device}")
    print(f"HF token present: {'yes' if (get_token() or os.environ.get('HF_TOKEN')) else 'NO'}")

    t0 = time.time()
    model, processor = load_backend(args.backend, model_id, device, dtype)
    print(f"loaded in {time.time()-t0:.1f}s")

    img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    res = segment_image(model, processor, img, [[100.0, 80.0, 300.0, 400.0]],
                        device, dtype, args.multimask)
    rle, score = res[0]
    area = int(mask_util.area(# type: ignore[arg-type]
        {"size": rle["size"], "counts": rle["counts"].encode()})) if rle else 0
    print(f"OK — 1 box → rle size={rle['size'] if rle else None} "
          f"area={area}px score={score:.3f}")
    print("preflight passed; safe to launch the array.")


def run(args) -> None:
    coco = json.load(open(args.coco))
    by_image = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    images = sorted(coco["images"], key=lambda im: im["id"])
    images = [im for im in images if by_image.get(im["id"])]
    images = images[args.shard:: args.num_shards]      # deterministic, disjoint
    if args.limit:
        images = images[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"masks_shard_{args.shard:05d}.jsonl"

    done = load_done_ids(out_path)
    todo = [im for im in images if im["id"] not in done]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_id = args.model_id or DEFAULT_MODELS[args.backend]

    print(f"shard {args.shard}/{args.num_shards} | backend={args.backend} "
          f"model={model_id} device={device}", flush=True)
    print(f"{len(images)} images assigned, {len(done)} already done, "
          f"{len(todo)} to process", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    model, processor = load_backend(args.backend, model_id, device, dtype)

    img_root = Path(args.images)
    t0 = time.time()
    n_img = n_box = n_empty = n_skip = n_fail = 0

    with out_path.open("a") as fout:
        for k, im in enumerate(todo):
            anns = by_image[im["id"]]
            path = img_root / im["file_name"]
            try:
                image = Image.open(path).convert("RGB")
            except Exception as e:
                n_fail += 1
                print(f"  !! unreadable {path}: {e}", flush=True)
                continue

            W, H = image.size
            if (W, H) != (im["width"], im["height"]):
                # Trust the file, not the json — boxes are in json coordinates.
                sx, sy = W / im["width"], H / im["height"]
            else:
                sx = sy = 1.0

            keep, boxes = [], []
            for ann in anns:
                b = xywh_to_xyxy_clipped(
                    [ann["bbox"][0] * sx, ann["bbox"][1] * sy,
                     ann["bbox"][2] * sx, ann["bbox"][3] * sy], W, H)
                if b is None:
                    n_skip += 1
                    continue
                keep.append(ann)
                boxes.append(b)

            record = {"image_id": im["id"], "file_name": im["file_name"],
                      "height": H, "width": W, "anns": []}
            if boxes:
                try:
                    res = segment_image(model, processor, image, boxes,
                                        device, dtype, args.multimask)
                except Exception as e:
                    n_fail += 1
                    print(f"  !! failed {im['file_name']}: {e}", flush=True)
                    continue
                for ann, (rle, score) in zip(keep, res):
                    if rle is None:
                        n_empty += 1
                    record["anns"].append(
                        {"id": ann["id"], "rle": rle, "score": round(score, 4)})
                    n_box += 1

            fout.write(json.dumps(record) + "\n")
            n_img += 1

            if n_img % args.log_every == 0:
                fout.flush()
                rate = n_img / (time.time() - t0)
                eta = (len(todo) - n_img) / max(rate, 1e-9) / 3600
                print(f"  {n_img}/{len(todo)} imgs | {n_box} masks | "
                      f"{rate:.2f} img/s | ETA {eta:.2f} h", flush=True)

    dt = time.time() - t0
    print(f"DONE shard {args.shard}: {n_img} images, {n_box} masks in {dt/60:.1f} min "
          f"({n_img/max(dt,1e-9):.2f} img/s)", flush=True)
    print(f"  empty masks: {n_empty} | degenerate boxes skipped: {n_skip} | "
          f"failed images: {n_fail}", flush=True)
    print(f"  -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Union mode (`.npy` packs)
# ---------------------------------------------------------------------------

def run_union(args) -> None:
    """Segment the union manifest built by `datagen/pack_mask_manifest.py`.

    Same model call as `run()`; the difference is only the join. Records are
    keyed by union index `u` and carry masks in union box order, which
    `pack_mask_manifest.py scatter` then fans back out to per-pack sidecars.
    """
    union_path = Path(args.union) / "union.jsonl"
    records = []
    with union_path.open() as f:
        for line in f:
            records.append(json.loads(line))
    records = records[args.shard:: args.num_shards]    # deterministic, disjoint
    if args.limit:
        records = records[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"masks_shard_{args.shard:05d}.jsonl"

    done = load_done_ids(out_path, key="u")
    todo = [r for r in records if r["u"] not in done]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_id = args.model_id or DEFAULT_MODELS[args.backend]

    print(f"shard {args.shard}/{args.num_shards} | backend={args.backend} "
          f"model={model_id} device={device}", flush=True)
    print(f"{len(records)} images assigned, {len(done)} already done, "
          f"{len(todo)} to process", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    model, processor = load_backend(args.backend, model_id, device, dtype)

    t0 = time.time()
    n_img = n_box = n_empty = n_fail = 0

    with out_path.open("a") as fout:
        for rec in todo:
            boxes = rec["boxes"]
            if not boxes:
                fout.write(json.dumps({"u": rec["u"], "rles": []}) + "\n")
                n_img += 1
                continue
            try:
                image = Image.open(rec["path"]).convert("RGB")
            except Exception as e:
                n_fail += 1
                print(f"  !! unreadable {rec['path']}: {e}", flush=True)
                continue

            W, H = image.size
            mw, mh = rec["wh"]
            if (W, H) != (mw, mh):
                # Trust the file, not the pack: boxes were built in pack coords.
                sx, sy = W / mw, H / mh
                boxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]

            # Packs run to 100 boxes/image against MegaSG's 5.6, so cap the
            # prompt batch. Chunking re-encodes the image, but only 0.7% of
            # images exceed the cap — cheaper than risking an OOM mid-shard.
            try:
                res = []
                for s in range(0, len(boxes), args.box_chunk):
                    res += segment_image(model, processor, image,
                                         boxes[s:s + args.box_chunk],
                                         device, dtype, args.multimask)
            except Exception as e:
                n_fail += 1
                print(f"  !! failed {rec['path']}: {e}", flush=True)
                continue

            rles = [r for r, _ in res]
            n_empty += sum(r is None for r in rles)
            n_box += len(rles)
            fout.write(json.dumps({"u": rec["u"], "rles": rles}) + "\n")
            n_img += 1

            if n_img % args.log_every == 0:
                fout.flush()
                rate = n_img / (time.time() - t0)
                eta = (len(todo) - n_img) / max(rate, 1e-9) / 3600
                print(f"  {n_img}/{len(todo)} imgs | {n_box} masks | "
                      f"{rate:.2f} img/s | ETA {eta:.2f} h", flush=True)

    dt = time.time() - t0
    print(f"DONE shard {args.shard}: {n_img} images, {n_box} masks in {dt/60:.1f} min "
          f"({n_img/max(dt,1e-9):.2f} img/s)", flush=True)
    print(f"  empty masks: {n_empty} | failed images: {n_fail}", flush=True)
    print(f"  -> {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge(args) -> None:
    """Fold every shard's RLEs into the COCO json's `segmentation` field."""
    rles, scores = {}, {}
    shards = sorted(Path(args.out).glob("masks_shard_*.jsonl"))
    if not shards:
        sys.exit(f"no shards found under {args.out}")

    for sp in shards:
        with sp.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for a in rec["anns"]:
                    if a["rle"] is not None:
                        rles[a["id"]] = a["rle"]
                        scores[a["id"]] = a["score"]
    print(f"merged {len(shards)} shards → {len(rles)} masks")

    coco = json.load(open(args.coco))
    n_hit = 0
    for ann in coco["annotations"]:
        rle = rles.get(ann["id"])
        if rle is not None:
            ann["segmentation"] = rle
            ann["sam_score"] = scores[ann["id"]]
            n_hit += 1
    total = len(coco["annotations"])
    print(f"filled {n_hit}/{total} annotations ({100*n_hit/max(total,1):.1f}%)")

    out = args.merged_coco or str(Path(args.out) / "coco_with_masks.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out, "w"))
    print(f"-> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coco", help="MegaSG COCO json (boxes); not needed for --check")
    p.add_argument("--images", help="image root directory")
    p.add_argument("--union", default=None,
                   help="manifest dir from pack_mask_manifest.py (`.npy` packs)")
    p.add_argument("--out", help="shard output directory; not needed for --check")
    p.add_argument("--backend", choices=["sam3", "sam2"], default="sam3")
    p.add_argument("--model_id", default=None, help="override the HF model id")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0, help="cap images (benchmarking)")
    p.add_argument("--multimask", action="store_true",
                   help="request 3 candidates and keep the highest-IoU one")
    p.add_argument("--box_chunk", type=int, default=64,
                   help="max boxes per forward (--union mode)")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--merge", action="store_true", help="merge shards, do not segment")
    p.add_argument("--merged_coco", default=None)
    p.add_argument("--check", action="store_true",
                   help="preflight: load the model, segment one synthetic box, exit")
    args = p.parse_args()

    if args.check:
        check(args)
        return
    if args.union:
        if not args.out:
            p.error("--out is required with --union")
        run_union(args)
        return
    if not args.coco or not args.out:
        p.error("--coco and --out are required unless --check")
    if args.merge:
        merge(args)
    else:
        if not args.images:
            p.error("--images is required unless --merge or --check")
        run(args)


if __name__ == "__main__":
    main()
