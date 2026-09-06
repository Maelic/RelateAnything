"""Run a prompt-baked open-vocab detector over a packed split's images.

Saves flat arrays (img_idx, xyxy pixels, conf, cls) to an.npz, aligned with
the pack's file_names.json order, for downstream IoU-matching against GT
boxes (deployment-gap eval / detector-box training).

Usage:
  python training/detect_boxes.py \
      --weights checkpoints/detectors/yoloe-11l-megasg497.pt \
      --pack runs/packed/megasg_50k/val \
      --out runs/detect/yoloe11l_val5k.npz
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pack", required=True, help="packed split dir (needs meta.json + file_names.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--max_det", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N images")
    ap.add_argument("--save_masks", default="",
                    help="segmentation weights only: also write per-image "
                         "instance-mask RLEs to this.jsonl, aligned with the "
                         "npz detection order, for mask AP. Forces "
                         "retina_masks so masks come back at ORIGINAL image "
                         "resolution (letterbox-padded masks would be "
                         "silently misaligned against GT).")
    ap.add_argument("--agnostic_nms", action="store_true",
                    help="class-agnostic NMS. Matters for a prompt-free detector, "
                         "whose 4,585-tag head fires several near-identical tags on "
                         "one region ('man'/'person'/'boy'); class-aware NMS keeps "
                         "them all and the relation head then pairs a region with "
                         "itself. Costs nested objects (a person and their shirt).")
    ap.add_argument("--set_classes", action="store_true",
                    help="prompt the detector with the PACK's categories at test time "
                         "(open-vocab, closed to the benchmark's list). Without it the "
                         "weights' baked-in vocabulary is used as-is -- which is what a "
                         "prompt-free ('-pf') checkpoint is for; `cls` then indexes the "
                         "detector's own names, saved into the npz as `names`.")
    args = ap.parse_args()

    pack = Path(args.pack)
    meta = json.load(open(pack / "meta.json"))
    file_names = json.load(open(pack / "file_names.json"))
    img_dir = Path(meta["img_dir"])
    paths = [str(img_dir / f) for f in file_names]
    if args.limit:
        paths = paths[: args.limit]
    print(f"{len(paths)} images from {img_dir}")

    # FastSAM is CLASS-AGNOSTIC ("everything" mode: one class, `object`) and
    # needs its own predictor for that postprocessing; the generic YOLO loader
    # dispatches on the checkpoint but not to this. It is the right front end
    # when the point is that NO taxonomy is imposed on the regions at all.
    if "fastsam" in os.path.basename(args.weights).lower():
        from ultralytics import FastSAM as _Loader
    else:
        from ultralytics import YOLO as _Loader

    model = _Loader(args.weights)
    print(f"loaded {args.weights} | task={model.task} | {len(model.names)} classes")

    if args.set_classes:
        # Open-world SGDet: the detector is PROMPTED with the benchmark's own
        # vocabulary at test time, so `cls` indexes meta["categories"] directly and no
        # cross-vocabulary remap is needed downstream. YOLOE needs its text prompt
        # embeddings passed explicitly; YOLO-World derives them internally.
        names = list(meta["categories"])
        try:
            model.set_classes(names, model.get_text_pe(names))   # YOLOE
        except (AttributeError, TypeError):
            model.set_classes(names)                             # YOLO-World
        print(f"  prompted with {len(names)} pack categories (open-vocab)")
        assert len(model.names) == len(meta["categories"]), \
            f"detector classes {len(model.names)} != pack categories {len(meta['categories'])}"

    names = [str(model.names[i]) for i in range(len(model.names))]
    if not args.set_classes:
        # A TEXT-PROMPT YOLOE checkpoint ships with no usable vocabulary: its
        # `names` are the placeholders "0".."79" and it detects nothing useful
        # until set_classes is called. The "-pf" prompt-free checkpoint is the
        # one carrying a real baked-in vocabulary (4,585 RAM tags). Refuse the
        # former rather than emit boxes labelled with integers.
        if all(n.isdigit() for n in names):
            raise SystemExit(
                f"!! {args.weights} has placeholder class names "
                f"('0'..'{len(names) - 1}') -- it is a text-prompt checkpoint with "
                f"an empty vocabulary. Pass --set_classes, or use a prompt-free "
                f"('-pf') checkpoint.")
        print(f"  prompt-free: the detector's own baked-in {len(names)}-class "
              f"vocabulary (pack categories are NOT imposed)")

    import torch

    kw = dict(conf=args.conf, iou=args.iou, max_det=args.max_det,
              imgsz=args.imgsz, verbose=False, save=False,
              agnostic_nms=args.agnostic_nms)
    if args.save_masks:
        if model.task != "segment":
            raise SystemExit(f"!! --save_masks needs segmentation weights, "
                             f"but {args.weights} has task={model.task}")
        kw["retina_masks"] = True
        from pycocotools import mask as mask_util
    img_idx, xyxy, conf, cls = [], [], [], []
    masks_by_idx = {}
    skipped = []
    seen = set()

    def consume(i, r):
        seen.add(i)
        b = r.boxes
        n = len(b)
        if n:
            img_idx.append(np.full(n, i, dtype=np.int32))
            xyxy.append(b.xyxy.cpu().numpy().astype(np.float32))
            conf.append(b.conf.cpu().numpy().astype(np.float32))
            cls.append(b.cls.cpu().numpy().astype(np.int32))
        if args.save_masks:
            rles = []
            m = getattr(r, "masks", None)
            for k in range(n):
                if m is None or k >= len(m.data):
                    rles.append(None)       # box without a mask: unscorable
                    continue
                arr = np.asfortranarray(
                    m.data[k].cpu().numpy().astype(np.uint8))
                rle = mask_util.encode(arr)
                rle["counts"] = rle["counts"].decode()
                rles.append(rle)
            masks_by_idx[i] = rles

    # Chunked inference with per-image OOM fallback: a single pathological
    # image can trigger a huge allocation (seen: 122 GiB conv on YOLOE-seg);
    # isolate + skip it rather than losing the whole job.
    chunk = 256
    for c0 in range(0, len(paths), chunk):
        sub = paths[c0:c0 + chunk]
        try:
            for j, r in enumerate(model.predict(sub, batch=args.batch,
                                                stream=True, **kw)):
                consume(c0 + j, r)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  OOM in chunk @{c0}; retrying per-image", flush=True)
            for j, p in enumerate(sub):
                if c0 + j in seen:
                    continue
                try:
                    r = model.predict(p, batch=1, stream=False, **kw)[0]
                    consume(c0 + j, r)
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    skipped.append(c0 + j)
                    print(f"  SKIP OOM image {c0 + j}: {p}", flush=True)
        done = min(c0 + chunk, len(paths))
        if done % 512 == 0 or done == len(paths):
            print(f"  {done}/{len(paths)}", flush=True)
    if skipped:
        print(f"WARNING: skipped {len(skipped)} OOM images: {skipped[:20]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        img_idx=np.concatenate(img_idx) if img_idx else np.zeros(0, np.int32),
        xyxy=np.concatenate(xyxy) if xyxy else np.zeros((0, 4), np.float32),
        conf=np.concatenate(conf) if conf else np.zeros(0, np.float32),
        cls=np.concatenate(cls) if cls else np.zeros(0, np.int32),
        n_images=np.int64(len(paths)),
        # The vocabulary `cls` indexes, saved WITH the detections: pack categories
        # under --set_classes, the detector's own names otherwise. Consumers must
        # read it from here -- re-deriving it from the.pt gives the PLACEHOLDER
        # names for a prompted run and silently mislabels every box.
        names=np.array(names),
)
    tot = sum(len(a) for a in img_idx)
    print(f"saved {out}: {tot} boxes over {len(paths)} images "
          f"({tot / max(len(paths), 1):.1f}/img)")

    if args.save_masks:
        mp = Path(args.save_masks)
        mp.parent.mkdir(parents=True, exist_ok=True)
        n_m = n_none = 0
        with mp.open("w") as f:
            for i in sorted(masks_by_idx):
                rles = masks_by_idx[i]
                n_m += sum(1 for r in rles if r is not None)
                n_none += sum(1 for r in rles if r is None)
                f.write(json.dumps({"idx": i, "rles": rles}) + "\n")
        print(f"saved {mp}: {n_m} masks ({n_none} boxes without one)")


if __name__ == "__main__":
    main()
