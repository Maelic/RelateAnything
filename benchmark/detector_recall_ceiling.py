"""detector_recall_ceiling.py — the hard upper bound the detector imposes on
relation recall, independent of the relation model.

A GT relation can only ever be recalled if BOTH of its endpoints were found by
the detector (the detbox protocol counts a missing endpoint as a permanent
miss). So `pair_recall` here is a CEILING: no relation model, however good,
can exceed it on the detbox protocol.

Box-only, so it runs on CPU in seconds — no images, no model, no GPU.

Usage:
    python training/detector_recall_ceiling.py \
        --dataset_root runs/packed/vg150 \
        --det runs/detect/yolo12m_vg150_val.npz \
        --det_weights.../yolo12m_vg150.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.eval_detboxes import cxcywh_to_xyxy, pairwise_iou, greedy_match  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--det", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--split", default="val",
                   help="pack split to measure against (test splits are the benchmark ones)")
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--det_conf", type=float, nargs="+",
                   default=[0.25, 0.10, 0.05, 0.01, 0.001])
    p.add_argument("--max_objects", type=int, nargs="+", default=[100])
    p.add_argument("--out", default="",
                   help="also write the rows as JSON (so consumers read the ceiling "
                        "instead of hand-copying it out of a log)")
    args = p.parse_args()
    rows = []

    name = args.name or os.path.basename(os.path.normpath(args.dataset_root))
    root = args.dataset_root
    split = args.split
    img_meta = np.load(os.path.join(root, split, "img_meta.npy"))
    boxes = np.load(os.path.join(root, split, "boxes.npy"), mmap_mode="r")
    rels = np.load(os.path.join(root, split, "rels.npy"), mmap_mode="r")
    n_img = len(json.load(open(os.path.join(root, split, "file_names.json"))))

    d = np.load(args.det)
    print(f"=== {name}: {n_img} images, {len(d['conf'])} raw detections ===")
    print(f"{'conf':>7} {'maxN':>5} {'det/img':>8} {'obj_recall':>11} "
          f"{'pair_recall':>12}  <- ceiling on relation recall")

    for conf_thr in args.det_conf:
        keep = d["conf"] >= conf_thr
        d_idx = d["img_idx"][keep]
        d_xyxy = d["xyxy"][keep].astype(np.float32)
        d_conf = d["conf"][keep]
        order = np.argsort(d_idx, kind="stable")
        d_idx, d_xyxy, d_conf = d_idx[order], d_xyxy[order], d_conf[order]
        starts = np.searchsorted(d_idx, np.arange(n_img + 1))

        for max_n in args.max_objects:
            n_gt = n_hit = 0
            n_rel = n_rel_ok = 0
            n_det_total = 0
            for i in range(n_img):
                s0, s1 = starts[i], starts[i + 1]
                c = d_conf[s0:s1]
                keep_n = min(max_n, len(c))
                top = np.argsort(-c)[:keep_n]
                det_xyxy_px = d_xyxy[s0:s1][top]
                n_det_total += len(top)

                _, W, H, b0, nb, r0, nr = img_meta[i]
                nb = min(int(nb), 400)
                if nb == 0:
                    continue
                gt_xyxy = cxcywh_to_xyxy(
                    np.array(boxes[b0:b0 + nb], dtype=np.float32))
                if len(top):
                    det_xyxy = det_xyxy_px / np.array([W, H, W, H], np.float32)
                    m = greedy_match(pairwise_iou(gt_xyxy, det_xyxy), args.iou_thr)
                else:
                    m = np.full(nb, -1, dtype=np.int64)

                n_gt += nb
                n_hit += int((m >= 0).sum())

                r = np.array(rels[r0:r0 + nr], dtype=np.int64)
                if r.size:
                    r = r[(r[:, 0] < nb) & (r[:, 1] < nb)]
                if r.size:
                    ok = (m[r[:, 0]] >= 0) & (m[r[:, 1]] >= 0)
                    n_rel += len(r)
                    n_rel_ok += int(ok.sum())

            obj_r = n_hit / max(n_gt, 1)
            pair_r = n_rel_ok / max(n_rel, 1)
            rows.append({"conf": conf_thr, "max_objects": max_n,
                         "det_per_img": n_det_total / n_img,
                         "obj_recall": obj_r, "pair_recall": pair_r,
                         "n_gt_rel": n_rel})
            print(f"{conf_thr:>7.3f} {max_n:>5} {n_det_total / n_img:>8.1f} "
                  f"{obj_r:>11.3f} {pair_r:>12.3f}")

    print()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump({"name": name, "dataset_root": root, "split": split,
                   "det": args.det, "iou_thr": args.iou_thr, "rows": rows},
                  open(args.out, "w"), indent=2, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
