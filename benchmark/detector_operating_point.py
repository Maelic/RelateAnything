"""detector_operating_point.py — pick a detector threshold by what it does to the
RELATIONS, not only to the boxes.

Box F1 alone is the wrong objective for an SGG front end. Raising the confidence
floor buys box precision and spends pair recall, and pair recall is the hard ceiling
on detection-mode relation recall. So this sweeps the operating point and reports
both sides on one grid, plus the quantity the A5 error anatomy showed actually
matters: how many SAME-CLASS box pairs the detector hands the relation head.

Columns per (conf, dedup) cell:
  box/img                boxes surviving the floor, the cap and the dedup
  boxR / boxR+cls        GT boxes recovered at IoU>=thr (localisation / +class)
  boxP                   detections matching some GT box (localisation)
  boxF1                  harmonic mean of boxR and boxP
  pairRec                fraction of GT relations with BOTH endpoints recovered
                         -- the ceiling on relation recall for ANY head
  sameCls/img            ordered same-class box pairs offered to the head
  dupPairs/img           of those, pairs overlapping at IoU>=0.3, i.e. the
                         fragmented-object pairs behind `sky in sky`

`--dedup` applies a post-hoc per-class greedy NMS to the CACHED detections, so the
duplicate-suppression question can be answered without re-running the detector.

Assumes `cls` indexes the pack's own categories (detect_boxes.py --set_classes).

    python benchmark/detector_operating_point.py \
        --pack runs/packed/psg/test --det runs/detect/yoloworld_ov_psg_test.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def to_xyxy(b: np.ndarray, W: float, H: float) -> np.ndarray:
    """Pack boxes are normalised cxcywh; detections are absolute xyxy."""
    if not len(b):
        return b.reshape(0, 4)
    cx, cy, w, h = b[:, 0] * W, b[:, 1] * H, b[:, 2] * W, b[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None,:, 0])
    y1 = np.maximum(a[:, None, 1], b[None,:, 1])
    x2 = np.minimum(a[:, None, 2], b[None,:, 2])
    y2 = np.minimum(a[:, None, 3], b[None,:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return (inter / np.clip(aa[:, None] + bb[None,:] - inter, 1e-9, None)).astype(np.float32)


def dedup_same_class(box: np.ndarray, conf: np.ndarray, cls: np.ndarray,
                     thr: float) -> np.ndarray:
    """Greedy per-class NMS on already-sorted-by-confidence detections."""
    keep = np.ones(len(box), bool)
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        if len(idx) < 2:
            continue
        M = iou_mat(box[idx], box[idx])
        for i in range(len(idx)):
            if not keep[idx[i]]:
                continue
            drop = idx[i + 1:][(M[i, i + 1:] >= thr) & keep[idx[i + 1:]]]
            keep[drop] = False
    return keep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pack", required=True)
    p.add_argument("--det", required=True)
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--conf", type=float, nargs="+",
                   default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
    p.add_argument("--dedup", type=float, nargs="+", default=[1.0],
                   help="per-class NMS IoU; 1.0 = off")
    p.add_argument("--dup_iou", type=float, default=0.3,
                   help="same-class pairs above this IoU are counted as fragments")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--name", default="")
    p.add_argument("--out", default="")
    a = p.parse_args()

    meta = json.load(open(os.path.join(a.pack, "meta.json")))
    img_meta = np.load(os.path.join(a.pack, "img_meta.npy"))
    pboxes = np.load(os.path.join(a.pack, "boxes.npy"), mmap_mode="r")
    pcats = np.load(os.path.join(a.pack, "box_cats.npy"), mmap_mode="r")
    prels = np.load(os.path.join(a.pack, "rels.npy"), mmap_mode="r")
    n_img = len(img_meta) if a.limit <= 0 else min(a.limit, len(img_meta))

    d = np.load(a.det)
    di = d["img_idx"]
    order = np.argsort(di, kind="stable")
    di = di[order]
    dx = d["xyxy"].astype(np.float32)[order]
    dc = d["conf"].astype(np.float32)[order]
    dl = d["cls"].astype(np.int64)[order]
    starts = np.searchsorted(di, np.arange(len(img_meta) + 1))

    rows = []
    for thr in a.conf:
        for dd in a.dedup:
            n_gt = hit = hit_cls = 0
            n_det = det_tp = 0
            n_rel = rel_cov = 0
            same_pairs = dup_pairs = 0
            for i in range(n_img):
                _, W, H, b0, nb, r0, nr = (int(x) for x in img_meta[i])
                g = to_xyxy(np.asarray(pboxes[b0:b0 + nb], np.float32), W, H)
                gc = np.asarray(pcats[b0:b0 + nb], np.int64)

                s, e = starts[i], starts[i + 1]
                bx, cf, cl = dx[s:e], dc[s:e], dl[s:e]
                k = cf >= thr
                bx, cf, cl = bx[k], cf[k], cl[k]
                o = np.argsort(-cf, kind="stable")[:a.max_objects]
                bx, cf, cl = bx[o], cf[o], cl[o]
                if dd < 1.0 and len(bx) > 1:
                    kp = dedup_same_class(bx, cf, cl, dd)
                    bx, cf, cl = bx[kp], cf[kp], cl[kp]

                M = iou_mat(g, bx)
                ok = M >= a.iou_thr
                n_gt += len(g)
                n_det += len(bx)
                if len(g) and len(bx):
                    hit += int(ok.any(1).sum())
                    same = gc[:, None] == cl[None,:]
                    covered = (ok & same).any(1)
                    hit_cls += int(covered.sum())
                    det_tp += int(ok.any(0).sum())
                    # same-class pairs offered to the head
                    S = cl[:, None] == cl[None,:]
                    np.fill_diagonal(S, False)
                    same_pairs += int(S.sum())
                    D = iou_mat(bx, bx) >= a.dup_iou
                    np.fill_diagonal(D, False)
                    dup_pairs += int((S & D).sum())
                    cov_idx = covered
                else:
                    cov_idx = np.zeros(len(g), bool)

                if nr:
                    rl = np.asarray(prels[r0:r0 + nr], np.int64)
                    valid = (rl[:, 0] < nb) & (rl[:, 1] < nb)
                    rl = rl[valid]
                    n_rel += len(rl)
                    if len(rl) and len(cov_idx):
                        rel_cov += int((cov_idx[rl[:, 0]] & cov_idx[rl[:, 1]]).sum())

            R = hit / max(n_gt, 1)
            Rc = hit_cls / max(n_gt, 1)
            P = det_tp / max(n_det, 1)
            F1 = 2 * P * R / max(P + R, 1e-9)
            rows.append(dict(conf=thr, dedup=dd, box_per_img=n_det / max(n_img, 1),
                             box_R=R, box_R_cls=Rc, box_P=P, box_F1=F1,
                             pair_recall=rel_cov / max(n_rel, 1),
                             same_cls_pairs_per_img=same_pairs / max(n_img, 1),
                             dup_pairs_per_img=dup_pairs / max(n_img, 1)))

    hdr = (f"{'conf':>5} {'dedup':>6} {'box/img':>8} {'boxR':>6} {'boxR+cls':>9} "
           f"{'boxP':>6} {'boxF1':>6} {'pairRec':>8} {'sameCls/img':>12} {'dupPairs/img':>13}")
    print(f"\n=== {a.name or a.det} — {n_img} images, IoU {a.iou_thr} ===")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        dd = "off" if r["dedup"] >= 1.0 else f"{r['dedup']:.2f}"
        print(f"{r['conf']:>5.2f} {dd:>6} {r['box_per_img']:>8.1f} "
              f"{100*r['box_R']:>5.1f}% {100*r['box_R_cls']:>8.1f}% {100*r['box_P']:>5.1f}% "
              f"{100*r['box_F1']:>5.1f}% {100*r['pair_recall']:>7.1f}% "
              f"{r['same_cls_pairs_per_img']:>12.1f} {r['dup_pairs_per_img']:>13.1f}")

    best_f1 = max(rows, key=lambda x: x["box_F1"])
    print(f"\nF1-optimal: conf {best_f1['conf']:.2f} dedup "
          f"{'off' if best_f1['dedup']>=1 else best_f1['dedup']:} "
          f"-> F1 {100*best_f1['box_F1']:.1f}%, pairRec {100*best_f1['pair_recall']:.1f}%, "
          f"dupPairs/img {best_f1['dup_pairs_per_img']:.1f}")
    base = rows[0]
    print(f"as-run (first row): conf {base['conf']:.2f} -> F1 {100*base['box_F1']:.1f}%, "
          f"pairRec {100*base['pair_recall']:.1f}%, dupPairs/img {base['dup_pairs_per_img']:.1f}")
    print(f"\nceiling cost of the F1-optimal point: "
          f"{100*(base['pair_recall']-best_f1['pair_recall']):+.1f} points of pair recall")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump({"pack": a.pack, "det": a.det, "name": a.name,
                   "iou_thr": a.iou_thr, "max_objects": a.max_objects,
                   "n_images": n_img, "rows": rows}, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
