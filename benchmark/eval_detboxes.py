"""Deployment-gap eval: score a checkpoint with DETECTOR boxes instead of GT.

Protocols (all reuse the standard SGCls evaluate() so numbers are directly
comparable with training summaries):
  gt          oracle GT boxes (sanity baseline — should reproduce known numbers)
  det-noise   GT boxes whose coords are replaced by the IoU>=0.5-matched
              detector box; unmatched boxes keep GT coords.  Isolates the
              effect of box-coordinate noise alone.
  det-strict  same substitution, but GT relations whose endpoints are NOT
              matched by any detection are dropped — recall on the sub-graph
              the detector actually covers (reported with coverage stats).
  jitter      synthetic gaussian box jitter (sigma relative to box size),
              a detector-free reference point.

Usage:
  python benchmark/eval_detboxes.py \
      --checkpoint runs/train/full_v2_swaphinge/checkpoint_best.pth \
      --data_root runs/packed/megasg_50k \
      --det runs/detect/yoloe11l_val5k.npz --det_name yoloe11l
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                    # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt       # noqa: E402
from relsgg.eval.evaluator import (SGClsEvaluator, SoftSGClsEvaluator,  # noqa: E402
                              build_match_matrix)
from relsgg.training.losses import PredicateOntology               # noqa: E402
from relsgg.training.engine import evaluate                        # noqa: E402


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def xyxy_to_cxcywh(b):
    x0, y0, x1, y1 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], -1)


def pairwise_iou(a, b):
    """a [G,4] xyxy, b [D,4] xyxy -> [G,D]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    tl = np.maximum(a[:, None,:2], b[None,:,:2])
    br = np.minimum(a[:, None, 2:], b[None,:, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None,:]
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def greedy_match(iou, thr):
    """Class-agnostic one-to-one greedy matching. Returns det idx per GT (-1)."""
    iou = iou.copy()
    match = np.full(iou.shape[0], -1, np.int64)
    while iou.size:
        g, d = np.unravel_index(np.argmax(iou), iou.shape)
        if iou[g, d] < thr:
            break
        match[g] = d
        iou[g,:] = -1
        iou[:, d] = -1
    return match


class BoxSubDataset(Dataset):
    """Wraps RelationDataset, substituting boxes (and optionally dropping
    relations with unmatched endpoints)."""

    def __init__(self, base, new_boxes=None, matched=None, strict=False,
                 jitter=0.0, seed=0):
        self.base, self.new_boxes, self.matched = base, new_boxes, matched
        self.strict, self.jitter = strict, jitter
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        image, boxes, target = self.base[i]
        nb = boxes.shape[0]
        if self.new_boxes is not None and self.new_boxes.get(i) is not None:
            boxes = torch.from_numpy(self.new_boxes[i][:nb].copy()).float()
        if self.jitter > 0:
            b = boxes.numpy().copy()
            eps = self.rng.standard_normal(b.shape).astype(np.float32)
            b[:, 0] += self.jitter * b[:, 2] * eps[:, 0]
            b[:, 1] += self.jitter * b[:, 3] * eps[:, 1]
            b[:, 2:] *= np.exp(self.jitter * eps[:, 2:])
            boxes = torch.from_numpy(np.clip(b, 1e-4, 1.0)).float()
        if self.strict and self.matched is not None:
            m = self.matched.get(i)
            rels = target["relations"]
            if m is not None and rels.numel():
                ok = torch.from_numpy(m)[rels[:, 0]] & torch.from_numpy(m)[rels[:, 1]]
                target = dict(target)
                target["relations"] = rels[ok]
                target["rel_flags"] = target["rel_flags"][ok]
                target["rel_weights"] = target["rel_weights"][ok]
        return image, boxes, target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="runs/packed/megasg_50k")
    p.add_argument("--canon_groups",
                   default="runs/packed/megasg/text_space/canonical_groups.json")
    p.add_argument("--pred_embeds",
                   default="runs/packed/megasg/text_space/pred_embeds_dinotxt_photo.npz")
    p.add_argument("--tau_eval", type=float, default=0.9311)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--det", required=True, help="detections.npz from detect_boxes.py")
    p.add_argument("--det_name", default="det")
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--det_conf", type=float, default=0.10,
                   help="min detector confidence used for matching")
    p.add_argument("--protocols", default="gt,det-noise,det-strict,jitter05")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=400)
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    pred_names = ckpt["pred_names"]

    ont = PredicateOntology.from_artifacts(
        os.path.join(args.data_root, "train", "meta.json"),
        args.canon_groups, args.pred_embeds, tau_ignore=args.tau_eval)
    emb = np.load(args.pred_embeds)["embeddings"]
    match_mat = build_match_matrix(ont.group_of, emb, args.tau_eval,
                                   ont.inverse_mask)

    train_ds = RelationDataset(root=args.data_root, split="train",
                               resolution=args.img_size)
    assert train_ds.predicate_names == pred_names, (
        "checkpoint predicate order != train pack order — wrong data_root?")
    base = RelationDataset(root=args.data_root, split="val",
                           resolution=args.img_size,
                           cat_to_idx=train_ds.cat_to_idx,
                           rel_cat_to_idx=train_ds.rel_cat_to_idx)

    # ---- Precompute detector-box substitutions -------------------------
    img_meta = np.load(os.path.join(args.data_root, "val", "img_meta.npy"))
    d = np.load(args.det)
    keep = d["conf"] >= args.det_conf
    d_idx, d_xyxy = d["img_idx"][keep], d["xyxy"][keep]
    order = np.argsort(d_idx, kind="stable")
    d_idx, d_xyxy = d_idx[order], d_xyxy[order]
    starts = np.searchsorted(d_idx, np.arange(len(base) + 1))

    new_boxes, matched_masks = {}, {}
    n_gt = n_match = n_rel = n_rel_cov = 0
    iou_sum = 0.0
    for i in range(len(base)):
        _, W, H, b0, nb, r0, nr = img_meta[i]
        nb = min(int(nb), base.max_objects)
        gt_cxcywh = np.array(base.boxes[b0:b0 + nb], dtype=np.float32)
        gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
        det = d_xyxy[starts[i]:starts[i + 1]].astype(np.float32)
        det /= np.array([W, H, W, H], np.float32)
        iou = pairwise_iou(gt_xyxy, det)
        m = greedy_match(iou, args.iou_thr)
        sub = gt_cxcywh.copy()
        hit = m >= 0
        if hit.any():
            sub[hit] = xyxy_to_cxcywh(det[m[hit]])
            iou_sum += iou[np.arange(nb)[hit], m[hit]].sum()
        new_boxes[i] = sub
        matched_masks[i] = hit
        n_gt += nb
        n_match += int(hit.sum())
        rels = np.array(base.rels[r0:r0 + nr], dtype=np.int64)
        rels = rels[(rels[:, 0] < nb) & (rels[:, 1] < nb)] if rels.size else rels
        if rels.size:
            n_rel += len(rels)
            n_rel_cov += int((hit[rels[:, 0]] & hit[rels[:, 1]]).sum())

    stats = {
        "det_name": args.det_name,
        "box_match_rate": float(n_match) / max(n_gt, 1),
        "mean_matched_iou": float(iou_sum) / max(n_match, 1),
        "rel_coverage": float(n_rel_cov) / max(n_rel, 1),
        "dets_per_img": float(len(d_idx)) / len(base),
    }
    print(f"[match] {json.dumps(stats, indent=2)}")

    # ---- Evaluate each protocol ---------------------------------------
    eval_args = SimpleNamespace(amp=device.type == "cuda",
                                amp_dtype_t=torch.bfloat16)
    score_mode = "sigmoid"
    results = {"match_stats": stats, "checkpoint": args.checkpoint}
    for proto in args.protocols.split(","):
        proto = proto.strip()
        if proto == "gt":
            ds = base
        elif proto == "det-noise":
            ds = BoxSubDataset(base, new_boxes, matched_masks)
        elif proto == "det-strict":
            ds = BoxSubDataset(base, new_boxes, matched_masks, strict=True)
        elif proto.startswith("jitter"):
            sigma = float(proto[len("jitter"):]) / 100.0
            ds = BoxSubDataset(base, jitter=sigma)
        else:
            raise ValueError(proto)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers,
                            pin_memory=True)
        evs = [
            SGClsEvaluator(topk=[20, 50, 100], num_predicates=len(pred_names),
                           score_mode=score_mode),
            SoftSGClsEvaluator(match_mat, ont.group_of, topk=[20, 50, 100],
                               score_mode=score_mode),
        ]
        with torch.no_grad():
            metrics = evaluate(model, loader, device, eval_args, evs,
                               eval_budget=args.eval_budget)
        results[proto] = metrics
        keys = ["R@50", "SoftR@50", "SoftmR@50", "SoftR@100", "GT_MRR"]
        line = "  ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
        print(f"[{proto:12s}] {line}", flush=True)

    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint), f"detbox_eval_{args.det_name}.json")
    json.dump(results, open(out_path, "w"), indent=2, default=float)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
