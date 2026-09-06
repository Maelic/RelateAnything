"""Dump per-pair logits + geometry for offline re-ranking experiments (E0/E1).

Saves, for EVERY sampled pair of every eval image, the full [V] predicate
logit row (fp16), the pair-existence logit, and the box pair — so any score
adjustment of the form z' = z_pred[v] + z_pair + Delta(v, geometry) can be
replayed offline on CPU, EXACTLY (no top-K approximation: with the full row,
argmax shifts under a per-predicate Delta are computed, not guessed).

This is the E1 enabler: the generative geometry prior is evaluated as a pure
re-ranking of this dump at matched emission volume, before any training run
is spent on it. GT triples and box categories ride along so per-predicate /
per-geometry-bucket recall is computable without touching the packs again.

Layout (npz, grouped by image; pair row i belongs to image searchsorted-style
via pair_counts):
    pred_names     [V]  str
    img_names      [I]  str
    pair_counts    [I]  int32   pairs per image (cumsum -> offsets)
    box_counts     [I]  int32   boxes per image
    boxes          [sum_N, 4]   fp16, normalized cxcywh (loader-truncated)
    box_cats       [sum_N]      int16
    sub, obj       [P]  int16   box indices WITHIN the image
    z_pair         [P]  fp16
    z_pred         [P, V] fp16  full fused predicate logits
    gt_counts      [I]  int32   GT triples per image (after loader truncation)
    gt             [sum_R, 3]   int16 (sub, obj, pred)

Size: VG150 test ~3.5M pairs x V=50 -> ~400 MB. Disk is at 99%, so fp16
everywhere and no duplication of per-image data into pair rows.

    python training/dump_pair_scores.py --checkpoint CK \
        --data_root runs/packed/vg150 --split test --out runs/analysis/e0/dump.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data.dataset import RelationDataset, collate_fn  # noqa: E402
from relsgg.model.geometry import RelGeomEncoder  # noqa: E402
from relsgg.text.student import encode_texts_student  # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="runs/packed/vg150")
    p.add_argument("--split", default="test")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--max_objects", type=int, default=32)
    p.add_argument("--geo_budget", type=int, default=992)
    p.add_argument("--final_budget", type=int, default=992)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ck, "ema").to(dev).eval()
    model.sampler.geo_budget = a.geo_budget
    model.sampler.final_budget = a.final_budget

    ds = RelationDataset(root=a.data_root, split=a.split, resolution=a.img_size,
                         max_objects=a.max_objects)
    names = list(ds.predicate_names)
    ck_args = ck.get("args") or {}
    ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    E = encode_texts_student(names, ck_args["text_student"],
                             templates=TRAIN_TEMPLATES, device=dev)
    model.vocab_head.set_vocabulary_matrix(names, E)
    model.reparameterize()

    sub_ds = ds
    if a.limit:
        sub_ds = torch.utils.data.Subset(ds, list(range(min(a.limit, len(ds)))))
    loader = DataLoader(sub_ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)

    pair_counts, box_counts, gt_counts = [], [], []
    all_boxes, all_cats, all_sub, all_obj = [], [], [], []
    all_zpair, all_zpred, all_gt = [], [], []
    n_img = 0
    for images, boxes, counts, targets in loader:
        images = images.to(dev, non_blocking=True)
        boxes_d = boxes.to(dev, non_blocking=True)
        counts_d = counts.to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            out = model(images, boxes_d, counts_d, targets=None)
        zp = out["logits"].float()
        za = (out["pair_logits"].float() if out.get("pair_logits") is not None
              else torch.zeros_like(zp[..., 0]))
        for b in range(zp.shape[0]):
            n_img += 1
            n = int(counts[b])
            box_counts.append(n)
            all_boxes.append(boxes[b][:n].numpy().astype(np.float16))
            all_cats.append(
                targets[b]["entity_labels"].numpy()[:n].astype(np.int16))
            mask = out["valid_mask"][b]
            k = int(mask.sum())
            pair_counts.append(k)
            all_sub.append(out["sub_idx"][b][mask].cpu().numpy().astype(np.int16))
            all_obj.append(out["obj_idx"][b][mask].cpu().numpy().astype(np.int16))
            all_zpair.append(za[b][mask].cpu().numpy().astype(np.float16))
            all_zpred.append(zp[b][mask].cpu().numpy().astype(np.float16))
            rels = targets[b].get("relations")
            if rels is None or not len(rels):
                gt_counts.append(0)
            else:
                r = rels.numpy().astype(np.int16)
                gt_counts.append(len(r))
                all_gt.append(r)
        if n_img % 3200 < a.batch_size:
            print(f"  {n_img} imgs, {sum(pair_counts):,} pairs", flush=True)

    fn = (ds.file_names if not a.limit
          else [ds.file_names[i] for i in range(len(sub_ds))])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(
        a.out,
        pred_names=np.array(names),
        img_names=np.array(fn),
        pair_counts=np.array(pair_counts, np.int32),
        box_counts=np.array(box_counts, np.int32),
        gt_counts=np.array(gt_counts, np.int32),
        boxes=np.concatenate(all_boxes) if all_boxes else np.zeros((0, 4), np.float16),
        box_cats=np.concatenate(all_cats) if all_cats else np.zeros(0, np.int16),
        sub=np.concatenate(all_sub), obj=np.concatenate(all_obj),
        z_pair=np.concatenate(all_zpair),
        z_pred=np.concatenate(all_zpred),
        gt=np.concatenate(all_gt) if all_gt else np.zeros((0, 3), np.int16),
        # Provenance so downstream readouts can tell a clean single-factor
        # ablation from a bundle without the reader having to remember which
        # flags each checkpoint carried (e0_report.py derives its caveat here).
        ckpt_path=np.array(a.checkpoint),
        ckpt_args=np.array(json.dumps(ck.get("args", {}), default=str)),
)
    mb = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}: {n_img} imgs, {sum(pair_counts):,} pairs, "
          f"{sum(gt_counts):,} GT triples, {mb:.0f} MB")


if __name__ == "__main__":
    main()
