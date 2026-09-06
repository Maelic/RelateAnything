"""SwapAcc / InvConsistency probe — the direction metric.

For every GT relation (s, o, g) in the val split, the sampler force-includes
both the GT slot (s, o) and the swapped slot (o, s) when targets are passed
(swap_include=True). The probe then asks:

  SwapAcc        P[ score(s,o,g) > score(o,s,g) ] — does the model prefer
                 the annotated direction for the annotated predicate?
                 Reported overall + spatial/semantic split (rel_flags bit0),
                 for the main head and the fast bilinear head (whose only
                 direction mechanism is the P_s/P_o asymmetry — RAM's bet).
  InvTop        for predicates with a spatial inverse (above/below,...):
                 P[ best inverse-predicate score at (o,s) > score of g at
                 (o,s) ] — on the swapped pair, does the model know the
                 relation flips rather than merely weakening?
  InvRank       median rank of the best inverse predicate at (o, s).

Usage:
    python training/swap_probe.py \
        --checkpoint runs/train/conv_50k_v31_pergroup/checkpoint_best.pth \
        --data_root runs/packed/megasg_50k \
        --canon_groups runs/packed/megasg/text_space/canonical_groups.json \
        --pred_embeds runs/packed/megasg/text_space/pred_embeds_dinotxt_photo.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                    # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt       # noqa: E402
from relsgg.training.losses import PredicateOntology               # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="runs/packed/megasg_50k")
    p.add_argument("--canon_groups",
                   default="runs/packed/megasg/text_space/canonical_groups.json")
    p.add_argument("--pred_embeds",
                   default="runs/packed/megasg/text_space/pred_embeds_dinotxt_photo.npz")
    p.add_argument("--tau_ignore", type=float, default=0.9311)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    pred_names = ckpt["pred_names"]
    V = len(pred_names)

    meta_path = os.path.join(args.data_root, "train", "meta.json")
    ont = PredicateOntology.from_artifacts(
        meta_path, args.canon_groups, args.pred_embeds,
        tau_ignore=args.tau_ignore)
    inv_mask = ont.inverse_mask.to(device)          # [V, V] bool
    has_inv = inv_mask.any(-1)                      # [V]

    # Each packed split carries its OWN vocabulary (val ids != train ids) —
    # labels must be remapped into the train/checkpoint id space, exactly as
    # train.py's eval does. Without this every L[..., g] below reads a wrong
    # predicate column.
    train_ds = RelationDataset(root=args.data_root, split="train",
                               resolution=args.img_size)
    assert train_ds.predicate_names == pred_names, (
        "checkpoint predicate order != train pack order — wrong data_root?")
    ds = RelationDataset(root=args.data_root, split="val",
                         resolution=args.img_size, max_objects=100,
                         cat_to_idx=train_ds.cat_to_idx,
                         rel_cat_to_idx=train_ds.rel_cat_to_idx)
    if args.limit:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=True)

    # tallies[head][split] = [n_correct, n_total]
    tallies = {h: defaultdict(lambda: [0, 0]) for h in ("main", "fast")}
    inv_top, inv_ranks = [0, 0], []

    amp = device.type == "cuda"
    with torch.no_grad():
        for images, boxes, box_counts, targets in tqdm(loader, desc="probe"):
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            box_counts = box_counts.to(device, non_blocking=True)
            for t in targets:
                t["relations"] = t["relations"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp,
                                    dtype=torch.bfloat16):
                out = model(images, boxes, box_counts, targets=targets)

            logit_heads = {"main": out["logits"].float()}
            if "logits_fast" in out:
                logit_heads["fast"] = out["logits_fast"].float()
            key = out["sub_idx"] * 1024 + out["obj_idx"]     # [B, K]

            for b, t in enumerate(targets):
                rels = t["relations"]
                if rels.numel() == 0:
                    continue
                flags = t.get("rel_flags")
                valid_b = out["valid_mask"][b]
                # first slot index per pair key
                key_b = key[b].masked_fill(~valid_b, -1)
                slot_of = {}
                for k_i, k_v in enumerate(key_b.tolist()):
                    if k_v >= 0 and k_v not in slot_of:
                        slot_of[k_v] = k_i

                for r_i in range(rels.shape[0]):
                    s, o, g = (int(rels[r_i, 0]), int(rels[r_i, 1]),
                               int(rels[r_i, 2]))
                    fwd = slot_of.get(s * 1024 + o)
                    bwd = slot_of.get(o * 1024 + s)
                    if fwd is None or bwd is None:
                        continue
                    split = ("spatial" if flags is not None
                             and int(flags[r_i]) & 1 else "semantic")
                    for h, L in logit_heads.items():
                        correct = bool(L[b, fwd, g] > L[b, bwd, g])
                        for sp in ("all", split):
                            tallies[h][sp][0] += int(correct)
                            tallies[h][sp][1] += 1
                    if has_inv[g]:
                        row = logit_heads["main"][b, bwd]        # [V]
                        inv_best = row[inv_mask[g]].max()
                        inv_top[0] += int(bool(inv_best > row[g]))
                        inv_top[1] += 1
                        inv_ranks.append(int((row > inv_best).sum()) + 1)

    report = {}
    for h in tallies:
        for sp, (c, n) in tallies[h].items():
            if n:
                report[f"SwapAcc_{h}_{sp}"] = round(c / n, 4)
                report[f"n_{h}_{sp}"] = n
    if inv_top[1]:
        report["InvTop_main"] = round(inv_top[0] / inv_top[1], 4)
        report["InvRank_median_main"] = int(np.median(inv_ranks))
        report["n_inv"] = inv_top[1]

    print("\n" + json.dumps(report, indent=2))
    out_path = os.path.join(os.path.dirname(args.checkpoint),
                            "swap_probe.json")
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"saved → {out_path}")


if __name__ == "__main__":
    main()
