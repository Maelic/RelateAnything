"""Is the relatedness head any good, and does it help or hurt downstream?

The head emits one logit per candidate pair and is trained with focal BCE against
"this pair carries >=1 GT relation" (relsgg/sampler.py). Two things follow that
are easy to conflate:

  * At eval the logit is added to EVERY predicate of its pair, so it cannot
    reorder predicates WITHIN a pair. It is purely a cross-pair term: it decides
    which pairs win the K slots under the graph constraint, and it is the only
    thing making one pair's score comparable to another's.
  * What it actually learns is ANNOTATION PROPENSITY — "would an annotator have
    labelled this pair" — which on a federated corpus is not the same as "these
    two objects are related", and is definitely not "this predicate is true".

This script measures the head on its own terms, per source:
  AP / AUC   ranking labelled-vs-unlabelled pairs among the pairs the sampler
             proposed. The honest ceiling: unlabelled != negative, so a perfect
             model cannot reach 1.0 and the number is a LOWER bound on quality.
  R@budget   fraction of GT-bearing pairs surviving into the top-N by relatedness
             — the recall this head is responsible for, at the budgets we deploy.
  prevalence share of proposed pairs that carry GT, i.e. the AP of chance.

    python research/eval_relatedness.py --checkpoint runs/train/v43_full_5ep/checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                      # noqa: E402
from relsgg.training.engine import evaluate                          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt        # noqa: E402


class RelatednessProbe:
    """Collect (relatedness logit, has-GT) for every proposed pair."""

    def __init__(self):
        self.lg, self.y, self.img = [], [], []
        self.n_gt_pairs = 0
        self.n_gt_found = 0

    @torch.no_grad()
    def update(self, out: dict, targets) -> None:
        pl = out.get("pair_logits")
        if pl is None:
            return
        sub_idx, obj_idx, valid = out["sub_idx"], out["obj_idx"], out["valid_mask"]
        for b in range(pl.shape[0]):
            mask = valid[b]
            gt_pairs = {(int(r[0]), int(r[1]))
                        for r in targets[b].get("relations", [])}
            self.n_gt_pairs += len(gt_pairs)
            if not int(mask.sum()):
                continue
            s_l = sub_idx[b][mask].tolist()
            o_l = obj_idx[b][mask].tolist()
            v = pl[b][mask].float().tolist()
            seen = set()
            for si, oi, lv in zip(s_l, o_l, v):
                if (si, oi) in seen:
                    continue
                seen.add((si, oi))
                self.lg.append(lv)
                self.y.append(1 if (si, oi) in gt_pairs else 0)
                self.img.append(int(targets[b]["index"]))
            self.n_gt_found += len(gt_pairs & seen)

    def compute(self):
        return {"n_pairs": float(len(self.lg))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--packs", nargs="+",
                   default=["runs/packed/psg", "runs/packed/vg150",
                            "runs/packed/indoorvg"])
    p.add_argument("--split", default="test")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=400)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", default="")
    a = p.parse_args()

    from sklearn.metrics import average_precision_score, roc_auc_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {a.checkpoint} (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, "ema").to(device).eval()

    res = {}
    for pack in a.packs:
        ds = RelationDataset(root=pack, split=a.split, resolution=a.img_size,
                             max_objects=40)
        if a.limit and len(ds) > a.limit:
            ds = torch.utils.data.Subset(ds, range(a.limit))
            ds.predicate_names = []  # unused here
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=a.num_workers,
                            pin_memory=True)
        ev = RelatednessProbe()
        eval_args = SimpleNamespace(amp=device.type == "cuda",
                                    amp_dtype_t=torch.bfloat16)
        evaluate(model, loader, device, eval_args, ev, eval_budget=a.eval_budget)
        y = np.asarray(ev.y)
        lg = np.asarray(ev.lg)
        name = os.path.basename(pack)
        if y.sum() == 0 or y.sum() == len(y):
            print(f"{name}: degenerate labels, skipped")
            continue
        prev = float(y.mean())
        row = {"AP": float(average_precision_score(y, lg)),
               "AUC": float(roc_auc_score(y, lg)),
               "prevalence": prev,
               "AP_lift_over_chance": float(average_precision_score(y, lg) / prev),
               "n_pairs": int(len(y)), "n_gt_pairs": int(ev.n_gt_pairs),
               "pair_recall_at_budget": float(ev.n_gt_found / max(ev.n_gt_pairs, 1))}
        # Per-image recall of GT pairs if we kept only the top-N by relatedness.
        by_img = {}
        for i, im in enumerate(ev.img):
            by_img.setdefault(im, []).append(i)
        for N in (20, 50, 100):
            hit = tot = 0
            for im, idxs in by_img.items():
                idxs = sorted(idxs, key=lambda i: -lg[i])[:N]
                hit += int(y[idxs].sum())
                tot += int(y[[i for i in by_img[im]]].sum())
            row[f"gt_pair_recall@{N}"] = hit / max(tot, 1)
        res[name] = row
        print(f"{name:12s} AP {row['AP']:.4f} (chance {prev:.4f}, "
              f"x{row['AP_lift_over_chance']:.1f})  AUC {row['AUC']:.4f}  "
              f"GT-pair recall @20/50/100: {row['gt_pair_recall@20']:.3f}/"
              f"{row['gt_pair_recall@50']:.3f}/{row['gt_pair_recall@100']:.3f}  "
              f"sampler coverage {row['pair_recall_at_budget']:.4f}")

    out = a.out or os.path.join(os.path.dirname(a.checkpoint), "relatedness.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
