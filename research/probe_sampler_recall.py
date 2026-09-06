"""Where does the pair sampler lose GT, and what would it cost to stop?

THE CEILING. On Haystack's adjudicated cells, 11.4% were never proposed by the
sampler at all — they score exactly 0.0 no matter what the relation head or the
calibration does. That is a hard recall cap of 88.6% that no scoring work can
touch, and it is currently the largest single lever in the system.

WHAT THIS PROBE ESTABLISHES, rather than assumes:

  1. WHICH STAGE loses them. relsgg/sampler.py is a cascade —
     geometry MLP over all N^2 ordered pairs -> top geo_budget (K1), then a
     learned relatedness score over the survivors -> top final_budget (K2).
     In the measurement that produced 11.4%, eval set
     final_budget = min(eval_budget=500, geo_budget=400) = 400 = K1, so stage 2
     was a NO-OP and every miss belongs to stage 1. This probe confirms that
     from the data instead of from arithmetic.

  2. WHAT A BETTER STAGE 1 WOULD BUY. The relatedness score is a two-tower
     bilinear form, s(i,j) = <f_s(v_i), f_o(v_j)>/sqrt(d). Scoring ALL N^2
     pairs with it is a single [N,d] x [d,N] matmul — at N=100, d=256 that is
     2.6 MFLOP, LESS than the geometry MLP already being evaluated on all N^2
     pairs. So the cascade's cheap-first-stage rationale does not hold at these
     N: we can rank every pair by the strong scorer for free. This measures the
     recall@K of geo-only, relatedness-only, and the fused score, so the
     redesign is chosen on evidence.

  3. WHAT BUDGET REACHES WHAT RECALL, including whether 100% is reachable at
     any K (it is not, if a GT box was dropped by max_objects first — that loss
     is upstream of the sampler and no budget fixes it).

Runs the backbone once per image and hooks the sampler's inputs, so the
expensive relation head never runs. Read-only: no model changes.

    python training/probe_sampler_recall.py --checkpoint CK \
        --data_root runs/packed/haystack --split test \
        --negatives runs/datamix/haystack_negatives.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data.dataset import RelationDataset, collate_fn  # noqa: E402
from relsgg.model.geometry import RelGeomEncoder  # noqa: E402
from relsgg.text.student import encode_texts_student  # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402

KS = [64, 128, 200, 300, 400, 600, 800, 1200, 1600, 2400, 3200, 6400, 10000]


@torch.no_grad()
def all_pair_scores(sampler, boxes, obj_feats, box_counts):
    """geo and relatedness scores for EVERY ordered pair. [B, N*N] each."""
    B, N, _ = obj_feats.shape
    dev = boxes.device
    ar = torch.arange(N, device=dev)
    valid_box = ar.unsqueeze(0) < box_counts.unsqueeze(1)
    pair_valid = (valid_box.unsqueeze(2) & valid_box.unsqueeze(1)
                  & (ar.unsqueeze(0) != ar.unsqueeze(1))).reshape(B, N * N)

    geo_feats = RelGeomEncoder.features(
        boxes.unsqueeze(2).expand(B, N, N, 4),
        boxes.unsqueeze(1).expand(B, N, N, 4),
).reshape(B, N * N, RelGeomEncoder.NUM_GEO)
    geo = sampler.geo_scorer(geo_feats).squeeze(-1)

    # The whole point: this is ONE matmul for all N^2 pairs, not a gather on
    # the survivors of a weaker stage.
    zs = sampler.f_sub(obj_feats)                       # [B, N, d]
    zo = sampler.f_obj(obj_feats)
    rel = torch.bmm(zs, zo.transpose(1, 2)).reshape(B, N * N) / (sampler.rel_dim ** 0.5)
    return geo.float(), rel.float(), pair_valid


def ranks_of(scores, pair_valid, cells, N):
    """Rank (0-based, descending) of each (s,o) cell under `scores`."""
    s = scores.masked_fill(~pair_valid, torch.finfo(scores.dtype).min)
    order = torch.argsort(s, descending=True)
    rank_of_flat = torch.empty_like(order)
    rank_of_flat.scatter_(0, order, torch.arange(len(order), device=s.device))
    out = []
    for (a, b) in cells:
        # A relation can reference a box index >= N: the pack stores original
        # indices and max_objects truncated the box list. That cell is
        # UNREACHABLE at any pair budget — the loss happened upstream of the
        # sampler — so it must be caught BEFORE a*N+b overflows the grid.
        if a >= N or b >= N:
            out.append(-1)
            continue
        f = a * N + b
        out.append(int(rank_of_flat[f]) if pair_valid[f] else -1)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="runs/packed/haystack")
    p.add_argument("--split", default="test")
    p.add_argument("--negatives", default="")
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ck, "ema").to(dev).eval()
    ds = RelationDataset(root=a.data_root, split=a.split, resolution=a.img_size,
                         max_objects=a.max_objects)
    names = ds.predicate_names
    ck_args = ck.get("args") or {}
    ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    E = encode_texts_student(names, ck_args["text_student"],
                             templates=TRAIN_TEMPLATES, device=dev)
    model.vocab_head.set_vocabulary_matrix(names, E)
    model.reparameterize()
    print(f"sampler: geo_budget={model.sampler.geo_budget}  "
          f"final_budget={model.sampler.final_budget}  "
          f"rel_dim={model.sampler.rel_dim}")

    neg_by_row = {}
    if a.negatives:
        side = json.load(open(a.negatives))
        pid = {n: i for i, n in enumerate(names)}
        row_of = {int(Path(f).stem.split("_")[-1]): i
                  for i, f in enumerate(ds.file_names)}
        for img_id, cells in side["by_image_id"].items():
            r = row_of.get(int(img_id))
            if r is not None:
                neg_by_row[r] = {(int(s), int(o)) for s, o, q in cells if q in pid}
        print(f"negatives joined on {len(neg_by_row):,} rows")

    # Hook the sampler's inputs; the relation head never has to run.
    grab = {}

    def hook(mod, args, kwargs):
        grab["boxes"] = kwargs.get("boxes")
        grab["obj_feats"] = kwargs.get("obj_feats")
        grab["box_counts"] = kwargs.get("box_counts")
        raise _Stop()

    class _Stop(Exception):
        pass

    h = model.sampler.register_forward_pre_hook(hook, with_kwargs=True)

    if a.limit:
        ds = torch.utils.data.Subset(ds, list(range(min(a.limit, len(ds)))))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)

    hits = {k: defaultdict(int) for k in ("geo", "rel", "fused", "oracle")}
    tot = 0
    n_hist, nsq_over = [], 0
    unreachable = 0
    row = -1
    for images, boxes_b, counts_b, targets in loader:
        images = images.to(dev, non_blocking=True)
        boxes_b = boxes_b.to(dev, non_blocking=True)
        counts_b = counts_b.to(dev, non_blocking=True)
        try:
            with torch.amp.autocast("cuda", enabled=dev.type == "cuda",
                                    dtype=torch.bfloat16):
                model(images, boxes_b, counts_b, targets=None)
        except _Stop:
            pass
        geo, rel, pv = all_pair_scores(model.sampler, grab["boxes"],
                                       grab["obj_feats"].float(),
                                       grab["box_counts"])
        N = grab["boxes"].shape[1]
        # z-score each so the sum is not dominated by whichever has more spread
        for b in range(geo.shape[0]):
            row += 1
            rels = targets[b].get("relations")
            cells = set()
            if rels is not None and len(rels):
                cells |= {(int(r[0]), int(r[1])) for r in rels}
            cells |= neg_by_row.get(row, set())
            cells = [c for c in cells if c[0] != c[1]]
            if not cells:
                continue
            n = int(counts_b[b])
            n_hist.append(n)
            nsq_over += (n * n > model.sampler.geo_budget)
            g, r_, v = geo[b], rel[b], pv[b]
            gz = (g - g[v].mean()) / g[v].std().clamp(min=1e-6)
            rz = (r_ - r_[v].mean()) / r_[v].std().clamp(min=1e-6)
            rk = {"geo": ranks_of(g, v, cells, N),
                  "rel": ranks_of(r_, v, cells, N),
                  "fused": ranks_of(gz + rz, v, cells, N)}
            for c, gr in zip(cells, rk["geo"]):
                tot += 1
                if gr < 0:            # cell references a box that does not exist
                    unreachable += 1  # (max_objects truncation) — no K fixes it
            for name in ("geo", "rel", "fused"):
                for x in rk[name]:
                    for k in KS:
                        if 0 <= x < k:
                            hits[name][k] += 1
            for k in KS:
                hits["oracle"][k] += sum(1 for x in rk["geo"] if x >= 0)

    h.remove()
    print(f"\ncells scored {tot:,}   images {len(n_hist):,}")
    print(f"boxes/img: mean {np.mean(n_hist):.1f} median {int(np.median(n_hist))} "
          f"p90 {int(np.percentile(n_hist,90))} max {max(n_hist)}")
    print(f"images where N^2 > geo_budget ({model.sampler.geo_budget}): "
          f"{nsq_over}/{len(n_hist)} = {nsq_over/max(len(n_hist),1):.1%}")
    print(f"cells UNREACHABLE at any K (box dropped by max_objects): "
          f"{unreachable:,} = {unreachable/max(tot,1):.2%}")
    reach = tot - unreachable
    print(f"\nrecall@K over the {reach:,} reachable cells "
          f"(ceiling {reach/max(tot,1):.4f} of all cells)")
    print(f"  {'K':>7s} {'geo (stage1 today)':>19s} {'relatedness':>13s} {'fused':>9s}")
    res = {}
    for k in KS:
        g = hits["geo"][k] / max(reach, 1)
        r_ = hits["rel"][k] / max(reach, 1)
        f = hits["fused"][k] / max(reach, 1)
        res[k] = {"geo": g, "rel": r_, "fused": f}
        print(f"  {k:7d} {g:19.4f} {r_:13.4f} {f:9.4f}")
    out = a.out or "runs/calib/sampler_recall.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"recall_at_k": res, "n_cells": tot, "unreachable": unreachable,
               "geo_budget": model.sampler.geo_budget,
               "boxes_per_image": {"mean": float(np.mean(n_hist)),
                                   "max": int(max(n_hist))}},
              open(out, "w"), indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
