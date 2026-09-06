"""Two graphs per image: a spatial LAYOUT graph and a semantic CONTENT graph.

MOTIVATION. The standard graph constraint keeps one predicate per ordered pair.
But `laptop on desk` and `laptop to the left of lamp`-style facts are not
alternatives — a pair can carry a layout relation AND an interaction at the same
time, and forcing one argmax makes them compete. Measured here, that competition
is not neutral: the relatedness term is a CONTACT prior
, so the contact predicate wins the slot
and the projective one is never emitted, whatever the model believes.

PROTOCOL (type-stratified graph constraint). Each ordered pair may emit at most
ONE spatial and ONE semantic edge — its argmax within each type. The two streams
are then ranked and cut at K independently, and each is scored against the GT of
its own type. This is still constrained (no synonym flooding, no pair emitting
ten near-duplicates), it is simply constrained per type rather than globally.

  NOT COMPARABLE to standard single-stream R@K: two streams of K get 2K
  predictions per image. Every number here is reported against its own
  type-restricted GT, and the standard number is printed beside it for context,
  never merged. Deployment reads the same way: layout graph + content graph.

TYPE ASSIGNMENT is data-derived, not hand-written. Each predicate STRING is
labelled from the training corpus's own per-relation `spatial` flag (pack meta
`flags_legend.bit0`), by majority over its instances. That statistic is
essentially bimodal — on megasg val, 2,195 predicates sit at 0.00 and ~27 at
0.80-1.00 with nothing between — so ANY cut in [0.05, 0.79] yields the same
partition, and the rule is a majority vote rather than a tuned threshold.
Benchmark predicates absent from the corpus fall back to semantic, and the count
of such fallbacks is printed so the assumption stays visible.

    python benchmark/eval_decomposed.py --checkpoint runs/train/v43_full_5ep/checkpoint_best.pth
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                      # noqa: E402
# Shared with deploy-time bank building — one definition of the type map
# ([[relsgg/decompose.py]]); this file keeps the evaluation semantics only.
from relsgg.decompose import corpus_spatial_map                   # noqa: E402
from relsgg.training.engine import evaluate                          # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES                          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt             # noqa: E402


class DecomposedEvaluator:
    """Per-type argmax per pair, independent top-K, GT scored within type."""

    def __init__(self, is_spatial: np.ndarray, topk=(20, 50, 100),
                 use_pair: bool = True):
        self.sp = torch.from_numpy(is_spatial.astype(bool))
        self.topk = list(topk)
        self.use_pair = use_pair
        self.tp = {s: {k: defaultdict(int) for k in self.topk}
                   for s in ("spatial", "semantic")}
        self.gt = {s: defaultdict(int) for s in ("spatial", "semantic")}

    @torch.no_grad()
    def update(self, out: dict, targets) -> None:
        logits, valid = out["logits"], out["valid_mask"]
        sub_idx, obj_idx = out["sub_idx"], out["obj_idx"]
        lg_all = logits.float()
        if self.use_pair and out.get("pair_logits") is not None:
            lg_all = lg_all + out["pair_logits"].float().unsqueeze(-1)
        sp = self.sp.to(logits.device)
        for b in range(logits.shape[0]):
            mask = valid[b]
            gt_by_type = {"spatial": set(), "semantic": set()}
            for r in targets[b].get("relations", []):
                s, o, p = int(r[0]), int(r[1]), int(r[2])
                t = "spatial" if bool(sp[p]) else "semantic"
                gt_by_type[t].add((s, o, p))
                self.gt[t][p] += 1
            if not int(mask.sum()):
                continue
            sc = lg_all[b][mask]                       # [K', V]
            s_l = sub_idx[b][mask].tolist()
            o_l = obj_idx[b][mask].tolist()
            for t, sel in (("spatial", sp), ("semantic", ~sp)):
                if not bool(sel.any()):
                    continue
                masked = sc.masked_fill(~sel.unsqueeze(0), float("-inf"))
                best, arg = masked.max(dim=-1)          # one edge per pair
                order = torch.argsort(best, descending=True).tolist()
                ranked = [(s_l[i], o_l[i], int(arg[i])) for i in order]
                for k in self.topk:
                    hit = set(ranked[:k]) & gt_by_type[t]
                    for (_, _, p) in hit:
                        self.tp[t][k][p] += 1

    def compute(self):
        res = {}
        for t in ("spatial", "semantic"):
            tot = sum(self.gt[t].values())
            for k in self.topk:
                got = sum(self.tp[t][k].values())
                res[f"{t}_R@{k}"] = got / tot if tot else float("nan")
                per = [self.tp[t][k][p] / c for p, c in self.gt[t].items() if c]
                res[f"{t}_mR@{k}"] = float(np.mean(per)) if per else float("nan")
            res[f"{t}_n_gt"] = float(tot)
            res[f"{t}_n_cls"] = float(len([c for c in self.gt[t].values() if c]))
        return res


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--packs", nargs="+", default=["runs/packed/vg150",
                                                  "runs/packed/psg",
                                                  "runs/packed/indoorvg"])
    p.add_argument("--split", default="test")
    p.add_argument("--corpus_pack", default="runs/packed/megasg")
    p.add_argument("--corpus_split", default="val")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=400)
    p.add_argument("--no_pair", action="store_true",
                   help="drop the relatedness term (it is a contact prior)")
    p.add_argument("--split_source", choices=["corpus", "alpha", "geo"],
                   default="corpus",
                   help="predicate type source: corpus flags (the measured "
                        "6/6 winner), the checkpoint's own gate alpha>=0.5 "
                        "(covers novel strings; under-routes unseen spatial "
                        "predicates), or 'geo' = geometry-predictability "
                        "(training/build_geo_type_map.py — spatial iff the "
                        "predicate is decidable from GT-box geometry alone; "
                        "the principled fix for the provenance-contaminated "
                        "corpus flag, which puts `on` and `resting on` in "
                        "different streams). Requires --type_map.")
    p.add_argument("--type_map", default="",
                   help="geo_type_map.json from build_geo_type_map.py "
                        "(used when --split_source geo)")
    p.add_argument("--out", default="")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {a.checkpoint} (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, "ema").to(device).eval()
    ck_args = ckpt.get("args") or {}
    ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    ts = ck_args.get("text_student") or ""
    from relsgg.text.student import encode_texts_student

    smap = corpus_spatial_map(a.corpus_pack, a.corpus_split)
    print(f"type map: {sum(smap.values())} spatial / {len(smap)} corpus predicates")
    geo_map = None
    if a.split_source == "geo":
        if not a.type_map:
            raise SystemExit("--split_source geo requires --type_map")
        geo_map = json.load(open(a.type_map))["packs"]

    res = {}
    for pack in a.packs:
        ds = RelationDataset(root=pack, split=a.split, resolution=a.img_size,
                             max_objects=40)
        names = ds.predicate_names
        is_sp = np.array([smap.get(n, False) for n in names])
        n_missing = sum(1 for n in names if n not in smap)
        loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=a.num_workers,
                            pin_memory=True)
        E = encode_texts_student(names, ts, templates=TRAIN_TEMPLATES,
                                 device=device)
        model.vocab_head.set_vocabulary_matrix(names, E)
        model.reparameterize()
        if a.split_source == "alpha":
            # The head's own router, evaluated on this vocabulary: alpha is
            # recomputed by set_vocabulary_matrix for exactly these rows.
            is_sp = (model.vocab_head.alpha.detach().cpu().numpy() >= 0.5)
            n_missing = 0
        elif a.split_source == "geo":
            gm = geo_map[os.path.basename(pack.rstrip("/"))]["map"]
            is_sp = np.array([gm.get(n, {}).get("spatial", False)
                              for n in names])
            n_missing = sum(1 for n in names if n not in gm)
        ev = DecomposedEvaluator(is_sp, use_pair=not a.no_pair)
        m = evaluate(model, loader, device,
                     SimpleNamespace(amp=device.type == "cuda",
                                     amp_dtype_t=torch.bfloat16),
                     ev, eval_budget=a.eval_budget)
        nm = os.path.basename(pack)
        res[nm] = m
        print(f"\n[{nm}] {int(is_sp.sum())}/{len(names)} predicates typed spatial "
              f"({n_missing} not in corpus -> semantic)")
        for t in ("spatial", "semantic"):
            print(f"   {t:9s} R@50 {m[f'{t}_R@50']:.4f}  mR@50 {m[f'{t}_mR@50']:.4f}"
                  f"   (GT {int(m[f'{t}_n_gt']):,} over {int(m[f'{t}_n_cls'])} classes)")

    out = a.out or os.path.join(
        os.path.dirname(a.checkpoint),
        "decomposed%s%s.json" % (
            {"corpus": "", "alpha": "_alpha", "geo": "_geo"}[a.split_source],
            "_nopair" if a.no_pair else ""))
    json.dump({"use_pair_logits": not a.no_pair,
               "split_source": a.split_source, "per_source": res},
              open(out, "w"), indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
