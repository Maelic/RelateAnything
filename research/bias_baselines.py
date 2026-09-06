"""How much of the LLM-generated MEGASG label distribution is guessable from
OBJECT-PAIR IDENTITY ALONE, with zero image information?

Two baselines, no pixels touched (CPU-only, no GPU/SLURM needed):

1. **FREQ / motif baseline** (Zellers et al. 2018, "Neural Motifs",
   arXiv:1711.06640) — a co-occurrence lookup table P(predicate | subj_cat,
   obj_cat) built from TRAIN counts, evaluated on held-out VAL. That paper's
   headline finding was that this trivial baseline came surprisingly close to
   full deep SGG models on VG150 — exposing how much of "scene graph
   prediction" was actually label-frequency memorization, not vision. We
   reproduce the same idea here to check whether the gemma-4-26B annotation
   pipeline baked in a similarly strong (subj,obj)->predicate prior (an LLM
   captioner has a lot of world-knowledge prior to lean on, e.g. person+horse
   -> "riding" almost regardless of the actual pose in the image).

2. **Trainable linear/bilinear probe** — Embedding(subj_cat) + Embedding
   (obj_cat) -> Linear -> predicate logits, trained by plain cross-entropy on
   train (subj_cat, obj_cat, pred_id) triples only. A smoothed, generalizing
   version of (1); tells us the CEILING of what a category-pair-only model can
   reach on this data (an embedding can share statistical strength across
   similar categories in ways a raw frequency table cannot).

Metric: given the pair's ranked predicate list, where does the TRUE predicate
land? We report Acc@1, R@5/10/20/50/100 (is it in the top-K), mR@50 (macro
over predicate classes — the project's usual mR@K convention) and MRR (the
project's GT_MRR convention) — split by spatial vs semantic (rels flags bit0)
since spatial predicates are largely geometry-derived (should be LESS
guessable from category identity) while interaction/semantic predicates often
carry a strong category-pair prior (should be MORE guessable).

NOTE on protocol: this is PER-EDGE top-K given the GT category pair for that
specific edge — not the model's per-IMAGE top-K-across-all-candidate-pairs
protocol (SGClsEvaluator). It answers a different, complementary question
("how determined is the predicate by category identity alone") and is most
comparable to the model's GT_MRR / R@50 (strict) numbers, not the graph-level
recall numbers. Report both baselines against the model's own peak epoch
values for context.

Usage:
    python training/bias_baselines.py --root runs/packed/megasg
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))  # avoid thread
# thrashing on a shared login node for what is a genuinely tiny model


def load_split(root: str, split: str):
    """Vectorized reconstruction of (subj_cat, obj_cat, pred_id, is_spatial)
    for every relation in a packed split — no images touched."""
    d = os.path.join(root, split)
    im = np.load(os.path.join(d, "img_meta.npy"))       # [n_img, 7]: id,W,H,b0,nb,r0,nr
    rels = np.load(os.path.join(d, "rels.npy"))          # [n_rel, 5]: sub_l,obj_l,pred,flags,raw
    box_cats = np.load(os.path.join(d, "box_cats.npy"))  # [n_box]

    b0, nr = im[:, 3], im[:, 6]
    rel_img_idx = np.repeat(np.arange(len(im)), nr)
    b0_per_rel = b0[rel_img_idx]
    global_sub = b0_per_rel + rels[:, 0]
    global_obj = b0_per_rel + rels[:, 1]

    subj_cat = box_cats[global_sub].astype(np.int64)
    obj_cat = box_cats[global_obj].astype(np.int64)
    pred_id = rels[:, 2].astype(np.int64)
    is_spatial = (rels[:, 3] & 1).astype(bool)
    return subj_cat, obj_cat, pred_id, is_spatial


# --------------------------------------------------------------------------
# Experiment 1: FREQ / motif co-occurrence baseline
# --------------------------------------------------------------------------

class FreqBaseline:
    """P(predicate | subj_cat, obj_cat) lookup table from train counts, with
    deterministic backoff to the global marginal for predicates never seen
    with a given pair (so every predicate always has a well-defined rank)."""

    def __init__(self, n_cat: int, n_pred: int):
        self.n_cat = n_cat
        self.n_pred = n_pred

    def fit(self, subj_cat, obj_cat, pred_id) -> None:
        pair_key = subj_cat * self.n_cat + obj_cat
        combined = pair_key.astype(np.int64) * (self.n_pred + 1) + pred_id.astype(np.int64)
        uniq, counts = np.unique(combined, return_counts=True)
        u_pair = uniq // (self.n_pred + 1)
        u_pred = uniq % (self.n_pred + 1)

        # global marginal ranking (backoff for unseen pairs / unseen preds-per-pair)
        marginal = np.bincount(pred_id, minlength=self.n_pred)
        order = np.argsort(-marginal, kind="stable")  # descending count, stable tie-break by id
        self.global_rank = np.empty(self.n_pred, dtype=np.int64)
        self.global_rank[order] = np.arange(self.n_pred)

        # Group by pair (primary key, ascending — for contiguous slicing),
        # then within each pair sort by count desc, tie-break by global_rank
        # asc. np.lexsort's LAST key is primary.
        sort_idx = np.lexsort((self.global_rank[u_pred], -counts, u_pair))
        u_pair_s, u_pred_s = u_pair[sort_idx], u_pred[sort_idx]

        uniq_pairs = np.unique(u_pair_s)
        starts = np.searchsorted(u_pair_s, uniq_pairs)
        ends = np.searchsorted(u_pair_s, uniq_pairs, side="right")

        self.pair_preds: Dict[int, np.ndarray] = {}
        self.pair_pred_sets: Dict[int, set] = {}
        self.pair_sorted_gr: Dict[int, np.ndarray] = {}
        for p, s, e in zip(uniq_pairs, starts, ends):
            preds = u_pred_s[s:e]
            self.pair_preds[int(p)] = preds
            self.pair_pred_sets[int(p)] = set(preds.tolist())
            self.pair_sorted_gr[int(p)] = np.sort(self.global_rank[preds])

        self.seen_pairs = set(uniq_pairs.tolist())

    def rank_of(self, subj_cat: int, obj_cat: int, true_pred: int) -> int:
        """0-indexed rank of the true predicate under this pair's merged
        (pair-specific-first, then global-backoff) ranking."""
        key = int(subj_cat) * self.n_cat + int(obj_cat)
        gr_true = int(self.global_rank[true_pred])
        if key not in self.seen_pairs:
            return gr_true
        if true_pred in self.pair_pred_sets[key]:
            return int(np.where(self.pair_preds[key] == true_pred)[0][0])
        primary_len = len(self.pair_preds[key])
        n_before = int(np.searchsorted(self.pair_sorted_gr[key], gr_true))
        return primary_len + (gr_true - n_before)


# --------------------------------------------------------------------------
# Experiment 2: trainable linear probe
# --------------------------------------------------------------------------

class PairPredicateProbe(nn.Module):
    def __init__(self, n_cat: int, n_pred: int, emb_dim: int = 64):
        super().__init__()
        self.sub_emb = nn.Embedding(n_cat, emb_dim)
        self.obj_emb = nn.Embedding(n_cat, emb_dim)
        self.fc = nn.Linear(emb_dim * 2, n_pred)

    def forward(self, subj_cat: torch.Tensor, obj_cat: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.sub_emb(subj_cat), self.obj_emb(obj_cat)], dim=-1)
        return self.fc(x)


def train_probe(subj_cat, obj_cat, pred_id, n_cat, n_pred, epochs=8, batch_size=8192,
                lr=3e-3, device="cpu", emb_dim=64):
    model = PairPredicateProbe(n_cat, n_pred, emb_dim=emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    s = torch.from_numpy(subj_cat).to(device)
    o = torch.from_numpy(obj_cat).to(device)
    p = torch.from_numpy(pred_id).to(device)
    n = len(s)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(s[idx], o[idx])
            loss = nn.functional.cross_entropy(logits, p[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.detach().item(); n_batches += 1
        print(f"  [probe] epoch {ep}: loss {tot_loss / n_batches:.4f}")
    model.eval()
    return model


@torch.no_grad()
def probe_ranks(model, subj_cat, obj_cat, pred_id, device="cpu", batch_size=16384):
    s = torch.from_numpy(subj_cat).to(device)
    o = torch.from_numpy(obj_cat).to(device)
    ranks = np.empty(len(subj_cat), dtype=np.int64)
    for i in range(0, len(s), batch_size):
        logits = model(s[i:i + batch_size], o[i:i + batch_size])
        order = torch.argsort(logits, dim=-1, descending=True)          # [B, n_pred]
        true = torch.from_numpy(pred_id[i:i + batch_size]).to(device).unsqueeze(-1)
        pos = (order == true).float().argmax(dim=-1)                     # rank of true class
        ranks[i:i + batch_size] = pos.cpu().numpy()
    return ranks


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def report(name: str, ranks: np.ndarray, pred_id: np.ndarray, is_spatial: np.ndarray,
          n_pred: int, topk=(1, 5, 10, 20, 50, 100)) -> dict:
    out = {"name": name, "n": len(ranks)}
    for split_name, mask in [("all", np.ones(len(ranks), bool)),
                             ("spatial", is_spatial), ("semantic", ~is_spatial)]:
        r = ranks[mask]
        pid = pred_id[mask]
        if len(r) == 0:
            continue
        sub = {}
        for k in topk:
            sub[f"R@{k}"] = float((r < k).mean())
        sub["MRR"] = float(np.mean(1.0 / (r + 1)))
        # MACRO over predicate classes present in this split, at every k --
        # not just k=50. On a 50-predicate pack mR@50 is trivially 1.000
        # (ranking all 50 always contains the truth), which made the macro
        # column look uninformative and hid the real comparison: FREQ's micro
        # win comes from always answering a category pair's MAJORITY predicate,
        # so its per-class accuracy at k=1 is the number that decides whether a
        # visual model is actually behind it. mAcc@1 == mR@1.
        by_cls = defaultdict(list)
        for ri, pi in zip(r, pid):
            by_cls[int(pi)].append(int(ri))
        for k in topk:
            sub[f"mR@{k}"] = float(np.mean(
                [float(np.mean(np.asarray(v) < k)) for v in by_cls.values()]))
        sub["mAcc@1"] = sub["mR@1"]
        sub["mMRR"] = float(np.mean(
            [float(np.mean(1.0 / (np.asarray(v) + 1.0)))
             for v in by_cls.values()]))
        sub["n_classes"] = int(len(by_cls))
        sub["n"] = int(len(r))
        out[split_name] = sub
    return out


def print_report(rep: dict) -> None:
    print(f"\n=== {rep['name']} ===")
    for split in ("all", "spatial", "semantic"):
        if split not in rep:
            continue
        s = rep[split]
        print(f"  [{split:>8}] n={s['n']:>8}  R@1={s['R@1']*100:5.2f}%  R@5={s['R@5']*100:5.2f}%  "
              f"R@10={s['R@10']*100:5.2f}%  R@20={s['R@20']*100:5.2f}%  R@50={s['R@50']*100:5.2f}%  "
              f"mR@50={s['mR@50']*100:5.2f}%  MRR={s['MRR']:.4f}")
        print(f"  {'':>10}  MACRO over {s['n_classes']} classes: "
              f"mAcc@1={s['mAcc@1']*100:5.2f}%  mR@5={s['mR@5']*100:5.2f}%  "
              f"mMRR={s['mMRR']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/packed/megasg")
    ap.add_argument("--probe_epochs", type=int, default=8)
    ap.add_argument("--probe_emb_dim", type=int, default=64)
    ap.add_argument("--skip_probe", action="store_true",
                   help="FREQ baseline only — skip the trainable linear probe "
                        "(on MEGASG it never beat the raw lookup table).")
    ap.add_argument("--eval_split", default="val",
                   help="split the baselines are SCORED on. Default 'val' "
                        "reproduces the original run; use 'test' to be "
                        "protocol-matched with the model's headline A1 "
                        "numbers, which are all TEST + graph-constrained.")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.root, "train", "meta.json")))
    n_cat, n_pred = len(meta["categories"]), len(meta["predicates"])
    print(f"[bias_baselines] {n_cat} categories, {n_pred} predicates")

    ev = args.eval_split
    print(f"[bias_baselines] loading train/{ev} (no images)...")
    tr_s, tr_o, tr_p, tr_sp = load_split(args.root, "train")
    va_s, va_o, va_p, va_sp = load_split(args.root, ev)
    print(f"[bias_baselines] train rels {len(tr_p)}  {ev} rels {len(va_p)}")
    if not va_sp.any():
        print(f"[bias_baselines] NOTE: the spatial flag (rels col3 bit0) is "
              f"all-zero in this pack, so the spatial/semantic split is "
              f"degenerate — 'semantic' == 'all' and 'spatial' is empty. "
              f"Only megasg-generated packs carry the flag.")

    unique_val_pairs = set((int(s) * n_cat + int(o)) for s, o in zip(va_s, va_o))
    unique_train_pairs = set((int(s) * n_cat + int(o)) for s, o in zip(tr_s, tr_o))
    coverage = len(unique_val_pairs & unique_train_pairs) / max(len(unique_val_pairs), 1)
    print(f"[bias_baselines] {coverage*100:.1f}% of {ev}'s (subj_cat,obj_cat) pairs "
          f"were seen (with SOME predicate) in train")

    results = {"coverage_pair_seen_in_train": coverage, "eval_split": ev}

    # ---- Experiment 1: FREQ / motif baseline ----
    print("\n[bias_baselines] fitting FREQ baseline (train co-occurrence)...")
    freq = FreqBaseline(n_cat, n_pred)
    freq.fit(tr_s, tr_o, tr_p)
    ranks = np.array([freq.rank_of(s, o, p) for s, o, p in zip(va_s, va_o, va_p)])
    rep_freq = report("FREQ baseline (Zellers-2018-style, category pair only)",
                      ranks, va_p, va_sp, n_pred)
    print_report(rep_freq)
    results["freq_baseline"] = rep_freq

    # ---- Experiment 2: trainable linear/bilinear probe ----
    if not args.skip_probe:
        print("\n[bias_baselines] training linear probe (category-pair embeddings)...")
        model = train_probe(tr_s, tr_o, tr_p, n_cat, n_pred, epochs=args.probe_epochs,
                            emb_dim=args.probe_emb_dim)
        ranks_probe = probe_ranks(model, va_s, va_o, va_p)
        rep_probe = report("Trainable linear probe (embed(subj)+embed(obj) -> Linear)",
                           ranks_probe, va_p, va_sp, n_pred)
        print_report(rep_probe)
        results["linear_probe"] = rep_probe
    else:
        print("\n[bias_baselines] --skip_probe set: FREQ baseline only.")

    if "megasg" in args.root:
        print("\n=== For reference: full visual model (full_v33a_50ep_v3, peak epoch, "
              "SAME MEGASG val, DIFFERENT protocol — per-image top-K not per-edge) ===")
        print("  R@50 (strict) peak = 0.2514 (epoch 8)   GT_MRR peak = 0.8971 (epoch 4-6)")

    out_path = args.out or os.path.join(
        args.root, f"bias_baselines{'' if ev == 'val' else '_' + ev}.json")
    json.dump(results, open(out_path, "w"), indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
