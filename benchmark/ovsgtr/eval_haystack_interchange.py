"""Score an interchange prediction file against Haystack's federated negatives.

Haystack is the only source in the benchmark that carries EXPLICIT negatives, so it is
the only place false positives on rare predicates can be measured at all — everywhere
else an unlisted relation is merely unlabelled. This runs the identical
`HaystackEvaluator` used for our own checkpoints, so fAP / P-AUC / PDD / PDO are
computed by the same code for both models.

COVERAGE IS NOT SYMMETRIC AND FAVOURS OvSGTR. An unsampled pair scores 0.0 (what a
deployed system emits when its sampler drops the pair). OvSGTR's graph_infer enumerates
every N*(N-1) pair, so its coverage is ~100%; our relatedness sampler prunes to
geo_budget, measured at 88.3%. Report `coverage` alongside the metrics.

Runs in the RELSGG venv.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from relsgg.eval.haystack import HaystackEvaluator  # noqa: E402

EPS = 1e-6


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True)
    p.add_argument("--pack", default="runs/packed/haystack/test")
    p.add_argument("--negatives", default="runs/datamix/haystack_negatives.json")
    p.add_argument("--score_mode", default="sigmoid", choices=["sigmoid", "softmax"])
    p.add_argument("--out", default=None)
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args()

    _npz = np.load(args.pred, allow_pickle=False)
    d = {k: _npz[k] for k in _npz.files}     # NpzFile re-inflates on every access
    info = json.loads(str(d["meta"][0]))
    if info.get("box_source") != "gt":
        raise SystemExit("Haystack cells are indexed by GT box id; requires --boxes gt")
    pred_names = [str(x) for x in d["predicates"]]
    bg = int(info.get("bg_column", 0))

    pack = Path(args.pack)
    meta = json.loads((pack / "meta.json").read_text())
    if list(meta["predicates"]) != pred_names:
        raise SystemExit("predicate vocabulary mismatch; refusing to score")
    file_names = json.loads((pack / "file_names.json").read_text())
    img_meta = np.load(pack / "img_meta.npy")
    rels_all = np.load(pack / "rels.npy")

    # Same join as benchmark/eval_haystack.py: the sidecar keys by Haystack image_id and
    # stores predicate NAMES, because the pack assigns predicate ids by first
    # appearance and its row order is its own.
    side = json.load(open(args.negatives))
    pid = {n: i for i, n in enumerate(pred_names)}
    row_of = {int(Path(f).stem.split("_")[-1]): i for i, f in enumerate(file_names)}
    neg_by_index, n_neg, n_skip = {}, 0, 0
    for img_id, cells in side["by_image_id"].items():
        r = row_of.get(int(img_id))
        if r is None:
            n_skip += len(cells)
            continue
        keep = [[int(s), int(o), pid[q]] for s, o, q in cells if q in pid]
        n_skip += len(cells) - len(keep)
        if keep:
            neg_by_index[r] = keep
            n_neg += len(keep)
    print(f"negatives: {n_neg:,} cells over {len(neg_by_index):,} images (skipped {n_skip})")
    assert n_skip == 0, "negative cells failed to join — check id/name mapping"

    ev = HaystackEvaluator(neg_by_index, num_predicates=len(pred_names),
                           score_mode=args.score_mode)

    pair_ptr = d["pair_ptr"]
    idxs = d["image_index"]
    V = len(pred_names)
    for s in range(0, len(idxs), args.batch):
        items = []
        for i in range(s, min(s + args.batch, len(idxs))):
            row = int(idxs[i])
            a, b = int(pair_ptr[i]), int(pair_ptr[i + 1])
            _iid, _w, _h, _b0, nb, r0, nr = (int(x) for x in img_meta[row])
            gt = np.asarray(rels_all[r0:r0 + nr], dtype=np.int64)
            gt = gt[(gt[:, 0] < nb) & (gt[:, 1] < nb)][:,:3] if gt.size else \
                np.zeros((0, 3), np.int64)
            prob = np.delete(d["rel_scores"][a:b].astype(np.float32), bg, axis=1)
            items.append((row, d["pairs"][a:b], prob, gt))
        if not items:
            continue

        K = max(1, max(len(x[1]) for x in items))
        B = len(items)
        logits = torch.full((B, K, V), -30.0)
        sub = torch.zeros(B, K, dtype=torch.long)
        obj = torch.zeros(B, K, dtype=torch.long)
        valid = torch.zeros(B, K, dtype=torch.bool)
        targets = []
        for j, (row, pairs, prob, gt) in enumerate(items):
            k = len(pairs)
            if k:
                # HaystackEvaluator re-applies the activation, so invert to logits.
                pr = torch.from_numpy(np.clip(prob, EPS, 1 - EPS))
                logits[j,:k] = torch.log(pr / (1 - pr)) if args.score_mode == "sigmoid" \
                    else torch.log(pr)
                sub[j,:k] = torch.from_numpy(pairs[:, 0].astype(np.int64))
                obj[j,:k] = torch.from_numpy(pairs[:, 1].astype(np.int64))
                valid[j,:k] = True
            targets.append({"index": row, "relations": torch.from_numpy(gt)})

        ev.update({"logits": logits, "sub_idx": sub, "obj_idx": obj,
                   "valid_mask": valid, "pair_logits": None}, targets)

    metrics = ev.compute()
    print("\n" + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items())))
    out = args.out
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"metrics": metrics,
                   "per_class": {pred_names[q]: {**v, "n_pos": ev.n_pos[q]}
                                 for q, v in ev.per_class.items()},
                   "score_mode": args.score_mode, "source": info},
                  open(out, "w"), indent=2)
        print("wrote", out)


if __name__ == "__main__":
    main()
