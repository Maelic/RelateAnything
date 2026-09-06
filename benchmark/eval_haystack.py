"""Score a checkpoint against Haystack's explicit negative annotations.

Reports federated per-predicate AP (the headline — Haystack is 8.1:1
negative:positive, where PR-AP behaves and ROC-AUC flatters), plus the
upstream P-AUC / PDD / PDO for comparability with their paper, bucketed by
positive support with the same LVIS-style split the recall metrics use.

Runs the CLOSED-VOCABULARY protocol: the head is reparameterized to Haystack's
56 predicates (byte-identical to our PSG pack), because PDD/PDO are defined
over the rank of a predicate within that 56-way score vector.

No model change: unsampled pairs score 0.0, which is what the deployed system
emits when the relatedness sampler drops a pair. See relsgg/haystack_eval.py.

    python benchmark/eval_haystack.py \
        --checkpoint runs/train/relsgg-vits16plus/model.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                      # noqa: E402
from relsgg.eval.haystack import HaystackEvaluator                # noqa: E402
from relsgg.training.engine import evaluate                          # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES                          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt             # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pack", default="runs/packed/haystack")
    p.add_argument("--negatives", default="runs/datamix/haystack_negatives.json")
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--score_mode", default="sigmoid", choices=["sigmoid", "softmax"])
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--out_dir", default="")
    p.add_argument("--no_pair", action="store_true",
                   help="drop the relatedness term (a contact prior). fAP is "
                        "cross-pair, so pair_logits dominates it exactly as on "
                        "SpatialSense; this is the control for whether a "
                        "calibration gain is representation or rescaling.")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {a.checkpoint} (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, a.weights).to(device).eval()

    ds = RelationDataset(root=a.pack, split="test", resolution=a.img_size,
                         max_objects=a.max_objects)
    pred_names = ds.predicate_names
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)

    # Join the sidecar (keyed by Haystack image_id, predicates by NAME) onto
    # pack row indices and the pack's own predicate order. Both remappings are
    # mandatory: the pack assigns predicate ids by first appearance, and its
    # row order is its own.
    side = json.load(open(a.negatives))
    pid = {n: i for i, n in enumerate(pred_names)}
    row_of = {int(Path(f).stem.split("_")[-1]): i
              for i, f in enumerate(ds.file_names)}
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
    print(f"negatives: {n_neg:,} cells over {len(neg_by_index):,} images "
          f"(skipped {n_skip})")
    assert n_skip == 0, "negative cells failed to join — check id/name mapping"

    # Reparameterize to Haystack's 56 predicates (== our PSG vocabulary).
    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    ts = ck_args.get("text_student") or ""
    if not ts:
        raise SystemExit("checkpoint was not trained in student text space")
    from relsgg.text.student import encode_texts_student
    E = encode_texts_student(pred_names, ts, templates=TRAIN_TEMPLATES,
                             device=device)
    model.vocab_head.set_vocabulary_matrix(pred_names, E)
    model.reparameterize()

    ev = HaystackEvaluator(neg_by_index, num_predicates=len(pred_names),
                           score_mode=a.score_mode, use_pair=not a.no_pair)
    eval_args = SimpleNamespace(amp=device.type == "cuda",
                                amp_dtype_t=torch.bfloat16)
    metrics = evaluate(model, loader, device, eval_args, ev,
                       eval_budget=a.eval_budget)

    print("\n" + "  ".join(f"{k}: {v:.4f}" for k, v in sorted(metrics.items())))
    order = sorted(ev.per_class, key=lambda p: ev.n_pos[p])
    print(f"\n{'predicate':22s} {'n_pos':>6} {'fAP':>7} {'P-AUC':>7} {'PDD':>6} {'PDO':>6}")
    for q in order[:8] + order[-5:]:
        m = ev.per_class[q]
        print(f"  {pred_names[q]:20s} {ev.n_pos[q]:>6} {m['fAP']:>7.4f} "
              f"{m.get('PAUC', float('nan')):>7.4f} {m['PDD']:>6.3f} {m['PDO']:>6.3f}")

    out_dir = a.out_dir or os.path.dirname(a.checkpoint)
    out = os.path.join(out_dir, "haystack_%s%s.json" % (
        a.score_mode, "_nopair" if a.no_pair else ""))
    json.dump({"metrics": metrics,
               "per_class": {pred_names[q]: {**v, "n_pos": ev.n_pos[q]}
                             for q, v in ev.per_class.items()},
               "score_mode": a.score_mode, "weights": a.weights,
               "eval_budget": a.eval_budget}, open(out, "w"), indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
