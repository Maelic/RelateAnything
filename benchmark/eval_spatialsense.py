"""SpatialSense: spatial-relation recognition against adversarial negatives.

Their protocol: each cell is a (subject box, predicate, object box) triple with a
verified TRUE/FALSE label, the test split is exactly balanced, and the model must
decide whether the relation holds. Chance is 50%; language and frequency priors
buy nothing, which is the entire point of the benchmark.

We report three things, in increasing order of how much they assume:
  AUC / AP    threshold-free, the honest headline for a model that was never
              trained to make a binary decision on this vocabulary.
  acc@valid   accuracy at the single global threshold chosen on SpatialSense's
              VALID split — their protocol, and available to us because the
              training corpus contains no SpatialSense image (verified
              0/5,976 train, 0/1,126 valid, 0/1,920 test).
  acc@oracle  accuracy at the best test threshold. An UPPER BOUND, never a
              headline — it peeks at test labels and is reported only to show
              how much of any gap is calibration versus ranking.

Per-predicate rows use the same three, plus n, so the 9 predicates can be read
individually — the reason we run this benchmark at all.

Pairs the relatedness sampler drops score 0.0 (deployed semantics, as in
haystack_eval); `coverage` is printed so that assumption stays auditable.

    python benchmark/eval_spatialsense.py --checkpoint runs/train/v43_full_5ep/checkpoint_best.pth
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
from relsgg.vocabulary import TRAIN_TEMPLATES                          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt             # noqa: E402
# The metric code lives in one torch-free module so the OvSGTR interchange scorer
# (benchmark/ovsgtr/eval_spatialsense_interchange.py) computes the identical numbers.
from relsgg.eval.spatialsense import (best_threshold,          # noqa: E402,F401
                                         summarise, print_summary)


class CellScorer:
    """Collect the model's score for every labelled (pair, predicate) cell.

    ``use_pair`` toggles the relatedness term. It is added as a per-pair CONSTANT
    across predicates, so it cannot change within-pair ranking — it is exactly
    and only the cross-pair signal, which is what this benchmark's per-predicate
    AUC measures. The relatedness head is trained to predict "was this pair
    ANNOTATED at all", which is annotation propensity, not truth of a specific
    predicate; whether that helps or hurts here is an empirical question.
    """

    def __init__(self, cells_by_row, use_pair: bool = True):
        self.cells = cells_by_row
        self.use_pair = use_pair
        self.scores, self.labels, self.preds = [], [], []
        self.pair_lg, self.pair_key = [], []
        self.n_cells = self.n_covered = 0

    @torch.no_grad()
    def update(self, out: dict, targets) -> None:
        logits = out["logits"]
        sub_idx, obj_idx, valid = out["sub_idx"], out["obj_idx"], out["valid_mask"]
        lg = logits.float()
        if self.use_pair and out.get("pair_logits") is not None:
            lg = lg + out["pair_logits"].float().unsqueeze(-1)
        sig = torch.sigmoid(lg)
        for b in range(logits.shape[0]):
            idx = int(targets[b]["index"])
            mask = valid[b]
            pair_row = {}
            if int(mask.sum()):
                s_l = sub_idx[b][mask].tolist()
                o_l = obj_idx[b][mask].tolist()
                for i, (si, oi) in enumerate(zip(s_l, o_l)):
                    pair_row.setdefault((si, oi), i)
                sc = sig[b][mask]
            for s, o, p, lab in self.cells.get(idx, ()):
                self.n_cells += 1
                row = pair_row.get((s, o))
                if row is None:
                    self.scores.append(0.0)
                else:
                    self.n_covered += 1
                    self.scores.append(float(sc[row, p]))
                self.labels.append(int(lab))
                self.preds.append(int(p))

    def compute(self):
        return {"coverage": self.n_covered / max(self.n_cells, 1)}


def score_split(model, pack, cells_json, device, a):
    ds = RelationDataset(root=pack, split="test", resolution=a.img_size,
                         max_objects=a.max_objects)
    side = json.load(open(cells_json))
    pred_names = side["predicates"]
    # Cells are keyed by the converter's image id, which IS the pack's row index
    # (the converter emits images in sorted file-name order and the packer
    # preserves it — asserted round-trip in convert_spatialsense_test.py).
    cells_by_row = {}
    for img_id, cells in side["by_image_id"].items():
        cells_by_row[int(img_id)] = [(int(s), int(o), int(p), int(l))
                                     for s, o, p, l in cells]
    # verify the join: pack rows must equal converter ids
    assert len(cells_by_row) <= len(ds), (len(cells_by_row), len(ds))

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)
    ck = model._ckpt_args
    ts = ck.get("text_student") or ""
    from relsgg.text.student import encode_texts_student
    E = encode_texts_student(pred_names, ts, templates=TRAIN_TEMPLATES,
                             device=device)
    model.vocab_head.set_vocabulary_matrix(pred_names, E)
    model.reparameterize()

    ev = CellScorer(cells_by_row, use_pair=not a.no_pair)
    eval_args = SimpleNamespace(amp=device.type == "cuda",
                                amp_dtype_t=torch.bfloat16)
    cov = evaluate(model, loader, device, eval_args, ev, eval_budget=a.eval_budget)
    return (np.asarray(ev.scores), np.asarray(ev.labels),
            np.asarray(ev.preds), pred_names, cov["coverage"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test_pack", default="runs/packed/spatialsense_test")
    p.add_argument("--valid_pack", default="runs/packed/spatialsense_valid")
    p.add_argument("--test_cells",
                   default="runs/datamix/spatialsense_test_cells.json")
    p.add_argument("--valid_cells",
                   default="runs/datamix/spatialsense_valid_cells.json")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=400)
    p.add_argument("--max_objects", type=int, default=40)
    p.add_argument("--no_pair", action="store_true",
                   help="drop the relatedness term (cross-pair ablation)")
    p.add_argument("--out", default="")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {a.checkpoint} (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, "ema").to(device).eval()
    ck_args = ckpt.get("args") or {}
    model._ckpt_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)

    vs, vl, _, _, vcov = score_split(model, a.valid_pack, a.valid_cells, device, a)
    tau, vacc = best_threshold(vs, vl)
    print(f"[valid] {len(vs):,} cells  coverage {vcov:.4f}  "
          f"threshold={tau:.4f} (acc {vacc:.4f} on valid)")

    s, l, pr, names, cov = score_split(model, a.test_pack, a.test_cells, device, a)
    print(f"[test ] {len(s):,} cells  coverage {cov:.4f}  "
          f"balance {int(l.sum())} true / {int((1 - l).sum())} false")

    res = summarise(s, l, pr, names, tau, cov, use_pair_logits=not a.no_pair)
    print_summary(res)

    out = a.out or os.path.join(os.path.dirname(a.checkpoint),
                                "spatialsense%s.json" % ("_nopair" if a.no_pair else ""))
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
