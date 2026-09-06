"""export_ours_interchange.py — write OUR model's SGDet predictions in the same
interchange format `run_ovsgtr_pack.py` writes for OvSGTR.

Why: `score_native_protocol.py` reproduces OvSGTR's published numbers exactly by
reimplementing their matcher. Once both models are in the same file format, that one
script scores both under EITHER protocol, and the OvR-SGG table finally becomes
comparable to the published leaderboard instead of only to itself.

Fairness contract (all four must hold or the comparison is not matched):
  * same boxes      -- pass --det pointing at the detections converted from the
                       OvSGTR checkpoint's own GroundingDINO output
  * same box labels -- taken from that same detection file, so object classes are
                       identical for both models (our model predicts no object class)
  * same ranking    -- pairs are sorted by pred * conf(sub) * conf(obj) descending,
                       which is graph_infer.py:99-107's rule, NOT our own
  * same cap        -- --det_max_objects must not clip below what the detector emitted

The stored `rel_scores` gets a zero background column at index 0 so the layout matches
OvSGTR's 51-column array and the consumer's `[:, 1:]` slice means the same thing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from relsgg.training.engine import evaluate                          # noqa: E402
from relsgg.scoring import ScoreContract                          # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402
from benchmark.eval_zeroshot_detbox import DetBoxDataset, collate, det_class_names  # noqa: E402


class InterchangeCollector:
    """Duck-types the evaluator interface so `evaluate()` drives it unchanged."""

    def __init__(self, score_mode: str, max_pairs: int, contract=None):
        self.score_mode, self.max_pairs = score_mode, max_pairs
        # MUST be the same object SGClsEvaluator uses, or the exported scores are
        # not the scores our reported numbers were computed from.
        self.contract = contract or ScoreContract()
        self.reset()

    def reset(self):
        self.pairs, self.scores, self.boxes, self.labels, self.bscores = [], [], [], [], []
        self.n_pairs, self.n_boxes = [], []

    @torch.no_grad()
    def update(self, out, targets):
        # Mirrors SGClsEvaluator.update exactly, including the pair_logits fusion.
        logits = out["logits"].float()
        pair_logits = (None if out.get("pair_logits") is None
                       else out["pair_logits"].float())
        if self.score_mode == "softmax":
            prob = torch.softmax(logits, dim=-1)
            if pair_logits is not None:
                prob = prob * torch.sigmoid(pair_logits).unsqueeze(-1)
        else:
            prob = self.contract.scores(logits, pair_logits)
        sub, obj = out["sub_idx"], out["obj_idx"]
        valid = out["valid_mask"]
        for b, t in enumerate(targets):
            v = valid[b]
            s, o = sub[b][v].cpu().numpy(), obj[b][v].cpu().numpy()
            p = prob[b][v].cpu().numpy().astype(np.float32)
            bx = t["det_boxes_px"].numpy().astype(np.float32)
            lb = t["det_labels"].numpy().astype(np.int32)
            bs = (t["box_scores"].numpy().astype(np.float32)
                  if "box_scores" in t else np.ones(len(bx), np.float32))
            # OvSGTR's ranking rule, applied to our scores so neither model gets a
            # ranking advantage: pred * conf(sub) * conf(obj), descending.
            if len(s):
                key = p.max(1) * bs[np.clip(s, 0, len(bs) - 1)] * bs[np.clip(o, 0, len(bs) - 1)]
                order = np.argsort(-key, kind="stable")[:self.max_pairs]
                s, o, p = s[order], o[order], p[order]
            self.pairs.append(np.stack([s, o], 1).astype(np.int32)
                              if len(s) else np.zeros((0, 2), np.int32))
            # prepend a zero background column -> OvSGTR's 51-column layout
            self.scores.append(np.concatenate(
                [np.zeros((len(p), 1), np.float32), p], 1).astype(np.float16))
            self.boxes.append(bx)
            self.labels.append(lb)
            self.bscores.append(bs)
            self.n_pairs.append(len(s))
            self.n_boxes.append(len(bx))

    def compute(self):
        return {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", default="runs/packed/vg150")
    p.add_argument("--split", default="test")
    p.add_argument("--det", required=True)
    p.add_argument("--det_vocab", default="pack", choices=["weights", "pack"])
    p.add_argument("--det_weights", default="")
    p.add_argument("--det_conf", type=float, default=0.0)
    p.add_argument("--det_max_objects", type=int, default=150)
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--score_mode", default="sigmoid", choices=["sigmoid", "softmax"])
    p.add_argument("--text_student", default=None)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=4000)
    p.add_argument("--geo_budget", type=int, default=4000,
                   help="the REAL stage-1 pair cap; eval_budget is clamped to it. "
                        "OvSGTR scores all N(N-1) pairs (~9.3k on its own boxes), so "
                        "leaving our default 400 here would compare a 96%% prune "
                        "against an exhaustive one.")
    p.add_argument("--max_pairs", type=int, default=300,
                   help="store this many top-ranked pairs/img. Only the top 100 can "
                        "affect R@20/50/100, so this is lossless with headroom.")
    p.add_argument("--limit", type=int, default=0,
                   help="debug: export only the first N images (image_index stays "
                        "pack-aligned, so the result is scorable as-is)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()

    meta = json.load(open(os.path.join(args.dataset_root, args.split, "meta.json")))
    pred_names = list(meta["predicates"])

    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    text_student = (args.text_student if args.text_student is not None
                    else ck_args.get("text_student") or "")
    print(f"reparameterizing vocab head to {len(pred_names)} predicates")
    if text_student:
        print(f"  vocabulary encoder: STUDENT ({text_student})")
        from relsgg.text.student import encode_texts_student
        E = encode_texts_student(pred_names, text_student,
                                 templates=TRAIN_TEMPLATES, device=device)
        model.vocab_head.set_vocabulary_matrix(pred_names, E)
    else:
        print("  vocabulary encoder: dino.txt TEACHER")
        raise SystemExit(
            "this checkpoint names no text student. The vocabulary has to be "
            "encoded by the encoder the head was trained against; pass "
            "--text_student, or use a released model, which ships its own.")
    model.reparameterize()

    raw = model.module if hasattr(model, "module") else model
    if args.geo_budget > 0:
        print(f"sampler geo_budget {raw.sampler.geo_budget} -> {args.geo_budget}")
        raw.sampler.geo_budget = args.geo_budget

    class_remap = {n: i for i, n in enumerate(meta["categories"])}
    d_names = (list(meta["categories"]) if args.det_vocab == "pack"
               else det_class_names(args.det_weights))
    ds = DetBoxDataset(args.dataset_root, args.det, d_names, args.iou_thr,
                       args.det_conf, args.det_max_objects, args.img_size,
                       require_class=False, class_remap=class_remap,
                       weight_by_conf=True, split=args.split)
    if args.limit > 0:
        ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=True)

    coll = InterchangeCollector(args.score_mode, args.max_pairs)
    eval_args = type("A", (), {"amp": device.type == "cuda",
                               "amp_dtype_t": torch.bfloat16})()
    evaluate(model, loader, device, eval_args, coll, eval_budget=args.eval_budget)

    n = len(coll.n_pairs)
    print(f"collected {n} images, {sum(coll.n_pairs)} pairs, {sum(coll.n_boxes)} boxes")
    info = {"model": "RelSGG", "checkpoint": args.checkpoint,
            "pack": f"{args.dataset_root}/{args.split}", "box_source": "det",
            "det": args.det, "det_conf": args.det_conf,
            "det_max_objects": args.det_max_objects,
            "score_semantics": args.score_mode, "bg_column": 0,
            "n_images": n, "label_base": 0,
            "geo_budget": args.geo_budget, "eval_budget": args.eval_budget,
            "max_pairs": args.max_pairs,
            "ranking": "pred * conf(sub) * conf(obj) [graph_infer.py:99-107]"}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        image_index=np.arange(n, dtype=np.int32),
        pair_ptr=np.concatenate([[0], np.cumsum(coll.n_pairs)]).astype(np.int64),
        box_ptr=np.concatenate([[0], np.cumsum(coll.n_boxes)]).astype(np.int64),
        pairs=np.concatenate(coll.pairs) if n else np.zeros((0, 2), np.int32),
        rel_scores=np.concatenate(coll.scores),
        boxes=np.concatenate(coll.boxes),
        labels=np.concatenate(coll.labels),
        box_scores=np.concatenate(coll.bscores),
        predicates=np.array(pred_names, dtype="<U32"),
        categories=np.array(list(meta["categories"]), dtype="<U32"),
        meta=np.array([json.dumps(info)], dtype=object).astype("<U4000"))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
