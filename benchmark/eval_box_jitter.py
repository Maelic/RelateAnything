"""eval_box_jitter.py — separate BOX-LOCALIZATION brittleness from DETECTOR
RECALL in the GT-box → detector-box gap.

The detbox evals (benchmark/eval_zeroshot_detbox.py) show −60..−80% vs GT
boxes, but that number conflates three things:
  1. the detector missing GT objects entirely (permanent misses, by the
     standard SGDet recall convention),
  2. the detector's class label disagreeing with GT (strict protocol only),
  3. the detector's boxes being *looser* than GT boxes.

This script isolates (3). It evaluates on GT boxes perturbed with synthetic
detector-like jitter: same object set, same box count, same GT relations
(relations reference box INDICES, which jitter does not change) — only the
coordinates move. So there are no missing objects and no label mismatch by
construction.

Read the result as:
  * jitter curve stays flat  → box noise is NOT the problem; the detbox gap
    is dominated by detector recall/labels. Fix the detector (or train on
    detector boxes); box-jitter augmentation would not help.
  * jitter curve collapses   → the model is brittle to box noise because it
    has only ever seen perfect GT boxes (there is no box augmentation in
    data/relation_dataset.py). Box-jitter augmentation during training is
    then a cheap, high-leverage fix.

The jitter is scale-relative (centre shift proportional to box size, size
scaled multiplicatively), so the achieved IoU depends only on the magnitude,
not on the box-size distribution — hence the nominal IoU per magnitude is
computed once by Monte Carlo and reported alongside each row.

magnitude 0.0 re-runs the plain GT-box protocol, so it must reproduce the
existing runs/train/<arm>/zeroshot_<name>.json numbers — a built-in check
that this harness matches the baseline it is being compared against.

Usage:
    python benchmark/eval_box_jitter.py \
        --checkpoint runs/train/full_v33a_50ep_v3/checkpoint_last.pth \
        --data_roots runs/packed/vg150 runs/packed/psg
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

from relsgg.data import RelationDataset, collate_fn                    # noqa: E402
from relsgg.eval.evaluator import SGClsEvaluator                     # noqa: E402
from relsgg.training.engine import evaluate                        # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402


def _perturb(cx, cy, w, h, mag, gen):
    """Detector-like jitter: centre shift proportional to box size, size
    scaled log-uniformly. Scale-relative, so IoU vs the original box is
    invariant to the box's absolute size."""
    shape = cx.shape

    def u():
        return (torch.rand(shape, generator=gen) * 2 - 1) * mag

    return (cx + u() * w, cy + u() * h,
            w * torch.exp(u()), h * torch.exp(u()))


def _iou_cxcywh(a, b):
    """Elementwise IoU between two cxcywh box tensors."""
    def to_xyxy(t):
        cx, cy, w, h = t
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    ax1, ay1, ax2, ay2 = to_xyxy(a)
    bx1, by1, bx2, by2 = to_xyxy(b)
    iw = (torch.min(ax2, bx2) - torch.max(ax1, bx1)).clamp(min=0)
    ih = (torch.min(ay2, by2) - torch.max(ay1, by1)).clamp(min=0)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union.clamp(min=1e-9)


def nominal_iou(mag, n=200_000, seed=0):
    """Expected IoU between a box and its jittered version at this magnitude.
    Scale-invariant, so a unit box is representative of the whole dataset."""
    if mag <= 0:
        return 1.0
    g = torch.Generator().manual_seed(seed)
    one = torch.ones(n)
    base = (one * 0.5, one * 0.5, one, one)
    return float(_iou_cxcywh(base, _perturb(*base, mag, g)).mean())


def make_jitter_collate(mag, seed):
    """collate_fn that jitters the valid boxes of every batch.

    Runs inside DataLoader workers, so it seeds a private generator per
    worker-visible call from a counter — reproducible per (mag, seed) run
    without needing to hand state back to the parent process.
    """
    state = {"n": 0}

    def _collate(batch):
        images, boxes, box_counts, targets = collate_fn(batch)
        if mag <= 0:
            return images, boxes, box_counts, targets
        g = torch.Generator().manual_seed(
            seed * 1_000_003 + os.getpid() * 9176 + state["n"])
        state["n"] += 1

        cx, cy, w, h = boxes.unbind(-1)
        ncx, ncy, nw, nh = _perturb(cx, cy, w, h, mag, g)
        # Keep boxes inside the frame; floor the size so nothing degenerates.
        nw = nw.clamp(1e-3, 1.0)
        nh = nh.clamp(1e-3, 1.0)
        ncx = ncx.clamp(nw / 2, 1 - nw / 2)
        ncy = ncy.clamp(nh / 2, 1 - nh / 2)
        jit = torch.stack([ncx, ncy, nw, nh], dim=-1)

        # Only perturb real boxes — padding must stay exactly zero.
        N = boxes.shape[1]
        valid = (torch.arange(N)[None,:] < box_counts[:, None]).unsqueeze(-1)
        return images, torch.where(valid, jit, boxes), box_counts, targets

    return _collate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_roots", nargs="+", required=True)
    p.add_argument("--jitter", type=float, nargs="+",
                   default=[0.0, 0.05, 0.10, 0.15, 0.20, 0.30],
                   help="Jitter magnitudes to sweep. 0.0 = plain GT boxes "
                        "(must reproduce the existing zeroshot_*.json).")
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--score_mode", default="sigmoid")
    p.add_argument("--text_student", default=None)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()

    eval_args = SimpleNamespace(amp=device.type == "cuda",
                                amp_dtype_t=torch.bfloat16)
    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    text_student = (args.text_student if args.text_student is not None
                    else ck_args.get("text_student") or "")

    iou_of = {m: nominal_iou(m) for m in args.jitter}
    print("jitter → nominal mean IoU: "
          + "  ".join(f"{m:.2f}→{iou_of[m]:.3f}" for m in args.jitter))

    for root in args.data_roots:
        name = os.path.basename(os.path.normpath(root))
        ds = RelationDataset(root=root, split="val",
                             resolution=args.img_size,
                             max_objects=args.max_objects)
        pred_names = ds.predicate_names

        # Vocabulary does not depend on jitter — reparameterize once per root.
        if text_student:
            from relsgg.text.student import encode_texts_student
            E = encode_texts_student(pred_names, text_student,
                                     templates=TRAIN_TEMPLATES, device=device)
            model.vocab_head.set_vocabulary_matrix(pred_names, E)
        else:
            raise SystemExit(
                "this checkpoint names no text student. The vocabulary has to be "
                "encoded by the encoder the head was trained against; pass "
                "--text_student, or use a released model, which ships its own.")
        model.reparameterize()

        print(f"\n=== [{name}] {len(ds)} images, {len(pred_names)} predicates ===")
        print(f"{'jitter':>7} {'IoU':>6} {'R@20':>8} {'R@50':>8} {'R@100':>8} "
              f"{'mR@20':>8} {'mR@50':>8} {'mR@100':>8}")

        rows = []
        for mag in args.jitter:
            loader = DataLoader(
                ds, batch_size=args.batch_size, shuffle=False,
                collate_fn=make_jitter_collate(mag, args.seed),
                num_workers=args.num_workers, pin_memory=True)
            ev = SGClsEvaluator(topk=[20, 50, 100],
                                num_predicates=len(pred_names),
                                score_mode=args.score_mode)
            m = evaluate(model, loader, device, eval_args, ev,
                         eval_budget=args.eval_budget)
            rows.append({"jitter": mag, "nominal_iou": iou_of[mag], **m})
            print(f"{mag:>7.2f} {iou_of[mag]:>6.3f} "
                  f"{m['R@20']:>8.4f} {m['R@50']:>8.4f} {m['R@100']:>8.4f} "
                  f"{m['mR@20']:>8.4f} {m['mR@50']:>8.4f} {m['mR@100']:>8.4f}")
            del loader

        base = rows[0]
        if base["jitter"] == 0.0:
            print(f"[{name}] relative to GT boxes:")
            for r in rows[1:]:
                d50 = 100 * (r["R@50"] - base["R@50"]) / max(base["R@50"], 1e-9)
                dm50 = 100 * (r["mR@50"] - base["mR@50"]) / max(base["mR@50"], 1e-9)
                print(f"  jitter {r['jitter']:.2f} (IoU {r['nominal_iou']:.3f}): "
                      f"R@50 {d50:+.1f}%  mR@50 {dm50:+.1f}%")

        out_path = os.path.join(out_dir, f"boxjitter_{name}.json")
        with open(out_path, "w") as f:
            json.dump({"checkpoint": args.checkpoint, "rows": rows,
                       "weights": args.weights, "score_mode": args.score_mode,
                       "seed": args.seed}, f, indent=2)
        print(f"[{name}] saved → {out_path}")

    print("BOX JITTER SWEEP DONE")


if __name__ == "__main__":
    main()
