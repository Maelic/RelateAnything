"""Score an interchange-format prediction file with the RelSGG evaluator.

This is the half of the comparison that makes it fair: `run_ovsgtr_pack.py` produces a
model-agnostic record, and this scores it with the SAME `SGClsEvaluator` and the SAME
graph-constraint / bucket / IDF settings used for our own checkpoints
. No metric
is reimplemented here.

Runs in the RELSGG venv (it imports relsgg.evaluator), NOT OvSGTR's.

SCORE SEMANTICS
---------------
The interchange stores post-activation probabilities, while SGClsEvaluator takes
pre-activation logits and applies the activation itself. We therefore invert:
  sigmoid mode -> logit(p),  so sigmoid(logit(p)) == p
  softmax mode -> log(p),    so softmax(log p)   == p  (softmax is shift-invariant)
This reuses the evaluator untouched instead of adding a probability branch to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from relsgg.eval.evaluator import SGClsEvaluator  # noqa: E402
from benchmark.eval_detboxes import cxcywh_to_xyxy, pairwise_iou, greedy_match  # noqa: E402

EPS = 1e-6
SENTINEL = 1 << 16  # never a valid box index; makes an unmatched GT a permanent miss


def load_pack_gt(pack: Path):
    meta = json.loads((pack / "meta.json").read_text())
    return (np.load(pack / "img_meta.npy"), np.load(pack / "rels.npy"),
            np.load(pack / "boxes.npy", mmap_mode="r"), list(meta["predicates"]))


def remap_gt_to_boxes(gt_rels, gt_boxes_cxcywh, pred_xyxy, W, H, iou_thr):
    """Re-index GT relation endpoints from GT-box space into predicted-box space.

    Required whenever the boxes are NOT ground truth: the evaluator compares predicted
    (sub_idx, obj_idx) against GT (sub, obj), and those index spaces only coincide
    under the GT-box protocol. Unmatched endpoints are mapped to SENTINEL rather than
    dropped, so GT relations the detector never recovered stay in the denominator —
    the standard SGDet recall convention, and the reason detector-box numbers are much
    lower than oracle-box ones.
    """
    gt_xyxy = cxcywh_to_xyxy(np.asarray(gt_boxes_cxcywh, dtype=np.float32))
    gt_xyxy = gt_xyxy * np.array([W, H, W, H], np.float32)
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        m = np.full(len(gt_xyxy), -1, dtype=np.int64)
    else:
        m = greedy_match(pairwise_iou(gt_xyxy, pred_xyxy), iou_thr)
    sub = np.where(m[gt_rels[:, 0]] >= 0, m[gt_rels[:, 0]], SENTINEL)
    obj = np.where(m[gt_rels[:, 1]] >= 0, m[gt_rels[:, 1]], SENTINEL)
    return np.stack([sub, obj, gt_rels[:, 2]], axis=1).astype(np.int64)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True, help="npz from run_ovsgtr_pack.py")
    p.add_argument("--pack", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--topk", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--graph_constraint", action="store_true", default=True)
    p.add_argument("--no_graph_constraint", dest="graph_constraint", action="store_false")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--iou_thr", type=float, default=0.5,
                   help="IoU for matching GT boxes to predicted boxes (non-GT box modes)")
    p.add_argument("--no_conf_weight", action="store_true",
                   help="rank by predicate score only, ignoring detector box confidence "
                        "(ablation). Default OFF: the SGDet convention, and OvSGTR's own, "
                        "is pred * conf(sub) * conf(obj).")
    p.add_argument("--max_pairs", type=int, default=0,
                   help="keep only the first N pairs per image (graph_infer sorted them "
                        "by ITS score, which multiplies in object confidences, while we "
                        "rank by predicate score alone -- so a prefix is NOT obviously "
                        "lossless for our top-K. 0 = keep all.")
    p.add_argument("--limit", type=int, default=0, help="score only the first N images")
    p.add_argument("--ovr_split", action="store_true",
                   help="also report base/novel subset recall using OvSGTR's own VG150 "
                        "OvR split (their datasets/vg.py). Only meaningful on a vg150 "
                        "pack; the subset breakdown is what their leaderboard reports.")
    args = p.parse_args()

    _npz = np.load(args.pred, allow_pickle=False)
    # NpzFile is lazy and DECOMPRESSES THE WHOLE ARRAY on every __getitem__, so
    # `d["rel_scores"][a:b]` inside the per-image loop would re-inflate ~47 MB 2,179
    # times. Materialise once.
    d = {k: _npz[k] for k in _npz.files}
    info = json.loads(str(d["meta"][0]))
    pred_names = [str(x) for x in d["predicates"]]
    bg = int(info.get("bg_column", 0))
    mode = info.get("score_semantics", "sigmoid")

    img_meta, rels_all, boxes_all, pack_preds = load_pack_gt(Path(args.pack))
    if pack_preds != pred_names:
        raise SystemExit("predicate vocabulary in prediction file != pack; refusing to score")

    box_source = info.get("box_source", "gt")
    needs_remap = box_source != "gt"
    if needs_remap:
        print(f"box_source={box_source}: IoU-remapping GT relations into predicted-box "
              f"index space at iou_thr={args.iou_thr}")

    subsets = None
    if args.ovr_split:
        from benchmark.eval_ovsgtr_novel import (VG150_NOVEL_PREDICATE,  # noqa: E402
                                                 VG150_BASE_PREDICATE)
        n2i = {n: i for i, n in enumerate(pred_names)}
        missing = [n for n in VG150_NOVEL_PREDICATE + VG150_BASE_PREDICATE if n not in n2i]
        if missing:
            raise SystemExit(f"--ovr_split: pack vocabulary is missing {missing[:5]} "
                             "— this flag is only valid on a VG150 pack")
        subsets = {"novel": [n2i[n] for n in VG150_NOVEL_PREDICATE],
                   "base":  [n2i[n] for n in VG150_BASE_PREDICATE]}
        print(f"--ovr_split: {len(subsets['novel'])} novel / {len(subsets['base'])} base")

    ev = SGClsEvaluator(topk=args.topk, num_predicates=len(pred_names),
                        score_mode=mode, graph_constraint=args.graph_constraint,
                        subsets=subsets)

    pair_ptr, box_ptr = d["pair_ptr"], d["box_ptr"]
    idxs = d["image_index"]
    n = len(idxs)
    n_scored = n_skipped_empty = n_skipped_nogt = 0

    if args.limit > 0:
        n = min(n, args.limit)
    for s in range(0, n, args.batch):
        chunk = range(s, min(s + args.batch, n))
        items = []
        for i in chunk:
            a, b = int(pair_ptr[i]), int(pair_ptr[i + 1])
            if b <= a:
                n_skipped_empty += 1
                continue
            if args.max_pairs > 0 and (b - a) > args.max_pairs:
                b = a + args.max_pairs
            _iid, w, h, b0, nb, r0, nr = (int(x) for x in img_meta[int(idxs[i])])
            gt = np.asarray(rels_all[r0:r0 + nr], dtype=np.int64)
            if gt.size == 0:
                n_skipped_nogt += 1
                continue
            gt = gt[(gt[:, 0] < nb) & (gt[:, 1] < nb)][:,:3]
            if gt.size == 0:
                n_skipped_nogt += 1
                continue
            if needs_remap:
                ba, bb = int(box_ptr[i]), int(box_ptr[i + 1])
                gt = remap_gt_to_boxes(gt, boxes_all[b0:b0 + nb], d["boxes"][ba:bb],
                                       w, h, args.iou_thr)
            prob = d["rel_scores"][a:b].astype(np.float32)
            prob = np.delete(prob, bg, axis=1)  # drop OvSGTR's background column
            # Standard SGDet triplet score is pred. conf(sub). conf(obj), and that is
            # what OvSGTR's own ranking uses (graph_infer.py:103). Our detector-box path
            # already passes box_scores (DetBoxDataset weight_by_conf=True); omitting it
            # HERE ranked their predictions by predicate score alone, which handicapped
            # the baseline in exactly the comparison this file exists to make fair.
            bs = None
            if not args.no_conf_weight:
                bb0, bb1 = int(box_ptr[i]), int(box_ptr[i + 1])
                bs = np.asarray(d["box_scores"][bb0:bb1], dtype=np.float32)
            items.append((d["pairs"][a:b], prob, gt, bs))

        if not items:
            continue

        K = max(len(x[0]) for x in items)
        V = len(pred_names)
        B = len(items)
        logits = torch.full((B, K, V), -30.0)
        sub = torch.zeros(B, K, dtype=torch.long)
        obj = torch.zeros(B, K, dtype=torch.long)
        valid = torch.zeros(B, K, dtype=torch.bool)
        targets = []
        for j, (pairs, prob, gt, bs) in enumerate(items):
            k = len(pairs)
            pr = torch.from_numpy(np.clip(prob, EPS, 1 - EPS))
            logits[j,:k] = torch.log(pr / (1 - pr)) if mode == "sigmoid" else torch.log(pr)
            sub[j,:k] = torch.from_numpy(pairs[:, 0].astype(np.int64))
            obj[j,:k] = torch.from_numpy(pairs[:, 1].astype(np.int64))
            valid[j,:k] = True
            t = {"relations": torch.from_numpy(gt)}
            if bs is not None and len(bs):
                t["box_scores"] = torch.from_numpy(bs)
            targets.append(t)
            n_scored += 1

        ev.update({"logits": logits, "sub_idx": sub, "obj_idx": obj,
                   "valid_mask": valid, "pair_logits": None}, targets)

    metrics = ev.compute()
    if subsets:
        # micro + macro per subset, from the evaluator's cumulative tp/gt — the same
        # subset_breakdown our side reports, so the two rows are directly comparable
        from benchmark.eval_ovsgtr_novel import subset_breakdown  # noqa: E402
        metrics.update(subset_breakdown(ev, pred_names, subsets["novel"],
                                        subsets["base"], args.topk))
    result = {"metrics": {k: float(v) if not isinstance(v, dict) else v
                          for k, v in metrics.items()},
              "source": info, "pack": str(args.pack),
              "graph_constraint": args.graph_constraint,
              "n_scored": n_scored, "n_skipped_empty": n_skipped_empty,
              "n_skipped_no_gt": n_skipped_nogt}
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    print(f"scored={n_scored} skipped_empty={n_skipped_empty} skipped_no_gt={n_skipped_nogt}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
