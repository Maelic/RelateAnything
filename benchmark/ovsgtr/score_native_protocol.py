"""score_native_protocol.py — score an interchange file with OvSGTR's OWN recall
code, transcribed from their source, as a REPRODUCTION GATE.

Why this file exists
--------------------
`eval_interchange.py` scores OvSGTR's predictions with OUR evaluator, which is the
right thing for a head-to-head (one protocol, both models). But it scored their
released checkpoint ~28% below their published number, and until that gap is
explained every downstream row is suspect. This script isolates the cause by
reimplementing `datasets/sgg_metrics.py::SGRecall` / `OvrSGZeroShotRecall` and
`_compute_pred_matches` line-for-line, then walking from their protocol to ours
one change at a time.

The two protocols differ in TWO independent ways, and this measures each:

  A  ovsgtr: triplet-EQUALITY match (sub_class, predicate, obj_class must all
                match GT exactly) + IoU>=0.5 on both endpoint boxes. Crucially
                MANY-TO-MANY: a GT object covered by k duplicate detections gives
                the model k independent chances to score the hit.
  B  ovsgtr_ca: same, but class-agnostic (drop the object-class equality), which
                is what our evaluator does. Isolates the cost of class matching.
  C  relsgg: greedy ONE-TO-ONE IoU assignment of GT boxes to predicted boxes,
                then box-INDEX equality. This is our protocol. Isolates the cost
                of forbidding duplicate credit.

Ranking is identical in all three (the stored order, which is graph_infer's
pred * conf(sub) * conf(obj) sort), so ranking cannot explain any difference.

Recall aggregation follows theirs exactly: per-image recall, then a plain mean
over images (image-MACRO), with images having no GT relation skipped; and the
novel subset averaged only over images that contain >=1 novel GT relation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark.eval_detboxes import pairwise_iou, greedy_match  # noqa: E402
from benchmark.eval_ovsgtr_novel import VG150_NOVEL_PREDICATE  # noqa: E402


def per_image_pred_to_gt(gt_trip, gt_boxes8, pred_trip, pred_boxes8, iou_thr,
                         mode="sgdet"):
    """Transcription of sgg_metrics._compute_pred_matches.

    Returns a list (len = n_pred) of lists of GT indices matched by that
    prediction. Note there is NO one-to-one constraint anywhere: every
    prediction whose triplet string matches and whose two boxes clear IoU is
    credited, so duplicate detections of the same object each get a chance.
    """
    n_pred = len(pred_trip)
    pred_to_gt = [[] for _ in range(n_pred)]
    if n_pred == 0 or len(gt_trip) == 0:
        return pred_to_gt
    keeps = (gt_trip[:, None,:] == pred_trip[None,:,:]).all(-1)
    gt_has_match = keeps.any(1)
    for gt_ind in np.where(gt_has_match)[0]:
        keep_inds = keeps[gt_ind]
        boxes = pred_boxes8[keep_inds]
        gt_box = gt_boxes8[gt_ind]
        if mode == "phrdet":
            gu = gt_box.reshape(2, 4)
            gu = np.concatenate((gu.min(0)[:2], gu.max(0)[2:]), 0)
            bu = boxes.reshape(-1, 2, 4)
            bu = np.concatenate((bu.min(1)[:,:2], bu.max(1)[:, 2:]), 1)
            inds = pairwise_iou(gu[None], bu)[0] >= iou_thr
        else:
            sub_iou = pairwise_iou(gt_box[None,:4], boxes[:,:4])[0]
            obj_iou = pairwise_iou(gt_box[None, 4:], boxes[:, 4:])[0]
            inds = (sub_iou >= iou_thr) & (obj_iou >= iou_thr)
        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


def recall_from_pred_to_gt(pred_to_gt, n_gt, topk):
    """Their SGRecall inner loop: union of GT hit by the top-k predictions."""
    out = {}
    for k in topk:
        match = set()
        for lst in pred_to_gt[:k]:
            match.update(lst)
        out[k] = (match, len(match) / float(n_gt))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--topk", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--per_class", action="store_true",
                   help="also record per-predicate tp@k and GT support. The aggregate "
                        "Novel column cannot say WHETHER a deficit is one predicate or "
                        "fifteen -- VG150's novel subset is ~93%% {on, of, in} by "
                        "instance count, so a single predicate can carry the whole "
                        "number and a per-class split is the only way to see it.")
    p.add_argument("--protocols", nargs="+",
                   default=["ovsgtr", "ovsgtr_1to1", "ovsgtr_ca", "relsgg"],
                   choices=["ovsgtr", "ovsgtr_ca", "ovsgtr_phrdet",
                            "relsgg", "relsgg_cls", "ovsgtr_1to1"])
    args = p.parse_args()

    _npz = np.load(args.pred, allow_pickle=False)
    d = {k: _npz[k] for k in _npz.files}   # NpzFile is lazy; materialise once
    info = json.loads(str(d["meta"][0]))
    pred_names = [str(x) for x in d["predicates"]]
    label_base = int(info.get("label_base", 1))

    pack = Path(args.pack)
    meta = json.loads((pack / "meta.json").read_text())
    if list(meta["predicates"]) != pred_names:
        raise SystemExit("predicate vocabulary mismatch; refusing to score")
    img_meta = np.load(pack / "img_meta.npy")
    rels_all = np.load(pack / "rels.npy")
    boxes_all = np.load(pack / "boxes.npy", mmap_mode="r")
    cats_all = np.load(pack / "box_cats.npy")

    novel = {pred_names.index(n) for n in VG150_NOVEL_PREDICATE if n in pred_names}
    print(f"novel predicates: {len(novel)}/{len(pred_names)}")

    pair_ptr, box_ptr, idxs = d["pair_ptr"], d["box_ptr"], d["image_index"]
    n = len(idxs) if args.limit <= 0 else min(len(idxs), args.limit)
    K = max(args.topk)

    acc = {pr: {"R": {k: [] for k in args.topk}, "N": {k: [] for k in args.topk},
                "tp": {k: 0 for k in args.topk}, "ntp": {k: 0 for k in args.topk}}
           for pr in args.protocols}
    tot_gt = tot_ngt = 0
    # per-predicate: {protocol: {k: {pred_id: tp}}} and {pred_id: gt_support}
    cls_tp = {pr: {k: {} for k in args.topk} for pr in args.protocols}
    cls_gt = {}
    dup_num = dup_den = 0      # predicted boxes per RECOVERED GT box
    ceil_num = ceil_den = 0    # GT relations with BOTH endpoints detected
    n_img = n_skip = 0

    for i in range(n):
        a, b = int(pair_ptr[i]), int(pair_ptr[i + 1])
        if b <= a:
            n_skip += 1
            continue
        _iid, W, H, b0, nb, r0, nr = (int(x) for x in img_meta[int(idxs[i])])
        gt = np.asarray(rels_all[r0:r0 + nr,:3], dtype=np.int64)
        gt = gt[(gt[:, 0] < nb) & (gt[:, 1] < nb)]
        if gt.size == 0:
            n_skip += 1
            continue

        # --- GT side: pixel xyxy boxes + category per box
        g = np.asarray(boxes_all[b0:b0 + nb], dtype=np.float32)
        gxyxy = np.stack([(g[:, 0] - g[:, 2] / 2) * W, (g[:, 1] - g[:, 3] / 2) * H,
                          (g[:, 0] + g[:, 2] / 2) * W, (g[:, 1] + g[:, 3] / 2) * H], 1)
        gcls = cats_all[b0:b0 + nb].astype(np.int64)
        gt_boxes8 = np.concatenate([gxyxy[gt[:, 0]], gxyxy[gt[:, 1]]], axis=1)

        # --- prediction side: their graph-constraint argmax over non-bg columns,
        #     in the stored (conf-weighted) order, truncated to the largest K.
        bA, bB = int(box_ptr[i]), int(box_ptr[i + 1])
        pboxes = d["boxes"][bA:bB]
        pcls = d["labels"][bA:bB].astype(np.int64) - label_base
        e = min(b, a + K)
        pairs = d["pairs"][a:e].astype(np.int64)
        scores = d["rel_scores"][a:e].astype(np.float32)[:, 1:]   # drop bg col 0
        pp = scores.argmax(1)                                     # pack predicate index
        pred_boxes8 = np.concatenate([pboxes[pairs[:, 0]], pboxes[pairs[:, 1]]], axis=1)

        if args.per_class:
            for r in gt:
                cls_gt[int(r[2])] = cls_gt.get(int(r[2]), 0) + 1
        novel_idx = [j for j, r in enumerate(gt) if int(r[2]) in novel]
        n_gt = len(gt)
        tot_gt += n_gt
        tot_ngt += len(novel_idx)

        # How many predicted boxes clear IoU 0.5 against each GT box that was found at
        # all? This is the mechanism behind the duplicate-credit cell: a GT object
        # covered by d detections gives the model d independent chances.
        if len(pboxes):
            iou_gd = pairwise_iou(gxyxy, pboxes)
            cover = (iou_gd >= args.iou_thr).sum(1)
            found = cover > 0
            dup_num += int(cover[found].sum())
            dup_den += int(found.sum())
            # Pair-recall CEILING on exactly these images: a GT relation is
            # recoverable only if BOTH endpoints were detected. Computed here so
            # an ablation that changes the boxes reports its own ceiling on its
            # own image subset, with no separate tool to keep in sync.
            ceil_num += int((found[gt[:, 0]] & found[gt[:, 1]]).sum())
        ceil_den += len(gt)

        for pr in args.protocols:
            if pr == "ovsgtr_1to1":
                # THEIR protocol plus an assignment constraint, and nothing else.
                # The assignment is CLASS-AWARE (only same-class detections are
                # eligible) and picks the highest-IoU eligible detection, so the GT
                # object is represented by the best box it could possibly be
                # represented by. Any drop from `ovsgtr` is therefore attributable to
                # duplicate credit alone, not to an unlucky greedy choice.
                if len(pboxes):
                    iou_gd = pairwise_iou(gxyxy, pboxes)
                    iou_gd = np.where(gcls[:, None] == pcls[None,:], iou_gd, 0.0)
                    m = greedy_match(iou_gd, args.iou_thr)
                else:
                    m = np.full(nb, -1, np.int64)
                SENT = 1 << 16
                gs = np.where(m[gt[:, 0]] >= 0, m[gt[:, 0]], SENT)
                go = np.where(m[gt[:, 1]] >= 0, m[gt[:, 1]], SENT)
                gt_set = {}
                for j, (s_, o_, p_) in enumerate(zip(gs, go, gt[:, 2])):
                    gt_set.setdefault((int(s_), int(o_), int(p_)), []).append(j)
                ptg = [gt_set.get((int(pairs[j, 0]), int(pairs[j, 1]), int(pp[j])), [])
                       for j in range(len(pairs))]
            elif pr in ("relsgg", "relsgg_cls"):
                # one-to-one: greedy GT->pred box assignment, then index equality.
                # `relsgg_cls` additionally demands the object classes match, which is
                # the ONLY difference from `ovsgtr` -- together the four cells form a
                # clean 2x2 over {assignment} x {class equality}.
                m = (greedy_match(pairwise_iou(gxyxy, pboxes), args.iou_thr)
                     if len(pboxes) else np.full(nb, -1, np.int64))
                SENT = 1 << 16
                ok = np.ones(len(gt), bool)
                if pr == "relsgg_cls" and len(pboxes):
                    ok = np.zeros(len(gt), bool)
                    both = (m[gt[:, 0]] >= 0) & (m[gt[:, 1]] >= 0)
                    ok[both] = ((pcls[m[gt[both, 0]]] == gcls[gt[both, 0]]) &
                                (pcls[m[gt[both, 1]]] == gcls[gt[both, 1]]))
                gs = np.where((m[gt[:, 0]] >= 0) & ok, m[gt[:, 0]], SENT)
                go = np.where((m[gt[:, 1]] >= 0) & ok, m[gt[:, 1]], SENT)
                gt_set = {}
                for j, (s_, o_, p_) in enumerate(zip(gs, go, gt[:, 2])):
                    gt_set.setdefault((int(s_), int(o_), int(p_)), []).append(j)
                ptg = [gt_set.get((int(pairs[j, 0]), int(pairs[j, 1]), int(pp[j])), [])
                       for j in range(len(pairs))]
            else:
                if pr == "ovsgtr_ca":
                    gt_trip = gt[:, 2:3]
                    pred_trip = pp[:, None]
                else:
                    gt_trip = np.stack([gcls[gt[:, 0]], gt[:, 2], gcls[gt[:, 1]]], 1)
                    pred_trip = np.stack([pcls[pairs[:, 0]], pp, pcls[pairs[:, 1]]], 1)
                ptg = per_image_pred_to_gt(
                    gt_trip, gt_boxes8, pred_trip, pred_boxes8, args.iou_thr,
                    mode="phrdet" if pr == "ovsgtr_phrdet" else "sgdet")

            res = recall_from_pred_to_gt(ptg, n_gt, args.topk)
            for k in args.topk:
                match, rec = res[k]
                acc[pr]["R"][k].append(rec)
                acc[pr]["tp"][k] += len(match)
                if args.per_class:
                    for j in match:
                        cls_tp[pr][k][int(gt[j][2])] = \
                            cls_tp[pr][k].get(int(gt[j][2]), 0) + 1
                hit = len(set(novel_idx) & match)
                acc[pr]["ntp"][k] += hit
                if novel_idx:
                    acc[pr]["N"][k].append(hit / float(len(novel_idx)))
        n_img += 1
        if n_img % 5000 == 0:
            print(f"  {n_img}/{n}", flush=True)

    print(f"\nscored {n_img} images, skipped {n_skip}")
    dup = dup_num / max(dup_den, 1)
    ceiling = ceil_num / max(ceil_den, 1)
    print(f"pair-recall ceiling on these images: {100*ceiling:.2f}% "
          f"({ceil_num:,}/{ceil_den:,} GT relations have both endpoints detected)")
    print(f"duplicate-detection factor: {dup:.2f} predicted boxes per RECOVERED GT box "
          f"(n={dup_den}); under the standard matcher each is an independent chance\n")
    hdr = f"{'protocol':<16}" + "".join(f"{'R@'+str(k):>9}" for k in args.topk) \
        + "  |" + "".join(f"{'nR@'+str(k):>9}" for k in args.topk)
    print(hdr)
    print("-" * len(hdr))
    out = {}
    for pr in args.protocols:
        row = {}
        line = f"{pr:<16}"
        for k in args.topk:
            v = 100 * float(np.mean(acc[pr]["R"][k]))
            row[f"R@{k}"] = v
            row[f"microR@{k}"] = 100 * acc[pr]["tp"][k] / max(tot_gt, 1)
            row[f"novel_microR@{k}"] = 100 * acc[pr]["ntp"][k] / max(tot_ngt, 1)
            line += f"{v:9.2f}"
        line += "  |"
        for k in args.topk:
            v = 100 * float(np.mean(acc[pr]["N"][k]))
            row[f"novel_R@{k}"] = v
            line += f"{v:9.2f}"
        print(line)
        # MEAN RECALL (SGG convention, same definition as eval_ovsgtr_novel.py:
        # mR@K = mean_c tp_c / gt_c over classes with GT support). Reported because
        # the micro column on this benchmark is close to a measurement of ONE
        # predicate: `on` alone is 62.3% of the novel GT by instance count, so a model
        # that transfers to `on` and nothing else wins Novel R@K outright. mR gives
        # every predicate the same vote, which is the whole reason SGG adopted it.
        if args.per_class:
            for k in args.topk:
                tpk = cls_tp[pr][k]
                for label, ids in (("", None), ("novel_", novel)):
                    cls = [i for i, g in cls_gt.items()
                           if g > 0 and (ids is None or i in ids)]
                    if not cls:
                        continue
                    row[f"{label}mR@{k}"] = 100 * sum(
                        tpk.get(i, 0) / cls_gt[i] for i in cls) / len(cls)
                    row[f"{label}n_classes@{k}"] = len(cls)
                    row[f"{label}nonzero_classes@{k}"] = sum(
                        tpk.get(i, 0) > 0 for i in cls)
        out[pr] = row

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"protocols": out, "source": info, "pack": str(args.pack),
             "n_images": n_img, "iou_thr": args.iou_thr,
             "duplicate_detection_factor": dup,
             "pair_recall_ceiling": ceiling,
             "n_gt_relations": tot_gt, "n_novel_gt_relations": tot_ngt,
             **({"per_class": {
                 "gt_support": {pred_names[i]: c for i, c in sorted(cls_gt.items())},
                 "novel": sorted(pred_names[i] for i in novel),
                 "tp": {pr: {str(k): {pred_names[i]: c
                                      for i, c in sorted(v.items())}
                             for k, v in d.items()}
                        for pr, d in cls_tp.items()}}}
                if args.per_class else {})},
            indent=2, sort_keys=True))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
