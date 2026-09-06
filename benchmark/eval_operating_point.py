"""Pick the deployment threshold, and show what it buys — honestly.

THE MEASUREMENT PROBLEM, stated first because it decides the whole design.
Ordinary F1 needs precision, and we cannot measure precision on PSG / VG150 /
IndoorVG: those packs annotate a fraction of the true relations, so a
prediction absent from GT is "unannotated", not "wrong". Measured on Haystack's
explicitly adjudicated negatives, the same checkpoint scores AP 0.5602 where
the PSG convention scores 0.0171 — a 33x gap that is entirely annotation
. So annotated precision is a LOWER
BOUND and an F1 built on it is pessimistic by an unknown factor.

Recall does NOT have this problem: a relation someone bothered to annotate is
in the denominator whether or not we find it, so recall against incomplete GT
is still a valid recall.

So we report TWO precisions per threshold and never average them silently:

  precision_annotated   TP / emitted, treating unannotated as wrong.
                        A hard LOWER BOUND.
  precision_calibrated  the mean CALIBRATED SCORE of the emitted set. Because
                        relsgg/scoring.py's contract is Platt-calibrated
                        against Haystack's adjudicated cells, the score is an
                        estimate of P(a human calls this true) — so its mean
                        over the kept set estimates precision WITHOUT needing
                        the incomplete GT at all. Valid exactly insofar as the
                        calibration transfers; ECE is reported so the reader
                        can judge that, and the Haystack column is the control
                        where the two can be compared against ground truth.

F1 is then reported both ways. The honest operating point sits between them.

Also emitted: ROC + AUC, the PR curve, the full threshold sweep, and a
PREDICATE CONFUSION MATRIX. The last one is the informative part of "confusion
matrix" for this task — a binary TP/FP table hides the difference between
"found the right pair, called it `on` instead of `sitting on`" and "invented a
pair nobody annotated". Those are different failures with different fixes.

Runs in the DEPLOYMENT configuration (max_objects / budgets from
deploy/pipeline.py, calibration installed), so the threshold it picks is the
one the product would use.

    python benchmark/eval_operating_point.py --checkpoint CK \
        --data_root runs/packed/psg --split test --out runs/calib/op_psg.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data.dataset import RelationDataset, collate_fn  # noqa: E402
from relsgg.model.geometry import RelGeomEncoder  # noqa: E402
from relsgg.scoring import ScoreContract  # noqa: E402
from relsgg.text.student import encode_texts_student  # noqa: E402
from benchmark.eval_deploy_metrics import auc, pr_curve, reliability  # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402


@torch.no_grad()
def collect(model, loader, device, contract, amp=True, neg_by_row=None):
    """Per-emitted-triplet score / TP / predicted+GT predicate, graph-constrained.

    With `neg_by_row` (Haystack's adjudicated negatives) a prediction is SCORED
    only when its cell was actually labelled — TP if in GT, FP if explicitly
    adjudicated negative, and EXCLUDED otherwise. That makes precision a real
    precision instead of a lower bound, which is the entire point of using
    Haystack as the control.
    """
    model.eval()
    sc, tp, pc, gc, img, scored = [], [], [], [], [], []
    n_gt_total, n_img, gt_per_img = 0, 0, []
    gt_cls_count = defaultdict(int)
    row = -1
    for images, boxes, counts, targets in loader:
        images = images.to(device, non_blocking=True)
        boxes = boxes.to(device, non_blocking=True)
        counts = counts.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
            out = model(images, boxes, counts, targets=None)
        zp = out["logits"].float()
        za = (out["pair_logits"].float() if out.get("pair_logits") is not None
              else torch.zeros_like(zp[..., 0]))
        probs = contract.scores(zp, za)
        for b in range(zp.shape[0]):
            row += 1
            mask = out["valid_mask"][b]
            rels = targets[b].get("relations")
            if rels is None or len(rels) == 0 or int(mask.sum()) == 0:
                continue
            negs = neg_by_row.get(row, set()) if neg_by_row is not None else None
            rels = rels.cpu()
            # pair -> set of GT predicates (a pair may carry several)
            gt = defaultdict(set)
            for r in rels:
                gt[(int(r[0]), int(r[1]))].add(int(r[2]))
                gt_cls_count[int(r[2])] += 1
            n_gt = sum(len(v) for v in gt.values())
            n_gt_total += n_gt
            gt_per_img.append(n_gt)
            n_img += 1
            p = probs[b][mask]
            s, best = p.max(dim=-1)                     # graph constraint
            sub = out["sub_idx"][b][mask].cpu()
            obj = out["obj_idx"][b][mask].cpu()
            s, best = s.cpu(), best.cpu()
            for k in range(len(s)):
                key = (int(sub[k]), int(obj[k]))
                g = gt.get(key)
                sc.append(float(s[k]))
                pc.append(int(best[k]))
                img.append(n_img - 1)
                cell = (int(sub[k]), int(obj[k]), int(best[k]))
                if g is None:
                    tp.append(0); gc.append(-1)          # pair not annotated
                elif int(best[k]) in g:
                    tp.append(1); gc.append(int(best[k]))
                else:
                    tp.append(0); gc.append(sorted(g)[0])  # pair hit, pred miss
                # federated: scorable only if this exact cell was adjudicated
                scored.append(1 if negs is None
                              else int(tp[-1] == 1 or cell in negs))
    return (np.array(sc), np.array(tp), np.array(pc), np.array(gc),
            np.array(img), n_gt_total, n_img, np.array(gt_per_img),
            dict(gt_cls_count), np.array(scored, dtype=bool))


def sweep(sc, tp, total_gt, n_img, taus, scored=None):
    rows = []
    for t in taus:
        m = sc >= t
        n = int(m.sum())
        ms = m if scored is None else (m & scored)
        if n == 0:
            rows.append({"tau": float(t), "n": 0, "per_img": 0.0,
                         "recall": 0.0, "prec_ann": 0.0, "prec_cal": 0.0,
                         "f1_ann": 0.0, "f1_cal": 0.0})
            continue
        rec = float(tp[m].sum() / max(total_gt, 1))
        if int(ms.sum()) == 0:
            continue
        pa = float(tp[ms].mean())
        pcal = float(sc[ms].mean())
        h = lambda p: 0.0 if (p + rec) <= 0 else 2 * p * rec / (p + rec)
        rows.append({"tau": float(t), "n": n, "per_img": n / max(n_img, 1),
                     "recall": rec, "prec_ann": pa, "prec_cal": pcal,
                     "f1_ann": h(pa), "f1_cal": h(pcal)})
    return rows


def roc(sc, tp, n=200):
    pos, neg = int(tp.sum()), int((1 - tp).sum())
    if pos == 0 or neg == 0:
        return []
    qs = np.unique(np.quantile(sc, np.linspace(0, 1, n)))
    out = []
    for t in qs[::-1]:
        m = sc >= t
        out.append({"tau": float(t),
                    "tpr": float(tp[m].sum() / pos),
                    "fpr": float((1 - tp[m]).sum() / neg)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--max_objects", type=int, default=32)
    p.add_argument("--geo_budget", type=int, default=992)
    p.add_argument("--final_budget", type=int, default=992)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--negatives", default="",
                   help="Haystack adjudicated-negative sidecar. Turns "
                        "precision_annotated into a REAL precision by scoring "
                        "only cells a human actually labelled — the control "
                        "for how far the lower bound sits from the truth.")
    p.add_argument("--n_confusion", type=int, default=18)
    p.add_argument("--tau", type=float, default=None,
                   help="compute the confusion matrix at THIS threshold "
                        "instead of the max-F1(calibrated) one. The F1 "
                        "criteria disagree by design here (calibrated F1 "
                        "prefers recall and lands at ~70 preds/img, which is "
                        "not a product operating point), so the shipped tau "
                        "is a judgement call and the confusion table should "
                        "describe the tau we actually ship.")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ck, "ema").to(dev).eval()
    # DEPLOYMENT configuration, not the eval defaults.
    model.sampler.geo_budget = a.geo_budget
    model.sampler.final_budget = a.final_budget
    contract = ScoreContract.for_checkpoint(a.checkpoint, required=True)
    print(f"contract: {contract.describe()}")
    print(f"deployment shapes: max_objects={a.max_objects} "
          f"geo_budget={a.geo_budget} final_budget={a.final_budget}")

    ds = RelationDataset(root=a.data_root, split=a.split, resolution=a.img_size,
                         max_objects=a.max_objects)
    names = list(ds.predicate_names)
    ck_args = ck.get("args") or {}
    ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    E = encode_texts_student(names, ck_args["text_student"],
                             templates=TRAIN_TEMPLATES, device=dev)
    model.vocab_head.set_vocabulary_matrix(names, E)
    model.reparameterize()
    neg_by_row = None
    if a.negatives:
        from pathlib import Path
        side = json.load(open(a.negatives))
        pid = {n: i for i, n in enumerate(names)}
        row_of = {int(Path(f).stem.split("_")[-1]): i
                  for i, f in enumerate(ds.file_names)}
        neg_by_row, n_neg = {}, 0
        for img_id, cells in side["by_image_id"].items():
            r = row_of.get(int(img_id))
            if r is None:
                continue
            keep = {(int(s_), int(o_), pid[q]) for s_, o_, q in cells if q in pid}
            if keep:
                neg_by_row[r] = keep
                n_neg += len(keep)
        print(f"FEDERATED: {n_neg:,} adjudicated negatives over "
              f"{len(neg_by_row):,} images — precision below is a REAL "
              f"precision, not a lower bound")

    if a.limit:
        ds = torch.utils.data.Subset(ds, list(range(min(a.limit, len(ds)))))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)

    sc, tp, pc, gc, img, total_gt, n_img, gt_pi, gt_cls, scored = collect(
        model, loader, dev, contract, neg_by_row=neg_by_row)
    print(f"\nimages {n_img:,}  emitted {len(sc):,}  GT {total_gt:,}  "
          f"GT/img {gt_pi.mean():.2f}")

    taus = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 101), np.quantile(sc, np.linspace(0, 1, 200))]))
    rows = sweep(sc, tp, total_gt, n_img, taus,
                 scored=scored if neg_by_row is not None else None)
    A = auc(sc, tp)
    _, _, ap, _ = pr_curve(sc, tp, total_gt)
    _, ece = reliability(sc, tp)

    best_cal = max(rows, key=lambda r: r["f1_cal"])
    best_ann = max(rows, key=lambda r: r["f1_ann"])
    # count-matched: emit as many as there are GT relations per image
    tgt = gt_pi.mean()
    best_cnt = min(rows, key=lambda r: abs(r["per_img"] - tgt))
    print(f"\nAUC {A:.4f}   AP {ap:.4f}   ECE {ece:.4f}")
    print(f"  {'criterion':26s} {'tau':>6s} {'preds/img':>10s} {'recall':>8s}"
          f" {'P_ann':>7s} {'P_cal':>7s} {'F1_ann':>7s} {'F1_cal':>7s}")
    for tag, r in (("max F1 (calibrated P)", best_cal),
                   ("max F1 (annotated P)", best_ann),
                   (f"count-matched ({tgt:.1f}/img)", best_cnt)):
        print(f"  {tag:26s} {r['tau']:6.3f} {r['per_img']:10.2f} "
              f"{r['recall']:8.4f} {r['prec_ann']:7.4f} {r['prec_cal']:7.4f} "
              f"{r['f1_ann']:7.4f} {r['f1_cal']:7.4f}")

    # ---- predicate confusion at the chosen tau --------------------------
    tau = a.tau if a.tau is not None else best_cal["tau"]
    keep = sc >= tau
    top = [c for c, _ in sorted(gt_cls.items(), key=lambda kv: -kv[1])][:a.n_confusion]
    idx = {c: i for i, c in enumerate(top)}
    M = np.zeros((len(top) + 1, len(top) + 1), dtype=np.int64)  # +1 = "other"
    pair_hit_pred_miss = 0
    for s_, t_, p_, g_ in zip(sc[keep], tp[keep], pc[keep], gc[keep]):
        if g_ < 0:
            continue                       # pair not annotated: not confusion
        r = idx.get(int(g_), len(top))
        c = idx.get(int(p_), len(top))
        M[r, c] += 1
        if not t_:
            pair_hit_pred_miss += 1
    n_emit = int(keep.sum())
    n_tp = int(tp[keep].sum())
    n_unann = n_emit - int((gc[keep] >= 0).sum())
    print(f"\nat tau={tau:.3f}: emitted {n_emit:,}  "
          f"TP {n_tp:,}  pair-hit/predicate-miss {pair_hit_pred_miss:,}  "
          f"pair-not-annotated {n_unann:,}  FN {total_gt - n_tp:,}")

    json.dump({
        "dataset": os.path.basename(a.data_root), "split": a.split,
        "federated": neg_by_row is not None,
        "contract": {"a": contract.calib_a, "b": contract.calib_b},
        "n_images": n_img, "n_emitted": len(sc), "total_gt": total_gt,
        "gt_per_image": float(gt_pi.mean()),
        "auc": A, "ap": ap, "ece": ece,
        "sweep": rows, "roc": roc(sc, tp),
        "best": {"f1_calibrated": best_cal, "f1_annotated": best_ann,
                 "count_matched": best_cnt},
        "confusion": {"labels": [names[c] for c in top] + ["other"],
                      "matrix": M.tolist(), "tau": tau},
        "counts_at_tau": {"emitted": n_emit, "tp": n_tp,
                          "pair_hit_predicate_miss": pair_hit_pred_miss,
                          "pair_not_annotated": n_unann,
                          "fn": total_gt - n_tp},
    }, open(a.out, "w"), indent=2)
    print(f"saved → {a.out}")


if __name__ == "__main__":
    main()
