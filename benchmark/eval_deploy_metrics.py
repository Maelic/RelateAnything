"""Deployment metrics: adaptive-K recall (R@n) and confidence reliability.

WHY THIS EXISTS, and why it is NOT another R@K script.

R@20/50/100 answers "if I am allowed 100 guesses, how many did I get?". Nothing
anyone deploys works that way. An SME frame has 2-15 real relations and the
product emits a handful; what matters is whether the FIRST few are right and
whether the number next to them means anything.

Two metrics, both deployment-shaped:

R@n  — adaptive-K recall. For an image with n GT relations, score the top n
       predictions. Because K equals the GT count, precision@n == recall@n
       exactly, so this single number is the model's accuracy at the operating
       point where it emits exactly as much as it should. No K to tune, no
       credit for a long tail of guesses. mR@n is the per-class macro version,
       which is what the rare predicates live or die on.
       R@2n / R@3n are reported alongside to show the SHAPE: a model whose
       R@2n is far above its R@n is right but badly ranked, which is a
       different (and easier) problem than being wrong.

Confidence reliability — the deployment question the user actually asked:
       "the top preds need to be correct and confidence should NOT be a
       confounder". Three things are measured, and they fail differently:
         1. CALIBRATION  — does score s mean precision s? (reliability table,
            ECE). A model can rank perfectly and still be uncalibrated, which
            makes any fixed product threshold arbitrary.
         2. DISCRIMINATION — does score separate TP from FP at all? (AUC,
            mean score TP vs FP). This is what ranking quality really is.
         3. THE CONFOUNDER — is a score of 0.6 worth the same on a head
            predicate as on a rare one? Calibration is reported PER FREQUENCY
            BUCKET for exactly this. If head and rare curves diverge, then a
            single global threshold silently trades away the tail, and the
            confidence number is reporting predicate frequency rather than
            correctness. That is the failure mode to catch before shipping.

Scoring uses relsgg.scoring.ScoreContract — the SAME object that
relsgg/evaluator.py, deploy/pipeline.py and the ONNX host import, so this is
not a transcription of the product's formula but literally the product's
formula. Graph-constrained (one predicate per pair). Pass --use_calibration to
score with the checkpoint's calibration.json: monotone, so every ranking
number is unchanged and only the threshold rows change meaning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data.dataset import RelationDataset, collate_fn  # noqa: E402
from relsgg.model.geometry import RelGeomEncoder  # noqa: E402
from relsgg.text.student import encode_texts_student  # noqa: E402
from relsgg.scoring import ScoreContract  # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES
from relsgg.checkpoint import build_model_from_ckpt


# Frequency buckets by GT instance count in the EVALUATED split, so the
# split's own tail defines "rare" rather than an imported constant.
def frequency_buckets(gt_counts: Dict[int, int]) -> Dict[int, str]:
    out = {}
    for c, n in gt_counts.items():
        out[c] = "freq" if n >= 100 else ("common" if n >= 10 else "rare")
    return out


@torch.no_grad()
def collect(model, loader, device, eval_budget: int, amp: bool = True,
            depth: int = 100, neg_by_index=None, contract=None):
    """Run the model and return (per-image records, flat prediction table).

    Every emitted triplet is kept with its score and TP flag — calibration is
    a property of the whole emission set, not of the top-K slice.

    `neg_by_index` maps dataset row -> set of EXPLICITLY NEGATIVE (s, o, p)
    cells (Haystack only). With it we can separate two very different claims
    about a false positive: "the model is wrong" and "nobody annotated it".
    Every emitted prediction is then tagged `lab`: 2 = labelled positive,
    1 = labelled negative, 0 = unlabelled (unknowable). We ALSO score every
    labelled cell directly, sampled or not, which is an unbiased sample of
    the labelled space rather than the emission-biased slice.
    """
    from relsgg.scoring import ScoreContract
    contract = contract or ScoreContract()
    model.eval()
    raw = model.module if hasattr(model, "module") else model
    orig = raw.sampler.final_budget
    raw.sampler.final_budget = min(eval_budget, raw.sampler.geo_budget)

    images_rec: List[dict] = []
    pred_score: List[float] = []
    pred_tp: List[int] = []
    pred_cls: List[int] = []
    pred_rank: List[int] = []
    pred_logit: List[float] = []
    pred_zpred: List[float] = []
    pred_zpair: List[float] = []
    pred_lab: List[int] = []
    pred_img: List[int] = []
    pred_ngt: List[int] = []
    cell_score: List[float] = []
    cell_tp: List[int] = []
    cell_cls: List[int] = []
    cell_seen: List[int] = []
    cell_zpred: List[float] = []
    cell_zpair: List[float] = []
    gt_counts: Dict[int, int] = defaultdict(int)
    row = -1

    try:
        for images, boxes, box_counts, targets in loader:
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            box_counts = box_counts.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp, dtype=torch.bfloat16):
                out = model(images, boxes, box_counts, targets=None)

            zpred = out["logits"].float()
            zpair = (out["pair_logits"].float()
                     if out.get("pair_logits") is not None
                     else torch.zeros_like(zpred[..., 0]))
            # THE shared contract, same object the evaluator and the deployed
            # host use (relsgg/scoring.py).
            lg = contract.fuse(zpred, zpair)
            probs_all = torch.sigmoid(lg)

            for b in range(lg.shape[0]):
                row += 1
                mask = out["valid_mask"][b]
                rels = targets[b].get("relations")
                if int(mask.sum()) == 0:
                    continue
                rels = rels.cpu() if rels is not None else torch.zeros(0, 3)
                gt_set = {(int(r[0]), int(r[1]), int(r[2])) for r in rels}
                n_gt = len(gt_set)
                # An image with no GT still carries information in federated
                # mode: Haystack's negatives live there too, and a prediction
                # on an explicitly-negative cell is a REAL false positive.
                # Outside federated mode it is unscorable, so skip it.
                has_neg = neg_by_index is not None and row in neg_by_index
                if n_gt == 0 and not has_neg:
                    continue
                for r in rels:
                    gt_counts[int(r[2])] += 1

                probs = probs_all[b][mask]                       # [k_valid, V]
                zs = lg[b][mask]                                 # pre-sigmoid
                sub_all = out["sub_idx"][b][mask].cpu()
                obj_all = out["obj_idx"][b][mask].cpu()
                # graph constraint: one predicate per pair
                best_s, best_p = probs.max(dim=-1)
                n_top = min(depth, best_s.numel())
                srt, pid = best_s.topk(n_top)
                sub = sub_all[pid.cpu()]
                obj = obj_all[pid.cpu()]
                prd = best_p[pid].cpu()
                srt = srt.cpu()

                neg_set = neg_by_index.get(row, set()) if neg_by_index else None
                tp = np.zeros(n_top, dtype=bool)
                lab = np.zeros(n_top, dtype=np.int8)
                for i in range(n_top):
                    key = (int(sub[i]), int(obj[i]), int(prd[i]))
                    if key in gt_set:
                        tp[i] = True
                        lab[i] = 2
                    elif neg_set is not None and key in neg_set:
                        lab[i] = 1

                # ---- unbiased cell-level pass over the labelled space ----
                if neg_set is not None:
                    pair_row = {(int(s), int(o)): i for i, (s, o)
                                in enumerate(zip(sub_all.tolist(),
                                                 obj_all.tolist()))}
                    pcpu = probs.cpu().numpy()
                    zp_cpu = zpred[b][mask].float().cpu().numpy()
                    za_cpu = zpair[b][mask].float().cpu().numpy()
                    for (s, o, q), y in ([(k, 1) for k in gt_set]
                                         + [(k, 0) for k in neg_set]):
                        i = pair_row.get((s, o))
                        # unsampled pair == what the deployed system emits for
                        # it: nothing. Score 0.0, same as relsgg/haystack_eval.
                        cell_score.append(float(pcpu[i, q]) if i is not None
                                          else 0.0)
                        cell_tp.append(y)
                        cell_cls.append(q)
                        cell_seen.append(int(i is not None))
                        # Keep the two terms apart on the LABELLED space too.
                        # On PSG/VG150 the relatedness term out-ranks the full
                        # score, but those labels reward predicting what an
                        # annotator wrote down. Only explicit negatives can say
                        # whether z_pair tracks TRUTH or ANNOTATION PROPENSITY,
                        # and that decides whether re-weighting toward it is a
                        # product win or benchmark-fitting.
                        cell_zpred.append(float(zp_cpu[i, q]) if i is not None
                                          else -30.0)
                        cell_zpair.append(float(za_cpu[i]) if i is not None
                                          else -30.0)

                per_cls_total: Dict[int, int] = defaultdict(int)
                for r in rels:
                    per_cls_total[int(r[2])] += 1
                if n_gt:                       # R@n needs a GT count to divide by
                    images_rec.append({
                        "n_gt": n_gt, "tp": tp, "cls": prd.numpy(),
                        "score": srt.numpy(),
                        "per_cls_total": dict(per_cls_total),
                    })
                pred_score.extend(srt.tolist())
                pred_tp.extend(tp.astype(int).tolist())
                pred_cls.extend(prd.tolist())
                pred_rank.extend(range(n_top))
                pred_lab.extend(lab.tolist())
                pred_img.extend([row] * n_top)
                pred_ngt.extend([n_gt] * n_top)
                # keep the PRE-sigmoid logits: calibration is fitted there, and
                # the sigmoid has already destroyed the resolution. Keep the
                # predicate and relatedness terms APART — they are different
                # signals and it matters which one carries discrimination.
                pred_logit.extend(
                    zs.gather(1, best_p.unsqueeze(1)).squeeze(1)[pid]
.float().cpu().tolist())
                pred_zpred.extend(
                    zpred[b][mask].gather(1, best_p.unsqueeze(1)).squeeze(1)[pid]
.float().cpu().tolist())
                pred_zpair.extend(zpair[b][mask][pid].float().cpu().tolist())
    finally:
        raw.sampler.final_budget = orig

    flat = {"score": np.array(pred_score), "tp": np.array(pred_tp),
            "cls": np.array(pred_cls), "rank": np.array(pred_rank),
            "logit": np.array(pred_logit), "lab": np.array(pred_lab),
            "z_pred": np.array(pred_zpred), "z_pair": np.array(pred_zpair),
            "img": np.array(pred_img), "n_gt": np.array(pred_ngt)}
    cells = {"score": np.array(cell_score), "tp": np.array(cell_tp),
             "cls": np.array(cell_cls), "seen": np.array(cell_seen),
             "z_pred": np.array(cell_zpred), "z_pair": np.array(cell_zpair)}
    return images_rec, flat, dict(gt_counts), cells


def pr_curve(score, tp, total_gt):
    """Precision-recall by descending score + average precision.

    THE POINT: a PR curve is INVARIANT to any monotonic rescaling of the
    score, so temperature scaling cannot move it by even one point. It is
    therefore the honest answer to "what operating points exist at all?",
    separate from "can a human address them with a threshold?".
    """
    order = np.argsort(-score, kind="mergesort")
    t = tp[order].astype(np.float64)
    cum = np.cumsum(t)
    prec = cum / np.arange(1, len(t) + 1)
    rec = cum / max(total_gt, 1)
    ap = float(np.sum((rec - np.concatenate(([0.0], rec[:-1]))) * prec))
    return prec, rec, ap, score[order]


def precision_targets(prec, rec, thr, targets=(0.9, 0.8, 0.7, 0.6, 0.5,
                                               0.4, 0.3, 0.2, 0.1)):
    """Best recall reachable at each precision floor (monotone envelope)."""
    # envelope: max precision achievable at or beyond each rank
    env = np.maximum.accumulate(prec[::-1])[::-1]
    rows = []
    for p in targets:
        idx = np.where(env >= p)[0]
        rows.append((p, float(rec[idx[-1]]), float(thr[idx[-1]]),
                     int(idx[-1]) + 1) if len(idx) else (p, 0.0, float("nan"), 0))
    return rows


def fit_temperature(logit, tp, lo=0.05, hi=50.0, iters=60):
    """Temperature T minimising NLL of sigmoid(z/T). Monotone => ranking-safe."""
    z = logit.astype(np.float64)
    y = tp.astype(np.float64)

    def nll(T):
        p = 1.0 / (1.0 + np.exp(-np.clip(z / T, -60, 60)))
        p = np.clip(p, 1e-12, 1 - 1e-12)
        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())

    # golden-section on a unimodal 1-D objective
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(iters):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    return float((a + b) / 2)


def fit_platt(logit, tp, iters: int = 300):
    """Platt scaling: fit sigmoid(a*z + b) by NLL.

    WHY NOT TEMPERATURE ALONE. sigmoid(z/T) -> 0.5 as T grows, so a
    temperature can never express a base rate far from 50%. Our positives are
    ~3.8% of emissions, so the temperature fit runs to its bound and parks
    every score near 0.5 — it removes the over-confidence but replaces it with
    a useless constant. The bias term b is what carries the base rate; a is
    what carries the resolution. Still strictly monotone in z (for a > 0), so
    every ranking metric is untouched.
    """
    z = torch.tensor(logit, dtype=torch.float64)
    y = torch.tensor(tp, dtype=torch.float64)
    # start from the identity in z and the empirical log-odds
    p0 = float(np.clip(tp.mean(), 1e-6, 1 - 1e-6))
    ab = torch.tensor([1.0, np.log(p0 / (1 - p0))], dtype=torch.float64,
                      requires_grad=True)
    opt = torch.optim.LBFGS([ab], lr=0.5, max_iter=iters,
                            tolerance_grad=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            ab[0] * z + ab[1], y)
        loss.backward()
        return loss

    opt.step(closure)
    a, b = ab.detach().tolist()
    return float(a), float(b)


def apply_platt(logit, a, b):
    return 1.0 / (1.0 + np.exp(-np.clip(a * logit + b, -60, 60)))


def auc_macro(score, tp, cls, min_pos=1, min_neg=1):
    """Mean of the WITHIN-PREDICATE AUCs, plus the pooled AUC for contrast.

    THE REAL CONFOUNDER TEST. Pooled AUC can be high purely because the model
    puts frequent predicates above rare ones and frequent predicates are more
    often right — that is prior-ranking, not instance-ranking. The within-class
    AUC asks the deployment question directly: among the pairs I called
    'on', does a higher score mean more likely correct? If pooled >> macro,
    the confidence number is largely reporting WHICH predicate it is.
    """
    per, ns = {}, {}
    for c in np.unique(cls):
        m = cls == c
        if int(tp[m].sum()) < min_pos or int((1 - tp[m]).sum()) < min_neg:
            continue
        per[int(c)] = auc(score[m], tp[m])
        ns[int(c)] = int(m.sum())
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return macro, per, ns


def adaptive_recall(records, mult: int = 1):
    """R@(mult*n) image-macro, and mR@(mult*n) per-class macro."""
    per_img, per_cls = [], defaultdict(list)
    for r in records:
        k = min(mult * r["n_gt"], len(r["tp"]))
        hit = r["tp"][:k]
        per_img.append(hit.sum() / r["n_gt"])
        cls_hits: Dict[int, int] = defaultdict(int)
        for c in r["cls"][:k][hit]:
            cls_hits[int(c)] += 1
        for c, tot in r["per_cls_total"].items():
            per_cls[c].append(cls_hits.get(c, 0) / tot)
    mR = float(np.mean([np.mean(v) for v in per_cls.values()])) if per_cls else 0.0
    return float(np.mean(per_img)), mR, len(per_cls)


def reliability(score, tp, n_bins: int = 10):
    """Equal-width reliability table + ECE."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows, ece, N = [], 0.0, len(score)
    for i in range(n_bins):
        m = (score >= edges[i]) & (score < edges[i + 1] if i < n_bins - 1
                                   else score <= edges[i + 1])
        n = int(m.sum())
        if n == 0:
            rows.append((edges[i], edges[i + 1], 0, float("nan"), float("nan")))
            continue
        conf, acc = float(score[m].mean()), float(tp[m].mean())
        ece += n / N * abs(acc - conf)
        rows.append((edges[i], edges[i + 1], n, conf, acc))
    return rows, float(ece)


def auc(score, tp):
    """ROC-AUC via rank statistic (no sklearn dependency)."""
    pos, neg = int(tp.sum()), int((1 - tp).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks over ties so the AUC is not inflated by score plateaus
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return float((ranks[tp == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="runs/packed/psg")
    p.add_argument("--split", default="test")
    p.add_argument("--weights", default="ema")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--max_objects", type=int, default=40)
    p.add_argument("--eval_budget", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--fit_temperature", action="store_true",
                   help="Fit a temperature on THIS split. Fit on val and apply "
                        "to test: fitting and reporting on the same split "
                        "overstates calibration.")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Apply a pre-fitted temperature (see --fit_temperature).")
    p.add_argument("--fit_platt", action="store_true",
                   help="Fit Platt (a, b) on THIS split; writes them to --out "
                        "so a test run can pick them up with --platt.")
    p.add_argument("--platt", default="",
                   help="'a,b' or a path to a deploy_metrics json holding a "
                        "platt fit from another (val) split.")
    p.add_argument("--negatives", default="",
                   help="Haystack negative-cell sidecar. Turns on FEDERATED "
                        "scoring: a prediction is a false positive only if it "
                        "was explicitly annotated negative, never merely "
                        "because it is absent from an incomplete GT.")
    p.add_argument("--use_calibration", action="store_true",
                   help="score with the checkpoint's calibration.json, i.e. "
                        "exactly what the deployed product computes. Monotone, "
                        "so every ranking metric is unchanged — it makes the "
                        "THRESHOLD rows mean what they say.")
    p.add_argument("--dump", default="",
                   help="Write the raw per-prediction table (score, tp, cls, "
                        "logits, image id) to this.npz. Every calibration "
                        "study then runs on CPU in seconds instead of "
                        "re-running the model on a GPU.")
    p.add_argument("--out", default="")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ck, a.weights).to(dev).eval()

    ds = RelationDataset(root=a.data_root, split=a.split,
                         resolution=a.img_size, max_objects=a.max_objects)
    names = ds.predicate_names
    ck_args = ck.get("args") or {}
    ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    E = encode_texts_student(names, ck_args["text_student"],
                             templates=TRAIN_TEMPLATES, device=dev)
    model.vocab_head.set_vocabulary_matrix(names, E)
    model.reparameterize()

    # Federated mode: join the negative sidecar onto pack row indices and the
    # pack's own predicate order (same remapping as benchmark/eval_haystack.py —
    # both are mandatory, the pack assigns predicate ids by first appearance).
    neg_by_index = None
    if a.negatives:
        from pathlib import Path
        side = json.load(open(a.negatives))
        pid_of = {n: i for i, n in enumerate(names)}
        row_of = {int(Path(f).stem.split("_")[-1]): i
                  for i, f in enumerate(ds.file_names)}
        neg_by_index, n_neg, n_skip = {}, 0, 0
        for img_id, cells in side["by_image_id"].items():
            r = row_of.get(int(img_id))
            if r is None:
                n_skip += len(cells)
                continue
            keep = {(int(s), int(o), pid_of[q]) for s, o, q in cells
                    if q in pid_of}
            n_skip += len(cells) - len(keep)
            if keep:
                neg_by_index[r] = keep
                n_neg += len(keep)
        print(f"federated: {n_neg:,} explicit negative cells over "
              f"{len(neg_by_index):,} images (skipped {n_skip})")

    if a.limit:
        ds = torch.utils.data.Subset(ds, list(range(min(a.limit, len(ds)))))
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)

    # Score exactly as the product does. Monotone, so R@n / AP / AUC are
    # unchanged; it is the THRESHOLD numbers that become meaningful.
    contract = (ScoreContract.for_checkpoint(a.checkpoint) if a.use_calibration
                else ScoreContract())
    print(f"score contract: {contract.describe()}")
    recs, flat, gt_counts, cells = collect(model, loader, dev, a.eval_budget,
                                           neg_by_index=neg_by_index,
                                           contract=contract)
    if a.dump:
        os.makedirs(os.path.dirname(os.path.abspath(a.dump)), exist_ok=True)
        np.savez_compressed(
            a.dump, **{k: v for k, v in flat.items()},
            **{f"cell_{k}": v for k, v in cells.items()},
            predicate_names=np.array(names, dtype=object),
            gt_cls=np.array(sorted(gt_counts)),
            gt_n=np.array([gt_counts[c] for c in sorted(gt_counts)]))
        print(f"dumped raw predictions → {a.dump}")
    print(f"\nimages scored: {len(recs)}   predictions kept: {len(flat['score']):,}"
          f"   GT relations: {sum(gt_counts.values()):,}")
    ngts = np.array([r["n_gt"] for r in recs])
    print(f"GT per image: mean {ngts.mean():.1f}  median {int(np.median(ngts))}"
          f"  p90 {int(np.percentile(ngts,90))}  max {ngts.max()}")

    # ---- 1. adaptive-K recall -------------------------------------------
    print("\n=== R@n — ADAPTIVE K (K = this image's GT count) ===")
    print("at K=n, precision@n == recall@n, so one number IS the operating point")
    print(f"  {'K':>5s}  {'R@K':>8s}  {'mR@K':>8s}  {'classes':>8s}")
    res = {}
    for m in (1, 2, 3):
        R, mR, nc = adaptive_recall(recs, m)
        res[f"R@{m}n"], res[f"mR@{m}n"] = R, mR
        print(f"  {m}n {'':>2s}  {R:8.4f}  {mR:8.4f}  {nc:8d}")

    # ---- 2. confidence: discrimination ----------------------------------
    sc, tp = flat["score"], flat["tp"]
    A = auc(sc, tp)
    print("\n=== CONFIDENCE — DISCRIMINATION ===")
    print(f"  ROC-AUC (score separates TP from FP): {A:.4f}")
    print(f"  mean score  TP {sc[tp==1].mean():.4f}   FP {sc[tp==0].mean():.4f}"
          f"   gap {sc[tp==1].mean()-sc[tp==0].mean():+.4f}")
    print(f"  TP rate overall (base precision): {tp.mean():.4f}")
    res["auc"] = A

    # ---- 3. confidence: calibration -------------------------------------
    rows, ece = reliability(sc, tp)
    res["ece"] = ece
    print("\n=== CONFIDENCE — CALIBRATION (is a score of s worth s?) ===")
    print(f"  {'bin':>12s}  {'n':>9s}  {'mean score':>10s}  {'precision':>9s}  {'gap':>7s}")
    for lo, hi, n, conf, acc in rows:
        if n == 0:
            continue
        print(f"  [{lo:.1f},{hi:.1f}) {n:9,d}  {conf:10.4f}  {acc:9.4f}  {acc-conf:+7.4f}")
    print(f"  ECE = {ece:.4f}   (0 = perfectly calibrated)")

    # ---- 4. threshold sweep ---------------------------------------------
    total_gt = sum(gt_counts.values())
    n_img = len(recs)
    print("\n=== OPERATING POINTS (what a product threshold buys) ===")
    print(f"  {'thr':>5s}  {'precision':>9s}  {'recall':>8s}  {'F1':>7s}"
          f"  {'preds/img':>9s}")
    sweep = []
    for t in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = sc >= t
        n = int(m.sum())
        if n == 0:
            continue
        prec = float(tp[m].mean())
        rec = float(tp[m].sum() / total_gt)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        sweep.append({"thr": t, "precision": prec, "recall": rec, "f1": f1,
                      "preds_per_img": n / n_img})
        print(f"  {t:5.2f}  {prec:9.4f}  {rec:8.4f}  {f1:7.4f}  {n/n_img:9.2f}")
    res["sweep"] = sweep

    # ---- 4b. WHAT OPERATING POINTS EXIST AT ALL? ------------------------
    # The PR curve is invariant to monotonic rescaling, so this is the part
    # calibration CANNOT fix. If no (precision, recall) pair here is usable,
    # the problem is the model, not the threshold.
    prec, rec, ap, thr_sorted = pr_curve(sc, tp, total_gt)
    res["average_precision"] = ap
    print("\n=== PR CURVE — detection-style (calibration CANNOT change this) ===")
    print(f"  Average Precision (AP) = {ap:.4f}   base rate = {tp.mean():.4f}")
    print(f"  {'precision floor':>15s}  {'best recall':>11s}  {'raw thr':>9s}  {'n preds':>9s}")
    for p_, r_, t_, n_ in precision_targets(prec, rec, thr_sorted):
        if n_ == 0:
            print(f"  {p_:15.2f}  {'UNREACHABLE':>11s}")
        else:
            print(f"  {p_:15.2f}  {r_:11.4f}  {t_:9.6f}  {n_:9,d}")

    # ---- 4b-fed. IS THE LOW PRECISION REAL, OR IS THE GT INCOMPLETE? ----
    # PSG/VG150 mark a prediction wrong for being unannotated. Haystack ships
    # explicit negatives, so here "false positive" can mean what it says.
    if neg_by_index is not None:
        lab = flat["lab"]
        cov = float((lab > 0).mean())
        print("\n=== FEDERATED (Haystack explicit negatives) ===")
        print(f"  emitted preds: {len(lab):,}   labelled: {int((lab>0).sum()):,}"
              f" ({cov:.2%})   labelled-pos {int((lab==2).sum()):,}"
              f"   labelled-neg {int((lab==1).sum()):,}")
        m = lab > 0
        if int(m.sum()) > 0:
            sc_f, tp_f = sc[m], tp[m]
            p_f, r_f, ap_f, thr_f = pr_curve(sc_f, tp_f, int(tp_f.sum()))
            print(f"  [emission-biased] AP {ap_f:.4f}  base rate {tp_f.mean():.4f}"
                  f"  AUC {auc(sc_f, tp_f):.4f}")
            print("  NOTE this slice is biased: it only contains emitted preds "
                  "that happened to be annotated.")
        # the unbiased view: score every labelled cell, sampled or not
        cs, ct = cells["score"], cells["tp"]
        if len(cs):
            p_c, r_c, ap_c, thr_c = pr_curve(cs, ct, int(ct.sum()))
            res["federated"] = {
                "coverage": cov, "cell_ap": ap_c,
                "cell_base_rate": float(ct.mean()),
                "cell_auc": auc(cs, ct), "n_cells": int(len(cs)),
                "sampled_frac": float(cells["seen"].mean())}
            print(f"\n  [unbiased, all labelled cells] n {len(cs):,}"
                  f"   pos {int(ct.sum()):,}   neg {int((1-ct).sum()):,}"
                  f"   sampled by the pair sampler {cells['seen'].mean():.2%}")
            print(f"  AP {ap_c:.4f}   base rate {ct.mean():.4f}"
                  f"   lift {ap_c/max(ct.mean(),1e-9):.2f}x"
                  f"   AUC {auc(cs, ct):.4f}")
            print(f"  vs the PSG-convention AP on this same run: {ap:.4f}"
                  f"  (base {tp.mean():.4f}, lift {ap/max(tp.mean(),1e-9):.2f}x)")
            print(f"  {'precision floor':>15s}  {'best recall':>11s}"
                  f"  {'raw thr':>9s}  {'n preds':>9s}")
            for p_, r_, t_, n_ in precision_targets(p_c, r_c, thr_c):
                if n_ == 0:
                    print(f"  {p_:15.2f}  {'UNREACHABLE':>11s}")
                else:
                    print(f"  {p_:15.2f}  {r_:11.4f}  {t_:9.6f}  {n_:9,d}")

    # ---- 4c. CALIBRATION: PLATT (a*z + b) --------------------------------
    # Monotone in the logit for a > 0 => every ranking metric (R@n, R@K, mR@K,
    # AP, AUC) is bit-identical. It buys ADDRESSABILITY, not accuracy.
    ab = None
    if a.fit_platt:
        ab = fit_platt(flat["logit"], tp)
    elif a.platt:
        if os.path.exists(a.platt):
            ab = tuple(json.load(open(a.platt))["platt"])
        else:
            ab = tuple(float(x) for x in a.platt.split(","))
    if ab is not None:
        res["platt"] = list(ab)
        sc_p = apply_platt(flat["logit"], *ab)
        rows_p, ece_p = reliability(sc_p, tp)
        res["ece_platt"] = ece_p
        print(f"\n=== AFTER PLATT SCALING (a = {ab[0]:.4f}, b = {ab[1]:.4f}) ===")
        print(f"  ECE {ece:.4f} -> {ece_p:.4f}")
        print(f"  AP  {ap:.4f} -> {pr_curve(sc_p, tp, total_gt)[2]:.4f}  "
              f"(must be identical — monotone rescale)")
        print(f"  score range {sc_p.min():.4f}.. {sc_p.max():.4f}")
        print(f"  {'bin':>12s}  {'n':>9s}  {'mean score':>10s}  {'precision':>9s}  {'gap':>7s}")
        for lo, hi, n, conf, acc in rows_p:
            if n == 0:
                continue
            print(f"  [{lo:.1f},{hi:.1f}) {n:9,d}  {conf:10.4f}  {acc:9.4f}"
                  f"  {acc-conf:+7.4f}")
        print("\n  calibrated operating points (thr now MEANS precision):")
        print(f"  {'thr':>5s}  {'precision':>9s}  {'recall':>8s}  {'preds/img':>9s}")
        sweep_p = []
        for t_ in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5):
            m = sc_p >= t_
            if int(m.sum()) == 0:
                continue
            sweep_p.append({"thr": t_, "precision": float(tp[m].mean()),
                            "recall": float(tp[m].sum() / total_gt),
                            "preds_per_img": int(m.sum()) / n_img})
            print(f"  {t_:5.2f}  {float(tp[m].mean()):9.4f}  "
                  f"{float(tp[m].sum()/total_gt):8.4f}  {int(m.sum())/n_img:9.2f}")
        res["sweep_platt"] = sweep_p

    # ---- 4d. WHAT IS THE SCORE ACTUALLY RANKING? -------------------------
    # Three questions the pooled AUC cannot answer.
    print("\n=== WHAT DOES THE CONFIDENCE NUMBER RANK? ===")
    a_pred = auc(flat["z_pred"], tp)
    a_pair = auc(flat["z_pair"], tp)
    print(f"  AUC of the predicate term alone (z_pred): {a_pred:.4f}")
    print(f"  AUC of the relatedness term alone (z_pair): {a_pair:.4f}")
    print(f"  AUC of the sum (what we ship): {A:.4f}")
    macro, per_cls_auc, per_cls_n = auc_macro(sc, tp, flat["cls"], min_pos=5,
                                              min_neg=5)
    print(f"\n  pooled AUC   {A:.4f}   (can be high from predicate priors alone)")
    print(f"  WITHIN-predicate macro AUC {macro:.4f} over {len(per_cls_auc)} "
          f"predicates with >=5 pos and >=5 neg")
    print("  -> if pooled >> within, the score is ranking WHICH predicate, "
          "not whether THIS instance is right")
    if per_cls_auc:
        srt_c = sorted(per_cls_auc.items(), key=lambda kv: kv[1])
        print(f"  {'predicate':24s} {'n':>8s} {'AUC':>7s}")
        for c, v in srt_c[:5] + srt_c[-5:]:
            nm = names[c] if c < len(names) else str(c)
            print(f"    {nm:22s} {per_cls_n[c]:8,d} {v:7.4f}")
    res["auc_z_pred"], res["auc_z_pair"] = a_pred, a_pair
    res["auc_within_predicate_macro"] = macro

    # ---- 4e. temperature scaling, an alternative to the Platt fit --------
    T = fit_temperature(flat["logit"], tp) if a.fit_temperature else a.temperature
    if T and T != 1.0:
        z = flat["logit"] / T
        sc_cal = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        rows_cal, ece_cal = reliability(sc_cal, tp)
        res["temperature"], res["ece_calibrated"] = T, ece_cal
        print(f"\n=== AFTER TEMPERATURE SCALING (T = {T:.3f}) ===")
        print(f"  ECE {ece:.4f} -> {ece_cal:.4f}")
        print(f"  AP  {ap:.4f} -> {pr_curve(sc_cal, tp, total_gt)[2]:.4f}  "
              f"(must be identical — monotone rescale)")
        print(f"  {'bin':>12s}  {'n':>9s}  {'mean score':>10s}  {'precision':>9s}  {'gap':>7s}")
        for lo, hi, n, conf, acc in rows_cal:
            if n == 0:
                continue
            print(f"  [{lo:.1f},{hi:.1f}) {n:9,d}  {conf:10.4f}  {acc:9.4f}  {acc-conf:+7.4f}")
        print("\n  calibrated operating points:")
        print(f"  {'thr':>5s}  {'precision':>9s}  {'recall':>8s}  {'preds/img':>9s}")
        for t_ in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7):
            m = sc_cal >= t_
            if int(m.sum()) == 0:
                continue
            print(f"  {t_:5.2f}  {float(tp[m].mean()):9.4f}  "
                  f"{float(tp[m].sum()/total_gt):8.4f}  {int(m.sum())/n_img:9.2f}")

    # ---- 5. THE CONFOUNDER TEST -----------------------------------------
    bucket = frequency_buckets(gt_counts)
    bsz = defaultdict(int)
    for c, b in bucket.items():
        bsz[b] += 1
    print("\n=== IS CONFIDENCE CONFOUNDED BY PREDICATE FREQUENCY? ===")
    print("Same score, different bucket — if precision differs, one global")
    print("threshold silently trades the tail away.")
    print("  buckets: " + ", ".join(f"{k}={v} predicates" for k, v in sorted(bsz.items())))
    cls_bucket = np.array([bucket.get(int(c), "rare") for c in flat["cls"]])
    print(f"\n  {'bin':>12s}  " + "  ".join(f"{b:>22s}" for b in ("freq", "common", "rare")))
    print(f"  {'':>12s}  " + "  ".join(f"{'n / prec / meanscore':>22s}" for _ in range(3)))
    per_bucket = {}
    for lo, hi, n, conf, acc in rows:
        if n == 0:
            continue
        cells = []
        for b in ("freq", "common", "rare"):
            m = ((sc >= lo) & (sc < hi if hi < 1.0 else sc <= hi)
                 & (cls_bucket == b))
            k = int(m.sum())
            cells.append(f"{k:6d} / {tp[m].mean():.3f} / {sc[m].mean():.3f}"
                         if k else f"{0:6d} /   --   /   --  ")
        print(f"  [{lo:.1f},{hi:.1f})  " + "  ".join(f"{c:>22s}" for c in cells))
    for b in ("freq", "common", "rare"):
        m = cls_bucket == b
        if m.sum():
            per_bucket[b] = {"n": int(m.sum()), "precision": float(tp[m].mean()),
                             "mean_score": float(sc[m].mean()),
                             "auc": auc(sc[m], tp[m])}
    print("\n  overall per bucket:")
    for b, v in per_bucket.items():
        print(f"    {b:7s} n={v['n']:8,d}  precision {v['precision']:.4f}"
              f"  mean score {v['mean_score']:.4f}  AUC {v['auc']:.4f}")
    res["per_bucket"] = per_bucket
    res["reliability"] = [{"lo": r[0], "hi": r[1], "n": r[2],
                           "conf": r[3], "acc": r[4]} for r in rows]

    out = a.out or os.path.join(os.path.dirname(a.checkpoint),
                                f"deploy_metrics_{os.path.basename(a.data_root)}"
                                f"_{a.split}.json")
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(f"\nsaved → {out}")


if __name__ == "__main__":
    main()
