"""HICO-DET HOI mAP on the EXACT RF-UC open-vocabulary split (SL-HOI protocol).

Computes composition-level (verb, object) mAP over the 600 HICO categories with
the standard 120-unseen / 480-seen RF-UC partition (VCL/GEN-VLKT lineage; list
verified row-identical to list_action.csv ordering against HOICLIP's
static_hico.py). A detection is (human box, object box, hoi category, score);
score = the model's sigmoid verb score for the pair, emitted only for hoi
categories whose object class matches the object box. TP = both boxes IoU>=0.5
with an unmatched GT of that category (greedy in score order), AP = all-point
interpolated PR area, per category, averaged per split.

HONEST-COMPARISON CAVEATS (state these wherever numbers are shown):
- GT boxes (PredCls-style): inflates us vs SL-HOI's end-to-end detection —
  localization is free and absent-object images contribute no false positives.
- Zero-shot cross-dataset: we never trained on HICO images or its verb list;
  SL-HOI trains on HICO's train set (480 seen categories). For us "seen" is
  just as unseen as "unseen".
- 80 of the 600 categories are no_interaction, which our converter uses as
  negatives; they are unscoreable. Means are reported over scoreable
  categories AND strict (no_interaction counted as AP=0) so the strict number
  is exactly their denominator.
- Unsampled pairs emit no detection (deployed semantics, cf. haystack_eval);
  pair coverage is reported.

    python benchmark/eval_hico_map.py --checkpoint runs/train/<arm>/checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                      # noqa: E402
from relsgg.training.engine import evaluate                          # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES                          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt             # noqa: E402
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

LIST_ACTION = str(DATASETS / "HICO_DET/list_action.csv")
# RF-UC 120 unseen hoi ids, 0-based == list_action.csv row order (verified).
RF_UC_UNSEEN = [8, 22, 27, 44, 50, 55, 62, 66, 70, 76, 77, 80, 83, 84, 90, 99,
    100, 104, 107, 127, 135, 136, 149, 158, 168, 179, 181, 184, 188, 189, 192,
    195, 198, 205, 206, 216, 222, 229, 238, 239, 254, 255, 257, 260, 261, 262,
    274, 279, 280, 281, 286, 289, 292, 303, 311, 315, 317, 325, 333, 334, 345,
    350, 351, 354, 358, 364, 379, 381, 389, 390, 391, 395, 397, 398, 399, 401,
    402, 403, 405, 407, 410, 416, 418, 427, 429, 436, 439, 440, 449, 463, 469,
    474, 482, 485, 498, 499, 504, 509, 517, 520, 522, 526, 531, 535, 539, 546,
    547, 548, 552, 555, 556, 560, 581, 586, 592, 593, 595, 596, 597, 599]


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    x, y, w, h = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=-1)


def iou_1vsN(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(a + b - inter, 1e-9)


def voc_ap(scores: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind="stable")
    tp = tps[order].astype(np.float64)
    fp = 1.0 - tp
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / n_gt
    pre = tp / np.maximum(tp + fp, 1e-9)
    # all-point interpolation: precision envelope, area under PR
    mrec = np.concatenate([[0.0], rec, [rec[-1] if len(rec) else 0.0]])
    mpre = np.concatenate([[0.0], pre, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


class HicoMAPEvaluator:
    """Collect (score, image, h, o) detections per hoi category from pair scores."""

    def __init__(self, hois_of_cat, cat_of_box_local, person_cat_id):
        self.hois_of_cat = hois_of_cat          # pack cat id -> [(hoi, pack verb id)]
        self.cat_of = cat_of_box_local          # row -> np[int] pack cat per local box
        self.person = person_cat_id
        self.dets = defaultdict(lambda: ([], [], [], []))  # hoi -> scores,row,h,o
        self.n_pairs = self.n_scored = 0

    @torch.no_grad()
    def update(self, out: dict, targets) -> None:
        logits = out["logits"]
        sub_idx, obj_idx, valid = out["sub_idx"], out["obj_idx"], out["valid_mask"]
        lg = logits.float()
        if out.get("pair_logits") is not None:
            lg = lg + out["pair_logits"].float().unsqueeze(-1)
        scores = torch.sigmoid(lg)
        for b in range(logits.shape[0]):
            idx = int(targets[b]["index"])
            cats = self.cat_of[idx]
            mask = valid[b]
            pair_row = {}
            if int(mask.sum()):
                s_l = sub_idx[b][mask].tolist()
                o_l = obj_idx[b][mask].tolist()
                for i, (si, oi) in enumerate(zip(s_l, o_l)):
                    pair_row.setdefault((si, oi), i)
            sc = scores[b][mask].cpu().numpy() if int(mask.sum()) else None
            for s in np.nonzero(cats == self.person)[0]:
                for o in range(len(cats)):
                    if o == s:
                        continue
                    self.n_pairs += 1
                    row = pair_row.get((int(s), int(o)))
                    if row is None:
                        continue        # unsampled pair -> no emission
                    self.n_scored += 1
                    for hoi, vid in self.hois_of_cat.get(int(cats[o]), ()):
                        d = self.dets[hoi]
                        d[0].append(float(sc[row, vid]))
                        d[1].append(idx); d[2].append(int(s)); d[3].append(int(o))

    def compute(self):                   # train_engine calls this at the end
        return {"pair_coverage": self.n_scored / max(self.n_pairs, 1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pack", default="runs/packed/hicodet")
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--out", default="")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {a.checkpoint} (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, a.weights).to(device).eval()

    ds = RelationDataset(root=a.pack, split="test", resolution=a.img_size,
                         max_objects=100)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)
    pred_names = ds.predicate_names
    pid = {n: i for i, n in enumerate(pred_names)}

    la = list(csv.DictReader(open(LIST_ACTION)))
    assert len(la) == 600
    hoi_pred = [r["vname_ing"].replace("_", " ") for r in la]
    hoi_obj = [r["nname"] for r in la]
    noint = {i for i, r in enumerate(la) if r["vname"] == "no_interaction"}

    # pack cat id -> [(hoi, pack verb id)] for real, in-vocab verbs
    hois_of_cat: dict = defaultdict(list)
    for h in range(600):
        if h in noint or hoi_pred[h] not in pid:
            continue
        c = ds.cat_to_idx.get(hoi_obj[h])
        if c is not None:
            hois_of_cat[c].append((h, pid[hoi_pred[h]]))

    # Per-row metadata straight from the pack arrays (order == dataset order).
    meta = np.asarray(ds.img_meta)
    box_cats = np.asarray(ds.box_cats)
    boxes_all = np.asarray(ds.boxes, dtype=np.float32)
    rels_all = np.asarray(ds.rels, dtype=np.int64)
    cat_of, xyxy_of, gt_of = {}, {}, defaultdict(lambda: defaultdict(list))
    n_gt = np.zeros(600, dtype=np.int64)
    for i in range(len(ds)):
        b0, nb, r0, nr = (int(meta[i, 3]), int(meta[i, 4]),
                          int(meta[i, 5]), int(meta[i, 6]))
        cat_of[i] = box_cats[b0:b0 + nb]
        xyxy_of[i] = cxcywh_to_xyxy(boxes_all[b0:b0 + nb])
        for s, o, pr in rels_all[r0:r0 + nr,:3]:
            verb = pred_names[pr]
            oc = int(box_cats[b0 + o])
            hoi = next((h for h, v in hois_of_cat.get(oc, ())
                        if pred_names[v] == verb), None)
            assert hoi is not None, (i, verb, oc)
            gt_of[hoi][i].append((int(s), int(o)))
            n_gt[hoi] += 1
    print(f"GT: {int(n_gt.sum()):,} instances over "
          f"{int((n_gt > 0).sum())} of 600 categories")

    person = ds.cat_to_idx["person"]
    ev = HicoMAPEvaluator(hois_of_cat, cat_of, person)

    # Reparameterize to the pack's 116 verbs in the checkpoint's text space.
    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    ts = ck_args.get("text_student") or ""
    from relsgg.text.student import encode_texts_student
    E = encode_texts_student(pred_names, ts, templates=TRAIN_TEMPLATES,
                             device=device)
    model.vocab_head.set_vocabulary_matrix(pred_names, E)
    model.reparameterize()

    eval_args = SimpleNamespace(amp=device.type == "cuda",
                                amp_dtype_t=torch.bfloat16)
    cov = evaluate(model, loader, device, eval_args, ev,
                   eval_budget=a.eval_budget)
    print(f"pair coverage: {cov['pair_coverage']:.4f}  "
          f"({ev.n_scored:,}/{ev.n_pairs:,} person->object pairs)")

    # Greedy IoU matching + AP per category.
    ap = np.full(600, np.nan)
    for hoi in range(600):
        if hoi in noint:
            continue
        sc, rows, hs, os_ = ev.dets.get(hoi, ([], [], [], []))
        sc = np.asarray(sc, dtype=np.float64)
        order = np.argsort(-sc, kind="stable")
        matched = {r: np.zeros(len(v), bool) for r, v in gt_of[hoi].items()}
        tp = np.zeros(len(sc), bool)
        for k in order:
            r = rows[k]
            gts = gt_of[hoi].get(r)
            if not gts:
                continue
            bx = xyxy_of[r]
            best, best_j = 0.5, -1
            for j, (gh, go) in enumerate(gts):
                if matched[r][j]:
                    continue
                mi = min(iou_1vsN(bx[hs[k]], bx[gh:gh + 1])[0],
                         iou_1vsN(bx[os_[k]], bx[go:go + 1])[0])
                if mi >= best:
                    best, best_j = mi, j
            if best_j >= 0:
                matched[r][best_j] = True
                tp[k] = True
        ap[hoi] = voc_ap(sc, tp, int(n_gt[hoi]))

    unseen = np.zeros(600, bool); unseen[RF_UC_UNSEEN] = True
    scoreable = ~np.isnan(ap)

    def mean_of(sel, strict):
        v = ap[sel & scoreable]
        if strict:            # unscoreable (no_interaction) counted as 0
            return float(np.nansum(ap[sel]) / int(sel.sum())), int(sel.sum())
        return float(v.mean()) if len(v) else float("nan"), int(len(v))

    res = {}
    for name, sel in [("unseen", unseen), ("seen", ~unseen),
                      ("full", np.ones(600, bool))]:
        m, n = mean_of(sel, False)
        ms, ns = mean_of(sel, True)
        res[f"mAP_{name}"] = m; res[f"n_{name}"] = n
        res[f"mAP_{name}_strict600"] = ms; res[f"n_{name}_strict"] = ns
        print(f"{name:7s} mAP {100 * m:6.2f} (over {n} scoreable)   "
              f"strict {100 * ms:6.2f} (over {ns} incl. no_interaction=0)")
    res["pair_coverage"] = cov["pair_coverage"]

    out = a.out or os.path.join(os.path.dirname(a.checkpoint),
                                "hico_map_rfuc.json")
    json.dump({"metrics": res,
               "per_class_ap": {i: (None if np.isnan(ap[i]) else float(ap[i]))
                                for i in range(600)},
               "protocol": "RF-UC 120 unseen, GT boxes, zero-shot",
               "weights": a.weights}, open(out, "w"), indent=2)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
