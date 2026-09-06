"""SpatialSense (A6) metrics, shared by every scorer that reports the axis.

Kept free of torch so the OvSGTR interchange scorer and our own evaluator compute
AUC / AP / accuracy-at-threshold from one implementation. A second copy in a second
script is how two models end up on two protocols; this module is the protocol.

Cells are (score, label, predicate-id) triples: one per labelled SpatialSense
(subject box, predicate, object box) with its verified TRUE/FALSE label. The test
split is exactly balanced, so chance is 0.5 on every headline.
"""
from __future__ import annotations

import numpy as np


def best_threshold(scores, labels):
    """Accuracy-maximising threshold over the mid-points between sorted scores."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    order = np.argsort(scores)
    s = scores[order]
    cand = np.unique(np.concatenate([[0.0], (s[:-1] + s[1:]) / 2, [1.0]]))
    accs = [(float(((scores >= t) == labels).mean()), float(t)) for t in cand]
    acc, t = max(accs)
    return t, acc


def summarise(scores, labels, preds, names, tau, coverage, use_pair_logits=True):
    """The result record eval_spatialsense.py has always written, from cell arrays.

    tau is the global threshold fitted on the VALID split (their protocol); the
    oracle threshold fitted on test is reported only as an upper bound.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    s = np.asarray(scores, dtype=np.float64)
    l = np.asarray(labels).astype(int)
    pr = np.asarray(preds).astype(int)
    lb = l.astype(bool)
    otau, oacc = best_threshold(s, l)
    res = {"use_pair_logits": bool(use_pair_logits),
           "AUC": float(roc_auc_score(l, s)),
           "AP": float(average_precision_score(l, s)),
           "acc@valid_tau": float(((s >= tau) == lb).mean()),
           "acc@0.5": float(((s >= 0.5) == lb).mean()),
           "acc@oracle": float(oacc), "valid_tau": float(tau),
           "oracle_tau": float(otau), "coverage": float(coverage), "n": int(len(s))}
    per = {}
    for i, nm in enumerate(names):
        m = pr == i
        if not m.any():
            continue
        si, li = s[m], l[m]
        row = {"n": int(m.sum()), "pos": int(li.sum()),
               "acc@valid_tau": float(((si >= tau) == li.astype(bool)).mean())}
        if 0 < li.sum() < len(li):
            row["AUC"] = float(roc_auc_score(li, si))
            row["AP"] = float(average_precision_score(li, si))
            row["acc@oracle"] = best_threshold(si, li)[1]
        per[nm] = row
    res["per_predicate"] = per
    return res


def print_summary(res: dict) -> None:
    print(f"\nOVERALL   AUC {res['AUC']:.4f}   AP {res['AP']:.4f}"
          f"   acc@valid_tau {res['acc@valid_tau']:.4f}"
          f"   acc@0.5 {res['acc@0.5']:.4f}"
          f"   acc@oracle {res['acc@oracle']:.4f}  (upper bound)")
    print(f"\n{'predicate':18s} {'n':>5s} {'pos':>5s} {'AUC':>7s} {'AP':>7s} "
          f"{'acc@vtau':>9s} {'acc@orac':>9s}")
    for nm, row in res["per_predicate"].items():
        print(f"{nm:18s} {row['n']:5d} {row['pos']:5d} "
              f"{row.get('AUC', float('nan')):7.4f} {row.get('AP', float('nan')):7.4f} "
              f"{row['acc@valid_tau']:9.4f} {row.get('acc@oracle', float('nan')):9.4f}")
