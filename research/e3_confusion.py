"""Predicate confusion analysis for two dump_pair_scores.py npz — CPU only.

Two complementary views, because they answer different questions and a single
"confusion matrix" for SGG silently conflates them:

  RECALL view  (--view recall, the classic PredCls confusion): restrict to pairs
      that carry a GT relation, take the model's argmax over the pack's
      predicates. Rows = GT predicate, cols = predicted. Answers "when the
      answer exists, what does the model say instead?" Blind to false positives.

  PRECISION view (--view precision): take the pairs the model ranks highest at a
      matched emission budget, and ask what fraction of each predicted predicate
      is backed by a GT edge of that predicate. Answers "when the model speaks,
      is it right?" This is the view that sees over-emission, which is the whole
      reason the background penalty exists.

Matched budget matters: a model that emits less looks more precise for free, so
both arms are cut at the SAME number of emitted edges (the GT triple count by
default) rather than at a shared score threshold, which these uncalibrated proxy
checkpoints do not share.

    python training/e3_confusion.py \
        runs/analysis/e3/dump_topk_vg150.npz runs/analysis/e3/dump_lse_vg150.npz \
        --labels topk lse --pack runs/packed/vg150/test
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def load(path, pack):
    z = np.load(path, allow_pickle=False)
    names = [str(s) for s in z["pred_names"]]
    pc = z["pair_counts"].astype(np.int64)
    gc = z["gt_counts"].astype(np.int64)
    pair_off = np.concatenate([[0], np.cumsum(pc)])[:-1]
    gt_off = np.concatenate([[0], np.cumsum(gc)])[:-1]
    return dict(z=z, names=names, pc=pc, gc=gc, pair_off=pair_off,
                gt_off=gt_off, sub=z["sub"].astype(np.int64),
                obj=z["obj"].astype(np.int64), gt=z["gt"].astype(np.int64),
                zpred=z["z_pred"], zpair=z["z_pair"].astype(np.float32),
                path=path)


def gt_pair_index(d):
    """Map every GT triple to its row in the flat pair arrays, or -1.

    Pairs are what the sampler kept, so a GT triple whose pair was not proposed
    has no row at all — those are counted and reported rather than dropped
    silently, since a recall-view matrix built on the survivors alone would
    overstate accuracy by exactly the sampler's miss rate.
    """
    n_img = len(d["pc"])
    out = np.full(len(d["gt"]), -1, np.int64)
    for i in range(n_img):
        ng, npr = d["gc"][i], d["pc"][i]
        if ng == 0 or npr == 0:
            continue
        p0, g0 = d["pair_off"][i], d["gt_off"][i]
        key = (d["sub"][p0:p0 + npr].astype(np.int64) << 20) + d["obj"][p0:p0 + npr]
        order = np.argsort(key, kind="stable")
        ks = key[order]
        g = d["gt"][g0:g0 + ng]
        want = (g[:, 0].astype(np.int64) << 20) + g[:, 1]
        pos = np.searchsorted(ks, want)
        ok = (pos < len(ks)) & (ks[np.clip(pos, 0, len(ks) - 1)] == want)
        out[g0:g0 + ng][ok] = p0 + order[pos[ok]]
    return out


def recall_matrix(d, gp, V):
    """Rows = GT predicate, cols = argmax prediction at that GT pair."""
    ok = gp >= 0
    rows = d["gt"][ok, 2]
    pred = np.asarray(d["zpred"][gp[ok]], np.float32).argmax(1)
    M = np.zeros((V, V), np.int64)
    np.add.at(M, (rows, pred), 1)
    return M, ok.mean()


def precision_edges(d, budget):
    """Top-`budget` (pair, predicate) edges by the deployed score, graph-
    constrained: one predicate per pair, matching how the model is decoded."""
    zp = np.asarray(d["zpred"], np.float32)
    best = zp.argmax(1)
    score = zp[np.arange(len(zp)), best] + d["zpair"]
    keep = np.argpartition(-score, budget)[:budget]
    return keep, best[keep]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs=2)
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--pack", default="runs/packed/vg150/test")
    p.add_argument("--top", type=int, default=12,
                   help="rows shown in the per-predicate tables")
    p.add_argument("--out", default="")
    a = p.parse_args()

    D = [load(pth, a.pack) for pth in a.dumps]
    names = D[0]["names"]
    V = len(names)
    assert D[1]["names"] == names, "arms use different predicate vocabularies"
    counts = json.load(open(f"{a.pack}/meta.json")).get("predicate_counts", {})

    GP = [gt_pair_index(d) for d in D]
    n_gt = len(D[0]["gt"])
    print(f"{n_gt:,} GT triples | {V} predicates | "
          f"sampler covers {100*(GP[0] >= 0).mean():.2f}% / "
          f"{100*(GP[1] >= 0).mean():.2f}% of them")

    # ---------------- recall view -----------------------------------------
    Ms, accs = [], []
    for d, gp, lab in zip(D, GP, a.labels):
        M, cov = recall_matrix(d, gp, V)
        Ms.append(M)
        acc = np.diag(M).sum() / max(M.sum(), 1)
        per = np.divide(np.diag(M), M.sum(1), out=np.zeros(V),
                        where=M.sum(1) > 0)
        accs.append(per)
        print(f"\n[{lab}] RECALL view: micro top-1 accuracy at GT pairs "
              f"{100*acc:.2f}%  |  macro {100*per[M.sum(1) > 0].mean():.2f}%")

    sup = Ms[0].sum(1)
    d_acc = accs[1] - accs[0]
    order = np.argsort(-np.abs(d_acc))
    order = [i for i in order if sup[i] >= 20][:a.top]
    print(f"\nLargest per-predicate top-1 accuracy changes "
          f"({a.labels[1]} - {a.labels[0]}, support>=20):")
    print(f"{'predicate':<18}{'n_gt':>7}{a.labels[0]:>9}{a.labels[1]:>9}"
          f"{'delta':>9}   top confusion (" + a.labels[1] + ")")
    for i in order:
        row = Ms[1][i].copy()
        row[i] = 0
        j = int(row.argmax())
        conf = f"{names[j]} {100*row[j]/max(sup[i],1):.0f}%" if row[j] else "-"
        print(f"{names[i]:<18}{sup[i]:>7,}{100*accs[0][i]:>8.1f}%"
              f"{100*accs[1][i]:>8.1f}%{100*d_acc[i]:>+8.1f}   {conf}")

    # Confusions that MOVED, aggregated off-diagonal.
    off = [M.copy() for M in Ms]
    for M in off:
        np.fill_diagonal(M, 0)
    dm = off[1].astype(np.int64) - off[0].astype(np.int64)
    flat = np.argsort(-np.abs(dm).ravel())[:a.top]
    print(f"\nOff-diagonal cells that moved most ({a.labels[1]} - {a.labels[0]}):")
    for f in flat:
        i, j = divmod(int(f), V)
        if dm[i, j] == 0:
            continue
        print(f"  GT {names[i]:<16} -> said {names[j]:<16} "
              f"{off[0][i,j]:>6,} -> {off[1][i,j]:>6,}  ({dm[i,j]:+,})")

    # ---------------- precision view --------------------------------------
    budget = n_gt
    print(f"\n{'='*74}\nPRECISION view @ matched budget = {budget:,} emitted "
          f"edges (= GT triple count)")
    prec, emitted = [], []
    for d, gp, lab in zip(D, GP, a.labels):
        keep, pred = precision_edges(d, budget)
        # a kept edge is correct iff that pair has a GT relation with that label
        truth = np.full(len(d["sub"]), -1, np.int64)
        okg = gp >= 0
        truth[gp[okg]] = d["gt"][okg, 2]
        hit = truth[keep] == pred
        cm = np.zeros(V, np.int64)
        np.add.at(cm, pred, 1)
        hm = np.zeros(V, np.int64)
        np.add.at(hm, pred[hit], 1)
        prec.append(np.divide(hm, cm, out=np.zeros(V), where=cm > 0))
        emitted.append(cm)
        # concentration: how much of the graph one predicate eats
        share = cm / max(cm.sum(), 1)
        ent = -(share[share > 0] * np.log(share[share > 0])).sum()
        print(f"[{lab}] micro precision {100*hit.mean():.2f}%  |  "
              f"distinct predicates used {int((cm > 0).sum())}/{V}  |  "
              f"top-1 share {100*share.max():.1f}% ({names[int(share.argmax())]})"
              f"  |  entropy {ent:.3f} nats (max {np.log(V):.3f})")

    dp = prec[1] - prec[0]
    de = emitted[1] - emitted[0]
    ordp = [i for i in np.argsort(-np.abs(de)) if emitted[0][i] + emitted[1][i] >= 50][:a.top]
    print(f"\nEmission volume + precision by predicate "
          f"({a.labels[1]} - {a.labels[0]}):")
    print(f"{'predicate':<18}{'n_train':>9}{'emit_'+a.labels[0]:>11}"
          f"{'emit_'+a.labels[1]:>11}{'d_emit':>9}{'prec_'+a.labels[0]:>11}"
          f"{'prec_'+a.labels[1]:>11}")
    for i in ordp:
        print(f"{names[i]:<18}{counts.get(names[i], 0):>9,}{emitted[0][i]:>11,}"
              f"{emitted[1][i]:>11,}{de[i]:>+9,}{100*prec[0][i]:>10.1f}%"
              f"{100*prec[1][i]:>10.1f}%")

    if a.out:
        np.savez_compressed(
            a.out, names=np.array(names),
            recall_cm_a=Ms[0], recall_cm_b=Ms[1],
            prec_a=prec[0], prec_b=prec[1],
            emit_a=emitted[0], emit_b=emitted[1],
            labels=np.array(a.labels))
        print(f"\nsaved -> {a.out}")


if __name__ == "__main__":
    main()
