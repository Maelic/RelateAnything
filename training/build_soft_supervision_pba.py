"""v42.2: P(B true | A written) as the POSITIVE weight — one estimator, both sides.

THE MEASURED FRONTIER (proxy arms wv2 vs wv2_sw, [[relsgg-50k-proxy-pack]]):
- cosine kernel: vocabulary-shift robust (best transfer mR ever) but head-diluting
  (`on` -69..-89%).
- support weighting: heads resurrected (+29..+96% micro) but the rare sibling of a
  frequent synonym starves — `near` (31K) under `beside`/`next to` (137K/141K),
  `under` (10.7K) under `below` (119K) — both -> 0.000 on transfer, because the
  benchmarks use exactly those strings.

THE UNIFICATION: what the positive target should encode is not "j is a synonym"
(kernel) nor "the annotator would write j" (support) but "j is TRUE for this
pair" — which is the also-true estimator's exact quantity, already validated
held-out for the negative side. Its learned asymmetry is the needed structure:
  specific->generic high  (beside => near: `near`'s column stays anchored)
  generic->specific low   (on =/> resting on: heads stop diluting)

WHAT THIS SCRIPT DOES: refits the estimator EXACTLY as build_soft_supervision.py
(same helpers, same seed, same fit split) and re-weights the EXISTING kernel
support (pos_i/pos_j pairs, cos floor inherited) with p on the CASE-CONTROL
scale — no King-Zeng deployment shift (that scale is correct for "how likely is
a random column also-true", i.e. the negatives; as positive weights next to
diag=1 it would flatten everything to ~1e-3) and no Elkan-Noto division (c=0.016
would saturate half the matrix at 1.0). Scale choice is a monotone-transform
call, made explicitly here. Negatives/symmetry/inverses copied verbatim; the
loss loader already zeroes neg_lw wherever a pair is positive.

KNOWN RISK, checked in the review table before any GPU: the estimator is trained
on co-annotations and TRUE LEXICAL SYNONYMS ARE NEVER CO-ANNOTATED (0/905,343),
so their `direct` feature is silent and their weight rides on cos_v2 + ctx. If
the table shows same-group weights at background level, this is v40 identity
positives in disguise — do not launch on it.

    python training/build_soft_supervision_pba.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from training.estimate_false_negatives import (coannotations, load_pack,  # noqa: E402
                                      pairs_from, sym_counts)
from training.build_soft_supervision import load_embeds  # noqa: E402

TS = PROJ / "runs/packed/datamix_v22/text_space"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(TS / "soft_supervision.npz"))
    ap.add_argument("--packs", nargs="+",
                    default=["runs/packed/megasg", "runs/packed/vg_raw"])
    ap.add_argument("--union_preds", default=str(TS / "union_predicates.json"))
    ap.add_argument("--kernel_embeds",
                    default=str(TS / "pred_embeds_studentv2_photo.npz"))
    ap.add_argument("--pred_context", default=str(TS / "pred_context_mc5.npz"))
    ap.add_argument("--neg_per_pos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(TS / "soft_supervision_pba.npz"))
    a = ap.parse_args()
    rs = np.random.RandomState(a.seed)

    z = dict(np.load(a.src, allow_pickle=False))
    names = [str(n) for n in z["predicates"]]
    idx = {n: i for i, n in enumerate(names)}
    V = len(names)
    E2 = load_embeds(Path(a.kernel_embeds), names)

    # ---- pooled relations + counts (identical to the builder) ----
    rel_u, img_u, off = [], [], 0
    counts = np.zeros(V, np.int64)
    for root in a.packs:
        pnames, rels, img = load_pack(Path(root), "train")
        remap = np.array([idx.get(n, -1) for n in pnames], np.int64)
        g = remap[rels[:, 2]]
        keep = g >= 0
        r = rels[keep].copy()
        r[:, 2] = g[keep]
        rel_u.append(r)
        img_u.append(img[keep] + off)
        off += img.max() + 1
        np.add.at(counts, g[keep], 1)
    rels = np.concatenate(rel_u)
    img = np.concatenate(img_u)

    # ---- estimator fit, verbatim from build_soft_supervision.py ----
    from sklearn.linear_model import LogisticRegression

    p_, st, ct = coannotations(rels, img)
    n_box = len(st)
    perm = rs.permutation(n_box)
    f, d = int(0.6 * n_box), int(0.8 * n_box)
    P = {k: pairs_from(p_, st[v], ct[v])
         for k, v in [("fit", perm[:f]), ("dev", perm[f:d]), ("test", perm[d:])]}
    C_fit = sym_counts(P["fit"], V)

    key_all = (img * 1024 + rels[:, 0].astype(np.int64)) * 1024 \
        + rels[:, 1].astype(np.int64)
    order_all = np.argsort(key_all, kind="stable")
    k_sorted = key_all[order_all]
    p_sorted = rels[order_all, 2].astype(np.int64)
    rev_key = (img * 1024 + rels[:, 1].astype(np.int64)) * 1024 \
        + rels[:, 0].astype(np.int64)
    r_lo = np.searchsorted(k_sorted, rev_key, side="left")
    r_hi = np.searchsorted(k_sorted, rev_key, side="right")
    C_rev = np.zeros((V, V), np.float32)
    for t in np.nonzero(r_hi > r_lo)[0]:
        C_rev[rels[t, 2], p_sorted[r_lo[t]:r_hi[t]]] += 1.0

    zc = np.load(a.pred_context, allow_pickle=False)
    cn = [str(x) for x in zc["predicates"]]
    pid = zc["pred_ids"].astype(np.int64)
    S_ctx = zc["ctx_sim"].astype(np.float32)
    ctx_row = np.full(V, -1, np.int64)
    for r_, u in enumerate(pid):
        j = idx.get(cn[u], -1)
        if j >= 0:
            ctx_row[j] = r_
    lfreq = np.log(counts + 1.0).astype(np.float32)

    def feats(A, B):
        cos = (E2[A] * E2[B]).sum(-1)
        direct = np.log((C_fit[A, B] + 1e-3) / (counts[A] + 1.0)).astype(np.float32)
        rev = np.log((C_rev[A, B] + 1e-3) / (counts[A] + 1.0)).astype(np.float32)
        ca, cb = ctx_row[A], ctx_row[B]
        ok = (ca >= 0) & (cb >= 0)
        cx = np.zeros(len(A), np.float32)
        cx[ok] = S_ctx[ca[ok], cb[ok]]
        return np.stack([direct, rev, cos, lfreq[B], cx], 1)

    def xy(split):
        pos = np.unique(P[split], axis=0)
        mA = np.bincount(pos.reshape(-1), minlength=V).astype(np.float64)
        mA /= mA.sum()
        nA = rs.choice(V, size=len(pos) * a.neg_per_pos * 2, p=mA)
        nB = rs.choice(V, size=len(pos) * a.neg_per_pos * 2)
        ng = np.stack([nA, nB], 1)
        ng = ng[ng[:, 0] != ng[:, 1]]
        sp = {tuple(sorted(t)) for t in pos}
        ng = np.array([t for t in ng if tuple(sorted(t)) not in sp])[
            :len(pos) * a.neg_per_pos]
        X = np.concatenate([feats(pos[:, 0], pos[:, 1]), feats(ng[:, 0], ng[:, 1])])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(ng))])
        return X, y

    Xd, yd = xy("dev")
    clf = LogisticRegression(max_iter=2000).fit(Xd, yd)
    print(f"[pba] refit coefs {dict(zip(['direct','rev','cos_v2','log_freq','ctx'], clf.coef_[0].round(3)))}"
          f"  sample prior {yd.mean():.4f}  (NO deployment shift, NO Elkan-Noto)")

    # ---- re-weight the existing kernel support with case-control p ----
    gi, gj = z["pos_i"].astype(np.int64), z["pos_j"].astype(np.int64)
    w_old = z["pos_w"].astype(np.float32)
    p_cc = np.empty(len(gi), np.float32)
    for s in range(0, len(gi), 2_000_000):
        e = min(s + 2_000_000, len(gi))
        p_cc[s:e] = clf.predict_proba(feats(gi[s:e], gj[s:e]))[:, 1]

    base = float(yd.mean())
    print(f"[pba] p_cc over kernel support: median {np.median(p_cc):.4f}  "
          f"p90 {np.percentile(p_cc, 90):.4f}  base rate {base:.4f}")

    # review table — the launch gate lives HERE
    def show(A, B):
        ia, ib = idx.get(A, -1), idx.get(B, -1)
        if ia < 0 or ib < 0:
            return
        p = float(clf.predict_proba(feats(np.array([ia]), np.array([ib])))[:, 1][0])
        m = (gi == ia) & (gj == ib)
        wo = float(w_old[m][0]) if m.any() else float("nan")
        print(f"    {A:>14s} -> {B:<14s}  kernel {wo:.3f}  p_cc {p:.3f}")
    print("[pba] review (kernel weight -> estimator weight):")
    for A, B in [("on", "resting on"), ("resting on", "on"),
                 ("beside", "near"), ("near", "beside"),
                 ("below", "under"), ("under", "below"),
                 ("in front of", "before"), ("before", "in front of"),
                 ("above", "below"), ("next to", "near"),
                 ("on", "on top of"), ("on top of", "on")]:
        show(A, B)

    def mass(t):
        m = gi == t
        return float(w_old[m].sum()) + 1.0, float(p_cc[m].sum()) + 1.0
    print(f"\n    {'predicate':>14s} {'mass kernel':>12s} {'mass p_cc':>10s} "
          f"{'exact share ->':>14s}")
    for name in ["on", "near", "in front of", "behind", "resting beside"]:
        if name in idx:
            mo, mn = mass(idx[name])
            print(f"    {name:>14s} {mo:>12.1f} {mn:>10.1f}   "
                  f"{1/mo:.3f} -> {1/mn:.3f}")

    keep = p_cc > 1e-4
    out = {k: v for k, v in z.items() if not k.startswith("pos_")}
    out.update(pos_i=gi[keep].astype(z["pos_i"].dtype),
               pos_j=gj[keep].astype(z["pos_j"].dtype),
               pos_w=p_cc[keep].astype(z["pos_w"].dtype))
    np.savez_compressed(a.out, **out)
    print(f"\nwrote {a.out}  ({int(keep.sum()):,} positive entries, "
          f"{(1 - keep.sum()/len(gi))*100:.1f}% dropped)")


if __name__ == "__main__":
    main()
