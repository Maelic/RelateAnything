"""Build the soft-supervision artifact the relation loss reads.

Every semantic constant the objective needs is estimated here rather than set
by hand: which predicates count as synonyms of which, how much an unannotated
column is likely to be true anyway, and how reciprocal a predicate is. Each of
those is a claim about meaning, and a threshold on a text cosine is a poor way
to make one — thresholds do not even survive a change of text encoder. Every
estimate here is validated on held-out data before a GPU run uses it.

The artifact (`soft_supervision.npz`) contains, all on the union vocabulary order:

  pos_w    sparse [i, j, w]  w = P(synonym | cos_v2), isotonic-fitted on the 1,177
           LEXICAL synonym pairs (scaffolding-stripped identity — independent of any
           table and of the encoder being scored; measured AP 0.739 for cos_v2 vs
           0.496 for the hand table). Replaces group positives; consumed as per-member
           weights in the L_out positive term (diag=1 implied). Only the SHAPE matters:
           the weighted mean normalises per row, so the neg_per_pos-dependent absolute
           calibration is not load-bearing.
  neg_lw   sparse [i, j, log(1-p)]  p = Elkan-Noto-corrected P(j also true | i
           annotated) from the co-annotation estimator (706K multi-annotated box
           pairs; fit/dev/test split ON BOX PAIRS). Additive log-weight on denominator
           logits: p->1 reproduces the old ignore mask as a limit, no threshold
           anywhere. Replaces tau_ignore, hard_lo banding, and the 0.3s.
  sym      [V]  posterior P(predicate is reciprocal) from a 2-component beta-binomial
           EM over reverse-annotation rates (`beside` 2.5% vs directional floor
           ~0.15%). Replaces the alpha<=0.5 hinge-eligibility rule; the hinge weight
           becomes (1 - sym[g]).
  inv_w    sparse [i, j, w]  kernel expansion of the four inverse seed pairs:
           w = P_syn(i,a)P_syn(j,b) + P_syn(i,b)P_syn(j,a). Replaces table expansion
           (the seeds themselves stay: measured 0/5 inverse binding in every text
           space, so they are not derivable from data we have).
  w_cooc   scalar  1 - P(also true | cooc-seen), measured on held-out co-annotations
           with the same Elkan-Noto correction. Replaces soft_neg_weight=0.3.

Numerical floors (0.01 on stored p and w) are sparsity cutoffs, not semantics: below
them the log-weight is <0.01 nats and the weighted-mean share <1%.

    python training/build_soft_supervision.py \\
        --out runs/packed/datamix_v22/text_space/soft_supervision.npz
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from training.estimate_false_negatives import (coannotations, load_pack,  # noqa: E402
                                      pairs_from, sym_counts)

TS = PROJ / "runs/packed/datamix_v22/text_space"
SEEDS = {"above": "below", "to the left of": "to the right of",
         "in front of": "behind", "on top of": "beneath"}
SCAFFOLD = r"^(is|are|being|located|positioned|placed|situated|arranged|directly|" \
           r"just|partially|seen|visible|shown|set)\s+"
FLOOR = 0.01     # numerical sparsity floor — see module docstring


def norm(p: str) -> str:
    q = p.strip().lower()
    while True:
        q2 = re.sub(SCAFFOLD, "", q)
        if q2 == q:
            return q
        q = q2


def load_embeds(path: Path, names: list[str]) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    assert [str(x) for x in z["predicates"]] == names, f"{path} vocabulary mismatch"
    E = z["embeddings"].astype(np.float32)
    return E / (np.linalg.norm(E, axis=-1, keepdims=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packs", nargs="+",
                    default=["runs/packed/megasg", "runs/packed/vg_raw"])
    ap.add_argument("--union_preds", default=str(TS / "union_predicates.json"))
    ap.add_argument("--kernel_embeds",
                    default=str(TS / "pred_embeds_studentv2_photo.npz"),
                    help="the text space the supervision is estimated in, "
                         "which need not be the space W is trained in")
    ap.add_argument("--pred_context", default=str(TS / "pred_context_mc5.npz"))
    ap.add_argument("--neg_per_pos", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(TS / "soft_supervision.npz"))
    a = ap.parse_args()
    rs = np.random.RandomState(a.seed)
    report: dict = {}

    names = json.load(open(a.union_preds))
    V = len(names)
    idx = {n: i for i, n in enumerate(names)}
    E2 = load_embeds(Path(a.kernel_embeds), names)                    # [V, D]
    print(f"[vocab] {V:,} predicates; oracle embeds {E2.shape}")

    # ---- pooled relations from all training packs, remapped to union ids ---------
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
        print(f"[packs] {root}: {keep.sum():,} relations "
              f"({(~keep).sum():,} OOV dropped)")
    rels = np.concatenate(rel_u)
    img = np.concatenate(img_u)

    # =========================================================================
    # 1. SYNONYM KERNEL — isotonic P(synonym | cos_v2) on lexical ground truth
    # =========================================================================
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score

    buckets: dict = {}
    for n in names:
        buckets.setdefault(norm(n), []).append(n)
    lex = np.array([[idx[x], idx[y]] for v in buckets.values() if len(v) > 1
                    for x, y in itertools.combinations(v, 2)], np.int64)
    marg = np.bincount(lex.reshape(-1), minlength=V).astype(np.float64)
    marg = marg * (counts + 1.0)
    marg /= marg.sum()
    neg = rs.choice(V, size=(len(lex) * a.neg_per_pos * 2, 2), p=marg)
    neg = neg[neg[:, 0] != neg[:, 1]]
    seen_lex = {tuple(sorted(t)) for t in lex}
    neg = np.array([t for t in neg if tuple(sorted(t)) not in seen_lex])[
:len(lex) * a.neg_per_pos]
    cs = np.concatenate([(E2[lex[:, 0]] * E2[lex[:, 1]]).sum(-1),
                         (E2[neg[:, 0]] * E2[neg[:, 1]]).sum(-1)])
    ys = np.concatenate([np.ones(len(lex)), np.zeros(len(neg))])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cs, ys)
    report["kernel"] = {
        "n_lex_pairs": int(len(lex)), "n_neg": int(len(neg)),
        "ap": round(float(average_precision_score(ys, cs)), 4),
        "curve": {round(c, 2): round(float(iso.predict([c])[0]), 4)
                  for c in np.arange(0.3, 1.01, 0.05)},
    }
    print(f"\n[kernel] isotonic on {len(lex):,} lexical pairs, AP(cos) "
          f"{report['kernel']['ap']}")
    print("[kernel] P(syn|cos):",
          {k: v for k, v in report["kernel"]["curve"].items() if v > 0.001})

    # pos_w sparse: pairs with kernel weight >= FLOOR (chunked V×V sweep)
    lo_cos = float(np.min([c for c in np.arange(0.0, 1.001, 0.001)
                           if iso.predict([c])[0] >= FLOOR], initial=1.0))
    pi, pj, pw = [], [], []
    for i in range(0, V, 1024):
        Sblk = E2[i:i + 1024] @ E2.T
        r, c = np.nonzero(Sblk >= lo_cos)
        keep = (r + i) != c
        r, c = r[keep], c[keep]
        pi.append(r + i)
        pj.append(c)
        pw.append(iso.predict(Sblk[r, c]).astype(np.float16))
    pos_i = np.concatenate(pi)
    pos_j = np.concatenate(pj)
    pos_v = np.concatenate(pw)
    report["kernel"]["nnz"] = int(len(pos_i))
    report["kernel"]["avg_members_per_class"] = round(float(len(pos_i)) / V, 2)
    print(f"[kernel] pos_w nnz {len(pos_i):,} "
          f"({report['kernel']['avg_members_per_class']}/class; cos floor {lo_cos:.3f})")

    # =========================================================================
    # 2. ALSO-TRUE ESTIMATOR — logistic + Elkan-Noto, then dense sweep
    # =========================================================================
    from sklearn.linear_model import LogisticRegression

    p_, st, ct = coannotations(rels, img)
    n_box = len(st)
    perm = rs.permutation(n_box)
    f, d = int(0.6 * n_box), int(0.8 * n_box)
    P = {k: pairs_from(p_, st[v], ct[v])
         for k, v in [("fit", perm[:f]), ("dev", perm[f:d]), ("test", perm[d:])]}
    C_fit = sym_counts(P["fit"], V)
    print(f"\n[alsotrue] {n_box:,} multi-annotated box pairs "
          f"({len(P['fit']):,}/{len(P['dev']):,}/{len(P['test']):,} events)")

    # Reverse-pair co-annotation: B annotated on (img,o,s) when A is on (img,s,o).
    # High for INVERSES ("A left of B" ⇒ "B right of A") and symmetric predicates,
    # ~0 otherwise. Its job is the antonym VETO the old tau_ctx_floor provided:
    # the ctx feature is antonym-blind (left/right ctx 0.892), and without a
    # counter-signal the estimator marks direction-opposites "also true" — the one
    # error class that destroys direction learning. Expected NEGATIVE coefficient.
    key_all = (img * 1024 + rels[:, 0].astype(np.int64)) * 1024 \
        + rels[:, 1].astype(np.int64)
    order_all = np.argsort(key_all, kind="stable")
    k_sorted = key_all[order_all]
    p_sorted = rels[order_all, 2].astype(np.int64)
    grp_start = np.searchsorted(k_sorted, key_all, side="left")
    grp_end = np.searchsorted(k_sorted, key_all, side="right")
    rev_key = (img * 1024 + rels[:, 1].astype(np.int64)) * 1024 \
        + rels[:, 0].astype(np.int64)
    r_lo = np.searchsorted(k_sorted, rev_key, side="left")
    r_hi = np.searchsorted(k_sorted, rev_key, side="right")
    C_rev = np.zeros((V, V), np.float32)
    has = np.nonzero(r_hi > r_lo)[0]
    for t in has:
        C_rev[rels[t, 2], p_sorted[r_lo[t]:r_hi[t]]] += 1.0
    del key_all, order_all, grp_start, grp_end, rev_key, r_lo, r_hi
    print(f"[alsotrue] reverse-pair co-annotation: {int(C_rev.sum()):,} events "
          f"(antonym veto feature)")

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

    def feats(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        cos = (E2[A] * E2[B]).sum(-1)
        direct = np.log((C_fit[A, B] + 1e-3) / (counts[A] + 1.0)).astype(np.float32)
        rev = np.log((C_rev[A, B] + 1e-3) / (counts[A] + 1.0)).astype(np.float32)
        ca, cb = ctx_row[A], ctx_row[B]
        ok = (ca >= 0) & (cb >= 0)
        cx = np.zeros(len(A), np.float32)
        cx[ok] = S_ctx[ca[ok], cb[ok]]
        return np.stack([direct, rev, cos, lfreq[B], cx], 1)

    def xy(split: str):
        """Fit/eval sample matched to the DEPLOYMENT distribution.

        The weight matrix is consumed row-wise: anchor A is a real GT predicate
        (frequency-weighted), column B is ANY of the V denominator columns
        (uniform). Negatives are sampled from exactly that product. Two dead ends
        led here: class_weight="balanced" calibrated p-hat to a 50% prior (build
        1: fully dense output), and matching BOTH endpoints to the positives'
        marginal — right for the ranking validation in
        estimate_false_negatives.py, wrong for calibration — left the rare-rare
        bulk off-distribution, where a negative log_freq coefficient extrapolated
        it above the floor (build 2: dense again, caught by the guard).
        """
        pos = np.unique(P[split], axis=0)
        mA = np.bincount(pos.reshape(-1), minlength=V).astype(np.float64)
        mA /= mA.sum()
        nA = rs.choice(V, size=len(pos) * a.neg_per_pos * 2, p=mA)
        nB = rs.choice(V, size=len(pos) * a.neg_per_pos * 2)          # uniform
        ng = np.stack([nA, nB], 1)
        ng = ng[ng[:, 0] != ng[:, 1]]
        sp = {tuple(sorted(t)) for t in pos}
        ng = np.array([t for t in ng if tuple(sorted(t)) not in sp])[
:len(pos) * a.neg_per_pos]
        X = np.concatenate([feats(pos[:, 0], pos[:, 1]), feats(ng[:, 0], ng[:, 1])])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(ng))])
        return X, y, pos

    Xd, yd, _ = xy("dev")
    clf = LogisticRegression(max_iter=2000).fit(Xd, yd)
    # King-Zeng case-control prior correction: the fit prior is 1/(1+neg_per_pos)
    # ~ 4.8%, but the DEPLOYMENT prior — P(random column labeled also-true for a
    # random anchor) — is computable exactly from the data: ordered companions per
    # anchor over V. Without this shift a featureless pair cannot score below the
    # sample prior, which (build 3) put the median random pair at corrected-p 2%.
    comp = float((ct.astype(np.float64) * (ct - 1.0)).sum())   # Σ k(k-1), ordered
    pi_lab = comp / (len(rels) * float(V))
    pi_smp = float(yd.mean())
    clf.intercept_ += (math.log(pi_lab / (1 - pi_lab))
                       - math.log(pi_smp / (1 - pi_smp)))
    print(f"[alsotrue] prior shift: sample {pi_smp:.4f} -> deployment "
          f"{pi_lab:.2e} (intercept {float(clf.intercept_[0]):+.2f})")
    Xt, yt, pos_t = xy("test")
    ap_t = average_precision_score(yt, clf.decision_function(Xt))
    # calibration sanity BEFORE the expensive sweep: p-hat over deployment-like
    # random pairs must be overwhelmingly below the storage floor
    sA = rs.choice(V, 1_000_000, p=np.bincount(
        np.concatenate(list(P.values())).reshape(-1), minlength=V) /
        max(np.concatenate(list(P.values())).size, 1))
    sB = rs.choice(V, 1_000_000)
    c_pre = float(clf.predict_proba(feats(pos_t[:, 0], pos_t[:, 1]))[:, 1].mean())
    p_bulk = np.clip(clf.predict_proba(feats(sA, sB))[:, 1] / max(c_pre, 1e-3), 0, 1)
    qs = {q: round(float(np.quantile(p_bulk, q)), 5)
          for q in (0.5, 0.9, 0.95, 0.99, 0.999)}
    print(f"[alsotrue] bulk EN-corrected p quantiles {qs}  (c={c_pre:.4f})")
    # Gate on what the numbers mean: p95 below the floor bounds storage at
    # about 1000 columns per class, and the bulk median has to sit at
    # background level. The effectively-removed rate is reported rather than
    # asserted, because it is a judgement call for the review table.
    assert qs[0.95] < FLOOR and qs[0.5] < FLOOR / 10, (
        f"corrected-p p95 {qs[0.95]} / median {qs[0.5]} — bulk not background, "
        "not sweeping 365M pairs with it")
    report["alsotrue_bulk_quantiles"] = qs
    # Elkan-Noto: c = E[P-hat | labeled positive], on held-out positives
    c_en = float(clf.predict_proba(feats(pos_t[:, 0], pos_t[:, 1]))[:, 1].mean())
    report["alsotrue"] = {
        "ap_test": round(float(ap_t), 4), "elkan_noto_c": round(c_en, 4),
        "coefficients": dict(zip(["direct", "rev", "cos_v2", "log_freq", "ctx"],
                                 clf.coef_[0].round(4).tolist())),
    }
    print(f"[alsotrue] test AP {ap_t:.4f}   Elkan-Noto c {c_en:.4f}   "
          f"coefs {report['alsotrue']['coefficients']}")

    # dense sweep: p_corr = min(1, p_hat/c); store log(1-p) where p >= FLOOR
    ni, nj, nl = [], [], []
    all_j = np.arange(V, dtype=np.int64)
    for i in range(V):
        X = feats(np.full(V, i, np.int64), all_j)
        p = clf.predict_proba(X)[:, 1] / c_en
        p = np.clip(p, 0.0, 1.0 - 1e-6)
        p[i] = 0.0
        keep = np.nonzero(p >= FLOOR)[0]
        ni.append(np.full(len(keep), i, np.int64))
        nj.append(keep)
        nl.append(np.log1p(-p[keep]).astype(np.float16))
        if i % 2000 == 0:
            print(f"  [alsotrue] dense sweep {i}/{V}", end="\r", flush=True)
    neg_i = np.concatenate(ni)
    neg_j = np.concatenate(nj)
    neg_lw = np.concatenate(nl)
    per_class = np.bincount(neg_i, minlength=V)
    report["alsotrue"]["nnz"] = int(len(neg_i))
    report["alsotrue"]["downweighted_per_class"] = {
        "mean": round(float(per_class.mean()), 1),
        "median": int(np.median(per_class)), "p95": int(np.percentile(per_class, 95))}
    report["alsotrue"]["frac_effectively_removed"] = round(
        float((neg_lw.astype(np.float32) < np.log(0.06)).mean()), 4)
    assert len(neg_i) < 40_000_000, (
        f"neg_lw is {len(neg_i):,} entries (~dense) — calibration is broken again; "
        "refusing to write a matrix that down-weights everything")
    print(f"\n[alsotrue] neg_lw nnz {len(neg_i):,} per-class "
          f"{report['alsotrue']['downweighted_per_class']} "
          f"(a hard cosine mask ignored 82.8 per class)")

    # =========================================================================
    # 3. SYMMETRY — beta-binomial 2-component EM over reverse-annotation rates
    # =========================================================================
    s, o, g = (rels[:, 0].astype(np.int64), rels[:, 1].astype(np.int64),
               rels[:, 2].astype(np.int64))
    fwd = ((img * 1024 + s) * 1024 + o) * 32768 + g
    bwd = ((img * 1024 + o) * 1024 + s) * 32768 + g
    has_rev = np.isin(bwd, fwd)
    n_g = np.bincount(g, minlength=V).astype(np.float64)
    k_g = np.bincount(g, weights=has_rev, minlength=V)
    obs = n_g >= 20                       # enough trials for the likelihood to speak
    # EM on binomial mixture with per-component rate (moment beta approx is overkill)
    r1, r2, w1 = 0.001, 0.05, 0.9
    for _ in range(200):
        l1 = np.where(obs, k_g * np.log(r1) + (n_g - k_g) * np.log1p(-r1), 0)
        l2 = np.where(obs, k_g * np.log(r2) + (n_g - k_g) * np.log1p(-r2), 0)
        m = np.maximum(l1, l2)
        z = (w1 * np.exp(l1 - m)) / (w1 * np.exp(l1 - m)
                                     + (1 - w1) * np.exp(l2 - m) + 1e-300)
        z = np.where(obs, z, w1)
        r1 = float((z * k_g)[obs].sum() / ((z * n_g)[obs].sum() + 1e-9))
        r2 = float(((1 - z) * k_g)[obs].sum() / (((1 - z) * n_g)[obs].sum() + 1e-9))
        w1 = float(z[obs].mean())
        if r1 > r2:
            r1, r2, w1 = r2, r1, 1 - w1
    sym = np.where(obs, 1 - z, 0.0).astype(np.float32)   # P(high-reciprocity comp.)
    report["symmetry"] = {
        "rate_directional": round(r1, 5), "rate_reciprocal": round(r2, 5),
        "n_scored": int(obs.sum()),
        "top": {names[i]: round(float(sym[i]), 3)
                for i in np.argsort(-sym)[:30] if sym[i] > 0.5},
    }
    print(f"\n[sym] EM: directional rate {r1:.4f}, reciprocal rate {r2:.4f}; "
          f"{int((sym > 0.5).sum())} predicates lean reciprocal")

    # =========================================================================
    # 4. INVERSE EXPANSION — kernel product around the four seeds
    # =========================================================================
    ii, ij, iw = [], [], []
    for a_s, b_s in SEEDS.items():
        ia, ib = idx.get(a_s), idx.get(b_s)
        if ia is None or ib is None:
            continue
        wa = iso.predict(E2 @ E2[ia])     # P_syn(·, seed_a)
        wb = iso.predict(E2 @ E2[ib])
        wa[ia] = wb[ib] = 1.0
        ra = np.nonzero(wa >= FLOOR)[0]
        rb = np.nonzero(wb >= FLOOR)[0]
        for x in ra:
            w = wa[x] * wb[rb]
            k = rb[w >= FLOOR]
            ii.append(np.full(len(k), x, np.int64))
            ij.append(k)
            iw.append((wa[x] * wb[k]).astype(np.float16))
    inv_i = np.concatenate(ii)
    inv_j = np.concatenate(ij)
    inv_v = np.concatenate(iw)
    report["inverse"] = {"nnz": int(len(inv_i)),
                        "n_strong": int((inv_v.astype(np.float32) > 0.5).sum())}
    print(f"[inv] kernel expansion: {len(inv_i):,} weighted pairs "
          f"({report['inverse']['n_strong']} with w>0.5; table had 455)")

    # =========================================================================
    # 5. COOC SOFT WEIGHT — P(also true | cooc-seen), held-out, EN-corrected
    # =========================================================================
    # Direct measurement on held-out co-annotations: of the pairs the fit-split
    # co-annotation table already links (proxy for "cooc-seen"), what fraction
    # recur in held-out data, corrected by the same c.
    ht = np.unique(P["test"], axis=0)
    seen_fit = C_fit[ht[:, 0], ht[:, 1]] > 0
    # rate among a matched sample of unlabeled pairs
    m2 = np.bincount(ht.reshape(-1), minlength=V).astype(np.float64)
    m2 /= m2.sum()
    ng = rs.choice(V, size=(len(ht) * 5, 2), p=m2)
    ng = ng[ng[:, 0] != ng[:, 1]]
    seen_ng = C_fit[ng[:, 0], ng[:, 1]] > 0
    p_seen = min(1.0, (seen_fit.mean() * len(ht))
                 / max(seen_fit.sum() + seen_ng.sum(), 1) / max(c_en, 1e-3))
    w_cooc = float(np.clip(1.0 - p_seen, 0.05, 1.0))
    report["w_cooc"] = round(w_cooc, 4)
    print(f"[cooc] fitted soft weight {w_cooc:.3f} (hand value was 0.3)")

    # =========================================================================
    np.savez_compressed(
        a.out,
        predicates=np.array(names),
        pos_i=pos_i.astype(np.int32), pos_j=pos_j.astype(np.int32), pos_w=pos_v,
        neg_i=neg_i.astype(np.int32), neg_j=neg_j.astype(np.int32), neg_lw=neg_lw,
        sym=sym,
        inv_i=inv_i.astype(np.int32), inv_j=inv_j.astype(np.int32), inv_w=inv_v,
        w_cooc=np.float32(w_cooc),
        kernel_cos_floor=np.float32(lo_cos),
        oracle=str(a.kernel_embeds),
)
    json.dump(report, open(str(a.out).replace(".npz", "_report.json"), "w"), indent=2)
    print(f"\n[out] wrote {a.out} "
          f"({Path(a.out).stat().st_size / 1e6:.1f} MB) + report json")

    # ---- review tables ------------------------------------------------------------
    print("\n================ REVIEW: kernel positives for the problem groups ========")
    from collections import defaultdict
    rows = defaultdict(list)
    for i_, j_, w_ in zip(pos_i, pos_j, pos_v):
        rows[i_].append((float(w_), j_))
    for p in ["in front of", "above", "behind", "near", "on", "wearing"]:
        i_ = idx[p]
        mem = sorted(rows.get(i_, []), reverse=True)[:8]
        print(f"  {p:<14s} " + "  ".join(f"{names[j]}:{w:.2f}" for w, j in mem))
    print("\n================ REVIEW: top down-weighted negatives of `on` ============")
    m_on = neg_i == idx["on"]
    o_j, o_l = neg_j[m_on], neg_lw[m_on].astype(np.float32)
    for k_ in np.argsort(o_l)[:10]:
        print(f"  on -> {names[o_j[k_]]:<24s} weight {float(np.exp(o_l[k_])):.3f}")
    print("\n================ REVIEW: top symmetry scores (with evidence) ============")
    for i_ in np.argsort(-sym)[:20]:
        if sym[i_] > 0.5:
            print(f"  {names[i_]:<26s} sym {sym[i_]:.2f}  "
                  f"n={int(n_g[i_]):>7,}  reversed={int(k_g[i_]):>6,} "
                  f"({k_g[i_] / max(n_g[i_], 1):.1%})")


if __name__ == "__main__":
    main()
