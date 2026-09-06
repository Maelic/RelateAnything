"""Estimate P(B also true | annotator wrote A) from data, replacing the cosine thresholds.

THE PROBLEM. Five hand-set knobs — `--tau_ignore 0.94`, `--tau_eval 0.94`,
`--hard_lo 0.85`, `--tau_dilate 0.94` (build_pair_cooc.py) and `--tau_ctx_floor 0.85` —
all answer one question: given the annotator wrote A for this pair, is B also true, so
that B must not be pushed down as a negative? Each is a threshold on text cosine, and
each is a patch for a weak synonym table (build_pair_cooc.py's own docstring: it dilates
by cosine "because canonical groups are near-identity, 10,086 groups for 10,102
predicates").

They are also not portable, which is what forces the issue. Between two text
spaces the cosine scale moves (mean off-diagonal 0.80 against 0.17), so a
threshold of 0.94 goes from ignoring 82.8 columns per class to 1.2 — removing
98.5% of the false-negative protection, a configuration that costs 33-68% of
rare-class recall. The size-matched equivalent in the second space (0.569)
selects 80% different pairs, so recalibrating per space does not preserve the
meaning either.

THE OBSERVATION THAT REPLACES THEM. Annotators sometimes wrote two different strings for
the SAME (image, subject box, object box). On megasg train: 706,454 box pairs (20.3%)
carry more than one predicate, 35.2% of all relations live on them, and they exhibit
34,270 distinct co-annotated predicate pairs. A co-annotation is a DIRECT observation
that two strings are interchangeable in context — exactly the quantity the thresholds
approximate. So estimate it, and validate the estimate on held-out co-annotations.

WHAT THIS SCRIPT DOES.
  1. Extracts co-annotations from the packed relations, and splits the CO-ANNOTATED BOX
     PAIRS (not the predicate pairs) into fit / dev / test. Splitting on box pairs is
     what makes the test honest: a predicate pair held out in test is one whose evidence
     was never counted.
  2. Builds features per predicate pair (A,B) from the fit split only:
       direct    log co-annotation rate C[A,B] / n[A]      — sparse but unambiguous
       text      cosine in whichever text space is passed   — dense, space-dependent
       context   PPMI distributional similarity             — dense, text-space-FREE
       prior     log marginal frequency of B                — popularity/hubness control
  3. Fits a logistic regression on dev (positives = dev co-annotations, negatives =
     sampled unlabeled pairs) — i.e. the mixture weights and the effective bandwidth are
     LEARNED, not set. Output is a calibrated probability, so the loss can use a
     continuous weight w = 1 - P instead of a binary mask.
  4. Reports on test, against the incumbent rule (cos >= tau) on the SAME test pairs:
     average precision, ROC-AUC, and the precision/recall the threshold actually
     achieves.

TWO CAVEATS, both real.
  * POSITIVE-UNLABELED. Absence of co-annotation is not evidence of falsity — annotators
    do not label exhaustively. Sampled "negatives" therefore contain unlabeled positives,
    which biases the intercept (and hence absolute calibration) but leaves the RANKING
    largely intact. AP/AUC comparisons here are sound; the absolute probabilities are an
    upper bound on precision and should be Elkan-Noto corrected before being trusted as
    calibrated weights.
  * This estimates the MARGINAL P(B|A), which is the right shape: the loss's ignore mask
    is [V, V], per-class not per-instance. Per-instance context is already handled
    separately by the cooc soft-mask in BatchLocalInfoNCE.

    python training/estimate_false_negatives.py \\
        --data_root runs/packed/megasg --split train \\
        --pred_embeds runs/packed/datamix_v22/text_space/pred_embeds_student_photo.npz \\
        --pred_context runs/packed/datamix_v22/text_space/pred_context.npz \\
        --tau 0.94 --out runs/benchmark/false_negative_estimator.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_pack(root: Path, split: str):
    meta = json.load(open(root / split / "meta.json"))
    rels = np.load(root / split / "rels.npy")
    im = np.load(root / split / "img_meta.npy")
    preds = meta["predicates"]
    # rels columns: [sub_box_local, obj_box_local, predicate, ?, ?]; verified by range.
    assert rels[:, 2].max() < len(preds), "predicate column is not col 2 for this pack"
    img = np.repeat(np.arange(len(im)), im[:, 6])
    assert len(img) == len(rels), (len(img), len(rels))
    return preds, rels, img


def coannotations(rels: np.ndarray, img: np.ndarray):
    """Group relations by (image, subject box, object box).

    Returns the sorted predicate array plus each group's [start, count), so callers can
    split on BOX PAIRS rather than on predicate pairs.
    """
    key = (img.astype(np.int64) << 40) | (rels[:, 0].astype(np.int64) << 20) \
        | rels[:, 1].astype(np.int64)
    order = np.argsort(key, kind="stable")
    k, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
    p = rels[order, 2].astype(np.int64)
    keep = cnt > 1
    return p, start[keep], cnt[keep]


def pairs_from(p, start, cnt):
    """Unordered predicate pairs co-annotated on the given box-pair groups."""
    out = []
    for s, c in zip(start, cnt):
        u = np.unique(p[s:s + c])
        if len(u) > 1:
            i, j = np.triu_indices(len(u), 1)
            out.append(np.stack([u[i], u[j]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), np.int64)


def sym_counts(pairs: np.ndarray, V: int):
    """Symmetric co-annotation count matrix as a dense [V, V] float32."""
    C = np.zeros((V, V), np.float32)
    if len(pairs):
        np.add.at(C, (pairs[:, 0], pairs[:, 1]), 1.0)
        C += C.T
    return C


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", nargs="+", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--pred_embeds", nargs="+", required=True,
                    help="one or more text spaces; each is scored separately so the "
                         "estimator's space-dependence is visible")
    ap.add_argument("--tags", nargs="+", default=None)
    ap.add_argument("--pred_context", default="", help="PPMI npz from build_predicate_context.py")
    ap.add_argument("--tau", type=float, default=0.94, help="incumbent threshold to beat")
    ap.add_argument("--neg_per_pos", type=int, default=20)
    ap.add_argument("--neg_sampling", default="freq_matched",
                    choices=["freq_matched", "uniform"],
                    help="uniform makes the task solvable by popularity alone; "
                         "freq_matched is the honest setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rs = np.random.RandomState(a.seed)

    # ---- co-annotations, split on BOX PAIRS -------------------------------------
    names0, rels, img = load_pack(Path(a.data_root[0]), a.split)
    p, start, cnt = coannotations(rels, img)
    n_box = len(start)
    perm = rs.permutation(n_box)
    f, d = int(0.6 * n_box), int(0.8 * n_box)
    splits = {"fit": perm[:f], "dev": perm[f:d], "test": perm[d:]}
    V0 = len(names0)
    print(f"[data] {len(rels):,} relations, {n_box:,} multi-annotated box pairs "
          f"-> fit {f:,} / dev {d - f:,} / test {n_box - d:,}")

    P = {k: pairs_from(p, start[v], cnt[v]) for k, v in splits.items()}
    for k, v in P.items():
        print(f"  {k:>4s}: {len(v):,} co-annotation events, "
              f"{len(np.unique(v, axis=0)):,} distinct predicate pairs")

    C_fit = sym_counts(P["fit"], V0)
    n_pred = np.bincount(rels[:, 2], minlength=V0).astype(np.float32)

    rows = []
    tags = a.tags or [Path(s).stem for s in a.pred_embeds]
    for path, tag in zip(a.pred_embeds, tags):
        z = np.load(path, allow_pickle=False)
        enames = [str(x) for x in z["predicates"]]
        key = [k for k in z.files if k != "predicates" and z[k].ndim == 2][0]
        E = z[key].astype(np.float32)
        E /= np.linalg.norm(E, axis=-1, keepdims=True) + 1e-8
        # INDEX SPACES: pack ids != embedding-file ids. Map by NAME, never by position.
        eidx = {n: i for i, n in enumerate(enames)}
        pack2emb = np.array([eidx.get(n, -1) for n in names0], np.int64)
        n_missing = int((pack2emb < 0).sum())

        ctx = None
        if a.pred_context:
            # THREE index spaces here, and they are all different sizes. The file stores
            # `predicates` = all 19,103 UNION names, `pred_ids` = the 814 union indices
            # that had enough support, and `ctx_sim` = an [814, 814] SIMILARITY MATRIX
            # (not embeddings) indexed by position within pred_ids. Map
            # pack name -> union id -> ctx_sim row, by NAME at the first hop.
            zc = np.load(a.pred_context, allow_pickle=False)
            cnames = [str(x) for x in zc["predicates"]]
            pid = zc["pred_ids"].astype(np.int64)
            S = zc["ctx_sim"].astype(np.float32)
            assert S.shape == (len(pid), len(pid)), (S.shape, len(pid))
            name2row = {cnames[u]: r for r, u in enumerate(pid)}
            pack2ctx = np.array([name2row.get(n, -1) for n in names0], np.int64)
            ctx = (S, pack2ctx)
            print(f"[ctx] {len(pid)} predicates with context vectors; "
                  f"{int((pack2ctx >= 0).sum())}/{len(names0)} of this pack covered")

        def feats(pr: np.ndarray) -> np.ndarray:
            A, B = pr[:, 0], pr[:, 1]
            ea, eb = pack2emb[A], pack2emb[B]
            ok = (ea >= 0) & (eb >= 0)
            cos = np.zeros(len(pr), np.float32)
            cos[ok] = (E[ea[ok]] * E[eb[ok]]).sum(-1)
            direct = np.log((C_fit[A, B] + 1e-3) / (n_pred[A] + 1.0))
            prior = np.log(n_pred[B] + 1.0)
            cols = [direct, cos, prior]
            if ctx is not None:
                S_, m = ctx
                ca, cb = m[A], m[B]
                okc = (ca >= 0) & (cb >= 0)
                cs = np.zeros(len(pr), np.float32)
                cs[okc] = S_[ca[okc], cb[okc]]      # precomputed similarity, not a dot
                cols.append(cs)
            return np.stack(cols, 1)

        def sample_neg(n: int, pos: np.ndarray) -> np.ndarray:
            """Negatives matched to the positives' MARGINAL predicate frequency.

            Uniform negatives would make this task trivially solvable by popularity:
            co-annotated pairs are, by construction, pairs of predicates frequent enough
            to be annotated twice, while a uniform draw over 10,102 predicates is almost
            always two rare ones. Drawing each endpoint from the positives' own marginal
            destroys the pairing while preserving frequency, so any remaining signal is
            about THIS pair being interchangeable, not about either being common.
            """
            if a.neg_sampling == "uniform":
                x = rs.randint(0, V0, size=(n * 2, 2))
            else:
                marg = np.bincount(pos.reshape(-1), minlength=V0).astype(np.float64)
                marg /= marg.sum()
                x = rs.choice(V0, size=(n * 2, 2), p=marg)
            x = x[x[:, 0] != x[:, 1]][:n]
            seen = {tuple(t) for t in np.unique(pos, axis=0)}
            keep = np.array([tuple(sorted(t)) not in seen for t in x], bool)
            return x[keep]

        def xy(split: str):
            pos = np.unique(P[split], axis=0)
            neg = sample_neg(len(pos) * a.neg_per_pos, pos)
            X = np.concatenate([feats(pos), feats(neg)])
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            return X, y, pos, neg

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, roc_auc_score

        Xd, yd, _, _ = xy("dev")
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xd, yd)
        Xt, yt, pos_t, neg_t = xy("test")
        s = clf.decision_function(Xt)
        ap_est = average_precision_score(yt, s)
        auc_est = roc_auc_score(yt, s)

        # incumbent rule on the SAME test pairs: cos >= tau, as a binary predictor
        cos_t = Xt[:, 1]
        ap_tau = average_precision_score(yt, cos_t)
        auc_tau = roc_auc_score(yt, cos_t)
        pred = cos_t >= a.tau
        tp = float((pred & (yt > 0)).sum())
        prec = tp / max(pred.sum(), 1)
        rec = tp / max((yt > 0).sum(), 1)
        # precision the estimator reaches at the SAME recall
        thr = np.quantile(s[yt > 0], 1 - rec) if rec > 0 else np.inf
        pe = s >= thr
        prec_est = float((pe & (yt > 0)).sum()) / max(pe.sum(), 1)

        fname = ["direct", "text_cos", "log_freq"] + (["ctx_sim"] if ctx else [])
        # Single-feature ablation: which signal is actually carrying the estimate, and
        # is any of it just popularity? Fit on dev, score on test, same splits.
        abl = {}
        for i, nm in enumerate(fname):
            c1 = LogisticRegression(max_iter=2000, class_weight="balanced")
            c1.fit(Xd[:, [i]], yd)
            abl[nm] = round(float(average_precision_score(
                yt, c1.decision_function(Xt[:, [i]]))), 4)
        # coverage: how much of the test set can the context feature even see?
        cov = None
        if ctx is not None:
            _, m = ctx
            cov = float(((m[pos_t[:, 0]] >= 0) & (m[pos_t[:, 1]] >= 0)).mean())

        row = {"space": tag, "n_missing_from_space": n_missing,
               "ap_single_feature": abl, "ctx_coverage_of_test_pos": cov,
               "neg_sampling": a.neg_sampling,
               "ap_estimator": ap_est, "auc_estimator": auc_est,
               "ap_cosine_only": ap_tau, "auc_cosine_only": auc_tau,
               f"precision_at_tau{a.tau}": prec, f"recall_at_tau{a.tau}": rec,
               "precision_estimator_at_same_recall": prec_est,
               "coefficients": dict(zip(fname, clf.coef_[0].round(4).tolist()))}
        rows.append(row)
        print(f"\n=== {tag} ===")
        print(f"  predicates absent from this space: {n_missing}")
        print(f"  estimator     AP {ap_est:.4f}   AUC {auc_est:.4f}")
        print(f"  cosine alone  AP {ap_tau:.4f}   AUC {auc_tau:.4f}")
        print(f"  cos >= {a.tau}: precision {prec:.4f}  recall {rec:.4f}")
        print(f"  estimator at that same recall: precision {prec_est:.4f} "
              f"({prec_est / max(prec, 1e-9):.2f}x)")
        print(f"  learned weights: {row['coefficients']}")
        print(f"  single-feature AP: {abl}")
        if cov is not None:
            print(f"  context feature covers {cov:.1%} of test positives")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
