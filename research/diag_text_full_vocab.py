"""Exact, full-vocabulary text-space metrics — no subsampling anywhere.

WHY. diag_text_spaces.py subsamples the columns that need the dense VxV Gram matrix
(mean cosine over 4,000 predicates, hubness over 3,000, the random side of
auc_syn_vs_rand), because at V=19,103 that matrix is 365M entries. The samples are
ample, but the headline claims about the v1/v2/teacher comparison should not rest on
"ample". This computes the same quantities over every one of the 19,103 predicates and
all 182,451,253 off-diagonal pairs, by chunking the matmul instead of sampling it.

Exact here means:
  mean_cos, p99_cos   all C(19103,2) pairs
  hubness_skew_Nk     k-occurrence with every predicate as both query and gallery
  auc_syn_vs_rand     synonym pairs vs the FULL off-diagonal distribution, computed as
                      a rank statistic against the exact histogram rather than a sample
  nn1_same_group      every multi-member-group predicate, full gallery
  effective_dim       unchanged (already exact)

    python training/diag_text_full_vocab.py \\
        --spaces $TS/pred_embeds_dinotxt_photo.npz $TS/pred_embeds_student_photo.npz \\
                 $TS/pred_embeds_studentv2_photo.npz \\
        --tags teacher student --canon_groups $TS/canonical_groups.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

NBINS = 20_000          # cosine histogram resolution for the exact AUC / percentiles
CHUNK = 1024


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def full_pass(E: np.ndarray, k: int = 10):
    """One chunked sweep over the full Gram matrix.

    Returns the exact cosine histogram over all off-diagonal pairs, the k-occurrence
    vector, and each row's argmax — everything the sampled columns approximate.
    """
    V = len(E)
    hist = np.zeros(NBINS, dtype=np.int64)
    nk = np.zeros(V, dtype=np.int64)
    argmax = np.empty(V, dtype=np.int64)
    csum = 0.0
    for i in range(0, V, CHUNK):
        S = (E[i:i + CHUNK] @ E.T).astype(np.float32)          # [c, V]
        rows = np.arange(i, min(i + CHUNK, V))
        S[np.arange(len(rows)), rows] = -2.0                   # drop self
        argmax[rows] = S.argmax(1)
        top = np.argpartition(-S, k, axis=1)[:,:k]
        np.add.at(nk, top.reshape(-1), 1)
        off = S[S > -1.5]
        csum += float(off.sum())
        hist += np.bincount(((off + 1.0) * 0.5 * (NBINS - 1)).astype(np.int32),
                            minlength=NBINS)
    return hist, nk, argmax, csum


def hist_quantile(hist, q):
    c = np.cumsum(hist)
    i = int(np.searchsorted(c, q * c[-1]))
    return i / (NBINS - 1) * 2.0 - 1.0


def hist_auc(hist, pos: np.ndarray):
    """P(random synonym pair > random off-diagonal pair), exact in the gallery."""
    c = np.cumsum(hist).astype(np.float64)
    tot = c[-1]
    idx = np.clip(((pos + 1.0) * 0.5 * (NBINS - 1)).astype(np.int32), 0, NBINS - 1)
    below = np.where(idx > 0, c[idx - 1], 0.0)
    ties = hist[idx]
    return float(((below + 0.5 * ties) / tot).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spaces", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--canon_groups", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    groups = json.load(open(a.canon_groups))

    rows = []
    for path, tag in zip(a.spaces, a.tags):
        z = np.load(path, allow_pickle=False)
        names = [str(x) for x in z["predicates"]]
        key = [k for k in z.files if k != "predicates" and z[k].ndim == 2][0]
        E = unit(z[key].astype(np.float32))
        V = len(E)
        idx = {n: i for i, n in enumerate(names)}
        mem = defaultdict(list)
        for n in names:
            mem[groups.get(n, n)].append(n)

        hist, nk, argmax, csum = full_pass(E)
        n_off = V * (V - 1)

        syn = np.array([float(E[idx[x]] @ E[idx[y]])
                        for ms in mem.values() if len(ms) > 1
                        for i, x in enumerate(ms) for y in ms[i + 1:]])
        multi = [n for n in names if len(mem[groups.get(n, n)]) > 1]
        hit = sum(groups.get(names[argmax[idx[n]]], names[argmax[idx[n]]])
                  == groups.get(n, n) for n in multi)
        ev = np.linalg.eigvalsh(np.cov((E - E.mean(0)).T))[::-1].clip(min=0)
        m, sd = nk.mean(), nk.std() + 1e-12

        rows.append({
            "tag": tag, "V": V, "n_off_pairs": n_off,
            "mean_cos": csum / n_off,
            "p99_cos": hist_quantile(hist, 0.99),
            "effective_dim": float(ev.sum() ** 2 / (ev ** 2).sum()),
            "hubness_skew_Nk": float(((nk - m) ** 3).mean() / sd ** 3),
            "max_Nk_over_uniform": float(nk.max() / m),
            "n_syn_pairs": len(syn), "syn_mean": float(syn.mean()),
            "auc_syn_vs_rand": hist_auc(hist, syn),
            "n_nn1_probes": len(multi), "nn1_same_group": hit / len(multi),
        })
        print(f"[{tag}] done  ({n_off:,} off-diagonal pairs, "
              f"{len(multi):,} NN@1 probes, {len(syn):,} synonym pairs)")

    cols = ["mean_cos", "p99_cos", "effective_dim", "hubness_skew_Nk",
            "max_Nk_over_uniform", "syn_mean", "auc_syn_vs_rand", "nn1_same_group"]
    print(f"\n{'space':<14s}" + "".join(f"{c[:13]:>15s}" for c in cols))
    for r in rows:
        print(f"  {r['tag']:<12s}" + "".join(f"{r[c]:>15.3f}" for c in cols))

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
