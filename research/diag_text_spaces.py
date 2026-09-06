"""Score every candidate predicate text space on one vocabulary, side by side.

WHY. The student encoder was distilled to separate ANTONYMS, and it succeeded
(left/right cos 0.32 where the teacher had ~0.94). Today's measurements say it paid for
that with general geometry: cos(`above`, `in front of`) = 0.961 (two different spatial
axes, indistinguishable), `beneath` within 0.94 of `depicts` / `reflected in`, and the
`in front of` canonical group spanning 0.57-0.99. The literature calls this shape:
narrow-cone ANISOTROPY (embeddings crowded into a cone, high cosine between unrelated
items) and it is exactly what a narrow fine-tuning objective produces when it deforms
the subspace that carried general knowledge.

So the question is not "is the student bad" but "bad COMPARED TO WHAT", and we happen to
have four encoders already computed on the megasg vocabulary. This scores them on the
properties the relation loss actually depends on:

  ANTONYM     inverse pairs must stay far apart, or direction is unlearnable.
              This is the one the student was built for.
  SYNONYM     canonical-group members must be closer than random pairs, or the
              multi-positive loss is supervising noise.
  ISOTROPY    mean off-diagonal cosine and effective dimension. A narrow cone makes
              every cosine large and the argmax unstable — the regime where a
              1,972-way argmax is decided by noise.
  NEIGHBOUR   is a predicate's nearest neighbour in its own group? The single number
              closest to "will the deployed argmax land on the right meaning".
  HUBNESS     do a few predicates dominate everyone's neighbourhoods (skew of N_k)?

Also applies two cheap post-hoc isotropy fixes to every space — mean-centring and
all-but-the-top (drop the mean and the leading principal components) — because if those
recover most of the gap, the fix costs one matrix operation and no retraining.

    python training/diag_text_spaces.py \\
        --spaces runs/packed/megasg/text_space/pred_embeds_dinotxt_photo.npz \\
                 runs/packed/megasg/text_space/pred_embeds_clip_photo.npz \\
                 runs/packed/megasg/text_space/pred_embeds_siglip2_photo.npz \\
                 runs/packed/text_student/pred_embeds_student_photo.npz \\
        --tags dinotxt clip siglip2 student \\
        --canon_groups runs/packed/megasg/text_space/canonical_groups.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

# Same four seed pairs the loss uses for direction supervision.
INVERSE = {"above": "below", "to the left of": "to the right of",
           "in front of": "behind", "on top of": "beneath"}
PROBE = [("above", "in front of"), ("beneath", "depicts"), ("in front of", "before"),
         ("above", "on top of"), ("to the left of", "to the right of"),
         ("in front of", "behind"), ("near", "beside"), ("on", "resting on")]


def unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def abtt(X, k):
    """All-but-the-top: drop the mean and the top-k principal directions."""
    Y = X - X.mean(0, keepdims=True)
    if k > 0:
        U, _, _ = np.linalg.svd(Y.T @ Y)
        D = U[:,:k]
        Y = Y - (Y @ D) @ D.T
    return unit(Y)


def auc(pos, neg, n=200_000, rs=None):
    """P(random positive > random negative) — rank-based, no threshold."""
    rs = rs or np.random.RandomState(0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = rs.choice(pos, min(n, len(pos)), replace=len(pos) < n)
    b = rs.choice(neg, min(n, len(neg)), replace=len(neg) < n)
    return float((a[:, None] > b[None,:min(2000, len(b))]).mean())


def score(E, names, groups, tag, rs):
    idx = {n: i for i, n in enumerate(names)}
    V, D = E.shape
    out = {"tag": tag, "V": V, "dim": D}

    # --- isotropy -----------------------------------------------------------------
    s = rs.choice(V, min(4000, V), replace=False)
    C = E[s] @ E[s].T
    off = C[~np.eye(len(s), dtype=bool)]
    out["mean_cos_offdiag"] = float(off.mean())
    out["p99_cos_offdiag"] = float(np.percentile(off, 99))
    ev = np.linalg.eigvalsh(np.cov((E - E.mean(0)).T))[::-1].clip(min=0)
    out["top1_var_share"] = float(ev[0] / ev.sum())
    # participation ratio: how many directions the space effectively uses
    out["effective_dim"] = float(ev.sum() ** 2 / (ev ** 2).sum())

    # --- synonym vs random ---------------------------------------------------------
    mem = defaultdict(list)
    for n in names:
        mem[groups.get(n, n)].append(n)
    syn = []
    for g, ms in mem.items():
        if len(ms) < 2:
            continue
        ii = [idx[m] for m in ms]
        B = E[ii] @ E[ii].T
        syn.extend(B[np.triu_indices(len(ii), 1)].tolist())
    rnd = off[rs.choice(len(off), min(200_000, len(off)), replace=False)]
    out["n_syn_pairs"] = len(syn)
    out["syn_mean"] = float(np.mean(syn)) if syn else float("nan")
    out["rand_mean"] = float(rnd.mean())
    out["auc_syn_vs_rand"] = auc(np.array(syn), rnd, rs=rs)

    # --- antonyms ------------------------------------------------------------------
    inv = []
    for a_, b_ in INVERSE.items():
        ga, gb = mem.get(groups.get(a_, a_), []), mem.get(groups.get(b_, b_), [])
        for x in ga:
            for y in gb:
                inv.append(float(E[idx[x]] @ E[idx[y]]))
    out["n_inv_pairs"] = len(inv)
    out["inv_mean"] = float(np.mean(inv)) if inv else float("nan")
    out["auc_syn_vs_inv"] = auc(np.array(syn), np.array(inv), rs=rs) if inv else float("nan")

    # --- nearest neighbour in own group? -------------------------------------------
    multi = [n for n in names if len(mem[groups.get(n, n)]) > 1]
    q = rs.choice(multi, min(1500, len(multi)), replace=False) if multi else []
    hit = 0
    for n in q:
        sim = E @ E[idx[n]]
        sim[idx[n]] = -2
        hit += groups.get(names[int(sim.argmax())], names[int(sim.argmax())]) == \
            groups.get(n, n)
    out["nn1_same_group"] = float(hit / len(q)) if len(q) else float("nan")

    # --- hubness -------------------------------------------------------------------
    k = 10
    sub = rs.choice(V, min(3000, V), replace=False)
    S = E[sub] @ E[sub].T
    np.fill_diagonal(S, -2)
    nk = np.bincount(np.argpartition(-S, k, axis=1)[:,:k].reshape(-1),
                     minlength=len(sub))
    m, sd = nk.mean(), nk.std() + 1e-8
    out["hubness_skew_Nk"] = float(((nk - m) ** 3).mean() / sd ** 3)

    out["probes"] = {f"{x} | {y}": round(float(E[idx[x]] @ E[idx[y]]), 3)
                     for x, y in PROBE if x in idx and y in idx}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spaces", nargs="+", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--canon_groups", required=True)
    ap.add_argument("--abtt", type=int, default=3,
                    help="principal directions removed by the all-but-the-top variant")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    assert len(a.spaces) == len(a.tags)
    groups = json.load(open(a.canon_groups))
    rs = np.random.RandomState(0)

    rows = []
    for path, tag in zip(a.spaces, a.tags):
        z = np.load(path, allow_pickle=False)
        names = [str(x) for x in z["predicates"]]
        key = [k for k in z.files if k != "predicates" and z[k].ndim == 2][0]
        E = z[key].astype(np.float32)
        print(f"[{tag}] {E.shape} from {path}")
        rows.append(score(unit(E), names, groups, tag, rs))
        # post-hoc isotropy fix on the SAME vocabulary, so the delta is attributable
        rows.append(score(abtt(E, a.abtt), names, groups, f"{tag}+abtt{a.abtt}", rs))

    def show(cols, title, fmt="{:>9.3f}"):
        print(f"\n{title}")
        print(f"{'space':<18s}" + "".join(f"{c.split('|')[0]:>13s}" for c in cols))
        for r in rows:
            print(f"  {r['tag']:<16s}" + "".join(
                (fmt.format(r[c]) if isinstance(r.get(c), float) else f"{r.get(c,''):>9}")
.rjust(13) for c in cols))

    print("\n" + "=" * 92)
    show(["auc_syn_vs_inv", "inv_mean", "auc_syn_vs_rand", "nn1_same_group"],
         "DISCRIMINATION — what the loss depends on "
         "(auc_syn_vs_inv: direction; auc_syn_vs_rand + nn1: meaning)")
    show(["mean_cos_offdiag", "p99_cos_offdiag", "top1_var_share", "effective_dim",
          "hubness_skew_Nk"],
         "GEOMETRY — narrow cone? (high mean cos + low effective_dim = anisotropic)")

    print("\nPROBE COSINES (the specific pathologies)")
    keys = list(rows[0]["probes"].keys())
    print(f"{'space':<18s}" + "".join(f"{k[:22]:>24s}" for k in keys[:4]))
    for r in rows:
        print(f"  {r['tag']:<16s}" + "".join(
            f"{r['probes'].get(k, float('nan')):>24.3f}" for k in keys[:4]))
    print(f"{'space':<18s}" + "".join(f"{k[:22]:>24s}" for k in keys[4:]))
    for r in rows:
        print(f"  {r['tag']:<16s}" + "".join(
            f"{r['probes'].get(k, float('nan')):>24.3f}" for k in keys[4:]))

    if a.out:
        json.dump(rows, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
