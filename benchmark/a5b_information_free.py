"""a5b_information_free.py — informativeness with NO external reference distribution.

The PSG train marginal is a convenience, not a requirement, and it costs us something:
it forces the headline to the 56 shared predicates, which makes the open-vocabulary
`vocab=train` arm unmeasurable (73% out-of-vocabulary). Every estimator here derives
its expectation from the RUN ITSELF, so an open vocabulary is scored rather than
excluded, and nothing has to be true of PSG.

All of them share the same skeleton as the referenced version -- sum surprisal over
relations the judge called TRUE -- and differ only in whose expectation is used.

  self    -log2 p_model(pred), p_model estimated from that system's own true relations.
          This is Shannon's own setup: information is relative to the receiver's
          expectation, and a receiver who knows the model's habits learns NOTHING from
          the 500th `on`. A model that always says `on` scores 0 bits for `on`, which
          is the correct answer, not a bug. Weakness: it is self-normalised, so a
          bigger usable vocabulary buys headroom (bounded by log2 V).

  pooled  the same, but p is estimated from ALL systems' relations combined -- a shared
          yardstick derived from the comparison itself rather than from an outside
          corpus. Removes self-normalisation; the number then depends on which systems
          are in the pool, so the pool must be stated.

  mdl     total description length of the system's true-predicate stream under a
          Krichevsky-Trofimov sequential code. This is the principled version: the
          first `on` is expensive and the 500th nearly free, which is the "worthless
          because it is the 500th" intuition formalised with no reference at all. It
          also CHARGES for vocabulary size -- the KT code carries the (V-1)/2 log2 N
          redundancy automatically -- so it answers the objection that `self` rewards
          a model merely for having a big predicate list. Order-independent: the
          Dirichlet-multinomial marginal depends only on the counts.

Reported beside them, needing no judge and no reference at all: distinct true
predicates per graph, and within-image predicate entropy -- repetition is a property
of a single graph, so it is visible without pooling anything.

    python benchmark/a5b_information_free.py --run <relation_precision.json>
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict

def load_run(path: str) -> dict:
    """Read a verdicts file, transparently handling the committed.gz copies.

    runs/benchmark/a5b/ stores the raw records gzipped (6 MB -> ~126 KB), which is the
    only reason they can live in the repo at all; analysis must not need a manual
    gunzip step to use what is committed.
    """
    import gzip
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)



def kt_code_length(counts: Counter) -> float:
    """Krichevsky-Trofimov code length in bits for a multiset of symbols.

    L = -log2 [ G(V/2)/G(N+V/2) * prod_x G(c_x+1/2)/G(1/2) ], the Dirichlet(1/2)
    marginal. Exchangeable, so the answer does not depend on the order the relations
    happened to be emitted in. Exceeds N*H(empirical) by ~(V-1)/2*log2(N), which is
    exactly the price of having to describe a larger vocabulary.
    """
    n = sum(counts.values())
    v = len(counts)
    if n == 0 or v == 0:
        return 0.0
    lg = math.lgamma
    ln2 = math.log(2.0)
    ll = lg(v / 2.0) - lg(n + v / 2.0)
    ll += sum(lg(c + 0.5) - lg(0.5) for c in counts.values())
    return -ll / ln2


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if not n:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--max_rank", type=int, default=10 ** 9)
    p.add_argument("--min_count", type=int, default=1,
                   help="Merge predicate types occurring fewer than this many times "
                        "into a single RARE symbol. Every estimator here treats the "
                        "predicate STRING as an atom, so a model whose decoder emits "
                        "malformed variants (`on of`, `in on`, `painted on a`) is paid "
                        "for them as if they were distinct relations. Sweeping this "
                        "shows whether a bits gap is real diversity or a fragmented "
                        "tail; a conclusion that only holds at min_count 1 is an "
                        "artifact of the tail.")
    p.add_argument("--out", default="")
    a = p.parse_args()

    recs = [r for r in load_run(a.run)["records"] if r["rank"] < a.max_rank]
    by = defaultdict(list)
    for r in recs:
        by[r["system"]].append(r)

    # Pooled expectation: every system's true relations, one shared yardstick that
    # never leaves this run.
    raw_pool = Counter(r["claim"][1] for v in by.values() for r in v
                       if not r["control"] and r["true"] == "yes")
    # Rare types are merged GLOBALLY (on the pooled count), so the same string is
    # treated the same way for every system and the merge cannot favour one of them.
    rare = {k for k, c in raw_pool.items() if c < a.min_count}

    def canon(pr: str) -> str:
        return "<RARE>" if pr in rare else pr

    pool = Counter(canon(k) for k, c in raw_pool.items() for _ in range(c))
    npool = sum(pool.values())

    rows = {}
    for name, v in by.items():
        nimg = len({r["row"] for r in v})
        judged = [r for r in v if not r["control"]]
        true = [r for r in judged if r["true"] == "yes"]
        # A control was emitted; only its verdict is unusable. Scale rates up to the
        # graph the model actually produced.
        scale = len(v) / max(1, len(judged))

        c_self = Counter(canon(r["claim"][1]) for r in true)
        n_true = sum(c_self.values())
        h_self = entropy(c_self)
        b_self = sum(-math.log2(c / n_true) * c for _, c in c_self.items()) if n_true else 0.0
        b_pool = sum(-math.log2(pool[pr] / npool) * c for pr, c in c_self.items()) if n_true else 0.0
        mdl = kt_code_length(c_self)

        per_img = defaultdict(Counter)
        for r in true:
            per_img[r["row"]][canon(r["claim"][1])] += 1
        d_pg = sum(len(c) for c in per_img.values()) / max(1, len(per_img))
        h_pg = sum(entropy(c) for c in per_img.values()) / max(1, len(per_img))

        rows[name] = {
            "rel_per_image": len(v) / nimg,
            "precision": len(true) / max(1, len(judged)),
            "true_rel_per_image": n_true * scale / nimg,
            "vocab_used": len(c_self),
            "bits_self_per_image": b_self * scale / nimg,
            "bits_pooled_per_image": b_pool * scale / nimg,
            "bits_mdl_per_image": mdl * scale / nimg,
            "bits_mdl_per_true_rel": mdl / max(1, n_true),
            "entropy_self": h_self,
            "distinct_true_pred_per_graph": d_pg,
            "within_image_entropy": h_pg,
        }

    hdr = (f"{'system':<24}{'trueRel/img':>12}{'V':>5}{'self':>9}{'pooled':>9}"
           f"{'MDL':>9}{'MDL/rel':>9}{'distinct/gr':>12}{'H_within':>10}")
    print(f"\nReference-free information per image   (source: {a.run})")
    print(f"pooled yardstick built from {npool} true relations across "
          f"{len(by)} systems in THIS run"
          + (f"; {len(rare)} predicate types seen < {a.min_count}x merged to <RARE>"
             if rare else ""))
    print(hdr)
    print("-" * len(hdr))
    for k, s in rows.items():
        print(f"{k:<24}{s['true_rel_per_image']:>12.1f}{s['vocab_used']:>5}"
              f"{s['bits_self_per_image']:>9.1f}{s['bits_pooled_per_image']:>9.1f}"
              f"{s['bits_mdl_per_image']:>9.1f}{s['bits_mdl_per_true_rel']:>9.2f}"
              f"{s['distinct_true_pred_per_graph']:>12.1f}{s['within_image_entropy']:>10.2f}")
    print("\nself/pooled/MDL are bits of TRUE information per image; V is the number of "
          "distinct\npredicates the system actually used. MDL charges for V, so it is "
          "the one to quote when\nthe systems' vocabularies differ. The last two "
          "columns use no judge-free reference at all.")

    if a.out:
        json.dump({"source": a.run, "pool_n": npool, "per_system": rows},
                  open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
