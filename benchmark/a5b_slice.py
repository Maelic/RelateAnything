"""a5b_slice.py — re-read an A5b per-relation run at a shallower graph depth.

WHY THIS IS SOUND, and where it is not. `relation_precision.py` ranks each image's
pairs by score and stamps every judged relation with its `rank`, so restricting to
rank < K reproduces exactly the graph that `--max_rel K --rel_frac 0.0` would have
emitted -- PROVIDED the original run's rel_frac did not already cut above K. It did
not for us (16.2 rel/img at cap 20), so our top-10 slice is exact. It bit slightly
into OvSGTR (9.68 rel/img under rel_frac 0.7 vs 9.8 at rel_frac 0.0), so a handful of
their rank<10 relations are missing here -- ~1% of their relations, and they are the
LOWEST-scoring ones, i.e. the slice flatters them a shade. Treat this as a fast read;
the reported number should come from a run actually configured at that depth.

WHY DEPTH MATTERS AT ALL. The deployed-graph comparison is confounded: ours emits ~16
relations, OvSGTR ~10, and `useful_per_image` rewards saying more. Equalising depth
removes that lever entirely -- at matched K, useful_per_image is just precision x K --
so the slice isolates PER-RELATION quality. It does not supersede the deployed number;
it answers a different question, and both belong in the paper.

Modal share is printed alongside because truth alone cannot see the `on` problem: a
model emitting one predicate for 85% of its relations scores well on precision and
says almost nothing. That is what the info axis prices, and what modal share exposes
judge-free.

    python benchmark/a5b_slice.py --run runs/judge/relation_precision_psg.json --max_rank 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.relation_precision import GENERIC_PREDICATES   # noqa: E402

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



def summarize(recs: list[dict], emitted: list[dict]) -> dict:
    """Rates over `recs` (non-control); graph SIZE over `emitted` (all, incl. controls).

    A control relation was emitted by the model and then corrupted before judging, so
    it counts toward how much the model SAYS while its verdict says nothing about the
    model. Rating it as a graph member but not as a verdict is the only combination
    that is unbiased in both. relation_precision.py's own summary drops controls from
    both, which leaves its rel_per_image -- and hence useful_per_image -- low by the
    control fraction (~12%): 16.2 rel/img reported where the graph is really 18.3.
    Since the factor is common to every system it never changed a ranking, but the
    absolute number belongs in the paper corrected."""
    n = len(recs)
    if not n:
        return {}
    imgs = {r["row"] for r in recs} | {r["row"] for r in emitted}
    yes = sum(r["true"] == "yes" for r in recs)
    info = [int(r["info"]) for r in recs]
    useful = sum(r["true"] == "yes" and int(r["info"]) >= 2 for r in recs)
    preds = Counter(r["claim"][1] for r in recs)
    per_img_pred = [len({x[1] for x in v}) for v in
                    _by_image(recs).values()]
    rel_per_image = len(emitted) / len(imgs)
    return {
        "n_judged": n,
        "n_emitted": len(emitted),
        "n_images": len(imgs),
        "rel_per_image": rel_per_image,
        "rel_per_image_noncontrol": n / len(imgs),
        "precision": yes / n,
        "mean_info": sum(info) / n,
        "mean_info_given_true": (sum(i for r, i in zip(recs, info) if r["true"] == "yes")
                                 / yes) if yes else 0.0,
        "useful_rate": useful / n,
        "useful_per_image": (useful / n) * rel_per_image,
        "useful_per_image_asreported": useful / len(imgs),
        "modal_predicate": preds.most_common(1)[0][0],
        "modal_share": preds.most_common(1)[0][1] / n,
        "distinct_predicates": len(preds),
        "distinct_pred_per_graph": sum(per_img_pred) / len(per_img_pred),
    }


def _by_image(recs):
    d = defaultdict(list)
    for r in recs:
        d[r["row"]].append(r["claim"])
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--max_rank", type=int, default=10)
    p.add_argument("--out", default="")
    a = p.parse_args()

    d = load_run(a.run)
    recs = d["records"]
    src = d.get("summary", {})

    # Controls are corrupted claims; they measure the JUDGE, not a system, and are
    # pooled across systems -- keep them out of every per-system number, but keep
    # reporting the gate, since a slice inherits the run's judge quality unchanged.
    at_depth = [r for r in recs if r["rank"] < a.max_rank]
    by_sys, emitted = defaultdict(list), defaultdict(list)
    for r in at_depth:
        emitted[r["system"]].append(r)
        if not r.get("control"):
            by_sys[r["system"]].append(r)

    # The control is depth-dependent, so a slice must recompute it rather than
    # inherit the parent run's: generic predicates concentrate at high rank, and a
    # corruption landing on one is often still true. Gate on specific replacements.
    ctl = [r for r in recs if r.get("control") and r["true"] and r["rank"] < a.max_rank]
    spec = [r for r in ctl if r["claim"][1] not in GENERIC_PREDICATES]
    fa_pool = sum(r["true"] == "yes" for r in ctl) / max(1, len(ctl))
    fa_spec = sum(r["true"] == "yes" for r in spec) / max(1, len(spec))

    out = {"source": a.run, "max_rank": a.max_rank,
           "false_accept": fa_pool, "false_accept_specific": fa_spec,
           "control_n": len(ctl), "control_n_specific": len(spec),
           "valid": bool(len(spec) >= 30 and fa_spec <= 0.25),
           "per_system": {k: summarize(v, emitted[k]) for k, v in by_sys.items()}}

    hdr = (f"{'system':<24}{'rel/img':>9}{'prec':>8}{'mInfo':>8}"
           f"{'useful/img':>12}{'modal':>10}{'mShare':>8}{'pred/graph':>12}")
    print(f"\nA5b at depth <= top-{a.max_rank}   (source: {a.run})")
    print(hdr)
    print("-" * len(hdr))
    for k, s in out["per_system"].items():
        print(f"{k:<24}{s['rel_per_image']:>9.1f}{s['precision']:>8.3f}"
              f"{s['mean_info']:>8.2f}{s['useful_per_image']:>12.2f}"
              f"{s['modal_predicate']:>10}{s['modal_share']:>8.2f}"
              f"{s['distinct_pred_per_graph']:>12.1f}")
    print("rel/img and useful/img count controls as emitted relations; rates do not "
          "(see summarize).")
    print(f"judge control AT THIS DEPTH: false_accept {out['false_accept']:.3f} pooled "
          f"(n={out['control_n']}) / {out['false_accept_specific']:.3f} specific "
          f"(n={out['control_n_specific']})   valid {out['valid']}")
    print("  gate = specific subset; a corruption landing on a generic predicate is "
          "often still true, and generic predicates concentrate at high rank.")

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
