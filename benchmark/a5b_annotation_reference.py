"""a5b_annotation_reference.py — put A5b's bits on a [0,1] scale.

`a5b_information.py` reports bits of true information per image
(sum of -log2 p_ref over the relations the judge accepted). That is the right
quantity to compare two systems with, but it has no ceiling, so it cannot be a
spoke on a chance-corrected radar where 1.0 means perfect.

The natural ceiling is the annotation itself. Scoring the ground-truth relations
of the SAME images under the SAME reference distribution gives a number in the
same units, and the ratio reads directly: what share of the information a human
annotator recorded did the system deliver. Floor is ~0 (a system whose claims
the judge rejects earns nothing); 1.0 is "as informative as the annotation".

Two properties worth stating, because both bound the number:
  * The ground truth counts every annotated relation, including those whose
    endpoints the shared detector never found. Both systems are penalised by
    detector recall identically (A4 measures that separately), so the ceiling is
    unreachable in detection mode by construction. This is deliberate: the
    alternative, restricting the ground truth to recovered pairs, would make the
    denominator depend on the detector's operating point.
  * The reference marginal and the ground truth are both PSG, so an `on`-heavy
    annotation is priced cheaply on both sides of the ratio.

    python benchmark/a5b_annotation_reference.py \\
        --run runs/benchmark/a5b/relation_precision_psg_top10.json.gz \\
        --info runs/benchmark/a5b/info_ref_top10_matched.json \\
        --pack runs/packed/psg/test --ref runs/packed/psg/train/meta.json \\
        --out runs/benchmark/a5b/annotation_reference_top10_matched.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def _load(path: str):
    p = Path(path)
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as fh:
        return json.load(fh)


def surprisal(ref_meta: str) -> tuple[np.ndarray, list[str]]:
    """Bits per predicate under the reference split's own marginal."""
    m = json.load(open(ref_meta))
    names = list(m["predicates"])
    counts = m["predicate_counts"]
    c = np.array([counts.get(n, 0) for n in names], dtype=float)
    if c.sum() <= 0:
        raise SystemExit(f"{ref_meta} carries no predicate counts")
    p = c / c.sum()
    return -np.log2(np.clip(p, 1e-12, None)), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True,
                    help="relation_precision json(.gz); only its image rows are read")
    ap.add_argument("--info", required=True, help="the a5b_information.py output to normalise")
    ap.add_argument("--pack", required=True, help="the pack the run scored (test split)")
    ap.add_argument("--ref", required=True, help="meta.json of the reference split (train)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    surp, names = surprisal(a.ref)
    pack = Path(a.pack)
    pmeta = json.load(open(pack / "meta.json"))
    if list(pmeta["predicates"]) != names:
        raise SystemExit("pack and reference disagree on the predicate index space")

    # The judged images, not the whole split: the ratio must have one denominator.
    rows = sorted({r["row"] for r in _load(a.run)["records"]})
    img = np.load(pack / "img_meta.npy")
    rels = np.load(pack / "rels.npy", mmap_mode="r")
    bits = n_rel = 0.0
    for r in rows:
        r0, nr = int(img[r][5]), int(img[r][6])
        pred = np.asarray(rels[r0:r0 + nr, 2], dtype=np.int64)
        bits += float(surp[pred].sum())
        n_rel += len(pred)
    n = len(rows)
    gt_bits = bits / n

    info = _load(a.info)["per_system"]
    out = {"run": a.run, "info": a.info, "pack": str(pack), "ref": a.ref,
           "n_images": n, "gt_rel_per_image": n_rel / n, "gt_bits_per_image": gt_bits,
           "per_system": {}}
    for sysname, v in info.items():
        b = v.get("true_bits_per_image")
        # bits_comparable is false when the system's predicates fall outside the
        # reference vocabulary; a share computed from a biased subsample is worse
        # than no share at all.
        ok = v.get("bits_comparable", True) and b is not None
        out["per_system"][sysname] = {
            "true_bits_per_image": b,
            "share_of_annotation": (b / gt_bits) if ok else None}

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"{n} images: ground truth {n_rel/n:.2f} rel/img, {gt_bits:.1f} bits/img")
    for k, v in out["per_system"].items():
        sh = "n/c" if v["share_of_annotation"] is None else f"{v['share_of_annotation']:.3f}"
        print(f"  {k:24} {v['true_bits_per_image'] or float('nan'):5.1f} bits   share {sh}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
