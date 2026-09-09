"""Predicate-string overlap: how much of a training corpus's relation mass lands on a
benchmark's exact predicate strings.

This is ONE COMPONENT of the shared triplet mass the report leads with (see
benchmark/SPEC.md). The headline statistic charges for the two object categories as
well -- the fraction of relation instances whose <subject category, predicate, object
category> triple the benchmark also annotates -- because a predicate string is not an
annotation: `on` between a person and a horse and `on` between a book and a table are
different acts. The triple is measured on the packed corpora, which carry the object
categories; what this script needs is only each pack's predicate histogram, which is why
it runs from meta.json alone. Read its output as the predicate component, not as the
triple.

WHY IT IS THE BENCHMARK'S MOST IMPORTANT NUMBER
-----------------------------------------------
A model trained on VG150 evaluated on VG150 shares the benchmark's 50 predicate strings
by construction, so there is NO vocabulary novelty to generalise to and micro R@K
mostly measures corpus match. This statistic makes that visible per (corpus, benchmark)
cell, so a reader can discount in-domain results instead of taking them at face value.
It involves ZERO image overlap — it is purely about annotation style, and is therefore
complementary to the image-id leakage check.

Definition: fraction of the corpus's relation INSTANCES (not distinct predicates —
mass, so the head dominates as it does in training) whose predicate string appears
verbatim in the benchmark's predicate vocabulary.

    python benchmark/annotation_overlap.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# (label, pack path). Training corpora on the rows, benchmarks on the columns.
CORPORA = [
    ("OURS megasg_clean", "runs/packed/megasg_clean/train"),
    ("OURS vg_raw", "runs/packed/vg_raw/train"),
    ("OvSGTR vg150/train", "runs/packed/vg150/train"),
]
BENCHMARKS = [
    ("vg150/test", "runs/packed/vg150/test"),
    ("psg/test", "runs/packed/psg/test"),
    ("indoorvg/test", "runs/packed/indoorvg/test"),
    ("haystack", "runs/packed/haystack/test"),
]


def predicate_mass(pack: Path):
    """{predicate string: number of relation instances} for a pack."""
    meta = json.loads((pack / "meta.json").read_text())
    names = list(meta["predicates"])
    counts = meta.get("predicate_counts")
    if counts:                      # emitted by the packer; avoids loading rels.npy
        return {str(k): int(v) for k, v in counts.items()}
    rels = np.load(pack / "rels.npy", mmap_mode="r")
    ids, n = np.unique(np.asarray(rels[:, 2]), return_counts=True)
    return {names[i]: int(c) for i, c in zip(ids, n) if i < len(names)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/benchmark/annotation_overlap.json")
    args = ap.parse_args()

    bench_vocab = {}
    for label, p in BENCHMARKS:
        meta = json.loads((Path(p) / "meta.json").read_text())
        bench_vocab[label] = set(meta["predicates"])

    rows, table = {}, []
    for clabel, cpath in CORPORA:
        mass = predicate_mass(Path(cpath))
        total = sum(mass.values())
        row = {}
        for blabel in bench_vocab:
            hit = sum(v for k, v in mass.items() if k in bench_vocab[blabel])
            row[blabel] = hit / total if total else 0.0
        rows[clabel] = {"total_relations": total,
                        # the key the report reads for the predicate component; "overlap"
                        # is kept for older readers of this file and means the same thing
                        "predicate_only": row, "overlap": row,
                        "distinct_predicates": len(mass)}
        table.append((clabel, row))

    hdr = f"{'training corpus':<24}" + "".join(f"{b:>16}" for b, _ in BENCHMARKS)
    print(hdr)
    print("-" * len(hdr))
    for clabel, row in table:
        print(f"{clabel:<24}" + "".join(f"{row[b]*100:>15.1f}%" for b, _ in BENCHMARKS))
    print()
    for clabel in rows:
        print(f"{clabel}: {rows[clabel]['total_relations']:,} relations, "
              f"{rows[clabel]['distinct_predicates']:,} distinct predicates")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
