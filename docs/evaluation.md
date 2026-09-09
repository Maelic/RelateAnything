# Evaluation

## The argument

Scene-graph benchmarks share their predicate vocabulary with the corpora models
train on. VG150's test split uses the same 50 predicate strings as its training
split, so a VG150-trained model faces **no vocabulary novelty at all**. The field
nonetheless ranks open-vocabulary methods on exactly that number.

We quantify this per (corpus, benchmark) cell as **shared triplet mass**: the
fraction of a training corpus's relation *instances* whose ⟨subject category,
predicate, object category⟩ triple the benchmark also annotates. The triple
rather than the predicate, because a predicate string is not an annotation —
`on` between a person and a horse and `on` between a book and a table are
different acts, and two corpora can agree on the string while never agreeing on
the pair it is asserted of. It involves zero image overlap, so it is
complementary to a leakage check, not a substitute for one.

| training corpus | matched on | VG150 | PSG | IndoorVG | Haystack |
|---|---|---|---|---|---|
| the released mixture (19,103 predicates) | predicate string | 53.4 % | 38.7 % | 49.5 % | 38.7 % |
| | both object categories | 44.4 % | 36.7 % | 26.8 % | 30.7 % |
| | **the whole triple** | **12.8 %** | **10.7 %** | **6.6 %** | **4.9 %** |
| VG150 train (the baseline's, 50 predicates) | predicate string | **100.0 %** | 57.3 % | **95.7 %** | 57.3 % |
| | both object categories | 100.0 % | 22.1 % | 12.3 % | 7.1 % |
| | **the whole triple** | **90.9 %** | 8.6 % | 10.4 % | 0.6 % |

**No existing benchmark is neutral, and the confound is concentrated
in-domain.** On VG150 the baseline's fine-tuning corpus reproduces 90.9 % of its
relation mass as triples the benchmark also annotates, against our 12.8 %. The
correspondence is empirical: micro R@50 tracks this statistic and the tail
metrics do not. Charging for the object names as well charges for taxonomy and
domain mismatch too, which is why the components are printed separately.

`python benchmark/annotation_overlap.py` reports the predicate-string component
(the first row of each block); the object-pair and triple components are
measured on the packed corpora and reported in the report's appendix.

## OV-SGG: six axes

The protocol, in full, is [`benchmark/SPEC.md`](../benchmark/SPEC.md).
Six axes, chosen so that no single one can be won by prior matching.

| axis | question | protocol | headline |
|---|---|---|---|
| **A1** Transfer | generalises across annotation styles? | GT boxes, closed vocab, graph-constrained, ≥3 sources of differing overlap | wR@50 + bucket split |
| **A2** Precision | hallucinates rare predicates? | Haystack's explicit negatives | fAP, P-AUC |
| **A3** Open-vocab | *means* the right relation? | full training vocabulary deployed, synonym matcher at calibrated τ | open-vocab mR@50 |
| **A4** Deployment | survives a real detector? | SGDet on a **shared** open-vocab detector | wR@50 vs pair-recall ceiling |
| **A5** Graph quality | is the graph true *and* informative? | VLM judge, one relation at a time, **no GT in the prompt**, each acceptance credited with its surprisal | true bits per image |
| **A6** Spatial | understands space, or co-occurrence? | SpatialSense balanced adversarial true/false | AUC (chance 0.5) |

Each axis is load-bearing, and each is individually gameable:

- **A1 alone** is won by corpus match.
- **A2 alone** is invariant to uniform score depression — fAP ranks *within* a
  predicate, so a model that knows a relation but never says it still scores well.
- **A3 alone** rewards head collapse: under a graph constraint the argmax of a
  collapsed model lands on `on`/`in`/`has`, which sit in nearly every accepted
  synonym set.
- **A4 alone** is bounded by the detector, not the relation model.
- **A6** is the only axis where a wrong answer is *provably* wrong — annotators
  wrote triples a model would get wrong, and the split is exactly balanced.

A6 earned its place empirically: **a sequence of recipe changes moved A1
by +40–47 % and A6 by nothing.** Recall-style gains are vocabulary and ranking
gains. Without A6 the suite could not tell those apart from spatial
understanding, and we would have claimed the latter.

The headline composite (`benchmark/overall_score.py`) spans A1, A2, A4, A5 and
A6, and reads **40.1** for the released tower against **11.8** for the baseline.
It is chance-corrected and combined as a harmonic mean, so a weak axis cannot be
averaged away, and two cells are normalised by a measured quantity instead of a
chance level: A4 by the shared detector's pair-recall ceiling, A5 by the
information in the images' own annotation. A3 is measured, reported and not
summed — it cannot be run on the baseline, whose vocabulary arrives as one
caption of about 150 strings.

Mind which axis set a composite covers, because they are not comparable. The
**model ladder** is scored on A1, A2, A4 and A6, since A5 costs one judge run
per arm and was run for the released tower only; **OVS-dev**, over A1, A2, A3
and A6, selected the recipe before A4 and A5 existed. Recomputing every
selection decision on the ladder composite changes no outcome.
**Never select a model on a composite unless the run targets its weakest
axis** — otherwise you are optimizing the aggregate rather than the deficiency.

## Running an evaluation

The packs come from the `maelic/OV-SGG-Bench` dataset repository and the
checkpoint from `maelic/relsgg-<model>` (see
[installation](installation.md#get-the-weights)); images are read from
`RA_DATASETS`.

```bash
python benchmark/eval_zeroshot.py \
  --checkpoint <snapshot>/model.pth \
  --data_roots runs/packed/psg runs/packed/vg150 runs/packed/indoorvg \
  --split test \
  --graph_constraint \
  --out_dir runs/eval/<name>
```

Open-vocabulary mode — the model is never told the benchmark's label set:

```bash
python benchmark/eval_zeroshot.py... --open_vocab --tau_eval 0.72
```

Other entry points:

| script | what |
|---|---|
| `benchmark/eval_zeroshot.py` | the main closed / open-vocabulary protocol (A1, A3) |
| `benchmark/detect_boxes.py`, `benchmark/eval_zeroshot_detbox.py`, `benchmark/eval_detboxes.py` | SGDet with a real detector (A4) |
| `benchmark/detector_recall_ceiling.py` | the pair-recall ceiling a detector imposes (A4) |
| `benchmark/eval_spatialsense.py` | adversarial spatial probe (A6) |
| `benchmark/eval_haystack.py`, `benchmark/eval_hico_map.py` | federated precision on explicit negatives (A2) |
| `benchmark/eval_decomposed.py` | the two-graph type-stratified protocol |
| `benchmark/llm_judge.py`, `benchmark/relation_precision.py` | GT-free graph quality, whole-graph and per-relation (A5) |
| `benchmark/aggregate.py`, `benchmark/overall_score.py` | the benchmark table and the OVS composite |
| `benchmark/release_gate.py` | the pre-registered ship/no-ship gate |
| `benchmark/ovsgtr/` | the OvSGTR baseline adapter (runs in its own venv, scored by our evaluator) |

## `--graph_constraint` is not optional

Under the graph constraint each pair may contribute **one** predicate to the
ranked list — the standard SGG protocol. Without it a pair contributes its whole
predicate column, and R@K inflates by **12–19 points**.

Every number in this repository's model cards, tables and README is
graph-constrained. Ours were not always: an earlier round of results was
unconstrained and therefore inflated, and comparisons against published methods
were invalid until it was fixed. If you are comparing against a baseline,
confirm which one they used.

## Metrics

- **R@K** — micro recall. Tracks vocabulary overlap; dominated by head predicates.
- **mR@K** — macro (per-predicate mean) recall. The tail metric.
- **F1@K** — harmonic mean of R@K and mR@K. A single number that neither head
  collapse nor tail-only tuning can win. Computable retroactively from existing
  per-class dumps (`benchmark/compute_f1.py`).
- **SoftR / SoftmR / SoftF1** — the open-vocabulary versions, where a prediction
  counts if it matches a ground-truth predicate through a synonym matcher at a
  calibrated cosine threshold.
- **fAP** — federated AP over labelled cells only, LVIS-style: absence is not a
  negative unless it was adjudicated.

## Detector boxes

Ground-truth boxes are a laboratory condition. With a real detector, retention
is **53–63 % on PSG and 29–32 % on IndoorVG** of the GT-box number.

The important finding is that this does **not** flatten the family: model
improvements measured with ground-truth boxes survive detector boxes at
+42–105 %. But retention
itself slid (91 % → 78 %) as models improved, so a gain measured on GT boxes
overstates the deployed gain. Report both.

The dominant term is **detector recall**, not box noise — jittering GT boxes
does not reproduce the drop. And the detector operating point is worth several
times the spread between published methods on the open-vocabulary leaderboard,
so an unreported one makes a comparison meaningless.

## Comparing against other methods

- **Use the same evaluator.** `benchmark/react_same_eval_table.py` and
  the OvSGTR harness under `benchmark/ovsgtr/` exist so baselines run through
  our scorer rather than through their reported numbers.
- **Check the matcher.** A matcher with no assignment constraint is worth as
  much as a backbone upgrade. Our OvSGTR reproduction gap closed to 0.35 points
  once duplicate-box credit was replaced by one-to-one matching.
- **Check the graph constraint** on both sides.
- **Report the detector operating point.**

Then read [pitfalls](pitfalls.md) before writing the number down.
