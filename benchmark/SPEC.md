# OV-SGG: an LVIS-style benchmark for open-vocabulary relation prediction

**Working name.** A protocol + metric suite for deciding whether a VRD/SGG model has
genuine open-vocabulary power, or has merely matched a benchmark's annotation style.

---

## 1. The problem with every existing SGG benchmark

Scene-graph benchmarks share their predicate vocabulary with the corpora models train
on. VG150 test uses the same 50 predicate strings as VG150 train, so a VG150-trained
model faces **no vocabulary novelty at all** — its recall is bounded below by string
agreement, not by understanding. The field nonetheless ranks open-vocabulary methods on
exactly that number.

We quantify this per (corpus, benchmark) cell as **annotation-style overlap**: the
fraction of a training corpus's relation *instances* whose predicate string appears
verbatim in the benchmark's vocabulary. It involves zero image overlap, so it is
complementary to (not a substitute for) an image-id leakage check.

Measured (`benchmark/annotation_overlap.py`):

| training corpus | #distinct pred | vg150/test | psg/test | indoorvg/test | haystack |
|---|---|---|---|---|---|
| OURS megasg_clean | 9,848 | 49.1% | 35.3% | 45.4% | 35.3% |
| OURS vg_raw | 17,352 | 74.6% | 47.9% | 71.4% | 47.9% |
| OvSGTR vg150/train | **50** | **100.0%** | 57.3% | **95.7%** | 57.3% |

**No existing benchmark is neutral.** VG150 is 100% in-domain for a VG150-trained model
against our 49.1%; IndoorVG is 95.7% against 45.4%. The `#distinct pred` column is the
other half of the story: a 50-predicate training vocabulary cannot be asked an
open-vocabulary question at all.

**Empirical confirmation.** Micro R@50 tracks the overlap gap and nothing else:
OvSGTR beats us on IndoorVG (50-pt gap in their favour, +48%) and loses on PSG (22-pt
gap, −10%), while **every tail metric goes our way on both**. That correspondence is
the benchmark's reason to exist.

---

## 2. Five axes

LVIS contributed three ideas — federated annotation (absence ≠ negative), a
support-based rare/common/frequent split, and AP over labelled cells only. Relations
need a fourth that LVIS does not, because SGG benchmarks leak annotation style.

| axis | question it alone can answer | protocol | headline |
|---|---|---|---|
| **A1 Transfer** | does it generalise across annotation styles? | GT boxes, closed vocab, graph-constrained, on ≥3 sources of differing overlap | wR@50 + bucket split |
| **A2 Precision** | does it hallucinate rare predicates? | Haystack's explicit negatives | fAP (n_pos≥5), P-AUC |
| **A3 Open-vocab** | does it *mean* the right relation? | full training vocabulary deployed, synonym matcher at calibrated τ=0.955 | mR@50 (open-vocab) |
| **A4 Deployment** | does it survive a real detector? | SGDet on a **shared** open-vocab detector | wR@50 vs pair-recall ceiling |
| **A5 Graph quality** | is the graph *as a whole* any good? | LLM oracle, pairwise, **no GT in the prompt** | win rate, gated on controls |
| **A6 Spatial** | does it understand space, or co-occurrence? | SpatialSense: balanced adversarial true/false triples | AUC (chance = 0.5) |

Each axis is load-bearing. A1 alone is gameable by corpus match. A2 alone is invariant
to uniform score depression (fAP ranks *within* a predicate, so a model that knows a
relation but never says it still scores well). A3 alone rewards head collapse, because
under a graph constraint the argmax of a collapsed model lands on `on`/`in`/`has`, which
sit in nearly every accepted synonym set. A4 alone is bounded by the detector.

### A6 — the only axis where a wrong answer is provably wrong

A1–A4 score against annotations, so a plausible-but-unlabelled relation is punished and
a frequency prior is rewarded. SpatialSense (Yang et al., ICCV'19) inverts that:
annotators were shown an image and asked to write relations a model would get **wrong**,
producing verified NEGATIVE triples. Its test split is exactly balanced (1,379 true /
1,379 false), so chance is 50% and knowing that `on` is common buys nothing.

This axis exists because measured: **every recipe change from v41 to v43 moved A1 by
+40–47% and A6 by nothing** (all arms AUC 0.65–0.68, CIs overlapping). Recall-style
gains are vocabulary and ranking gains; without A6 the suite cannot tell those apart
from spatial understanding, and the paper would have claimed the latter.

The upstream warning applies to us too: their own **boxes-only baseline (no image at
all) scores 68.8**, within 2.5 points of the best trained model, so a good A6 is not by
itself evidence of visual reasoning — it is evidence of *not* being fooled by priors.

### A5 — the only GT-free axis

A1–A4 all compare against annotations and therefore inherit their blind spots: a
relation that is true but simply unlabelled is scored as a false positive, and a model
can be no better than the corpus it is measured against. A5 removes ground truth from
the question entirely. The judge sees the image with numbered boxes and two candidate
graphs, and is asked which is the better *description of this scene* — never a relation
at a time, because global coherence (no contradictions, salient interactions rather
than trivia, usefulness to someone who has only the graph) is exactly what per-triplet
matching cannot see. It is the closest available proxy for a downstream task without
committing to one downstream task.

An LLM oracle is only worth as much as its controls, so `llm_judge.py` enforces six:

1. **Not Gemma.** Our RA-4M supervision was generated by `gemma-4-26B-A4B-it`; a Gemma
   judge would reward our own model's output distribution. The script hard-refuses one.
   Default judge is Qwen2.5-VL (different family, different pretraining).
2. **Position bias** — every pair is judged in both orders; a verdict counts only if it
   survives the swap. `flip_rate` is the judge's noise floor.
3. **Length is measured, not equalised.** Comprehensiveness is a genuine SGG virtue — a
   model that correctly describes more of the scene is a better model — so truncating
   both graphs to a common K would define that advantage out of the measurement. Each
   model instead emits what it would deploy: every pair scoring within `rel_frac=0.7` of
   that image's best pair, capped at `max_rel=20`. The rule is **per-image relative**
   because the two models' scores sit on different scales (OvSGTR's top-pair score is
   ~0.14 at the median), so an absolute threshold would rank calibration rather than
   content, while a relative one lets each model's own confidence spread decide how much
   it says. Measured: OvSGTR emits a median of 13 relations/graph, 36% at the cap.
4. **Padding control** — 15% of comparisons pit a graph against *itself plus fabricated
   relations* (same objects, same predicate strings, on pairs the model did not assert):
   strictly longer, strictly less true. `padding_resistance` is the share preferring the
   shorter truthful graph, and it converts the judge's length preference from an
   assumption into a number. The head-to-head is additionally reported **stratified by
   which graph was longer** (`by_length`), so a win that only appears in the
   winner-was-longer stratum is visible as such.
5. **Degenerate control** — 15% of comparisons pit a graph against a *scrambled* copy of
   itself (predicates permuted across the same pairs, so length, vocabulary and object
   names are all preserved). A judge that cannot prefer the intact graph is not
   measuring relational content, and `valid: false` invalidates the head-to-head.
   Only graphs with ≥3 distinct predicates where ≥50% of slots actually move are used:
   a graph that is four copies of `on` permutes into itself, and charging the judge for
   that measures our head collapse, not its discrimination. Controls are assigned to
   feasible images *before* judging, or the sample starves — measured, only 17% of
   OvSGTR's graphs are diverse enough to scramble at all.
6. **Vocabulary breadth** — we can emit ~10k predicate strings, OvSGTR at most ~150
   (its prompt is capped at `max_text_len=512`). Both arms are run: `pack` restricts us
   to the benchmark's predicates (isolating relation *choice*), `train` deploys the full
   vocabulary (the system as shipped). A win present in both arms is not a wording win.

---

## 3. Sources, and why each is present

| source | images | role | why it cannot be dropped |
|---|---|---|---|
| **VG150 test** | 26,404 | in-domain **control** | Included *to be discounted*. It shows what a 100%-overlap cell looks like, so readers can calibrate every other number. |
| **PSG test** | 2,179 | primary transfer | Least-unfair recall source (22-pt gap). COCO images, 56 curated predicates. |
| **IndoorVG test** | 4,403 | domain shift | Indoor scenes; also the highest-overlap non-VG150 source, so it isolates *style* match from *content* shift. |
| **Haystack** | 11,368 | federated negatives | SA-1B images → **zero image overlap** with any training corpus. The only source with explicit negatives, hence the only one that can measure precision. |
| **HICO-DET test** | 9,546 | external verbs | 116 verbs, none of them our vocabulary's shape. Serves A1 (verb R@K) and A2 (290,941 federated negative cells from image-level negative captions + `no_interaction` pairs). Also carries the RF-UC 120-unseen composition split for comparison with the open-vocabulary HOI literature. |
| **SpatialSense test** | 1,920 | adversarial spatial | 2,758 balanced true/false triples over 9 spatial predicates. The only source where a wrong answer is *verified* wrong. Zero image overlap with megasg_clean or vg_raw (checked on all three of its splits); its valid split is equally unseen and is therefore the legitimate place to tune the decision threshold. |

Haystack's negatives are model-assisted and deliberately adversarial, so A2 measures
discrimination against hard negatives, **not** deployment precision. Do not convert fAP
into a claimed real-world precision, and do not compute ECE/Brier on it.

---

## 4. Metrics, and what each is for

- **R@K (micro)** — reported *only* for comparability with the literature. It is the
  metric most inflated by annotation-style overlap and should never be the headline.
- **mR@K (macro)** — per-class mean; the standard long-tail metric.
- **R@K rare/common/freq** — LVIS-style buckets at <50 / 50–500 / >500 GT relations,
  with `n_cls_*` reported so the split stays interpretable. **Head collapse cannot hide
  in a bucket split.**
- **wR@K** — IDF-weighted recall, `w_c ∝ log(N/n_c)`; one scalar grading the tail
  continuously. Noisiest of the three (tail weight costs variance monotonically), so it
  is the headline only alongside the buckets.
- **fAP / P-AUC / PDD / PDO** (A2) — federated per-predicate AP over labelled cells.
  Predicates with `n_pos < 5` are excluded from the headline mean: AP over a handful of
  positives is near-bimodal.

### OVS — the one permitted scalar (`benchmark/overall_score.py`)

An earlier revision of this spec excluded *any* aggregate score, on the grounds that
cross-source averages are not meaningful and invite corpus-match gaming. **That
objection is against an arithmetic mean of raw metrics, and it still stands.** OVS is
admitted because it is built so the objection does not apply:

1. **Chance correction before combination.** `norm = clip((x − chance)/(1 − chance))`,
   with chance *derived*, never chosen: `1/V` for recall over a V-class benchmark
   vocabulary, the dataset's positive prevalence for fAP, 0.5 for AUC. Raw averaging
   gets this badly wrong — AUC 0.66 and mR@50 0.22 are not "0.44 on average", they are
   0.32 and 0.20 above their respective floors.
2. **Harmonic mean across axes.** Minimised by imbalance, so excellence on one axis
   cannot pay for uselessness on another. Precedent: generalised zero-shot learning
   replaced the arithmetic mean of seen/unseen accuracy with the harmonic mean for
   exactly this reason.
3. **It never replaces the vector.** The per-axis components print with every score,
   and the axis table remains the headline.

Corpus match therefore *buys less* under OVS, not more: matching VG150's annotation
style lifts one A1 cell of four and nothing on A2/A4/A6, and the harmonic mean pulls
the total back toward the model's weakest axis.

**The composite spans A1, A2, A4 and A6.** A3 and A5 are measured and reported and
neither is summed. A3 cannot be run on the open-vocabulary baseline at all — its
predicate vocabulary is a single caption capped at 512 word pieces, about 150 strings,
against A3's 19,103 — so a composite containing A3 exists for one of the two models
being compared. A5 is a *pairwise* preference: the two models' win rates sum to 1, so
the cell describes the pair rather than either model, and under the chance correction
above a model that loses the comparison scores 0 on that axis, which collapses the
harmonic mean to 0 whatever the other four axes say. A4's cell is divided by the
measured pair-recall ceiling before correction, so it scores the share of recoverable
pairs recovered rather than the detector's limit.

Reported together, always: **OVS** (harmonic), **OVS_arith**, the **weakest axis**, and
**balance = OVS/OVS_arith** ∈ (0,1] — 1.0 exactly when all axes are equal, so it reads
directly as how specialised the model is.

> **OVS is comparable only between models scored on the same axis set.** Adding an axis
> changes every score. The axis set is printed with the table and stored in the json.

### Deliberately excluded
- **zR@K** (zero-shot triplet recall) — excluded by decision. It would also be
  misleading here: every test predicate *string* exists in our training vocabulary, so
  this is cross-dataset **transfer**, not zero-shot.
- **IMR and informativeness-reweighted mean recalls.** They re-weight within the *same
  closed vocabulary* and so cannot see a model emitting a correct synonym outside it —
  the central open-vocabulary failure mode. They also need an informativeness prior
  estimated from the same biased annotations they are correcting for. The bucket split
  plus IDF weighting achieves the tail sensitivity transparently and decomposably.

---

## 5. Protocol rules

1. **Graph constraint is mandatory** and must be stated. Unconstrained R@K runs 12–19
   points higher. OvSGTR's published numbers are graph-constrained — verified in
   `datasets/sgg_metrics.py:91-92`, where each pair contributes one argmax triplet
   unconditionally (`multiple_preds` is written but never read).
2. **One evaluator for all models.** Models emit a common interchange record
   (`image_id, boxes, pair indices, per-predicate scores`) and are scored by identical
   code. Never compare across two papers' metric implementations.
   The record's `label_base` states whether object labels are 0-based or reserve index 0
   for background (OvSGTR's do; ours do not). Anything that *displays* object names must
   honour it — and in a side-by-side comparison both graphs must be named from ONE
   shared label source, since they address the same numbered boxes. Getting this wrong
   is invisible to every index-matched metric and catastrophic in a judged comparison:
   a one-category shift made OvSGTR's graphs read as nonsense and produced a spurious
   99% win rate before the shared-naming rule and box-identity assert were added.
3. **Report the overlap statistic in every cell.** A number without its overlap is
   uninterpretable.
4. **Report the pair-recall ceiling for A4.** It bounds every model identically;
   measured 0.696 (PSG/YOLO-World @0.05), 0.484 (IndoorVG), 0.823 (Haystack/YOLOE).
   The detector ranking *flips by dataset*, so never assume one detector dominates.
5. **State the input contract.** Protocols differ in what the model is *given*:

   | contract | boxes | labels |
   |---|---|---|
   | PredCls (OvSGTR, Motifs, …) | GT | **GT** |
   | ours | GT | none — architecturally cannot consume them |
   | SGDet (shared) | detector | detector's predictions |

   Our model never sees object categories at inference; baselines do. This asymmetry
   **favours the baselines** and must accompany any table.

---

## 6. Known asymmetries (all favour the baseline)

- **Labels.** As above.
- **Pair coverage.** OvSGTR enumerates every N·(N−1) pair; our sampler prunes to
  `geo_budget=400`. On A2 this is visible as `coverage` (~100% vs 88.3%), and
  unsampled cells score 0. *Measured negative result: uncapping our budget to 3600
  makes results slightly **worse** (R@50 .1551→.1529), so the pruning is not what
  costs us head recall — do not re-investigate.*
- **NMS.** OvSGTR's postprocessor applies NMS at IoU 0.5 even when handed GT boxes with
  tied scores, silently suppressing them (measured: 23→21 on one PSG image). We disable
  it for supplied-box modes so both models receive an identical box set; this only
  helps them.

---

## 7. Running it

```bash
# 0. neutrality matrix (CPU, seconds)
python benchmark/annotation_overlap.py

# A1, A3: GT boxes, closed and open vocabulary
python benchmark/eval_zeroshot.py --checkpoint <snapshot>/model.pth \
    --data_roots runs/packed/vg150 runs/packed/psg runs/packed/indoorvg runs/packed/hicodet \
    --split test --graph_constraint --out_dir runs/eval/<name>
python benchmark/eval_zeroshot.py ... --open_vocab --tau_eval <tau from calibrate_match_tau.py>

# A2: explicit negatives
python benchmark/eval_haystack.py --checkpoint <snapshot>/model.pth --pack runs/packed/haystack
python benchmark/eval_hico_map.py --checkpoint <snapshot>/model.pth --pack runs/packed/hicodet

# A4: shared open-vocab detections, the pair-recall ceiling, SGDet
python benchmark/detect_boxes.py --weights <detector.pt> --pack runs/packed/psg/test --out runs/det/psg_test.npz --set_classes
python benchmark/detector_recall_ceiling.py --dataset_root runs/packed/psg --split test --det runs/det/psg_test.npz
python benchmark/eval_zeroshot_detbox.py --checkpoint <snapshot>/model.pth --dataset_root runs/packed/psg \
    --dataset_name psg --split test --det runs/det/psg_test.npz

# A5: the oracle (needs a VLM; see docs/design/a5-graph-quality-metric.md)
python benchmark/dump_relsgg_interchange.py ...      # our predictions in the interchange format
python benchmark/relation_precision.py --system ours=<ours.npz> --system baseline=<baseline.npz> --pack runs/packed/psg/test --highlight

# A6: adversarial spatial
python benchmark/eval_spatialsense.py --checkpoint <snapshot>/model.pth

# baseline (OvSGTR venv): run their model over the same packs, score with OUR evaluator
python benchmark/ovsgtr/run_ovsgtr_pack.py --pack runs/packed/psg/test --config <cfg> --checkpoint <ckpt> --out runs/ovsgtr/psg_test.npz
python benchmark/ovsgtr/eval_interchange.py --pred runs/ovsgtr/psg_test.npz --pack runs/packed/psg/test

# assemble
python benchmark/aggregate.py
python benchmark/overall_score.py
```

Inference for a non-RelSGG model runs in its own venv and communicates only through the
interchange `.npz`; see `benchmark/ovsgtr/` for the reference adapter.
