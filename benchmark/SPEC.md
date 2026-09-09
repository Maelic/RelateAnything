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

We quantify this per (corpus, benchmark) cell as **shared triplet mass**: the fraction
of a training corpus's relation *instances* whose ⟨subject category, predicate, object
category⟩ triple the benchmark also annotates. The triple rather than the predicate,
because a predicate string is not an annotation: `on` between a person and a horse and
`on` between a book and a table are different acts, and two corpora can agree on the
string while never agreeing on the pair it is asserted of. It involves zero image
overlap, so it is complementary to (not a substitute for) an image-id leakage check.

| training corpus | matched on | VG150 | PSG | IndoorVG | Haystack |
|---|---|---|---|---|---|
| the released mixture (19,103 predicates) | predicate string | 53.4 % | 38.7 % | 49.5 % | 38.7 % |
| | both object categories | 44.4 % | 36.7 % | 26.8 % | 30.7 % |
| | **the whole triple** | **12.8 %** | **10.7 %** | **6.6 %** | **4.9 %** |
| VG150 train (the baseline's, 50 predicates) | predicate string | **100.0 %** | 57.3 % | **95.7 %** | 57.3 % |
| | both object categories | 100.0 % | 22.1 % | 12.3 % | 7.1 % |
| | **the whole triple** | **90.9 %** | 8.6 % | 10.4 % | 0.6 % |

**No existing benchmark is neutral, and the confound is concentrated in-domain.** On
VG150 the baseline's fine-tuning corpus reproduces 90.9% of its relation mass as triples
the benchmark also annotates, against our 12.8%; on PSG the two are 8.6% and 10.7%. A
50-predicate training vocabulary is also the other half of the story: it cannot be asked
an open-vocabulary question at all.

**Empirical confirmation.** Micro R@50 tracks this statistic and the tail metrics do
not: OvSGTR beats us on IndoorVG micro recall and loses on PSG, while **every tail
metric goes our way on both**. That correspondence is the benchmark's reason to exist.

`benchmark/annotation_overlap.py` reports the predicate-string component; the object-pair
and triple components come from the same packs and are reported in the report.

---

## 2. Six axes

LVIS contributed three ideas — federated annotation (absence ≠ negative), a
support-based rare/common/frequent split, and AP over labelled cells only. Relations
need a fourth that LVIS does not, because SGG benchmarks leak annotation style.

| axis | question it alone can answer | protocol | headline |
|---|---|---|---|
| **A1 Transfer** | does it generalise across annotation styles? | GT boxes, closed vocab, graph-constrained, on ≥3 sources of differing overlap | wR@50 + bucket split |
| **A2 Precision** | does it hallucinate rare predicates? | Haystack's explicit negatives | fAP (n_pos≥5), P-AUC |
| **A3 Open-vocab** | does it *mean* the right relation? | full training vocabulary deployed, synonym matcher at the τ calibrated for the shipped text space (0.72) | mR@50 (open-vocab) |
| **A4 Deployment** | does it survive a real detector? | SGDet on a **shared** open-vocab detector | wR@50 vs pair-recall ceiling |
| **A5 Graph quality** | is the graph true *and* informative? | VLM judge, one relation at a time, **no GT in the prompt** | true bits per image, gated on controls |
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

This axis exists because it was measured to be independent of the others: over a
sequence of recipe changes A1 moved by +40–47% while A6 moved by nothing (every
arm between AUC 0.65 and 0.68, confidence intervals overlapping). Recall-style
gains are vocabulary and ranking gains; without A6 the suite cannot tell them
apart from spatial understanding.

The upstream warning applies to us too: their own **boxes-only baseline (no image at
all) scores 68.8**, within 2.5 points of the best trained model, so a good A6 is not by
itself evidence of visual reasoning — it is evidence of *not* being fooled by priors.

### A5 — the only GT-free axis

A1–A4 all compare against annotations and therefore inherit their blind spots: a
relation that is true but simply unlabelled is scored as a false positive, and a model
can be no better than the corpus it is measured against. A5 removes ground truth from
the question entirely.

**What is scored.** The judge is shown the image with the subject and object boxes drawn,
**one relation at a time**, and asked only whether the claim holds for that photograph.
Truth alone is not the axis, because truth alone favours `on`, which is almost always
true and almost never informative: a system answering `on` everywhere would be accepted
more often than either model here. Each accepted relation is therefore credited with its
surprisal under a reference distribution,

    I = Σ over relations the judge accepted of  −log₂ p_ref(predicate)

with `p_ref` the PSG **training** marginal restricted to the predicates both systems can
emit, so no smoothing constant influences the comparison and neither factor can be
increased by repetition. `on` is priced at 2.31 bits against 7.04 for `riding`.

**Why not the whole-graph comparison.** An earlier revision of this spec scored A5 as a
pairwise preference over whole graphs. That verdict turns out to depend on graph length:
shown two graphs the judge prefers the baseline's at deployed length (49 of 58 decisive
comparisons, restricted-vocabulary arm), and prefers ours when each system contributes
its top ten pairs (40 of 64). Both verdicts are reported in the paper, and neither is the
axis, because a preference between graphs of different lengths cannot separate "the extra
relations are false" from "the judge prefers shorter text". Eq. (4) is the only verdict
of the three that does not depend on length. A judge also cannot perceive repetition:
asked to rate informativeness directly it scored the baseline's `on` relations *above*
its other predicates, the reverse of the intended ordering, which is why the
informativeness term is measured against a reference distribution rather than elicited.

**Measured, PSG test, shared detector boxes, one judge:**

| depth | model | asserted/img | accepted/img | bits/accepted | true bits/img |
|---|---|---|---|---|---|
| matched (top 10 pairs each) | RelateAnything | 9.8 | 4.07 | 4.57 | **18.6** |
| | OvSGTR | 9.8 | 4.33 | 3.09 | 13.4 |
| deployed (own scores) | RelateAnything | 18.3 | 6.53 | 4.39 | **28.7** |
| | OvSGTR | 10.3 | 4.64 | 3.04 | 14.1 |

The judge accepts slightly *more* of the baseline's relations at matched depth; each one
is worth about a third less, because most of them are `on` — 66 % of the baseline's
relations, against a modal-predicate share of 0.85 of an average graph versus 0.52 for
ours. Since bits per image have no upper bound, the composite divides by the same
quantity computed on the images' own annotation (27.9 bits over those 199 images), giving
0.67 against 0.48. The matched-depth cell is the one that enters the composite, because
the deployed-depth ratio exceeds 1.0 — the annotation is sparse.

**Controls.** A fixed share of relations are corrupted before judging by substituting
another predicate from the same image, and the run reports how often a corruption is
accepted: 0.135 on specific corruptions at deployed depth and 0.201 at matched depth
(0.275 pooled over all corruption types), a difference due to the rendering rather than
the depth. The headline cell is thus the arm with the *weaker* control. It is reported
anyway, because the acceptance rate is an upper bound rather than a false-positive rate —
a swapped predicate is sometimes true of the pair it lands on — and because the ordering
holds under both renderings and at both depths. Re-running the matched-depth arm in the
stronger rendering remains to be done.

**Judge family.** RA-4M supervision was generated by `gemma-4-26B-A4B-it`; a Gemma judge
would reward our own output distribution, and `llm_judge.py` hard-refuses one. The judge
is from a different family, and an earlier development model was scored by a second
family as a cross-check.

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
  metric most inflated by shared triplet mass and should never be the headline.
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

**The headline composite spans A1, A2, A4, A5 and A6**, and reads 40.1 for the released
tower against 11.8 for the baseline. A3 is measured, reported and not summed: it cannot
be run on the open-vocabulary baseline at all — its predicate vocabulary is a single
caption capped at 512 word pieces, about 150 strings, against A3's 19,103 — so a
composite containing A3 would exist for one of the two models being compared. A5 entered
the composite only once it was scored per model rather than pairwise (see above); a
pairwise win rate could not be summed, since the two rates describe the pair rather than
either model. Two cells are normalised by a measured quantity rather than a chance level:
A4 by the pair-recall ceiling of the shared detector and A5 by the information in the
images' own annotation, so that each reports the share of what is recoverable that a
model delivers.

**Three composites exist and they are not comparable.** The five-axis OVS above is the
headline. The **model ladder** (`docs/training.md`, the released family) is scored on
A1, A2, A4 and A6, because A5 costs one judge run per arm and was run for the released
tower only. **OVS-dev**, over A1, A2, A3 and A6, selected the recipe before A4 and A5
existed; recomputing every selection decision on the ladder composite changes no
outcome, which is what makes the selection not an artefact of the axis set. Withholding
each axis of the headline composite in turn leaves the ordering of the two systems
unchanged, at ratios between 2.0 and 3.7×.

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
  makes results slightly **worse** (R@50.1551→.1529), so the pruning is not what
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
python benchmark/eval_zeroshot.py... --open_vocab --tau_eval <tau from calibrate_match_tau.py>

# A2: explicit negatives
python benchmark/eval_haystack.py --checkpoint <snapshot>/model.pth --pack runs/packed/haystack
python benchmark/eval_hico_map.py --checkpoint <snapshot>/model.pth --pack runs/packed/hicodet

# A4: shared open-vocab detections, the pair-recall ceiling, SGDet
python benchmark/detect_boxes.py --weights <detector.pt> --pack runs/packed/psg/test --out runs/det/psg_test.npz --set_classes
python benchmark/detector_recall_ceiling.py --dataset_root runs/packed/psg --split test --det runs/det/psg_test.npz
python benchmark/eval_zeroshot_detbox.py --checkpoint <snapshot>/model.pth --dataset_root runs/packed/psg \
    --dataset_name psg --split test --det runs/det/psg_test.npz

# A5: the oracle (needs a VLM)
python benchmark/dump_relsgg_interchange.py...      # our predictions in the interchange format
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
