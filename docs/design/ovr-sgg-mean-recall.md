# OvR-SGG: why the Novel column should be mean recall

**Status:** measured 2026-09-05 on Swin-T boxes; Swin-B pending. All numbers from
`benchmark/ovr_meanrecall_table.py`, which reads the `--per_class` output of
`score_native_protocol.py` (OvSGTR's own recall code, transcribed).

---

## 1. What we changed about our own row first

Our previously reported OvR row came from a tower that had seen all 50 VG150
predicate **strings** in training — never from VG150 images, but seen (megasg,
vg_raw, hicodet). OvSGTR's Novel column is only meaningful because *they* never saw
those 15 with a label. Same-boxes and same-matcher already removed the detector and
protocol confounds; the remaining one is training vocabulary.

So we retrained the shipped recipe with those 15 predicates held out
(the shipped recipe on a text space rebuilt without them by `training/build_heldout_text_space.py --heldout training/ovr15.json`; 12.53% of relations removed, images unchanged).
Our Novel R@50 on Swin-T falls **24.14 → 11.82**. That drop is real and we report it.

**The honest instance-weighted result is that we lose the Novel column**: 11.82
against OvSGTR's 13.21. We still lead Base+Novel, 22.40 against 20.44.

## 2. Then look at what that column is actually measuring

VG150's novel subset is not a balanced held-out set:

| predicate | GT | share of novel |
|---|---|---|
| `on` | 52,950 | **62.3%** |
| `of` | 16,259 | 19.1% |
| `in` | 9,501 | 11.2% |
| *the other twelve* | 6,283 | 7.4% |

**92.6% of the novel GT is three predicates.** Instance-weighted Novel R@K is
therefore close to a measurement of `on`. Per predicate, at R@50 on Swin-T:

| predicate | GT share | ours (held out) | OvSGTR |
|---|---|---|---|
| on | 62.3% | 11.9 | **18.7** |
| of | 19.1% | **11.7** | 4.8 |
| in | 11.2% | **16.3** | 4.7 |
| riding | 1.7% | **47.5** | 0.0 |
| walking on | 0.6% | **13.0** | 0.0 |
| eating | 0.3% | **4.8** | 0.0 |
| painted on | 0.1% | **1.2** | 0.0 |
| walking in | 0.1% | **1.0** | 0.0 |
| wears, attached to, belonging to, for, part of, playing, says | 4.5% | 0.0 | 0.0 |

**OvSGTR is non-zero on 3 of the 15 novel predicates, and on 13 of all 50.** Its
top-50 never contains the other 37. We are non-zero on 8 and 40 respectively.

Their entire Novel advantage is `on`. They win it by +6.8 there, and `on` carries
62.3% of the weight.

## 3. Mean recall, which is the SGG convention for exactly this reason

`mR@K = mean_c (tp_c / gt_c)` over classes with GT support — the definition already
used by `eval_ovsgtr_novel.py`.

| subset | K | ours (held out) | OvSGTR | ratio | ours non-zero | OvSGTR non-zero |
|---|---|---|---|---|---|---|
| Base+Novel (50) | 20 | **11.06** | 3.07 | 3.6× | 38/50 | 13/50 |
| Base+Novel (50) | 50 | **14.86** | 4.06 | 3.7× | 40/50 | 13/50 |
| Base+Novel (50) | 100 | **17.61** | 4.77 | 3.7× | 40/50 | 13/50 |
| Novel (15) | 20 | **5.06** | 1.29 | 3.9× | 7/15 | 3/15 |
| Novel (15) | 50 | **7.17** | 1.88 | 3.8× | 8/15 | 3/15 |
| Novel (15) | 100 | **8.98** | 2.37 | 3.8× | 8/15 | 3/15 |

The ranking inverts and holds at every K and on both subsets.

## 4. This is the same degeneracy A5 measured independently

On PSG, judge-free, OvSGTR emits **1.6–2.2 distinct predicates per graph with a modal
share of 0.61**, and draws **52% of its total true information from `on` alone**
(`docs/design/a5-graph-quality-metric.md`). The 13-of-50 non-zero count here is that
same behaviour surfacing in their own benchmark. Two unrelated evaluations, one
finding: the model transfers to the head predicates and essentially nowhere else.

## 5. Reporting recommendation

Report **both**, and the per-class table beside them:

* **Novel R@50 (micro)** — we concede: 11.82 vs 13.21.
* **Novel mR@50** — we lead 3.8×: 7.17 vs 1.88, on 8 non-zero classes against 3.

The micro row is the published convention and should not be dropped; the mR row and
the non-zero-class count are the generalisation claim. Stating only micro credits a
model for one predicate; stating only mR would look like metric-shopping. The
per-class table makes both unarguable.

**Do not describe this as concept-level holdout.** It is a string-level holdout
matching OvSGTR's own split: we still train on 625,111 instances of `on`-synonyms
(`resting on`, `above`, `on top of`, `sitting on`), and `wears` is held out while
`wearing` survives with 334,197. Their base set leaks identically (`sitting on`,
`standing on`, `parked on`, `mounted on` are base while `on` is novel). That symmetry
is what makes the rows comparable; neither model is concept-blind.

## 6. Conditions, verified rather than assumed

* **Boxes byte-identical.** 26,404 images, 2,563,082 boxes, `max_abs_diff 0.00e+00`
  on coordinates and `box_scores`; their detections consumed with `det_conf 0.0`
  (no re-thresholding) and `det_max_objects 150`, which never binds (their max is 100).
* **`labels` differ only by index base** — theirs 1-indexed, ours 0-indexed,
  `theirs == ours + 1` for all 2,563,082 entries. Each file declares its own
  `label_base`; the scorer subtracts it. Declarations were checked against the actual
  ranges, since a mis-declared base would silently corrupt every triplet match.
* **Protocol `ovsgtr` is MANY-TO-MANY**, as published: a GT object covered by *k*
  duplicate detections gives *k* chances. `duplicate_detection_factor` = **2.99**
  here, so absolute numbers are inflated for both models equally. `ovsgtr_1to1`
  (19.76 / 10.09) and `relsgg` (17.35 / 9.32) are reported alongside.
* **Vocabulary reparametrised from text at eval.** The trained head has no row for the
  held-out 15; `export_ours_interchange.py` discards that matrix and encodes all 50
  VG150 predicate names with the text student (`reparameterizing vocab head to 50
  predicates`, `novel predicates: 15/50`). Sampling 107k pairs, the argmax lands on a
  held-out predicate 16.3% of the time — so 11.82 is real generalisation, not a floor.
* **Pair budget favours them**, mildly: we export 150 pairs/image, they store 500. The
  scorer reads only the top 100, so R@100 is unaffected.
* **Pair-recall ceiling** on these boxes: 84.40%.
