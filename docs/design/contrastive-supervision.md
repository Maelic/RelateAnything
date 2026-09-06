# Contrastive supervision for open-vocabulary SGG: what a source knows

*Design note, 2026-08-26. For anyone adding a dataset, a pseudo-label source, or a new
loss term to RelSGG. Read this before touching `relsgg/loss_synonym.py:BatchLocalInfoNCE`.*

## 1. The problem in one sentence

A pair's label set is never complete, and **the incompleteness has structure by source**.
The contrastive loss is a claim about what the data knows; every regression we have
measured on the loss side was that claim being wrong somewhere.

Measured instances (all in `runs/benchmark/ovs.json` / memory notes):

| what was treated as a negative | what it actually was | cost |
|---|---|---|
| a synonym of the GT ("riding" vs "riding on") | a positive | tail collapse, fixed by synonym groups → v42 soft kernel |
| a co-occurring predicate ("holding" for a "looking at" pair) | unknown, likely true | fixed by co-occurrence soft negatives (w = 0.3, later fitted) |
| "on"/"above" for a HICO "riding" pair | true, never annotatable by HICO | projective SpatialSense AUC −0.05, share-independent (2026-08-26) |
| a pair's other GT predicates (single-label targets) | positives | multi-hot targets, per-group aggregation |

Each fix was local. This note writes down the general rule so the next source does not
need a new fix.

## 2. Vocabulary

For an anchor `a` (a directed pair in an image) from source `s`:

- `L_a` — its annotated predicates (multi-hot).
- `C_s` — the **label space of the source**: the set of columns the annotator was
  *choosing among*. HICO: 117 verbs. VG/RA-4M: open text (effectively "anything").
  PSG: 56. A pseudo-label source: exactly the columns the labeller was asked about.
- `e_s` — **exhaustiveness within `C_s`**: P(a true predicate in `C_s` is annotated).
  HICO annotates every verb per pair → e ≈ 1. VG annotators write one relation and move
  on → e is low. This is a *source* property and it is estimable (§5).
- `N_a` — explicit negative cells (HICO "no interaction" captions, Haystack negatives,
  a verifier's "no").
- `inv(L_a)` — spatial inverses / antonyms of the GT: true negatives *by structure*, the
  one case where a negative is certain without annotation.
- `k(v | L_a, cats_a)` — the co-annotation kernel: P(v also true | L_a, subject/object
  categories), fitted on pairs that carry >1 label (`training/build_pair_cooc.py`).
- `syn(v | L_a)` — the synonym kernel P(v is a paraphrase of L_a | text cosine), fitted
  (v42), not thresholded.

Everything below is a function of these six objects and nothing else.

## 3. The status of a column, per anchor

For a column `v` in the batch set `S`, exactly one of:

| status | condition | denominator weight | positive weight |
|---|---|---|---|
| positive | `syn(v \| L_a)` high | 0 | `syn` |
| structural negative | `v ∈ inv(L_a)` | 1 | 0 |
| explicit negative | `v ∈ N_a` | 1 | 0 |
| in-vocabulary negative | `v ∈ C_s \ L_a` | `e_s · (1 − k(v \| L_a, cats))` | 0 |
| **unknown** | `v ∉ C_s` | **0** (ignored) | 0 |

Today's code implements rows 1–3 exactly, row 4 with `e_s = 1` for every source and `k`
from the cooc kernel, and row 5 only when `--restrict_neg_sources` names the source.
The two gaps are therefore: **(a) `e_s` is silently 1 for VG-style sources**, so their
unannotated in-vocabulary columns are pushed down at full weight; **(b) the unknown row is
opt-in** rather than derived from the source's label space.

The rule generalises with no new hyperparameters: a source contributes negatives only
inside its own label space, scaled by how exhaustive it is there, discounted by how often
the column co-occurs with what was annotated.

## 4. What goes into the batch set `S`

The current set is `GT(batch) ∪ text-hard ∪ inverses ∪ uniform fill to n_neg`. Three
changes, in order of expected value:

1. **Pair-conditional hard negatives.** The discriminative negatives for a person–horse
   pair are *feeding / leading / riding / walking*, not *casting light on*. They come from
   the same cooc/object→predicate prior that gives soft negatives: high prior plausibility
   for `cats_a`, low `k` given `L_a`. Text-neighbour negatives (current "hard" pool) are
   confusable in *language*; these are confusable in *the world*. Graphical Contrastive
   Losses (Zhang et al. 2019) made the same distinction with entity-instance and proximal
   relationship negatives, on VG.
2. **Family-stratified fill.** The dual expert routes by family (spatial / semantic /
   action). Uniform fill over 19K columns gives an action anchor mostly irrelevant
   negatives; stratify the fill so every anchor sees its own family at a fixed share.
3. **Density normalisation.** Once columns are masked per source, anchors have different
   effective negative counts (HICO ≈ 116 vs RA-4M ≈ 512). InfoNCE's gradient scales with
   the number of negatives, so masked sources train *softer*. Add `log(N_ref / |A_a|)` to
   the negative logits (a weight, not a change of objective) so the denominator has the
   same mass for every source. ScaleDet and puNCE both need the same correction for the
   same reason.

## 5. Estimating the source parameters (no hand thresholds)

`C_s` is read from the pack's `meta.json`. `N_a` is the sidecar. `k` and `syn` are already
fitted. `e_s` is the new one, and there are two estimators:

- **Internal:** among pairs carrying ≥2 labels in a source, how often does a predicate
  that the kernel says is co-true (`k > 0.5`) actually appear? That ratio is a lower
  bound on `e_s` within `C_s`.
- **External (preferred once the loop exists):** run the verifier on a sample of the
  source's pairs over `C_s`; `e_s` = fraction of verified-true cells that were annotated.
  The HICO pilot already produced this for HICO (recall of GT against the verifier ≈ 0.6
  at matched precision) and it is the same machinery as per-verb reliability.

Both give a number per source; neither is a threshold.

## 6. The cross-pair term is not optional

InfoNCE normalises over columns *within* a pair. It never trains "is this pair's score for
`on` higher than that pair's score for `on`" — cross-pair comparability, which is exactly
what per-predicate AUC, the haystack precision test and any truth judgment measure. The
per-cell sigmoid auxiliary (`lambda_sigmoid`) is that term, and it must obey the same
status table: a cell is a labelled negative only under rows 2–4, unknown cells get a
PU prior weight, never a hard zero. Structurally this is OWL-ViT's federated loss and the
SPML "positive label is all you need" position: train on what is known, do not invent
negatives.

## 7. What a new source must ship

A **source card** next to the pack (proposal: `<root>/train/source.json`), consumed by
`train.py` instead of per-source flags:

```json
{
  "label_space": ["riding", "holding", "..."],   // C_s (or "open")
  "exhaustive": 0.93,                             // e_s, estimated, with its estimator
  "negatives_sidecar": "runs/datamix/hicodet_negatives_train.json",
  "families": {"riding": "action", "...": "..."},
  "inverses": [["above", "below"], ...],
  "provenance": "human | vlm-verified | model-proposed"
}
```

`--restrict_neg_sources` is the interim form of the `label_space` field; the `exhaustive`
field is the missing one. A pseudo-label source from the loop is then a first-class
citizen: its label space is the candidate set it was asked about, its exhaustiveness is
the verifier's recall, and its per-cell confidence rides on the positive weight.

## 8. Where this sits in the literature

- Partial label spaces across datasets: Detic-style "Simple multi-dataset detection"
  penalises only within each dataset's label space; UniDet / ScaleDet unify label spaces
  with hard + semantic-soft assignment. Our status table is that idea at the predicate
  level, with the exhaustiveness term detectors do not need (boxes are exhaustive per
  image, relations are not).
- Positive-unlabeled contrast: puNCE / puCL re-weight unlabeled columns as soft
  positive–negative mixtures; SPML treats unobserved labels as unknown. Row 5 and the PU
  prior in §6 are those.
- SGG-specific negatives: Graphical Contrastive Losses (typed hard negatives), Lang3DSG
  (negatives from the label set, always including "no relation"), ReLIC-SGG (unannotated
  relations as latent variables to complete, not negatives), CAGE-SGG-style contradiction
  sets (our inverse mask, generalised to antonyms from the text student).

## 9. Test plan on the proxy ladder (2.5 GPU-h per arm)

1. `NEGMASK=hicodet` — row 5 for HICO. **Running (6961179).** Read: A6 back in the ctl
   band with HICO F1 intact.
2. `e_s` for VG-style sources (RA-4M/vg_raw) < 1 — the internal estimator, applied as the
   row-4 weight. Read: tail SoftmR and A3 (fewer false negatives on the open-vocab tail).
3. Density normalisation on top of 1. Read: HICO F1 unchanged, spatial unchanged, loss
   curves per source aligned.
4. Pair-conditional hard negatives replacing half the text-hard pool. Read: A2 precision
   and HICO per-verb acc@1 (the eating / eating-at confusions are exactly this kind).

Arms 1–3 are flag-only. Arm 4 needs the object→predicate prior table as a sampler input.

## 10. Deriving column status from a knowledge source instead of from annotation

The status table in §3 is built from what a *source* observed. A second, independent
axis is what the *predicates themselves* imply about each other, which no annotation
records and which is the same for every source. For a positive `L_a`, an external
relation table `T(v | L_a)` can say:

| relation | example | status it implies |
|---|---|---|
| paraphrase | riding on ↔ riding | positive (weight from kernel) |
| entailment | riding ⇒ on, sitting at ⇒ near | **soft positive** — the column is true, even if the source could never label it |
| compatible | holding ∧ looking at | soft / unknown — never a full-weight negative |
| exclusive | in front of ⊥ behind, sitting on ⊥ standing on | **hard negative by logic** — extends `inv()` beyond spatial inverses |
| unrelated | riding vs casting light on | ordinary negative, subject to the source rows |

This is strictly stronger than the source mask: masking makes "on" *unknown* for a HICO
"riding" pair; entailment makes it a *positive*. And it fixes the near-collision the
embedding cannot: "eating" vs "eating at" have text cosine 0.98 but the table says
*distinct, compatible, different object types* — not paraphrases.

### Which knowledge source gives which layer

1. **Word / sentence embeddings** (our text student, dino.txt) — *similarity*, symmetric.
   They cannot tell paraphrase from antonym (in front of / behind are close; the antonym-
   aware student was built to patch exactly that) or entailment direction. Measured limits
   here: the A3 τ bug (a cosine threshold calibrated in one space accepted 0.6% of true
   synonyms in another), the eating / eating-at collision, "higher than" → beneath. Use them
   for what they are good at: **candidate generation** (which pairs are worth asking about)
   and a **tail prior** where no data exists, always through a *fitted* kernel, never a
   threshold.
2. **Entailment / LLM judgment** (an NLI model or an instruction-tuned LM on templated
   triplets "a person is riding a horse" vs "a person is on a horse") — *relation type*,
   directional. This is the missing "certified cross-lemma resource" named in the
   commit-late decoding post-mortem. It gives the lattice ReLIC-SGG completes and the
   contradiction sets CAGE-SGG builds, from one judge, for any vocabulary.
3. **World knowledge / affordance tables** (HICO's object→verb table, VerbNet/ConceptNet
   style priors, or the same LM asked "can a `<subj>` `<pred>` a `<obj>`?") — *category-
   conditional plausibility*. This is what turns pair-conditional hard negatives (§4.1)
   from a data statistic into something that generalises to unseen categories. Note the
   FREQ baseline beats the model per-edge on micro recall: plausibility priors are strong,
   and a text-derived one is the open-vocabulary version.

### How it plugs in without new hyperparameters

The judge produces a label per (v, u[, subject type, object type]); the *weight* attached
to a label is still fitted from data, exactly as `syn(v | L)` is today: the 34K co-annotated
pairs give P(both true | judge says entailed / compatible / exclusive), which calibrates the
judge per relation type (the same per-class reliability estimate we used for the Qwen
verifier). Pairs the judge never saw fall back to the source rows. So the loss becomes

    weight(v | a) = source_term(v, s(a)) × logic_term(v, L_a, cats_a)

with `source_term` from §3 and `logic_term` from the calibrated lattice; either factor
alone reproduces today's behaviour.

### Cost and validation

V = 19,103 makes V² impossible; judge only each predicate's top-k text neighbourhood plus
its family (k = 50 → ~1M pairs, ~200K for the predicates that actually occur as positives).
An NLI model does that on CPU in hours; an LM at 7B does it on one GPU in a night.
Validation is GT-free and threshold-free: on co-annotated pairs the judge must never say
*exclusive*; on high-support never-co-annotated pairs *exclusive* must be enriched; on the
known spatial inverse list *exclusive* must be ≥ the inverse mask's own coverage; and the
eating / eating-at, in-front-of / behind, higher-than / beneath cases must come out as
compatible / exclusive / exclusive respectively. Then the on-target reads are A3, A6 and
HICO per-verb acc@1.

This is arm 5 of §9, and the one that survives the cluster: it needs a judge and CPU, not
the A40s.

## 11. Our own predicate space: predicates as regions, not points

Everything above patches a *symmetric* space: the frozen text bank `W` scores a pair by
cosine, and cosine cannot encode direction (riding ⇒ on but not the reverse), exclusion
(in front of ⊥ behind) or independence. The status table and the lattice are bolted on as
masks and weights because the geometry cannot carry them. The alternative is a space whose
geometry *is* the lattice.

### The candidate: probabilistic box embeddings

Each predicate `v` is an axis-aligned box `B_v = [m_v, M_v] ⊂ R^d` (Vilnis et al. 2018;
Gumbel-smoothed boxes, Dasgupta et al. 2020). Volumes are probabilities:

| relation | geometry | probability |
|---|---|---|
| paraphrase | near-identical boxes | P(A\|B) ≈ P(B\|A) ≈ 1 |
| A entails B | `B_A ⊂ B_B` | P(B\|A) = 1, P(A\|B) < 1 |
| compatible | overlap, neither contains | 0 < P(A∧B) < min |
| exclusive | disjoint | P(A∧B) = 0 |
| unrelated | independent | P(A∧B) = P(A)·P(B) |

"Unrelated" gets a *definition* (independence) instead of a default. A pair's visual query
becomes a point `x` (or a small box for uncertainty) and `score(pair, v) = log P(x ∈ B_v)`,
which is O(V·d) exactly like the dot product, so the open-vocabulary head stays free.

### Why our own measurements point here

- **Region alignment already won.** The synonym-group logsumexp (positives = a *region* of
  text space) is what transferred to the tail; identity positives (points) lost 33–68% on
  rare classes ([[relsgg-region-vs-point-alignment]]). Boxes make "a predicate is a region"
  the primitive rather than an emergent property of the loss.
- **Entailment at inference for free.** A "riding" point inside `B_riding ⊂ B_on` scores high
  for "on" by construction. Today a HICO pair that was never labelled "on" must be masked;
  in a box space the geometry says it is on.
- **Exclusion by construction.** Disjoint boxes make "in front of" and "behind" mutually
  exclusive without an inverse mask; the swap-direction hinge becomes a geometric constraint.
- **Winner-take-all disappears.** eating / eating at are overlapping-but-distinct boxes;
  a point can be in both, and neither is pushed out of the other by the loss.
- **Open vocabulary is preserved** by keeping `W` as the *input*: a text→box mapper
  `g: w_v ↦ (m_v, M_v)` is trained on known predicates and applied to any new phrase, so a
  novel predicate gets a box at test time with no retraining (Task2Box uses the same trick
  for asymmetric task relations).

### The losses, and how the status table collapses into them

1. **Membership (image supervision):** for GT `v`, maximise `log P(x ∈ B_v)`; for columns
   in status rows 2–4, minimise it with the row's weight; unknown columns: no term. The
   InfoNCE partition disappears — this is the per-cell sigmoid term of §6 in box form, which
   is what trains cross-pair comparability in the first place.
2. **Lattice (predicate–predicate):** regression of `vol(B_A ∩ B_B)/vol(B_A)` onto the
   co-annotation conditional `k(B|A)` where it is supported; PPDB / NLI / inverse-list labels
   as soft targets (entailed → 1, exclusive → 0, independent → `P(A)·P(B)`). Weights fitted
   on the co-annotated pairs as everywhere else. This term needs **no images**.
3. **Marginals:** `vol(B_v)` free, or tied to predicate frequency as a prior.

The source rows of §3 still say *which cells are supervised*; the lattice rows of §10
become *geometry* instead of per-anchor weights.

### What could go wrong

- Boxes need enough dimensions to hold thousands of partially overlapping regions; d ≈ 64–128
  in the literature for label spaces of this size (Patel et al. 2022, multi-label boxes).
- Gumbel smoothing has a temperature; it is a fitted model parameter, not a data threshold.
- The dual expert must still produce two points (or one point in a product space) — the
  spatial/semantic gate is orthogonal and stays.
- Calibration: `P(x ∈ B_v)` is a probability, but the Platt step should be re-checked, not
  assumed away.

### The pilot that survives the cluster

No retraining is needed to test the *space*: take the trained ViT-S+ queries for the eval
pairs (the edge-dump / attribution pass already exports them), freeze them as points, fit
`g` and the boxes with losses 1–2 on the training split, and re-score the same queries with
`log P(x ∈ B_v)` instead of cosine (the rescoring harness is not part of the release). Reads:
A3 (does region scoring transfer to the open vocabulary), A6 and HICO per-verb acc@1 (do
entailment / exclusion resolve the eating / eating-at and in-front-of / behind cases), and
held-out co-annotation conditionals (does the lattice generalise). CPU-scale; hours.
If it wins on frozen queries, the joint version (train the visual head into the box space)
is the next architecture, and the loss design of §§3–6 is its supervision.

### Pilot results (2026-08-26, frozen HICO-0.05 S+ queries; `training/box_pilot_*.py`)

| version | objective on frozen queries | control vs base | box vs base | lattice (held-out) |
|---|---|---|---|---|
| v1 | membership BCE, 512→64 from scratch | collapses (vg150 mR .43→.17) | worse | Spearman 0.77, direction + exclusion recovered |
| v2 | membership BCE, residual-in-text-space (starts at base) | still collapses the tail (.43→.21) while micro R rises | worse | MAE 0.04, exclusion 0.006, compatible pairs shrink to 0 |
| v3 | the checkpoint's own InfoNCE | **stays at base** (±0.02) | **matches base** (HICO mR .797 vs .798) | **collapses to 0** — widths → 0, boxes become points |
| v4 | own InfoNCE + width floor 0.02, λ=20, 30 ep | at base | −3…−8 % | flat at 0.985 — never learned |
| v5 | PCA-64 basis both sides (EV 0.98), own InfoNCE, λ=20 | at init −8…−11 % mR (HICO equal); after fit = base | −5…−15 % mR | flat — boxes never overlap at init, gradient dead |
| v6 | v5 + density score (log member − log vol/d), widths ×3, β_vol 0.2 | = base | **−60…−75 % mR** — at 3× width every box contains every query, so the score is −log vol alone = a box-size prior | flat at 0.985 (synonym-pair conditional at init 2e-5); untrained held-out Spearman 0.54 |
| v7a/b | v5 widths + **log-space BCE lattice loss** (± density), λ=20 | = base | **−80 % mR** (member loss rose 4.3→5.7: boxes inflated to plateaus) | loss 0.40→0.15 by *inflation*: inverse cond 0.73, held-out MAE 0.76 (≈ predicting 1), Spearman 0.63 — the loss had no "unrelated → 0" pairs |
| v8a | v7 + 50 % never-co-annotated zero pairs (target 0), λ=20 | = base | −57…−71 % mR (HICO −40 %) | Spearman 0.60, MAE 0.35; direction P(on\|riding) .77 vs .14; inverse .43, zero .32 |
| v8b | same, λ=2 | = base | −42…−58 % mR (HICO −24 %) | Spearman 0.55, MAE 0.26; P(riding\|on) .03, P(behind\|in front of) .09, P(below\|above) .16; inverse .30, zero .27 |

Three facts, in order of confidence:

1. **The relation types are learnable as geometry from data we already have.** Co-annotation
   conditionals + spatial inverses train a text→box mapper that generalises to held-out
   pairs with direction (P(on|riding) = 1.00, P(riding|on) = 0.00), exclusion (in front of /
   behind, above / below → 0), and compatibility (eating / eating at ≈ 0.23, not a paraphrase).
2. **On frozen queries the objective dominates the space.** Any refit that is not the training
   objective re-allocates mass to the head first — the point control proves it. Only the
   model's own InfoNCE leaves the control at base; under it, box membership is a drop-in
   equal of cosine (no gain, no loss).
3. **Membership is not a ranking score among nested predicates.** A point inside
   `B_riding ⊂ B_on` is a member of both with probability 1; only a *density*,
   `log P(x ∈ B_v) − log vol(B_v)`, ranks the specific box above the general one that
   contains it. That single change is what makes entailment (big general boxes) and
   ranking (prefer the specific) compatible in one space — **but only with widths at the
   query spread**: v6 widened boxes ×3 to force overlap and the membership term became a
   plateau, leaving `−log vol` alone = a box-size prior (−60…−75 % mR). And the lattice
   gradient was never "dead from no overlap": in d = 64 the conditional is a product of 64
   per-dim overlaps (0.83⁶⁴ ≈ 6e-6), so any loss on the *probability* has gradient ∝ p ≈ 0 —
   the loss must be BCE on `log P(b|a)` (1.5e6× the gradient at init; v7a/b).
4. **Ranking and region structure conflict under naive weighting.** InfoNCE's partition
   rewards sharpness and shrinks regions to points (v3); the membership objective keeps
   regions but collapses the tail (v2). The joint model needs both terms plus a floor or
   volume prior on widths — v4 tests the floor on frozen queries; the real test is joint
   training of the visual head into the space, which is post-cluster work.
5. **On frozen queries the trade is a Pareto line, not a tuning problem** (v8, the balanced
   lattice with a live gradient and zero pairs). λ = 0 / 2 / 20 gives vg150 mR@50 .37 / .18 / .13
   against held-out lattice Spearman .17 / .55 / .60; every unit of entailment geometry
   (boxes that contain the specific box and exclude the inverse) is paid in ranking, because
   those widths cannot also sit at the query spread. The space is fine (v3: box = cosine),
   the lattice is learnable (v1, v8) — what is fixed is the queries. **Pilot closed**; the
   next experiment is joint training, with both losses from step 1, so the queries can move
   into the box geometry. Spec: fixed PCA-64 basis, `log_cond` BCE lattice with zero pairs,
   own InfoNCE on membership, widths floored at the query spread, λ swept 0.5–5.

## 12 · A patch-level relational objective: design and verdict (2026-08-26)

### 12.1 What the probes established, and why it is the expected optimum

`analyze_patch_similarity.py` and `visualize_affordance_similarity.py` (report §6b–6c) measured the
dense features of the pretrained DINOv3, the canonical run and the negmask run:

* relation fine-tuning **sharpened object identity** (class selectivity at L12 0.242 → 0.281,
  object-class 1-NN from contact patches 0.71 → 0.76) and left patch-to-patch similarity
  **relation-blind** (partner − unrelated 0.133 → 0.138; P(partner > same-class other) 0.47 → 0.38);
* the hand region becomes mutually similar with the object it handles (contact advantage 0.36 →
  0.44) but encodes *which object*, not *what is done*: verb 1-NN 0.25 → 0.27, below the 0.42
  majority baseline.

This is not a defect of the recipe; it is what the architecture asks for. The relation score is
`cos(pair_context(SSP(s), SSP(o), deformable read, geometry), text_p)`. Every input to the pair
context is an *object-centric* pooled token plus geometry, so the backbone is rewarded for clean
identity and parts (hand, handlebar, saddle) and for nothing pairwise. Cosine similarity between
two patches is symmetric and predicate-free, so even in principle it can carry "related" but not
"rides" vs "is ridden by". Relations in this model are *composed* from identity + geometry in the
head; they are not a property of a patch.

### 12.2 Three designs

**D1 — naive partner contrastive on the base features (rejected).** Anchor = patches of s,
positives = patches of o, negatives = other objects. It makes a cup patch resemble a hand patch
*regardless of predicate*, i.e. it teaches relatedness, not relations, and it fights the object
head (`loss_obj`), the SSP pooling and the entity probes that the whole head is built on.
Predicted probe outcome: P(partner > same-class) rises, object 1-NN and entity accuracy fall,
OVS falls. Symmetric, undirected, predicate-agnostic — do not build this.

**D2 — dense relational head on top of the shared backbone (recommended).**
Keep the base features identity-organised and give relations their own space:

```
f_i           fused patch token           [P, d]      (P = 28×28 at 448 px)
u_i = A f_i   as-SUBJECT projection       [P, d_r]    d_r = 128
v_j = B f_j   as-OBJECT projection        [P, d_r]
M_p = reshape(H t_p)   predicate-conditioned bilinear, rank r = 16, from the FROZEN text bank
s_p(i → j) = u_iᵀ M_p v_j                 directional, predicate-specific patch affinity
```

`H` is a small hypernetwork (512 → 2·d_r·r) so every predicate in the 19K bank has an affinity
form without per-predicate parameters — open vocabulary is preserved. Two losses, both on GT
relations only, sampled at k = 8 patches per box and ≤ 64 relations per image:

* **L_partner (patch retrieval, "where is the partner")** — for anchor i ∈ s and predicate p,
  InfoNCE over patches j with positives j ∈ o (mask-weighted when SAM `cov` rasters are attached,
  box otherwise) and negatives, in order of importance: patches of *other instances of s's
  class* (identity negative — the one the current features cannot beat), the *nearest unrelated
  object* (proximity negative), background, and patches from other images in the batch.
* **L_pred (predicate retrieval, "what is the relation here")** — for a sampled (i ∈ s, j ∈ o),
  the existing `BatchLocalInfoNCE` over predicate columns with `s_p(i→j)` as the logit, the
  *reversed* pair (j → i) as a directional hard negative for non-symmetric predicates, and the
  source-aware `col_allow` mask so HOI pairs never push spatial predicates down (§3–4 of this
  doc). Soft ontology targets apply unchanged.

Weight λ_dense ∈ {0.1, 0.3}; warm start after epoch 2 so identity forms first; gradient flows
into the backbone (a stop-gradient arm isolates "does the backbone need to change at all").
Cost: `u`/`v` are two linear layers; the affinity `u M_p vᵀ` on sampled patches is
64 × 8 × 784 × 16 flops per predicate column — under 10 % of a step.

What D2 buys, in order of likelihood:
1. **A box-free pair proposer.** `max_p s_p(i→j)` aggregated over patches is a relation affinity
   field. The deployment gap is detector-box retention (53–63 % on PSG, 29–32 % IndoorVG); a field
   that says "these two regions are related, this way round" does not care where the detector put
   the box edge, and can rescore or replace the geometric pair sampler on detector boxes.
2. **The maps the affordance question asked for** — dense, predicate-conditioned, directional,
   with the hand–object region as a first-class output rather than a heuristic query.
3. A regulariser that keeps contact parts distinct in the shared backbone. Evidence it is wanted:
   the last-layer rewrite (CKA 0.58) went *toward* parts already.

**D3 — verb-conditioned contact contrastive (the affordance-specific variant).** Contact region
= person ∩ dilated object (mask if available). Feature = mean of contact patches after a
projection C. Positives = contact regions of the same verb in other images **with a different
object class**; negatives = same object class, different verb. That is the only formulation that
can make "hands doing X" cluster by X rather than by the object, because it explicitly removes
object identity from the positive set. It needs dense verb labels: HICO's 116 verbs (62 of them
dead in the tower without HOI data) plus the loop-verified HICO-train data. It is a study, not a
shipping component: it answers "can affordance be *taught* into these features", after §6c
answered "it does not *emerge*".

### 12.3 Is it a good idea for the current model?

Not for in-domain accuracy, yes for deployment and interpretability — and the reasons are
measured, not guessed:

* **Pair recall on GT boxes is 99.8 %** and the head already separates verbs at rank 2 (66 % of
  HICO GT verbs, 82 % at rank 3). A better dense field does not add pairs the sampler misses or
  verbs the head lacks; the predicted R@K change is inside the 1.3 % noise floor.
* **Pixels matter through the pair read** (attribution: −55 % without pixels, object→text
  channel 0). The relational signal the head needs is already delivered by SSP + the deformable
  read on identity-organised features. D1 would degrade that; D2 leaves it alone.
* **The two open problems it can touch** are detector-box deployment and mask-free operation.
  The two it cannot touch are the HICO sibling-verb collapse (a decode/loss problem, §5 of the
  report) and the spatial axis (geometry is explicit; the model ignores precise angles).
* **Risks**: identity interference (watch object 1-NN 0.76, `loss_obj`, class selectivity);
  head collapse of the affinity field toward frequent predicates (needs the same source mask and
  density normalisation as the pair head); +8–10 % step time.

### 12.4 Pre-registered test (proxy ladder, 2.5 GPU-h per arm)

Arms: control · D2 λ=0.1 · D2 λ=0.3 · D2 stop-grad · D3 (HICO mix). Gates, all from existing
tools: OVS-F1 within noise of control (`overall_score.py --metric F1`); P(partner > same-class)
measured in the **u/v space** rises above 0.5 while staying ≈ 0.38 in the base features
(`analyze_patch_similarity.py`, add a `--space uv` switch); detector-box retention
(`eval_detboxes.py`) up by more than the seed spread; for D3, verb 1-NN from contact features in
C-space above the 0.42 majority baseline (`visualize_affordance_similarity.py`). Any arm that
buys its probe gains with an OVS loss beyond noise is rejected — the point is to add a capability
without paying for it in the head.

### 12.5 Anchors in the literature

Dense interaction objectives exist and work with CNN backbones — PPDM and CDN detect HOI as an
*interaction point* between human and object centres with offset regression; Relationformer
gives relations their own tokens next to object tokens. On the affordance side, LOCATE and
Hotspots recover affordance regions from DINO-family features only through an explicit
objective (exocentric → egocentric transfer, or video-derived hotspots), never from similarity
alone — consistent with §6c. D2 is the ViT-native, open-vocabulary version of the interaction
point idea: a text-conditioned affinity field instead of a fixed verb head.
