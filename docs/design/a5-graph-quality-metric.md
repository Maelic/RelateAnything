# A5 — measuring scene-graph quality with an LLM oracle

**Status:** definition settled 2026-09-04. Supersedes the `useful relations per
image` headline used in earlier A5b runs, which is shown below to be precision in
disguise. Raw verdicts and every derived table are committed under
`runs/benchmark/a5b/`.

---

## 1. The question A5 exists to answer

Recall against PSG ground truth cannot separate the two systems we care about,
because both are open-vocabulary and PSG's annotation is sparse and closed. A
relation can be correct, useful, and absent from the annotation. So A5 asks a
vision-language model to act as the oracle instead.

There are two ways to do that, and they measure different things. **Both are kept.**

| | what it sees | what it is good for | file |
|---|---|---|---|
| **A5 (pairwise)** | two whole graphs, side by side | graph-level quality: coverage, diversity, redundancy | `benchmark/llm_judge.py` |
| **A5b (per-relation)** | one claim at a time | per-claim truth | `benchmark/relation_precision.py` |

They disagree, and the disagreement is informative rather than a problem — see §6.

---

## 2. What the judge is allowed to decide

> **The judge decides TRUTH. Nothing else.**

This is the central design decision, and it was reached by failing twice.

The original A5b asked for two axes in one reply: `true` (yes/no/unclear) and
`info` (0–3, "how much does this say beyond what the two object *names* already
imply"). The `info` axis existed to price the **`on` problem** — a baseline that
emits `on` for most of its relations scores near-perfect precision while saying
almost nothing.

### Failure 1 — the joint axis collapsed onto truth

Measured over the 200-image run, for **every** system:

```
P(info = 0 | true = no )  = 0.998 – 0.999
P(info ≥ 2 | true = yes)  = 0.92  – 0.94
```

So `useful_rate ≡ 0.92 × precision`, exactly, and the second axis contributed
nothing. Worse, it **inverted**: OvSGTR's `on` relations scored `info` **1.10**
against **0.85** for their own non-`on` relations — the axis *rewarded* the modal
predicate, because `on` is the easiest predicate to verify.

The rubric already contained the sentence *"A claim can be TRUE and still score 0
for INFO — judge the two questions independently and do not let one drive the
other."* Exhortation does not work.

### Failure 2 — decoupling it structurally did not help either

`--info_mode text` moves informativeness to a second, image-free pass over the
distinct triples: truth is absent from the context, so it *cannot* collapse. It
did decouple —

```
P(info ≥ 2 | true = yes) = 0.947
P(info ≥ 2 | true = no ) = 0.924      # decoupled, and useless
```

— but only by calling everything informative. The **calibration anchors** added
for exactly this purpose caught it:

```
vacuous  : cup/on/table = 2   sky/above/road = 3   person/near/car = 1   tree/in/park = 2   → mean 2.00
specific : man/riding/horse=3  woman/slicing/cake=2  dog/catching/frisbee=3  child/hugging/bear=2 → mean 2.50
separated = False   (requires ≥ 1.5)
```

`cup on table` scoring 2 out of 3 for specificity is the whole failure in one line.

### Why no rubric can fix this

A per-relation judge is **structurally blind** to the `on` problem.
`cup on table` shown in isolation is a good relation: true, sensible, worth saying
once. Its worthlessness is a property of the **distribution** — the 500th `on` is
empty *because it is the 500th* — and a judge holding one relation at a time cannot
know which one it holds.

**Conclusion: informativeness is measured, not judged.**

---

## 3. The metric

> **Bits of true information per image**
> `B = Σ over relations judged TRUE of −log₂ p(pred)`

The judge supplies only the truth filter — the half it does well
(`false_accept` 0.135 on specific corruptions). Informativeness is the surprisal
of the predicate, which is a measurement, not an opinion:
`on` = 2.31 bits, `riding` = 7.04 bits against the PSG train marginal.

It cannot be gamed in either direction:

* **repeating a cheap predicate** — each copy is worth its 2.31 bits, forever;
* **emitting rare predicates at random** — those are false, and the judge deletes
  them. Rarity pays only when it survives verification.

### 3.1 Choice of `p` — four estimators, all implemented

`p` is an expectation, and there is more than one defensible one. The ranking is
robust across all of them; the magnitude is not, so state which you used.

| estimator | `p` from | script | note |
|---|---|---|---|
| **referenced** | PSG train marginal | `a5b_information.py` | needs an external corpus; forces the shared 56-predicate vocabulary |
| **self** | the system's own true relations | `a5b_information_free.py` | Shannon's own setup: a receiver who knows the model always says `on` learns nothing from the 500th. Self-normalised, so a bigger vocabulary buys headroom |
| **pooled** | all systems' relations in this run | `a5b_information_free.py` | shared yardstick derived from the comparison itself; **recommended** |
| **MDL** | none — Krichevsky–Trofimov sequential code | `a5b_information_free.py` | the first `on` is expensive, the 500th nearly free. Order-independent (Dirichlet–multinomial depends only on counts) and **charges for vocabulary size** via the `(V−1)/2·log₂N` redundancy |

**No external distribution is required.** For our closed-vocabulary arm the pooled
estimator gives **28.2 bits** where the PSG-referenced one gives **28.7** — the
external corpus was buying essentially nothing.

### 3.2 The vocabulary trap — why `--min_count` exists

Every estimator treats the predicate **string** as an atom, so a decoder emitting
malformed variants is paid for them as if they were novel relations. Our
`vocab=train` arm has `V = 211`, of which a real share is garbage: `on of` (39),
`in on` (32), `painted on a`, `reflected in a`, `eating from food near`, `on her`,
plus 79 singleton types.

`--min_count k` merges types seen fewer than `k` times **globally** (identical
treatment for every system, so the merge cannot favour one). Sweep it. A conclusion
that only holds at `min_count 1` is an artifact of the tail.

MDL bits/image, highlighted verdicts, deployed depth:

| `min_count` | ours (pack) | ours (train) | OvSGTR |
|---|---|---|---|
| 1 | 25.0 | 47.1 | 7.2 |
| 3 | 24.8 | 40.5 | 7.2 |
| 10 | **23.7** | **28.9** | **7.2** |

`ours-pack` vs `OvSGTR` is stable at every setting (ratio 3.3–3.5×) and OvSGTR does
not move at all — `V = 12`, there is no tail to collapse. The train arm loses 39%,
so **do not quote its 47.1**; at `min_count 10` it is still genuinely ahead, so the
diversity is real but much smaller than the raw count suggests.

### 3.3 Judge-free companions

Neither needs an oracle or a reference, and both show the `on` problem directly:

* **distinct true predicates per graph** — 2.4 (ours) vs 1.2 (OvSGTR)
* **within-image predicate entropy** — 1.01 (ours) vs 0.20 (OvSGTR)

---

## 4. The control gate — read this before quoting any `false_accept`

A5b corrupts `--control_frac` of relations before judging by swapping in a different
predicate **from the same image's graph**, and refuses to certify if the judge
accepts corrupted claims too often.

**The pooled `false_accept` is depth-dependent and must not be used as a
judge-quality number.** If the swap lands on a *generic* predicate the claim is
frequently **still true**, so the judge accepting it is not judge error:

```
accepted:  beside 0.57   on 0.42   in 0.31   |   wearing 0.05   holding 0.11
```

Generic predicates concentrate at high rank, so **within one run**:

```
false_accept at rank < 10  = 0.276
false_accept at rank ≥ 10  = 0.186
```

The deployed-depth run certified (0.236) only by averaging the two, and the matched
top-10 run "failed" (0.275) for this reason alone — not because of the judge or the
configuration.

**Gate on the specific subset** (`GENERIC_PREDICATES` in `relation_precision.py`):
0.135 – 0.201 across all runs, all passing. `summary.valid` now reflects this.

### 4.1 Colour highlighting improves grounding (adopt it)

`HIGHLIGHT=1` renders one image per relation — subject in a thick **blue** box,
object in thick **orange**, everything else thin grey — instead of one numbered
image per photo shared by all its relations.

| | pooled | specific |
|---|---|---|
| plain | 0.236 | 0.170 |
| highlighted | **0.196** | **0.135** |

Same `n = 986`, about 3 standard errors. It also makes the judge **stricter on us**
(our precision 0.377 → 0.356; theirs 0.448 → 0.450), meaning some apparent precision
was the judge failing to ground among ~100 identical boxes and defaulting to yes.
Costs prefix caching; still cheap (974 relations in 76 s).

---

## 5. Two accounting fixes that change published numbers

1. **`rel_per_image` dropped controls**, so every `useful/img` was ~12% low. A
   control relation *was* emitted; only its verdict is unusable. Corrected
   deployed-depth values: **6.34 / 6.70 / 4.33** (was 5.61 / 5.92 / 4.07). The
   check: corrected `rel/img` now reproduces judge-free `graph_stats` exactly
   (18.3 / 10.3).
2. **Depth is now explicit.** `a5b_slice.py` re-reads any run at any depth without
   re-judging, recomputing the control at that depth rather than inheriting it.

---

## 6. Results

PSG test, 200 images, judge **Qwen3-VL-8B-Instruct**, both systems on the **same
YOLO-World boxes**. Never a Gemma judge — megasg supervision came from Gemma, so a
Gemma judge measures self-preference.

### 6.1 Matched length (top-10 both sides, `--max_rel 10 --rel_frac 0.0`)

Graph length equal by construction — 9.8 relations each.

| system | rel/img | precision | bits (referenced) | modal share | preds/graph |
|---|---|---|---|---|---|
| ours (pack) | 9.8 | 0.414 | **18.6** | `on` 0.22 | 3.3 |
| ours (train) | 9.8 | 0.414 | n/c | `on` 0.06 | 4.9 |
| OvSGTR | 9.8 | **0.441** | 13.4 | `on` **0.61** | 2.2 |

Control 0.201 specific (n = 369) — passes.

**At matched length OvSGTR is more precise per relation and we deliver 39% more
true information.** Decomposition — the sentence for the paper:

> OvSGTR draws **52%** of its total information from `on` alone; we draw **11%**.
> They have **more** true in-vocabulary relations (799 vs 723) and **fewer** bits
> (2468 vs 3305).

### 6.2 Deployed graphs (each model emits what it would ship)

| system | rel/img | precision | bits (referenced) | bits (MDL, min_count 10) | H(pred) |
|---|---|---|---|---|---|
| ours (pack) | 18.3 | 0.356 | **28.7** | 23.7 | 3.84 |
| ours (train) | 18.8 | 0.360 | n/c | 28.9 | 6.90 |
| OvSGTR | 10.3 | **0.450** | 14.1 | 7.2 | 1.98 |

`n/c` = the train arm emits 73% of its predicates outside PSG's vocabulary, so a
referenced bits total over the remainder is a biased subsample. This is what the
reference-free estimators exist for.

### 6.3 The two oracles disagree, coherently

The pairwise oracle, both vocabulary arms, with the length confound visible:

| run | cmp | decisive | win (ours) | rel ours | rel theirs | A longer | equal |
|---|---|---|---|---|---|---|---|
| deployed, pack | 140 | 58 | 0.16 | 18.1 | 10.4 | 0.10 (101) | 0.40 (39) |
| **matched top-10, pack** | 142 | 64 | **0.62** | 9.8 | 9.8 | — (0) | 0.62 (142) |
| deployed, train | 141 | 46 | 0.24 | 18.6 | 10.2 | 0.16 (106) | 0.62 (35) |
| **matched top-10, train** | 140 | 63 | **0.68** | 9.9 | 9.9 | — (0) | 0.68 (140) |

All four certify (control accuracy 0.94–1.00, primacy 0.16–0.24, ≥17 decisive
controls). **The deployed-graph numbers are not usable as a quality verdict.** Our
graph is longer in essentially every comparison and theirs is NEVER longer
(`b_longer` n = 0 in all four runs), so 0.16 / 0.24 substantially measures a
verbosity prior. Split by length, the equal-length bucket of the deployed runs
already reads 0.40 / 0.62; forcing equality by construction gives 0.62 / 0.68 on
4–6× more decisive comparisons.

So at matched length the **pairwise** judge has us winning 0.62–0.68 while the
**per-relation** judge has OvSGTR ahead on precision. Both certify. This is not a
contradiction:

> Pairwise sees the whole graph, where diversity and coverage are visible.
> Per-relation sees one claim in isolation, where only truth matters.
> **Our graph is better as a graph; their individual claims are likelier to be
> true.**

The bits metric is the quantitative form of the first half of that sentence.

---

## 7. Reproducing

```bash
# per-relation oracle; HIGHLIGHT=1 is recommended, INFO_MODE stays joint (info is
# taken from the bits metric, not from the judge)
python benchmark/relation_precision.py --system ours=<ours.npz> --system baseline=<baseline.npz> \
    --pack runs/packed/psg/test --highlight --info_mode joint
python benchmark/relation_precision.py ... --highlight --max_rel 10 --rel_frac 0.0   # matched length

# pairwise oracle at matched length
python benchmark/llm_judge.py --a <ours.npz> --b <baseline.npz> --pack runs/packed/psg/test --max_rel 10

# analysis — all read committed verdicts, no GPU
python benchmark/a5b_slice.py            --run <verdicts.json> --max_rank 10
python benchmark/a5b_information.py      --run <verdicts.json>
python benchmark/a5b_information_free.py --run <verdicts.json> --min_count 3
```

Raw verdicts are committed gzipped under `runs/benchmark/a5b/*.json.gz`
(6 MB → ~126 KB); every script reads them directly via `gzip.open` or after
`gunzip`. Derived tables are the `.txt`/`.json` pairs beside them.

## 8. Open

* The `vocab=train` decoder emits malformed predicates (`on of`, `in on`,
  `painted on a`). Worth fixing on its own merits; until then quote that arm at
  `min_count ≥ 10`.
* Highlighted verdicts exist at deployed depth only. A matched-length highlighted
  run would make §6.1 and §6.2 use the same grounding.
