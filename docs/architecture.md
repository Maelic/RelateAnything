# Architecture

## The shape of the problem

A relation model that is genuinely open-vocabulary cannot have a predicate
classifier. A classifier's output layer *is* the vocabulary; changing it means
retraining. So the design constraint is:

> the predicate vocabulary must enter at inference time, as data.

RelateAnything satisfies this by scoring a visual pair representation against
**text embeddings** of the predicates, with exactly one learned transformation
on the classification path — and it is on the visual side.

The second constraint is decoupling. The model receives pixels and boxes and
**never** object class labels, so it cannot learn `person + horse → riding`,
the frequency prior that dominates scene-graph recall. That costs less than it
sounds like: 92 % of the relation logit's variance comes from pair context
rather than from either endpoint.

## The data path

```
image ──▶ DINOv3 backbone ──▶ dense patch features F  (multi-tap, fused)
                                    │
boxes ──┬──▶ SoftSpatialPool ──────▶ v_sub, v_obj, v_union, v_contact
        │
        └──▶ GeoEncoder ───────────▶ g_ij   (19 geometry features)
                                    │
                        pair_proj ──▶ fused pair representation
                                    │
                     PairSampler ───▶ K candidate pairs  (400 → 128)
                                    │
              RelationTransformer ──▶ context-aware pair repr  r
                                    │
              (DeformableRelRead) ──▶ + box-anchored scene read
                                    │
                       VocabHead ───▶ cosine(proj(r), W) ──▶ [K, V] logits
```

`W` is the `[V, d]` matrix of normalized text embeddings for the current
vocabulary. It is **never modified by learned parameters** — there is no
`text_proj`. That is what makes an unseen predicate string work: InfoNCE pushes
`proj(r)` toward the exact text direction of the ground-truth predicate, so
there is no intermediate learned subspace that could only represent the
training vocabulary.

`reparameterize()` seals `W` and verifies normalization. From then on the
forward pass is a matmul, with zero language-model cost at runtime.

## Modules

| module | file | what it does |
|---|---|---|
| Backbone | [`backbone.py`](../relsgg/backbone.py) | DINOv3 ViT or ConvNeXt, LoRA or full fine-tune. Reads **three taps** (`[-6, -3, -1]`) and fuses them with softmax weights |
| Region pooling | [`roi.py`](../relsgg/roi.py) | `SoftSpatialPool` — pools patch features under a box (or a mask) into subject / object / union / contact-zone features |
| Geometry | [`geometry.py`](../relsgg/geometry.py) | 19 pairwise box features, Fourier box-corner tokens, scene positional encoding |
| Pair sampler | [`sampler.py`](../relsgg/sampler.py) | prunes N² pairs to a bounded budget in two stages |
| Relation transformer | [`transformer.py`](../relsgg/transformer.py) | self-attention across pairs (inter-pair dependency) + cross-attention to the scene |
| Deformable read | [`deformable.py`](../relsgg/deformable.py) | box-anchored sparse sampling of the feature map, additive behind a zero-init gate |
| Vocab head | [`vocab.py`](../relsgg/vocab.py) | the open-vocabulary scoring layer and re-parameterization |
| Losses | [`loss.py`](../relsgg/loss.py), [`loss_synonym.py`](../relsgg/loss_synonym.py) | InfoNCE with synonym groups, relatedness BCE, direction hinge, background suppression |
| Score contract | [`scoring.py`](../relsgg/scoring.py) | the one definition of a relation score, shared by eval and deploy |
| Decomposition | [`decompose.py`](../relsgg/decompose.py) | splits one forward pass into spatial and semantic graphs |
| Public API | [`api.py`](../relsgg/api.py) | `RelateAnything` — load, set vocabulary, predict |

Every knob is a field on `RelSGGConfig` in
[`model.py`](../relsgg/model.py), each documented inline with why it exists and
what it measured. That dataclass is the real reference; this page is the map.

## Two scores, multiplied — and why

The model emits two separate quantities per candidate pair:

- **pair existence** (`pair_logit`), from the relatedness head — *is there a
  relation here at all?*
- **predicate identity** (`pred_logit`), from the vocab head — *which one?*

They are fused as `sigmoid(a·(pred + w·pair) + b)`. Splitting them is what lets
"no relation" be supervised at all: absent relations in machine-generated
annotation are *unlabeled*, not false, so the relatedness head is trained with
positive-unlabeled-aware down-weighted negatives rather than hard zeros.

The relatedness term is also the most interesting failure mode in the project.
It raises recall on every annotation-derived benchmark and **lowers** accuracy
on adjudicated negatives, because it is partly a model of *which pairs a human
bothered to annotate* rather than which pairs are related. Hence `pair_weight`
is a knob, not a constant. See [pitfalls](pitfalls.md).

## The pair sampler

Scoring every ordered pair is O(N²) in boxes, and most pairs are nothing. Two
stages cut it down:

1. **Geometry pre-scorer** — a small MLP on raw geometry features only, no
   vision. Keeps the top `geo_budget` (400) pairs.
2. **Learned relatedness** — an asymmetric score
   `s(i,j) = ⟨f_s(v_i), f_o(v_j)⟩/√d`. Keeps the top `final_budget` (128).

Stage 2 is learned rather than a cosine proxy for a specific reason: cosine
similarity selects *similar* objects, not *interacting* ones, and the
interaction-vs-non-interaction confusion is the dominant noise source in
open-vocabulary SGG.

The sampler is not a bottleneck — it recovers 99.79 % of ground-truth positive
pairs, and exhaustive scoring costs only 1.02× — so it is not the place to look
for accuracy. It is fully batched and traceable, which is what makes ONNX
export possible.

## Backbone taps and fusion

The backbone is read at three depths, not just the last layer, and the taps are
fused by learned softmax weights. Without normalization those weights are
confounded by activation scale: the combiner *looks* uniform (.358/.330/.312)
while the tap norms are 52.9 / 192.4 / 668.0, so the fusion is really ~72 %
last-layer. `norm_taps=True` puts a LayerNorm on each tap first, which decouples
importance from scale and is in the shipped recipe.

## Backbone families

| family | LoRA | notes |
|---|---|---|
| DINOv3 ViT-S/16, S/16+, B/16 | yes | the released family |
| DINOv3 ConvNeXt T/S/B/L | no | supported and runs; full fine-tune only, since LoRA has no attention projections to attach to |

A tuned ConvNeXt-T is ViT-S-class in accuracy and the fastest at batch size 1,
but the ViT family ships: the level axis on ConvNeXt measured flat, with all
arms at the noise floor.

## What is *not* in here

- **A detector.** Boxes come from outside. That is the product contract.
- **Object classification.** No class labels in, none out.
- **A language model at inference.** The text encoder runs once, when the
  vocabulary is set.
