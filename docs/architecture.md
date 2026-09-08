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
| Backbone | [`model/backbone.py`](../relsgg/model/backbone.py) | the DINOv3 tower, fully fine-tuned. Reads **three taps** (`[-6, -3, -1]`) and fuses them with softmax weights, plus the interaction block |
| Region pooling | [`model/pooling.py`](../relsgg/model/pooling.py) | `SoftSpatialPool` — pools patch features under a box (or a mask) into subject, object, union and contact-zone features |
| Geometry | [`model/geometry.py`](../relsgg/model/geometry.py) | 19 pairwise region features, Fourier box-corner tokens, scene positional encoding |
| Pair sampler | [`model/sampler.py`](../relsgg/model/sampler.py) | prunes N² pairs to a bounded budget in two stages |
| Relation transformer | [`model/transformer.py`](../relsgg/model/transformer.py) | self-attention across pairs, then cross-attention to the scene and the box tokens |
| Deformable read | [`model/deformable.py`](../relsgg/model/deformable.py) | box-anchored sparse sampling of the feature map, additive behind a zero-initialised gate |
| Vocabulary head | [`model/vocab_head.py`](../relsgg/model/vocab_head.py) | the open-vocabulary scoring layer and its two query experts |
| The network | [`model/relsgg.py`](../relsgg/model/relsgg.py) | composes all of the above, and carries the training objective |
| Configuration | [`config.py`](../relsgg/config.py) | every hyperparameter, defaulting to the released recipe |
| Objective | [`training/losses.py`](../relsgg/training/losses.py) | batch-local contrastive loss with estimated synonym weights, relatedness BCE, direction hinge, background suppression |
| Score contract | [`scoring.py`](../relsgg/scoring.py) | the one definition of a relation score, shared by evaluation and deployment |
| Decomposition | [`decompose.py`](../relsgg/decompose.py) | splits one forward pass into a spatial and a semantic graph |
| Text student | [`text/student.py`](../relsgg/text/student.py) | the encoder that turns predicate strings into the head's vocabulary |
| Public API | [`api.py`](../relsgg/api.py) | `RelateAnything` — load, set vocabulary, predict |

Every hyperparameter is a field on `RelSGGConfig` in
[`config.py`](../relsgg/config.py), and its defaults are the released recipe:
`RelSGGConfig()` builds the model that ships, and a checkpoint's own arguments
round-trip through `config_from_args`.

## Two scores, added — and why

The model emits two separate quantities per candidate pair:

- **pair existence** (`pair_logit`), from the relatedness head — *is there a
  relation here at all?*
- **predicate identity** (`pred_logit`), from the vocab head — *which one?*

They are fused as `sigmoid(a·(pred + w·pair) + b)` — additively, in logit
space. The multiplicative form was measured and is slightly worse (0.904 AUC
against 0.911); relatedness alone reaches 0.748. Splitting them is what lets
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
while the tap norms are 52.9 / 192.4 / 668.0, so an unnormalised fusion is
really about 72 % last layer. Each tap is therefore LayerNorm-ed before the
weighted sum, which separates "how important is this depth" from "how large are
its activations".

## Backbone families

The released models are DINOv3 ViT-S/16, ViT-S/16+ and ViT-B/16, each fully
fine-tuned at a learning rate eight times below the head's. Any tower whose
Hugging Face configuration `AutoModel` can build and whose hidden states the
three taps can read will load; pass it with `--backbone_model`.

## What is *not* in here

- **A detector.** Boxes come from outside. That is the product contract.
- **Object classification.** No class labels in, none out.
- **A language model at inference.** The text encoder runs once, when the
  vocabulary is set.
