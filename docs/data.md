# Data

## RA-4M

The training corpus. **474,413 images, 4,282,531 relations, 10,102 distinct
free-text predicates**, machine-generated and geometrically verified.

| | |
|---|---|
| images | Objects365 / COCO / OpenImages re-crawl — **identifiers only, never redistributed** |
| annotator | `gemma-4-26B-A4B-it` via a vLLM pipeline ([`datagen/`](../datagen/)) |
| filtering | deterministic geometric gates, not model self-scoring |
| object categories | 497, used for the *aux* object loss only — never as model input |
| predicates | free text; **synonyms are deliberately never collapsed** |

That last row is a project rule, not an oversight. `riding` and `riding on` stay
separate columns. Surface-form diversity *is* the label space: collapsing
synonyms into a canonical form destroys exactly the signal an open-vocabulary
model is supposed to have. Canonical groups exist, but they are used for loss
computation (so a synonym is never scored as a negative), never for emitted
data.

The deployed vocabulary in the released models is larger than RA-4M's own —
**19,103 strings**, the union over RA-4M, raw Visual Genome, and HICO-DET.

### Getting it

The corpus is the dataset repository
[`maelic/RA-4M`](https://huggingface.co/datasets/maelic/RA-4M): the relation
annotations (`annotations/ra4m_{train,val}.jsonl.gz`, one record per image),
their statistics, the image manifest with the extractor that rebuilds the
image set from [JosephZ/mega_1m](https://huggingface.co/datasets/JosephZ/mega_1m),
the three generator prompts, and the training-ready packs. The pipeline that
produced it is [`datagen/`](../datagen/). See
[installation](installation.md#get-the-data) for the download commands.

## Other sources

| pack | role |
|---|---|
| `vg_raw` | raw Visual Genome relations, 17,742 predicates — mixture partner |
| `hicodet` | HICO-DET verbs — mixture partner, human-object interaction coverage |
| `vg150`, `psg`, `indoorvg`, `haystack`, `spatialsense` | evaluation only |

The shipped models train on `megasg_clean + vg_raw + hicodet`.

## The pack format

Everything is packed to a memmap-backed directory, one per split, written by
[`training/pack_megasg.py`](../training/pack_megasg.py):

```
runs/packed/<name>/<split>/
  meta.json        predicates, categories, counts, provenance, flags legend
  file_names.json  image file names
  img_meta.npy     per-image width/height/offsets
  boxes.npy        normalized boxes
  box_cats.npy     object-category indices
  rels.npy         (sub_idx, obj_idx, pred_label) + flags + weights
```

All id resolution and box normalization happen at pack time. The loader
([`relsgg/data/dataset.py`](../relsgg/data/dataset.py)) only decodes images
and slices arrays, so worker startup is instant and resident memory stays flat
regardless of corpus size.

Batch contract:

```
images      [B, 3, H, W]  float32 in [0, 1], square-resized
boxes       [B, max_N, 4] normalized cxcywh, zero-padded
box_counts  [B]           valid boxes per image
targets     per image: relations [R,3], rel_flags [R], rel_weights [R],
                       entity_labels [N]
```

**There is no horizontal-flip augmentation, on purpose.** The vocabulary
contains directional predicates — `to the left of`, `to the right of` — and a
flip silently falsifies them.

## Mixtures

Sources are packed independently, then trained jointly through one shared
vocabulary ([`relsgg/data/multipack.py`](../relsgg/data/multipack.py)). Sizes differ by two
orders of magnitude, so a plain concatenation would drown the small sources; a
`DistributedWeightedSampler` draws each epoch from a per-sample multinomial
that realizes target per-source fractions, sharded across DDP ranks.

```bash
--data_roots runs/packed/megasg_clean runs/packed/vg_raw runs/packed/hicodet
--mix_fractions 0.7274 0.063 0.2096
```

> **`mix_fractions` is per-image, and the loss is per-relation.** Images from
> different sources carry very different relation counts, so a nominal 50 %
> source can contribute 16 % of the actual supervision. If you are reasoning
> about how much a source *teaches*, convert to relation share first. This has
> caused a wrong conclusion at least once — see [pitfalls](pitfalls.md).

## Adding a source

1. Convert to COCO-SGG JSON. There are converters to copy from:
   `training/convert_hicodet.py`, `convert_vg_raw.py`, `convert_spatialsense.py`,
   `convert_openimages_ext.py`.
2. Pack it: `python training/pack_megasg.py --ann <json> --img_dir <dir> --out runs/packed/<name>`.
3. Rebuild the union vocabulary: `python training/build_union_vocab.py`.
4. Re-encode predicate embeddings for the union with `training/encode_vocab_text.py`.
5. Check for evaluation leakage — see below.
6. Add it to `--data_roots` / `--mix_fractions`.

Then read [`docs/objective.md`](objective.md)
**before** you train on it. A source's *label space* is a claim about what
absence means: HICO-DET annotates 117 verbs and nothing else, so treating `on`
as a negative for a HICO pair teaches the model something false. That is what
`--restrict_neg_sources` exists for, and it is in the shipped recipe.

## Evaluation leakage

Non-negotiable check before any new source enters the mixture. Image ids leak
across corpora under different spellings — IndoorVG holdout images appear in
both `vg_raw` and `megasg` under two id conventions, and the datamix registry
does not catch it by itself.

```bash
python training/audit_image_overlap.py --pack runs/packed/<name> --against runs/packed/psg...
python train.py... --exclude_ids runs/datamix/indoorvg_holdout.json     # shipped in maelic/OV-SGG-Bench
```

Vocabulary overlap is a separate axis and equally load-bearing — see
[evaluation](evaluation.md).

## Licensing

RA-4M annotations are Gemma-generated and carry the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) notice; downstream use
must comply with the Gemma Prohibited Use Policy. Images are not redistributed
and remain under their source datasets' licenses. `vg_raw` derives from Visual
Genome (CC BY 4.0). Full detail:
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
