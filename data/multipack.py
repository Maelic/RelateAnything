"""Multi-source training mixture over separately-packed datasets.

Each source is packed independently (its own img_dir, its own local predicate /
category ids). They are trained jointly by loading every source's RelationDataset
with ONE shared vocabulary:

  * predicates  → the union order from build_union_vocab (rel_cat_to_idx);
  * categories  → the union order from build_union_vocab's --category_roots
                  (cat_to_idx). A source excluded from that union (e.g. svg_vg,
                  whose "categories" are free-form captions, not a taxonomy)
                  maps entirely to -1 and skips the category-dependent paths
                  (object-aux loss, cooc soft mask) — by design, not a gap.

Sources have wildly different sizes (MEGASG ~463K vs SpatialSense ~4K), so a
plain concat would drown the small sources. `DistributedWeightedSampler` draws
each epoch from a per-sample multinomial whose mass realises a target
per-source fraction, and shards the draw across DDP ranks the same way
DistributedSampler does (identical seed per epoch, rank takes its stride).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from data.multiscale import ResConcatDataset
from data.relation_dataset import RelationDataset


def build_mixture_datasets(
    base_root: str,
    extra_roots: Sequence[str],
    union_predicates: List[str],
    resolution: int,
    max_objects: int,
    union_categories: Optional[List[str]] = None,
    val_root: Optional[str] = None,
    augment: float = 0.0,
    geometric_weight: float = 1.0,
    drop_geometric_roots: Optional[Sequence[str]] = None,
    cat_aliases: Optional[dict] = None,
    exclude_ids: Optional[set] = None,
    rasters: Optional[str] = None,
    mask_dropout: float = 0.0,
) -> Tuple[ConcatDataset, RelationDataset, np.ndarray, List[str]]:
    """Return (train_concat, val_ds, source_of_index, source_names).

    ``source_of_index`` is an int array aligned with the concat's global index
    space (0 = base, 1.. = extra_roots order), for the weighted sampler.
    ``union_categories`` is the shared category vocab (default: base_root's own
    categories only, the pre-fix behavior). ``val_root`` supplies the val split
    (default: base_root); the base train pack may be train-only (e.g.
    megasg_clean), so point val at a pack that has a val/ split (e.g. the
    original megasg pack — its val is not leak-affected).
    """
    rel_cat_to_idx = {p: i for i, p in enumerate(union_predicates)}
    base_meta = json.load(open(Path(base_root) / "train" / "meta.json"))
    cat_names = union_categories if union_categories is not None \
        else base_meta["categories"]
    cat_to_idx = {n: i for i, n in enumerate(cat_names)}
    # Aliases map a source's free-form box strings onto EXISTING union
    # categories (see training/build_cat_aliases.py). They add spellings, never
    # taxonomy entries, so W_obj / cooc / every other artifact stay valid.
    if cat_aliases:
        n_added = 0
        for src_name, union_name in cat_aliases.items():
            if src_name not in cat_to_idx and union_name in cat_to_idx:
                cat_to_idx[src_name] = cat_to_idx[union_name]
                n_added += 1
        print(f"[mixture] category aliases: +{n_added:,} spellings mapped onto "
              f"the {len(cat_names):,}-entry union taxonomy (taxonomy unchanged)")
    val_root = val_root or base_root

    roots = [base_root, *extra_roots]
    source_names = [os.path.basename(os.path.normpath(r)) for r in roots]
    dg = set(drop_geometric_roots or ())
    if dg == {"all"}:
        dg = set(source_names)
    unknown = dg - set(source_names)
    assert not unknown, (
        f"--drop_geometric names {sorted(unknown)} which are not sources "
        f"in this run ({source_names}); check the spelling")

    subsets: List[RelationDataset] = []
    source_of_index_parts: List[np.ndarray] = []
    for si, r in enumerate(roots):
        ds = RelationDataset(
            root=r, split="train", resolution=resolution,
            max_objects=max_objects,
            cat_to_idx=cat_to_idx, rel_cat_to_idx=rel_cat_to_idx,
            augment=augment,       # train only — val stays deterministic below
            geometric_weight=geometric_weight,   # train only, same reason
            drop_geometric=source_names[si] in dg,
            exclude_ids=exclude_ids,
            rasters=rasters,
            mask_dropout=mask_dropout,   # train only — val is scored both ways
            source_idx=si,
        )
        subsets.append(ds)
        source_of_index_parts.append(np.full(len(ds), si, dtype=np.int64))
        # Geometric share matters when --geometric_weight != 1: GQA is ~88%
        # auto-derived left/right, so the downweight lands almost entirely on
        # that source. Print it so the arm is auditable from the log.
        geo = float((np.asarray(ds.rels[:, 3]) & 2).astype(bool).mean()) \
            if len(ds.rels) else 0.0
        print(f"[mixture] {source_names[si]:16s} {len(ds):>7,} imgs"
              f"  (dropped {ds.n_rels_oov_dropped:,} OOV rels"
              + (f", {ds.n_excluded:,} held-out imgs" if ds.n_excluded else "")
              + ")"
              f"  geometric {100 * geo:.1f}%"
              + ("  -> DROPPED" if source_names[si] in dg
                 else (f" @w={geometric_weight}" if geometric_weight != 1.0 else "")))

    # ResConcatDataset (not stock ConcatDataset) so an (index, resolution)
    # pair from MultiScaleBatchSampler survives the sub-dataset dispatch.
    train_concat = ResConcatDataset(subsets)
    source_of_index = np.concatenate(source_of_index_parts)
    assert len(source_of_index) == len(train_concat)

    val_ds = RelationDataset(
        root=val_root, split="val", resolution=resolution,
        max_objects=max_objects,
        cat_to_idx=cat_to_idx, rel_cat_to_idx=rel_cat_to_idx,
        rasters=rasters,             # val keeps masks ON; dropout stays 0
    )
    return train_concat, val_ds, source_of_index, source_names


def fractions_from_temperature(counts: Sequence[int], alpha: float) -> np.ndarray:
    """Size-proportional-with-temperature source fractions: ``frac_s ∝ N_s^alpha``.

    The interpretable knob for multi-source repetition (mT5 / XLM-R style):

    * ``alpha = 1.0`` — fractions exactly proportional to size. Every IMAGE is
      equally likely, so no source is oversampled at all: each source is passed
      over at the same rate. Maximum unique-data coverage, minimum tail weight.
    * ``alpha = 0.5`` — square-root sampling. The usual compromise: small
      sources get lifted well above proportional but nowhere near uniform.
    * ``alpha = 0.0`` — every source contributes equally regardless of size,
      i.e. maximum oversampling of the small ones.

    With the hand-set 0.50/0.20/0.25/0.05 fractions, spatialsense's 4,274
    images were drawn ~11.6x more often per image than megasg's 463,657 — over
    a long run that memorizes the small sources. alpha makes that tradeoff
    explicit and sweepable instead of implicit in four hand-picked numbers.
    """
    c = np.asarray(counts, dtype=np.float64)
    w = np.where(c > 0, np.power(np.maximum(c, 1.0), float(alpha)), 0.0)
    return w / w.sum()


def sample_weights_from_fractions(
    source_of_index: np.ndarray,
    target_fractions: Sequence[float],
    max_passes: Optional[float] = None,
    draws_per_epoch: Optional[int] = None,
    verbose: bool = True,
) -> np.ndarray:
    """Per-sample weight so that, in expectation, source s contributes
    ``target_fractions[s]`` of drawn samples regardless of its raw size.

    weight(sample in source s) = target_fractions[s] / N_s.

    ``max_passes`` caps how many times a source's images may be drawn per
    epoch, redistributing the surplus to uncapped sources. Fixed fractions
    massively oversample small sources: at 0.50/0.20/0.25/0.05 over
    megasg/svg_vg/gqa/spatialsense with 50K draws/epoch, spatialsense's 4,274
    images are drawn 4.9x per epoch while megasg's 463K are drawn 0.05x — so
    over a long run the small sources are memorized while the big one is barely
    seen once. Requires ``draws_per_epoch`` to convert fractions into passes.
    """
    counts = np.bincount(source_of_index, minlength=len(target_fractions))
    frac = np.asarray(target_fractions, dtype=np.float64)
    frac = frac / frac.sum()

    if max_passes is not None and draws_per_epoch:
        # passes_s = draws_per_epoch * frac_s / N_s ; cap it, give back the
        # surplus to sources still under the cap, in proportion to their
        # current fraction. Iterate — capping one source can push another over.
        for _ in range(len(frac)):
            with np.errstate(divide="ignore", invalid="ignore"):
                passes = np.where(counts > 0,
                                  draws_per_epoch * frac / np.maximum(counts, 1),
                                  0.0)
            over = passes > max_passes + 1e-12
            if not over.any():
                break
            capped = np.where(counts > 0, max_passes * counts / draws_per_epoch, 0.0)
            surplus = float((frac[over] - capped[over]).sum())
            frac[over] = capped[over]
            room = ~over & (counts > 0)
            if not room.any() or surplus <= 0:
                break
            frac[room] += surplus * (frac[room] / frac[room].sum())
        frac = frac / frac.sum()
        if verbose:
            passes = np.where(counts > 0,
                              draws_per_epoch * frac / np.maximum(counts, 1), 0.0)
            print(f"[mixture] max_passes={max_passes} → effective fractions "
                  + ", ".join(f"{f:.3f} ({p:.2f} passes)"
                              for f, p in zip(frac, passes)))

    per_source_w = np.where(counts > 0, frac / np.maximum(counts, 1), 0.0)
    w = per_source_w[source_of_index]
    return (w / w.sum()).astype(np.float64)


class DistributedWeightedSampler(Sampler[int]):
    """Weighted-with-replacement sampling, sharded across DDP ranks.

    Each epoch every rank builds the SAME global multinomial draw (seeded by
    ``seed + epoch``) of length ``num_samples`` (rounded up to a multiple of
    ``num_replicas``), then takes ``[rank::num_replicas]``. With replacement,
    so small upsampled sources repeat within an epoch by design.
    """

    def __init__(
        self,
        weights: np.ndarray,
        num_replicas: int = 1,
        rank: int = 0,
        num_samples: Optional[int] = None,
        seed: int = 0,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = max(1, num_replicas)
        self.rank = rank
        n = num_samples if num_samples is not None else len(weights)
        # pad the global draw to a multiple of world size, like DistributedSampler
        self.total_size = int(np.ceil(n / self.num_replicas)) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(self.weights, self.total_size,
                                replacement=True, generator=g)
        yield from idx[self.rank:self.total_size:self.num_replicas].tolist()

    def __len__(self) -> int:
        return self.num_samples
