"""Multi-source training mixture over separately packed datasets.

Each source is packed on its own; they are trained jointly under one union
vocabulary of predicates and categories (``training/build_union_vocab.py``).
Sources differ in size by orders of magnitude, so ``DistributedWeightedSampler``
draws each epoch from a per-sample multinomial that realises the requested
per-source fractions, sharded across processes like ``DistributedSampler``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Sampler

from.multiscale import ResConcatDataset
from.dataset import RelationDataset


def build_mixture_datasets(
    base_root: str,
    extra_roots: Sequence[str],
    union_predicates: List[str],
    resolution: int,
    max_objects: int,
    union_categories: Optional[List[str]] = None,
    val_root: Optional[str] = None,
    augment: float = 0.0,
    exclude_ids: Optional[set] = None,
    rasters: Optional[str] = None,
    mask_dropout: float = 0.0,
) -> Tuple[ConcatDataset, RelationDataset, np.ndarray, List[str]]:
    """Return ``(train_concat, val_ds, source_of_index, source_names)``.

    ``source_of_index`` maps every index of the concatenated training set to
    its source (0 = base, then ``extra_roots`` in order). ``val_root``
    supplies the validation split (default: the base root).
    """
    rel_cat_to_idx = {p: i for i, p in enumerate(union_predicates)}
    base_meta = json.load(open(Path(base_root) / "train" / "meta.json"))
    cat_names = union_categories if union_categories is not None else base_meta["categories"]
    cat_to_idx = {n: i for i, n in enumerate(cat_names)}
    val_root = val_root or base_root

    roots = [base_root, *extra_roots]
    source_names = [os.path.basename(os.path.normpath(r)) for r in roots]
    subsets: List[RelationDataset] = []
    source_of_index_parts: List[np.ndarray] = []
    for si, r in enumerate(roots):
        ds = RelationDataset(
            root=r, split="train", resolution=resolution, max_objects=max_objects,
            cat_to_idx=cat_to_idx, rel_cat_to_idx=rel_cat_to_idx, augment=augment,
            exclude_ids=exclude_ids, rasters=rasters, mask_dropout=mask_dropout,
            source_idx=si)
        subsets.append(ds)
        source_of_index_parts.append(np.full(len(ds), si, dtype=np.int64))
        print(f"[mixture] {source_names[si]:16s} {len(ds):>7,} images"
              f"  (dropped {ds.n_rels_oov_dropped:,} out-of-vocabulary relations"
              + (f", {ds.n_excluded:,} held-out images" if ds.n_excluded else "") + ")")

    # ResConcatDataset keeps the (index, resolution) pairs of the multi-scale
    # batch sampler intact through the sub-dataset dispatch.
    train_concat = ResConcatDataset(subsets)
    source_of_index = np.concatenate(source_of_index_parts)
    val_ds = RelationDataset(root=val_root, split="val", resolution=resolution,
                             max_objects=max_objects, cat_to_idx=cat_to_idx,
                             rel_cat_to_idx=rel_cat_to_idx, rasters=rasters)
    return train_concat, val_ds, source_of_index, source_names


def sample_weights_from_fractions(source_of_index: np.ndarray,
                                  target_fractions: Sequence[float],
                                  draws_per_epoch: Optional[int] = None) -> np.ndarray:
    """Per-sample weights such that source ``s`` contributes
    ``target_fractions[s]`` of the draws in expectation, whatever its size:
    ``weight = fraction_s / N_s``."""
    counts = np.bincount(source_of_index, minlength=len(target_fractions))
    frac = np.asarray(target_fractions, dtype=np.float64)
    frac = frac / frac.sum()
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
