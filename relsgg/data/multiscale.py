"""Multi-scale training: one square resolution per batch.

A square resize is the identity in normalised coordinates, so boxes,
coverage rasters and the scene position encoding need no change when the
canvas changes; only the patch grid and the pixel size of objects do. The
resolution therefore rides along with the sample index: the batch sampler
yields ``(index, resolution)`` pairs, seeded by ``seed + epoch`` so every
process draws the same sequence.
"""
from __future__ import annotations

import bisect
from typing import List, Sequence

import torch
from torch.utils.data import ConcatDataset, Sampler


def scale_ladder(base: int, lo: float, hi: float, n: int,
                 patch: int = 16) -> List[int]:
    """``n`` resolutions spanning ``[lo, hi] * base``, each a multiple of ``patch``.

    Snapping to the patch size is mandatory, not cosmetic: the backbone asserts
    H and W are divisible by patch_size (it computes its grid as H // patch).
    Duplicates after snapping are collapsed, so a tight range with a large n
    simply yields fewer distinct scales rather than repeating one.
    """
    if n < 1:
        raise ValueError(f"--multi_scale_n must be >= 1, got {n}")
    if not (0 < lo <= hi):
        raise ValueError(f"bad scale range: lo={lo} hi={hi}")
    if n == 1:
        vals = [(lo + hi) / 2.0]
    else:
        vals = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    out = sorted({max(patch, int(round(base * s / patch)) * patch) for s in vals})
    return out


class ResConcatDataset(ConcatDataset):
    """ConcatDataset that forwards an ``(index, resolution)`` pair intact.

    Stock ConcatDataset bisects on an int index, so a tuple index raises. This
    resolves the sub-dataset from the int part and hands the sub-dataset the pair,
    which is what lets one multi-scale sampler drive a multi-source mixture.
    """

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            return super().__getitem__(idx)
        i, res = int(idx[0]), int(idx[1])
        if i < 0:
            if -i > len(self):
                raise ValueError("absolute index out of range")
            i = len(self) + i
        d = bisect.bisect_right(self.cumulative_sizes, i)
        local = i if d == 0 else i - self.cumulative_sizes[d - 1]
        return self.datasets[d][(local, res)]


class MultiScaleBatchSampler(Sampler[List[tuple]]):
    """Batch an index sampler, attaching one shared resolution per batch.

    Args:
        sampler:     the index sampler to wrap (RandomSampler,
                     DistributedSampler, DistributedWeightedSampler,...).
        batch_size:  items per batch.
        resolutions: the ladder to draw from (see ``scale_ladder``).
        drop_last:   drop a trailing short batch (True matches the train loader).
        seed:        RNG seed; combined with the epoch, and deliberately NOT with
                     the rank, so every rank draws the same resolution sequence.
    """

    def __init__(self, sampler, batch_size: int, resolutions: Sequence[int],
                 drop_last: bool = True, seed: int = 42):
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.resolutions = [int(r) for r in resolutions]
        if not self.resolutions:
            raise ValueError("empty resolution ladder")
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Forwarded to the inner sampler too — train.py only knows about us."""
        self.epoch = int(epoch)
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)

    def _draw(self, g) -> int:
        j = int(torch.randint(len(self.resolutions), (1,), generator=g).item())
        return self.resolutions[j]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        batch: List[int] = []
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                r = self._draw(g)
                yield [(i, r) for i in batch]
                batch = []
        if batch and not self.drop_last:
            r = self._draw(g)
            yield [(i, r) for i in batch]

    def __len__(self) -> int:
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else \
            (n + self.batch_size - 1) // self.batch_size
