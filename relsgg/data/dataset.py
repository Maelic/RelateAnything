"""Packed COCO-SGG relation dataset.

Reads the memmap format written by ``training/pack_megasg.py`` (one directory
per split with meta.json / file_names.json / img_meta.npy / boxes.npy /
box_cats.npy / rels.npy). All id resolution and box normalisation happened at
pack time; this loader only decodes images and slices arrays, so worker
startup is instant and resident memory stays ~flat regardless of dataset size.

Batch contract (consumed by relsgg.training.engine and relsgg.model.RelSGG):

    images      [B, 3, H, W]  float32 in [0, 1] (square-resized)
    boxes       [B, max_N, 4] normalized cxcywh, zero-padded
    box_counts  [B]           valid boxes per image
    targets     list of dicts per image:
        relations      LongTensor [R, 3]  (sub_idx, obj_idx, pred_label)
        rel_flags      LongTensor [R]     bit0 spatial, bit1 geometric,
                                          bits2+ round (see pack meta)
        rel_weights    FloatTensor [R]    per-relation loss weight
        entity_labels  LongTensor [N]     contiguous object-category index

NOTE deliberately no horizontal-flip augmentation: the vocabulary contains
directional predicates ("to the left of", "to the right of") that a flip
silently falsifies.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


from.rasters import box_raster as _box_raster


class TargetList(list):
    """Per-image target dicts, carrying the batched region rasters (``cov``,
    ``fill``) as attributes when the loader was built with masks."""

    cov = None
    fill = None

FLAG_SPATIAL = 1
FLAG_GEOMETRIC = 2

# ImageNet-ish luma weights, for saturation jitter around the grey image.
_LUMA = torch.tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def _photometric_jitter(image: torch.Tensor, strength: float) -> torch.Tensor:
    """Brightness / contrast / saturation jitter on a [3,H,W] float image in
    [0,1]. Geometry-preserving, so box coordinates remain valid — that is the
    whole reason only photometric ops are used here (see RelationDataset.augment).

    ``strength`` is the half-width of the uniform factor range, e.g. 0.4 draws
    each factor from U(0.6, 1.4). Uses the ambient RNG so DataLoader workers
    (seeded per worker per epoch by torch) decorrelate naturally.
    """
    if strength <= 0.0:
        return image

    def _f() -> float:
        return float(1.0 + (torch.rand(()) * 2.0 - 1.0) * strength)

    image = image * _f()                                    # brightness
    mean = image.mean(dim=(1, 2), keepdim=True)
    image = (image - mean) * _f() + mean                    # contrast
    grey = (image * _LUMA.to(image.dtype)).sum(dim=0, keepdim=True)
    image = (image - grey) * _f() + grey                    # saturation
    return image.clamp_(0.0, 1.0)


class RelationDataset(Dataset):
    """Memmap-backed SGCls dataset over a packed COCO-SGG split.

    Args:
        root:            Packed dataset root containing ``<split>/meta.json``
                         (build with ``training/pack_megasg.py``).
        split:           ``"train"`` or ``"val"``.
        resolution:      Square resize target (must be divisible by the ViT
                         patch size).
        max_objects:     Additional per-image box cap on top of the pack-time
                         cap; relations referencing trimmed boxes are dropped.
        cat_to_idx:      Optional category mapping from another split (train)
                         to keep entity label ids consistent.
        rel_cat_to_idx:  Optional predicate mapping from another split. Own
                         predicates are remapped into it; relations whose
                         predicate string is absent are dropped and counted
                         in ``n_rels_oov_dropped`` (they are the natural
                         out-of-vocabulary set — evaluate them separately).
        geometric_weight: Loss weight for relations flagged as derived from
                         box geometry rather than annotated. Default 1.0.
        exclude_ids: File-name stems to drop entirely (held-out eval images).
    """

    def __init__(
        self,
        root: str,
        split: str,
        resolution: int = 224,
        max_objects: int = 100,
        cat_to_idx: Optional[Dict[str, int]] = None,
        rel_cat_to_idx: Optional[Dict[str, int]] = None,
        geometric_weight: float = 1.0,
        augment: float = 0.0,
        drop_geometric: bool = False,
        exclude_ids: Optional[Set[str]] = None,
        rasters: Optional[str] = None,
        mask_dropout: float = 0.0,
        source_idx: int = -1,
) -> None:
        # Position of this pack in the --data_roots mixture (-1 = single source).
        # Stamped into every target as "src" so source-aware losses (per-source
        # contrast-column masking, --restrict_neg_sources) know where an anchor came from.
        self.source_idx = int(source_idx)
        self.split_dir = Path(root) / split
        meta_path = self.split_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"{meta_path} not found — this loader reads packed datasets. "
                f"Build one with: python training/pack_megasg.py --preset <name> "
                f"(see training/pack_megasg.py --help)"
)
        self.meta = json.load(open(meta_path))
        self.img_dir = Path(self.meta["img_dir"])
        self.resolution = resolution
        self.max_objects = max_objects
        self.geometric_weight = geometric_weight
        self.drop_geometric = drop_geometric
        # Photometric jitter only: brightness, contrast and saturation cannot
        # move a box. Images are resized to a square without preserving the
        # aspect ratio; that resize is the identity in normalised coordinates,
        # so boxes, rasters and the scene position encoding are unchanged.
        self.augment = float(augment)

        # ---- region rasters (optional) ----------------------------------
        # cov.npy holds a g x g coverage raster per annotated box, indexed
        # through cov_index by ABSOLUTE box offset (so view-packs such as
        # megasg_proxy50k, whose img_meta points into a parent boxes.npy,
        # resolve correctly). A box with no raster is not "missing data" — the
        # loader rasterizes the rectangle, which is that region's true shape.
        self.cov = self.cov_index = self.fill = None
        self.cov_res = 32
        self.mask_dropout = float(mask_dropout)
        if rasters:
            # Keyed by the pack DIRECTORY name, matching build_mask_rasters'
            # `ds:split` spec — meta["dataset"] disagrees for derived packs
            # (runs/packed/megasg_proxy50k declares dataset="megasg_clean_proxy50k").
            pack_name = Path(root).name
            rd = Path(rasters) / pack_name / split
            if (rd / "cov.npy").exists():
                self.cov = np.load(rd / "cov.npy", mmap_mode="r")
                self.cov_index = np.load(rd / "cov_index.npy")
                self.fill = np.load(rd / "fill.npy")
                self.cov_res = int(json.load(open(rd / "meta.json"))["res"])
            else:
                raise FileNotFoundError(
                    f"--rasters given but {rd/'cov.npy'} not found. Build with: "
                    f"python datagen/build_mask_rasters.py --packs "
                    f"{pack_name}:{split}")

        self.file_names: List[str] = json.load(
            open(self.split_dir / "file_names.json")
)
        self.img_meta = np.load(self.split_dir / "img_meta.npy")
        self.boxes = np.load(self.split_dir / "boxes.npy", mmap_mode="r")
        self.box_cats = np.load(self.split_dir / "box_cats.npy", mmap_mode="r")
        self.rels = np.load(self.split_dir / "rels.npy", mmap_mode="r")

        # ---- held-out image removal -------------------------------------
        # Drops whole images by file-name stem. Rows are removed from img_meta
        # and file_names only; boxes/box_cats/rels keep their original layout,
        # and img_meta carries absolute (offset, count) pairs into them, so the
        # surviving rows still address the right slices. Stems are compared
        # literally, which is why the exclusion list must spell out every id
        # space a pack might use (see training/build_indoorvg_holdout.py: the
        # same VG photo is named "2351750" in vg_raw and "000000123456" in
        # megasg, and matching only one of the two silently leaves the leak in).
        self.n_excluded = 0
        if exclude_ids:
            keep = np.array(
                [Path(f).stem not in exclude_ids for f in self.file_names],
                dtype=bool,
)
            self.n_excluded = int((~keep).sum())
            if self.n_excluded:
                self.file_names = [f for f, k in zip(self.file_names, keep) if k]
                self.img_meta = self.img_meta[keep]

        # ---- category mapping (entity labels) --------------------------
        own_cats: List[str] = self.meta["categories"]
        if cat_to_idx is None:
            self.cat_to_idx = {n: i for i, n in enumerate(own_cats)}
            self._cat_remap = None  # identity
        else:
            self.cat_to_idx = cat_to_idx
            self._cat_remap = np.array(
                [cat_to_idx.get(n, -1) for n in own_cats], dtype=np.int64
)

        # ---- predicate mapping ------------------------------------------
        own_preds: List[str] = self.meta["predicates"]
        self.n_rels_oov_dropped = 0
        if rel_cat_to_idx is None:
            self.predicate_names = list(own_preds)
            self.rel_cat_to_idx = {n: i for i, n in enumerate(own_preds)}
            self._pred_remap = None  # identity
        else:
            self.rel_cat_to_idx = rel_cat_to_idx
            self.predicate_names = [None] * len(rel_cat_to_idx)
            for n, i in rel_cat_to_idx.items():
                self.predicate_names[i] = n
            self._pred_remap = np.array(
                [rel_cat_to_idx.get(n, -1) for n in own_preds], dtype=np.int64
)
            n_missing = int((self._pred_remap < 0).sum())
            if n_missing:
                own_counts = self.meta.get("predicate_counts", {})
                self.n_rels_oov_dropped = sum(
                    own_counts.get(n, 0)
                    for j, n in enumerate(own_preds)
                    if self._pred_remap[j] < 0
)
                print(
                    f"[RelationDataset {self.meta['dataset']}/{split}] "
                    f"{n_missing}/{len(own_preds)} predicates absent from the "
                    f"provided vocabulary → {self.n_rels_oov_dropped} relations "
                    f"dropped (out-of-vocabulary set)"
)

    def __len__(self) -> int:
        return len(self.img_meta)

    def load_raw(self, idx: int) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        """Return (PIL image, boxes [N,4] cxcywh-normalized, rels [R,5]) —
        untouched by resolution / remapping. For visualisation scripts."""
        _, _, _, b0, nb, r0, nr = self.img_meta[idx]
        img = Image.open(self.img_dir / self.file_names[idx]).convert("RGB")
        return img, np.array(self.boxes[b0:b0 + nb]), np.array(self.rels[r0:r0 + nr])

    def __getitem__(self, idx):
        # idx may be a plain int, or an (index, resolution) pair emitted by
        # data.multiscale.MultiScaleBatchSampler. The override has to arrive
        # through the index because the resolution must be constant across a
        # BATCH (the images are stacked) while the dataset only ever sees one
        # item at a time. Resizing later in collate_fn would mean resampling an
        # already-resampled image, which throws away exactly the high-frequency
        # detail the large scales exist to provide.
        res = self.resolution
        if isinstance(idx, tuple):
            idx, res = int(idx[0]), int(idx[1])
        _, _, _, b0, nb, r0, nr = self.img_meta[idx]
        nb = min(int(nb), self.max_objects)

        img = Image.open(self.img_dir / self.file_names[idx]).convert("RGB")
        img = img.resize((res, res), Image.BILINEAR)
        image = torch.from_numpy(
            np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
)
        if self.augment > 0.0:
            image = _photometric_jitter(image, self.augment)

        # No box remapping: a square resize is the IDENTITY in normalized
        # cxcywh, so the packed coordinates are already correct for it.
        boxes = torch.from_numpy(np.array(self.boxes[b0:b0 + nb], dtype=np.float32))

        cov = fill = None
        if self.cov is not None:
            # Modality dropout, per IMAGE not per object: at deployment you
            # either ran a segmenter on the frame or you did not. Dropping the
            # mask means replacing the region with its bounding rectangle —
            # a data-level substitution, not an architectural switch, so the
            # model never learns there were two modes.
            drop = self.mask_dropout > 0.0 and random.random() < self.mask_dropout
            g = self.cov_res
            cov = np.empty((nb, g, g), dtype=np.uint8)
            fill = np.ones(nb, dtype=np.float32)
            for j in range(nb):
                row = -1 if drop else int(self.cov_index[b0 + j])
                if row < 0:
                    cx, cy, w, h = (float(v) for v in self.boxes[b0 + j])
                    cov[j] = np.clip(_box_raster(cx - w / 2, cy - h / 2,
                                                 cx + w / 2, cy + h / 2, g)
                                     * 255.0, 0, 255).astype(np.uint8)
                else:
                    cov[j] = self.cov[row]
                    fill[j] = float(self.fill[row])
            cov = torch.from_numpy(cov)
            fill = torch.from_numpy(fill)

        cats = np.array(self.box_cats[b0:b0 + nb], dtype=np.int64)
        if self._cat_remap is not None:
            cats = self._cat_remap[cats]

        rels = np.array(self.rels[r0:r0 + nr], dtype=np.int64)  # [R, 5]
        if rels.size:
            keep = (rels[:, 0] < nb) & (rels[:, 1] < nb)
            rels = rels[keep]
        if rels.size and self._pred_remap is not None:
            rels[:, 2] = self._pred_remap[rels[:, 2]]
            rels = rels[rels[:, 2] >= 0]
        # Drop auto-derived (source=geometric) edges outright rather than
        # down-weighting them. For GQA these are exactly its 1.6M
        # left/right edges — 88.5% of that source — which otherwise dominate
        # the mixture. Dropped edges become unlabelled pairs, which the
        # co-occurrence soft-negative table already handles.
        if rels.size and self.drop_geometric:
            rels = rels[(rels[:, 3] & FLAG_GEOMETRIC) == 0]

        if rels.size:
            relations = torch.from_numpy(rels[:,:3])
            rel_flags = torch.from_numpy(rels[:, 3])
            rel_weights = torch.where(
                (rel_flags & FLAG_GEOMETRIC).bool(),
                torch.full((len(rels),), self.geometric_weight),
                torch.ones(len(rels)),
)
        else:
            relations = torch.zeros((0, 3), dtype=torch.long)
            rel_flags = torch.zeros((0,), dtype=torch.long)
            rel_weights = torch.zeros((0,), dtype=torch.float32)

        target = {
            "relations": relations,
            "rel_flags": rel_flags,
            "rel_weights": rel_weights,
            "entity_labels": torch.from_numpy(cats),
            # Row index into this pack, so an evaluator can join against a
            # sidecar keyed by image (HaystackEvaluator's negative cells).
            # Every other consumer ignores it.
            "index": int(idx),
            "src": self.source_idx,
        }
        if cov is not None:
            target["cov"] = cov
            target["fill"] = fill
        return image, boxes, target


def collate_fn(batch, pad_to: "int | None" = None):
    """Pad boxes to the batch max (or to ``pad_to`` slots) and stack.

    Returns ``(images [B,3,H,W], boxes [B,N,4], box_counts [B], targets)``.
    With ``pad_to`` every batch has the same box dimension, which keeps the
    pair sampler's shapes constant for compiled and exported graphs; padded
    slots are masked by ``box_counts`` either way, so nothing else changes.
    """
    images = torch.stack([b[0] for b in batch])
    box_counts = torch.tensor([b[1].shape[0] for b in batch], dtype=torch.long)
    max_n = pad_to if pad_to is not None else max(1, int(box_counts.max()))
    assert pad_to is None or int(box_counts.max()) <= pad_to, (
        f"batch contains an image with {int(box_counts.max())} boxes > "
        f"pad_to={pad_to} — raise pad_to (should be >= max_objects)")

    boxes = torch.zeros(len(batch), max_n, 4, dtype=torch.float32)
    for i, (_, bx, _) in enumerate(batch):
        boxes[i,: bx.shape[0]] = bx

    targets = TargetList(b[2] for b in batch)

    if batch and "cov" in batch[0][2]:
        g = batch[0][2]["cov"].shape[-1]
        cov = torch.zeros(len(batch), max_n, g, g, dtype=torch.uint8)
        fill = torch.ones(len(batch), max_n, dtype=torch.float32)
        for i, (_, _, t) in enumerate(batch):
            n = t["cov"].shape[0]
            cov[i,:n] = t["cov"]
            fill[i,:n] = t["fill"]
        targets.cov = cov
        targets.fill = fill

    return images, boxes, box_counts, targets
