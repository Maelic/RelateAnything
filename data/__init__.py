"""Data loading for RelAnything training (packed COCO-SGG datasets)."""

from .relation_dataset import RelationDataset, collate_fn

__all__ = ["RelationDataset", "collate_fn"]
