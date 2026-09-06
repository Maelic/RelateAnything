"""RelateAnything: open-vocabulary relation prediction from boxes or masks.

    from relsgg import RelateAnything
    ra = RelateAnything.from_pretrained("maelic/relsgg-vits16plus")
    triplets = ra.predict(image, boxes_xyxy)

``relsgg.scoring`` and ``relsgg.decompose`` are torch-free so the ONNX
runtime can import them; the package therefore loads its model lazily.
"""
from.config import RelSGGConfig
from.vocabulary import DEFAULT_PREDICATES, TRAIN_TEMPLATES

__all__ = ["RelateAnything", "RelSGG", "RelSGGConfig", "DEFAULT_PREDICATES", "TRAIN_TEMPLATES"]


def __getattr__(name):
    if name == "RelateAnything":
        from.api import RelateAnything
        return RelateAnything
    if name == "RelSGG":
        from.model import RelSGG
        return RelSGG
    raise AttributeError(f"module 'relsgg' has no attribute {name!r}")
