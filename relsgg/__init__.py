"""RelAnything — label-free re-parametrizable relation head.

Public API
----------
    from relsgg import RelSGG, RelSGGConfig

    model = RelSGG()
    model.encode_vocabulary(["above", "behind", "next to", "holding"])
    model.reparameterize()                  # fuses text encoder; zero LLM overhead

    # Inference
    triplets = model.predict(images, boxes) # List[List[dict]]

    # Re-parametrize to a new vocabulary at any time
    model.encode_vocabulary(["in front of", "behind"])
    model.reparameterize()
"""

# Lazy (PEP 562) so torch-free submodules stay torch-free: the laptop deploy
# path imports relsgg.scoring / relsgg.decompose with no torch installed, and
# an eager `from .model import ...` here would pull torch in at package import.
__all__ = ["RelSGG", "RelSGGConfig"]


def __getattr__(name):
    if name in __all__:
        from . import model
        return getattr(model, name)
    raise AttributeError(f"module 'relsgg' has no attribute {name!r}")
