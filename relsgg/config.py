"""Model configuration.

The defaults are the released recipe: every published checkpoint was trained
with these values except ``backbone_model``, which names the DINOv3 tower of
that model. ``RelSGGConfig()`` therefore builds a RelateAnything model as
shipped; ``config_from_args`` rebuilds the config a checkpoint was trained
with from the ``args`` dictionary stored inside it.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class RelSGGConfig:
    # -- backbone ---------------------------------------------------------
    backbone_model: str = "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    backbone_pretrained: bool = True   # False: build from config only, weights come
                                       # from the checkpoint (released models)
    patch_size: int = 16

    # -- pair features ----------------------------------------------------
    d_model: int = 512                 # width of the relation head
    pe_num_freqs: int = 16             # Fourier bands per box coordinate
    pe_max_octave: float = 7.0         # top frequency of the Fourier ladder (2**7)

    # -- pair sampler -----------------------------------------------------
    geo_budget: int = 400              # pairs kept by the geometry pre-scorer
    final_budget: int = 128            # pairs kept by the relatedness head (K)
    rel_neg_weight: float = 0.3        # floor weight of unlabelled pairs in the
                                       # relatedness loss (positive-unlabelled data)

    # -- relation transformer and interaction block -----------------------
    n_self_layers: int = 2
    n_cross_layers: int = 2
    n_dep_layers: int = 2
    n_gnd_layers: int = 1
    n_heads: int = 8
    ffn_ratio: float = 2.0
    dropout: float = 0.2

    # -- deformable scene read -------------------------------------------
    deformable_points: int = 4         # sampled points per anchor and head
    deformable_heads: int = 8
    deformable_nulls: int = 2          # learnable non-image slots per (head, anchor)

    # -- text-space head --------------------------------------------------
    text_dim: int = 512                # dimension of the text student's embeddings
    proj_layers: int = 2               # depth of the visual-to-text projection
    logit_scale_init: float = 5.0
    infonce_temp: float = 0.07

    # -- training-time terms (unused at inference) ------------------------
    box_token_dropout: float = 0.3     # per-image chance of hiding the box tokens
    lambda_geo: float = 1.0            # geometry pre-scorer BCE
    lambda_rel: float = 1.0            # relatedness BCE
    lambda_obj: float = 0.1            # object-category alignment of the query parts
    lambda_swap: float = 0.5           # direction hinge between (s,o) and (o,s)
    swap_margin: float = 0.05
    lambda_sigmoid: float = 0.25       # per-cell sigmoid auxiliary
    lambda_bg: float = 0.05            # background suppression on unlabelled pairs
    bg_topk: int = 5
    cfa_prob: float = 0.5              # same-predicate feature mixing probability
    cfa_alpha: float = 1.0             # Beta(alpha, alpha) mixing coefficient

    @property
    def deployed_text_dim(self) -> int:
        return self.text_dim


def config_from_args(args, backbone_pretrained: bool) -> RelSGGConfig:
    """Config of a checkpoint from the ``args`` it stores.

    Every field is read under its own name; fields the checkpoint does not
    carry (or carries as ``None``) keep the released default.
    """
    a = dict(args if isinstance(args, dict) else vars(args))
    kw = {}
    for f in fields(RelSGGConfig):
        if f.name == "backbone_pretrained":
            continue
        v = a.get(f.name)
        if v is None or (f.name == "backbone_model" and not v):
            continue
        kw[f.name] = v
    return RelSGGConfig(backbone_pretrained=backbone_pretrained, **kw)
