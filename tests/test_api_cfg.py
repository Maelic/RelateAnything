"""Config round-trip: checkpoint args -> RelSGGConfig.

_cfg_from_args once silently dropped fields that CREATE PARAMETERS
(patch_size, beta_relatedness, lambda_sigmoid) — a rebuilt model was correct
only for checkpoints that happened to use the defaults, and a v44-style
sigmoid/beta checkpoint would load without its extra modules. The fixture is
v43's real args.json; the synthetic case flips the once-dropped fields to
non-defaults and asserts they survive.
"""
import json
import os

from relsgg.api import _cfg_from_args

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "args_v43.json")


def test_v43_fixture_roundtrip():
    a = json.load(open(FIXTURE))
    cfg = _cfg_from_args(a, backbone_pretrained=False)
    assert cfg.backbone_type == a["backbone_type"]
    assert cfg.lora_rank == a["lora_rank"]
    assert cfg.d_model == a["d_model"]
    assert cfg.dual_spatial_head == a["dual_spatial_head"]
    # the once-dropped trio, at v43's values
    assert cfg.patch_size == 16
    assert cfg.beta_relatedness is False
    assert cfg.lambda_sigmoid == 0.0


def test_nondefault_parameter_creating_fields_survive():
    a = json.load(open(FIXTURE))
    a.update(patch_size=14, beta_relatedness=True, lambda_sigmoid=0.25,
             n_heads=4, ffn_ratio=4.0)
    cfg = _cfg_from_args(a, backbone_pretrained=False)
    assert cfg.patch_size == 14
    assert cfg.beta_relatedness is True
    assert cfg.lambda_sigmoid == 0.25
    assert cfg.n_heads == 4
    assert cfg.ffn_ratio == 4.0


def test_missing_optional_fields_fall_back():
    # Old checkpoints predate these args entirely — .get defaults must hold.
    a = json.load(open(FIXTURE))
    for k in ("patch_size", "beta_relatedness", "lambda_sigmoid",
              "n_heads", "ffn_ratio"):
        a.pop(k, None)
    cfg = _cfg_from_args(a, backbone_pretrained=False)
    assert cfg.patch_size == 16
    assert cfg.beta_relatedness is False
    assert cfg.lambda_sigmoid == 0.0
