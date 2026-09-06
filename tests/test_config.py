"""The configuration defaults are the released recipe, and a checkpoint's
own arguments round-trip through it.

``RelSGGConfig()`` must build the model that ships: a newcomer instantiating
it, and every loader rebuilding a checkpoint, has to land on the same
network. The values below are read from the released ``vits16plus``
configuration in ``training/configs/``.
"""
import json
import os

import pytest

from relsgg.config import RelSGGConfig, config_from_args

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "training", "configs")
RELEASED = [f for f in sorted(os.listdir(CONFIGS)) if f.endswith(".json")]


def _load(name):
    with open(os.path.join(CONFIGS, name)) as fh:
        return json.load(fh)


@pytest.mark.parametrize("name", RELEASED)
def test_released_runs_only_differ_in_documented_fields(name):
    """Every released model uses the default recipe except for its backbone."""
    cfg = config_from_args(_load(name), backbone_pretrained=False)
    default = RelSGGConfig()
    differing = {f for f in RelSGGConfig.__dataclass_fields__
                 if getattr(cfg, f) != getattr(default, f)}
    assert differing <= {"backbone_model", "backbone_pretrained"}, differing


def test_defaults_are_the_shipped_architecture():
    c = RelSGGConfig()
    assert c.d_model == 512 and c.text_dim == 512
    assert c.deformable_points == 4 and c.deformable_heads == 8 and c.deformable_nulls == 2
    assert c.pe_num_freqs == 16 and c.pe_max_octave == 7.0
    assert c.final_budget == 128 and c.geo_budget == 400
    assert c.backbone_pretrained is True


def test_missing_and_null_fields_keep_the_default():
    """Older checkpoints predate some arguments, and argparse writes None for
    unset ones; neither may change the config."""
    a = _load("relsgg-vits16plus.json")
    for k in ("dropout", "proj_layers", "deformable_heads"):
        a.pop(k, None)
    a["pe_max_octave"] = None
    cfg = config_from_args(a, backbone_pretrained=False)
    d = RelSGGConfig()
    assert (cfg.dropout, cfg.proj_layers, cfg.deformable_heads, cfg.pe_max_octave) == (
        d.dropout, d.proj_layers, d.deformable_heads, d.pe_max_octave)


def test_backbone_is_the_only_per_model_difference():
    towers = {config_from_args(_load(n), backbone_pretrained=False).backbone_model
              for n in RELEASED}
    assert towers == {"facebook/dinov3-vits16-pretrain-lvd1689m",
                      "facebook/dinov3-vits16plus-pretrain-lvd1689m",
                      "facebook/dinov3-vitb16-pretrain-lvd1689m"}
