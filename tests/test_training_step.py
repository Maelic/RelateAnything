"""One training step of the real objective, on a tiny model.

Training a released model needs GPUs, so what CI can check is that the
objective runs end to end and that every trainable parameter receives a
gradient from it. The second half is the point: a module the loss never
reaches trains as dead weight and nothing else notices, because rank-based
metrics are invariant to whatever it does.
"""
import json
import os
import tempfile

import pytest

torch = pytest.importorskip("torch")

from relsgg.config import RelSGGConfig          # noqa: E402
from relsgg.model import RelSGG                 # noqa: E402
from relsgg.training.losses import PredicateOntology  # noqa: E402

V = 12          # predicates
N = 5           # boxes per image
B = 2           # images


def _tiny_backbone_dir(tmp):
    """A DINOv3 config small enough to run on a CPU, built from config only."""
    cfg = {"model_type": "dinov3_vit", "hidden_size": 32, "intermediate_size": 64,
           "num_hidden_layers": 2, "num_attention_heads": 2, "patch_size": 16,
           "image_size": 64, "num_channels": 3, "hidden_act": "silu",
           "use_gated_mlp": True, "num_register_tokens": 1, "layer_norm_eps": 1e-5,
           "rope_theta": 100.0, "layerscale_value": 1.0, "query_bias": True,
           "key_bias": False, "value_bias": True, "proj_bias": True, "mlp_bias": True,
           "pos_embed_rescale": 2.0, "drop_path_rate": 0.0}
    with open(os.path.join(tmp, "config.json"), "w") as fh:
        json.dump(cfg, fh)
    return tmp


def _ontology():
    """Identity positives, no down-weighting, no inverses — the shape the loss
    reads, with every estimated table at its neutral value."""
    pos_w = torch.eye(V, dtype=torch.float16)
    neg_lw = torch.zeros(V, V, dtype=torch.float16)
    sym = torch.zeros(V)
    inverse = torch.zeros(V, V, dtype=torch.bool)
    return PredicateOntology([f"p{i}" for i in range(V)], pos_w, neg_lw, sym, inverse)


def _batch(device="cpu"):
    torch.manual_seed(0)
    images = torch.rand(B, 3, 64, 64)
    cx, cy = torch.rand(B, N) * 0.6 + 0.2, torch.rand(B, N) * 0.6 + 0.2
    boxes = torch.stack([cx, cy, torch.full((B, N), 0.2), torch.full((B, N), 0.2)], -1)
    box_counts = torch.full((B,), N, dtype=torch.long)
    targets = [{"relations": torch.tensor([[0, 1, 3], [1, 2, 7], [3, 4, 3]]),
                "rel_weights": torch.ones(3),
                "entity_labels": torch.arange(N),
                "src": 0} for _ in range(B)]
    return images, boxes, box_counts, targets


@pytest.fixture(scope="module")
def model():
    tmp = tempfile.mkdtemp(prefix="ra_test_backbone_")
    cfg = RelSGGConfig(backbone_model=_tiny_backbone_dir(tmp), backbone_pretrained=False,
                       d_model=32, text_dim=16, geo_budget=12, final_budget=6,
                       n_self_layers=1, n_cross_layers=1, n_dep_layers=1, n_gnd_layers=1,
                       n_heads=2, deformable_points=2, deformable_heads=2,
                       deformable_nulls=1, proj_layers=2, dropout=0.0)
    m = RelSGG(cfg)
    m.vocab_head.set_vocabulary_matrix([f"p{i}" for i in range(V)], torch.randn(V, 16))
    m.install_losses(_ontology(), n_neg=8)
    m.set_object_vocabulary([f"c{i}" for i in range(N)], torch.randn(N, 16))
    return m


def test_one_step_produces_a_finite_loss(model):
    model.train()
    out = model(*_batch())
    assert torch.isfinite(out["loss"]), out["loss_dict"]
    for k, v in out["loss_dict"].items():
        assert torch.isfinite(v), f"{k} is not finite"
    assert out["logits"].shape == (B, model.config.final_budget, V)


def test_every_trainable_parameter_receives_gradient(model):
    """The default-on audit in the training loop, as a test."""
    model.train()
    model.zero_grad(set_to_none=True)
    model(*_batch())["loss"].backward()
    dead = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    # The pooling shape gate is inactive by design when no masks are given.
    dead = [n for n in dead if n != "spatial_pool.cov_lambda"]
    assert not dead, f"no gradient reached: {dead}"


def test_inference_needs_no_targets_and_no_labels(model):
    model.eval()
    images, boxes, box_counts, _ = _batch()
    with torch.no_grad():
        out = model(images, boxes, box_counts=box_counts, targets=None)
    assert "loss" not in out
    assert out["pair_logits"].shape == (B, model.config.final_budget)


def test_masks_change_the_geometry_features_only_where_they_differ(model):
    """A region raster of the box itself has to reproduce the box result: the
    geometry features take their box values when a region fills its box."""
    model.eval()
    images, boxes, box_counts, _ = _batch()
    g = 32
    cx, cy, w, h = boxes.unbind(-1)
    e = torch.arange(g + 1, dtype=torch.float32) / g
    ix = (torch.minimum((cx + w / 2).unsqueeze(-1), e[1:])
          - torch.maximum((cx - w / 2).unsqueeze(-1), e[:-1])).clamp(min=0)
    iy = (torch.minimum((cy + h / 2).unsqueeze(-1), e[1:])
          - torch.maximum((cy - h / 2).unsqueeze(-1), e[:-1])).clamp(min=0)
    cov = (iy.unsqueeze(-1) * ix.unsqueeze(-2)) * (g * g)
    with torch.no_grad():
        plain = model(images, boxes, box_counts=box_counts, targets=None)
        boxed = model(images, boxes, box_counts=box_counts, targets=None,
                      cov=cov, fill=torch.ones(B, N))
    assert torch.allclose(plain["logits"], boxed["logits"], atol=2e-4)
