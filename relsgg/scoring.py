"""The relation score contract, shared by evaluation and deployment.

    score = sigmoid(a * (pred_logit + w * pair_logit) + b)

``pred_logit`` is the predicate head's scaled cosine, ``pair_logit`` the
sampler's pair-existence logit. ``w`` (``pair_weight``) is 1 as trained;
``(a, b)`` is a calibration fitted after training on a validation split
(``calibration.json`` next to a checkpoint). The head's affine is trained
against balanced positives and negatives, while a real frame has few true
pairs, so raw scores crowd into [0.9, 1.0); the calibration is monotone, so
ranking metrics are unchanged and only thresholds gain a meaning.

Works on torch tensors and numpy arrays, so the GPU evaluator and the
torch-free ONNX runtime run the same arithmetic. ``tests/test_score_parity.py``
checks that.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

try:
    import torch
except ImportError:            # the ONNX runtime path runs without torch
    torch = None

CONTRACT = "sigmoid(a * (pred_logit + w * pair_logit) + b)"


def _is_tensor(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _sigmoid(z):
    """Stable sigmoid that is exact at the infinities (masked columns are -inf)."""
    if _is_tensor(z):
        return torch.sigmoid(z)
    z = np.asarray(z)
    out = np.empty(z.shape, dtype=np.result_type(z.dtype, np.float32))
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass(frozen=True)
class ScoreContract:
    calib_a: float = 1.0
    calib_b: float = 0.0
    pair_weight: float = 1.0

    def __post_init__(self):
        if self.calib_a <= 0:
            raise ValueError(f"calib_a must be > 0 to keep the contract monotone; got {self.calib_a}")

    @property
    def is_calibrated(self) -> bool:
        return (self.calib_a, self.calib_b) != (1.0, 0.0)

    def fuse(self, pred_logit, pair_logit=None):
        """``[..., K, V]`` predicate logits and ``[..., K]`` pair logits -> ``[..., K, V]``."""
        z = pred_logit
        if pair_logit is not None and self.pair_weight:
            if _is_tensor(pair_logit):
                z = z + self.pair_weight * pair_logit.unsqueeze(-1)
            else:
                z = z + self.pair_weight * np.asarray(pair_logit)[..., None]
        return self.calib_a * z + self.calib_b

    def scores(self, pred_logit, pair_logit=None):
        return _sigmoid(self.fuse(pred_logit, pair_logit))

    @classmethod
    def from_json(cls, path: str, pair_weight: float = 1.0) -> "ScoreContract":
        with open(path) as fh:
            d = json.load(fh)
        return cls(calib_a=float(d["a"]), calib_b=float(d["b"]), pair_weight=pair_weight)

    @classmethod
    def for_checkpoint(cls, ckpt_path: str, pair_weight: float = 1.0,
                       required: bool = False) -> "ScoreContract":
        """The ``calibration.json`` next to a checkpoint, or the identity.
        Ranking metrics do not need one; thresholds do (``required=True``)."""
        p = os.path.join(os.path.dirname(ckpt_path), "calibration.json")
        if os.path.exists(p):
            return cls.from_json(p, pair_weight=pair_weight)
        if required:
            raise FileNotFoundError(
                f"no calibration.json next to {ckpt_path}; a threshold on raw scores is "
                "not meaningful. Fit one with benchmark/eval_deploy_metrics.py --fit_platt.")
        return cls(pair_weight=pair_weight)

    def describe(self) -> str:
        return (f"{CONTRACT}  [a={self.calib_a:.4f} b={self.calib_b:.4f} w={self.pair_weight:.2f}"
                + ("" if self.is_calibrated else ", uncalibrated") + "]")


def graph_constrained(scores, valid_mask=None):
    """Boolean mask keeping one predicate per pair (its argmax column)."""
    if _is_tensor(scores):
        best = scores.argmax(dim=-1, keepdim=True)
        keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, best, True)
        if valid_mask is not None:
            keep &= valid_mask.unsqueeze(-1)
        return keep
    best = np.asarray(scores).argmax(axis=-1)
    keep = np.zeros(np.shape(scores), dtype=bool)
    np.put_along_axis(keep, best[..., None], True, axis=-1)
    if valid_mask is not None:
        keep &= np.asarray(valid_mask)[..., None]
    return keep
