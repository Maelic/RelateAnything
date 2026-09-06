"""THE relation score contract. One definition, imported by eval AND deploy.

WHY THIS FILE EXISTS. The score was independently re-implemented in five
places — relsgg/evaluator.py, relsgg/model.py:predict, relsgg/api.py,
deploy/pipeline.py and deploy/postprocess.py — and they did not agree. Eval
scored `sigmoid(pred + rel)`; the product scored `sigmoid(pred) * sigmoid(rel)`.
Those rank pairs differently, so no number we ever reported came from the
formula a user actually ran. Worse, the benchmarks PREFERRED the wrong one:
on VG150 the multiplicative form reads AUC 0.832 against the additive 0.763,
because multiplying up-weights the relatedness term and relatedness is an
annotation-propensity prior ([[relsgg-relatedness-contact-prior]]) — it
predicts which pairs a human bothered to annotate. On Haystack's explicitly
adjudicated negatives the ordering INVERTS (additive 0.9108, multiplicative
0.9038, relatedness-alone worst at 0.7475). See [[relsgg-confidence-knob]].

So the contract is fixed here, once:

    score = sigmoid(a * (pred_logit + w * pair_logit) + b)

  w = pair_weight  1.0 is the trained fusion; 0.0 drops relatedness entirely.
  (a, b)           the deployment calibration. The head's own logit_scale /
                   logit_bias are trained by a mass-balanced BCE, i.e. against
                   a 50/50 prior, while a real frame is 0.2-4% positive — so
                   raw scores pile into [0.9, 1.0) and a threshold is a knob
                   connected to nothing. (a, b) is that missing fit. Monotone
                   for a > 0, so every ranking metric is bit-identical and only
                   THRESHOLDS change meaning.

Works on torch tensors and numpy arrays alike, so the GPU evaluator and the
ONNX/numpy laptop path run the same arithmetic rather than two transcriptions
of the same sentence. tests/test_score_parity.py proves it rather than
asserting it, and includes a test that the two formulas genuinely differ so
the parity test cannot pass vacuously.

WHAT STILL DIFFERS BETWEEN EVAL AND DEPLOYMENT. The score function is now one
function, but a reported number is not yet a deployment number. These are the
remaining knobs, all of them deliberate, none of them silent:

  axis            eval default            deploy default        effect
  --------------- ----------------------- --------------------- --------------
  score fn        this contract           this contract         UNIFIED
  calibration     identity                fitted (a, b)         thresholds only
                                                                (ranking is
                                                                invariant)
  graph constraint opt-in                 always on             +12-19 pts R@K
                  (--graph_constraint)    (max_per_pair=1)      when left off
  selection       global top-K            threshold, then top-K different
                                                                operating point
  boxes           GT (oracle)             YOLOE detections      the big one
  max_objects     100                     16
  geo_budget      400                     160                   fewer candidate
  final_budget    128 (eval_budget 500)   64                    pairs survive
  rank weight     predicate score only    x conf(sub) conf(obj) SGDet
                  unless box_scores given                       convention

The oracle-box row dominates the rest: see [[relsgg-deployment-probe-e1e3]]
for the measured GT->detector drop. Anything quoting a deployment precision
from an oracle-box eval is quoting an upper bound.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:                       # torch is only used for isinstance dispatch; the
    import torch           # laptop deploy path (onnx/openvino backends) must
except ImportError:        # work without it installed at all.
    torch = None

CONTRACT = "sigmoid(a * (pred_logit + w * pair_logit) + b)"


def _is_tensor(x) -> bool:
    return torch is not None and isinstance(x, torch.Tensor)


def _sigmoid(z):
    """Numerically stable, and EXACT at the infinities.

    Not `1/(1+exp(-clip(z, -60, 60)))`: clipping maps -inf to 8.8e-27 instead
    of 0, and decode_decomposed masks whole predicate columns with -inf and
    expects them gone. The branch form overflows nowhere (each exp argument is
    <= 0) and returns exactly 0.0 / 1.0 at -inf / +inf, matching torch.
    """
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
    """How a predicate logit and a relatedness logit become one number."""

    calib_a: float = 1.0
    calib_b: float = 0.0
    pair_weight: float = 1.0

    def __post_init__(self):
        if self.calib_a <= 0:
            raise ValueError(
                f"calib_a must be > 0 or the contract stops being monotone and "
                f"ranking metrics change under calibration; got {self.calib_a}")

    @property
    def is_calibrated(self) -> bool:
        return (self.calib_a, self.calib_b) != (1.0, 0.0)

    # -- the contract ------------------------------------------------------
    def fuse(self, pred_logit, pair_logit=None):
        """[..., K, V] predicate logits + [..., K] pair logits -> [..., K, V]."""
        z = pred_logit
        if pair_logit is not None and self.pair_weight:
            if _is_tensor(pair_logit):
                z = z + self.pair_weight * pair_logit.unsqueeze(-1)
            else:
                z = z + self.pair_weight * np.asarray(pair_logit)[..., None]
        return self.calib_a * z + self.calib_b

    def scores(self, pred_logit, pair_logit=None):
        return _sigmoid(self.fuse(pred_logit, pair_logit))

    # -- persistence -------------------------------------------------------
    @classmethod
    def from_json(cls, path: str, pair_weight: float = 1.0) -> "ScoreContract":
        with open(path) as fh:
            d = json.load(fh)
        return cls(calib_a=float(d["a"]), calib_b=float(d["b"]),
                   pair_weight=pair_weight)

    @classmethod
    def for_checkpoint(cls, ckpt_path: str, pair_weight: float = 1.0,
                       required: bool = False) -> "ScoreContract":
        """Load `calibration.json` sitting next to a checkpoint, if present.

        Uncalibrated is the default rather than an error: it is the right
        behaviour for every ranking metric (R@K, mR@K, AP, AUC are invariant)
        and only misleads when someone applies a THRESHOLD. Callers that
        threshold should pass required=True.
        """
        p = os.path.join(os.path.dirname(ckpt_path), "calibration.json")
        if os.path.exists(p):
            return cls.from_json(p, pair_weight=pair_weight)
        if required:
            raise FileNotFoundError(
                f"no calibration.json next to {ckpt_path}: a threshold against "
                "the raw head is meaningless (~97% of scores sit in [0.9, 1.0)). "
                "Fit one with benchmark/eval_deploy_metrics.py --fit_platt.")
        return cls(pair_weight=pair_weight)

    def describe(self) -> str:
        return (f"{CONTRACT}  [a={self.calib_a:.4f} b={self.calib_b:.4f} "
                f"w={self.pair_weight:.2f}"
                + ("" if self.is_calibrated else ", UNCALIBRATED") + "]")


# The contract as it was before this file existed, kept ONLY so the deploy
# regression test can assert the two differ and quantify by how much. Do not
# use it for anything else.
def legacy_multiplicative(pred_logit, pair_logit=None):
    s = _sigmoid(pred_logit)
    if pair_logit is None:
        return s
    p = _sigmoid(pair_logit)
    p = p.unsqueeze(-1) if _is_tensor(p) else p[..., None]
    return s * p


def graph_constrained(scores, valid_mask=None):
    """One predicate per pair — each pair keeps only its argmax column.

    Deploy has always done this (`ThresholdConfig.max_per_pair == 1`) while
    eval defaulted to unconstrained, which inflated R@K by 12-19 points
    ([[relsgg-eval-protocol-graph-constraint]]). Same helper both sides now.
    Returns a boolean mask of kept cells, NOT a modified score array.
    """
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
