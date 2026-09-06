"""Eval and deployment must compute the SAME number. Proven, not asserted.

This is the regression test for the class of bug that made every reported
number come from a formula the product did not run: relsgg/evaluator.py scored
sigmoid(pred + rel) while deploy/pipeline.py and deploy/postprocess.py scored
sigmoid(pred) * sigmoid(rel). Both are defensible; having both is not. And the
benchmarks could not catch it — on VG150 the wrong one reads BETTER (AUC 0.832
vs 0.763), because multiplying up-weights the relatedness term and relatedness
predicts annotation propensity rather than truth. Only Haystack's adjudicated
negatives invert that (0.9038 vs 0.9108).

So the guard cannot be "the numbers look right". It has to be that every path
is the same code. These tests pin that down:

  1. the torch and numpy implementations of the contract agree to fp32
  2. the evaluator's ranking == the deploy decoder's ranking on shared input
  3. calibration is MONOTONE: it may not reorder anything, ever
  4. the legacy multiplicative form actually differs — i.e. this test would
     have failed before the fix, rather than passing vacuously
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deploy.postprocess import ThresholdConfig, decode  # noqa: E402
from relsgg.scoring import (ScoreContract, graph_constrained,  # noqa: E402
                            legacy_multiplicative)


def _fake(K=24, V=7, seed=0):
    """Logits with the shape and SCALE the real head produces.

    logit_scale.exp() is ~21 and logit_bias ~-4.5, so predicate logits live
    around [-5, +9] and saturate the sigmoid — using N(0,1) here would make
    every test pass for the wrong reason.
    """
    g = np.random.default_rng(seed)
    pred = (g.normal(0.35, 0.12, (K, V)) * 21.26 - 4.505).astype(np.float32)
    pair = g.normal(0.0, 3.0, K).astype(np.float32)
    sub = g.integers(0, 6, K).astype(np.int64)
    obj = (sub + 1 + g.integers(0, 5, K)) % 6
    valid = np.ones(K, dtype=bool)
    valid[-3:] = False
    return pred, pair, sub, obj.astype(np.int64), valid


CONTRACTS = [
    ScoreContract(),                                        # uncalibrated
    ScoreContract(calib_a=0.5472, calib_b=-6.1098),         # PSG-val fit
    ScoreContract(calib_a=0.4344, calib_b=-2.4435),         # Haystack fit
    ScoreContract(calib_a=1.0, calib_b=0.0, pair_weight=0.0),   # no relatedness
    ScoreContract(calib_a=0.9, calib_b=-3.0, pair_weight=2.0),
]


@pytest.mark.parametrize("c", CONTRACTS)
def test_torch_and_numpy_agree(c):
    """The GPU evaluator and the torch-free laptop host run one contract."""
    pred, pair, *_ = _fake()
    n = c.scores(pred, pair)
    t = c.scores(torch.from_numpy(pred), torch.from_numpy(pair)).numpy()
    assert np.abs(n - t).max() < 1e-6, np.abs(n - t).max()


@pytest.mark.parametrize("c", CONTRACTS[1:3])
def test_calibration_preserves_the_true_ranking(c):
    """Calibration changes what a threshold MEANS, never the order.

    The reference order is the one in LOGIT space, which is exact. Comparing
    against the RAW PROBABILITY instead would be wrong, and instructively so —
    the identity contract is deliberately EXCLUDED here because its "score" is
    that raw probability, which cannot represent the order at all. See
    test_raw_probability_cannot_represent_the_ranking.
    """
    pred, pair, *_ = _fake(seed=3)
    truth = ScoreContract(pair_weight=c.pair_weight).fuse(pred, pair).ravel()
    cal = c.scores(pred, pair).ravel()
    assert np.array_equal(np.argsort(-truth, kind="mergesort"),
                          np.argsort(-cal, kind="mergesort"))


def test_raw_probability_cannot_represent_the_ranking():
    """The uncalibrated score is so saturated that fp32 loses the order.

    ~97% of real scores land in [0.9, 1.0), where float32 steps by ~6e-8, so
    genuinely different logits collapse onto the same float and their relative
    order becomes whatever the sort's tie-breaking says. Calibration is not
    only about interpretability: it moves the scores back into a range the
    number format can actually express. This asserts the failure exists (so
    nobody "fixes" it by reverting) and that calibration removes it.
    """
    pred, pair, *_ = _fake(seed=3)
    truth = np.argsort(-ScoreContract().fuse(pred, pair).ravel(), kind="mergesort")
    raw = np.argsort(-ScoreContract().scores(pred, pair).ravel(), kind="mergesort")
    cal = ScoreContract(calib_a=0.4344, calib_b=-2.4435)
    calr = np.argsort(-cal.scores(pred, pair).ravel(), kind="mergesort")

    p = np.asarray(ScoreContract().scores(pred, pair)).ravel()
    z = np.asarray(ScoreContract().fuse(pred, pair)).ravel()
    n_tied = len(p) - len(np.unique(p))
    assert not np.array_equal(truth, raw), (
        "raw fp32 probabilities preserved the order on this fixture — pick a "
        "harder one, or the saturation claim no longer holds")
    assert np.array_equal(truth, calr), "calibration must restore the order"
    # Concretely, on this fixture logits 11.3392 and 11.3412 both become the
    # bit-identical float 0x1.fffe700000000p-1.
    assert n_tied > 0 and p.max() > 0.9999, (n_tied, p.max())
    assert (p > 0.9).mean() > 0.5, "fixture no longer reproduces the saturation"
    # the tied pair really does come from DISTINCT logits
    _, first = np.unique(p, return_index=True)
    dup = np.setdiff1d(np.arange(len(p)), first)
    assert all(len(np.unique(z[p == p[i]])) > 1 for i in dup)


def test_evaluator_ranking_matches_deploy_decoder():
    """SGClsEvaluator's top-K and deploy's decode() pick the same triplets."""
    from relsgg.evaluator import SGClsEvaluator
    pred, pair, sub, obj, valid = _fake(seed=7)
    c = ScoreContract(calib_a=0.4344, calib_b=-2.4435)
    preds = [f"p{i}" for i in range(pred.shape[1])]

    # deploy path: numpy, graph-constrained, thresholded at 0
    got = decode(pred, pair, sub, obj, valid, preds,
                 ThresholdConfig(threshold=0.0, topk=10, max_per_pair=1,
                                 box_score_weight=False,
                                 calib_a=c.calib_a, calib_b=c.calib_b))
    deploy_rank = [(t.subject_idx, t.object_idx, t.predicate) for t in got]

    # eval path: torch, same contract, same graph constraint
    ev = SGClsEvaluator(topk=[10], num_predicates=len(preds),
                        score_mode="sigmoid", graph_constraint=True, contract=c)
    s = ev.contract.scores(torch.from_numpy(pred), torch.from_numpy(pair))
    keep = graph_constrained(s, torch.from_numpy(valid))
    flat = torch.where(keep, s, torch.full_like(s, -1.0)).ravel()
    top = flat.topk(10).indices
    eval_rank = [(int(sub[int(i) // s.shape[1]]), int(obj[int(i) // s.shape[1]]),
                  preds[int(i) % s.shape[1]]) for i in top]

    assert deploy_rank == eval_rank, f"\ndeploy {deploy_rank}\neval   {eval_rank}"


def test_the_old_bug_would_have_been_caught():
    """Guard against a vacuous test: the two formulas must really differ.

    If additive and multiplicative ever agreed on this input, the parity test
    above would pass no matter which one each side used.
    """
    pred, pair, *_ = _fake(seed=11)
    add = ScoreContract().scores(pred, pair).ravel()
    mul = legacy_multiplicative(pred, pair).ravel()
    ra = np.argsort(-add, kind="mergesort")
    rm = np.argsort(-mul, kind="mergesort")
    assert not np.array_equal(ra, rm), "formulas agree — test proves nothing"


def test_pair_weight_zero_drops_relatedness():
    pred, pair, *_ = _fake(seed=5)
    a = ScoreContract(pair_weight=0.0).scores(pred, pair)
    b = ScoreContract(pair_weight=0.0).scores(pred, None)
    assert np.allclose(a, b)


def test_calib_a_must_stay_positive():
    """a <= 0 would flip the ranking; the contract refuses to be constructed."""
    with pytest.raises(ValueError):
        ScoreContract(calib_a=0.0)
    with pytest.raises(ValueError):
        ScoreContract(calib_a=-1.0)


def test_decode_masks_with_neg_inf_in_logit_space():
    """decode_decomposed masks columns with -inf; it must survive the contract."""
    pred, pair, sub, obj, valid = _fake(seed=13)
    pred = pred.copy()
    pred[:, 2] = -np.inf
    c = ScoreContract(calib_a=0.4344, calib_b=-2.4435)
    s = c.scores(pred, pair)
    assert np.all(s[:, 2] == 0.0)
    assert np.isfinite(s).all()
