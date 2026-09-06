"""Evaluator arithmetic on hand-computable cases.

Every reported number flows through SGClsEvaluator, and two of its properties
are easy to lose silently:
  * the graph constraint (one predicate per ordered pair) — without it R@K
    reads 12 to 19 points high and spatial classes fall to zero;
  * F1@K, the harmonic mean of R@K and mR@K — it must equal the closed form
    and must be 0 when either side is 0, since a tail-blind model cannot score.
"""
import torch

from relsgg.eval.evaluator import SGClsEvaluator, _harmonic

NAMES = ["on", "holding", "behind", "wearing"]
V = len(NAMES)


def _out(logits):
    K = logits.shape[1]
    return {
        "logits": logits,
        "sub_idx": torch.tensor([[0, 1, 0, 2, 1, 2][:K]]),
        "obj_idx": torch.tensor([[1, 0, 2, 0, 2, 1][:K]]),
        "valid_mask": torch.ones(1, K, dtype=torch.bool),
    }


def test_harmonic_closed_form():
    assert _harmonic(0.0, 0.0) == 0.0
    assert _harmonic(1.0, 0.0) == 0.0          # tail-blind -> zero, not 0.5
    assert abs(_harmonic(0.5, 0.5) - 0.5) < 1e-12
    # a published pair, computed by hand: 2*.3597*.2656/(.3597+.2656)
    assert abs(_harmonic(0.3597, 0.2656) - 0.30558) < 1e-4


def test_f1_keys_consistent_with_r_and_mr():
    ev = SGClsEvaluator(topk=[20, 50, 100], num_predicates=V,
                        score_mode="sigmoid", graph_constraint=True)
    torch.manual_seed(0)
    ev.update(_out(torch.randn(1, 6, V)),
              [{"relations": torch.tensor([[0, 1, 0], [0, 2, 1], [1, 2, 2]])}])
    m = ev.compute()
    for k in (20, 50, 100):
        assert abs(m[f"F1@{k}"] - _harmonic(m[f"R@{k}"], m[f"mR@{k}"])) < 1e-12


def test_perfect_prediction_scores_one():
    # Logits that rank the GT predicate first for every GT pair.
    logits = torch.full((1, 3, V), -10.0)
    gt = [(0, 1, 0), (0, 2, 1), (1, 2, 2)]
    for slot, (_, _, p) in enumerate(gt):
        logits[0, slot, p] = 10.0
    out = {
        "logits": logits,
        "sub_idx": torch.tensor([[s for s, _, _ in gt]]),
        "obj_idx": torch.tensor([[o for _, o, _ in gt]]),
        "valid_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    ev = SGClsEvaluator(topk=[20], num_predicates=V,
                        score_mode="sigmoid", graph_constraint=True)
    ev.update(out, [{"relations": torch.tensor(gt)}])
    m = ev.compute()
    assert m["R@20"] == 1.0 and m["mR@20"] == 1.0 and m["F1@20"] == 1.0


def test_graph_constraint_one_predicate_per_pair():
    """One pair whose top-2 predicates BOTH match distinct GT relations on the
    same ordered pair: unconstrained top-K credits both; constrained keeps only
    the argmax, so exactly one of the two GT rows can ever be hit."""
    logits = torch.full((1, 1, V), -10.0)
    logits[0, 0, 0] = 10.0     # 'on'      — argmax
    logits[0, 0, 1] = 9.0      # 'holding' — runner-up on the SAME pair
    out = {
        "logits": logits,
        "sub_idx": torch.tensor([[0]]),
        "obj_idx": torch.tensor([[1]]),
        "valid_mask": torch.ones(1, 1, dtype=torch.bool),
    }
    gt = [{"relations": torch.tensor([[0, 1, 0], [0, 1, 1]])}]

    unc = SGClsEvaluator(topk=[20], num_predicates=V,
                         score_mode="sigmoid", graph_constraint=False)
    unc.update(out, gt)
    con = SGClsEvaluator(topk=[20], num_predicates=V,
                         score_mode="sigmoid", graph_constraint=True)
    con.update(out, gt)

    assert unc.compute()["R@20"] == 1.0      # both GT rows credited
    assert con.compute()["R@20"] == 0.5      # only the argmax predicate counts
