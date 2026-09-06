"""relsgg.spatialsense_metrics: one implementation behind both A6 scorers.

benchmark/eval_spatialsense.py (ours) and benchmark/ovsgtr/eval_spatialsense_interchange.py
(the baseline) both call summarise(); if the numbers ever diverge it is the inputs,
not the metric. These pin the threshold search and the record layout.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
from relsgg.eval.spatialsense import best_threshold, summarise  # noqa: E402


def test_best_threshold_separable():
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    l = np.array([0, 0, 0, 1, 1, 1])
    t, acc = best_threshold(s, l)
    assert acc == 1.0 and 0.3 < t < 0.7


def test_best_threshold_is_accuracy_maximising():
    rng = np.random.default_rng(0)
    s = rng.uniform(size=200); l = rng.integers(0, 2, 200)
    t, acc = best_threshold(s, l)
    grid = np.linspace(0, 1, 1001)
    brute = max(((s >= g) == l.astype(bool)).mean() for g in grid)
    assert acc >= brute - 1e-9


def test_summarise_record_layout_and_chance():
    rng = np.random.default_rng(1)
    names = ["behind", "on", "under"]
    n = 600
    l = rng.integers(0, 2, n); pr = rng.integers(0, 3, n)
    s = np.clip(0.5 + 0.3 * (2 * l - 1) + rng.normal(0, 0.1, n), 0, 1)   # planted signal
    res = summarise(s, l, pr, names, tau=0.5, coverage=0.9)
    for k in ("AUC", "AP", "acc@valid_tau", "acc@0.5", "acc@oracle", "valid_tau",
              "oracle_tau", "coverage", "n", "per_predicate", "use_pair_logits"):
        assert k in res
    assert res["n"] == n and res["coverage"] == 0.9
    assert res["AUC"] > 0.95 and set(res["per_predicate"]) == set(names)
    # a label-independent score sits at chance
    flat = summarise(rng.uniform(size=n), l, pr, names, tau=0.5, coverage=1.0)
    assert abs(flat["AUC"] - 0.5) < 0.08


def test_per_predicate_row_without_both_classes_has_no_auc():
    names = ["a", "b"]
    s = np.array([0.9, 0.8, 0.2, 0.1]); l = np.array([1, 1, 0, 0]); pr = np.array([0, 0, 1, 1])
    res = summarise(s, l, pr, names, tau=0.5, coverage=1.0)
    assert "AUC" not in res["per_predicate"]["a"] and "AUC" not in res["per_predicate"]["b"]
    assert res["per_predicate"]["a"]["n"] == 2
