"""Two-graph split semantics (pure numpy — CI-safe).

Pins the properties the measured protocol depends on: one argmax edge per
pair per stream, independent ranking, a pair may appear in BOTH streams, and
the hybrid type rule (corpus flag first, gate alpha only for novel strings).
The checkpoint-level parity check (api vs benchmark/eval_decomposed.py) needs
weights and runs locally, not in CI.
"""
import numpy as np

from relsgg.decompose import split_ranked, type_vector
from postprocess import ThresholdConfig, decode_decomposed

PREDS = ["on", "behind", "holding", "looking at"]     # spatial: on, behind
IS_SP = np.array([True, True, False, False])


def test_pair_can_appear_in_both_streams():
    # One pair, strong scores in both types.
    scores = np.array([[5.0, 1.0, 4.0, 0.0]], np.float32)
    out = split_ranked(scores, np.array([0]), np.array([1]),
                       np.array([True]), IS_SP, topk=5)
    assert out["spatial"][0][:3] == (0, 1, 0)      # argmax within spatial: on
    assert out["semantic"][0][:3] == (0, 1, 2)     # argmax within semantic: holding


def test_one_edge_per_pair_per_stream():
    # Two spatial predicates both high for the same pair -> ONE spatial edge.
    scores = np.array([[5.0, 4.9, 0.0, 0.0]], np.float32)
    out = split_ranked(scores, np.array([0]), np.array([1]),
                       np.array([True]), IS_SP, topk=5)
    assert len(out["spatial"]) == 1 and out["spatial"][0][2] == 0


def test_streams_ranked_independently():
    scores = np.array([[3.0, 0.0, 1.0, 0.0],       # pair A: spatial 3.0, sem 1.0
                       [2.0, 0.0, 4.0, 0.0]],      # pair B: spatial 2.0, sem 4.0
                      np.float32)
    out = split_ranked(scores, np.array([0, 1]), np.array([1, 2]),
                       np.array([True, True]), IS_SP, topk=1)
    assert out["spatial"][0][:2] == (0, 1)          # A tops spatial
    assert out["semantic"][0][:2] == (1, 2)         # B tops semantic


def test_invalid_pairs_dropped():
    scores = np.array([[9.0, 0.0, 9.0, 0.0]], np.float32)
    out = split_ranked(scores, np.array([0]), np.array([1]),
                       np.array([False]), IS_SP, topk=5)
    assert out["spatial"] == [] and out["semantic"] == []


def test_type_vector_hybrid_rule():
    names = ["on", "hovering over", "holding"]
    cmap = {"on": True, "holding": False}           # corpus knows 2 of 3
    alpha = np.array([0.0, 0.9, 0.9])               # gate would say spatial
    is_sp, src = type_vector(names, corpus_map=cmap, alpha=alpha)
    assert list(is_sp) == [True, True, False]
    # corpus wins where known (even against alpha), gate only for the novel one
    assert list(src) == ["corpus", "gate", "corpus"]


def test_decode_decomposed_thresholds_per_stream():
    # v2 contract: LOGITS in. logit(0.9)=2.197, logit(0.6)=0.405,
    # logit(1e-6)~-13.8 for the two columns meant to be far below the floor.
    kw = dict(
        pred_logits=np.array([[2.197, -13.8, 0.405, -13.8]], np.float32),
        pair_logits=np.array([0.0], np.float32),   # logit(0.5): additive identity
        sub_idx=np.array([0]), obj_idx=np.array([1]),
        valid_mask=np.array([True]), predicates=PREDS, is_spatial=IS_SP,
        cfg=ThresholdConfig(threshold=0.5, pair_weight=0.0),
)
    out = decode_decomposed(**kw)
    assert [t.predicate for t in out["spatial"]] == ["on"]
    assert [t.predicate for t in out["semantic"]] == ["holding"]


def test_parity_with_eval_decomposed_ranking():
    """split_ranked must reproduce benchmark/eval_decomposed.py's ranking math
    (masked_fill -> max per pair -> argsort desc -> cut) bit-for-bit. Random
    distinct floats, 50 trials, both streams, full ordered edge lists."""
    import torch

    rng = np.random.default_rng(0)
    for _ in range(50):
        K, V = int(rng.integers(1, 12)), int(rng.integers(2, 9))
        is_sp = rng.random(V) < 0.5
        if not is_sp.any() or is_sp.all():
            is_sp[0], is_sp[-1] = True, False
        sc = rng.standard_normal((K, V)).astype(np.float32)
        sub = rng.integers(0, 6, K)
        obj = rng.integers(0, 6, K)
        valid = rng.random(K) < 0.9

        got = split_ranked(sc, sub, obj, valid, is_sp, topk=K)

        # the eval's construction, verbatim semantics
        sc_t = torch.from_numpy(sc)[torch.from_numpy(valid)]
        s_l = sub[valid].tolist()
        o_l = obj[valid].tolist()
        for tag, sel in (("spatial", torch.from_numpy(is_sp)),
                         ("semantic", ~torch.from_numpy(is_sp))):
            if not bool(sel.any()) or sc_t.numel() == 0:
                assert got[tag] == []
                continue
            masked = sc_t.masked_fill(~sel.unsqueeze(0), float("-inf"))
            best, arg = masked.max(dim=-1)
            order = torch.argsort(best, descending=True).tolist()
            want = [(s_l[i], o_l[i], int(arg[i])) for i in order
                    if torch.isfinite(best[i])]
            assert [e[:3] for e in got[tag]] == want, (tag, got[tag], want)
