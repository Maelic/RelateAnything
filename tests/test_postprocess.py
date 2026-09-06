"""deploy/postprocess.decode — the numpy-only edge decode.

This is the code that turns raw ONNX outputs into triplets on devices with no
torch. Tests pin the threshold semantics (global floor, per-predicate
override, pair_weight) and the padding guard: the ONNX graph is built for a
FIXED box count and zero-padded, so pair slots can reference boxes that do
not exist — decode must drop them, not gather out of range.

decode now takes LOGITS (v2 export), and fuses them ADDITIVELY under
relsgg.scoring.ScoreContract rather than multiplying two sigmoids. The tests
are still written in probability terms and converted with `_lg`, so the intent
stays readable and the expected values stay comparable to the old ones.
"""
import numpy as np

from postprocess import ThresholdConfig, decode

PREDS = ["on", "holding", "behind"]


def _lg(p):
    """probability -> logit, so tests can keep stating what they mean."""
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def _inputs(scores, pair=None):
    K = len(scores)
    return dict(
        pred_logits=_lg(scores),
        # 0.5 is the additive identity in probability terms: logit(0.5) = 0.
        pair_logits=_lg(pair if pair is not None else [0.5] * K),
        sub_idx=np.array([0, 1, 0][:K]),
        obj_idx=np.array([1, 0, 2][:K]),
        valid_mask=np.ones(K, bool),
        predicates=PREDS,
)


def test_global_threshold_floors_scores():
    kw = _inputs([[0.9, 0.1, 0.1], [0.3, 0.2, 0.1]])
    out = decode(**kw, cfg=ThresholdConfig(threshold=0.5, pair_weight=0.0))
    assert len(out) == 1 and out[0].predicate == "on" and out[0].score > 0.5


def test_per_predicate_override_beats_global():
    kw = _inputs([[0.9, 0.1, 0.1], [0.1, 0.6, 0.1]])
    cfg = ThresholdConfig(threshold=0.5, pair_weight=0.0,
                          per_predicate={"holding": 0.7})
    out = decode(**kw, cfg=cfg)
    # 'holding' at 0.6 passes the 0.5 global but fails its own 0.7 override.
    assert [t.predicate for t in out] == ["on"]


def test_thresholds_vector_layout():
    cfg = ThresholdConfig(threshold=0.4, per_predicate={"behind": 0.9})
    v = cfg.thresholds_vector(PREDS)
    assert v.tolist() == [np.float32(0.4), np.float32(0.4), np.float32(0.9)]


def test_pair_weight_zero_recovers_the_predicate_probability():
    """With w=0 and no calibration the score IS sigmoid(pred_logit)."""
    kw = _inputs([[0.9, 0.1, 0.1]], pair=[0.01])
    out = decode(**kw, cfg=ThresholdConfig(threshold=0.0, pair_weight=0.0))
    assert abs(out[0].score - 0.9) < 1e-5


def test_pair_weight_gates_scores():
    # Same predicate scores, strong vs weak pair evidence: with pair_weight=1
    # the weak-pair slot must fall below the floor. Additively,
    # sigmoid(logit(.8) + logit(.1)) = 0.31 while
    # sigmoid(logit(.8) + logit(.99)) = 0.997.
    kw = _inputs([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]], pair=[0.99, 0.1])
    out = decode(**kw, cfg=ThresholdConfig(threshold=0.5, pair_weight=1.0))
    assert len(out) == 1 and out[0].subject_idx == 0


def test_calibration_shifts_the_threshold_not_the_order():
    """The deployment calibration is the reason a threshold means anything."""
    kw = _inputs([[0.99, 0.97, 0.1], [0.98, 0.2, 0.1]])
    raw = decode(**kw, cfg=ThresholdConfig(threshold=0.5, pair_weight=0.0,
                                           max_per_pair=0))
    cal = decode(**kw, cfg=ThresholdConfig(threshold=0.5, pair_weight=0.0,
                                           max_per_pair=0,
                                           calib_a=0.4344, calib_b=-2.4435))
    # uncalibrated, everything near 1.0 survives a 0.5 floor; calibrated, none
    # of it does — same ranking, different meaning.
    assert len(raw) == 3 and len(cal) == 0
    order = decode(**kw, cfg=ThresholdConfig(threshold=0.0, pair_weight=0.0,
                                             max_per_pair=0,
                                             calib_a=0.4344, calib_b=-2.4435))
    assert [(t.subject_idx, t.predicate) for t in order][:3] == \
           [(t.subject_idx, t.predicate) for t in raw][:3]


def test_padding_indices_are_dropped():
    # Slot references box 7 of a 2-box frame — the zero-padding case.
    kw = _inputs([[0.9, 0.1, 0.1], [0.9, 0.1, 0.1]])
    kw["sub_idx"] = np.array([0, 7])
    kw["obj_idx"] = np.array([1, 1])
    out = decode(**kw, cfg=ThresholdConfig(threshold=0.5, pair_weight=0.0),
                 box_scores=np.array([0.9, 0.9]))
    assert len(out) == 1 and out[0].subject_idx == 0
