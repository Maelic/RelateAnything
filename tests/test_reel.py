"""The reel's model-free parts: region dedup, edge selection, mask geometry.

None of this needs a checkpoint or a detector — which is the point, because the
detectors it normally runs against are AGPL and are not in this repository, so
CI would otherwise cover none of it.
"""
from __future__ import annotations

import numpy as np
import pytest

from deploy.make_reel import bezier, dedupe_regions, select_edges
from deploy.postprocess import Triplet
from deploy.seg_detector import COLOUR_WORDS, colour_bins, decode_masks


def _t(s: int, p: str, o: int, score: float = 0.9) -> Triplet:
    return Triplet(subject_idx=s, predicate=p, score=score, object_idx=o)


# --------------------------------------------------------------------------- #
# dedupe_regions
# --------------------------------------------------------------------------- #

def test_dedupe_collapses_the_same_object_under_two_names():
    """`persian cat` and `feline` survive class-aware NMS; only one is a region."""
    boxes = np.array([[10, 10, 100, 100],       # kept: first, so highest conf
                      [12, 11, 99, 102],        # the same cat, another name
                      [200, 200, 260, 260]], np.float32)
    out, labels, masks = dedupe_regions(boxes, ["persian cat", "feline", "lamp"], None)
    assert labels == ["persian cat", "lamp"]
    assert len(out) == 2 and masks is None


def test_dedupe_keeps_nested_regions():
    """A person and their shirt overlap far less than the 0.7 threshold."""
    boxes = np.array([[0, 0, 100, 300],         # person
                      [20, 40, 80, 150]], np.float32)   # shirt
    out, labels, _ = dedupe_regions(boxes, ["person", "shirt"], None)
    assert labels == ["person", "shirt"] and len(out) == 2


def test_dedupe_reindexes_masks_with_boxes():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 70, 70]], np.float32)
    masks = np.zeros((3, 80, 80), bool)
    masks[2, 55:65, 55:65] = True
    _, labels, out = dedupe_regions(boxes, ["a", "b", "c"], masks)
    assert labels == ["a", "c"]
    assert out.shape[0] == 2 and out[1].sum() == 100      # the third mask moved to 1


# --------------------------------------------------------------------------- #
# select_edges
# --------------------------------------------------------------------------- #

BOXES = np.array([[0, 0, 200, 400],            # 0 person
                  [40, 60, 160, 200],          # 1 shirt
                  [30, 250, 170, 380],         # 2 shorts
                  [0, 300, 300, 500],          # 3 bicycle
                  [700, 20, 712, 50]], np.float32)   # 4 far too small to see
SHAPE = (600, 900)


def test_one_edge_per_pair():
    trips = [_t(0, "riding", 3), _t(3, "carrying", 0), _t(0, "wearing", 1)]
    out = select_edges(trips, BOXES, SHAPE, max_edges=4)
    assert [(e.subject_idx, e.predicate, e.object_idx) for e in out] == \
        [(0, "riding", 3), (0, "wearing", 1)]


def test_one_edge_per_predicate():
    """Otherwise a six-shot reel shows four predicates: clothing dominates."""
    trips = [_t(0, "wearing", 1), _t(0, "wearing", 2), _t(0, "riding", 3)]
    out = select_edges(trips, BOXES, SHAPE, max_edges=4)
    assert [e.predicate for e in out] == ["wearing", "riding"]


def test_tiny_endpoints_are_dropped():
    small = _t(0, "looking at", 4)
    assert select_edges([small], BOXES, SHAPE, max_edges=4) == []


def test_short_edges_are_kept():
    """`person riding skateboard` has nearly concentric endpoints; an earlier
    minimum-length rule dropped exactly the interaction relations."""
    concentric = np.array([[0, 0, 400, 400], [150, 150, 250, 250]], np.float32)
    out = select_edges([_t(0, "riding", 1)], concentric, SHAPE, max_edges=4)
    assert len(out) == 1


def test_ranking_order_is_preserved():
    trips = [_t(0, "riding", 3, 0.99), _t(0, "wearing", 1, 0.80)]
    out = select_edges(trips, BOXES, SHAPE, max_edges=4)
    assert [e.score for e in out] == [0.99, 0.80]


def test_out_of_range_indices_are_ignored():
    """The head is built for a fixed box count and zero-padded."""
    assert select_edges([_t(0, "on", 99)], BOXES, SHAPE, max_edges=4) == []


# --------------------------------------------------------------------------- #
# geometry and colour
# --------------------------------------------------------------------------- #

def test_bezier_reaches_both_ends_and_bows():
    pts = bezier((0, 0), (100, 0), bulge=0.2, min_bow=10)
    assert np.allclose(pts[0], (0, 0)) and np.allclose(pts[-1], (100, 0))
    assert abs(pts[len(pts) // 2][1]) > 5          # bowed off the chord


def test_bezier_bows_short_edges_by_a_floor():
    """A proportional bow draws a short edge as a smudge under its own chip."""
    pts = bezier((0, 0), (6, 0), bulge=0.18, min_bow=46)
    assert abs(pts[len(pts) // 2][1]) > 20


def test_bezier_sign_flips_the_side():
    up = bezier((0, 0), (100, 0), bulge=0.2)[24][1]
    down = bezier((0, 0), (100, 0), bulge=-0.2)[24][1]
    assert np.sign(up) == -np.sign(down)


def test_decode_masks_crops_to_its_own_box():
    """An uncropped prototype mask fires on every instance of the class."""
    proto = np.zeros((1, 8, 8), np.float32)
    proto[0] = 1.0                                  # logit > 0 everywhere
    coef = np.ones((1, 1), np.float32)
    lb = np.array([[0, 0, 32, 32]], np.float32)     # top-left quarter, imgsz=64
    idx = decode_masks(coef, proto, np.array([0]), lb, imgsz=64)
    assert idx[:4, :4].all()                        # inside the box
    assert not idx[4:, :].any() and not idx[:, 4:].any()   # nowhere else


def test_decode_masks_paints_small_boxes_last():
    """A small object in front of a large one keeps its own pixels."""
    proto = np.ones((1, 8, 8), np.float32)
    coef = np.ones((1, 2), np.float32)
    lb = np.array([[0, 0, 64, 64], [0, 0, 16, 16]], np.float32)   # big, then small
    idx = decode_masks(coef, proto, np.array([0, 1]), lb, imgsz=64)
    assert idx[0, 0] == 2                           # the small one won
    assert idx[7, 7] == 1


@pytest.mark.parametrize("rgb,word", [
    ((0.02, 0.02, 0.02), "black"),
    ((0.97, 0.97, 0.97), "white"),
    ((0.50, 0.50, 0.50), "gray"),
    ((0.90, 0.05, 0.05), "red"),
    ((0.05, 0.65, 0.15), "green"),
    ((0.05, 0.15, 0.85), "blue"),
])
def test_colour_bins_names_the_obvious_cases(rgb, word):
    got = COLOUR_WORDS[int(colour_bins(np.array([[rgb]], np.float32))[0, 0])]
    assert got == word


def test_colour_bins_is_vectorised_elementwise():
    img = np.random.default_rng(0).random((7, 5, 3)).astype(np.float32)
    whole = colour_bins(img)
    assert whole.shape == (7, 5)
    for y in range(7):
        for x in range(5):
            assert whole[y, x] == colour_bins(img[y, x][None, None])[0, 0]
