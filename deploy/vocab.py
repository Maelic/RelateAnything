"""Deployment vocabularies.

Both vocabularies are swappable — that is the whole point of RelateAnything:
  * OBJECT_VOCAB     the open-vocab detector's classes (YOLO-World / YOLOE)
  * PREDICATE_VOCAB  the relation head's predicates (distilled student encoder)

OBJECT_VOCAB defaults to MEGASG's own 497 categories, which is exactly what the
shipped detector checkpoints were re-parameterized to
(`checkpoints/detectors/yolov8x-worldv2_megasg497.pt`, `yoloe-11l-megasg497.pt`)
— verified to match `runs/packed/megasg/train/meta.json` in order. Keeping the
detector's label space equal to the relation model's training label space is
what makes the two halves agree about what an object *is*.
"""
from __future__ import annotations

import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

# --- objects: MEGASG's 497 categories, in the detector's class order ---------
with open(os.path.join(_HERE, "megasg_categories.json")) as _f:
    MEGASG_OBJECT_VOCAB = json.load(_f)

# A small everyday-desk subset, for when you want a faster/cleaner demo than
# 497 classes gives. Not the default — pass --objects webcam to use it.
WEBCAM_OBJECT_VOCAB = [
    "person", "chair", "desk", "table", "laptop", "computer keyboard",
    "computer mouse", "computer monitor", "mobile phone", "cup", "mug",
    "bottle", "book", "pen", "backpack", "bag", "headphones", "glasses",
    "hat", "shirt", "jacket", "watch", "houseplant", "lamp", "clock",
    "picture frame", "window", "door", "couch", "pillow", "remote control",
    "television", "box", "can", "bowl", "plate", "food", "banana", "apple",
    "ball", "guitar", "bicycle", "dog", "cat",
]

# --- predicates -------------------------------------------------------------
# The SELECTION below is editorial (which predicates a demo viewer cares
# about) and stays in code. Every NUMBER — per-predicate recall, thresholds,
# alpha — is per-checkpoint and lives in that model's generated artifacts
# (predicate_bank.npz via build_predicate_bank.py, thresholds.json via
# calibrate_thresholds.py). Score scales are specific to a checkpoint, so no
# threshold belongs in this file — read them with load_bank().
#
# The list drops "near", "next to" and "under": the first two are the
# chattiest, least informative spatial relations and "beside" covers their
# meaning. Synonyms are deliberately kept apart ("on" / "resting on" / "on top
# of"): the model was trained on a synonym-rich vocabulary, and suppressing
# variants distorts its scores. They compete at argmax time.
PREDICATE_VOCAB = [
    # people doing things (the interesting half)
    "wearing", "riding", "playing", "sitting on", "sitting at", "holding",
    "sitting in", "looking at", "using", "watching", "standing on",
    "carrying", "talking to", "smiling at", "standing beside", "walking past",
    "posing with", "leaning against",
    # object-object structure
    "part of", "resting on", "on", "covering", "inside", "on top of",
    "contained in", "hanging from", "surrounding", "attached to",
    # spatial frame
    "in front of", "beside", "to the left of", "to the right of", "behind",
    "above", "below",
]


def load_bank(path_or_dir):
    """Load a per-model predicate bank (deploy/dist/<model_id>/).

    Returns a dict of aligned arrays: names [V], W [V,768] float32 unit-norm,
    alpha [V], thr [V] (NaN = uncalibrated), best_f1 [V], is_spatial [V] bool,
    type_source [V], recall [V], gt [V], plus provenance (checkpoint, student).
    Numpy-only — safe on torch-less edge devices.
    """
    import numpy as _np
    p = path_or_dir
    if os.path.isdir(p):
        p = os.path.join(p, "predicate_bank.npz")
    z = _np.load(p, allow_pickle=True)
    out = {k: z[k] for k in z.files}
    out["names"] = [str(x) for x in out["names"]]
    out["is_spatial"] = out["is_spatial"].astype(bool)
    return out


def _dedup(xs):
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


MEGASG_OBJECT_VOCAB = _dedup(MEGASG_OBJECT_VOCAB)
WEBCAM_OBJECT_VOCAB = _dedup(WEBCAM_OBJECT_VOCAB)
PREDICATE_VOCAB = _dedup(PREDICATE_VOCAB)

# Default object vocabulary = MEGASG's, matching the shipped detector weights.
OBJECT_VOCAB = MEGASG_OBJECT_VOCAB
