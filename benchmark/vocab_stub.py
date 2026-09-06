"""vocab_stub.py -- shape-correct stand-in vocabulary matrices for COST measurement.

WHY THIS EXISTS
---------------
The vocabulary head is reparameterised into a static (V x text_dim) matrix, so its
latency and its FLOPs depend on the SHAPE of that matrix and on nothing else -- the
values change what the model predicts, never how much work it does. That makes a
stand-in matrix exact for a cost benchmark and worthless for anything else.

It is needed because the 512-d text student
(`runs/packed/text_student_v2_512/student.pt`) and its precomputed embeddings
(`.../pred_embeds_studentv2_512_photo.npz`) did not survive the cluster migration; the
released checkpoints reference both by path. Rather than block the cost table on
re-distilling a text encoder, the cost tools may substitute a matrix of the right
shape -- but ONLY when asked explicitly, and every consumer must record that it did.

NEVER import this from an accuracy path. `stub_vocabulary` returns noise; any recall,
mR or F1 computed against it is meaningless. The call sites gate it behind an explicit
--synthetic_vocab flag and stamp `synthetic_vocab: true` into their output json.
"""
from __future__ import annotations

import os

import numpy as np
import torch


def real_vocabulary(names, text_student: str, pred_embeds: str, device, templates):
    """Return (names, E) from the real encoder/embeddings, or None if absent."""
    if pred_embeds and os.path.exists(pred_embeds):
        z = np.load(pred_embeds)
        return [str(q) for q in z["predicates"]], z["embeddings"]
    if text_student and os.path.exists(text_student):
        from relsgg.text.student import encode_texts_student
        return list(names), encode_texts_student(list(names), text_student,
                                                 templates=templates, device=device)
    return None


def stub_vocabulary(names, text_dim: int, device, seed: int = 0):
    """Unit-norm Gaussian rows: the shape the head will actually see.

    Rows are L2-normalised because the head scores by cosine against them -- an
    unnormalised stand-in would leave the logits at a wildly different scale, and
    while that cannot change the FLOP count it can change how often downstream
    thresholds fire, which WOULD change measured work in the decode path.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    E = torch.randn(len(names), text_dim, generator=g)
    E = torch.nn.functional.normalize(E, dim=-1)
    return list(names), E.to(device)
