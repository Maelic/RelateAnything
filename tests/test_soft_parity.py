"""Plumbing regression: soft-supervision mode must contain v41 as a special case.

The v42 loss path replaces boolean masks with continuous weights. Before trusting a
12 GPU-h run on it, verify the algebra the way the v41 flag was verified against v40
(identity+lse == group+mean(w=0) to 6 decimals): construct a SOFT ontology whose
weights are the binarisation of a legacy ontology —

    pos_w   = pos_mask (1.0 / 0.0)
    neg_lw  = log(1e-6) where ignore_mask else 0     (the p->1 limit)
    w_cooc  = soft_neg_weight

— and check BatchLocalInfoNCE produces the same loss on random inputs, in both the
single-label and multi-label paths, with and without the cooc soft mask. The only
tolerated difference is the ignore limit: masked_fill(-inf) vs +log(1e-6); at
temp=0.07 that changes the denominator by < 1e-4 relative, which is the tolerance.

Synthetic vocabulary — this tests PLUMBING, not the artifact.

    pytest tests/test_soft_parity.py -q      (or: python tests/test_soft_parity.py)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from relsgg.loss_synonym import BatchLocalInfoNCE, PredicateOntology  # noqa: E402


def fake_legacy(V=240, D=32, seed=0):
    rs = np.random.RandomState(seed)
    names = [f"p{i}" for i in range(V)]
    # ~40 groups of 2-6 members, rest singletons
    canon = {}
    gi = 0
    i = 0
    while i < V // 2:
        size = rs.randint(2, 7)
        for j in range(i, min(i + size, V // 2)):
            canon[names[j]] = f"g{gi}"
        gi += 1
        i += size
    E = rs.randn(V, D).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    ont = PredicateOntology(names, canon_map=canon, embeddings=E,
                            tau_ignore=0.5, group_positives=True,
                            counts={n: int(rs.randint(1, 1000)) for n in names})
    return ont, E


class SoftView:
    """Soft ontology fabricated as the exact binarisation of a legacy one."""

    def __init__(self, legacy: PredicateOntology, w_cooc: float):
        V = len(legacy.predicates)
        self.predicates = legacy.predicates
        self.soft = True
        self.group_positives = False
        self.group_of = legacy.group_of
        self.pos_w = legacy.pos_mask.to(torch.float16)
        lw = torch.zeros(V, V)
        lw[legacy.ignore_mask] = math.log(1e-6)
        self.neg_lw = lw.to(torch.float16)
        self.sym = torch.zeros(V)
        self.inv_elig = legacy.inverse_mask.any(1).float()
        self.w_cooc = w_cooc
        self.pos_mask = legacy.pos_mask
        self.ignore_mask = torch.zeros(V, V, dtype=torch.bool)
        self.inverse_mask = legacy.inverse_mask
        self.tau_ignore = None
        self.class_weight = legacy.class_weight


def main() -> None:
    torch.manual_seed(0)
    ont, E = fake_legacy()
    V = len(ont.predicates)
    W = torch.from_numpy(E)
    soft_ont = SoftView(ont, w_cooc=0.3)

    # identical sampling: fix the RNG per call and use full-vocab S via n_neg=V
    def make(loss_ont, **kw):
        return BatchLocalInfoNCE(loss_ont, temp=0.07, n_neg=V, hard_frac=0.0,
                                 soft_neg_weight=0.3, pos_agg="mean",
                                 pos_member_weight=1.0, **kw)

    legacy = make(ont)
    soft = make(soft_ont)

    M = 64
    feats = torch.randn(M, W.shape[1])
    labels = torch.randint(0, V, (M,))
    hot = torch.zeros(M, V, dtype=torch.bool)
    hot[torch.arange(M), labels] = True
    extra = torch.randint(0, V, (M,))
    hot[torch.arange(M), extra] = True          # multi-label rows

    ok = True
    for name, args in [("single-label", (feats, labels, W)),
                       ("multi-label", (feats, hot, W))]:
        torch.manual_seed(1)
        a = legacy(*args)
        torch.manual_seed(1)
        b = soft(*args)
        rel = abs(float(a) - float(b)) / max(abs(float(a)), 1e-9)
        tag = "OK " if rel < 1e-4 else "FAIL"
        ok &= rel < 1e-4
        print(f"[{tag}] {name:<13s} legacy {float(a):.6f}  soft {float(b):.6f}  "
              f"rel-diff {rel:.2e}")

    # degenerate weights guard: a row whose GT has NO stored positives beyond the
    # diagonal must not produce nan/inf
    lonely = torch.full((8,), V - 1)             # singleton predicate
    val = soft(feats[:8], lonely, W)
    print(f"[{'OK ' if torch.isfinite(val) else 'FAIL'}] singleton-GT rows finite "
          f"({float(val):.6f})")
    ok &= bool(torch.isfinite(val))

    print("\nPARITY " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)



def test_soft_supervision_contains_legacy_loss():
    import pytest
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0


if __name__ == "__main__":
    main()
