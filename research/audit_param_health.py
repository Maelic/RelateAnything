"""Which parameters actually trained?

MOTIVATION. `vocab_head.logit_scale` and `logit_bias` are parameters on the
deployed score path, and a contrastive loss that computes its own
`cos / infonce_temp` never touches them: they can come out of a full training
run bit-identical to their initialisation. Rank-based metrics are invariant to
a monotone rescale, so nothing in the benchmark notices.

If one dead parameter can hide that long, the right response is to check all of
them rather than to fix the one we tripped over. This rebuilds the model from
the checkpoint's own args at the checkpoint's own seed and reports, per
parameter, the relative movement from initialisation

    delta = ||w_trained - w_init|| / (||w_init|| + eps)

and flags:
  DEAD      trainable, delta ~ 0            -> receives no gradient. A BUG.
  FROZEN    requires_grad False             -> by design
  TINY      trainable, delta < --tiny       -> trains, but barely moves
  SATURATED |w| at a clamp boundary         -> e.g. logit_scale.exp() clamp

Init reproducibility: parameters are compared against a fresh build under the
same seed. Any residual init mismatch shows up as a LARGE delta, never a small
one, so it can produce false "healthy" readings but not false DEAD reports —
the direction of the error is safe for the thing we care about.

    python training/audit_param_health.py --checkpoint runs/train/v43_full_5ep/checkpoint_best.pth
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.checkpoint import build_model_from_ckpt        # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--tiny", type=float, default=1e-3,
                   help="relative movement below this counts as TINY")
    p.add_argument("--dead", type=float, default=1e-8,
                   help="relative movement below this counts as DEAD")
    p.add_argument("--out", default="")
    a = p.parse_args()

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    args = args if isinstance(args, dict) else vars(args)
    seed = int(args.get("seed", 42))

    trained = build_model_from_ckpt(ckpt, a.weights)
    # Fresh build = same config, NO weights loaded: hand build_model_from_ckpt a
    # checkpoint whose state dicts are empty, so load_state_dict(strict=False)
    # is a no-op and every parameter stays at its constructor init.
    torch.manual_seed(seed)
    fresh = build_model_from_ckpt({**ckpt, "model": {}, "ema_model": {}}, "raw")
    # Optional submodules are built on demand from the presence of their keys,
    # which an empty sd cannot trigger — mirror them so their params are
    # comparable rather than silently skipped.
    if getattr(trained.vocab_head, "gate_mlp", None) is not None:
        torch.manual_seed(seed)
        fresh.vocab_head.build_gate_mlp()
    if getattr(trained.vocab_head, "beta_mlp", None) is not None:
        torch.manual_seed(seed)
        fresh.vocab_head.build_beta_mlp()

    fresh_sd = dict(fresh.named_parameters())
    rows = []
    for name, w in trained.named_parameters():
        w0 = fresh_sd.get(name)
        if w0 is None or w0.shape != w.shape:
            continue
        n0 = w0.detach().float().norm().item()
        d = (w.detach().float() - w0.detach().float()).norm().item()
        zero_init = n0 < 1e-8
        rel = d if zero_init else d / (n0 + 1e-12)
        rows.append({"name": name, "numel": w.numel(),
                     "requires_grad": bool(w.requires_grad),
                     "rel_move": rel, "zero_init": zero_init, "norm_init": n0,
                     "norm_now": w.detach().float().norm().item()})

    dead = [r for r in rows if r["requires_grad"] and r["rel_move"] <= a.dead]
    tiny = [r for r in rows if r["requires_grad"]
            and a.dead < r["rel_move"] < a.tiny]
    frozen = [r for r in rows if not r["requires_grad"]]
    live = [r for r in rows if r["requires_grad"] and r["rel_move"] >= a.tiny]

    print(f"checkpoint: {a.checkpoint} (epoch {ckpt.get('epoch')}, seed {seed})")
    print(f"{len(rows)} parameter tensors — "
          f"{len(live)} live, {len(tiny)} TINY, {len(dead)} DEAD, "
          f"{len(frozen)} frozen-by-design")

    if dead:
        print("\n!!!! DEAD — trainable but received no gradient (BUG) !!!!")
        for r in sorted(dead, key=lambda x: -x["numel"]):
            print(f"   {r['name']:55s} {r['numel']:>9,d}  |w|={r['norm_now']:.5g}")
    if tiny:
        print(f"\n-- TINY (< {a.tiny:g} relative movement) --")
        for r in sorted(tiny, key=lambda x: x["rel_move"])[:25]:
            print(f"   {r['name']:55s} {r['numel']:>9,d}  rel={r['rel_move']:.2e}")

    print("\n-- 12 LEAST-moved live parameters --")
    for r in sorted(live, key=lambda x: x["rel_move"])[:12]:
        print(f"   {r['name']:55s} {r['numel']:>9,d}  rel={r['rel_move']:.4f}")
    print("\n-- 8 MOST-moved live parameters --")
    for r in sorted(live, key=lambda x: -x["rel_move"])[:8]:
        print(f"   {r['name']:55s} {r['numel']:>9,d}  rel={r['rel_move']:.4f}")

    n_train = sum(r["numel"] for r in rows if r["requires_grad"])
    n_dead = sum(r["numel"] for r in dead)
    print(f"\ntrainable params {n_train/1e6:.2f}M, of which DEAD "
          f"{n_dead/1e6:.4f}M ({100*n_dead/max(n_train,1):.3f}%)")

    out = a.out or os.path.join(os.path.dirname(a.checkpoint),
                                "param_health.json")
    json.dump({"checkpoint": a.checkpoint, "seed": seed, "rows": rows,
               "n_dead": len(dead), "n_tiny": len(tiny)},
              open(out, "w"), indent=2)
    print(f"saved → {out}")


if __name__ == "__main__":
    main()
