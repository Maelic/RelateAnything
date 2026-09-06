"""WiSE-FT backbone blend: theta_alpha = alpha * theta_ft + (1-alpha) * theta_init.

Interpolates ONLY the DINOv3-pretrained backbone weights toward their
pretrained init (Wortsman et al., robust fine-tuning). Everything trained
from scratch keeps its fine-tuned value:
  - the whole relation head (no meaningful init anchor),
  - backbone.layer_weights (zero-init scalar combiner, from scratch),
Both "model" and "ema_model" are blended; optimizer/scheduler state is
dropped so the output stays ~1 GB on a 98%-full filesystem.

Usage:
  python training/wise_ft_blend.py --checkpoint <ckpt> --alpha 0.8 \
      --out runs/train/<run>_wise0.8/checkpoint_best.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from relsgg.model.backbone import Backbone

SKIP_SUBSTRINGS = ("layer_weights",)


def blend_state_dict(sd: dict, init_sd: dict, alpha: float) -> tuple[dict, int, int]:
    out, n_blend, n_keep = {}, 0, 0
    for k, v in sd.items():
        blendable = (
            k.startswith("backbone.")
            and not any(s in k for s in SKIP_SUBSTRINGS)
            and k in init_sd
            and torch.is_floating_point(v)
            and init_sd[k].shape == v.shape
)
        if blendable:
            out[k] = alpha * v.float() + (1.0 - alpha) * init_sd[k].float()
            out[k] = out[k].to(v.dtype)
            n_blend += 1
        else:
            out[k] = v
            n_keep += 1
    return out, n_blend, n_keep


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = ck["args"]
    if not isinstance(a, dict):
        a = vars(a)

    backbone = Backbone(
        backbone_type=a["backbone_type"],
        **({"model_name": a["backbone_model"]} if a.get("backbone_model") else {}),
        **({"patch_size": a["patch_size"]} if a.get("patch_size") else {}),
        pretrained=True,
)
    init_sd = {f"backbone.{k}": v for k, v in backbone.state_dict().items()}

    slim = {k: v for k, v in ck.items()
            if k not in ("optimizer", "scheduler", "scaler")}
    for sdk in ("model", "ema_model"):
        sd = slim.get(sdk)
        if not sd:
            continue
        blended, n_blend, n_keep = blend_state_dict(sd, init_sd, args.alpha)
        # drift sanity: mean |ft - init| over blended tensors, before/after
        drift = torch.stack([
            (sd[k].float() - init_sd[k].float()).abs().mean()
            for k in sd if k in init_sd and torch.is_floating_point(sd[k])
            and not any(s in k for s in SKIP_SUBSTRINGS) and k.startswith("backbone.")
        ]).mean()
        slim[sdk] = blended
        print(f"{sdk}: blended {n_blend} backbone tensors at alpha={args.alpha}"
              f" (kept {n_keep}); mean |ft-init| drift {drift:.5f}"
              f" -> {args.alpha * drift:.5f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, out)
    print(f"saved {out} ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
