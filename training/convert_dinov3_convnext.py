"""Convert torchhub DINOv3-ConvNeXt weights into an offline HF model dir.

WHY THIS EXISTS. `checkpoints/dinov3_convnext_{tiny,small,base,large}_*.pth` are
raw state dicts from the original dinov3 repo, but `relsgg/backbone.py` builds
its backbone with `AutoModel.from_pretrained(...)` under `HF_HUB_OFFLINE=1`.
transformers 5.x ships `DINOv3ConvNextModel`, so the architecture is available
locally — only the weights are in the wrong layout and there is no HF snapshot
to download. This writes one.

KEY MAPPING (verified by exact-coverage assertions, not by eyeball):

    downsample_layers.{i}.{k}.*   ->  model.stages.{i}.downsample_layers.{k}.*
    stages.{i}.{j}.gamma          ->  model.stages.{i}.layers.{j}.gamma
    stages.{i}.{j}.dwconv.*       ->  model.stages.{i}.layers.{j}.depthwise_conv.*
    stages.{i}.{j}.norm.*         ->  model.stages.{i}.layers.{j}.layer_norm.*
    stages.{i}.{j}.pwconv{1,2}.*  ->  model.stages.{i}.layers.{j}.pointwise_conv{1,2}.*
    norm.*                        ->  layer_norm.*

`norms.{i}.*` (the dinov3 repo's per-stage output norms) have NO counterpart in
the HF module: HF normalises once at the end. They are dropped DELIBERATELY and
reported, rather than silently ignored, because a dropped tensor is exactly the
kind of thing that turns into a mysterious accuracy deficit later.

Geometry is read off the checkpoint (`hidden_sizes` from the downsample convs,
`depths` by counting blocks) instead of hardcoded per variant, so a variant we
have not looked at cannot be mis-declared.

    python training/convert_dinov3_convnext.py --variant tiny
    python training/convert_dinov3_convnext.py --all
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import torch

_OUT_ROOT = "checkpoints/hf"


def _load_raw(path: str) -> dict:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return sd.get("model", sd)


def _geometry(sd: dict) -> tuple[list[int], list[int]]:
    """hidden_sizes from the stem/downsample convs, depths by counting blocks."""
    hidden = []
    for i in range(4):
        w = sd[f"downsample_layers.{i}.0.weight"]
        # stage 0's downsample is [conv, norm]; stages 1-3 are [norm, conv]
        hidden.append(w.shape[0] if w.dim() == 4
                      else sd[f"downsample_layers.{i}.1.weight"].shape[0])
    depths = []
    for i in range(4):
        blocks = {int(m.group(1)) for k in sd
                  if (m:= re.match(rf"stages\.{i}\.(\d+)\.", k))}
        depths.append(len(blocks))
    return hidden, depths


def _remap(sd: dict) -> tuple[dict, list[str]]:
    out, dropped = {}, []
    sub = {"dwconv": "depthwise_conv", "norm": "layer_norm",
           "pwconv1": "pointwise_conv1", "pwconv2": "pointwise_conv2"}
    for k, v in sd.items():
        if k.startswith("norms."):
            dropped.append(k)
            continue
        if k.startswith("norm."):
            out["layer_norm." + k[len("norm."):]] = v
        elif (m:= re.match(r"downsample_layers\.(\d+)\.(\d+)\.(.+)", k)):
            i, j, rest = m.group(1), m.group(2), m.group(3)
            out[f"model.stages.{i}.downsample_layers.{j}.{rest}"] = v
        elif (m:= re.match(r"stages\.(\d+)\.(\d+)\.(\w+)\.?(.*)", k)):
            i, j, part, rest = m.groups()
            if part == "gamma":
                out[f"model.stages.{i}.layers.{j}.gamma"] = v
            else:
                out[f"model.stages.{i}.layers.{j}.{sub[part]}."
                    f"{rest}"] = v
        else:
            raise KeyError(f"unmapped source key: {k}")
    return out, dropped


def convert(variant: str, dry_run: bool = False) -> str:
    from transformers import DINOv3ConvNextConfig, DINOv3ConvNextModel

    hits = sorted(glob.glob(f"checkpoints/dinov3_convnext_{variant}_*.pth"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly 1 checkpoint for {variant}, got {hits}")
    src = hits[0]

    sd = _load_raw(src)
    hidden, depths = _geometry(sd)
    cfg = DINOv3ConvNextConfig(hidden_sizes=hidden, depths=depths)
    model = DINOv3ConvNextModel(cfg)

    new_sd, dropped = _remap(sd)
    target = dict(model.state_dict())

    missing = sorted(set(target) - set(new_sd))
    unexpected = sorted(set(new_sd) - set(target))
    bad_shape = [k for k in set(new_sd) & set(target)
                 if tuple(new_sd[k].shape) != tuple(target[k].shape)]

    print(f"[{variant}] {src}")
    print(f"  hidden_sizes={hidden} depths={depths}")
    print(f"  source keys {len(sd)} -> mapped {len(new_sd)}, "
          f"target expects {len(target)}")
    print(f"  dropped (no HF counterpart): {len(dropped)}"
          + (f" e.g. {dropped[:3]}" if dropped else ""))
    if missing:
        print(f"  !! MISSING {len(missing)}: {missing[:6]}")
    if unexpected:
        print(f"  !! UNEXPECTED {len(unexpected)}: {unexpected[:6]}")
    if bad_shape:
        print(f"  !! SHAPE MISMATCH {len(bad_shape)}: {bad_shape[:6]}")
    if missing or unexpected or bad_shape:
        raise SystemExit(f"[{variant}] conversion is not exact — refusing to write")

    model.load_state_dict(new_sd, strict=True)

    # Forward parity is what actually matters; the key mapping only makes it
    # possible. Cannot check against the source model (the dinov3 repo is not
    # installed), so check the invariants we depend on downstream instead:
    # 4 stage maps at strides 4/8/16/32 with the declared channel counts.
    model.eval()
    with torch.no_grad():
        o = model(pixel_values=torch.zeros(1, 3, 448, 448),
                  output_hidden_states=True)
    hs = o.hidden_states
    print(f"  hidden_states: {[tuple(h.shape) for h in hs]}")
    strides = [448 // h.shape[-1] for h in hs]
    assert strides[-4:] == [4, 8, 16, 32], f"unexpected strides {strides}"

    out_dir = os.path.join(_OUT_ROOT, f"convnext_{variant}_lvd1689m")
    if dry_run:
        print(f"  dry-run: would write {out_dir}")
        return out_dir
    model.save_pretrained(out_dir)
    print(f"  saved -> {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="tiny",
                   choices=["tiny", "small", "base", "large"])
    p.add_argument("--all", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()
    for v in (["tiny", "small", "base", "large"] if a.all else [a.variant]):
        convert(v, a.dry_run)


if __name__ == "__main__":
    main()
