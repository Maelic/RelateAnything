#!/usr/bin/env python
"""Convert local Meta-format DINOv3 checkpoints (checkpoints/*.pth) to HF
format without touching the gated facebook/dinov3-* hub repos.

Wraps the official transformers conversion script (fetched from GitHub, since
pip wheels don't ship convert_*.py) and monkeypatches hf_hub_download to
return the local file. The official script then verifies the converted model
against hardcoded expected outputs — a real correctness check.

Usage (login node,.venv):
    python training/convert_dinov3_local.py --script /path/to/convert_dinov3.py \
        --models vitb16_lvd1689m vits16_lvd1689m vits16plus_lvd1689m

Output: checkpoints/hf/<model_name>/ — loadable via
    AutoModel.from_pretrained("checkpoints/hf/vitb16_lvd1689m")
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
CKPT_DIR = PROJ / "checkpoints"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True,
                    help="Path to the official convert_dinov3_vit_to_hf.py")
    ap.add_argument("--models", nargs="+", default=["vitb16_lvd1689m"])
    ap.add_argument("--save_dir", default=str(CKPT_DIR / "hf"))
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("convert_dinov3", args.script)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["convert_dinov3"] = mod
    spec.loader.exec_module(mod)

    def local_download(repo_id: str, filename: str) -> str:
        p = CKPT_DIR / filename
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found — place the Meta checkpoint there "
                f"(expected for {repo_id})"
)
        print(f"[local] using {p}")
        return str(p)

    mod.hf_hub_download = local_download

    # Meta checkpoints carry CUDA storages; login nodes are CPU-only.
    import torch
    _orig_load = torch.load

    def _cpu_load(*a, **k):
        k.pop("mmap", None)
        k["map_location"] = "cpu"
        return _orig_load(*a, **k)

    mod.torch.load = _cpu_load

    for name in args.models:
        print(f"\n=== converting {name} ===")
        ns = argparse.Namespace(model_name=name, save_dir=args.save_dir,
                                push_to_hub=False)
        mod.convert_and_test_dinov3_checkpoint(ns)


if __name__ == "__main__":
    main()
