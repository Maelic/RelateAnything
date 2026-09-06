"""Merge LoRA adapters into the backbone base weights for release/export.

WHY. A LoRA checkpoint's backbone is a PEFT wrapper: every adapted projection
runs base_W @ x + B @ A @ x at inference. Exporting that to ONNX bakes the
unmerged lora_A/lora_B matmuls into the graph — numerically correct but
larger and slower on the stage that already dominates frame time (the
backbone is ~69% of the laptop budget). `merge_and_unload()` folds
B@A*scale into the base weights, after which the model is indistinguishable
from a full-FT one.

The re-saved checkpoint's args get lora_rank=0 so EVERY downstream consumer
(export, bank, thresholds, api) treats it as full-FT — nothing else needs a
LoRA special case. Full-FT checkpoints skip this script entirely.

    python deploy/merge_lora.py --checkpoint runs/train/<run>/checkpoint_best.pth
    -> runs/train/<run>/checkpoint_best_merged.pth  (+ parity report)
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".hf_cache"))


def _merge_sd(ckpt: dict, which: str) -> dict:
    """Build the model from ckpt[which], merge PEFT, return a plain state dict
    whose keys match a lora_rank=0 construction."""
    from relsgg.checkpoint import build_model_from_ckpt

    ck = dict(ckpt)
    # build_model_from_ckpt reads 'ema_model'/'model' internally via `which`
    model = build_model_from_ckpt(ck, which)
    peft_model = model.backbone.model
    if not hasattr(peft_model, "merge_and_unload"):
        raise SystemExit("[merge] backbone is not PEFT-wrapped — this is "
                         "already a full-FT/frozen checkpoint; nothing to do.")
    model.backbone.model = peft_model.merge_and_unload()
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default=None,
                    help="default: <checkpoint>_merged.pth alongside")
    ap.add_argument("--parity_images", type=int, default=32,
                    help="megasg val images for the merged-vs-unmerged check")
    ap.add_argument("--parity_tol", type=float, default=1e-4)
    ap.add_argument("--data_root", default="runs/packed/megasg")
    ap.add_argument("--skip_parity", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    args = ckpt.get("args") or {}
    args = dict(args if isinstance(args, dict) else vars(args))
    if int(args.get("lora_rank", 0)) <= 0:
        raise SystemExit(f"[merge] lora_rank={args.get('lora_rank')} — "
                         "not a LoRA checkpoint, nothing to merge.")

    from relsgg.checkpoint import build_model_from_ckpt

    out_path = a.out or a.checkpoint.replace(".pth", "_merged.pth")
    new_ckpt = {k: v for k, v in ckpt.items()
                if k not in ("model", "ema_model", "optimizer", "scheduler",
                             "scaler")}
    new_args = dict(args)
    new_args["lora_rank"] = 0          # downstream treats it as full-FT
    new_args["lora_layers"] = None
    new_args["_merged_from_lora_rank"] = args.get("lora_rank")
    new_ckpt["args"] = new_args

    merged = {}
    for which in ("model", "ema_model"):
        if which not in ckpt or not ckpt[which]:
            continue
        sd, merged_model = _merge_sd(ckpt, "ema" if which == "ema_model" else "raw")
        merged[which] = sd
        print(f"[merge] {which}: {len(sd)} tensors after merge_and_unload")
    new_ckpt.update(merged)

    # --- parity: merged (fresh lora_rank=0 build) vs original wrapper --------
    if not a.skip_parity:
        from data.relation_dataset import RelationDataset, collate_fn
        from torch.utils.data import DataLoader, Subset
        import numpy as np

        orig = build_model_from_ckpt(ckpt, "ema").eval()
        m_ck = copy.deepcopy(new_ckpt)
        rebuilt = build_model_from_ckpt(m_ck, "ema").eval()

        tr = RelationDataset(root=a.data_root, split="train", resolution=448)
        ds = RelationDataset(root=a.data_root, split="val", resolution=448,
                             max_objects=40, cat_to_idx=tr.cat_to_idx,
                             rel_cat_to_idx=tr.rel_cat_to_idx)
        idx = np.random.default_rng(0).permutation(len(ds))[:a.parity_images]
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=4,
                            shuffle=False, collate_fn=collate_fn)
        worst = 0.0
        with torch.no_grad():
            for images, boxes, counts, _ in loader:
                o1 = orig(images, boxes, counts, targets=None)
                o2 = rebuilt(images, boxes, counts, targets=None)
                d = (o1["logits"].float() - o2["logits"].float()).abs().max()
                worst = max(worst, float(d))
        print(f"[merge] parity over {a.parity_images} images: "
              f"max |dlogit| = {worst:.2e} (tol {a.parity_tol:g})")
        if worst > a.parity_tol:
            raise SystemExit("[merge] PARITY FAILED — not writing the merged "
                             "checkpoint.")

    torch.save(new_ckpt, out_path)
    print(f"[merge] wrote {out_path} "
          f"({os.path.getsize(out_path)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
