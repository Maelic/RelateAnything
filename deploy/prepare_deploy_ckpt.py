"""Bake a training checkpoint into a self-contained, portable deploy bundle.

Run this ONCE where dino.txt is available (e.g. the training box). It:
  1. loads the trained checkpoint,
  2. re-parameterizes the relation head to the deployment PREDICATE_VOCAB using
     the dino.txt text tower (this is the relation-detector re-parametrization),
  3. bundles the model weights (frozen backbone included), the baked vocabulary,
     and the backbone's config into ONE .pt file.

The resulting file needs NO dino.txt, NO HF download and NO network on the
target machine — just `pip install torch transformers` and this file.

    python deploy/prepare_deploy_ckpt.py \
        --checkpoint runs/train/full_v33a_50ep_v3/checkpoint_best.pth \
        --dinotxt checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
        --out relateanything_deploy.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relsgg.api import RelateAnything          # noqa: E402
from deploy.vocab import PREDICATE_VOCAB       # noqa: E402


def _embed_text_encoder(student_path: str, fp16: bool) -> dict:
    """Pack the distilled student AND the CLIP-BPE tokenizer into the bundle.

    Without the tokenizer the student is useless offline: it would try to pull
    openai/clip-vit-base-patch32 from the Hub on the first encode. Only the
    tokenizer files are taken (a few MB) — not the 1.2 GB the HF cache holds
    for that repo, almost all of which is CLIP's image tower.
    """
    from transformers import CLIPTokenizer
    from relsgg.text_student import (CLIP_TOKENIZER_ID, CLIP_TOKENIZER_FILES,
                                     CLIP_TOKENIZER_LAYOUTS)

    if not student_path or not os.path.exists(student_path):
        raise SystemExit(
            f"[prepare] --embed-text-encoder needs the text student; "
            f"got {student_path!r}. Pass --text-student explicitly.")

    ck = torch.load(student_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    if fp16:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}

    # materialise the tokenizer to a temp dir, then slurp its files as bytes
    import tempfile
    d = tempfile.mkdtemp(prefix="ra_tok_")
    CLIPTokenizer.from_pretrained(CLIP_TOKENIZER_ID).save_pretrained(d)
    files = {f: open(os.path.join(d, f), "rb").read()
             for f in CLIP_TOKENIZER_FILES if os.path.exists(os.path.join(d, f))}
    if not any(all(f in files for f in layout) for layout in CLIP_TOKENIZER_LAYOUTS):
        raise SystemExit(
            f"[prepare] tokenizer incomplete: got {sorted(files)}, need one of "
            f"{[list(l) for l in CLIP_TOKENIZER_LAYOUTS]}")

    return {"cfg": ck["cfg"], "clip_ids": ck["clip_ids"], "state_dict": sd,
            "tokenizer_files": files,
            "source": os.path.basename(student_path)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dinotxt", default="",
                    help="dino.txt weights (.pth) — v33 and earlier checkpoints")
    ap.add_argument("--text-student", dest="text_student", default=None,
                    help="distilled student ckpt (v34+); default: read from checkpoint args")
    ap.add_argument("--out", default="relateanything_deploy.pt")
    ap.add_argument("--predicates", nargs="*", default=None,
                    help="override predicate vocabulary (default deploy/vocab)")
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--score_mode", default="sigmoid", choices=["sigmoid", "softmax"])
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--fp16", action="store_true",
                    help="store weights as fp16 (halves the file; from_deploy "
                         "upcasts to fp32 at load, so compute is unchanged)")
    ap.add_argument("--embed-text-encoder", dest="embed_text_encoder",
                    action="store_true",
                    help="embed the distilled text student + its CLIP-BPE "
                         "tokenizer so the target can call set_vocabulary() on "
                         "ARBITRARY strings offline. Costs ~24 MB at --fp16 — "
                         "less than shipping the 19K-row trained W bank, and "
                         "unbounded rather than limited to trained predicates.")
    args = ap.parse_args()

    preds = args.predicates if args.predicates else PREDICATE_VOCAB
    print(f"[prepare] re-parameterizing to {len(preds)} predicates")
    ra = RelateAnything.from_checkpoint(
        args.checkpoint, preds, dinotxt_weights=args.dinotxt,
        text_student=args.text_student,
        device="cpu", weights=args.weights, score_mode=args.score_mode,
        img_size=args.img_size)
    print(f"[prepare] text encoder: "
          f"{'student ' + str(ra.text_student) if ra.text_student else 'dino.txt'}")

    # backbone config so the target can rebuild the arch from config alone
    bb_cfg = ra.model.backbone.model.config
    backbone_config = bb_cfg.to_dict() if hasattr(bb_cfg, "to_dict") else None

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    a = ck["args"]; a = a if isinstance(a, dict) else vars(a)

    sd = ra.model.state_dict()
    if args.fp16:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}

    bundle = {
        "model": sd,                      # backbone + head + BAKED W/alpha/gates
        "args": a,
        "backbone_config": backbone_config,
        "predicates": preds,
        "img_size": args.img_size,
        "score_mode": args.score_mode,
        "storage_dtype": "fp16" if args.fp16 else "fp32",
        "source_checkpoint": os.path.basename(args.checkpoint),
    }

    if args.embed_text_encoder:
        bundle["text_encoder"] = _embed_text_encoder(ra.text_student, args.fp16)
        print(f"[prepare] embedded text encoder + CLIP-BPE tokenizer "
              f"-> set_vocabulary() works offline on arbitrary strings")
    torch.save(bundle, args.out)
    sz = os.path.getsize(args.out) / 1e6
    print(f"[prepare] wrote {args.out}  ({sz:.0f} MB, self-contained)")
    print(f"[prepare] predicates baked: {preds[:8]}{'...' if len(preds) > 8 else ''}")


if __name__ == "__main__":
    main()
