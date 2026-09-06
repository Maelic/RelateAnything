"""Shared helpers for running OvSGTR checkpoints outside their training harness.

OvSGTR's own eval path (`main.py` -> `datasets/vg.py` -> `SggEvaluator`) is welded to
VG150's `stanford_filtered` HDF5 layout and to their metric implementation. We need
neither: for a fair cross-model comparison both models must be scored by the SAME
evaluator on the SAME images. So we import only their *model*, run it ourselves, and
emit a model-agnostic interchange record that our evaluator consumes.

Runs under OvSGTR's isolated venv (python 3.11 / torch 2.1.2), NOT the RelSGG venv.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence

OVSGTR_ROOT = Path(
    os.environ.get("OVSGTR_ROOT", "third_party/OvSGTR")
)


def _ensure_ovsgtr_on_path() -> None:
    """OvSGTR uses implicit top-level imports (`models`, `util`, `datasets`)."""
    root = str(OVSGTR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def build_args(config_file: str | Path, **overrides) -> argparse.Namespace:
    """Replicate main.py's config->args merge without its argparse/DDP scaffolding.

    main.py raises if a config key collides with an existing arg; here the config is
    the base and `overrides` win, which is what lets us flip `use_gt_box` per protocol.
    """
    _ensure_ovsgtr_on_path()
    from util.slconfig import SLConfig

    cfg = SLConfig.fromfile(str(config_file))
    args = argparse.Namespace(**cfg._cfg_dict.to_dict())

    # Defaults main.py injects outside the config file.
    for k, v in dict(device="cuda", use_ema=False, debug=False, amp=False).items():
        if not hasattr(args, k):
            setattr(args, k, v)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def load_ovsgtr(config_file: str | Path, checkpoint: str | Path, device: str = "cuda",
                use_gt_box: bool = False, **overrides):
    """Build the model + postprocessor and load a released checkpoint.

    Returns (model, postprocessor, args). `strict=False` mirrors main.py:412 - the
    released checkpoints legitimately omit criterion-side buffers - so the caller MUST
    inspect the returned key report rather than trusting a silent load.
    """
    import torch

    _ensure_ovsgtr_on_path()
    # Inlined from main.py:99-106. Importing main.py itself would pull in wandb,
    # termcolor and the whole DDP training stack for two lines of registry lookup.
    import models  # noqa: F401  (registers the build funcs as a side effect)
    from models.registry import MODULE_BUILD_FUNCS
    from util.misc import clean_state_dict

    args = build_args(config_file, use_gt_box=use_gt_box, device=device, **overrides)
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict, \
        f"modelname {args.modelname} not registered"
    model, _criterion, postprocessors = MODULE_BUILD_FUNCS.get(args.modelname)(args)

    ckpt = torch.load(str(checkpoint), map_location="cpu")
    state = clean_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    model.to(device).eval()
    post = postprocessors["bbox"]

    # main.py:326-328 attaches the relation heads to the postprocessor AFTER
    # build_model_main returns. Without this graph_infer() receives rln_proj=None
    # and dies inside the no-grad forward, so it must be replicated here.
    # (The Hungarian matcher needed by use_gt_box is already wired inside
    # build_groundingdino at groundingdino.py:1001.)
    post.rln_proj = getattr(model, "rln_proj", None)
    post.rln_classifier = getattr(model, "rln_classifier", None)
    post.rln_freq_bias = getattr(model, "rln_freq_bias", None)
    if getattr(post, "do_sgg", False) and post.rln_proj is None:
        raise RuntimeError("do_sgg=True but model has no rln_proj; wrong config/checkpoint pair?")

    post.to(device)
    post.eval()
    return model, post, args, {"missing": list(missing), "unexpected": list(unexpected),
                               "epoch": ckpt.get("epoch")}


def install_vocabulary(post, nouns: Sequence[str], predicates: Sequence[str]) -> None:
    """Attach the benchmark vocabulary, mirroring engine.py:247-250.

    `name2classes` drives the object-label decoding; `name2predicates` indexes the
    columns of `all_relation`. Both must agree with the prompts from build_prompts().
    """
    post.name2classes = {n: i + 1 for i, n in enumerate(nouns)}
    post.name2predicates = name2predicates(predicates)


def preprocess_caption(caption: str) -> str:
    """Byte-identical to GroundingDINO's own helper (lowercase + trailing period)."""
    result = caption.lower().strip()
    return result if result.endswith(".") else result + "."


def build_prompts(nouns: Sequence[str], predicates: Sequence[str]) -> Dict[str, str]:
    """Construct the two text prompts exactly as datasets/vg.py:331,345 does.

    This is the open-vocabulary hook: OvSGTR scores relations by dot-product against
    the encoded `rel_caption`, and `graph_infer` maps token spans back to predicate
    names by decoding between separators. So an arbitrary benchmark vocabulary can be
    installed here with no model change - the same contract our own head has.

    NOTE: predicate names containing '.' would break the separator-based span split
    used in graph_infer (input_ids 101/102/1012 = [CLS]/[SEP]/'.'), so they are
    rejected loudly rather than silently mis-decoded.
    """
    bad = [p for p in predicates if "." in p] + [n for n in nouns if "." in n]
    if bad:
        raise ValueError(f"'.' is the prompt separator; offending entries: {bad[:5]}")
    return {
        "caption": preprocess_caption(". ".join(nouns)),
        "rel_caption": ". ".join(predicates) + ".",
    }


def name2predicates(predicates: Sequence[str], bg: str = "__background__") -> Dict[str, int]:
    """graph_infer indexes `all_relation` columns via this map.

    Two constraints, both load-bearing:
      * Index 0 must be background. graph_infer ranks with `all_relation[:, 1:].max(1)`
        (graph_infer.py:99), so a real predicate at column 0 is excluded from ranking.
      * The background entry must be PRESENT, because graph_infer allocates
        `torch.zeros((P, len(name2predicates)))` (graph_infer.py:87) and then writes at
        column `name2predicates[name]`. Omitting it makes the array one column short
        and the last predicate raises IndexError.
    This mirrors datasets/vg.py:122, where index 0 is VG150's '__background__'.
    """
    return {bg: 0, **{p: i + 1 for i, p in enumerate(predicates)}}
