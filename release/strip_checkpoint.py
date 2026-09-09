"""Strip a training checkpoint to its releasable core.

A training checkpoint carries optimizer moments, scheduler state, scaler and
history. None of it belongs in a public artifact, and the optimizer alone is
about two thirds of the file for full fine-tuning runs. The stripped file keeps
exactly what inference needs:

    model            the EMA weights (what every reported evaluation used),
                     less the entries the current model no longer defines
    args             the training config (from_checkpoint rebuilds from these),
                     with machine-local paths removed
    backbone_config  the backbone's Hugging Face config, so the tower is built
                     from config and never downloaded (no gated-repo login)
    pred_names       the training vocabulary (open-vocabulary eval + provenance)
    epoch            provenance

The text student the checkpoint names is copied next to the output as
``text_student.pt`` (with its CLIP tokenizer files), and the training
vocabulary is encoded with it once into ``predicate_embeddings.npz``, so that
``from_checkpoint(full_vocabulary=True)`` does not re-encode 19k strings on
every machine. That is the layout every released model repository uses.

Verified after writing: ``RelateAnything.from_checkpoint(strict=True)``
on the stripped file with the Hugging Face hub disabled. A stripped checkpoint
that cannot strict-load offline must not ship, so verification is the exit
condition.

    python release/strip_checkpoint.py \
        --checkpoint runs/train/<run>/checkpoint_last.pth \
        --out deploy/dist/<model_id>/model.pth
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".hf_cache"))

from relsgg.checkpoint import hub_id_for_backbone  # noqa: E402
from relsgg.text.student import resolve_student_path  # noqa: E402

KEEP = ("args", "pred_names", "epoch")
#: State-dict entries the current model no longer defines.
from relsgg.checkpoint import OBSOLETE_KEYS  # noqa: E402
#: Arg values that name machine-local files. They are rewritten to the
#: repository-relative form the docs use, or dropped when they only mattered
#: on the training machine.
PATH_ARGS_DROP = (
    # machine-local
    "data_root", "output_dir", "resume", "init_from", "wandb_dir",
    # training-only inputs: packs, supervision tables, selection sets. None of
    # them is read at inference, and a released checkpoint must load on a
    # machine that has none of them.
    "data_roots", "dev_root", "exclude_ids", "neg_rate_table", "soft_supervision",
    "pred_embeds", "pred_context", "pair_cooc", "cat_aliases", "canonical_groups",
    "syn_kernel", "heldout_predicates",
)
PRIVATE_MARKERS = ("/home/", "/mimer/", "/proj/", "/tmp/", "/scratch/", "/cephyr/")


def scrub_args(a: dict) -> dict:
    a = dict(a)
    for k in PATH_ARGS_DROP:
        if k in a and a[k]:
            a[k] = None if not isinstance(a[k], list) else []
    for k, v in list(a.items()):
        if isinstance(v, str) and any(m in v for m in PRIVATE_MARKERS):
            # keep the repository-relative tail when there is one
            tail = v.split("RelateAnything_project/")[-1] if "RelateAnything_project/" in v else None
            a[k] = tail
        elif isinstance(v, list) and v and all(isinstance(x, str) for x in v):
            a[k] = [x.split("RelateAnything_project/")[-1] if any(m in x for m in PRIVATE_MARKERS) else x
                    for x in v]
    return a


def backbone_config_for(args: dict, explicit: str | None) -> dict:
    """The backbone's HF config as a dict: from ``--backbone_config``, from the
    local converted directory, or from the hub id it maps to."""
    import json
    from transformers import AutoConfig
    if explicit:
        return json.load(open(explicit))
    name = args.get("backbone_model") or ""
    if name and os.path.isdir(name):
        return AutoConfig.from_pretrained(name).to_dict()
    hub = hub_id_for_backbone(name)
    try:
        return AutoConfig.from_pretrained(hub).to_dict()
    except Exception as e:  # gated repo without login, or offline
        raise SystemExit(
            f"[strip] cannot fetch the backbone config for {name!r} ({hub}): {type(e).__name__}: {e}\n"
            "        Either `huggingface-cli login` after accepting Meta's DINOv3 terms, "
            "or pass --backbone_config <path/to/config.json> from a local converted copy.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"],
                    help="ema = what every reported eval used")
    ap.add_argument("--text_student", default=None,
                    help="student.pt to copy next to --out as text_student.pt "
                         "(default: the path recorded in the checkpoint args)")
    ap.add_argument("--backbone_config", default=None,
                    help="config.json of the backbone, when the hub is not reachable")
    ap.add_argument("--skip_embeddings", action="store_true",
                    help="do not write predicate_embeddings.npz (the training "
                         "vocabulary encoded with the shipped student)")
    ap.add_argument("--skip_verify", action="store_true")
    a = ap.parse_args()
    os.chdir(REPO)

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    raw_args = ckpt["args"]
    raw_args = dict(raw_args if isinstance(raw_args, dict) else vars(raw_args))
    sd = (ckpt.get("ema_model") if a.weights == "ema" else None) or ckpt["model"]

    out = {k: ckpt[k] for k in KEEP if k in ckpt}
    out["backbone_config"] = backbone_config_for(raw_args, a.backbone_config)
    args = scrub_args(raw_args)
    args["backbone_model"] = hub_id_for_backbone(raw_args.get("backbone_model"))
    student_src = a.text_student or raw_args.get("text_student")
    if student_src:
        # The source checkpoint may name the student by an absolute path (a
        # training run) or as the file beside it (an already-stripped one).
        student_src = resolve_student_path(student_src, near=a.checkpoint)
    if student_src and os.path.exists(student_src):
        out_dir = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(student_src, os.path.join(out_dir, "text_student.pt"))
        copied = 0
        for fname in ("tokenizer.json", "vocab.json", "merges.txt",
                      "tokenizer_config.json", "special_tokens_map.json"):
            p = os.path.join(os.path.dirname(student_src), fname)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(out_dir, fname)); copied += 1
        if not copied:
            # The student tokenizes with CLIP's BPE. Save a copy next to it so
            # the released files work with the hub disabled.
            try:
                from transformers import CLIPTokenizer
                CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32").save_pretrained(out_dir)
                print(f"[strip] saved the CLIP tokenizer next to {a.out}")
            except Exception as e:  # offline without a cached copy
                print(f"[strip] warning: could not save the CLIP tokenizer ({type(e).__name__}); "
                      "from_checkpoint will fetch openai/clip-vit-base-patch32 on first use")
        args["text_student"] = "text_student.pt"
    elif student_src:
        print(f"[strip] warning: text student {student_src} not found; "
              "from_checkpoint will need text_student= or embeddings=")
    out["args"] = args
    out["model"] = {k: v.detach().cpu() for k, v in sd.items()
                    if k not in OBSOLETE_KEYS}
    out["stripped_weights"] = a.weights
    out["stripped_from_run"] = os.path.basename(os.path.dirname(os.path.abspath(a.checkpoint)))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    torch.save(out, a.out)
    src = os.path.getsize(a.checkpoint) / 1e9
    dst = os.path.getsize(a.out) / 1e9
    print(f"[strip] {src:.2f} GB -> {dst:.2f} GB  ({a.out})")

    # The training vocabulary, encoded once with the student that ships beside
    # the weights. Encoding 19k strings costs a minute and a half on a CPU, and
    # every user of full_vocabulary=True would pay it; the file is what that
    # encoding produces, so loading it is the same vocabulary, not an
    # approximation of it.
    out_dir = os.path.dirname(os.path.abspath(a.out))
    student_out = os.path.join(out_dir, "text_student.pt")
    if not a.skip_embeddings and out.get("pred_names") and os.path.exists(student_out):
        import numpy as np
        from relsgg.text.student import encode_texts_student
        from relsgg.vocabulary import TRAIN_TEMPLATES
        names = [str(n) for n in out["pred_names"]]
        W = encode_texts_student(names, student_out, templates=TRAIN_TEMPLATES, device="cpu")
        emb = os.path.join(out_dir, "predicate_embeddings.npz")
        np.savez_compressed(emb, names=np.array(names, dtype=object),
                            W=W.cpu().numpy().astype(np.float16))
        print(f"[strip] {len(names)} predicate embeddings -> {emb} "
              f"({os.path.getsize(emb)/1e6:.1f} MB)")

    if not a.skip_verify:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from relsgg.api import RelateAnything
        ra = RelateAnything.from_checkpoint(
            a.out, ["on", "holding"], device="cpu", strict=True)
        n = sum(p.numel() for p in ra.model.parameters())
        print(f"[strip] verify: strict offline from_checkpoint OK ({n/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
