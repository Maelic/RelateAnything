"""Precompute a bank of predicate embeddings for the runtime-swappable ONNX head.

`deploy/export_onnx.py --vocab-mode input` leaves W [V,768] and alpha [V] as
graph INPUTS, so the predicate vocabulary can change per frame. Encoding new
predicate strings normally needs the distilled student text encoder (torch) —
which we do NOT want to ship to a laptop. So we encode a generous bank ONCE,
here, and the laptop just slices rows out of it by name. Result: a genuinely
dynamic vocabulary with nothing but onnxruntime + numpy on the target.

PER-CHECKPOINT, NOT GLOBAL. Three things in this bank belong to one specific
checkpoint and silently corrupt inference if mixed across models:
  * W       — must be encoded by the SAME text student the checkpoint trained
              with. The student is therefore resolved from the checkpoint's own
              args (like relsgg/api.py), never from a hard-coded default: the
              old default silently produced v1-space embeddings for v2-space
              models, and cosine against a wrong-space W is garbage, not error.
  * alpha   — recomputed through THIS checkpoint's gate MLP for these rows
              (the checkpoint's stored alpha indexes the training vocabulary).
  * thr     — per-predicate operating points measured on THIS checkpoint's
              scores (deploy/calibrate_thresholds.py). Score scales are
              specific to a checkpoint, so thresholds never transfer.

Also bakes the two-graph type vector (is_spatial/type_source): corpus flag when
the string is known, gate alpha>=0.5 for novel strings — and caches the corpus
type map to deploy/corpus_type_map.json so runtime never needs the packs.

    python deploy/build_predicate_bank.py \
        --checkpoint runs/train/<run>/checkpoint_best.pth \
        --out deploy/dist/<model_id>/predicate_bank.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".hf_cache"))

from deploy.vocab import PREDICATE_VOCAB  # noqa: E402

CORPUS_MAP_CACHE = os.path.join(REPO, "deploy", "corpus_type_map.json")


def _load_ckpt_args(path: str) -> tuple[dict, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    a = ck.get("args") or {}
    a = dict(a if isinstance(a, dict) else vars(a))
    sd = ck.get("ema_model") or ck["model"]
    return a, sd


def _corpus_map(pack: str, split: str) -> dict:
    """Corpus type map, cached to a json so it survives pack evacuation."""
    if os.path.exists(CORPUS_MAP_CACHE):
        return json.load(open(CORPUS_MAP_CACHE))
    from relsgg.decompose import corpus_spatial_map
    m = corpus_spatial_map(pack, split)
    json.dump(m, open(CORPUS_MAP_CACHE, "w"), separators=(",", ":"))
    print(f"[bank] corpus type map cached -> {CORPUS_MAP_CACHE} ({len(m)} strings)")
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--student", default=None,
                    help="text student path. Default: resolved from the "
                         "checkpoint's own args (the only safe source).")
    ap.add_argument("--recall", default=None,
                    help="per_class_recall.json. Default: <run_dir>/per_class_recall.json")
    ap.add_argument("--thresholds", default=None,
                    help="deploy_thresholds.json from calibrate_thresholds.py. "
                         "Default: runs/analysis/<run>/deploy_thresholds.json; "
                         "missing values become NaN (and a warning), never 0.")
    ap.add_argument("--corpus_pack", default="runs/packed/megasg")
    ap.add_argument("--corpus_split", default="val")
    ap.add_argument("--min-gt", type=int, default=30,
                    help="include every trained predicate with at least this much GT support")
    ap.add_argument("--out", default="deploy/dist/predicate_bank.npz")
    ap.add_argument("--skip_verify", action="store_true",
                    help="skip the cosine cross-check against the training "
                         "pred_embeds pack (needed once packs are evacuated)")
    args = ap.parse_args()

    os.chdir(REPO)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    run_name = os.path.basename(run_dir)

    ck_args, sd = _load_ckpt_args(args.checkpoint)

    # --- text student: from the checkpoint, or fail --------------------------
    student = args.student or ck_args.get("text_student")
    if not student or not os.path.exists(student):
        raise SystemExit(
            f"[bank] no usable text student (checkpoint says "
            f"{ck_args.get('text_student')!r}). W must be encoded in the "
            "checkpoint's own text space — pass --student explicitly only if "
            "you know it is the SAME file the model trained with.")
    print(f"[bank] checkpoint {run_name}  student {student}")

    # Template ensemble must match training — import the single source of
    # truth rather than keeping a copy that can drift.
    from relsgg.vocabulary import TRAIN_TEMPLATES as PHOTO_TEMPLATES

    # --- which predicates -----------------------------------------------------
    recall_path = args.recall or os.path.join(run_dir, "per_class_recall.json")
    recall = {}
    names: list[str] = list(PREDICATE_VOCAB)      # curated set always included
    if os.path.exists(recall_path):
        d = json.load(open(recall_path))
        for c in d["classes"]:
            recall[c["name"]] = (c["recall"], c["gt"])
            if c["gt"] >= args.min_gt:
                names.append(c["name"])
    else:
        print(f"[bank] WARNING no per-class recall at {recall_path} — "
              "bank restricted to the curated set")
    seen, ordered = set(), []
    for n in names:
        if n not in seen:
            seen.add(n); ordered.append(n)
    print(f"[bank] {len(ordered)} predicates "
          f"({len(PREDICATE_VOCAB)} curated + trained with gt>={args.min_gt})")

    # --- encode with the student ---------------------------------------------
    from relsgg.text.student import encode_texts_student
    W = encode_texts_student(ordered, student, templates=PHOTO_TEMPLATES,
                             device="cpu").float()
    W = torch.nn.functional.normalize(W, dim=-1)
    print(f"[bank] W {tuple(W.shape)}")

    # --- verify the space against the training pack --------------------------
    # Shared strings must land where training put them (cos > 0.999). This is
    # the tripwire for the exact bug the old default caused: a bank silently
    # encoded in the WRONG student's space.
    if not args.skip_verify:
        pe_path = ck_args.get("pred_embeds", "")
        if pe_path and os.path.exists(pe_path):
            pk = np.load(pe_path, allow_pickle=True)
            pack_names = [str(x) for x in pk["predicates"]]
            pack_W = torch.from_numpy(pk["embeddings"].astype(np.float32))
            pack_W = torch.nn.functional.normalize(pack_W, dim=-1)
            idx = {n: i for i, n in enumerate(pack_names)}
            shared = [(j, idx[n]) for j, n in enumerate(ordered) if n in idx]
            if shared:
                a_ = torch.stack([W[j] for j, _ in shared])
                b_ = torch.stack([pack_W[i] for _, i in shared])
                cos = (a_ * b_).sum(-1)
                print(f"[bank] space check: {len(shared)} shared strings, "
                      f"min cos {float(cos.min()):.5f}")
                if float(cos.min()) < 0.999:
                    raise SystemExit(
                        "[bank] encoded W does NOT match the training text "
                        "space — wrong student or wrong templates. Refusing "
                        "to write a corrupt bank.")
        else:
            print(f"[bank] WARNING cannot verify space ({pe_path!r} missing)")

    # --- alpha from the TRAINED gate MLP -------------------------------------
    # alpha routes each predicate between the semantic and spatial experts and
    # is a function of its text embedding, so it must be recomputed for these
    # rows — reusing the checkpoint's alpha (which indexes the training
    # vocabulary) would misalign every row.
    gm = torch.nn.Sequential(torch.nn.Linear(W.shape[1], 128), torch.nn.GELU(),
                             torch.nn.Linear(128, 1))
    with torch.no_grad():
        gm[0].weight.copy_(sd["vocab_head.gate_mlp.0.weight"])
        gm[0].bias.copy_(sd["vocab_head.gate_mlp.0.bias"])
        gm[2].weight.copy_(sd["vocab_head.gate_mlp.2.weight"])
        gm[2].bias.copy_(sd["vocab_head.gate_mlp.2.bias"])
        alpha = torch.sigmoid(gm(W).squeeze(-1))
    print(f"[bank] alpha {tuple(alpha.shape)}  "
          f"[{float(alpha.min()):.3f}, {float(alpha.max()):.3f}] "
          f"(high = routed to the spatial expert)")

    # --- two-graph type vector (corpus flag, gate fallback) ------------------
    from relsgg.decompose import type_vector
    cmap = _corpus_map(args.corpus_pack, args.corpus_split)
    is_spatial, type_source = type_vector(ordered, corpus_map=cmap,
                                          alpha=alpha.numpy())
    n_src = {s: int((type_source == s).sum()) for s in ("corpus", "gate", "default")}
    print(f"[bank] type vector: {int(is_spatial.sum())} spatial "
          f"(sources {n_src})")

    # --- per-predicate calibrated thresholds ---------------------------------
    thr_path = args.thresholds or os.path.join("runs/analysis", run_name,
                                               "deploy_thresholds.json")
    thr = np.full(len(ordered), np.nan, np.float32)
    best_f1 = np.full(len(ordered), np.nan, np.float32)
    if os.path.exists(thr_path):
        td = json.load(open(thr_path))
        rows = {r["name"]: r for r in td["predicates"]}
        for i, n in enumerate(ordered):
            if n in rows:
                thr[i] = rows[n]["best_f1_thr"]
                best_f1[i] = rows[n]["best_f1"]
        print(f"[bank] thresholds: {int(np.isfinite(thr).sum())}/{len(ordered)} "
              f"calibrated from {thr_path}")
    else:
        print(f"[bank] WARNING no calibration at {thr_path} — thresholds are "
              "NaN. Run deploy/calibrate_thresholds.py; per-checkpoint "
              "calibration is MANDATORY for release banks (score scales do "
              "not transfer between checkpoints).")

    np.savez_compressed(
        args.out,
        names=np.array(ordered, dtype=object),
        W=W.numpy().astype(np.float32),
        alpha=alpha.numpy().astype(np.float32),
        thr=thr,
        best_f1=best_f1,
        is_spatial=is_spatial.astype(np.uint8),
        type_source=np.array(list(type_source), dtype=object),
        recall=np.array([recall.get(n, (float("nan"), 0))[0] for n in ordered], np.float32),
        gt=np.array([recall.get(n, (0, 0))[1] for n in ordered], np.int64),
        default=np.array(list(PREDICATE_VOCAB), dtype=object),
        # provenance
        checkpoint=np.array(run_name),
        student=np.array(student),
)
    print(f"[bank] wrote {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
