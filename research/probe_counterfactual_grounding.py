"""Cheap TDE-style counterfactual grounding probe.

Tang et al., "Unbiased Scene Graph Generation from Biased Training" (CVPR
2020) measure geometric/frequency shortcut reliance via a Total Direct
Effect: compare the model's real prediction against a counterfactual one
where the causal path through genuine visual content is cut, holding
everything else (here: which boxes, which pair) fixed. This script runs
that comparison directly, at inference, no retraining:

  factual        — real image + real boxes -> P_f
  counterfactual — boxes UNCHANGED, image content replaced with a flat gray
                   "no evidence" image -> P_cf

For a FIXED (subject, object) pair, if the model's predicate distribution
barely moves between P_f and P_cf, it didn't need the image for that
relation — it decoded box geometry (+ whatever appearance prior box size/
position implies), not visual evidence.

Getting a fair comparison requires the SAME pair to be scored in both
conditions. RelatednessPairSampler picks pairs using obj_feats, which are a
function of the (real or blanked) backbone features — so naively forwarding
the blanked image would let the sampler pick a DIFFERENT set of pairs,
confounding "the pair changed" with "the prediction changed". This script
runs the factual pass first, then monkey-patches model.sampler.forward for
the counterfactual pass to return the exact pairs (and relatedness/
pair-existence scores) the factual pass produced — so the counterfactual
score differs from the factual one ONLY through the predicate-classification
logits' response to the image, isolating exactly the mechanism this
conversation has been probing (see training/visualize_relation_attention.py,
training/analyze_relation_attention.py).

Reports, split by whether a predicate is majority-geometric in its training
data (relsgg's own `spatial_predicate_names` heuristic, reused from
benchmark/eval_zeroshot.py) vs not: geometric predicates ("above", "left of")
SHOULD show high factual/counterfactual agreement — that's correct
behavior, not a shortcut. Semantic/contact predicates ("riding", "eating",
"holding") showing high agreement IS the shortcut signature: the model
claims to know a visually-grounded relation without ever having looked.

Usage
-----
    python training/probe_counterfactual_grounding.py \\
        --checkpoints runs/train/A/checkpoint_best.pth runs/train/B/checkpoint_best.pth \\
        --labels A B \\
        --data_root runs/packed/psg --split val \\
        --n_images 60 --n_pairs 6 \\
        --out runs/analysis/counterfactual_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from relsgg.data.dataset import RelationDataset          # noqa: E402
from relsgg.model.geometry import RelGeomEncoder                  # noqa: E402
from relsgg.text.student import encode_texts_student        # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES                    # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt       # noqa: E402
from benchmark.eval_zeroshot import spatial_predicate_names  # noqa: E402

EPS = 1e-8
_GEO_WEIGHT_KEYS = ("geo_encoder.mlp.0.weight", "sampler.geo_scorer.0.weight")


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) over the last dim, both already softmax probabilities."""
    return (p * ((p + EPS).log() - (q + EPS).log())).sum(-1)


@torch.no_grad()
def probe_checkpoint(
    checkpoint: str,
    ds: RelationDataset,
    pred_names: list[str],
    spatial_names: set,
    device: torch.device,
    n_images: int,
    n_pairs: int,
    score_mode: str,
    seed: int,
    weights: str,
) -> dict:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, weights).to(device).eval()

    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    text_student = ck_args.get("text_student") or ""
    if text_student:
        E = encode_texts_student(pred_names, text_student,
                                 templates=TRAIN_TEMPLATES, device=device)
        model.vocab_head.set_vocabulary_matrix(pred_names, E)
    else:
        raise SystemExit(
            "this checkpoint names no text student. The vocabulary has to be "
            "encoded by the encoder the head was trained against; pass "
            "--text_student, or use a released model, which ships its own.")
    model.reparameterize()

    is_spatial = torch.tensor([n in spatial_names for n in pred_names], device=device)

    orig_sampler_forward = model.sampler.forward

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_images, len(ds)))

    rows = {"top1_agree": [], "score_drop": [], "kl": [], "is_spatial": []}

    def _decode(logits, pair_logits, valid):
        lg = logits[valid].float()
        if pair_logits is not None:
            pl = pair_logits[0][valid].float()
            if score_mode == "sigmoid":
                lg = lg + pl.unsqueeze(-1)
        scores = lg.sigmoid() if score_mode == "sigmoid" else lg.softmax(-1)
        if score_mode == "softmax" and pair_logits is not None:
            scores = scores * pl.sigmoid().unsqueeze(-1)
        return scores

    for idx in indices:
        image, boxes, target = ds[idx]
        box_count = boxes.shape[0]
        max_objects = ds.max_objects
        boxes_padded = torch.zeros(max_objects, 4, dtype=torch.float32)
        boxes_padded[:box_count] = boxes

        images_b = image.unsqueeze(0).to(device)
        boxes_b = boxes_padded.unsqueeze(0).to(device)
        counts_b = torch.tensor([box_count], device=device)

        # ---- factual pass ----
        out_f = model(images_b, boxes_b, counts_b, targets=None)
        logits_f = out_f["logits"]
        sub_idx = out_f["sub_idx"]; obj_idx = out_f["obj_idx"]
        valid = out_f["valid_mask"]; pred_labels = out_f["pred_labels"]
        pair_logits = out_f.get("pair_logits")

        scores_f = _decode(logits_f[0], pair_logits, valid[0])
        top_score_f, top_pred_f = scores_f.max(-1)
        valid_idx = valid[0].nonzero(as_tuple=True)[0]
        order = top_score_f.argsort(descending=True)

        seen_pairs = set()
        picked = []
        for o in order.tolist():
            s = int(sub_idx[0, valid[0]][o]); ob = int(obj_idx[0, valid[0]][o])
            if (s, ob) in seen_pairs:
                continue
            seen_pairs.add((s, ob))
            picked.append(o)
            if len(picked) >= n_pairs:
                break

        # ---- counterfactual pass: same pairs, blanked image ----
        geo_loss_zero = torch.zeros((), device=device)
        # The frozen tuple MUST match relsgg/model.py's own arity contract,
        # which branches on len(sampler_out) == 7 (RelatednessPairSampler:
        # geo_loss, rel_loss, rel_logits) vs 5 (CascadePairSampler). This probe
        # hardcoded a 6-tuple written before `rel_loss` was added to the
        # relatedness return, so it matched NEITHER branch and died with
        # "too many values to unpack (expected 5)" — the whole 4-probe suite was
        # silently unrunnable on the current lineage, which uses
        # sampler_type=relatedness. Built from pair_logits rather than a literal
        # arity so it follows whichever sampler the checkpoint actually used.
        if pair_logits is not None:
            fixed = (sub_idx, obj_idx, valid, pred_labels,
                     geo_loss_zero, geo_loss_zero, pair_logits)
        else:
            fixed = (sub_idx, obj_idx, valid, pred_labels, geo_loss_zero)

        def _fixed_sampler(*_a, **_kw):
            return fixed

        model.sampler.forward = _fixed_sampler
        images_blank = torch.full_like(images_b, 0.5)
        out_cf = model(images_blank, boxes_b, counts_b, targets=None)
        model.sampler.forward = orig_sampler_forward

        logits_cf = out_cf["logits"]
        scores_cf = _decode(logits_cf[0], pair_logits, valid[0])

        p_f = logits_f[0][valid[0]].float().softmax(-1)
        p_cf = logits_cf[0][valid[0]].float().softmax(-1)
        kl_vals = _kl(p_f, p_cf)  # [K']

        for o in picked:
            p_id = int(top_pred_f[o])
            argmax_cf = int(scores_cf[o].argmax())
            rows["top1_agree"].append(float(argmax_cf == p_id))
            rows["score_drop"].append(
                float(top_score_f[o].item() - scores_cf[o, p_id].item()))
            rows["kl"].append(float(kl_vals[o].item()))
            rows["is_spatial"].append(bool(is_spatial[p_id].item()))

    arr = {k: np.array(v) for k, v in rows.items()}
    spa = arr["is_spatial"]

    def _stats(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        return {
            "n": n,
            "top1_agree": float(arr["top1_agree"][mask].mean()),
            "score_drop": float(arr["score_drop"][mask].mean()),
            "kl": float(arr["kl"][mask].mean()),
        }

    return {
        "overall": _stats(np.ones_like(spa, dtype=bool)),
        "spatial_predicates": _stats(spa),
        "semantic_predicates": _stats(~spa),
        "_ckpt_args": {"backbone_type": ck_args.get("backbone_type"),
                       "epoch": ckpt.get("epoch")},
    }


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--spatial_root", default="",
                   help="Packed root to source the geometric/semantic predicate "
                        "split from (needs real source flags in train/rels.npy — "
                        "VG150/PSG/GQA packs are all-zero there; megasg/megasg_clean "
                        "carry them). Defaults to --data_root.")
    p.add_argument("--split", default="val")
    p.add_argument("--n_images", type=int, default=60)
    p.add_argument("--n_pairs", type=int, default=6)
    p.add_argument("--max_objects", type=int, default=40)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--score_mode", default="sigmoid", choices=["sigmoid", "softmax"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    assert len(args.checkpoints) == len(args.labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = RelationDataset(root=args.data_root, split=args.split,
                        resolution=args.img_size, max_objects=args.max_objects)
    pred_names = ds.predicate_names
    spatial_names = spatial_predicate_names(args.spatial_root or args.data_root)
    print(f"{len(spatial_names)}/{len(pred_names)} predicates flagged spatial "
         f"(majority-geometric source): {sorted(spatial_names)}")

    results = {}
    for ckpt_path, label in zip(args.checkpoints, args.labels):
        print(f"==== {label}: {ckpt_path} ====")
        summary = probe_checkpoint(
            ckpt_path, ds, pred_names, spatial_names, device,
            args.n_images, args.n_pairs, args.score_mode, args.seed, args.weights)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        results[label] = summary

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")

    labels = list(results.keys())
    groups = ["overall", "spatial_predicates", "semantic_predicates"]
    metrics = ["top1_agree", "score_drop", "kl"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.5))
    x = np.arange(len(labels))
    width = 0.25
    for ax, m in zip(axes, metrics):
        for i, g in enumerate(groups):
            vals = [results[l][g].get(m, 0.0) for l in labels]
            ax.bar(x + (i - 1) * width, vals, width, label=g)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(m, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    png_path = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
