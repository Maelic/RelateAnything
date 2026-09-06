"""Does the RelationTransformer's cross-attention scene access (beyond what
SoftSpatialPool already pools from inside the sub/obj/union/contact boxes)
actually change predictions? A 3-way extension of
training/probe_counterfactual_grounding.py's TDE-style probe.

SSP is structurally LOCAL: it only ever pools features from inside the
pair's own boxes. The cross-attention layer is the only mechanism with
access to the scene OUTSIDE those boxes (occlusion, supporting surfaces,
broader layout) — so whether it's worth fixing / redesigning (deformable
attention, DAB-DETR-style box-conditioned modulation,...) hinges on
whether that outside-the-box context is ever actually used. This probe
tests it directly, per pair, three conditions:

  factual     — real image, real boxes                          -> P_f
  local-only  — real pixels ONLY inside an expanded union box,
                everything else flat gray                        -> P_loc
  full-blank  — everything flat gray (from the earlier probe)    -> P_blank

If P_loc ~= P_f (local context is enough, matches factual almost as well as
having the whole image), the broader scene isn't earning its keep for this
model/task and investing in a fancier context mechanism has low expected
payoff. If P_loc drifts toward P_blank (local context is NOT enough), the
model is using scene context beyond the boxes and that pathway is worth
protecting/improving.

Same sampler-freezing trick as probe_counterfactual_grounding.py: the
factual pass's (sub_idx, obj_idx, valid_mask, pred_labels, pair_logits) are
captured once per image and forced for every condition, so all three
conditions score the literal same relation instances.

Usage
-----
    python training/probe_context_window.py \\
        --checkpoints runs/train/A/checkpoint_best.pth... \\
        --labels A... \\
        --data_root runs/packed/psg --split val --spatial_root runs/packed/megasg_clean \\
        --n_images 30 --n_pairs 4 --margin 0.15 \\
        --out runs/analysis/context_window_probe.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from relsgg.data.dataset import RelationDataset          # noqa: E402
from relsgg.model.geometry import RelGeomEncoder                  # noqa: E402
from relsgg.model.pooling import union_box                             # noqa: E402
from relsgg.text.student import encode_texts_student        # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES                    # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt       # noqa: E402
from benchmark.eval_zeroshot import spatial_predicate_names  # noqa: E402

EPS = 1e-8
_GEO_WEIGHT_KEYS = ("geo_encoder.mlp.0.weight", "sampler.geo_scorer.0.weight")


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return (p * ((p + EPS).log() - (q + EPS).log())).sum(-1)


def _local_image(image: torch.Tensor, box_xyxy_px: tuple) -> torch.Tensor:
    """Flat gray image with real pixels restored inside box_xyxy_px."""
    x0, y0, x1, y1 = box_xyxy_px
    out = torch.full_like(image, 0.5)
    out[:, y0:y1, x0:x1] = image[:, y0:y1, x0:x1]
    return out


@torch.no_grad()
def probe_checkpoint(
    checkpoint: str,
    ds: RelationDataset,
    pred_names: list[str],
    spatial_names: set,
    device: torch.device,
    n_images: int,
    n_pairs: int,
    margin: float,
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

    rows = {"top1_agree_local": [], "score_drop_local": [], "kl_local": [],
            "top1_agree_blank": [], "score_drop_blank": [], "kl_blank": [],
            "is_spatial": []}

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
        _, H, W = image.shape
        boxes_padded = torch.zeros(max_objects, 4, dtype=torch.float32)
        boxes_padded[:box_count] = boxes

        images_b = image.unsqueeze(0).to(device)
        boxes_b = boxes_padded.unsqueeze(0).to(device)
        counts_b = torch.tensor([box_count], device=device)

        out_f = model(images_b, boxes_b, counts_b, targets=None)
        logits_f = out_f["logits"]
        sub_idx = out_f["sub_idx"]; obj_idx = out_f["obj_idx"]
        valid = out_f["valid_mask"]; pred_labels = out_f["pred_labels"]
        pair_logits = out_f.get("pair_logits")

        scores_f = _decode(logits_f[0], pair_logits, valid[0])
        top_score_f, top_pred_f = scores_f.max(-1)
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

        boxes_np = boxes_b[0].cpu().numpy()
        image_blank = torch.full_like(image, 0.5)

        for o in picked:
            s_box = int(sub_idx[0, valid[0]][o]); o_box = int(obj_idx[0, valid[0]][o])
            p_id = int(top_pred_f[o])

            u = union_box(
                torch.from_numpy(boxes_np[s_box:s_box + 1]).unsqueeze(0),
                torch.from_numpy(boxes_np[o_box:o_box + 1]).unsqueeze(0),
)[0, 0]
            cx, cy, w, h = u.tolist()
            hw, hh = w / 2 * (1 + margin), h / 2 * (1 + margin)
            x0 = max(0, int((cx - hw) * W)); x1 = min(W, int((cx + hw) * W) + 1)
            y0 = max(0, int((cy - hh) * H)); y1 = min(H, int((cy + hh) * H) + 1)
            x1 = max(x1, x0 + 1); y1 = max(y1, y0 + 1)

            image_local = _local_image(image, (x0, y0, x1, y1)).unsqueeze(0).to(device)

            model.sampler.forward = _fixed_sampler
            out_loc = model(image_local, boxes_b, counts_b, targets=None)
            out_bl = model(image_blank.unsqueeze(0).to(device), boxes_b, counts_b, targets=None)
            model.sampler.forward = orig_sampler_forward

            scores_loc = _decode(out_loc["logits"][0], pair_logits, valid[0])
            scores_bl = _decode(out_bl["logits"][0], pair_logits, valid[0])
            p_f_full = logits_f[0][valid[0]].float().softmax(-1)[o]
            p_loc_full = out_loc["logits"][0][valid[0]].float().softmax(-1)[o]
            p_bl_full = out_bl["logits"][0][valid[0]].float().softmax(-1)[o]

            argmax_loc = int(scores_loc[o].argmax())
            argmax_bl = int(scores_bl[o].argmax())
            rows["top1_agree_local"].append(float(argmax_loc == p_id))
            rows["top1_agree_blank"].append(float(argmax_bl == p_id))
            rows["score_drop_local"].append(
                float(top_score_f[o].item() - scores_loc[o, p_id].item()))
            rows["score_drop_blank"].append(
                float(top_score_f[o].item() - scores_bl[o, p_id].item()))
            rows["kl_local"].append(float(_kl(p_f_full.unsqueeze(0), p_loc_full.unsqueeze(0)).item()))
            rows["kl_blank"].append(float(_kl(p_f_full.unsqueeze(0), p_bl_full.unsqueeze(0)).item()))
            rows["is_spatial"].append(bool(is_spatial[p_id].item()))

    arr = {k: np.array(v) for k, v in rows.items()}
    spa = arr["is_spatial"]

    def _stats(mask):
        n = int(mask.sum())
        if n == 0:
            return {"n": 0}
        return {"n": n,
                "top1_agree_local": float(arr["top1_agree_local"][mask].mean()),
                "top1_agree_blank": float(arr["top1_agree_blank"][mask].mean()),
                "score_drop_local": float(arr["score_drop_local"][mask].mean()),
                "score_drop_blank": float(arr["score_drop_blank"][mask].mean()),
                "kl_local": float(arr["kl_local"][mask].mean()),
                "kl_blank": float(arr["kl_blank"][mask].mean())}

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
    p.add_argument("--spatial_root", default="")
    p.add_argument("--split", default="val")
    p.add_argument("--n_images", type=int, default=30)
    p.add_argument("--n_pairs", type=int, default=4)
    p.add_argument("--margin", type=float, default=0.15,
                   help="Fractional padding added around the union box for "
                        "the local-only condition (0.15 = 15% larger on each side).")
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
    print(f"{len(spatial_names)}/{len(pred_names)} predicates flagged spatial")

    results = {}
    for ckpt_path, label in zip(args.checkpoints, args.labels):
        print(f"==== {label}: {ckpt_path} ====")
        summary = probe_checkpoint(
            ckpt_path, ds, pred_names, spatial_names, device,
            args.n_images, args.n_pairs, args.margin, args.score_mode,
            args.seed, args.weights)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        results[label] = summary

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")

    labels = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(labels))
    width = 0.35
    for ax, metric in zip(axes, ["top1_agree", "score_drop", "kl"]):
        loc = [results[l]["overall"].get(f"{metric}_local", 0.0) for l in labels]
        bl = [results[l]["overall"].get(f"{metric}_blank", 0.0) for l in labels]
        ax.bar(x - width / 2, loc, width, label="local-only (union box + margin)")
        ax.bar(x + width / 2, bl, width, label="full-blank")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    png_path = os.path.splitext(args.out)[0] + ".png"
    fig.savefig(png_path, dpi=130)
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
