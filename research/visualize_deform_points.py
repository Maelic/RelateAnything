"""Where do the learned deformable points actually land on real images?

The parameter-space tracker (track_deform_points.py) showed the DEFAULT
pattern never leaves the anchor centers and all learned structure is
query-conditional (union anchor row-norm 0.75, others static). This script
shows the query-conditional behavior itself, on real validation pairs:

  per image: one ROW per top-scoring distinct pair, six panels:
               [0] colour image, sub (blue) / obj (orange) boxes and the TOTAL
                   attention-mass map of the read (all heads, all anchors);
               [1-4] one panel per anchor (sub / obj / union / contact): the
                   anchor box, that anchor's mass map and its 32 sampled
                   points (8 heads x 4) as small fixed-size dots. Mass maps
                   are splatted with sigma = half a patch, i.e. the true
                   footprint of a bilinear sample on the stride-16 map, and
                   share ONE colour scale across the four anchor panels so a
                   4 %-share anchor looks faint next to a 48 % one;
               [5] per-head allocation of the softmax mass over the four
                   anchors + the null slots (stacked bars).
               v1 encoded weight as marker AREA (up to ~10^4 pt^2 with 8 heads)
               and head as marker SHAPE — 128 overlapping glyphs per panel that
               hid the boxes and the image. Weight is now a heatmap, position
               a dot, head a bar.
  heads_XXX: for the top pair of the first --n_head_images images, one
               panel per head (its own mass map, points coloured by anchor).
  deform_offset_fields.png (corpus): where the points land RELATIVE to their
               anchor, in half-extent units — attention-weighted 2-D histogram
               per anchor, for all pairs / spatial / semantic / the most
               frequent top-1 predicates; the mean partner-centre position is
               marked so "do contact points move toward the object?" is
               answered by eye.
  corpus stats (over --n_stats images), split spatial vs semantic predicate:
               per-anchor sampled-|offset| distribution (half-extent units),
               fraction of samples OUTSIDE their anchor box (the
               evidence-seeking rate), and attention-weight share per anchor.

Usage:
    python training/visualize_deform_points.py \
        --checkpoint runs/train/..._def4_.../checkpoint_best.pth \
        --data_root runs/packed/psg --split val \
        --n_images 6 --n_stats 200 --out_dir runs/analysis/deform_vis
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data.dataset import RelationDataset                  # noqa: E402
from relsgg.model.geometry import RelGeomEncoder                         # noqa: E402
from relsgg.text.student import encode_texts_student               # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES                           # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt              # noqa: E402

ANCHORS = ["sub", "obj", "union", "contact"]
COLORS = {"sub": "#2a78d6", "obj": "#eb6834",
          "union": "#1baf7a", "contact": "#8a63d2"}
SPATIAL_NAMES = {
    "above", "at the back of", "behind", "below", "beneath", "beside",
    "in front of", "inside", "near", "next to", "on", "on the left of",
    "on the left side of", "on the right of", "on the right side of",
    "on top of", "over", "to the left of", "to the right of", "under",
    "underneath",
}


@torch.no_grad()
def run_image(model, image, boxes, box_count, device, score_mode="sigmoid"):
    """Forward one image with deformable capture; return per-pair records."""
    images_b = image.unsqueeze(0).to(device)
    boxes_b = boxes.unsqueeze(0).to(device)
    counts_b = torch.tensor([box_count], device=device)

    model.deformable_read.capture = True
    out = model(images_b, boxes_b, counts_b, targets=None)
    model.deformable_read.capture = False

    logits = out["logits"][0]
    sub_idx, obj_idx = out["sub_idx"][0], out["obj_idx"][0]
    valid = out["valid_mask"][0]
    pair_logits = out.get("pair_logits")

    lg = logits[valid].float()
    if pair_logits is not None and score_mode == "sigmoid":
        lg = lg + pair_logits[0][valid].float().unsqueeze(-1)
    scores = lg.sigmoid()
    top_score, top_pred = scores.max(-1)

    dr = model.deformable_read
    H, P = dr.heads, dr.n_points
    L = getattr(dr, "n_levels", 1)
    # Multi-level: positions/weights carry a level axis [K',H,A,L,P]. Collapse
    # it for the point-cloud panels (a point's PLACE is level-independent —
    # offsets are in half-extent units) and report the level split separately,
    # since level collapse is the known failure mode of multi-scale deformable
    # attention and must be visible, not inferred.
    pos = dr.last_pos[0][valid]                              # [K',H,A,(L),P,2]
    w = dr.last_w[0][valid].view(-1, H, len(ANCHORS), L, P)  # [K',H,A,L,P]
    level_w = (dr.last_level_w[0][valid] if getattr(dr, "last_level_w", None)
               is not None else None)                        # [K',H,A,L]
    if L > 1:
        pos = pos.view(-1, H, len(ANCHORS), L, P, 2)[:,:,:, 0]  # any level
        w = w.sum(3)                                         # mass at each place
    else:
        pos = pos.view(-1, H, len(ANCHORS), P, 2)
        w = w.view(-1, H, len(ANCHORS), P)
    anch = dr.last_anchors[0][valid]                         # [K',A,4]
    # null mass is per (head, anchor); zero-filled when the arm has no nulls so
    # every downstream consumer sees the same record shape.
    nw = (dr.last_null_w[0][valid] if dr.last_null_w is not None
          else w.new_zeros(w.shape[:-1]))                    # [K',H,A]
    return {
        "pos": pos.cpu(), "w": w.cpu(), "anch": anch.cpu(), "null_w": nw.cpu(),
        "score": top_score.cpu(), "pred": top_pred.cpu(),
        "sub": sub_idx[valid].cpu(), "obj": obj_idx[valid].cpu(),
        "level_w": level_w.cpu() if level_w is not None else None,
    }


def offsets_in_units(rec):
    """Sampled offsets in anchor half-extent units. [K',H,A,P,2]"""
    # anchors are shared by all heads -> broadcast a head axis in.
    centers = rec["anch"][:, None,:, None,:2]
    half = (rec["anch"][:, None,:, None, 2:].clamp_min(0.05) * 0.5)
    return (rec["pos"] - centers) / half


def _entity(ent_names, labels, k):
    if ent_names is None or labels is None:
        return f"#{k}"
    j = int(labels[k]) if k < len(labels) else -1
    return ent_names[j] if 0 <= j < len(ent_names) else f"#{k}"


def _pick_pairs(rec, n_pairs):
    order = rec["score"].argsort(descending=True)
    seen, picked = set(), []
    for i in order.tolist():
        key = (int(rec["sub"][i]), int(rec["obj"][i]))
        if key in seen:
            continue
        seen.add(key)
        picked.append(i)
        if len(picked) >= n_pairs:
            break
    return picked


def splat(pos, w, H, W, sigma):
    """Attention mass -> image-space density. pos [...,2] in [0,1] (x,y),
    w [...] same leading shape. Each sample is a bilinear read of ONE stride-16
    cell neighbourhood, so a Gaussian of sigma = half a patch is its honest
    footprint, not a stylistic blur."""
    from scipy.ndimage import gaussian_filter
    grid = np.zeros((H, W), dtype=np.float64)
    p = pos.reshape(-1, 2).numpy()
    ww = w.reshape(-1).numpy()
    xs = np.clip((p[:, 0] * W).astype(int), 0, W - 1)
    ys = np.clip((p[:, 1] * H).astype(int), 0, H - 1)
    np.add.at(grid, (ys, xs), ww)
    return gaussian_filter(grid, sigma)


def heat_rgba(heat, vmax, cmap="inferno", gamma=0.6, alpha_max=0.85):
    """Colour = density, alpha = density (gamma-compressed) so the image stays
    visible where the read does not look."""
    v = np.clip(heat / max(vmax, 1e-12), 0, 1)
    rgba = plt.get_cmap(cmap)(v)
    rgba[..., 3] = alpha_max * v ** gamma
    return rgba


def _box_patch(ax, cxcywh, W, H, color, lw=2.0, ls="-", alpha=0.95):
    cx, cy, bw, bh = cxcywh
    ax.add_patch(plt.Rectangle(((cx - bw / 2) * W, (cy - bh / 2) * H),
                               bw * W, bh * H, fill=False, ec=color, lw=lw,
                               ls=ls, alpha=alpha, zorder=6))


def _lightened_gray(img_np):
    g = img_np.mean(-1)
    return 0.35 + 0.65 * g


def _clamped_frac(pos):
    p = pos.reshape(-1, 2)
    return float(((p <= 0.0) | (p >= 1.0)).any(-1).float().mean())


def draw_image(img_np, rec, pred_names, n_pairs, path, ent_names=None,
               ent_labels=None, patch_px=16.0):
    picked = _pick_pairs(rec, n_pairs)
    if not picked:
        return
    H, W = img_np.shape[:2]
    Hh, A, P = rec["pos"].shape[1], rec["pos"].shape[2], rec["pos"].shape[3]
    sigma = 0.5 * patch_px
    gray = _lightened_gray(img_np)
    ncol = 6
    fig, axes = plt.subplots(len(picked), ncol,
                             figsize=(3.3 * ncol, 3.55 * len(picked)),
                             squeeze=False,
                             gridspec_kw={"width_ratios": [1.15, 1, 1, 1, 1, 0.9]})
    for r, i in enumerate(picked):
        wgt = rec["w"][i]                                       # [H,A,P]
        nullm = rec["null_w"][i]                                # [H,A]
        share = (wgt.sum(-1).sum(0) / Hh).tolist()              # per anchor
        null_share = float(nullm.sum()) / Hh
        off = offsets_in_units({"anch": rec["anch"][i:i + 1],
                                "pos": rec["pos"][i:i + 1]})[0]  # [H,A,P,2]
        outside = (off.abs().max(-1).values > 1.0).float()      # [H,A,P]
        out_w = float((outside * wgt).sum() / wgt.sum().clamp_min(1e-9))
        clamped = _clamped_frac(rec["pos"][i])
        heats = [splat(rec["pos"][i,:, ai], wgt[:, ai], H, W, sigma)
                 for ai in range(A)]
        total = sum(heats)
        vmax_anchor = max(h.max() for h in heats)

        s_name = _entity(ent_names, ent_labels, int(rec["sub"][i]))
        o_name = _entity(ent_names, ent_labels, int(rec["obj"][i]))
        pred = pred_names[int(rec["pred"][i])]
        sc = float(rec["score"][i])

        # [0] colour image + sub/obj boxes + total mass
        ax = axes[r][0]
        ax.imshow(img_np)
        ax.imshow(heat_rgba(total, total.max()), interpolation="bilinear",
                  zorder=4)
        _box_patch(ax, rec["anch"][i, 0].tolist(), W, H, COLORS["sub"], lw=2.4)
        _box_patch(ax, rec["anch"][i, 1].tolist(), W, H, COLORS["obj"], lw=2.4)
        ax.set_title(f"{s_name} —{pred}→ {o_name}   [{sc:.2f}]",
                     fontsize=10.5, fontweight="bold", loc="left")
        ax.text(0.02, 0.02,
                f"null {null_share:.0%} · mass outside box {out_w:.0%}"
                f" · clamped {clamped:.0%}",
                transform=ax.transAxes, fontsize=8, color="white",
                va="bottom", ha="left", zorder=8,
                bbox=dict(fc="black", alpha=0.55, lw=0, pad=2.5))

        # [1-4] per-anchor panels on a lightened grey image, SHARED scale
        for ai, name in enumerate(ANCHORS):
            ax = axes[r][1 + ai]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
            ax.imshow(heat_rgba(heats[ai], vmax_anchor), zorder=4,
                      interpolation="bilinear")
            _box_patch(ax, rec["anch"][i, ai].tolist(), W, H, COLORS[name],
                       lw=2.2, ls="-" if ai < 2 else "--")
            pts = rec["pos"][i,:, ai].reshape(-1, 2).numpy()
            ax.scatter(pts[:, 0] * W, pts[:, 1] * H, s=16, color=COLORS[name],
                       edgecolors="white", linewidths=0.5, zorder=7, alpha=0.95)
            out_a = float((outside[:, ai] * wgt[:, ai]).sum()
                          / wgt[:, ai].sum().clamp_min(1e-9))
            ax.set_title(f"{name}  ·  {share[ai]:.0%} of mass"
                         f"  ·  {out_a:.0%} outside box", fontsize=9.5,
                         color=COLORS[name])
        # [5] per-head allocation, stacked bars
        ax = axes[r][5]
        alloc = torch.cat([wgt.sum(-1), nullm.sum(-1, keepdim=True)], -1).numpy()  # [H,5]
        left = np.zeros(Hh)
        for ai, name in enumerate(ANCHORS + ["null"]):
            ax.barh(np.arange(Hh), alloc[:, ai], left=left,
                    color=COLORS.get(name, "#9a9a9a"), edgecolor="white",
                    linewidth=0.6, height=0.78)
            left += alloc[:, ai]
        ax.set_yticks(np.arange(Hh))
        ax.set_yticklabels([f"h{h}" for h in range(Hh)], fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.tick_params(labelsize=8)
        ax.invert_yaxis()
        ax.set_title("mass per head", fontsize=9.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for c in range(5):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    fig.suptitle("deformable read — where the attention mass lands (heat = "
                 "softmax weight splatted at half-patch sigma; dots = sampled "
                 "points; anchor panels share one colour scale)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


def draw_heads(img_np, rec, pred_names, path, ent_names=None,
               ent_labels=None, patch_px=16.0):
    """One panel per head for the top pair: does any head look somewhere the
    others do not? (corpus head_divergence says mostly no; this shows it)."""
    picked = _pick_pairs(rec, 1)
    if not picked:
        return
    i = picked[0]
    H, W = img_np.shape[:2]
    Hh, A, P = rec["pos"].shape[1], rec["pos"].shape[2], rec["pos"].shape[3]
    sigma = 0.5 * patch_px
    gray = _lightened_gray(img_np)
    wgt = rec["w"][i]
    heats = [splat(rec["pos"][i, h], wgt[h], H, W, sigma) for h in range(Hh)]
    vmax = max(h.max() for h in heats)
    ncol = 4
    nrow = (Hh + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.6 * nrow),
                             squeeze=False)
    for h in range(nrow * ncol):
        ax = axes[h // ncol][h % ncol]
        ax.set_xticks([]); ax.set_yticks([])
        if h >= Hh:
            ax.axis("off"); continue
        ax.imshow(gray, cmap="gray", vmin=0, vmax=1)
        ax.imshow(heat_rgba(heats[h], vmax), zorder=4, interpolation="bilinear")
        _box_patch(ax, rec["anch"][i, 0].tolist(), W, H, COLORS["sub"], lw=1.8)
        _box_patch(ax, rec["anch"][i, 1].tolist(), W, H, COLORS["obj"], lw=1.8)
        for ai, name in enumerate(ANCHORS):
            pts = rec["pos"][i, h, ai].numpy()
            ax.scatter(pts[:, 0] * W, pts[:, 1] * H, s=22, color=COLORS[name],
                       edgecolors="white", linewidths=0.5, zorder=7)
        sh = wgt[h].sum(-1)
        nl = float(rec["null_w"][i, h].sum())
        ax.set_title(f"head {h}  ·  sub {sh[0]:.0%} obj {sh[1]:.0%} "
                     f"uni {sh[2]:.0%} con {sh[3]:.0%} null {nl:.0%}",
                     fontsize=8.5)
    s_name = _entity(ent_names, ent_labels, int(rec["sub"][i]))
    o_name = _entity(ent_names, ent_labels, int(rec["obj"][i]))
    fig.suptitle(f"per-head reads for {s_name} —"
                 f"{pred_names[int(rec['pred'][i])]}→ {o_name} "
                 f"[{float(rec['score'][i]):.2f}]  (dots coloured by anchor: "
                 "sub blue / obj orange / union green / contact purple; one "
                 "colour scale)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=115, bbox_inches="tight")
    plt.close(fig)


FIELD_R, FIELD_BINS = 3.0, 48


class OffsetFields:
    """Attention-weighted 2-D histograms of sampled offsets in anchor
    half-extent units, keyed by group -> anchor, plus the mean partner-centre
    position in the same frame (obj centre in sub units / sub centre in obj
    units) so the field can be read against where the other box is."""

    def __init__(self):
        self.hist = {}
        self.partner = {}
        self.count = {}

    def add(self, key, off, wgt, anch):
        # off [H,A,P,2], wgt [H,A,P], anch [A,4]
        if key not in self.hist:
            self.hist[key] = np.zeros((len(ANCHORS), FIELD_BINS, FIELD_BINS))
            self.partner[key] = np.zeros((2, 2))
            self.count[key] = 0
        for ai in range(len(ANCHORS)):
            o = off[:, ai].reshape(-1, 2).numpy()
            ww = wgt[:, ai].reshape(-1).numpy()
            hst, _, _ = np.histogram2d(o[:, 1], o[:, 0], bins=FIELD_BINS,
                                       range=[[-FIELD_R, FIELD_R]] * 2,
                                       weights=ww)
            self.hist[key][ai] += hst
        half_s = anch[0, 2:].clamp_min(0.05) * 0.5
        half_o = anch[1, 2:].clamp_min(0.05) * 0.5
        self.partner[key][0] += ((anch[1,:2] - anch[0,:2]) / half_s).numpy()
        self.partner[key][1] += ((anch[0,:2] - anch[1,:2]) / half_o).numpy()
        self.count[key] += 1

    def figure(self, keys, out_path, title):
        keys = [k for k in keys if k in self.hist]
        if not keys:
            return
        fig, axes = plt.subplots(len(keys), len(ANCHORS),
                                 figsize=(3.1 * len(ANCHORS), 3.15 * len(keys)),
                                 squeeze=False)
        ext = (-FIELD_R, FIELD_R, FIELD_R, -FIELD_R)   # image y down
        for r, key in enumerate(keys):
            n = self.count[key]
            for ai, name in enumerate(ANCHORS):
                ax = axes[r][ai]
                h = self.hist[key][ai]
                h = h / max(h.sum(), 1e-12)
                ax.imshow(np.sqrt(h), cmap="inferno", extent=ext,
                          interpolation="nearest")
                ax.add_patch(plt.Rectangle((-1, -1), 2, 2, fill=False,
                                           ec=COLORS[name], lw=1.8,
                                           ls="-" if ai < 2 else "--"))
                ax.axhline(0, color="white", lw=0.4, alpha=0.5)
                ax.axvline(0, color="white", lw=0.4, alpha=0.5)
                if ai < 2:
                    px, py = self.partner[key][ai] / max(n, 1)
                    pc = COLORS["obj"] if ai == 0 else COLORS["sub"]
                    ax.plot(np.clip(px, -FIELD_R, FIELD_R),
                            np.clip(py, -FIELD_R, FIELD_R), marker="x",
                            ms=11, mew=2.4, color=pc)
                ax.set_xlim(-FIELD_R, FIELD_R); ax.set_ylim(FIELD_R, -FIELD_R)
                ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
                ax.tick_params(labelsize=7)
                if r == 0:
                    ax.set_title(f"{name} anchor", color=COLORS[name],
                                 fontsize=10)
                if ai == 0:
                    ax.set_ylabel(f"{key}\n(n={n})", fontsize=9)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)


def stats_figure(stats, out_path):
    panels = ("mag", "outside", "wshare", "nullshare")
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.6))
    x = np.arange(len(ANCHORS))
    for gi, grp in enumerate(("spatial", "semantic")):
        off = 0.38 * gi - 0.19
        hatch = None if gi == 0 else "//"
        for pi, key in enumerate(panels):
            vals = [np.mean(stats[grp][key][a]) if stats[grp][key][a] else 0.0
                    for a in ANCHORS]
            axes[pi].bar(x + off, vals, 0.36,
                         color=[COLORS[a] for a in ANCHORS],
                         hatch=hatch, edgecolor="white", label=grp)
    axes[0].set_title("mean sampled |offset| (half-extent units)")
    axes[0].axhline(1.0, color="#999999", lw=1, ls="--")
    axes[1].set_title("fraction of samples OUTSIDE their anchor")
    axes[2].set_title("attention-weight share per anchor")
    axes[2].axhline(0.25, color="#999999", lw=1, ls="--")
    axes[3].set_title("NULL-slot mass per anchor (v3)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ANCHORS)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("solid = spatial predicate top-1, hatched = semantic", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="runs/packed/psg")
    p.add_argument("--split", default="val")
    p.add_argument("--n_images", type=int, default=6)
    p.add_argument("--n_stats", type=int, default=200)
    p.add_argument("--n_pairs", type=int, default=4)
    p.add_argument("--n_head_images", type=int, default=3,
                   help="per-head panels for the top pair of this many images")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_objects", type=int, default=40)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--weights", default="ema")
    p.add_argument("--out_dir", default="runs/analysis/deform_vis")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    assert hasattr(model, "deformable_read"), "checkpoint has no deformable read"

    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    ds = RelationDataset(root=args.data_root, split=args.split,
                         resolution=args.img_size, max_objects=args.max_objects)
    pred_names = ds.predicate_names
    ent_names = [None] * len(ds.cat_to_idx)
    for n_, i_ in ds.cat_to_idx.items():
        ent_names[i_] = n_
    patch_px = args.img_size / model.backbone_grid if hasattr(
        model, "backbone_grid") else 16.0
    E = encode_texts_student(pred_names, ck_args["text_student"],
                             templates=TRAIN_TEMPLATES, device=device)
    model.vocab_head.set_vocabulary_matrix(pred_names, E)
    model.reparameterize()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    vis_idx = rng.sample(range(len(ds)), min(args.n_images, len(ds)))
    stat_idx = rng.sample(range(len(ds)), min(args.n_stats, len(ds)))

    # ---- qualitative panels (same seed 42 => same images as prior vis runs)
    for rank, idx in enumerate(vis_idx):
        image, boxes, target = ds[idx]
        bc = boxes.shape[0]
        bp = torch.zeros(args.max_objects, 4)
        bp[:bc] = boxes
        rec = run_image(model, image, bp, bc, device)
        img_np = image.permute(1, 2, 0).numpy()
        draw_image(img_np, rec, pred_names, args.n_pairs,
                   os.path.join(args.out_dir, f"img_{rank:03d}_ds{idx}.png"),
                   ent_names=ent_names, ent_labels=target["entity_labels"],
                   patch_px=patch_px)
        if rank < args.n_head_images:
            draw_heads(img_np, rec, pred_names,
                       os.path.join(args.out_dir, f"heads_{rank:03d}_ds{idx}.png"),
                       ent_names=ent_names, ent_labels=target["entity_labels"],
                       patch_px=patch_px)
        print(f"vis [{rank + 1}/{len(vis_idx)}] ds{idx}")

    # ---- corpus stats
    stats = {g: {"mag": {a: [] for a in ANCHORS},
                 "outside": {a: [] for a in ANCHORS},
                 "wshare": {a: [] for a in ANCHORS},
                 # V3: mass parked on the learnable null slots. This is the
                 # honest version of what v1 faked by flinging points
                 # off-image, so it is the number that says whether the fix
                 # took: null share up AND union |offset| down = it worked.
                 "nullshare": {a: [] for a in ANCHORS},
                 # samples pinned to the frame edge by clamp_to_image: they read
                 # the border cell, not the place the offset asked for.
                 "clamped": {a: [] for a in ANCHORS}}
             for g in ("spatial", "semantic")}
    fields = OffsetFields()
    pred_count = {}
    head_share = {g: [] for g in ("spatial", "semantic")}  # [H,A] per pair
    level_share = {g: [] for g in ("spatial", "semantic")}  # [L] per pair
    for n, idx in enumerate(stat_idx):
        image, boxes, target = ds[idx]
        bc = boxes.shape[0]
        bp = torch.zeros(args.max_objects, 4)
        bp[:bc] = boxes
        rec = run_image(model, image, bp, bc, device)
        off = offsets_in_units(rec)                      # [K',H,A,P,2]
        Hh, A, P = off.shape[1], off.shape[2], off.shape[3]
        wgt, nw = rec["w"], rec["null_w"]                # [K',H,A,P], [K',H,A]
        for i in range(off.shape[0]):
            grp = ("spatial" if pred_names[int(rec["pred"][i])] in SPATIAL_NAMES
                   else "semantic")
            head_share[grp].append((wgt[i].sum(-1) + nw[i]).numpy())
            pname = pred_names[int(rec["pred"][i])]
            pred_count[pname] = pred_count.get(pname, 0) + 1
            for key in ("all", grp, pname):
                fields.add(key, off[i], wgt[i], rec["anch"][i])
            if rec["level_w"] is not None:
                # [H,A,L] -> L, averaged over heads (each head's softmax sums
                # to 1) so the share is on a 0-1 scale like wshare.
                level_share[grp].append(
                    rec["level_w"][i].sum(1).mean(0).numpy())
            for ai, aname in enumerate(ANCHORS):
                o = off[i,:, ai]                        # [H,P,2]
                stats[grp]["mag"][aname].append(float(o.norm(dim=-1).mean()))
                stats[grp]["outside"][aname].append(
                    float((o.abs().max(-1).values > 1.0).float().mean()))
                # per-head softmaxes each sum to 1, so divide by H to keep the
                # share on a 0-1 scale comparable with the single-head arm.
                stats[grp]["wshare"][aname].append(float(wgt[i,:, ai].sum()) / Hh)
                stats[grp]["nullshare"][aname].append(float(nw[i,:, ai].sum()) / Hh)
                stats[grp]["clamped"][aname].append(
                    _clamped_frac(rec["pos"][i,:, ai]))
        if (n + 1) % 50 == 0:
            print(f"stats {n + 1}/{len(stat_idx)}")

    stats_figure(stats, os.path.join(args.out_dir, "deform_point_stats.png"))
    top_preds = [k for k, _ in sorted(pred_count.items(), key=lambda kv: -kv[1])
                 if pred_count[k] >= 25][:6]
    fields.figure(["all", "spatial", "semantic"] + top_preds,
                  os.path.join(args.out_dir, "deform_offset_fields.png"),
                  "where the read samples RELATIVE to its anchor (half-extent "
                  "units; box = the anchor; × = mean partner centre: obj in "
                  "the sub frame, sub in the obj frame; sqrt colour scale)")
    summary = {g: {m: {a: (round(float(np.mean(stats[g][m][a])), 4)
                           if stats[g][m][a] else None)
                       for a in ANCHORS}
                   for m in ("mag", "outside", "wshare", "nullshare", "clamped")}
               for g in ("spatial", "semantic")}
    summary["top_predicates"] = {k: pred_count[k] for k in top_preds}
    # per-head anchor allocation: the single-head arm MEASURED 62-75% of all
    # mass on `sub`, which one shared softmax cannot undo. H independent
    # softmaxes can, so report whether the heads actually diverged.
    for g in ("spatial", "semantic"):
        if head_share[g]:
            hs = np.stack(head_share[g]).mean(0)          # [H,A]
            summary[g]["head_anchor_share"] = [
                [round(float(v), 4) for v in row] for row in hs]
            summary[g]["head_divergence"] = round(
                float(np.abs(hs - hs.mean(0, keepdims=True)).mean()), 4)
        if level_share[g]:
            # LEVEL COLLAPSE INSTRUMENT. Uniform would be 1/L per level (times
            # the non-null mass). A share near 1.0 on one level means the level
            # axis bought nothing and the arm should be read as single-level.
            ls = np.stack(level_share[g]).mean(0)         # [L]
            summary[g]["level_share"] = [round(float(v), 4) for v in ls]
            summary[g]["level_collapse"] = round(float(ls.max() / max(ls.sum(), 1e-9)), 4)
    json.dump(summary, open(os.path.join(args.out_dir,
                                         "deform_point_stats.json"), "w"),
              indent=2)
    print(json.dumps(summary, indent=1))
    print("done ->", args.out_dir)


if __name__ == "__main__":
    main()
