"""Patch-similarity maps (DINOv3-paper style) and a relation-vs-class similarity statistic.

QUESTION. DINOv3's dense features are famous for "same object, same colour": a query patch on a
dog lights up every dog patch. After relation fine-tuning the head reads pairs, so the features
may have moved toward "same RELATION": a patch on the rider should now resemble the horse it
rides more than the other riders in the image. This script measures that, per layer, against
the pretrained backbone, and draws the query→similarity maps the paper shows.

STATISTIC. For every GT relation (s, o) in an image, take the patches whose centres fall in the
subject box only (not in the object box), L2-normalise, and compute the mean cosine to the
patches of five target regions:
    partner       the object box o (minus s)                          — relation
    same_class    other instances of s's category (not s, not o, not overlapping s) — class
    unrelated     boxes of another category with NO GT relation to s in either direction
    unrelated_nn  the single unrelated box nearest to s (proximity control)
    background    patches outside every box
and, for the *_class/relation contrast, the fraction of relations where cos(partner) >
cos(same_class) (only when both exist). Everything is reported per model, per layer
(hidden states + the fused map the head reads), and split by rel_flags bit0 (spatial vs
semantic) and by whether s and o touch (IoU > 0), because touching pairs share adjacent
patches trivially.

MAPS. For --n_viz images: query = centre patch of s, of o, and one background patch; one row per
query, columns = image with boxes | pretrained L | model L (each fine-tuned arm) | fused (first
arm). Fixed colour scale [0, 1] on cosine so panels are comparable across models.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relsgg.data.dataset import RelationDataset  # noqa: E402
from relsgg.model.backbone import Backbone  # noqa: E402
from research.analyze_backbone_shift import hidden_stack  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

TARGETS = ["self", "partner", "same_class", "unrelated", "unrelated_nn", "background"]
MAP_NORM = "stretch"
C_SUB, C_OBJ = "#2a78d6", "#eb6834"


def load_models(ckpts, labels, weights, device):
    ck0 = torch.load(ckpts[0], map_location="cpu", weights_only=False)
    a0 = ck0["args"] if isinstance(ck0["args"], dict) else vars(ck0["args"])
    kw = dict(backbone_type=a0.get("backbone_type", "dinov3"), model_name=a0.get("backbone_model") or None,
              pretrained=True)
    models = {"pretrained": Backbone(**kw, norm_taps=False).to(device).eval()}
    for p, lab in zip(ckpts, labels):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        key = "ema_model" if weights == "ema" and "ema_model" in ck else "model"
        sd = {k[len("backbone."):]: v for k, v in ck[key].items() if k.startswith("backbone.")}
        a = ck["args"] if isinstance(ck["args"], dict) else vars(ck["args"])
        bb = Backbone(**kw, norm_taps=bool(a.get("norm_taps", False))).to(device).eval()
        _, unexpected = bb.load_state_dict(sd, strict=False)
        assert not unexpected, unexpected[:5]
        models[lab] = bb
    return models


def box_patch_masks(boxes, hp):
    """[N, hp*hp] bool: patch centre inside box (cxcywh normalised)."""
    ys, xs = np.meshgrid((np.arange(hp) + 0.5) / hp, (np.arange(hp) + 0.5) / hp, indexing="ij")
    xs, ys = xs.reshape(-1), ys.reshape(-1)
    m = np.zeros((len(boxes), hp * hp), bool)
    for i, (cx, cy, w, h) in enumerate(boxes):
        m[i] = (xs >= cx - w / 2) & (xs <= cx + w / 2) & (ys >= cy - h / 2) & (ys <= cy + h / 2)
    return m


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx0, by0, bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih; return inter / max(a[2] * a[3] + b[2] * b[3] - inter, 1e-9)


CENTER = False


def _norm(x):
    """L2-normalise patch tokens; with --center, subtract the per-image mean token first (removes the
    shared anisotropy direction that puts every pretrained last-layer patch at cos ~0.8)."""
    if CENTER:
        x = x - x.mean(0, keepdim=True)
    return F.normalize(x, dim=-1)


def feats_for(models, image, n_patch, layers):
    """{model: {layer_key: [P, d] normalised}}"""
    out = {}
    with torch.no_grad():
        for name, bb in models.items():
            hs, _, fused, _, _, _ = hidden_stack(bb, image[None], n_patch)
            d = {str(l): _norm(hs[l][0].float()) for l in layers}
            d["fused"] = _norm(fused[0].float())
            out[name] = d
    return out


def image_stats(feats, boxes, cats, rels, hp, acc):
    masks = box_patch_masks(boxes, hp); any_box = masks.any(0)
    N = len(boxes)
    related = np.zeros((N, N), bool)
    for s, o in rels[:,:2]:
        related[s, o] = related[o, s] = True
    for s, o, p, fl in rels[:,:4]:
        if s == o:
            continue
        q = masks[s] & ~masks[o]
        if q.sum() < 2:
            continue
        tgt = {}
        tgt["partner"] = masks[o] & ~masks[s]
        sc = np.zeros(hp * hp, bool); un = np.zeros(hp * hp, bool); nn = None; nnd = 9
        for j in range(N):
            if j in (s, o) or iou(boxes[j], boxes[s]) > 0:
                continue
            if cats[j] == cats[s]:
                sc |= masks[j]
            elif not related[s, j]:
                un |= masks[j]
                dd = float(np.hypot(boxes[j][0] - boxes[s][0], boxes[j][1] - boxes[s][1]))
                if dd < nnd:
                    nnd, nn = dd, j
        tgt["same_class"] = sc & ~masks[s] & ~masks[o]
        tgt["unrelated"] = un & ~masks[s] & ~masks[o]
        tgt["unrelated_nn"] = (masks[nn] & ~masks[s] & ~masks[o]) if nn is not None else np.zeros(hp * hp, bool)
        tgt["background"] = ~any_box
        touch = iou(boxes[s], boxes[o]) > 0
        split = ("spatial" if (int(fl) & 1) else "semantic", "touch" if touch else "apart")
        for name, fd in feats.items():
            for lk, X in fd.items():
                Q = X[torch.from_numpy(q).to(X.device)]
                S = Q @ X.T                                          # [nq, P]
                row = {"self": float((Q @ Q.T).sum() - Q.shape[0]) / max(Q.shape[0] * (Q.shape[0] - 1), 1)}
                for t, m in tgt.items():
                    row[t] = float(S[:, torch.from_numpy(m).to(X.device)].mean()) if m.sum() else None
                for key in (("all", "all"), split, (split[0], "all"), ("all", split[1])):
                    acc.setdefault(name, {}).setdefault(lk, {}).setdefault(key, []).append(row)


def summarise(acc):
    out = {}
    for name, per_layer in acc.items():
        out[name] = {}
        for lk, per_split in per_layer.items():
            out[name][lk] = {}
            for key, rows in per_split.items():
                r = {}
                for t in TARGETS:
                    v = [x[t] for x in rows if x.get(t) is not None]
                    r[t] = float(np.mean(v)) if v else None
                    r[f"n_{t}"] = len(v)
                both = [x for x in rows if x.get("partner") is not None and x.get("same_class") is not None]
                r["p_partner_gt_sameclass"] = float(np.mean([x["partner"] > x["same_class"] for x in both])) if both else None
                r["n_both"] = len(both)
                bu = [x for x in rows if x.get("partner") is not None and x.get("unrelated_nn") is not None]
                r["p_partner_gt_unrelated_nn"] = float(np.mean([x["partner"] > x["unrelated_nn"] for x in bu])) if bu else None
                r["n_partner_vs_nn"] = len(bu)
                if r["partner"] is not None and r["unrelated"] is not None:
                    r["relation_selectivity"] = r["partner"] - r["unrelated"]
                if r["same_class"] is not None and r["unrelated"] is not None:
                    r["class_selectivity"] = r["same_class"] - r["unrelated"]
                if r["partner"] is not None and r["same_class"] is not None:
                    r["partner_minus_sameclass"] = r["partner"] - r["same_class"]
                out[name][lk]["|".join(key)] = r
    return out


def plot_curves(summary, layers, names, out_path, split="all|all"):
    lks = [str(l) for l in layers] + ["fused"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    cols = {"pretrained": "#9aa3ad"}
    palette = ["#2a78d6", "#1baf7a", "#8a63d2", "#eb6834"]
    for i, n in enumerate([n for n in names if n != "pretrained"]):
        cols[n] = palette[i % len(palette)]
    x = np.arange(len(lks))
    for ax, (metric, title) in zip(axes, [("relation_selectivity", "partner − unrelated  (relation selectivity)"),
                                          ("class_selectivity", "same-class other − unrelated  (class selectivity)"),
                                          ("partner_minus_sameclass", "partner − same-class other"),
                                          ("p_partner_gt_sameclass", "P(partner more similar than same-class other)")]):
        for n in names:
            ys = [summary[n][lk][split].get(metric) for lk in lks]
            ax.plot(x, [np.nan if v is None else v for v in ys], "-o", color=cols[n], lw=2, ms=4, label=n)
        ax.set_xticks(x); ax.set_xticklabels([("L" + k if k != "fused" else "fused") for k in lks], fontsize=8)
        ax.set_title(title, fontsize=9); ax.grid(alpha=.25)
        if metric.startswith("p_"):
            ax.axhline(0.5, ls=":", color="#999", lw=1); ax.set_ylim(0, 1)
        else:
            ax.axhline(0, ls=":", color="#999", lw=1)
    axes[0].legend(fontsize=8); fig.suptitle(f"mean patch cosine from subject-box patches, split = {split}", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)


def draw_maps(models, ds, idx, layers, names, hp, n_patch, device, out_path, img_size):
    pil, boxes, rels = ds.load_raw(idx)
    _, _, _, b0, nb, r0, nr = ds.img_meta[idx]
    cats = np.array(ds.box_cats[b0:b0 + nb]); cat_names = ds.meta["categories"]
    boxes = boxes[:nb]; rels = rels[(rels[:, 0] < nb) & (rels[:, 1] < nb)]
    if len(rels) == 0:
        return False
    # pick the relation with the largest subject box that is not the whole image
    areas = boxes[rels[:, 0], 2] * boxes[rels[:, 0], 3]
    ok = (areas < 0.6) & (rels[:, 0] != rels[:, 1])
    if not ok.any():
        return False
    r = rels[np.argmax(np.where(ok, areas, -1))]
    s, o = int(r[0]), int(r[1]); pred = ds.predicate_names[int(r[2])] if int(r[2]) < len(ds.predicate_names) else str(r[2])
    img = pil.resize((img_size, img_size))
    image = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0).to(device)
    feats = feats_for(models, image, n_patch, layers)
    masks = box_patch_masks(boxes, hp)
    def centre_patch(b):
        return int(min(hp - 1, b[1] * hp)) * hp + int(min(hp - 1, b[0] * hp))
    bg = np.where(~masks.any(0))[0]
    queries = [("subject: " + cat_names[cats[s]], centre_patch(boxes[s])), ("object: " + cat_names[cats[o]], centre_patch(boxes[o]))]
    if len(bg):
        queries.append(("background", int(bg[len(bg) // 2])))
    L = str(layers[-1])
    cols = [("image", None, None)] + [(n, L, n) for n in names] + [(names[1] + " fused", "fused", names[1])]
    fig, axes = plt.subplots(len(queries), len(cols), figsize=(3.1 * len(cols), 3.2 * len(queries)), squeeze=False)
    for qi, (qname, qp) in enumerate(queries):
        qy, qx = divmod(qp, hp)
        for ci, (title, lk, mname) in enumerate(cols):
            ax = axes[qi][ci]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if lk is None:
                for b, c in ((boxes[s], C_SUB), (boxes[o], C_OBJ)):
                    ax.add_patch(Rectangle(((b[0] - b[2] / 2) * img_size, (b[1] - b[3] / 2) * img_size), b[2] * img_size, b[3] * img_size, fill=False, ec=c, lw=2))
                ax.set_title(f"{cat_names[cats[s]]} → {pred} → {cat_names[cats[o]]}", fontsize=8)
            else:
                X = feats[mname][lk]; sim = (X @ X[qp]).reshape(hp, hp).cpu().numpy()
                if MAP_NORM == "stretch":
                    lo, hi = np.percentile(sim, 2), np.percentile(sim, 98)
                    sim = (sim - lo) / max(hi - lo, 1e-6)
                up = np.kron(sim, np.ones((img_size // hp, img_size // hp)))
                ax.imshow(np.clip(up, 0, 1), cmap="magma", alpha=0.75, vmin=0, vmax=1)
                ax.set_title(f"{title} @L{lk}" if lk != "fused" else title, fontsize=8)
            ax.plot((qx + 0.5) * img_size / hp, (qy + 0.5) * img_size / hp, "o", ms=7, mfc="#ff2a2a", mec="white", mew=1.2)
            if ci == 0:
                ax.set_ylabel("query: " + qname, fontsize=8)
    fig.suptitle("cosine similarity to the query patch (red); " + ("per-panel 2–98 % stretch" if MAP_NORM == "stretch" else "fixed scale 0–1")
                 + ("; per-image mean-centred tokens" if CENTER else ""), fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--weights", default="ema")
    ap.add_argument("--data_root", default="runs/packed/psg")
    ap.add_argument("--split", default="val")
    ap.add_argument("--layers", default="4,8,12")
    ap.add_argument("--n_stats", type=int, default=300)
    ap.add_argument("--n_viz", type=int, default=6)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--center", action="store_true", help="subtract the per-image mean patch token before cosine")
    ap.add_argument("--map_norm", choices=["fixed", "stretch"], default="stretch")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    global CENTER, MAP_NORM
    CENTER = a.center; MAP_NORM = a.map_norm
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [int(x) for x in a.layers.split(",")]
    models = load_models(a.checkpoints, a.labels, a.weights, device); names = list(models)
    ds = RelationDataset(root=a.data_root, split=a.split, resolution=a.img_size)
    hp = a.img_size // models["pretrained"].patch_size; n_patch = hp * hp
    rng = random.Random(a.seed); order = list(range(len(ds))); rng.shuffle(order)
    acc = {}; n_done = 0
    for idx in order:
        if n_done >= a.n_stats:
            break
        pil, boxes, rels = ds.load_raw(idx)
        _, _, _, b0, nb, r0, nr = ds.img_meta[idx]
        boxes = boxes[:nb]; rels = rels[(rels[:, 0] < nb) & (rels[:, 1] < nb)]
        if len(rels) == 0 or len(boxes) < 3:
            continue
        cats = np.array(ds.box_cats[b0:b0 + nb])
        img = pil.resize((a.img_size, a.img_size))
        image = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0).to(device)
        feats = feats_for(models, image, n_patch, layers)
        image_stats(feats, boxes, cats, rels, hp, acc)
        n_done += 1
        if n_done % 50 == 0:
            print(f"[sim] {n_done}/{a.n_stats} images", flush=True)
    summary = summarise(acc)
    json.dump({"n_images": n_done, "layers": layers, "targets": TARGETS, "summary": summary}, open(os.path.join(a.out_dir, "patch_similarity.json"), "w"), indent=1)
    for split in ("all|all", "spatial|all", "semantic|all", "all|touch", "all|apart"):
        if all(split in summary[n][str(layers[0])] for n in names):
            plot_curves(summary, layers, names, os.path.join(a.out_dir, f"curves_{split.replace('|', '_')}.png"), split)
    # console table
    lk = "fused"
    print(f"\n{'model':12s} layer   self  partner same_cls unrel  unrel_nn  bg   | rel_sel cls_sel  partner-same  P(partner>same) n")
    for n in names:
        for lk in [str(l) for l in layers] + ["fused"]:
            r = summary[n][lk]["all|all"]
            f = lambda k: ("  --  " if r.get(k) is None else f"{r[k]:6.3f}")
            print(f"{n:12s} {lk:5s} {f('self')} {f('partner')} {f('same_class')} {f('unrelated')} {f('unrelated_nn')} {f('background')} | {f('relation_selectivity')} {f('class_selectivity')} {f('partner_minus_sameclass')}   {f('p_partner_gt_sameclass')} {r['n_both']}")
    for split in ("spatial|all", "semantic|all", "all|touch", "all|apart"):
        print(f"-- {split}: fused P(partner>same_class) " + "  ".join(f"{n} {summary[n]['fused'][split]['p_partner_gt_sameclass']}" for n in names if split in summary[n]["fused"]))
    # maps
    done = 0
    for idx in order[a.n_stats:]:
        if done >= a.n_viz:
            break
        if draw_maps(models, ds, idx, layers, names, hp, n_patch, device, os.path.join(a.out_dir, f"maps_{done:02d}_ds{idx}.png"), a.img_size):
            done += 1
    print(f"done -> {a.out_dir}")


if __name__ == "__main__":
    main()
