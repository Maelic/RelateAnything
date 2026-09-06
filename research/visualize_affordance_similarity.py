"""Hand/contact-centred DINOv3 similarity maps on HICO + an emergent-affordance retrieval test.

Complements analyze_patch_similarity.py (which showed fine-tuning sharpens CLASS similarity).
Here the query is placed where the action happens, on human-object interactions:

  contact patch   the person-box patch inside the person∩object intersection (or the person
                  patch nearest the object centre when boxes do not overlap) — for holding /
                  cutting with / texting on / drinking with this is the hand.
  far patch       the person-box patch farthest from the object (head / torso) — control.
  object patch    the object-box centre.

MAPS  (--n_viz images, hand-verb relations only): rows = the three queries, columns = image |
pretrained | fine-tuned arms | fused. Mean-centred tokens, per-panel 2-98 % stretch.

STATISTICS over --n_stats hand-verb relations
  contact advantage   cos(contact -> object patches) - cos(far -> object patches), per model /
                      layer. Positive = the hand region resembles the object it acts on more
                      than the rest of the person does.
  affordance retrieval  one feature per relation = mean of its contact patches (and, as a
                      control, of its far patches). Leave-one-out 1-NN across images; accuracy
                      of predicting the VERB and of predicting the OBJECT CLASS, vs the
                      majority-class chance. If contact features retrieve the verb better than
                      the object class explains, the patch features carry affordance.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research.analyze_patch_similarity as aps  # noqa: E402
from relsgg.data.dataset import RelationDataset  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HAND_VERBS = {"holding", "carrying", "wielding", "cutting with", "drinking with", "eating", "brushing with",
              "texting on", "typing on", "talking on", "swinging", "throwing", "catching", "hitting", "reading",
              "opening", "washing", "stirring", "pouring", "peeling", "feeding", "petting", "painting", "repairing",
              "pushing", "pulling", "lifting", "picking up", "filling", "squeezing", "tying", "adjusting", "zipping",
              "cutting", "operating", "packing", "cleaning", "scratching", "stabbing", "serving", "milking", "grinding",
              "loading", "pointing", "inspecting", "waving", "dragging", "flipping", "blowing", "hugging", "kicking"}


def contact_and_far(boxes, masks, s, o, hp):
    """indices of (contact patches, far patch, contact query patch) for subject s and object o."""
    ys, xs = np.divmod(np.arange(hp * hp), hp); cx, cy = xs / hp + 0.5 / hp, ys / hp + 0.5 / hp
    person = np.where(masks[s])[0]
    if len(person) == 0:
        return None
    ocx, ocy = boxes[o][0], boxes[o][1]
    d = np.hypot(cx[person] - ocx, cy[person] - ocy)
    inter = person[masks[o][person]]
    contact = inter if len(inter) else person[np.argsort(d)[:max(1, len(person) // 8)]]
    # the query = contact patch nearest the object centre
    dc = np.hypot(cx[contact] - ocx, cy[contact] - ocy); q = int(contact[np.argmin(dc)])
    far = int(person[np.argmax(d)])
    return contact, far, q


def draw(models, ds, idx, r, layers, names, hp, n_patch, device, out_path, img_size):
    pil, boxes, rels = ds.load_raw(idx)
    _, _, _, b0, nb, r0, nr = ds.img_meta[idx]
    cats = np.array(ds.box_cats[b0:b0 + nb]); cat_names = ds.meta["categories"]
    boxes = boxes[:nb]
    s, o, p = int(r[0]), int(r[1]), int(r[2]); pred = ds.predicate_names[p]
    img = pil.resize((img_size, img_size))
    image = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0).to(device)
    feats = aps.feats_for(models, image, n_patch, layers)
    masks = aps.box_patch_masks(boxes, hp)
    cf = contact_and_far(boxes, masks, s, o, hp)
    if cf is None:
        return False
    contact, far, q = cf
    oc = int(min(hp - 1, boxes[o][1] * hp)) * hp + int(min(hp - 1, boxes[o][0] * hp))
    queries = [("contact (hand)", q), ("far person", far), ("object: " + cat_names[cats[o]], oc)]
    L = str(layers[-1])
    cols = [("image", None, None)] + [(n, L, n) for n in names] + [(names[1] + " fused", "fused", names[1])]
    fig, axes = plt.subplots(len(queries), len(cols), figsize=(3.1 * len(cols), 3.2 * len(queries)), squeeze=False)
    for qi, (qname, qp) in enumerate(queries):
        qy, qx = divmod(qp, hp)
        for ci, (title, lk, mname) in enumerate(cols):
            ax = axes[qi][ci]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
            if lk is None:
                for b, c in ((boxes[s], aps.C_SUB), (boxes[o], aps.C_OBJ)):
                    ax.add_patch(Rectangle(((b[0] - b[2] / 2) * img_size, (b[1] - b[3] / 2) * img_size), b[2] * img_size, b[3] * img_size, fill=False, ec=c, lw=2))
                ax.set_title(f"person → {pred} → {cat_names[cats[o]]}", fontsize=8)
            else:
                X = feats[mname][lk]; sim = (X @ X[qp]).reshape(hp, hp).cpu().numpy()
                lo, hi = np.percentile(sim, 2), np.percentile(sim, 98); sim = (sim - lo) / max(hi - lo, 1e-6)
                up = np.kron(sim, np.ones((img_size // hp, img_size // hp)))
                ax.imshow(np.clip(up, 0, 1), cmap="magma", alpha=0.75, vmin=0, vmax=1)
                ax.set_title(f"{title} @L{lk}" if lk != "fused" else title, fontsize=8)
            ax.plot((qx + 0.5) * img_size / hp, (qy + 0.5) * img_size / hp, "o", ms=7, mfc="#ff2a2a", mec="white", mew=1.2)
            if ci == 0:
                ax.set_ylabel("query: " + qname, fontsize=8)
    fig.suptitle("cosine similarity to the query patch (red); mean-centred tokens, per-panel 2–98 % stretch", fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)
    return True


def loo_1nn(F, y):
    """leave-one-out 1-NN accuracy on L2-normalised features F [n, d], labels y [n]."""
    S = F @ F.T; np.fill_diagonal(S, -np.inf)
    return float((y[S.argmax(1)] == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--data_root", default="runs/packed/hicodet")
    ap.add_argument("--split", default="test")
    ap.add_argument("--layers", default="8,12")
    ap.add_argument("--n_stats", type=int, default=600, help="hand-verb relations for the statistics")
    ap.add_argument("--n_viz", type=int, default=8)
    ap.add_argument("--min_verb_n", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    aps.CENTER = True; aps.MAP_NORM = "stretch"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [int(x) for x in a.layers.split(",")]
    models = aps.load_models(a.checkpoints, a.labels, "ema", device); names = list(models)
    ds = RelationDataset(root=a.data_root, split=a.split, resolution=a.img_size)
    hp = a.img_size // models["pretrained"].patch_size; n_patch = hp * hp
    pn = ds.predicate_names; hand_ids = {i for i, n in enumerate(pn) if n in HAND_VERBS}
    cat_names = ds.meta["categories"]
    rng = random.Random(a.seed); order = list(range(len(ds))); rng.shuffle(order)
    lks = [str(l) for l in layers] + ["fused"]
    feat = {n: {lk: {"contact": [], "far": []} for lk in lks} for n in names}
    adv = {n: {lk: [] for lk in lks} for n in names}
    verbs, objs = [], []; used = []
    for idx in order:
        if len(verbs) >= a.n_stats:
            break
        pil, boxes, rels = ds.load_raw(idx)
        _, _, _, b0, nb, r0, nr = ds.img_meta[idx]
        boxes = boxes[:nb]; rels = rels[(rels[:, 0] < nb) & (rels[:, 1] < nb)]
        rels = rels[[int(r[2]) in hand_ids and r[0] != r[1] for r in rels]] if len(rels) else rels
        if len(rels) == 0:
            continue
        cats = np.array(ds.box_cats[b0:b0 + nb])
        img = pil.resize((a.img_size, a.img_size))
        image = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0).to(device)
        feats = aps.feats_for(models, image, n_patch, layers)
        masks = aps.box_patch_masks(boxes, hp)
        seen = set()
        for r in rels:
            s, o, p = int(r[0]), int(r[1]), int(r[2])
            if (s, o) in seen:
                continue
            seen.add((s, o))
            cf = contact_and_far(boxes, masks, s, o, hp)
            if cf is None:
                continue
            contact, far, q = cf
            obj_only = np.where(masks[o] & ~masks[s])[0]
            if len(obj_only) == 0:
                continue
            for n in names:
                for lk in lks:
                    X = feats[n][lk]
                    c = X[torch.as_tensor(contact, device=X.device)].mean(0); f = X[far]
                    c = c / c.norm().clamp(min=1e-6)
                    O = X[torch.as_tensor(obj_only, device=X.device)]
                    adv[n][lk].append(float((O @ c).mean() - (O @ f).mean()))
                    feat[n][lk]["contact"].append(c.cpu().numpy()); feat[n][lk]["far"].append(f.cpu().numpy())
            verbs.append(p); objs.append(int(cats[o])); used.append((idx, s, o, p))
        if len(verbs) % 100 < len(seen):
            print(f"[aff] {len(verbs)} relations", flush=True)
    verbs = np.array(verbs); objs = np.array(objs)
    vc = Counter(verbs.tolist()); keep = np.array([vc[v] >= a.min_verb_n for v in verbs])
    res = {"n_relations": int(len(verbs)), "n_kept": int(keep.sum()), "verbs_kept": sorted({pn[v] for v in verbs[keep]}),
           "chance_verb": float(max(Counter(verbs[keep].tolist()).values()) / keep.sum()),
           "chance_object": float(max(Counter(objs[keep].tolist()).values()) / keep.sum()), "models": {}}
    print(f"\nkept {keep.sum()} relations over {len(res['verbs_kept'])} verbs; chance verb {res['chance_verb']:.3f} object {res['chance_object']:.3f}")
    print(f"{'model':11s} layer  contact_adv  | 1-NN verb: contact  far   | 1-NN object: contact  far")
    for n in names:
        res["models"][n] = {}
        for lk in lks:
            Fc = np.stack(feat[n][lk]["contact"])[keep]; Ff = np.stack(feat[n][lk]["far"])[keep]
            Fc /= np.linalg.norm(Fc, axis=1, keepdims=True).clip(1e-6); Ff /= np.linalg.norm(Ff, axis=1, keepdims=True).clip(1e-6)
            r = {"contact_advantage": float(np.mean(adv[n][lk])), "contact_advantage_pos_frac": float(np.mean(np.array(adv[n][lk]) > 0)),
                 "verb_1nn_contact": loo_1nn(Fc, verbs[keep]), "verb_1nn_far": loo_1nn(Ff, verbs[keep]),
                 "object_1nn_contact": loo_1nn(Fc, objs[keep]), "object_1nn_far": loo_1nn(Ff, objs[keep])}
            res["models"][n][lk] = r
            print(f"{n:11s} {lk:5s}  {r['contact_advantage']:+.4f} ({r['contact_advantage_pos_frac']:.2f}) |   {r['verb_1nn_contact']:.3f}   {r['verb_1nn_far']:.3f}  |   {r['object_1nn_contact']:.3f}   {r['object_1nn_far']:.3f}")
    json.dump(res, open(os.path.join(a.out_dir, "affordance.json"), "w"), indent=1)
    # figure: verb vs object retrieval, contact vs far, per model at fused + L12
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    cols = {"pretrained": "#9aa3ad"}; pal = ["#2a78d6", "#1baf7a", "#8a63d2"]
    for i, n in enumerate([n for n in names if n != "pretrained"]):
        cols[n] = pal[i % 3]
    x = np.arange(len(lks)); w = 0.8 / len(names)
    for ax, key, title, chance in ((axes[0], "verb", "1-NN retrieval of the VERB (contact patches, filled; far patches, hollow)", res["chance_verb"]),
                                   (axes[1], "object", "1-NN retrieval of the OBJECT CLASS", res["chance_object"])):
        for i, n in enumerate(names):
            ax.bar(x + (i - len(names) / 2 + 0.5) * w, [res["models"][n][lk][f"{key}_1nn_contact"] for lk in lks], w, color=cols[n], label=n)
            ax.bar(x + (i - len(names) / 2 + 0.5) * w, [res["models"][n][lk][f"{key}_1nn_far"] for lk in lks], w, fill=False, ec=cols[n], lw=1.5, ls="--")
        ax.axhline(chance, ls=":", color="#999", lw=1); ax.set_xticks(x); ax.set_xticklabels([("L" + k if k != "fused" else "fused") for k in lks])
        ax.set_title(title, fontsize=9); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.25)
    axes[0].legend(fontsize=8); fig.suptitle(f"HICO {a.split}: {int(keep.sum())} hand-verb relations, {len(res['verbs_kept'])} verbs (dotted = majority-class chance)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "affordance_retrieval.png"), dpi=150); plt.close(fig)
    # maps: prefer small objects (hand-sized) with overlap
    cand = [(idx, s, o, p) for (idx, s, o, p) in used]
    rng.shuffle(cand)
    done = 0
    for idx, s, o, p in cand:
        if done >= a.n_viz:
            break
        pil, boxes, rels = ds.load_raw(idx); _, _, _, b0, nb, r0, nr = ds.img_meta[idx]
        if boxes[o][2] * boxes[o][3] > 0.15 or aps.iou(boxes[s], boxes[o]) == 0:
            continue
        if draw(models, ds, idx, np.array([s, o, p]), layers, names, hp, n_patch, device,
                os.path.join(a.out_dir, f"aff_{done:02d}_{pn[p].replace(' ', '_')}_ds{idx}.png"), a.img_size):
            done += 1
    print(f"done -> {a.out_dir}")


if __name__ == "__main__":
    main()
