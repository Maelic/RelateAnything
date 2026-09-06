"""Decodability probe: do the fine-tuned DINOv3 dense features carry RELATEDNESS information?

Cosine similarity (analyze_patch_similarity.py) cannot see information that is linearly present
but not aligned with the dominant (identity) directions. This probe asks the question the right
way: freeze the backbone, mean-pool patch tokens inside each GT box, and train the SAME linear
probe on the SAME pairs for the pretrained and each fine-tuned backbone.

Per ordered pair (i, j) of GT boxes in an image (label = a GT relation i -> j exists):
  feature sets    so     [f_s, f_o]                 pooled subject + object tokens
                  sou    [f_s, f_o, f_union]        + the union-box pooling (interaction region)
                  so+geo [f_s, f_o, geometry]
  baselines       geo    12-d box geometry only (both boxes, deltas, IoU)
                  cls    one-hot categories of both boxes
                  cls+geo
Readouts (train/test split BY IMAGE, standardised features, class-balanced logistic regression):
  relatedness AUC            over all pairs
  within-class-pair AUC      AUC computed inside each (cat_s, cat_o) group that has >= 5 positives
                             and >= 5 negatives in the test split, averaged — identity cannot
                             explain this number, geometry is not in the features
  direction accuracy         among related pairs, which way round (i->j vs j->i)
  predicate accuracy / macro-F1   among related pairs, predicates with >= min_pred train samples
The gap between a fine-tuned backbone and the pretrained one under the identical probe is the
evidence for (or against) relatedness information in the features.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research.analyze_patch_similarity as aps  # noqa: E402
from relsgg.data.dataset import RelationDataset  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def geometry(bs, bo):
    iou = aps.iou(bs, bo)
    return np.array([bs[0], bs[1], bs[2], bs[3], bo[0], bo[1], bo[2], bo[3],
                     bo[0] - bs[0], bo[1] - bs[1], np.log((bo[2] * bo[3] + 1e-6) / (bs[2] * bs[3] + 1e-6)), iou], np.float32)


def union_box(a, b):
    x0 = min(a[0] - a[2] / 2, b[0] - b[2] / 2); y0 = min(a[1] - a[3] / 2, b[1] - b[3] / 2)
    x1 = max(a[0] + a[2] / 2, b[0] + b[2] / 2); y1 = max(a[1] + a[3] / 2, b[1] + b[3] / 2)
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], np.float32)


def pool(X, mask):
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return X.mean(0)
    return X[torch.as_tensor(idx, device=X.device)].mean(0)


def fit_probe(Xtr, ytr, Xte, multi=False):
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=400, C=0.5, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    Z = sc.transform(Xte)
    return clf.predict(Z) if multi else clf.predict_proba(Z)[:, 1]


def within_class_auc(p, y, cs, co, min_n=5):
    key = cs.astype(np.int64) * 100000 + co
    aucs, ws = [], []
    for k in np.unique(key):
        m = key == k
        if y[m].sum() >= min_n and (~y[m]).sum() >= min_n:
            aucs.append(roc_auc_score(y[m], p[m])); ws.append(m.sum())
    return (float(np.average(aucs, weights=ws)) if aucs else float("nan")), len(aucs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--data_root", default="runs/packed/psg")
    ap.add_argument("--split", default="val")
    ap.add_argument("--layers", default="8,12")
    ap.add_argument("--n_images", type=int, default=1500)
    ap.add_argument("--max_boxes", type=int, default=16)
    ap.add_argument("--neg_per_pos", type=int, default=6, help="cap on unrelated pairs kept per related pair, per image")
    ap.add_argument("--min_pred", type=int, default=30)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [int(x) for x in a.layers.split(",")]; lks = [str(l) for l in layers] + ["fused"]
    models = aps.load_models(a.checkpoints, a.labels, "ema", device); names = list(models)
    ds = RelationDataset(root=a.data_root, split=a.split, resolution=a.img_size)
    hp = a.img_size // models["pretrained"].patch_size; n_patch = hp * hp
    rng = random.Random(a.seed); order = list(range(len(ds))); rng.shuffle(order); order = order[:a.n_images]
    n_cat = len(ds.meta["categories"])
    F = {n: {lk: {"s": [], "o": [], "u": []} for lk in lks} for n in names}
    G, CS, CO, Y, PRED, IMG = [], [], [], [], [], []
    for t, idx in enumerate(order):
        pil, boxes, rels = ds.load_raw(idx)
        _, _, _, b0, nb, r0, nr = ds.img_meta[idx]
        nb = min(int(nb), a.max_boxes); boxes = boxes[:nb]
        rels = rels[(rels[:, 0] < nb) & (rels[:, 1] < nb)] if len(rels) else rels
        if nb < 2 or len(rels) == 0:
            continue
        cats = np.array(ds.box_cats[b0:b0 + nb])
        img = pil.resize((a.img_size, a.img_size))
        image = torch.from_numpy(np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0).to(device)
        with torch.no_grad():
            feats = {}
            for n, bb in models.items():
                hs, _, fused, _, _, _ = aps.hidden_stack(bb, image[None], n_patch)
                feats[n] = {str(l): hs[l][0].float() for l in layers}; feats[n]["fused"] = fused[0].float()
        masks = aps.box_patch_masks(boxes, hp)
        pooled = {n: {lk: [pool(feats[n][lk], masks[i]) for i in range(nb)] for lk in lks} for n in names}
        gt = {}
        for r in rels:
            gt.setdefault((int(r[0]), int(r[1])), int(r[2]))
        pos = list(gt); neg = [(i, j) for i in range(nb) for j in range(nb) if i != j and (i, j) not in gt]
        rng.shuffle(neg); neg = neg[:a.neg_per_pos * len(pos)]
        for (i, j) in pos + neg:
            ub = union_box(boxes[i], boxes[j]); um = aps.box_patch_masks(ub[None], hp)[0]
            for n in names:
                for lk in lks:
                    F[n][lk]["s"].append(pooled[n][lk][i].cpu().numpy()); F[n][lk]["o"].append(pooled[n][lk][j].cpu().numpy())
                    F[n][lk]["u"].append(pool(feats[n][lk], um).cpu().numpy())
            G.append(geometry(boxes[i], boxes[j])); CS.append(int(cats[i])); CO.append(int(cats[j]))
            Y.append((i, j) in gt); PRED.append(gt.get((i, j), -1)); IMG.append(idx)
        if (t + 1) % 200 == 0:
            print(f"[probe] {t + 1}/{len(order)} images, {len(Y)} pairs", flush=True)
    G = np.stack(G); CS = np.array(CS); CO = np.array(CO); Y = np.array(Y); PRED = np.array(PRED); IMG = np.array(IMG)
    imgs = np.unique(IMG); rng.shuffle(imgs.tolist()); te_imgs = set(np.random.RandomState(a.seed).permutation(imgs)[: len(imgs) * 3 // 10].tolist())
    te = np.array([i in te_imgs for i in IMG]); tr = ~te
    onehot = np.zeros((len(Y), 2 * n_cat), np.float32); onehot[np.arange(len(Y)), CS] = 1; onehot[np.arange(len(Y)), n_cat + CO] = 1
    print(f"pairs {len(Y)} ({Y.sum()} related) train {tr.sum()} test {te.sum()} images {len(imgs)}")
    res = {"n_pairs": int(len(Y)), "n_related": int(Y.sum()), "n_images": int(len(imgs)), "sets": {}}

    def readouts(name, X):
        p = fit_probe(X[tr], Y[tr], X[te])
        auc = roc_auc_score(Y[te], p); wauc, ng = within_class_auc(p, Y[te], CS[te], CO[te])
        # direction: among related pairs, is (i,j) or (j,i) the GT direction? build swapped copies
        rel = Y & tr; relte = Y & te
        out = {"auc": float(auc), "within_class_auc": wauc, "n_class_groups": ng}
        # predicate readout
        pm = PRED >= 0
        cnt = np.bincount(PRED[pm & tr], minlength=PRED.max() + 1); keep_p = np.where(cnt >= a.min_pred)[0]
        kp = np.isin(PRED, keep_p) & pm
        if len(keep_p) >= 3 and (kp & te).sum() > 20:
            yp = fit_probe(X[kp & tr], PRED[kp & tr], X[kp & te], multi=True)
            out["pred_acc"] = float((yp == PRED[kp & te]).mean()); out["pred_macro_f1"] = float(f1_score(PRED[kp & te], yp, average="macro"))
            out["n_predicates"] = int(len(keep_p)); out["pred_majority"] = float(np.bincount(PRED[kp & te]).max() / (kp & te).sum())
        res["sets"][name] = out
        print(f"{name:26s} AUC {auc:.3f}  within-class AUC {wauc:.3f} ({ng} groups)  " + (f"pred acc {out['pred_acc']:.3f} macroF1 {out['pred_macro_f1']:.3f} (n_pred {out['n_predicates']}, majority {out['pred_majority']:.3f})" if "pred_acc" in out else ""), flush=True)

    readouts("geo", G); readouts("cls", onehot); readouts("cls+geo", np.concatenate([onehot, G], 1))
    for n in names:
        for lk in lks:
            S = np.stack(F[n][lk]["s"]); O = np.stack(F[n][lk]["o"]); U = np.stack(F[n][lk]["u"])
            readouts(f"{n}@{lk} so", np.concatenate([S, O], 1))
            readouts(f"{n}@{lk} sou", np.concatenate([S, O, U], 1))
            readouts(f"{n}@{lk} so+geo", np.concatenate([S, O, G], 1))
    json.dump(res, open(os.path.join(a.out_dir, "feature_relatedness_probe.json"), "w"), indent=1)
    # figure: AUC / within-class AUC / predicate acc for so and sou per model at each layer, plus baselines
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cols = {"pretrained": "#9aa3ad"}; pal = ["#2a78d6", "#1baf7a", "#8a63d2"]
    for i, n in enumerate([n for n in names if n != "pretrained"]):
        cols[n] = pal[i % 3]
    for ax, key, title in zip(axes, ["auc", "within_class_auc", "pred_acc"], ["relatedness AUC (all pairs)", "within-class-pair AUC (identity removed)", "predicate accuracy (related pairs)"]):
        x = np.arange(len(lks) * 2); w = 0.8 / len(names)
        for i, n in enumerate(names):
            vals = [res["sets"].get(f"{n}@{lk} {fs}", {}).get(key, np.nan) for lk in lks for fs in ("so", "sou")]
            ax.bar(x + (i - len(names) / 2 + 0.5) * w, vals, w, color=cols[n], label=n)
        for bname, ls in (("geo", ":"), ("cls", "--"), ("cls+geo", "-.")):
            v = res["sets"][bname].get(key)
            if v is not None and v == v:
                ax.axhline(v, ls=ls, color="#555", lw=1, label=bname)
        ax.set_xticks(x); ax.set_xticklabels([f"{('L' + lk) if lk != 'fused' else 'fused'}\n{fs}" for lk in lks for fs in ("so", "sou")], fontsize=8)
        ax.set_title(title, fontsize=9); ax.grid(axis="y", alpha=.25)
        if key != "pred_acc":
            ax.set_ylim(0.4, 1.0)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{a.data_root.split('/')[-1]} {a.split}: linear probe on frozen pooled features, {res['n_pairs']} pairs / {res['n_related']} related, split by image", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out_dir, "feature_relatedness_probe.png"), dpi=150); plt.close(fig)
    print(f"done -> {a.out_dir}")


if __name__ == "__main__":
    main()
