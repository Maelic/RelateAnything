"""Readable qualitative triptychs: GT | arm A | arm B, edges drawn as arrows.

Supersedes the box-only renderer in e3_qualitative.py, which drew every
referenced box on ONE canvas and left the two graphs as text in a JSON file —
unreadable as a comparison, and its labels ran off the top edge when a box
touched y=0.

What changed:
  * three panels on one canvas, so the two arms are actually side by side;
  * EDGES are drawn (arrow subject -> object, predicate at the midpoint), not
    just boxes, because the edge set is the thing being compared;
  * only boxes touched by the drawn edges are outlined, and each panel draws
    only ITS OWN edges' boxes;
  * --focus restricts every panel to one predicate family (default: wearing),
    which is what keeps the picture legible — a 14-edge graph drawn in full is
    a hairball regardless of layout;
  * label boxes are clamped inside the canvas.

Both arms are cut at the same per-image budget (that image's GT triple count),
so no panel can win by drawing more edges.

    python training/e3_panels.py \
        runs/analysis/e3/dump_topk_vg150.npz runs/analysis/e3/dump_lse_vg150.npz \
        --labels topk lse --pack runs/packed/vg150/test --n 6 \
        --out runs/analysis/e3/panels
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WEAR_RX = re.compile(r"wear|dressed in|has on|clothed", re.I)
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (213, 94, 0),
           (0, 114, 178), (204, 121, 167), (140, 86, 75), (127, 127, 127)]
PANEL_W = 460


def font(sz, bold=True):
    for p in (f"/usr/share/fonts/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
              f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def load(path):
    z = np.load(path, allow_pickle=False)
    pc = z["pair_counts"].astype(np.int64)
    bc = z["box_counts"].astype(np.int64)
    gc = z["gt_counts"].astype(np.int64)
    return dict(
        names=[str(s) for s in z["pred_names"]],
        imgs=[str(s) for s in z["img_names"]],
        pc=pc, bc=bc, gc=gc,
        p_off=np.concatenate([[0], np.cumsum(pc)])[:-1],
        b_off=np.concatenate([[0], np.cumsum(bc)])[:-1],
        g_off=np.concatenate([[0], np.cumsum(gc)])[:-1],
        sub=z["sub"].astype(np.int64), obj=z["obj"].astype(np.int64),
        zpred=z["z_pred"], zpair=z["z_pair"].astype(np.float32),
        boxes=z["boxes"].astype(np.float32),
        cats=z["box_cats"].astype(np.int64), gt=z["gt"].astype(np.int64))


def edges_for(d, i, budget):
    p0, n = d["p_off"][i], d["pc"][i]
    if n == 0 or budget <= 0:
        return np.zeros((0, 3), np.int64)
    zp = np.asarray(d["zpred"][p0:p0 + n], np.float32)
    best = zp.argmax(1)
    score = zp[np.arange(n), best] + d["zpair"][p0:p0 + n]
    k = min(budget, n)
    keep = np.argpartition(-score, k - 1)[:k]
    keep = keep[np.argsort(-score[keep])]
    return np.stack([d["sub"][p0 + keep], d["obj"][p0 + keep], best[keep]], 1)


def xyxy(b):
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def draw_panel(img, boxes_n, cats, catnames, edges, names, title, colour_of):
    """One panel: the image, the boxes its edges touch, and the edges."""
    W, H = img.size
    canvas = Image.new("RGB", (PANEL_W, int(H * PANEL_W / W) + 34), (250, 250, 250))
    im = img.resize((PANEL_W, int(H * PANEL_W / W)), Image.LANCZOS)
    canvas.paste(im, (0, 34))
    dr = ImageDraw.Draw(canvas, "RGBA")
    fs, fb = font(12, False), font(13)
    dr.rectangle([0, 0, PANEL_W, 33], fill=(32, 32, 32))
    dr.text((8, 9), title, fill=(255, 255, 255), font=fb)
    w, h = im.size

    used = sorted({int(x) for e in edges for x in (e[0], e[1])})
    pos = {}
    for n_, bi in enumerate(used):
        x1, y1, x2, y2 = xyxy(boxes_n[bi:bi + 1])[0]
        bx = [x1 * w, y1 * h + 34, x2 * w, y2 * h + 34]
        c = colour_of(bi)
        dr.rectangle(bx, outline=c, width=3)
        tag = f"{bi}:{catnames[cats[bi]]}"
        tw = dr.textlength(tag, font=fs)
        ty = max(34, bx[1] - 15)                       # clamp inside canvas
        tx = min(max(0, bx[0]), PANEL_W - tw - 6)
        dr.rectangle([tx, ty, tx + tw + 6, ty + 15], fill=c)
        dr.text((tx + 3, ty + 1), tag, fill=(0, 0, 0), font=fs)
        pos[bi] = ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2)

    for s, o, p_ in edges:
        s, o, p_ = int(s), int(o), int(p_)
        if s not in pos or o not in pos:
            continue
        (x1, y1), (x2, y2) = pos[s], pos[o]
        c = colour_of(s)
        dr.line([x1, y1, x2, y2], fill=c + (235,), width=3)
        vx, vy = x2 - x1, y2 - y1
        L = max((vx * vx + vy * vy) ** 0.5, 1e-6)
        ux, uy = vx / L, vy / L
        dr.polygon([(x2, y2),
                    (x2 - 11 * ux + 6 * uy, y2 - 11 * uy - 6 * ux),
                    (x2 - 11 * ux - 6 * uy, y2 - 11 * uy + 6 * ux)],
                   fill=c + (235,))
        lab = names[p_]
        tw = dr.textlength(lab, font=fs)
        mx = min(max(0, (x1 + x2) / 2 - tw / 2 - 3), PANEL_W - tw - 6)
        my = min(max(34, (y1 + y2) / 2 - 8), canvas.size[1] - 17)
        dr.rectangle([mx, my, mx + tw + 6, my + 16], fill=(255, 255, 255, 235))
        dr.text((mx + 3, my + 1), lab, fill=(20, 20, 20), font=fs)
    if not len(edges):
        dr.text((10, 44), "(no edge in this family)", fill=(120, 120, 120), font=fs)
    return canvas


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dumps", nargs=2)
    p.add_argument("--labels", nargs=2, default=["A", "B"])
    p.add_argument("--pack", default="runs/packed/vg150/test")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--focus", default="wear",
                   help="regex; only edges whose predicate matches are drawn. "
                        "'.' draws everything (usually a hairball).")
    p.add_argument("--min_gt", type=int, default=4)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    A, B = load(a.dumps[0]), load(a.dumps[1])
    meta = json.load(open(f"{a.pack}/meta.json"))
    catnames, img_dir = meta["categories"], meta["img_dir"]
    names = A["names"]
    rx = re.compile(a.focus, re.I)
    foc = np.array([bool(rx.search(n)) for n in names])

    cand = []
    for i in range(len(A["imgs"])):
        ng = int(A["gc"][i])
        if ng < a.min_gt or A["pc"][i] == 0:
            continue
        ea, eb = edges_for(A, i, ng), edges_for(B, i, ng)
        fa = {(int(s), int(o), int(q)) for s, o, q in ea if foc[q]}
        fb = {(int(s), int(o), int(q)) for s, o, q in eb if foc[q]}
        if not fa and not fb:
            continue
        nd = len(fa ^ fb)
        if nd == 0:
            continue
        if max(len(fa), len(fb)) > 7:      # keep the picture readable
            continue
        cand.append((nd, i))
    cand.sort(key=lambda t: (-t[0], t[1]))
    picked = sorted([i for _, i in cand[:a.n]])
    os.makedirs(a.out, exist_ok=True)

    for i in picked:
        ng = int(A["gc"][i])
        img = Image.open(os.path.join(img_dir, A["imgs"][i])).convert("RGB")
        b0, nb = A["b_off"][i], A["bc"][i]
        bx, ct = A["boxes"][b0:b0 + nb], A["cats"][b0:b0 + nb]
        g0 = A["g_off"][i]
        gt = A["gt"][g0:g0 + ng]
        gt = np.array([e for e in gt if foc[e[2]]] or [], np.int64).reshape(-1, 3)
        ea = np.array([e for e in edges_for(A, i, ng) if foc[e[2]]] or [],
                      np.int64).reshape(-1, 3)
        eb = np.array([e for e in edges_for(B, i, ng) if foc[e[2]]] or [],
                      np.int64).reshape(-1, 3)
        col = {}

        def colour_of(bi):
            if bi not in col:
                col[bi] = PALETTE[len(col) % len(PALETTE)]
            return col[bi]

        panels = [
            draw_panel(img, bx, ct, catnames, gt, names,
                       f"GROUND TRUTH  ({len(gt)} edges)", colour_of),
            draw_panel(img, bx, ct, catnames, ea, names,
                       f"{a.labels[0]}  ({len(ea)} edges)", colour_of),
            draw_panel(img, bx, ct, catnames, eb, names,
                       f"{a.labels[1]}  ({len(eb)} edges)", colour_of),
        ]
        Hh = max(q.size[1] for q in panels)
        out = Image.new("RGB", (PANEL_W * 3 + 16, Hh), (255, 255, 255))
        for k, q in enumerate(panels):
            out.paste(q, (k * (PANEL_W + 8), 0))
        out.save(os.path.join(a.out, f"{os.path.splitext(A['imgs'][i])[0]}.png"))
        print(f"{A['imgs'][i]}: GT {len(gt)} | {a.labels[0]} {len(ea)} | "
              f"{a.labels[1]} {len(eb)} edges (budget {ng})")
    print(f"\n{len(picked)} triptychs -> {a.out}")


if __name__ == "__main__":
    main()
