"""Side-by-side qualitative panels: one image, its numbered boxes, both models' graphs.

Selection is RULE-BASED and deterministic, never hand-picked — otherwise a qualitative
figure is just an argument with pictures. Images are taken in pack order, keeping the
first `--n` that satisfy legibility constraints stated in the output JSON:

  * both models emit at least `--min_rel` relations (so there is something to compare),
  * the union of boxes the two graphs actually reference is <= `--max_marks`
    (more numbered boxes than that and the overlay is unreadable),
  * the image has at least `--min_boxes` detections.

Both graphs are named from ONE shared label source (see `obj_labels`): the two records
address the same detector boxes, and OvSGTR's labels are 1-based, so a per-record naming
path silently renames every object by one category.

    python benchmark/make_qualitative.py \
        --a runs/judge/relsgg_psg_test_yoloworld_pack.npz --a_name RelSGG \
        --b runs/sgdet/ovdr_mega_psg_test_yoloworld.npz --b_name OvSGTR \
        --pack runs/packed/psg/test --n 10 --out runs/judge/qualitative_psg.json
"""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.llm_judge import load_npz, obj_labels, triplets  # noqa: E402

# Distinct, colour-blind-safe marker colours cycled across the numbered boxes.
PALETTE = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66),
           (0, 114, 178), (213, 94, 0), (204, 121, 167), (150, 150, 150)]


def render(img_path, boxes, marks, max_side=680):
    """Draw only the boxes the two graphs actually reference, numbered to match."""
    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    sc = min(1.0, max_side / max(W, H))
    if sc < 1.0:
        im = im.resize((int(W * sc), int(H * sc)), Image.LANCZOS)
    dr = ImageDraw.Draw(im, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for j, i in enumerate(sorted(marks)):
        if i >= len(boxes):
            continue
        col = PALETTE[j % len(PALETTE)]
        x0, y0, x1, y1 = [v * sc for v in boxes[i]]
        dr.rectangle([x0, y0, x1, y1], outline=col + (255,), width=3)
        tag = str(i)
        tw = dr.textlength(tag, font=font)
        dr.rectangle([x0, y0, x0 + tw + 10, y0 + 22], fill=col + (255,))
        dr.text((x0 + 5, y0 + 3), tag, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", required=True); p.add_argument("--a_name", default="ours")
    p.add_argument("--b", required=True); p.add_argument("--b_name", default="baseline")
    p.add_argument("--pack", required=True)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--max_rel", type=int, default=20)
    p.add_argument("--rel_frac", type=float, default=0.7)
    p.add_argument("--min_rel", type=int, default=6)
    p.add_argument("--max_marks", type=int, default=10)
    p.add_argument("--min_boxes", type=int, default=5)
    p.add_argument("--show", type=int, default=10, help="relations listed per graph")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    A, B = load_npz(a.a), load_npz(a.b)
    pack = Path(a.pack)
    meta = json.loads((pack / "meta.json").read_text())
    cat_names = list(meta["categories"])
    file_names = json.loads((pack / "file_names.json").read_text())
    img_dir = Path(meta["img_dir"])

    rowA = {int(v): i for i, v in enumerate(A["image_index"])}
    rowB = {int(v): i for i, v in enumerate(B["image_index"])}

    panels = []
    for row in sorted(set(rowA) & set(rowB)):
        ia, ib = rowA[row], rowB[row]
        pa, pb = A["box_ptr"], B["box_ptr"]
        xa = A["boxes"][int(pa[ia]):int(pa[ia + 1])]
        xb = B["boxes"][int(pb[ib]):int(pb[ib + 1])]
        assert xa.shape == xb.shape and np.allclose(xa, xb, atol=1e-3), (
            f"row {row}: the two records do not share a box set")
        if len(xa) < a.min_boxes:
            continue
        lab = obj_labels(A, ia)
        ta = triplets(A, ia, a.max_rel, a.rel_frac, cat_names, labels=lab)[:a.show]
        tb = triplets(B, ib, a.max_rel, a.rel_frac, cat_names, labels=lab)[:a.show]
        if len(ta) < a.min_rel or len(tb) < a.min_rel:
            continue
        marks = set()
        for t in ta + tb:
            for e in (t[0], t[2]):
                marks.add(int(e.rsplit("#", 1)[1]))
        if len(marks) > a.max_marks:
            continue
        panels.append(dict(
            row=row, file=file_names[row], n_boxes=int(len(xa)),
            image=render(img_dir / file_names[row], xa, marks),
            a=[list(t) for t in ta], b=[list(t) for t in tb]))
        print(f"  [{len(panels)}] row {row}  {file_names[row]}  "
              f"marks={len(marks)}  {a.a_name}={len(ta)} {a.b_name}={len(tb)}")
        if len(panels) >= a.n:
            break

    out = dict(a_name=a.a_name, b_name=a.b_name, pack=str(pack), n=len(panels),
               selection=(f"first {a.n} images in pack order with >={a.min_boxes} "
                          f"detections, both models emitting >={a.min_rel} relations, "
                          f"and <={a.max_marks} distinct boxes referenced; "
                          f"top {a.show} relations shown per graph"),
               panels=panels)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    mb = Path(a.out).stat().st_size / 1e6
    print(f"wrote {a.out}  panels={len(panels)}  ({mb:.1f} MB)")


if __name__ == "__main__":
    main()
