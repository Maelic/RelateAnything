"""Convert OvSGTR's OWN predicted boxes into our detect-npz format.

Removes the last confound in the OvR-SGG comparison. Until now our row used a
YOLO detector and theirs used GroundingDINO, so a recall gap could always be
blamed on the detector. Feeding OUR relation head THEIR boxes makes the two
rows differ in the relation model alone.

Input: the npz written by benchmark/ovsgtr/run_ovsgtr_pack.py --boxes det
        (their native SGDet: GroundingDINO predicts boxes, labels, scores)
Output: runs/detect/<name>.npz with img_idx / xyxy / conf / cls / n_images,
        readable by eval_zeroshot_detbox.py and eval_ovsgtr_novel.py with
        --det_vocab pack.

Their postprocessor reserves category index 0 for __background__, so stored
labels are 1-BASED against `categories` (meta.label_base). We subtract it, and
we REFUSE to guess: if label_base is absent the script fails rather than
silently shifting every object class by one (the bug that once handed the LLM
judge a fake 99% win rate).

    python benchmark/ovsgtr/ovsgtr_boxes_to_detnpz.py \
        --pred runs/ovsgtr/ovr_swint_vg150_test_det.npz \
        --pack runs/packed/vg150/test \
        --out runs/detect/ovsgtr_ovr_swint_vg150_test.npz
"""
from __future__ import annotations
import argparse, json, os
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--pack", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    d = np.load(a.pred, allow_pickle=True)
    meta = json.loads(str(d["meta"][0]))
    if meta.get("box_source") != "det":
        raise SystemExit(f"--pred was produced with box_source={meta.get('box_source')!r}; "
                         "this converter is only meaningful for 'det' (their own boxes)")
    base = meta.get("label_base")
    if base is None:
        raise SystemExit("meta.label_base missing — refusing to guess the label origin")

    pack = json.load(open(os.path.join(a.pack, "meta.json")))
    cats_pack = list(pack["categories"])
    cats_pred = [str(x) for x in d["categories"]]
    if cats_pred != cats_pack:
        raise SystemExit("category lists differ between prediction npz and pack; "
                         "cls would index the wrong space")

    n_img = int(pack["num_images"])
    idx, ptr = d["image_index"], d["box_ptr"]
    boxes, labels, scores = d["boxes"], d["labels"], d["box_scores"]

    img_idx = np.concatenate([np.full(ptr[i + 1] - ptr[i], idx[i], dtype=np.int32)
                              for i in range(len(idx))]) if len(idx) else np.zeros(0, np.int32)
    cls = labels.astype(np.int64) - int(base)
    keep = cls >= 0                      # background-labelled boxes carry no category
    if (~keep).any():
        print(f"dropping {int((~keep).sum())} background-labelled boxes")
    if cls[keep].max(initial=-1) >= len(cats_pack):
        raise SystemExit("label out of range after subtracting label_base")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    np.savez_compressed(a.out,
                        img_idx=img_idx[keep].astype(np.int32),
                        xyxy=boxes[keep].astype(np.float32),
                        conf=scores[keep].astype(np.float32),
                        cls=cls[keep].astype(np.int32),
                        n_images=np.int64(n_img))
    print(f"wrote {a.out}: {int(keep.sum()):,} boxes over {n_img:,} images "
          f"({keep.sum()/max(n_img,1):.1f}/img), conf "
          f"[{scores[keep].min():.3f}, {scores[keep].max():.3f}]")
    print(f"  source: {meta.get('checkpoint')}  label_base={base}")


if __name__ == "__main__":
    main()
