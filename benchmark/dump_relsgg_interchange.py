"""Emit RelSGG predictions in the common interchange format.

Until now our model's predictions were scored inline and never materialised, while
OvSGTR's went through `benchmark/ovsgtr/run_ovsgtr_pack.py`. For the LLM-oracle axis both
models must be rendered and judged by IDENTICAL code, so our side needs the same record.

Reuses `DetBoxDataset` / the same forward path `eval_zeroshot_detbox.py` scores, so the
dumped predictions are exactly the ones behind the reported metrics — not a
re-derivation that could drift.

    python benchmark/dump_relsgg_interchange.py \
        --checkpoint runs/train/relsgg-vits16plus/model.pth \
        --dataset_root runs/packed/psg --split test --dataset_name psg_test \
        --det runs/detect/yoloworld_ov_psg_test.npz --det_vocab pack \
        --out runs/judge/relsgg_psg_test_yoloworld.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from relsgg.checkpoint import build_model_from_ckpt           # noqa: E402
# DetBoxDataset yields 4-tuples (image, boxes, target, n_boxes); it needs the collate
# that ships beside it, NOT data.collate_fn (3-tuples, and a different return order).
from benchmark.eval_zeroshot_detbox import (DetBoxDataset, TRAIN_TEMPLATES,   # noqa: E402
                                  det_class_names, collate)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--det", required=True)
    p.add_argument("--det_vocab", default="pack", choices=["weights", "pack"])
    p.add_argument("--det_weights", default="")
    p.add_argument("--det_conf", type=float, default=0.05)
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--max_objects", type=int, default=60)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--vocab", default="pack",
                   help="'pack' = reparameterize to the benchmark's predicates; "
                        "'train' = keep the FULL training vocabulary (open-vocab arm)")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--emit", default="full", choices=["full", "topk"],
                   help="'full' stores every pair x V score matrix (only viable for a "
                        "small V). 'topk' stores the K best triplets per image — "
                        "required for the open-vocabulary arm, where V~19k would make "
                        "the full matrix tens of GB.")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, a.weights).to(device).eval()

    meta = json.load(open(os.path.join(a.dataset_root, a.split, "meta.json")))
    cat_names = meta["categories"]
    class_remap = {n: i for i, n in enumerate(cat_names)}

    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    ts = ck_args.get("text_student") or ""
    if not ts:
        raise SystemExit("checkpoint not trained in student text space")
    from relsgg.text.student import encode_texts_student

    if a.vocab == "pack":
        pred_names = list(meta["predicates"])
    else:
        # Open-vocabulary arm: deploy the full training vocabulary, so the graph the
        # model actually emits in the wild is what gets judged.
        npz = np.load(ck_args["pred_embeds"], allow_pickle=False)
        pred_names = [str(x) for x in npz["predicates"]]
    E = encode_texts_student(pred_names, ts, templates=TRAIN_TEMPLATES, device=device)
    model.vocab_head.set_vocabulary_matrix(pred_names, E)
    model.reparameterize()
    print(f"[{a.dataset_name}] vocabulary: {len(pred_names)} predicates ({a.vocab})")

    d_names = list(cat_names) if a.det_vocab == "pack" else det_class_names(a.det_weights)
    ds = DetBoxDataset(a.dataset_root, a.det, d_names, a.iou_thr, a.det_conf,
                       a.max_objects, a.img_size, False, class_remap, split=a.split)
    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=a.num_workers,
                        pin_memory=True)

    raw = model.module if hasattr(model, "module") else model
    raw.sampler.final_budget = min(a.eval_budget, raw.sampler.geo_budget)

    # Detector boxes in pixel xyxy, so the record is renderable without the dataset.
    # Detector boxes must be filtered and ordered EXACTLY as DetBoxDataset does, or the
    # box indices in `pairs` would not address the boxes we store.
    det = np.load(a.det)
    keep = det["conf"] >= a.det_conf
    d_idx, d_xyxy = det["img_idx"][keep], det["xyxy"][keep].astype(np.float32)
    d_conf, d_cls = det["conf"][keep], det["cls"][keep].astype(np.int64)
    order = np.argsort(d_idx, kind="stable")
    d_idx, d_xyxy, d_conf, d_cls = d_idx[order], d_xyxy[order], d_conf[order], d_cls[order]
    #...and labels must land in PACK category space, so a renderer can name them.
    remap = np.array([class_remap.get(n, -1) for n in d_names], dtype=np.int64)
    d_cls = remap[d_cls]
    starts = np.searchsorted(d_idx, np.arange(len(ds) + 1))

    V = len(pred_names)
    out_index, out_pairs, out_scores = [], [], []
    out_boxes, out_labels, out_bscores = [], [], []
    pair_ptr, box_ptr = [0], [0]
    n_done = 0

    with torch.no_grad():
        for bi, (images, boxes, box_counts, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            boxes = boxes.to(device, non_blocking=True)
            box_counts = box_counts.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda",
                                    dtype=torch.bfloat16):
                out = model(images, boxes, box_counts, targets=None)

            lg = out["logits"].float()
            if out.get("pair_logits") is not None:
                lg = lg + out["pair_logits"].float().unsqueeze(-1)
            probs = torch.sigmoid(lg)              # deploy contract
            sub, obj, vm = out["sub_idx"], out["obj_idx"], out["valid_mask"]

            for j in range(lg.shape[0]):
                i = n_done + j
                m = vm[j]
                k = int(m.sum())
                s0, s1 = int(starts[i]), int(starts[i + 1])
                conf = d_conf[s0:s1]
                topn = min(a.max_objects, len(conf))
                top = np.argsort(-conf)[:topn]

                out_index.append(i)
                if k:
                    pj = probs[j][m]                       # [k, V]
                    sj = sub[j][m].cpu().numpy()
                    oj = obj[j][m].cpu().numpy()
                    if a.emit == "topk":
                        # Keep only each pair's best predicate, then the K best pairs.
                        # This is exactly what the renderer consumes, and it keeps the
                        # open-vocab arm (V~19k) from exploding to tens of GB.
                        bs, ba_ = pj.max(1)
                        sel = torch.argsort(bs, descending=True)[:a.topk].cpu().numpy()
                        oh = np.zeros((len(sel), 2), np.float32)
                        oh[:, 0] = ba_.cpu().numpy()[sel]          # predicate id
                        oh[:, 1] = bs.cpu().numpy()[sel]           # score
                        out_pairs.append(np.stack([sj[sel], oj[sel]], 1).astype(np.int32))
                        out_scores.append(oh.astype(np.float32))
                        k = len(sel)
                    else:
                        out_pairs.append(np.stack([sj, oj], 1).astype(np.int32))
                        out_scores.append(pj.cpu().numpy().astype(np.float16))
                out_boxes.append(d_xyxy[s0:s1][top])
                out_labels.append(d_cls[s0:s1][top].astype(np.int32))
                out_bscores.append(conf[top].astype(np.float32))
                pair_ptr.append(pair_ptr[-1] + k)
                box_ptr.append(box_ptr[-1] + topn)
            n_done += lg.shape[0]
            if a.limit and n_done >= a.limit:
                break
            if bi % 20 == 0:
                print(f"  {n_done}/{len(ds)}", flush=True)

    def cat(chunks, shape, dtype):
        return np.concatenate(chunks, 0) if chunks else np.zeros(shape, dtype)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(
        a.out,
        image_index=np.asarray(out_index, np.int32),
        pair_ptr=np.asarray(pair_ptr, np.int64),
        box_ptr=np.asarray(box_ptr, np.int64),
        pairs=cat(out_pairs, (0, 2), np.int32),
        rel_scores=cat(out_scores, (0, 2 if a.emit == "topk" else V),
                       np.float32 if a.emit == "topk" else np.float16),
        boxes=cat(out_boxes, (0, 4), np.float32),
        labels=cat(out_labels, (0,), np.int32),
        box_scores=cat(out_bscores, (0,), np.float32),
        predicates=np.asarray(pred_names),
        categories=np.asarray(cat_names),
        meta=np.asarray([json.dumps({
            "model": "RelSGG", "checkpoint": a.checkpoint,
            "pack": f"{a.dataset_root}/{a.split}", "box_source": "external",
            "det": a.det, "det_conf": a.det_conf, "vocab": a.vocab,
            "score_semantics": "sigmoid", "bg_column": -1, "emit": a.emit,
            "topk": a.topk if a.emit == "topk" else None,
            "n_images": len(out_index),
        })]),
)
    print(f"wrote {a.out}  images={len(out_index)} pairs={pair_ptr[-1]}")


if __name__ == "__main__":
    main()
