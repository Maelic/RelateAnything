"""Zero-shot transfer with a REAL detector's own boxes (SGDet-style), not GT
boxes (that's eval_zeroshot.py's SGCls protocol).

Pipeline: reparameterize the trained vocab head to the target dataset's own
predicate vocabulary (same trick as eval_zeroshot.py) → feed the DETECTOR's
own proposed boxes (not GT boxes, not GT box count) to the model → score
predicates → match against GT relations using the union of (a) IoU≥thr
box correspondence and, for the "strict" protocol, (b) the detector's own
predicted class agreeing with the GT class — exactly the two conditions a
real two-stage detector+relation pipeline is judged on in the literature.

GT relations whose endpoints have no valid detector correspondence are kept
in the denominator (mapped to a sentinel box index the model can never
predict a pair for) so they count as permanent misses — matching standard
SGDet recall convention (recall over ALL annotated GT relations, not just
the ones a detector happened to recover). This is why this script does NOT
just reuse eval_detboxes.py's box-substitution trick (which keeps the GT box
count/order and is designed to isolate coordinate noise, not real detector
coverage) — see PLAN.md 2026-07-19 deployment-gap eval for that other use.

Prereq: run training/detect_boxes.py first to produce the detections npz.

Usage:
    python benchmark/eval_zeroshot_detbox.py \
        --checkpoint runs/train/full_v33a_50ep_v3/checkpoint_best.pth \
        --dataset_root runs/packed/vg150 --dataset_name vg150 \
        --det_weights.../BACKBONES/yolo12m_vg150.pt \
        --det runs/detect/yolo12m_vg150_val.npz \
        --protocols lenient,strict
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.checkpoint import build_model_from_ckpt    # noqa: E402
from benchmark.eval_detboxes import cxcywh_to_xyxy, pairwise_iou, greedy_match  # noqa: E402
from relsgg.eval.evaluator import SGClsEvaluator                  # noqa: E402
from relsgg.training.engine import evaluate                     # noqa: E402
from relsgg.data.dataset import TargetList, _box_raster     # noqa: E402
from datagen.build_mask_rasters import mask_raster            # noqa: E402

TRAIN_TEMPLATES = ["{p}", "one object is {p} another object",
                   "a photo of something {p} something"]
SENTINEL = 1 << 16  # never a valid sub/obj idx for any image's box list


def det_class_names(det_weights: str):
    """Detector's own class-name list, in ITS index order (may differ from
    the packed dataset's category order — must remap by name, not index)."""
    import torch as _torch
    ck = _torch.load(det_weights, map_location="cpu", weights_only=False)
    names = ck["model"].names
    return [names[i] for i in range(len(names))]


class DetBoxDataset(Dataset):
    """Per-image: detector's OWN boxes (not GT), GT relations remapped into
    detector-box index space (sentinel index if endpoint unmatched)."""

    def __init__(self, root, det_npz, det_names, iou_thr, det_conf,
                 max_objects, img_size, require_class, class_remap,
                 weight_by_conf=True, split="val", det_masks="", cov_res=32,
                 min_area_frac=0.0):
        self.meta = json.load(open(os.path.join(root, split, "meta.json")))
        self.img_dir = self.meta["img_dir"]
        self.file_names = json.load(open(os.path.join(root, split, "file_names.json")))
        self.img_meta = np.load(os.path.join(root, split, "img_meta.npy"))
        self.boxes = np.load(os.path.join(root, split, "boxes.npy"), mmap_mode="r")
        self.box_cats = np.load(os.path.join(root, split, "box_cats.npy"), mmap_mode="r")
        self.rels = np.load(os.path.join(root, split, "rels.npy"), mmap_mode="r")
        self.max_objects = max_objects
        self.img_size = img_size
        self.require_class = require_class
        self.weight_by_conf = weight_by_conf

        # --rasters equivalent for DETECTOR regions. The pack rasters under
        # runs/sam_masks/rasters are indexed by GT box, so they cannot be used
        # here: a detector box needs the DETECTOR's own mask. detect_boxes.py
        # --save_masks writes one jsonl record per image whose `rles` list is
        # aligned with that image's detections in npz order, so the mapping is
        # the within-image ordinal, carried through the same conf filter and
        # sort as every other column below.
        self.cov_res = cov_res
        self.rles = None
        if det_masks:
            self.rles = {}
            with open(det_masks) as f:
                for line in f:
                    rec = json.loads(line)
                    self.rles[int(rec["idx"])] = rec["rles"]

        d = np.load(det_npz)
        raw_idx = d["img_idx"]
        if self.rles is not None and len(raw_idx):
            assert np.all(np.diff(raw_idx) >= 0), (
                "detections are not grouped by image; the mask jsonl's "
                "per-image RLE order cannot be aligned")
            ordinal = (np.arange(len(raw_idx), dtype=np.int64)
                       - np.searchsorted(raw_idx, raw_idx, side="left"))
        else:
            ordinal = np.zeros(len(raw_idx), dtype=np.int64)
        keep = d["conf"] >= det_conf
        if min_area_frac > 0 and len(raw_idx):
            # A class-agnostic segmenter (FastSAM) returns object PARTS as well
            # as objects — a shirt, a sleeve, a patch of trouser — and at 70+
            # segments an image the confidence ranking is happy to spend the
            # whole budget on them. Area is the cheap, honest discriminator.
            # Default 0.0 leaves every existing evaluator bit-identical.
            wh = self.img_meta[raw_idx][:, 1:3].astype(np.float64)
            b = d["xyxy"].astype(np.float64)
            frac = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
                    / np.maximum(wh[:, 0] * wh[:, 1], 1.0))
            keep &= frac >= min_area_frac
        d_idx, d_xyxy = d["img_idx"][keep], d["xyxy"][keep].astype(np.float32)
        d_conf, d_cls = d["conf"][keep], d["cls"][keep]
        # remap detector class idx -> dataset category idx by name; -1 if absent
        cls_remap_arr = np.array(
            [class_remap.get(det_names[c], -1) for c in range(len(det_names))],
            dtype=np.int64)
        order = np.argsort(d_idx, kind="stable")
        self.d_idx, self.d_xyxy = d_idx[order], d_xyxy[order]
        self.d_conf = d_conf[order]
        self.d_cls_remap = cls_remap_arr[d_cls[order]]
        self.d_ord = ordinal[keep][order]
        n_img = len(self.file_names)
        self.starts = np.searchsorted(self.d_idx, np.arange(n_img + 1))
        self.iou_thr = iou_thr

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.img_dir, self.file_names[i])).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        image = torch.from_numpy(
            np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0)

        s0, s1 = self.starts[i], self.starts[i + 1]
        # top-confidence detections, capped at max_objects
        conf = self.d_conf[s0:s1]
        keep_n = min(self.max_objects, len(conf))
        top = np.argsort(-conf)[:keep_n]
        det_xyxy_px = self.d_xyxy[s0:s1][top]
        det_cls = self.d_cls_remap[s0:s1][top]
        det_scores = conf[top].astype(np.float32)
        n_det = len(det_xyxy_px)

        _, W, H, b0, nb, r0, nr = self.img_meta[i]
        nb = min(int(nb), 400)
        det_xyxy = det_xyxy_px / np.array([W, H, W, H], np.float32)
        det_cxcywh = np.zeros((max(n_det, 1), 4), np.float32)
        if n_det:
            x0, y0, x1, y1 = det_xyxy[:, 0], det_xyxy[:, 1], det_xyxy[:, 2], det_xyxy[:, 3]
            det_cxcywh[:n_det] = np.stack(
                [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], -1)
        boxes = torch.from_numpy(det_cxcywh[:max(n_det, 1)]).float()
        n_boxes = n_det if n_det else 1  # keep collate happy on an empty image

        # GT boxes/relations for this image
        gt_cxcywh = np.array(self.boxes[b0:b0 + nb], dtype=np.float32)
        gt_cats = np.array(self.box_cats[b0:b0 + nb], dtype=np.int64)
        gt_xyxy = cxcywh_to_xyxy(gt_cxcywh)
        rels = np.array(self.rels[r0:r0 + nr], dtype=np.int64)
        rels = rels[(rels[:, 0] < nb) & (rels[:, 1] < nb)] if rels.size else rels

        if n_det and nb:
            iou = pairwise_iou(gt_xyxy, det_xyxy)
            m = greedy_match(iou, self.iou_thr)  # det idx per GT box, -1 if none
            if self.require_class:
                class_ok = np.zeros(nb, dtype=bool)
                hit = m >= 0
                class_ok[hit] = det_cls[m[hit]] == gt_cats[hit]
                m = np.where(class_ok, m, -1)
        else:
            m = np.full(nb, -1, dtype=np.int64)

        if rels.size:
            sub_map = np.where(m[rels[:, 0]] >= 0, m[rels[:, 0]], SENTINEL)
            obj_map = np.where(m[rels[:, 1]] >= 0, m[rels[:, 1]], SENTINEL)
            relations = torch.from_numpy(
                np.stack([sub_map, obj_map, rels[:, 2]], axis=1).astype(np.int64))
        else:
            relations = torch.zeros((0, 3), dtype=torch.long)

        target = {"relations": relations}
        if self.rles is not None:
            from pycocotools import mask as mask_util
            g = self.cov_res
            ords = self.d_ord[s0:s1][top]
            rles_i = self.rles.get(int(i)) or []
            n_slot = max(n_det, 1)
            cov = np.empty((n_slot, g, g), np.uint8)
            fill = np.ones(n_slot, np.float32)
            for j in range(n_slot):
                m = None
                if j < n_det and int(ords[j]) < len(rles_i):
                    rle = rles_i[int(ords[j])]
                    if rle is not None:
                        r = dict(rle)
                        if isinstance(r["counts"], str):
                            r["counts"] = r["counts"].encode()
                        m = mask_util.decode(r)
                        if m.ndim == 3:
                            m = m[..., 0]
                        if not m.any():
                            m = None
                if m is None:
                    # No mask for this detection (or the dummy slot of an image
                    # with no detections): rasterize the BOX, exactly what the
                    # training loader does for a missing sidecar row, so the
                    # region is still correct rather than absent.
                    if j < n_det:
                        x0, y0, x1, y1 = (float(v) for v in det_xyxy[j])
                    else:
                        x0, y0, x1, y1 = 0.0, 0.0, 1.0, 1.0
                    cov[j] = np.clip(_box_raster(x0, y0, x1, y1, g) * 255.0,
                                     0, 255).astype(np.uint8)
                else:
                    cov[j] = np.clip(mask_raster(m.astype(np.float64), g) * 255.0,
                                     0, 255).astype(np.uint8)
                    bw = float(det_xyxy_px[j, 2] - det_xyxy_px[j, 0])
                    bh = float(det_xyxy_px[j, 3] - det_xyxy_px[j, 1])
                    fill[j] = float(m.sum()) / max(bw * bh, 1.0)
            target["cov"] = torch.from_numpy(cov)
            target["fill"] = torch.from_numpy(fill)
            target["mode"] = 1.0
            # Which sidecar RLE each kept detection came from, in the model's own
            # slot order. Carried so a visualiser can draw the SAME masks the model
            # pooled over instead of re-deriving the conf filter and top-k sort and
            # risking an off-by-one against them. Inert for evaluators.
            target["det_rle_ord"] = torch.from_numpy(
                np.ascontiguousarray(ords[:n_det]).astype(np.int64) if n_det
                else np.full(1, -1, np.int64))
        # Carried for the interchange EXPORT path only (score_native_protocol.py
        # needs the detector's own boxes and class labels to build triplets the way
        # OvSGTR's `_triplet` does). Evaluators ignore keys they do not read, so
        # this is inert for every normal eval.
        target["det_boxes_px"] = torch.from_numpy(
            np.ascontiguousarray(det_xyxy_px[:n_det]) if n_det
            else np.zeros((1, 4), np.float32))
        target["det_labels"] = torch.from_numpy(
            np.ascontiguousarray(det_cls[:n_det]) if n_det
            else np.full(1, -1, np.int64))
        if self.weight_by_conf:
            box_scores = np.zeros(max(n_det, 1), np.float32)
            if n_det:
                box_scores[:n_det] = det_scores
            target["box_scores"] = torch.from_numpy(box_scores)
        return image, boxes, target, n_boxes


def collate(batch):
    images = torch.stack([b[0] for b in batch])
    box_counts = torch.tensor([b[3] for b in batch], dtype=torch.long)
    max_n = max(1, int(box_counts.max()))
    boxes = torch.zeros(len(batch), max_n, 4, dtype=torch.float32)
    for i, (_, bx, _, n) in enumerate(batch):
        boxes[i,:n] = bx[:n]
    targets = TargetList(b[2] for b in batch)
    if batch and "cov" in batch[0][2]:
        g = batch[0][2]["cov"].shape[-1]
        cov = torch.zeros(len(batch), max_n, g, g, dtype=torch.uint8)
        fill = torch.ones(len(batch), max_n, dtype=torch.float32)
        for i, b in enumerate(batch):
            n = b[2]["cov"].shape[0]
            cov[i,:n] = b[2]["cov"]
            fill[i,:n] = b[2]["fill"]
        targets.cov, targets.fill = cov, fill
        targets.mode = torch.tensor([float(b[2].get("mode", 1.0)) for b in batch])
    return images, boxes, box_counts, targets


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--keep_W", action="store_true",
                   help="Keep the checkpoint's own W when the target vocabulary "
                        "equals the checkpoint's (closed-set fine-tunes).")
    p.add_argument("--dataset_root", required=True)
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--split", default="val",
                   help="pack split to evaluate; the benchmark protocols use 'test'")
    p.add_argument("--det_weights", default="",
                   help="only needed for --det_vocab weights (to read the baked class names)")
    p.add_argument("--det_vocab", default="weights", choices=["weights", "pack"],
                   help="'pack' for detections made with detect_boxes.py --set_classes, "
                        "whose cls already indexes the pack's categories")
    p.add_argument("--det", required=True, help="detections.npz from detect_boxes.py")
    p.add_argument("--det_masks", default="",
                   help="instance-mask RLE jsonl from detect_boxes.py --save_masks, "
                        "aligned with --det. Given, the head reads the DETECTOR's "
                        "masks as its regions instead of its boxes: the deployment "
                        "cell of paper2 s7 (detector boxes + detector masks). "
                        "Detections without a mask fall back to their box raster.")
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--score_mode", default="softmax", choices=["sigmoid", "softmax"],
                   help="softmax matches the literature SGDet convention")
    p.add_argument("--text_student", default=None,
                   help="Student text-encoder ckpt for reparameterization. "
                        "Defaults to the one the checkpoint names.")
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--det_conf", type=float, default=0.10)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--geo_budget", type=int, default=0,
                   help="override the sampler's stage-1 geometry budget (0 = leave as "
                        "trained). This is the REAL pair cap; --eval_budget is clamped "
                        "to it, so raising eval_budget alone does nothing.")
    p.add_argument("--max_objects", type=int, default=60)
    p.add_argument("--protocols", default="lenient,strict")
    p.add_argument("--no_graph_constraint", action="store_true",
                   help="Emit every (pair, predicate) cell instead of one "
                        "predicate per pair. OFF by default: unconstrained R@K "
                        "is inflated 12-19 points and is not comparable to any "
                        "GT-box number we report. Only for reproducing the "
                        "pre-2026-07-31 detbox numbers.")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()

    meta = json.load(open(os.path.join(args.dataset_root, args.split, "meta.json")))
    pred_names = meta["predicates"]
    cat_names = meta["categories"]
    class_remap = {n: i for i, n in enumerate(cat_names)}

    print(f"[{args.dataset_name}] {len(pred_names)} predicates, {len(cat_names)} "
          f"categories — reparameterizing vocab head")
    # The vocabulary has to be encoded by the encoder the head was trained
    # against: a head scored against a different text space measures nothing.
    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    text_student = (args.text_student if args.text_student is not None
                    else ck_args.get("text_student") or "")
    if args.keep_W and list(ckpt.get("pred_names") or []) == list(pred_names):
        print(f"[{args.dataset_name}] --keep_W: vocabulary matches the checkpoint's "
              f"{len(pred_names)} predicates; keeping its W")
        model.vocab_head.pred_names = list(pred_names)
    elif text_student:
        print(f"[{args.dataset_name}] vocabulary encoder: STUDENT ({text_student})")
        from relsgg.text.student import encode_texts_student
        E = encode_texts_student(pred_names, text_student,
                                 templates=TRAIN_TEMPLATES, device=device)
        model.vocab_head.set_vocabulary_matrix(pred_names, E)
    else:
        raise SystemExit(
            "this checkpoint names no text student. The vocabulary has to be "
            "encoded by the encoder the head was trained against; pass "
            "--text_student, or use a released model, which ships its own.")
    model.reparameterize()

    if args.geo_budget > 0:
        # evaluate() sets final_budget = min(eval_budget, sampler.geo_budget), so
        # geo_budget (default 400) is the REAL cap and raising --eval_budget alone is a
        # no-op. Stage 1 of the sampler is a geometry filter, so widening it at eval
        # time changes only how aggressively pairs are pruned, not any learned weight —
        # needed to compare pair-for-pair against models that score all N*(N-1) pairs.
        raw = model.module if hasattr(model, "module") else model
        print(f"[{args.dataset_name}] sampler geo_budget "
              f"{raw.sampler.geo_budget} -> {args.geo_budget}")
        raw.sampler.geo_budget = args.geo_budget

    if args.det_vocab == "pack":
        # Detections produced by detect_boxes.py --set_classes: the detector was
        # PROMPTED with this pack's categories at test time, so `cls` already indexes
        # cat_names. Reading names from the checkpoint would return the weights' baked
        # vocabulary (e.g. 80 COCO names for yolov8m-worldv2) and silently remap almost
        # every detection to -1.
        d_names = list(cat_names)
    else:
        d_names = det_class_names(args.det_weights)
    n_absent = sum(1 for n in d_names if n not in class_remap)
    print(f"[{args.dataset_name}] detector classes: {len(d_names)} (vocab={args.det_vocab}) "
          f"(name mismatch vs pack categories: {n_absent})")

    eval_args = type("A", (), {"amp": device.type == "cuda",
                               "amp_dtype_t": torch.bfloat16})()

    results = {"checkpoint": args.checkpoint, "detector": args.det_weights}
    for proto in args.protocols.split(","):
        proto = proto.strip()
        require_class = proto == "strict"
        ds = DetBoxDataset(args.dataset_root, args.det, d_names, args.iou_thr,
                          args.det_conf, args.max_objects, args.img_size,
                          require_class, class_remap, split=args.split,
                          det_masks=args.det_masks)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=args.num_workers,
                            pin_memory=True)
        # Graph-constrained by default, matching the ground-truth-box
        # evaluation. Mixing the two conventions inflates the detector side by
        # 12-19 points and understates the deployment gap.
        ev = SGClsEvaluator(topk=[20, 50, 100], num_predicates=len(pred_names),
                            score_mode=args.score_mode,
                            graph_constraint=not args.no_graph_constraint)
        with torch.no_grad():
            metrics = evaluate(model, loader, device, eval_args, ev,
                               eval_budget=args.eval_budget)
        results[proto] = metrics
        line = "  ".join(f"{k}={metrics[k]:.4f}" for k in
                         ["R@20", "R@50", "R@100", "mR@20", "mR@50", "mR@100"]
                         if k in metrics)
        print(f"[{args.dataset_name}/{proto}] {line}", flush=True)

    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint), f"zeroshot_detbox_{args.dataset_name}.json")
    json.dump(results, open(out_path, "w"), indent=2, default=float)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
