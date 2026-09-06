"""OvSGTR-style open-vocabulary RELATION (OvR-SGG) evaluation on VG150.

Reproduces the base/novel relation split from OvSGTR (Chen et al., ECCV'24,
"Expanding Scene Graph Boundaries", arXiv:2311.10988). Their OvR-SGG setting
holds out 15 of the 50 VG150 predicates as "novel" (never seen with a label
during training) and reports Recall@K on the Novel subset and on Base+Novel,
under the SGDET protocol (detector boxes).

Split source: OvSGTR repo datasets/vg.py (VG150_NOVEL_PREDICATE /
VG150_BASE_PREDICATE). Reproduced verbatim below.

IMPORTANT interpretive note (see PLAN.md critique): for OvSGTR the split is
meaningful because they TRAIN on VG150 base relations then test on novel. For
*our* model the split is cosmetic — we never train on VG150 at all (not base,
not novel; not even VG150 images), so all 50 predicates are equally zero-shot.
We compute the identical numbers only to place our model on the same axis as
their table. The novel subset is also 93% just {on, of, in} by instance count,
so "Novel R@K" (micro) mostly measures the three easiest head predicates.

Two box regimes are reported:
  * gtbox  — oracle GT boxes (upper bound; ~PredCls/SGCls, NOT OvSGTR's number)
  * detbox — a real detector's own boxes (SGDET; the OvSGTR-comparable number),
             lenient (IoU only) and strict (IoU + detector class == GT class)

For each we report, per subset {novel, base, all}:
  * R@K   micro       = sum_c tp_c / sum_c gt_c        (instance recall)
  * R@K   image-macro = mean over images of per-image subset recall (SGG conv.)
  * mR@K  macro       = mean_c (tp_c / gt_c)           (mean recall over classes)

Usage (offline compute node, detections already cached by detect_boxes.py):
    python benchmark/eval_ovsgtr_novel.py \
        --checkpoint runs/train/full_v33a_50ep_v3/checkpoint_best.pth \
        --dataset_root runs/packed/vg150 \
        --det_weights.../BACKBONES/yolo12m_vg150.pt \
        --det runs/detect/yolo12m_vg150_val.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                      # noqa: E402
from relsgg.eval.evaluator import SGClsEvaluator                       # noqa: E402
from relsgg.training.engine import evaluate                          # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402
from benchmark.eval_zeroshot_detbox import DetBoxDataset, collate, det_class_names  # noqa: E402

# --- OvSGTR VG150 OvR split (verbatim from their datasets/vg.py) -------------
VG150_NOVEL_PREDICATE = [
    "belonging to", "part of", "riding", "walking in", "in", "of",
    "painted on", "playing", "for", "walking on", "says", "attached to",
    "eating", "on", "wears",
]
VG150_BASE_PREDICATE = [
    "between", "to", "made of", "looking at", "along", "laying on", "using",
    "carrying", "against", "mounted on", "sitting on", "flying in", "covering",
    "from", "over", "near", "hanging from", "across", "at", "above",
    "watching", "covered in", "wearing", "holding", "and", "standing on",
    "lying on", "growing on", "under", "on back of", "with", "has",
    "in front of", "behind", "parked on",
]

# OvSGTR published SGDET Novel(Relation) / Base+Novel(Relation) R@50/R@100
OVSGTR_PUBLISHED = {
    "SwinT (no distill)":  {"novel_R@50": 0.34,  "novel_R@100": 0.41},
    "SwinT (distill)":     {"novel_R@50": 13.45, "novel_R@100": 16.19,
                            "all_R@50": 20.46, "all_R@100": 23.86},
    "SwinB (distill)":     {"novel_R@50": 16.39, "novel_R@100": 19.72,
                            "all_R@50": 22.89, "all_R@100": 26.65},
}


def subset_breakdown(ev, pred_names, novel_ids, base_ids, topk):
    """Micro + macro base/novel/all recall from the evaluator's tp/gt counts."""
    out = {}
    for k in topk:
        rows = {r["class_id"]: r for r in ev.compute_per_class(k, pred_names)}
        for name, ids in (("novel", novel_ids), ("base", base_ids),
                          ("all", novel_ids + base_ids)):
            tp = sum(rows[c]["tp"] for c in ids if c in rows)
            gt = sum(rows[c]["gt"] for c in ids if c in rows)
            recs = [rows[c]["recall"] for c in ids if c in rows and rows[c]["gt"] > 0]
            out[f"{name}_R@{k}_micro"] = tp / gt if gt else 0.0
            out[f"{name}_mR@{k}_macro"] = float(np.mean(recs)) if recs else 0.0
            out[f"{name}_gt@{k}"] = gt
    # Per-class breakdown of the 15 novel predicates at @50. The literature reports
    # only micro R@K on this subset, where on/of/in carry 92.6% of the mass -- so a
    # strong Novel R@K is compatible with recovering nothing else. This is the column
    # that says which of the 15 actually work.
    rows50 = {r["class_id"]: r for r in ev.compute_per_class(50, pred_names)}
    out["novel_per_class@50"] = {
        rows50[c]["name"]: {"tp": rows50[c]["tp"], "gt": rows50[c]["gt"],
                            "recall": rows50[c]["recall"]}
        for c in novel_ids if c in rows50}
    return out


def run_gtbox(model, root, pred_names, novel_ids, base_ids, args, device, eval_args):
    ds = RelationDataset(root=root, split=args.split, resolution=args.img_size,
                         max_objects=args.max_objects)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers,
                        pin_memory=True)
    ev = SGClsEvaluator(topk=[20, 50, 100], num_predicates=len(pred_names),
                        score_mode=args.score_mode,
                        graph_constraint=not args.no_graph_constraint,
                        subsets={"novel": novel_ids, "base": base_ids})
    metrics = evaluate(model, loader, device, eval_args, ev,
                       eval_budget=args.eval_budget)
    metrics.update(subset_breakdown(ev, pred_names, novel_ids, base_ids,
                                    [20, 50, 100]))
    return metrics


def run_detbox(model, args, pred_names, novel_ids, base_ids, proto, device, eval_args):
    meta = json.load(open(os.path.join(args.dataset_root, args.split, "meta.json")))
    class_remap = {n: i for i, n in enumerate(meta["categories"])}
    # --det_vocab pack: detect_boxes.py --set_classes PROMPTED the detector with this
    # pack's categories, so `cls` already indexes them. Reading the baked names off an
    # open-vocab checkpoint instead returns its pretraining vocabulary (80 COCO names
    # for yolov8m-worldv2) and silently remaps almost every detection to -1.
    d_names = (list(meta["categories"]) if args.det_vocab == "pack"
               else det_class_names(args.det_weights))
    ds = DetBoxDataset(args.dataset_root, args.det, d_names, args.iou_thr,
                       args.det_conf, args.det_max_objects, args.img_size,
                       require_class=(proto == "strict"), class_remap=class_remap,
                       weight_by_conf=not args.no_conf_weight, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate, num_workers=args.num_workers,
                        pin_memory=True)
    ev = SGClsEvaluator(topk=[20, 50, 100], num_predicates=len(pred_names),
                        score_mode=args.det_score_mode,
                        graph_constraint=not args.no_graph_constraint,
                        subsets={"novel": novel_ids, "base": base_ids})
    metrics = evaluate(model, loader, device, eval_args, ev,
                       eval_budget=args.eval_budget)
    metrics.update(subset_breakdown(ev, pred_names, novel_ids, base_ids,
                                    [20, 50, 100]))
    return metrics


def print_table(tag, m):
    print(f"\n=== {tag} ===")
    print(f"{'subset':<7} {'R@50 micro':>11} {'R@50 imac':>10} {'mR@50 mac':>10} "
          f"{'R@100 micro':>12} {'gt@50':>8}")
    for s in ("novel", "base", "all"):
        imac = m.get(f"{s}_R@50", float('nan')) if s != "all" else m.get("R@50", float('nan'))
        print(f"{s:<7} {m[f'{s}_R@50_micro']*100:>10.2f}% {imac*100:>9.2f}% "
              f"{m[f'{s}_mR@50_macro']*100:>9.2f}% {m[f'{s}_R@100_micro']*100:>11.2f}% "
              f"{m[f'{s}_gt@50']:>8}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset_root", default="runs/packed/vg150")
    p.add_argument("--det_weights", default="",
                   help="only needed for --det_vocab weights (to read baked class names)")
    p.add_argument("--det_vocab", default="weights", choices=["weights", "pack"],
                   help="'pack' when the detector was PROMPTED with the pack's "
                        "categories (open-vocab detectors via --set_classes)")
    p.add_argument("--split", default="test",
                   help="VG150 split. OvSGTR reports TEST; default is test here.")
    p.add_argument("--text_student", default=None,
                   help="Student text-encoder ckpt. Defaults to whatever the "
                        "checkpoint names.")
    p.add_argument("--no_graph_constraint", action="store_true",
                   help="Emit every (pair, predicate) cell instead of one predicate "
                        "per pair. OFF by default: OvSGTR's published R@K IS graph-"
                        "constrained (their sgg_metrics.py builds exactly one triplet "
                        "per pair and never reads multiple_preds), and unconstrained "
                        "R@K runs 12-19 points higher, so leaving this on would make "
                        "every comparison to their table invalid.")
    p.add_argument("--det", required=True)
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--score_mode", default="sigmoid", choices=["sigmoid", "softmax"],
                   help="gtbox scoring. sigmoid matches every other cell this project "
                        "reports (synonym-trained head: softmax over a synonym-sharing "
                        "vocabulary deflates each synonym's probability)")
    p.add_argument("--det_score_mode", default="sigmoid", choices=["sigmoid", "softmax"])
    p.add_argument("--iou_thr", type=float, default=0.5)
    p.add_argument("--det_conf", type=float, default=0.10)
    p.add_argument("--det_conf_sweep", default="",
                   help="comma list of det_conf values; detbox eval loops over "
                        "them reusing the reparameterized model (find the "
                        "achieved-recall-vs-threshold peak cheaply).")
    p.add_argument("--no_conf_weight", action="store_true",
                   help="disable SGDet triplet scoring conf(sub)*conf(obj)*"
                        "pred (ablation: rank by predicate score only).")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--geo_budget", type=int, default=0,
                   help="override the sampler's stage-1 geometry budget (0 = as trained, "
                        "400). This is the REAL pair cap: evaluate() sets final_budget = "
                        "min(eval_budget, sampler.geo_budget), so raising --eval_budget "
                        "alone is a NO-OP. Matters here because OvSGTR scores all N(N-1) "
                        "pairs (~9.9k/img on its own ~100 boxes) while we keep 400.")
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--det_max_objects", type=int, default=60)
    p.add_argument("--modes", default="gtbox,detbox")
    p.add_argument("--out", default="")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch')})")
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()

    meta = json.load(open(os.path.join(args.dataset_root, args.split, "meta.json")))
    pred_names = meta["predicates"]
    name2id = {n: i for i, n in enumerate(pred_names)}
    novel_ids = [name2id[n] for n in VG150_NOVEL_PREDICATE]
    base_ids = [name2id[n] for n in VG150_BASE_PREDICATE]
    assert len(novel_ids) == 15 and len(base_ids) == 35, "split/vocab mismatch"

    # reparameterize to ALL 50 VG150 predicates (base+novel) — identical to
    # OvSGTR's inference vocabulary in the OvR setting.
    # The vocabulary has to be encoded by the encoder the head was trained
    # against: a head scored against a different text space measures nothing.
    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    text_student = (args.text_student if args.text_student is not None
                    else ck_args.get("text_student") or "")
    print(f"reparameterizing vocab head to {len(pred_names)} VG150 predicates")
    if text_student:
        print(f"  vocabulary encoder: STUDENT ({text_student})")
        from relsgg.text.student import encode_texts_student
        E = encode_texts_student(pred_names, text_student,
                                 templates=TRAIN_TEMPLATES, device=device)
        model.vocab_head.set_vocabulary_matrix(pred_names, E)
    else:
        print("  vocabulary encoder: dino.txt TEACHER")
        raise SystemExit(
            "this checkpoint names no text student. The vocabulary has to be "
            "encoded by the encoder the head was trained against; pass "
            "--text_student, or use a released model, which ships its own.")
    model.reparameterize()

    if args.geo_budget > 0:
        raw = model.module if hasattr(model, "module") else model
        print(f"sampler geo_budget {raw.sampler.geo_budget} -> {args.geo_budget}")
        raw.sampler.geo_budget = args.geo_budget

    eval_args = type("A", (), {"amp": device.type == "cuda",
                               "amp_dtype_t": torch.bfloat16})()
    results = {"checkpoint": args.checkpoint, "detector": args.det_weights,
               "det_npz": args.det, "det_vocab": args.det_vocab,
               "data_split": args.split, "score_mode": args.score_mode,
               "det_score_mode": args.det_score_mode,
               "graph_constraint": not args.no_graph_constraint,
               "text_student": text_student,
               "split": {"novel": VG150_NOVEL_PREDICATE, "base": VG150_BASE_PREDICATE},
               "ovsgtr_published": OVSGTR_PUBLISHED, "modes": {}}

    modes = [m.strip() for m in args.modes.split(",")]
    if "gtbox" in modes:
        m = run_gtbox(model, args.dataset_root, pred_names, novel_ids, base_ids,
                      args, device, eval_args)
        results["modes"]["gtbox"] = m
        print_table("gtbox (oracle GT boxes — NOT OvSGTR-comparable, upper bound)", m)
    if "detbox" in modes:
        confs = ([float(c) for c in args.det_conf_sweep.split(",")]
                 if args.det_conf_sweep else [args.det_conf])
        for c in confs:
            args.det_conf = c
            tag = f"@conf{c:.2f}" if len(confs) > 1 else ""
            for proto in ("lenient", "strict"):
                m = run_detbox(model, args, pred_names, novel_ids, base_ids, proto,
                               device, eval_args)
                results["modes"][f"detbox_{proto}{tag}"] = m
                print_table(f"detbox/{proto}{tag} (SGDET — OvSGTR-comparable)", m)

    print("\n=== OvSGTR published (SGDET, Novel Relation R@50) ===")
    for name, v in OVSGTR_PUBLISHED.items():
        print(f"  {name:<20} novel_R@50={v.get('novel_R@50')}  "
              f"all_R@50={v.get('all_R@50', '-')}")

    out_path = args.out or os.path.join(
        os.path.dirname(args.checkpoint), "ovsgtr_novel_vg150.json")
    json.dump(results, open(out_path, "w"), indent=2, default=float)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
