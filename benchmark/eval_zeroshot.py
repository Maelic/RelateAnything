"""Zero-shot transfer evaluation: reparameterize a trained checkpoint to a
new predicate vocabulary (VG150 / PSG /...) and run the strict SGCls
protocol on its packed val split.

This exercises the actual product contract: no fine-tuning, no ontology —
load checkpoint, encode the target vocabulary with the frozen dino.txt
tower (same template ensemble as training), reparameterize, score. The
dual-head spatialness gate re-routes the new vocabulary automatically from
its text embeddings.

Usage (offline compute node):
    python benchmark/eval_zeroshot.py \
        --checkpoint runs/train/full_v1_dualspa/checkpoint_best.pth \
        --data_roots runs/packed/vg150 runs/packed/psg
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                     # noqa: E402
from relsgg.eval.evaluator import (SGClsEvaluator, SoftSGClsEvaluator,  # noqa: E402
                              build_cross_match_matrix)
from relsgg.config import RelSGGConfig
from relsgg.model import RelSGG                   # noqa: E402
from relsgg.training.engine import evaluate                        # noqa: E402

# The template ensemble the training W was built with, and the shared loader.
from relsgg.vocabulary import TRAIN_TEMPLATES                          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt             # noqa: E402


def spatial_predicate_names(root: str) -> set:
    """Predicate NAMES a packed root labels spatial, by majority vote over its
    train rel-flags bit0 — the same rule train.py fits the spatialness gate
    against (train.py:predicate_spatial_flags).

    Returned as names rather than ids because the benchmark packs carry no
    spatial flags of their own (VG150/PSG are all-zero); the subset has to
    cross vocabularies by string.
    """
    meta = json.load(open(os.path.join(root, "train", "meta.json")))
    preds = meta["predicates"]
    rels = np.load(os.path.join(root, "train", "rels.npy"), mmap_mode="r")
    pid = np.asarray(rels[:, 2])
    sbit = (np.asarray(rels[:, 3]) & 1).astype(np.float64)
    cnt = np.bincount(pid, minlength=len(preds))
    spa = np.bincount(pid, weights=sbit, minlength=len(preds))
    return {preds[i] for i in range(len(preds))
            if cnt[i] > 0 and spa[i] >= 0.5 * cnt[i]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_roots", nargs="+", required=True,
                   help="Packed dataset roots with val/ splits.")
    p.add_argument("--split", default="val",
                   help="Pack split subdir to evaluate (val/test). Non-val "
                        "splits are suffixed into the output filename.")
    p.add_argument("--graph_constraint", action="store_true",
                   help="Rank one triplet per object pair (its arg-max "
                        "predicate) — the convention behind most published "
                        "R@K numbers. Default is the unconstrained top-K over "
                        "pairs x predicates, which scores higher. Results are "
                        "written to zeroshot_<name>_gc*.json.")
    p.add_argument("--weights", default="ema", choices=["ema", "raw"])
    p.add_argument("--score_mode", default="sigmoid",
                   choices=["sigmoid", "softmax"])
    p.add_argument("--text_student", default=None,
                   help="Student text-encoder checkpoint for vocabulary "
                        "encoding. Default: auto — taken from the training "
                        "run's args when the checkpoint was trained in "
                        "student space (encoding a student-trained head "
                        "with the teacher, or vice versa, is meaningless).")
    p.add_argument("--open_vocab", action="store_true",
                   help="OPEN-VOCABULARY protocol: keep the model's full "
                        "training vocabulary deployed instead of "
                        "reparameterizing down to the benchmark's own "
                        "predicates, and score a prediction correct when it "
                        "MEANS the GT predicate (text cosine >= --tau_eval, "
                        "exact string, never an inverse). The default "
                        "closed-vocabulary protocol lets the model answer "
                        "only in the benchmark's 37-56 words, which penalises "
                        "a synonym-preserving model for saying 'on top of' "
                        "where the benchmark wrote 'on'. Always "
                        "graph-constrained. Results go to "
                        "zeroshot_<name>_ov*.json. Use a small --batch_size "
                        "(8): logits are [B, budget, 19103] here.")
    p.add_argument("--ov_inverse_mask", action="store_true",
                   help="Also block inverse pairs in the --open_vocab matcher. "
                        "OFF by default: measured inverse leakage is 0.00%% at "
                        "every tau >= 0.90, so this only costs ~1.1 GB of "
                        "[V,V] ontology masks. Turn on to prove the property "
                        "rather than rely on it.")
    p.add_argument("--tau_eval", type=float, default=None,
                   help="Cosine threshold for --open_vocab synonym matching. "
                        "DEFAULT None = read it from --tau_calibration, which "
                        "is checked against the checkpoint's own text space. "
                        "A cosine threshold is only meaningful in the space it "
                        "was fitted in. Carried into another space, a "
                        "threshold that accepted 72%% of true synonyms can "
                        "accept 0.6%% of them and turn A3 into near-exact "
                        "string matching. Pass a float only to override "
                        "deliberately.")
    p.add_argument("--tau_calibration", default="",
                   help="Calibration json from training/calibrate_match_tau.py. "
                        "DEFAULT empty = resolve it from the CHECKPOINT'S OWN "
                        "text space, runs/benchmark/tau_calibration_<stem of "
                        "its pred_embeds>.json. Each text space needs its own "
                        "threshold, so whatever is used, its `pred_embeds` "
                        "must match the checkpoint's or the run aborts.")
    p.add_argument("--no_templates", action="store_true",
                   help="Encode bare predicate names (ablation).")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--eval_budget", type=int, default=500)
    p.add_argument("--max_objects", type=int, default=100)
    p.add_argument("--limit", type=int, default=0,
                   help="Debug: evaluate only the first N images.")
    p.add_argument("--rasters", default="",
                   help="Root of precomputed region rasters (datagen/"
                        "build_mask_rasters.py) — evaluates in MASK mode. "
                        "The rasters are model INPUTS, not labels; "
                        "train_engine.evaluate() forwards them even with "
                        "targets=None. Pair with --out_dir so mask-mode "
                        "results never overwrite the box-mode OVS jsons. "
                        "Unset = box mode (unchanged).")
    p.add_argument("--out_dir", default="",
                   help="Where to write zeroshot_<name>.json "
                        "(default: checkpoint dir).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch')}, "
          f"best {ckpt.get('best_recall', 0):.4f})")
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    train_vocab = set(ckpt.get("pred_names") or [])
    eval_args = SimpleNamespace(amp=device.type == "cuda",
                                amp_dtype_t=torch.bfloat16)
    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    # Created now rather than at write time: the json lands after every dataset
    # has been scored, and a missing directory would discard the whole run on
    # its last line.
    os.makedirs(out_dir, exist_ok=True)
    templates = None if args.no_templates else TRAIN_TEMPLATES

    # Match the text encoder to the one the head was trained against.
    ck_args = ckpt.get("args") or {}
    if not isinstance(ck_args, dict):
        ck_args = vars(ck_args)
    text_student = (args.text_student if args.text_student is not None
                    else ck_args.get("text_student") or "")
    if text_student:
        print(f"vocabulary encoder: STUDENT ({text_student})")

    # ---- open-vocabulary setup (loaded once; identical for every root) ----
    ov_train_preds = ov_E_train = ov_inverse_mask = None
    if args.open_vocab:
        z = np.load(ck_args["pred_embeds"])
        ov_train_preds = [str(p) for p in z["predicates"]]
        ov_E_train = z["embeddings"]
        # The npz IS the W the head was trained against, so its order is
        # authoritative; re-encoding would risk a silent drift.
        assert ov_train_preds == list(ckpt["pred_names"]), (
            "pred_embeds order does not match the checkpoint's pred_names")
        print(f"deploying the full {len(ov_train_preds):,}-predicate "
              f"training vocabulary from {ck_args['pred_embeds']}")
    if args.open_vocab and args.tau_eval is None:
        # A cosine threshold is a property of one embedding space, so the
        # space identity is checked rather than trusted.
        cal_path = args.tau_calibration or os.path.join(
            "runs/benchmark",
            "tau_calibration_%s.json" % os.path.splitext(
                os.path.basename(ck_args["pred_embeds"]))[0])
        if not os.path.exists(cal_path):
            raise SystemExit(
                f"no tau calibration for this checkpoint's text space.\n"
                f"  expected: {cal_path}\n"
                f"  run: python training/calibrate_match_tau.py --pred_embeds "
                f"{ck_args['pred_embeds']} --out {cal_path}")
        cal = json.load(open(cal_path))
        if os.path.realpath(cal["pred_embeds"]) != os.path.realpath(ck_args["pred_embeds"]):
            raise SystemExit(
                f"tau calibration was fitted in a different text space:\n"
                f"  calibration: {cal['pred_embeds']}\n"
                f"  checkpoint: {ck_args['pred_embeds']}\n"
                f"Re-run training/calibrate_match_tau.py --pred_embeds "
                f"{ck_args['pred_embeds']}, or pass --tau_eval to override.")
        args.tau_eval = float(cal["chosen"]["tau"])
        print(f"tau_eval={args.tau_eval} from {cal_path} "
              f"(synonym recall {100 * cal['chosen']['syn_recall']:.1f}%, "
              f"inverse leak {100 * cal['chosen']['inv_leak']:.2f}%, "
              f"random FPR {100 * cal['chosen']['rand_fpr']:.3f}%)")
    for root in args.data_roots:
        name = os.path.basename(os.path.normpath(root))
        ds = RelationDataset(root=root, split=args.split,
                             resolution=args.img_size,
                             max_objects=args.max_objects,
                             rasters=args.rasters or None)
        if args.limit:
            ds = torch.utils.data.Subset(ds, range(min(args.limit, len(ds))))
            ds.predicate_names = ds.dataset.predicate_names
        pred_names = ds.predicate_names
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=args.num_workers, pin_memory=True)

        print(f"\n[{name}] {len(ds)} images, {len(pred_names)} predicates — "
              f"reparameterizing (templates={'train-ensemble' if templates else 'bare'})")
        if not text_student:
            raise SystemExit(
                "this checkpoint names no text student. The vocabulary has to be "
                "encoded by the encoder the head was trained against; pass "
                "--text_student, or use a released model, which ships its own.")
        from relsgg.text.student import encode_texts_student
        E_bench = encode_texts_student(pred_names, text_student,
                                       templates=templates, device=device)
        if args.open_vocab:
            # The head answers over its whole training vocabulary; the
            # benchmark's own embeddings become the matcher's right-hand side.
            model.vocab_head.set_vocabulary_matrix(ov_train_preds, ov_E_train)
        else:
            model.vocab_head.set_vocabulary_matrix(pred_names, E_bench)
        model.reparameterize()
        # alpha is a mixture weight, not a hard route, so print the values.
        order = sorted(zip(pred_names, model.vocab_head.alpha.tolist()),
                       key=lambda t: -t[1])
        print(f"[{name}] gate alpha (1 = spatial expert, 0 = semantic): "
              + ", ".join(f"{n}={v:.2f}" for n, v in order))
        if args.open_vocab:
            # Open-vocabulary contract: the head keeps its FULL training
            # vocabulary (installed above) and answers in its own words; a
            # prediction counts when it MEANS the GT predicate. Always
            # graph-constrained — see SoftSGClsEvaluator.graph_constraint.
            M_cross = build_cross_match_matrix(
                ov_train_preds, pred_names, ov_E_train, E_bench,
                tau_eval=args.tau_eval, inverse_mask=ov_inverse_mask)
            per_gt = M_cross.sum(0)
            print(f"[{name}] open-vocab matcher: {M_cross.shape[0]:,} deployed "
                  f"predicates -> {M_cross.shape[1]} GT classes, "
                  f"{float(per_gt.float().mean()):.1f} accepted spellings per GT "
                  f"(min {int(per_gt.min())}, max {int(per_gt.max())}) @tau={args.tau_eval}")
            assert int(per_gt.min()) >= 1, (
                "a GT predicate has no accepted spelling — tau_eval too high")
            ev = SoftSGClsEvaluator(
                M_cross, torch.arange(len(pred_names)),
                topk=[20, 50, 100], score_mode=args.score_mode,
                graph_constraint=True)
        else:
            ev = SGClsEvaluator(topk=[20, 50, 100],
                                num_predicates=len(pred_names),
                                score_mode=args.score_mode,
                                graph_constraint=args.graph_constraint)
        metrics = evaluate(model, loader, device, eval_args, ev,
                           eval_budget=args.eval_budget)

        if args.open_vocab:
            # Soft evaluator keys its per-class stats by GT class id.
            per_cls = {k: {i: [ev._group_tp[k].get(i, 0) / n]
                           for i, n in ev._group_gt.items() if n > 0}
                       for k in ev.topk}
        else:
            # mR over predicates whose exact string is in the training
            # vocabulary, which separates "never saw the word" from "saw it
            # but did not transfer".
            overlap = [i for i, n in enumerate(pred_names) if n in train_vocab]
            per_cls = {k: dict(ev._per_class_recall[k]) for k in ev.topk}
            for k in ev.topk:
                vals = [float(np.mean(per_cls[k][i]))
                        for i in overlap if i in per_cls[k]]
                metrics[f"mR@{k}_in_train_vocab"] = float(np.mean(vals)) if vals else 0.0
            metrics["n_pred_in_train_vocab"] = len(overlap)

        print(f"[{name}] " + "  ".join(f"{k}: {v:.4f}"
                                       for k, v in sorted(metrics.items())))
        suffix = ""
        if args.open_vocab:
            suffix = f"_ov{suffix}"
        if args.img_size != 448 and not args.out_dir:
            # Keep a resolution sweep from overwriting the canonical 448 px
            # results, which the tables read. Skipped when --out_dir is set:
            # the caller has already separated the outputs by directory (the
            # convention --masks uses) and the canonical files are not at risk.
            suffix = f"_r{args.img_size}{suffix}"
        if args.graph_constraint or args.open_vocab:
            suffix = f"_gc{suffix}"
        if args.split != "val":
            suffix = f"_{args.split}{suffix}"
        out_path = os.path.join(out_dir, f"zeroshot_{name}{suffix}.json")
        with open(out_path, "w") as f:
            json.dump({"metrics": metrics,
                       "per_class_recall": {
                           str(k): {pred_names[i]: float(np.mean(v))
                                    for i, v in per_cls[k].items()}
                           for k in ev.topk},
                       "score_mode": args.score_mode,
                       "weights": args.weights,
                       "templates": templates,
                       "predicates": pred_names}, f, indent=2)
        print(f"[{name}] saved → {out_path}")


if __name__ == "__main__":
    main()
