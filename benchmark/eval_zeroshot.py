"""Zero-shot transfer evaluation: reparameterize a trained checkpoint to a
new predicate vocabulary (VG150 / PSG / ...) and run the strict SGCls
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

from data import RelationDataset, collate_fn                     # noqa: E402
from relsgg.evaluator import (SGClsEvaluator, SoftSGClsEvaluator,  # noqa: E402
                              build_cross_match_matrix)
from relsgg.model import RelSGG, RelSGGConfig                   # noqa: E402
from relsgg.train_engine import evaluate                        # noqa: E402

# The template ensemble the training W was built with, and the shared loader.
from relsgg.api import TRAIN_TEMPLATES                          # noqa: E402
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


_HEAD_STACKS = {
    # spec name -> (module attribute on RelSGG, ModuleList attribute)
    "self":  ("rel_transformer", "self_layers"),
    "cross": ("rel_transformer", "cross_layers"),
    "dep":   ("rel_interaction", "dep_layers"),
    "gnd":   ("rel_interaction", "gnd_layers"),
}


def ablate_head_layers(model, spec: str) -> str:
    """Drop the last N layers of one HEAD stack. Returns a filename tag.

    WHY THIS IS A VALID ABLATION. Every layer in these stacks is a residual
    transformer block and the forward pass is a bare `for layer in stack` loop
    (transformer.py:356-366), so removing a layer leaves the residual stream
    intact -- it is exactly an identity substitution, not a corrupted forward.

    WHAT IT ANSWERS. Head depth has never been varied in ANY run (the four
    --n_*_layers flags exist in train.py but are absent from the runner and no
    run directory has ever set them). Before spending ~14 GPU-h on a deeper
    head, this asks the cheaper question on the EXISTING checkpoint: does the
    head use the depth it already has? If deleting a layer costs ~nothing, the
    stack is not depth-limited and adding more cannot help.

    NOTE `cross` removes the STANDARD cross layer only. `rel_transformer` keeps
    `last_cross` (the ExposedCrossAttentionLayer) outside the ModuleList and
    always applies it, so cross depth goes 2 -> 1 and never to 0.
    """
    if not spec:
        return ""
    stack, _, n_s = spec.partition(":")
    n = int(n_s or 1)
    if stack not in _HEAD_STACKS:
        raise SystemExit(f"--ablate_head: unknown stack {stack!r}; "
                         f"expected one of {sorted(_HEAD_STACKS)}")
    mod_name, attr = _HEAD_STACKS[stack]
    mod = getattr(model, mod_name, None)
    if mod is None:
        raise SystemExit(f"--ablate_head: this checkpoint has no {mod_name} "
                         f"(use_rel_interaction was off?)")
    ml = getattr(mod, attr)
    before = len(ml)
    keep = before - n
    if keep < 0:
        raise SystemExit(f"--ablate_head: {mod_name}.{attr} has only {before} "
                         f"layer(s); cannot drop {n}")
    setattr(mod, attr, type(ml)(list(ml)[:keep]))
    print(f"[ablate_head] {mod_name}.{attr}: {before} -> {keep} layers "
          f"(dropped the last {n})")
    return f"{stack}{n}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--keep_W", action="store_true",
                   help="Keep the checkpoint's own W when the target vocabulary "
                        "equals the checkpoint's (closed-set fine-tunes, incl. "
                        "--learn_W); otherwise W is re-encoded from text.")
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
    p.add_argument("--compile", default="",
                   help="torch.compile mode to run the model under (e.g. "
                        "'default'). Empty = eager. Use to price inductor's "
                        "numerical drift in benchmark metrics rather than logits.")
    p.add_argument("--score_mode", default="sigmoid",
                   choices=["sigmoid", "softmax"])
    p.add_argument("--dinotxt_weights",
                   default="checkpoints/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth")
    p.add_argument("--text_student", default=None,
                   help="Student text-encoder checkpoint for vocabulary "
                        "encoding. Default: auto — taken from the training "
                        "run's args when the checkpoint was trained in "
                        "student space (encoding a student-trained head "
                        "with the teacher, or vice versa, is meaningless).")
    p.add_argument("--csls_lambda", type=float, default=0.0,
                   help="CSLS hubness correction strength (0 = off). Rescores "
                        "cos' = cos − λ·0.5(r_q + r_e): r_e = per-predicate "
                        "mean top-k cosine over a calibration query sample, "
                        "r_q = per-slot mean top-k over the vocab. Derived "
                        "entirely from the DEPLOYED vocab + query stream — no "
                        "training priors, transfers to any inference vocab. "
                        "Measured on v34/PSG: λ=1 gives mR@50 +32%%, R@50 +6%%. "
                        "Results are written to zeroshot_<name>_csls<λ>.json "
                        "so baselines are never clobbered.")
    p.add_argument("--csls_k", type=int, default=10)
    p.add_argument("--csls_calib", type=int, default=200,
                   help="Images for the r_e calibration pass (the deploy "
                        "equivalent is warmup frames).")
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
                        "was fitted in: 0.955 was calibrated in student_v1 "
                        "(synonym cos 0.961, random 0.805) and silently "
                        "carried into student_v2 (synonym 0.755, random "
                        "0.167), where it admits 0.6%% of true synonyms "
                        "instead of 72%% — A3 measured near-exact string match "
                        "for every run from v38 on. Pass a float only to "
                        "override deliberately.")
    p.add_argument("--tau_calibration", default="",
                   help="Calibration json from training/calibrate_match_tau.py. "
                        "DEFAULT empty = resolve it from the CHECKPOINT'S OWN "
                        "text space, runs/benchmark/tau_calibration_<stem of "
                        "its pred_embeds>.json. Arms are split across spaces "
                        "(v41/v42 are student_v1, wv2/v43/v44 are student_v2) "
                        "and each needs its own threshold, so one shared "
                        "default cannot be right for all of them. Whatever is "
                        "used, its `pred_embeds` must match the checkpoint's "
                        "or the run aborts.")
    p.add_argument("--decode", default="exact", choices=["exact", "codebook"],
                   help="CODEBOOK decoding (commit-late plan E1): keep the "
                        "full training vocabulary deployed, assign each of its "
                        "columns to its best benchmark string in the SEMANTIC "
                        "map space (student_v2 + the isotonic synonym kernel, "
                        "argmax assignment — no thresholds), and score each "
                        "benchmark class as the aggregate of its assigned "
                        "codewords' kernel-weighted sigmoid scores. Unlike "
                        "--open_vocab this does NOT change the metric: the "
                        "standard closed-vocabulary evaluator runs on the "
                        "aggregated [*, V_bench] scores. Results are suffixed "
                        "_cb<agg>.")
    p.add_argument("--codebook_agg", default="max",
                   choices=["max", "sum", "umax"],
                   help="Aggregator over a benchmark class's codeword cell. "
                        "max/sum are kernel-weighted votes; umax is the "
                        "CHiLS-faithful variant — membership by argmax "
                        "assignment, votes UNWEIGHTED (the kernel-weighted "
                        "max measured as a no-op: calibrated synonym "
                        "probabilities are 0.03-0.14 at the cosines where "
                        "cross-lemma siblings live, and a decode max cannot "
                        "integrate small weights the way training does). "
                        "Selection between variants on PSG-val dev only.")
    p.add_argument("--codebook_map",
                   default="runs/packed/datamix_v22/text_space/pred_embeds_studentv2_photo.npz",
                   help="Union-vocabulary embeddings in the MAP space "
                        "(student_v2 — best measured synonymy signal, AP "
                        "0.739). Scores still come from the checkpoint's own "
                        "head/space; only the codeword->benchmark assignment "
                        "lives here.")
    p.add_argument("--syn_kernel",
                   default="runs/packed/datamix_v22/text_space/syn_kernel_v2.npz",
                   help="Dense isotonic P(synonym|cos_v2) curve "
                        "(training/export_syn_kernel.py) used as the codeword "
                        "vote weight — junk codewords self-silence at P~0.")
    p.add_argument("--codebook_encoder",
                   default="runs/packed/text_student_v2/student.pt",
                   help="Encoder for benchmark strings on the MAP side (must "
                        "match --codebook_map's space).")
    p.add_argument("--no_templates", action="store_true",
                   help="Encode bare predicate names (ablation).")
    p.add_argument("--spatial_only", action="store_true",
                   help="Restrict BOTH the deployed vocabulary and the GT to "
                        "the spatial predicates (--spatial_from), i.e. the "
                        "spatial-only product: one head, one predicate set. "
                        "Non-spatial GT relations are dropped, so R@K is over "
                        "the spatial subset only and is NOT comparable to the "
                        "full-vocabulary numbers. Results are written to "
                        "zeroshot_<name>_spa*.json.")
    p.add_argument("--spatial_from", default="",
                   help="Packed root whose train rel-flags bit0 defines which "
                        "predicate NAMES count as spatial (majority vote, the "
                        "same rule train.py fits the gate against). Default: "
                        "the checkpoint's own training root. The eval pack's "
                        "predicates are intersected with that set by name — "
                        "VG150/PSG packs carry no spatial flags of their own.")
    p.add_argument("--force_alpha", type=float, default=None,
                   help="Pin the dual-head routing to a constant instead of "
                        "the trained gate: 1.0 = spatial expert alone (the "
                        "spatial-only deployment — vocab_head.proj drops out "
                        "of the graph entirely), 0.0 = semantic expert alone. "
                        "Run 1.0 / 0.0 / unset over the same --spatial_only "
                        "vocabulary to read off whether q_spa is "
                        "self-sufficient or is being carried by the mixture.")
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
    p.add_argument("--ablate_head", default="",
                   help="DIAGNOSTIC: delete the last N layers of one head "
                        "stack at eval — self:1 / cross:1 / dep:1 / gnd:1. "
                        "These are residual blocks, so removal is a clean "
                        "identity ablation. Measures whether the head USES the "
                        "depth it has, which gates whether a DEEPER head is "
                        "worth training. Results are written to "
                        "zeroshot_<name>_habl<spec>*.json.")
    p.add_argument("--tap_ablate", type=int, default=-1,
                   help="DIAGNOSTIC: mean-ablate backbone tap i (0=shallowest) "
                        "before fusion, i.e. keep its magnitude but destroy its "
                        "spatial content. Answers whether a tap is LOAD-BEARING "
                        "rather than merely large. -1 = off (default).")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch')}, "
          f"best {ckpt.get('best_recall', 0):.4f})")
    model = build_model_from_ckpt(ckpt, args.weights).to(device).eval()
    if args.tap_ablate >= 0:
        model.backbone.tap_ablate = args.tap_ablate
        print(f"[tap_ablate] mean-ablating tap {args.tap_ablate} "
              f"(layer_offsets={model.backbone.layer_offsets})")
        if getattr(model, "ms_scene", None) is not None:
            print("[tap_ablate] WARNING: this arm has a multi-level read, which "
                  "consumes the RAW taps directly and is NOT ablated here — the "
                  "ablation only reaches the fused map.")
    _habl = ablate_head_layers(model, args.ablate_head)
    if getattr(args, "compile", ""):
        # Inductor fusion changes reduction order and fuses epilogues, so it
        # perturbs logits (measured max|dlogit| 1.15, top-50 triplet Jaccard
        # 0.91 at bs1). Whether that is FREE is a question about METRICS, not
        # about logits — this flag exists so the benchmark can answer it.
        model = torch.compile(model, mode=args.compile, dynamic=False)
        print(f"torch.compile(mode={args.compile}) enabled")

    train_vocab = set(ckpt.get("pred_names") or [])
    eval_args = SimpleNamespace(amp=device.type == "cuda",
                                amp_dtype_t=torch.bfloat16)
    out_dir = args.out_dir or os.path.dirname(args.checkpoint)
    # Create it NOW, not at write time: the json is written after every dataset
    # has been scored, so a missing --out_dir used to throw away a completed
    # multi-source eval on the last line.
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
    # --decode codebook shares the full-vocabulary deployment (the head answers
    # over all trained columns) but NOT the soft matcher/evaluator.
    ov_train_preds = ov_E_train = ov_inverse_mask = None
    if args.open_vocab or args.decode == "codebook":
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
        # A cosine threshold is a property of ONE embedding space. Reading it
        # from a calibration that was fitted in a DIFFERENT space is how A3
        # came to accept 0.6% of true synonyms for every run from v38 on, so
        # the space identity is checked rather than trusted.
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
                f"  checkpoint : {ck_args['pred_embeds']}\n"
                f"Re-run training/calibrate_match_tau.py --pred_embeds "
                f"{ck_args['pred_embeds']}, or pass --tau_eval to override.")
        args.tau_eval = float(cal["chosen"]["tau"])
        print(f"tau_eval={args.tau_eval} from {cal_path} "
              f"(synonym recall {100 * cal['chosen']['syn_recall']:.1f}%, "
              f"inverse leak {100 * cal['chosen']['inv_leak']:.2f}%, "
              f"random FPR {100 * cal['chosen']['rand_fpr']:.3f}%)")
    cb_kernel = cb_E_map = None
    if args.decode == "codebook":
        if args.open_vocab or args.spatial_only:
            raise SystemExit("--decode codebook is a closed-vocabulary "
                             "SCORING change; it cannot combine with "
                             "--open_vocab (different metric) or "
                             "--spatial_only.")
        if not text_student:
            raise SystemExit("--decode codebook assumes a student-space "
                             "checkpoint (scores) plus the v2 map space.")
        zm = np.load(args.codebook_map)
        assert [str(q) for q in zm["predicates"]] == ov_train_preds, (
            "codebook_map predicate order != checkpoint vocabulary — "
            "refusing to build a misaligned assignment (index-space bug)")
        cb_E_map = zm["embeddings"].astype(np.float32)
        cb_E_map /= np.linalg.norm(cb_E_map, axis=1, keepdims=True) + 1e-8
        cb_kernel = np.load(args.syn_kernel)
        print(f"codebook decode: map={args.codebook_map} agg={args.codebook_agg}")
        # Inverse masking is OPT-IN because it is provably redundant here:
        # training/calibrate_match_tau.py measures inverse leakage at 0.00%
        # for every tau >= 0.90 (inverse cos 0.279 vs synonym 0.961 — the
        # antonym-aware distillation already separates them). Building it
        # costs ~1.1 GB of [V,V] masks for a guard that never fires.
        if args.ov_inverse_mask and ck_args.get("canon_groups") \
                and ck_args.get("ontology_meta"):
            from relsgg.loss_synonym import PredicateOntology
            ont = PredicateOntology.from_artifacts(
                meta_path=ck_args["ontology_meta"],
                canon_groups_path=ck_args["canon_groups"],
                embeds_path=ck_args.get("pred_embeds") or None,
                tau_ignore=ck_args.get("tau_ignore", 0.9))
            ov_inverse_mask = ont.inverse_mask
            print(f"open-vocab: inverse mask loaded "
                  f"({int(ov_inverse_mask.sum()):,} blocked pairs) — "
                  f"'above' can never be scored as 'below'")

    # Spatial-only product configuration: which predicate names survive.
    spatial_names = set()
    if args.spatial_only:
        src = args.spatial_from or (ck_args.get("data_roots") or [None])[0] \
            or ck_args.get("data_root")
        if not src:
            raise SystemExit("--spatial_only needs --spatial_from: the "
                             "checkpoint records no training pack root.")
        spatial_names = spatial_predicate_names(src)
        print(f"spatial vocabulary: {len(spatial_names)} predicate names from "
              f"{src}")

    for root in args.data_roots:
        name = os.path.basename(os.path.normpath(root))
        rel_cat_to_idx = None
        if args.spatial_only:
            own = json.load(open(os.path.join(root, args.split, "meta.json")))
            keep = [q for q in own["predicates"] if q in spatial_names]
            if not keep:
                print(f"[{name}] no spatial predicates in this pack — skipped")
                continue
            rel_cat_to_idx = {q: i for i, q in enumerate(keep)}
        ds = RelationDataset(root=root, split=args.split,
                             resolution=args.img_size,
                             max_objects=args.max_objects,
                             rel_cat_to_idx=rel_cat_to_idx,
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
        E_bench = None
        _keep = (args.keep_W and not args.open_vocab and args.decode != "codebook"
                 and list(ckpt.get("pred_names") or []) == list(pred_names))
        if _keep:
            # Closed-set fine-tune: the checkpoint's own (possibly trained) W is
            # already loaded by build_model_from_ckpt; do not re-encode it.
            print(f"[{name}] --keep_W: vocabulary matches the checkpoint's "
                  f"{len(pred_names)} predicates; keeping its W")
            model.vocab_head.pred_names = list(pred_names)
        elif text_student:
            from relsgg.text_student import encode_texts_student
            E_bench = encode_texts_student(pred_names, text_student,
                                           templates=templates, device=device)
            if args.open_vocab or args.decode == "codebook":
                # Deploy the training vocabulary, not the benchmark's. The
                # benchmark embeddings are still needed — as the MATCHER's
                # right-hand side (open_vocab) or not at all (codebook, whose
                # map side is encoded separately in the v2 space).
                model.vocab_head.set_vocabulary_matrix(ov_train_preds, ov_E_train)
            else:
                model.vocab_head.set_vocabulary_matrix(pred_names, E_bench)
        else:
            if args.open_vocab:
                raise SystemExit("--open_vocab requires the student text "
                                 "encoder (--text_student / a student-trained "
                                 "checkpoint): the matcher and the head must "
                                 "live in one text space.")
            model.vocab_head.encode_vocabulary_dinotxt(
                pred_names, dinotxt_weights=args.dinotxt_weights,
                templates=templates)
        if args.force_alpha is not None:
            model.vocab_head.set_alpha_override(args.force_alpha)
        model.reparameterize()
        if model.config.dual_spatial_head and args.decode != "codebook":
            al = model.vocab_head.alpha
            if args.force_alpha is not None:
                print(f"[{name}] routing PINNED to alpha={args.force_alpha} "
                      f"({'spatial' if args.force_alpha >= 0.5 else 'semantic'}"
                      f" expert only) — gate bypassed")
            else:
                # alpha is a mixture weight, not a hard route: print the
                # values, not a count over an arbitrary 0.5 threshold.
                order = sorted(zip(pred_names, al.tolist()), key=lambda t: -t[1])
                print(f"[{name}] gate alpha (1=spatial expert, 0=semantic): "
                      + ", ".join(f"{n}={v:.2f}" for n, v in order))

        # ---- optional CSLS hubness correction (see --csls_lambda help) ----
        orig_sqd = model.vocab_head.score_query_dual
        if args.csls_lambda > 0:
            scale = float(model.vocab_head.logit_scale.exp().clamp(max=100.0))
            bias = float(model.vocab_head.logit_bias)
            rows, n_cal = [], 0
            with torch.no_grad():
                for images, boxes, box_counts, _tg in loader:
                    out = model(images.to(device), boxes.to(device),
                                box_counts.to(device), targets=None)
                    cos = (out["logits"].float() - bias) / scale
                    rows.append(cos[out["valid_mask"]].cpu())
                    n_cal += images.shape[0]
                    if n_cal >= args.csls_calib:
                        break
            allq = torch.cat(rows)
            k_e = min(args.csls_k, allq.shape[0])
            r_e = allq.topk(k_e, dim=0).values.mean(0).to(device)   # [V]
            lam, k_q = args.csls_lambda, args.csls_k
            print(f"[{name}] CSLS on: λ={lam} k={k_q} calib={n_cal} imgs "
                  f"(hubbiest: "
                  + ", ".join(f"{pred_names[i]}={r_e[i]:.2f}"
                              for i in r_e.topk(3).indices.tolist()) + ")")

            def wrapped(q_sem, q_spa, _o=orig_sqd, _re=r_e):
                lg = _o(q_sem, q_spa)
                cos = (lg.float() - bias) / scale
                r_q = cos.topk(min(k_q, cos.shape[-1]),
                               dim=-1).values.mean(-1, keepdim=True)
                cos = cos - lam * 0.5 * (r_q + _re)
                return (cos * scale + bias).to(lg.dtype)

            model.vocab_head.score_query_dual = wrapped

        if args.decode == "codebook":
            # ---- vocabulary-induced codebook assignment (map space) ----
            # Every trained column votes for its argmax benchmark string,
            # weighted by the isotonic kernel P(syn|cos) at that cosine —
            # junk columns self-silence. Coverage guard: a benchmark string
            # whose own best codeword was claimed by a sibling keeps that
            # codeword too (bipartite cover, still no thresholds).
            from relsgg.text_student import encode_texts_student
            E_bm = encode_texts_student(pred_names, args.codebook_encoder,
                                        templates=TRAIN_TEMPLATES, device=device)
            E_bm = torch.nn.functional.normalize(
                E_bm.float(), dim=-1).cpu().numpy()
            cosm = cb_E_map @ E_bm.T                          # [V_U, V_B]
            a_of = cosm.argmax(1)
            k_of = np.interp(cosm[np.arange(len(a_of)), a_of],
                             cb_kernel["cos"], cb_kernel["p"])
            src = [np.arange(len(a_of))]
            dst = [a_of]
            wts = [k_of]
            top_m = cosm.argmax(0)                            # [V_B]
            need = np.nonzero(a_of[top_m] != np.arange(cosm.shape[1]))[0]
            if len(need):
                src.append(top_m[need])
                dst.append(need)
                wts.append(np.interp(cosm[top_m[need], need],
                                     cb_kernel["cos"], cb_kernel["p"]))
            cb_src = torch.from_numpy(np.concatenate(src)).long().to(device)
            cb_dst = torch.from_numpy(np.concatenate(dst)).long().to(device)
            cb_w = torch.from_numpy(np.concatenate(wts)).float().to(device)
            eff = torch.zeros(len(pred_names))
            eff.scatter_add_(0, cb_dst.cpu(), (cb_w.cpu() > 0.01).float())
            print(f"[{name}] codebook: {len(cb_src):,} codewords -> "
                  f"{len(pred_names)} classes; effective members (k>0.01) "
                  f"median {int(eff.median())}, min {int(eff.min())}, "
                  f"max {int(eff.max())}; agg={args.codebook_agg}")
            _cb_prev = model.vocab_head.score_query_dual

            def cb_wrapped(q_sem, q_spa, _o=_cb_prev, _src=cb_src,
                           _dst=cb_dst, _w=cb_w, _vb=len(pred_names),
                           _agg=args.codebook_agg):
                lg = _o(q_sem, q_spa)                          # [..., V_U]
                p_all = torch.sigmoid(lg.float())
                v = p_all.index_select(-1, _src)
                if _agg != "umax":
                    v = v * _w                                 # [..., nnz]
                out = torch.zeros(*p_all.shape[:-1], _vb,
                                  device=v.device, dtype=v.dtype)
                ix = _dst.view(*(1,) * (v.dim() - 1), -1).expand_as(v)
                out.scatter_reduce_(-1, ix, v,
                                    reduce="sum" if _agg == "sum" else "amax")
                out = out.clamp_(min=1e-7, max=1.0 - 1e-6)
                return torch.log(out / (1.0 - out)).to(lg.dtype)

            model.vocab_head.score_query_dual = cb_wrapped

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
        try:
            metrics = evaluate(model, loader, device, eval_args, ev,
                               eval_budget=args.eval_budget)
        finally:
            model.vocab_head.score_query_dual = orig_sqd

        if args.open_vocab:
            # Soft evaluator keys its per-class stats by GT class id.
            per_cls = {k: {i: [ev._group_tp[k].get(i, 0) / n]
                           for i, n in ev._group_gt.items() if n > 0}
                       for k in ev.topk}
        else:
            # mR restricted to predicates whose exact string exists in the
            # training vocabulary — separates "never taught the word" from
            # "taught but not transferred" (plan P4 framing).
            overlap = [i for i, n in enumerate(pred_names) if n in train_vocab]
            per_cls = {k: dict(ev._per_class_recall[k]) for k in ev.topk}
            for k in ev.topk:
                vals = [float(np.mean(per_cls[k][i]))
                        for i in overlap if i in per_cls[k]]
                metrics[f"mR@{k}_in_train_vocab"] = float(np.mean(vals)) if vals else 0.0
            metrics["n_pred_in_train_vocab"] = len(overlap)

        print(f"[{name}] " + "  ".join(f"{k}: {v:.4f}"
                                       for k, v in sorted(metrics.items())))
        suffix = f"_csls{args.csls_lambda}" if args.csls_lambda > 0 else ""
        if _habl:
            # MUST be in the filename: without it an ablation run overwrites the
            # arm's canonical zeroshot_<name>_test_gc.json with a deliberately
            # crippled model's numbers.
            suffix = f"_habl{_habl}{suffix}"
        if args.tap_ablate >= 0:
            # MUST be in the filename: without it an ablation run overwrites the
            # arm's canonical zeroshot_<name>_test_gc.json with a deliberately
            # crippled model's numbers.
            suffix = f"_tabl{args.tap_ablate}{suffix}"
        if args.force_alpha is not None:
            suffix = f"_a{args.force_alpha:g}{suffix}"
        if args.spatial_only:
            suffix = f"_spa{suffix}"
        if args.decode == "codebook":
            suffix = f"_cb{args.codebook_agg}{suffix}"
        if args.open_vocab:
            suffix = f"_ov{suffix}"
        if args.img_size != 448 and not args.out_dir:
            # ANTI-CLOBBER, same reason as --tap_ablate above: a resolution
            # sweep would otherwise overwrite the arm's canonical 448px
            # zeroshot_<name>_test_gc.json, which is what the paper tables and
            # overall_score.py read. Skipped when --out_dir is set, because then
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
                       "spatial_only": args.spatial_only,
                       "force_alpha": args.force_alpha,
                       "predicates": pred_names}, f, indent=2)
        print(f"[{name}] saved → {out_path}")


if __name__ == "__main__":
    main()
