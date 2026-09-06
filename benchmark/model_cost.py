"""model_cost.py — parameters and FLOPs for the released RelateAnything family,
in the same accounting the baselines are measured under.

Latency already lives in benchmark/latency.py (relation head, A40) and
deploy/bench_react_compare.py (detector + relation head, end to end). This adds the
two static costs, and — because the models being compared are not the same KIND of
model — it reports them split:

  relation      our tower alone: the number to compare against another relation head
  detector      the external detector our contract requires (params from its own
                checkpoint; FLOPs from a forward at its native input size)
  pipeline      the sum, which is what compares to an end-to-end model like OvSGTR

FLOPs are counted with torch.utils.flop_counter on one real image at the deployed
resolution and the mean box count of the pack, since a relation head's cost depends
on how many pairs it scores.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402
from benchmark.vocab_stub import real_vocabulary, stub_vocabulary  # noqa: E402


def gflops(fn) -> float | None:
    try:
        from torch.utils.flop_counter import FlopCounterMode
        fc = FlopCounterMode(display=False)
        with fc, torch.no_grad():
            fn()
        return fc.get_total_flops() / 1e9
    except Exception as exc:                # noqa: BLE001
        print("  FLOP counting failed:", exc)
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="NAME=run_dir")
    p.add_argument("--pack", default="runs/packed/vg150/test")
    p.add_argument("--det_weights", nargs="*", default=[],
                   help="NAME=path.pt detectors to cost alongside")
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--det_imgsz", type=int, default=640)
    p.add_argument("--n_boxes", type=int, default=0,
                   help="boxes per image for the relation forward; 0 = pack mean")
    p.add_argument("--eval_budget", type=int, default=400)
    p.add_argument("--vocab", choices=["closed", "open"], default="closed",
                   help="closed = the pack's V (the A1 setting); open = the checkpoint's "
                        "full training vocabulary. The head's FLOPs scale with V, so the "
                        "two answer different questions and both belong in the table.")
    p.add_argument("--synthetic_vocab", action="store_true",
                   help="substitute a shape-correct stand-in when the checkpoint's text "
                        "student / pred_embeds are absent (see vocab_stub.py). Exact for "
                        "params and FLOPs; meaningless for accuracy.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    meta = json.load(open(Path(args.pack) / "meta.json"))
    pred_names = list(meta["predicates"])
    img_meta = np.load(Path(args.pack) / "img_meta.npy")
    n_boxes = args.n_boxes or int(round(float(np.mean(img_meta[:, 4]))))
    print(f"pack {args.pack}: {n_boxes} boxes/image (mean), V={len(pred_names)}")

    dev = torch.device(args.device)
    synthetic = False
    out = {"pack": args.pack, "img_size": args.img_size, "n_boxes": n_boxes,
           "vocab_regime": args.vocab,
           "V": len(pred_names), "eval_budget": args.eval_budget,
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "relation": {}, "detector": {}}

    for spec in args.runs:
        name, d = spec.split("=", 1)
        ckpt_path = os.path.join(d, "checkpoint_last.pth")
        if not os.path.exists(ckpt_path):
            print(f"skip {name}: no {ckpt_path}")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_ckpt(ckpt, "ema").to(dev).eval()
        ck_args = ckpt.get("args") or {}
        if not isinstance(ck_args, dict):
            ck_args = vars(ck_args)
        ts = ck_args.get("text_student") or ""
        pe = ck_args.get("pred_embeds") or ""
        names = ([str(q) for q in ckpt.get("pred_names", [])]
                 if args.vocab == "open" else pred_names)
        real = real_vocabulary(names, ts, pe, dev, TRAIN_TEMPLATES)
        if real is not None and args.vocab == "open":
            names, E = real
        elif ts and os.path.exists(ts):
            from relsgg.text.student import encode_texts_student
            E = encode_texts_student(names, ts, templates=TRAIN_TEMPLATES, device=dev)
        elif args.synthetic_vocab:
            names, E = stub_vocabulary(names, int(ck_args.get("text_dim") or 512), dev)
            synthetic = True
            print(f"  [synthetic vocab] V={len(names)} stand-in; params/FLOPs only")
        elif not ts:
            raise SystemExit(
                "this checkpoint names no text student. The vocabulary has to be "
                "encoded by the encoder the head was trained against; pass "
                "--text_student, or use a released model, which ships its own.")
            E = None
        else:
            raise SystemExit(f"{name}: text student '{ts}' absent; pass --synthetic_vocab")
        if E is not None:
            model.vocab_head.set_vocabulary_matrix(names, E)
        model.reparameterize()
        raw = model.module if hasattr(model, "module") else model
        raw.sampler.final_budget = min(args.eval_budget, raw.sampler.geo_budget)

        n_par = sum(p.numel() for p in model.parameters())
        img = torch.randn(1, 3, args.img_size, args.img_size, device=dev)
        boxes = torch.rand(1, n_boxes, 4, device=dev) * 0.5 + 0.25
        counts = torch.tensor([n_boxes], device=dev)
        g = gflops(lambda: model(img, boxes, counts, targets=None))
        out["relation"][name] = {"params_M": n_par / 1e6, "GFLOPs": g,
                                 "run": d, "epoch": ckpt.get("epoch"),
                                 "V": len(names)}
        print(f"{name:<12} relation: {n_par/1e6:6.1f}M params, "
              f"{'n/a' if g is None else f'{g:7.1f}'} GFLOPs")
        del model
        torch.cuda.empty_cache()

    for spec in args.det_weights:
        name, w = spec.split("=", 1)
        if not os.path.exists(w):
            print(f"skip detector {name}: {w} missing")
            continue
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        from ultralytics import YOLO
        y = YOLO(w)
        m = y.model.to(dev).eval()
        n_par = sum(p.numel() for p in m.parameters())
        x = torch.randn(1, 3, args.det_imgsz, args.det_imgsz, device=dev)
        g = gflops(lambda: m(x))
        out["detector"][name] = {"params_M": n_par / 1e6, "GFLOPs": g,
                                 "weights": w, "imgsz": args.det_imgsz}
        print(f"{name:<12} detector: {n_par/1e6:6.1f}M params, "
              f"{'n/a' if g is None else f'{g:7.1f}'} GFLOPs")
        del y, m
        torch.cuda.empty_cache()

    out["synthetic_vocab"] = synthetic
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
