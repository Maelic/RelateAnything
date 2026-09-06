"""bench_ovsgtr_cost.py — parameters, FLOPs and A40 batch-1 latency for OvSGTR,
measured on the SAME images and the SAME harness protocol as our own family
(benchmark/latency.py: CUDA-event timed, warmup then N timed images, bs1).

Runs in the OvSGTR venv.

What is counted, and why it is the whole model. OvSGTR is end-to-end: one forward
produces boxes, object labels and relation features, and `graph_infer` then scores
EVERY ordered pair of detected objects. Our model is relation-only and needs an
external detector, so the comparable quantity on our side is detector + relation head
(deploy/bench_react_compare.py). Reporting their relation head alone would flatter
them and reporting our pipeline against their head would flatter us.

The pair count matters and is reported: their relation cost is quadratic in the number
of surviving objects (~97/image on VG150), so `graph_infer` is timed separately from
the transformer forward.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

parents = Path(__file__).resolve().parents

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

if str(parents[2]) not in sys.path:
    sys.path.insert(0, str(parents[2]))
from benchmark.ovsgtr.run_ovsgtr_pack import (Pack, resized_hw,  # noqa: E402
                                              IMAGENET_MEAN, IMAGENET_STD)
from benchmark.ovsgtr.ovsgtr_common import (OVSGTR_ROOT, load_ovsgtr, build_prompts,  # noqa: E402
                           install_vocabulary, _ensure_ovsgtr_on_path)

_ensure_ovsgtr_on_path()          # OVSGTR_ROOT, overridable via $OVSGTR_ROOT
from util.misc import nested_tensor_from_tensor_list  # noqa: E402


def host_info():
    """Node + CPU model. At batch 1 a chunk of the wall time is kernel-launch and
    python overhead, so the host CPU is part of the measurement, not trivia."""
    import platform, re
    cpu = ""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"node": platform.node(), "cpu": cpu,
            "torch": torch.__version__, "cuda": torch.version.cuda}


def count_params(*modules):
    """Distinct parameter tensors across the model and its postprocessor heads.

    De-duplicated by id: the postprocessor holds references to the same rln_proj /
    rln_classifier tensors the model does, so a naive sum double-counts them.
    """
    seen, total = set(), 0
    for m in modules:
        if m is None:
            continue
        for p in getattr(m, "parameters", lambda: [])():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pack", default="runs/packed/vg150/test")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n_images", type=int, default=200)
    p.add_argument("--n_warmup", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--img_dir", default=None,
                   help="override the pack's recorded image directory (packs built on "
                        "another cluster carry that cluster's absolute path)")
    p.add_argument("--tf32", action="store_true",
                   help="allow TF32 for fp32 matmuls (cuDNN convs already default to it "
                        "in torch>=1.12). Off = strict fp32, the torch>=1.12 default; on "
                        "= what torch 1.9 did by default, which is the regime OvSGTR's "
                        "own published 3090 latency was measured in, and what anyone "
                        "deploying on Ampere/Hopper would actually get. Reduced precision "
                        "is NOT an option here: their ms_deform_attn CUDA kernel is "
                        "AT_DISPATCH_FLOATING_TYPES, so fp16/bf16 raise at the first "
                        "encoder layer.")
    p.add_argument("--flops", action="store_true",
                   help="also run torch's FlopCounterMode over the transformer forward "
                        "AND the postprocessor, on --flops_images images (their box "
                        "count, and hence the quadratic pair term, varies per image)")
    p.add_argument("--flops_images", type=int, default=20)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    pack = Pack(Path(args.pack))
    if args.img_dir:
        pack.img_dir = Path(args.img_dir)
    if not pack.img_dir.is_dir():
        raise SystemExit(f"image dir {pack.img_dir} does not exist; pass --img_dir")
    model, post, ck_args, info = load_ovsgtr(args.config, args.checkpoint, args.device)
    prompts = build_prompts(pack.categories, pack.predicates)
    install_vocabulary(post, pack.categories, pack.predicates)

    n_params = count_params(model, post,
                            getattr(post, "rln_proj", None),
                            getattr(post, "rln_classifier", None),
                            getattr(post, "rln_freq_bias", None))
    print(f"parameters: {n_params/1e6:.2f}M")

    dev = torch.device(args.device)
    imgs = []
    for i in range(args.n_warmup + args.n_images):
        it = pack.item(i % len(pack))
        im = Image.open(pack.img_dir / it["file_name"]).convert("RGB")
        W, H = im.size
        oh, ow = resized_hw(W, H)
        t = TF.normalize(TF.to_tensor(TF.resize(im, [oh, ow])), IMAGENET_MEAN, IMAGENET_STD)
        imgs.append((t, W, H))

    target = {"caption": prompts["caption"], "rel_caption": prompts["rel_caption"]}
    fwd_ms, post_ms, tot_ms, n_boxes, n_pairs = [], [], [], [], []

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32

    @torch.no_grad()
    def one(t, W, H, timed):
        samples = nested_tensor_from_tensor_list([t.to(dev)])
        orig = torch.as_tensor([[H, W]], device=dev)
        if not timed:
            out = model(samples, [target])
            post(out, orig)
            return
        e = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        e[0].record()
        out = model(samples, [target])
        e[1].record()
        res = post(out, orig)          # includes graph_infer: all N(N-1) pairs
        e[2].record()
        torch.cuda.synchronize()
        fwd_ms.append(e[0].elapsed_time(e[1]))
        post_ms.append(e[1].elapsed_time(e[2]))
        tot_ms.append(e[0].elapsed_time(e[2]))
        g = res[0].get("graph", {})
        pr = g.get("all_node_pairs")
        bx = g.get("pred_boxes")
        n_pairs.append(0 if pr is None else len(pr))
        n_boxes.append(0 if bx is None else len(bx))

    print(f"warmup {args.n_warmup}...", flush=True)
    for t, W, H in imgs[:args.n_warmup]:
        one(t, W, H, timed=False)
    torch.cuda.synchronize()
    print(f"timing {args.n_images}...", flush=True)
    t0 = time.time()
    for t, W, H in imgs[args.n_warmup:]:
        one(t, W, H, timed=True)
    wall = time.time() - t0

    def stats(a):
        a = np.asarray(a, dtype=np.float64)
        return {"mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "std": float(a.std()),
                "min": float(a.min()), "max": float(a.max())}

    res = {"model": "OvSGTR", "checkpoint": args.checkpoint, "config": args.config,
           "precision": "fp32-tf32" if args.tf32 else "fp32-strict",
           "tf32_matmul": bool(args.tf32), "tf32_cudnn": bool(args.tf32),
           "batch_size": 1, "ovsgtr_root": str(OVSGTR_ROOT),
           "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
           "host": host_info(),
           "n_images": args.n_images, "n_warmup": args.n_warmup,
           "params_M": n_params / 1e6,
           "transformer_ms": stats(fwd_ms),
           "postprocess_graph_infer_ms": stats(post_ms),
           "total_ms": stats(tot_ms),
           "fps": 1000.0 / float(np.mean(tot_ms)),
           "boxes_per_img": float(np.mean(n_boxes)),
           "pairs_per_img": float(np.mean(n_pairs)),
           "wall_s": wall, "epoch": info.get("epoch")}

    if args.flops:
        # graph_infer's pair scoring is NOT invisible to the counter: rln_proj and the
        # relation einsum are ordinary aten matmuls, so a TorchDispatchMode sees them
        # even though they are driven from a python loop. Counting the two stages
        # separately is what exposes the quadratic term -- the transformer cost is
        # fixed by image size, the postprocessor cost grows with boxes^2.
        try:
            from torch.utils.flop_counter import FlopCounterMode
            fwd_gf, post_gf, boxes_at = [], [], []
            for t, W, H in imgs[args.n_warmup:args.n_warmup + args.flops_images]:
                samples = nested_tensor_from_tensor_list([t.to(dev)])
                orig = torch.as_tensor([[H, W]], device=dev)
                with torch.no_grad():
                    fc = FlopCounterMode(display=False)
                    with fc:
                        out = model(samples, [target])
                    f_fwd = fc.get_total_flops()
                    fc = FlopCounterMode(display=False)
                    with fc:
                        r = post(out, orig)
                    f_post = fc.get_total_flops()
                fwd_gf.append(f_fwd / 1e9)
                post_gf.append(f_post / 1e9)
                bx = r[0].get("graph", {}).get("pred_boxes")
                boxes_at.append(0 if bx is None else len(bx))
            res["transformer_GFLOPs"] = stats(fwd_gf)
            res["postprocess_graph_infer_GFLOPs"] = stats(post_gf)
            res["total_GFLOPs"] = stats([a + b for a, b in zip(fwd_gf, post_gf)])
            res["flops_boxes_per_img"] = float(np.mean(boxes_at))
            res["flops_n_images"] = len(fwd_gf)
            res["flops_note"] = (
                "FlopCounterMode (matmul/conv/sdpa only, 2*MACs convention), "
                "batch 1, native OvSGTR test resize (short side 800, long side <=1333). "
                "Averaged over flops_n_images real VG150-test images: the transformer "
                "term varies with image shape and the postprocess term with boxes^2.")
            print(f"GFLOPs transformer {res['transformer_GFLOPs']['mean']:.1f} + "
                  f"graph_infer {res['postprocess_graph_infer_GFLOPs']['mean']:.1f} = "
                  f"{res['total_GFLOPs']['mean']:.1f}")
        except Exception as exc:            # noqa: BLE001
            res["flops_error"] = repr(exc)
            print("FLOP counting failed:", exc)

    print(f"\nparams {res['params_M']:.1f}M | boxes/img {res['boxes_per_img']:.1f} | "
          f"pairs/img {res['pairs_per_img']:.0f}")
    print(f"transformer {res['transformer_ms']['mean']:.1f} ms | "
          f"graph_infer {res['postprocess_graph_infer_ms']['mean']:.1f} ms | "
          f"total {res['total_ms']['mean']:.1f} ms ({res['fps']:.1f} FPS)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, sort_keys=True))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
