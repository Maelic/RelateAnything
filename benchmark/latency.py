"""Per-image inference latency for the OVS benchmark.

WHY LATENCY IS A COLUMN AND NOT AN AXIS
---------------------------------------
OVS combines CAPABILITY axes with a harmonic mean. Latency is a COST, and cost
must never enter that mean: a model can drive any cost term to its optimum by
doing less work, so folding speed into a capability composite would let "fast
and useless" outrank "slow and correct". Latency is therefore reported BESIDE
OVS, and the trade is left visible rather than pre-decided. `overall_score.py`
prints ms/img next to OVS and, separately, OVS-per-100ms for anyone who wants
the efficiency view explicitly.

WHAT IS MEASURED
----------------
Per-image wall time of `model(images, boxes, box_counts, targets=None)` at
BATCH SIZE 1 — the deployment-relevant quantity. Batched throughput (img/s at
batch 32) is reported alongside, because the two answer different questions and
a serving stack cares about both.

Inputs are REAL batches from a pack, not random tensors: box count drives the
sampler and the three soft-pooling calls, so a synthetic 40-box image would
overstate the cost of a typical 12-box one. The same fixed set of images is
replayed for every arm, so arms are comparable.

TWO VOCABULARY REGIMES, because the head's cost scales with the deployed
vocabulary and the benchmark uses both:
  closed  V = 50    (VG150 — the A1 setting)
  open    V = 19103 (full training vocabulary — the A3 setting)

STAGE BREAKDOWN via CUDA events on submodules, so the number is actionable
rather than just a verdict. `spatial_pool` runs three times per forward
(objects, unions, contacts) and its events are summed. Stages are measured in
a SEPARATE pass from the headline timing — event recording perturbs the
critical path, so mixing them would bias the total we report.

    python benchmark/latency.py --runs v43_full_5ep v44_full_5ep
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from relsgg.data import RelationDataset, collate_fn                      # noqa: E402
from relsgg.vocabulary import TRAIN_TEMPLATES                          # noqa: E402
from relsgg.checkpoint import build_model_from_ckpt             # noqa: E402

STAGES = ["backbone", "spatial_pool", "sampler", "rel_transformer", "vocab_head"]


def timed(fn, n_warmup: int, n_iter: int) -> list:
    """Wall time per call in ms, CUDA-synchronised, warmup discarded."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return out


def stage_profile(model, batches, n_iter: int) -> dict:
    """Sum of each stage's own CUDA time, ms/img. Separate pass — the events
    perturb the critical path, so this never feeds the headline total.

    Stages are wrapped at the METHOD that the forward pass actually calls, not
    via register_forward_hook: `backbone.extract(...)` and
    `vocab_head.score_query_dual(...)` are invoked directly rather than through
    __call__, so module hooks never fire on them and the two most interesting
    stages silently report 0.0.
    """
    acc = defaultdict(float)
    # (attribute owner, method name, stage label)
    targets = [(model.backbone, "extract", "backbone"),
               (model.spatial_pool, "forward", "spatial_pool"),
               (model.sampler, "forward", "sampler"),
               (model.rel_transformer, "forward", "rel_transformer"),
               (model.vocab_head, "score_query", "vocab_head"),
               (model.vocab_head, "score_query_dual", "vocab_head")]
    originals = []

    def wrap(owner, meth, label):
        fn = getattr(owner, meth, None)
        if fn is None:
            return
        originals.append((owner, meth, fn))

        def timed_call(*args, **kw):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn(*args, **kw)
            e.record()
            acc[("__pending__", label)] = acc.get(("__pending__", label), [])
            acc[("__pending__", label)].append((s, e))
            return out
        setattr(owner, meth, timed_call)

    for owner, meth, label in targets:
        wrap(owner, meth, label)
    try:
        for i in range(n_iter):
            for k in [k for k in acc if isinstance(k, tuple)]:
                acc.pop(k)
            im, bx, bc = batches[i % len(batches)]
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                model(im, bx, bc, targets=None)
            torch.cuda.synchronize()
            for k in [k for k in acc if isinstance(k, tuple)]:
                label = k[1]
                for s, e in acc[k]:
                    acc[label] = acc.get(label, 0.0) + s.elapsed_time(e)
    finally:
        for owner, meth, fn in originals:
            setattr(owner, meth, fn)
        for k in [k for k in list(acc) if isinstance(k, tuple)]:
            acc.pop(k)
    return {k: v / n_iter for k, v in acc.items() if isinstance(k, str)}


def summarize(ms: list) -> dict:
    a = np.asarray(ms, dtype=float)
    return {"mean": float(a.mean()), "min": float(a.min()), "max": float(a.max()),
            "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "std": float(a.std()), "n": int(a.size)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--pack", default="runs/packed/vg150")
    p.add_argument("--split", default="test")
    p.add_argument("--n_images", type=int, default=64,
                   help="distinct real images replayed (identical across arms)")
    p.add_argument("--n_warmup", type=int, default=20)
    p.add_argument("--n_iter", type=int, default=200)
    p.add_argument("--eval_budget", type=int, default=400,
                   help="pair budget. DEFAULT 400 = what eval_zeroshot uses, so "
                        "the latency matches the accuracy numbers it sits beside.")
    p.add_argument("--deploy_budget", type=int, default=128,
                   help="second budget to report (the trained final_budget)")
    p.add_argument("--batch_throughput", type=int, default=32)
    p.add_argument("--img_size", type=int, default=448)
    p.add_argument("--ckpt_root", default="runs/train",
                   help="directory holding the run dirs (each with checkpoint_last.pth)")
    p.add_argument("--synthetic_vocab", action="store_true",
                   help="if the checkpoint's text student / pred_embeds are missing, "
                        "substitute a shape-correct stand-in matrix. Cost is a function "
                        "of the vocabulary SHAPE only, so this is exact for latency and "
                        "FLOPs and meaningless for anything else -- the output json is "
                        "stamped synthetic_vocab: true. See benchmark/vocab_stub.py")
    p.add_argument("--out", default="runs/benchmark/latency.json")
    a = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("latency must be measured on the GPU it will be quoted for")
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(0)}")

    ds = RelationDataset(root=a.pack, split=a.split, resolution=a.img_size,
                         max_objects=40)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn,
                        num_workers=4)
    batches, counts = [], []
    for i, (im, bx, bc, _t) in enumerate(loader):
        if i >= a.n_images:
            break
        batches.append((im.to(device), bx.to(device), bc.to(device)))
        counts.append(int(bc.sum()))
    # The batched input is re-collated rather than concatenated from the bs1
    # tensors: collate_fn pads boxes to the max box count IN THE BATCH, so at
    # batch 1 every image carries a different max_N and they cannot be stacked.
    bloader = DataLoader(ds, batch_size=a.batch_throughput, shuffle=False,
                         collate_fn=collate_fn, num_workers=4)
    bim, bbx, bbc, _ = next(iter(bloader))
    big = (bim.to(device), bbx.to(device), bbc.to(device))
    print(f"{len(batches)} real images from {a.pack}/{a.split}, "
          f"{np.mean(counts):.1f} boxes/img (min {min(counts)}, max {max(counts)})")

    from relsgg.text.student import encode_texts_student
    from benchmark.vocab_stub import real_vocabulary, stub_vocabulary
    results = {}
    for run in a.runs:
        # checkpoint_LAST first: runs launched with NOSAVEBEST=1 write only
        # _last (the FINAL-epoch protocol ranks on it and the chained OVS evals
        # pin it), so a _best-only lookup silently SKIPS every such arm — which
        # is how the ConvNeXt row went missing from this table once. Timing is a
        # property of the architecture, not the weights, so which epoch is loaded
        # does not change the measurement; only whether the arm gets measured.
        ck_path = next((p for p in (f"{a.ckpt_root}/{run}/checkpoint_last.pth",
                                    f"{a.ckpt_root}/{run}/checkpoint_best.pth")
                        if os.path.exists(p)), "")
        if not ck_path:
            print(f"!! {run}: no checkpoint_last or _best, skipped")
            continue
        print(f"[{run}] {os.path.basename(ck_path)}")
        ckpt = torch.load(ck_path, map_location="cpu", weights_only=False)
        model = build_model_from_ckpt(ckpt, "ema").to(device).eval()
        ck_args = ckpt.get("args") or {}
        ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
        ts = ck_args.get("text_student") or ""
        pe = ck_args.get("pred_embeds") or ""

        # OPEN regime: the full training vocabulary. Its names live in the checkpoint
        # itself (`pred_names`), so the open-vocab SHAPE is recoverable even when the
        # embedding npz the run was trained against is not on this machine.
        open_names = [str(q) for q in ckpt.get("pred_names", [])]
        real = real_vocabulary(open_names, ts, pe, device, TRAIN_TEMPLATES)
        if real is not None:
            open_names, open_E = real
            closed_E = encode_texts_student(ds.predicate_names, ts,
                                            templates=TRAIN_TEMPLATES, device=device)
            synthetic = False
        elif a.synthetic_vocab:
            dim = int(ck_args.get("text_dim") or 512)
            _, open_E = stub_vocabulary(open_names, dim, device)
            _, closed_E = stub_vocabulary(ds.predicate_names, dim, device)
            synthetic = True
            print(f"  [synthetic vocab] text student '{ts}' and embeddings '{pe}' are "
                  f"absent; using shape-correct stand-ins (V={len(open_names)}/"
                  f"{len(ds.predicate_names)}, dim={dim}). Cost only.")
        else:
            raise SystemExit(
                f"{run}: neither '{pe}' nor the text student '{ts}' is on this machine, "
                f"so the vocabulary matrix cannot be built. Pass --synthetic_vocab to "
                f"measure cost with a shape-correct stand-in.")
        vocabs = {"closed": (ds.predicate_names, closed_E),
                  "open": (open_names, open_E)}

        res = {"n_params_M": sum(q.numel() for q in model.parameters()) / 1e6}
        for regime, (names, E) in vocabs.items():
            model.vocab_head.set_vocabulary_matrix(names, E)
            model.reparameterize()
            for tag, budget in (("eval", a.eval_budget),
                                ("deploy", a.deploy_budget)):
                model.sampler.final_budget = min(budget, model.sampler.geo_budget)
                k = 0

                @torch.no_grad()
                def one():
                    nonlocal k
                    im, bx, bc = batches[k % len(batches)]
                    k += 1
                    with torch.amp.autocast("cuda", enabled=True,
                                            dtype=torch.bfloat16):
                        model(im, bx, bc, targets=None)

                res[f"{regime}_{tag}_bs1"] = summarize(
                    timed(one, a.n_warmup, a.n_iter))
                res[f"{regime}_{tag}_bs1"]["V"] = len(names)

            # batched throughput at the eval budget
            model.sampler.final_budget = min(a.eval_budget,
                                             model.sampler.geo_budget)

            @torch.no_grad()
            def batched():
                with torch.amp.autocast("cuda", enabled=True,
                                        dtype=torch.bfloat16):
                    model(big[0], big[1], big[2], targets=None)

            bt = timed(batched, 5, 30)
            res[f"{regime}_batch{a.batch_throughput}_img_s"] = (
                a.batch_throughput / (statistics.fmean(bt) / 1000.0))

            with torch.no_grad():
                res[f"{regime}_stages_ms"] = stage_profile(model, batches, 30)

        res["synthetic_vocab"] = synthetic
        results[run] = res
        s = res[f"open_eval_bs1"]
        c = res[f"closed_eval_bs1"]
        st = res["open_stages_ms"]
        print(f"\n[{run}] {res['n_params_M']:.1f}M params")
        print(f"  bs1 closed V={c['V']:<6d} {c['mean']:6.1f} ms  "
              f"(min {c['min']:.1f} / max {c['max']:.1f}, p95 {c['p95']:.1f})")
        print(f"  bs1 open   V={s['V']:<6d} {s['mean']:6.1f} ms  "
              f"(min {s['min']:.1f} / max {s['max']:.1f}, p95 {s['p95']:.1f})")
        print(f"  stages (open): "
              + "  ".join(f"{n} {st.get(n, 0.0):.1f}" for n in STAGES))
        print(f"  batch{a.batch_throughput}: "
              f"{res[f'open_batch{a.batch_throughput}_img_s']:.1f} img/s")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"device": torch.cuda.get_device_name(0),
               "synthetic_vocab": any(r.get("synthetic_vocab") for r in results.values()),
               "pack": a.pack, "split": a.split, "n_images": len(batches),
               "boxes_per_img_mean": float(np.mean(counts)),
               "n_warmup": a.n_warmup, "n_iter": a.n_iter,
               "eval_budget": a.eval_budget, "deploy_budget": a.deploy_budget,
               "note": "latency is a COST, reported beside OVS and never inside "
                       "its harmonic mean",
               "runs": results}, open(a.out, "w"), indent=2)
    print(f"\nsaved → {a.out}")


if __name__ == "__main__":
    main()
