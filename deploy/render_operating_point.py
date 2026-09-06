"""Render real frames at the CHOSEN deployment threshold, end to end.

Not a qualitative cherry-pick and not an oracle-box visualisation: this runs
`deploy.pipeline.ParallelScenePipeline` exactly as the product does —
YOLOE detections (no GT boxes), max_objects/geo_budget/final_budget from
PipelineConfig, the calibrated score contract — and applies the single
threshold chosen by benchmark/eval_operating_point.py. What you see is what a
user would see.

The only deliberate departure from a webcam run: images come from the PSG val
split so GT is available, which lets each predicted edge be marked against the
annotation. That marking is INFORMATIVE, NOT A VERDICT — PSG annotates a
fraction of true relations, so an edge marked "not annotated" is frequently
correct and simply unlabelled. It is
drawn in a third, neutral style for exactly that reason; treating it as red
would reproduce the error this project spent a day disproving.

    python deploy/render_operating_point.py --tau 0.47 --n 12 \
        --out runs/calib/render
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None,:, 0])
    y1 = np.maximum(a[:, None, 1], b[None,:, 1])
    x2 = np.minimum(a[:, None, 2], b[None,:, 2])
    y2 = np.minimum(a[:, None, 3], b[None,:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None,:] - inter + 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tau", type=float, default=None,
                   help="threshold; default = the max-F1(calibrated) tau in "
                        "--op_json, i.e. the one the analysis chose")
    p.add_argument("--op_json", default="runs/calib/op_psg.json")
    # Which checkpoint the product runs. PipelineConfig.ckpt still defaults to
    # the ViT-B full-recipe model, so leaving this unset reproduces every
    # earlier render byte-for-byte. Point it elsewhere and the calibration must
    # come with it: tau and the Platt (a, b) are properties of a checkpoint's
    # logit_scale/logit_bias, which got no gradient under the ranking loss and
    # are therefore an accident of init per model.
    # Reusing one model's tau on another is the mistake this flag exists to make
    # visible, hence the is_calibrated assertion below.
    p.add_argument("--checkpoint", default=None,
                   help="relation checkpoint; default = PipelineConfig.ckpt")
    p.add_argument("--data_root", default="runs/packed/psg")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--iou", type=float, default=0.5,
                   help="IoU for matching a DETECTION to a GT box, so a "
                        "predicted edge can be compared with the annotation")
    p.add_argument("--out", default="runs/calib/render")
    a = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from relsgg.data.dataset import RelationDataset
    from deploy.pipeline import ParallelScenePipeline, PipelineConfig

    tau = a.tau
    if tau is None:
        tau = json.load(open(a.op_json))["best"]["f1_calibrated"]["tau"]
    print(f"threshold tau = {tau:.4f}")

    os.makedirs(a.out, exist_ok=True)
    ds = RelationDataset(root=a.data_root, split=a.split, resolution=448,
                         max_objects=32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pcfg = PipelineConfig(
            det_weights="checkpoints/detectors/yoloe-11m-seg-pf.pt",
            overlap=True)
        if a.checkpoint:
            pcfg.ckpt = a.checkpoint
        pipe = ParallelScenePipeline(pcfg)
    print(f"pipeline: ckpt={pipe.cfg.ckpt}")
    print(f"pipeline: max_objects={pipe.cfg.max_objects} "
          f"geo_budget={pipe.cfg.geo_budget} "
          f"final_budget={pipe.cfg.final_budget}")
    print(f"contract: {pipe._contract.describe()}")
    # Hard stop rather than a warning: this script's entire output is "what a
    # user sees at tau", and against the raw head ~97% of scores sit in
    # [0.9, 1.0), so an uncalibrated render silently shows a threshold that
    # does nothing. ParallelScenePipeline only warns, because ranking callers
    # are legitimately uncalibrated.
    if not pipe._contract.is_calibrated:
        raise SystemExit(
            f"!! no calibration.json next to {pipe.cfg.ckpt} — tau would be "
            "meaningless. Fit one with deploy/fit_calibration.py.")

    tally = defaultdict(int)
    made = 0
    for i in range(a.start, min(a.start + a.n * 4, len(ds))):
        img, bx, tgt = ds[i]
        rels = tgt.get("relations")
        if rels is None or len(rels) == 0:
            continue
        frame = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:,:,::-1].copy()
        H, W = frame.shape[:2]
        res = pipe(frame, top_k=40, score_thr=float(tau))
        if not res.triplets:
            continue

        # map detections -> GT boxes so each edge can be marked
        gb = bx.numpy()
        gt_xyxy = np.stack([(gb[:, 0] - gb[:, 2] / 2) * W, (gb[:, 1] - gb[:, 3] / 2) * H,
                            (gb[:, 0] + gb[:, 2] / 2) * W, (gb[:, 1] + gb[:, 3] / 2) * H], 1)
        M = _iou_matrix(np.asarray(res.boxes_xyxy, np.float32), gt_xyxy)
        det2gt = {d: int(M[d].argmax()) for d in range(len(M))
                  if M.shape[1] and M[d].max() >= a.iou}
        gt_pairs = defaultdict(set)
        for r in rels:
            gt_pairs[(int(r[0]), int(r[1]))].add(int(r[2]))

        edges = []
        for t in res.triplets:
            si, pred, oi, s = (t[0], t[1], t[2], t[3]) if not hasattr(t, "score") \
                else (t.subject_idx, t.predicate, t.object_idx, t.score)
            gs, go = det2gt.get(int(si)), det2gt.get(int(oi))
            if gs is None or go is None:
                kind = "unmatched_box"
            else:
                g = gt_pairs.get((gs, go))
                if g is None:
                    kind = "not_annotated"
                else:
                    names = ds.predicate_names
                    kind = "correct" if pred in {names[c] for c in g} else "wrong_predicate"
            tally[kind] += 1
            edges.append((int(si), pred, int(oi), float(s), kind))

        # ---- draw ----
        fig, ax = plt.subplots(figsize=(11, 7.2), dpi=120)
        ax.imshow(frame[:,:,::-1])
        ax.axis("off")
        STYLE = {"correct": ("#2E7D32", "-"), "wrong_predicate": ("#EF6C00", "-"),
                 "not_annotated": ("#5C6BC0", (0, (4, 3))),
                 "unmatched_box": ("#78909C", (0, (1, 3)))}
        for k, b in enumerate(res.boxes_xyxy):
            ax.add_patch(Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                   fill=False, ec="#FFFFFF", lw=1.6, alpha=.85))
            lab = res.labels[k] if k < len(res.labels) else str(k)
            ax.text(b[0] + 2, b[1] + 12, f"{k}:{lab}", color="#FFFFFF", fontsize=7,
                    bbox=dict(fc="#00000099", ec="none", pad=1.2))
        cen = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        for j, (si, pred, oi, s, kind) in enumerate(edges[:14]):
            if si >= len(res.boxes_xyxy) or oi >= len(res.boxes_xyxy):
                continue
            c, ls = STYLE[kind]
            x0, y0 = cen(res.boxes_xyxy[si]); x1, y1 = cen(res.boxes_xyxy[oi])
            ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), color=c, lw=1.8,
                                         linestyle=ls, alpha=.9,
                                         arrowstyle="-|>", mutation_scale=13,
                                         connectionstyle=f"arc3,rad={0.12 + 0.05*(j%3)}"))
            ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"{pred} {s:.2f}", color="#fff",
                    fontsize=7.5, ha="center", va="center",
                    bbox=dict(fc=c, ec="none", alpha=.92, pad=1.4))
        n_ok = sum(1 for e in edges if e[4] == "correct")
        ax.set_title(f"deployment @ tau={tau:.3f}  |  YOLOE {len(res.boxes_xyxy)} boxes, "
                     f"{len(edges)} edges above threshold, {n_ok} match PSG GT  |  "
                     f"{res.timing.total:.0f} ms", fontsize=9)
        fig.tight_layout()
        fp = os.path.join(a.out, f"frame_{i:05d}.png")
        fig.savefig(fp, bbox_inches="tight"); plt.close(fig)
        made += 1
        print(f"  {fp}  boxes {len(res.boxes_xyxy):2d}  edges {len(edges):2d}  "
              f"correct {n_ok}")
        if made >= a.n:
            break

    tot = sum(tally.values())
    print(f"\nedge tally over {made} frames ({tot} edges above tau):")
    for k in ("correct", "wrong_predicate", "not_annotated", "unmatched_box"):
        print(f"  {k:16s} {tally[k]:5d}  {tally[k]/max(tot,1):6.1%}")
    print("\n'not_annotated' is NOT an error rate: PSG annotates a fraction of")
    print("true relations, so much of that bucket is correct-but-unlabelled.")
    json.dump({"tau": tau, "frames": made, "tally": dict(tally)},
              open(os.path.join(a.out, "tally.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
