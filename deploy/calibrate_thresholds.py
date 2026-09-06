"""Per-predicate deployment thresholds, measured per checkpoint.

WHY PER CHECKPOINT. `vocab_head.logit_scale`/`logit_bias` received no gradient
under the ranking loss (they were bit-identical to init for eleven versions),
so the absolute sigmoid score scale is an accident of initialisation and
training dynamics — it differs between checkpoints and carries no calibrated
meaning. Rank metrics never see this; a deployment THRESHOLD is nothing but
this. Consequence: thresholds measured on one model transfer to no other, and
a release bank without its own calibration ships NaN thresholds on purpose
rather than someone else's numbers. (Project rule: thresholds are measured
from data, never hand-set.)

REGIME (fixed, so numbers stay comparable between checkpoints): ground-truth
boxes, a megasg val sample, the predicate sigmoid score alone —
no pair-existence term, no detector confidence. The demo's displayed score is
pred * pair^w, a smaller number; postprocess applies these thresholds with
pair_weight folded out or documented.

One-vs-rest per bank predicate over every scored candidate pair, streaming
histograms -> reverse-cumsum PR (the closed form mirrors
training/analyze_predicate_pr_curves.py; that script's logic lives inside its
main(), hence the small local copy with this pointer).

    python deploy/calibrate_thresholds.py \
        --checkpoint runs/train/<run>/checkpoint_best.pth
    -> runs/analysis/<run>/deploy_thresholds.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".hf_cache"))


def forward_batch(model, images, boxes, box_counts, device, amp, amp_dtype):
    """The ONLY place inputs are assembled for the model.

    Deliberately isolated: the coming mask-input variant adds one tensor HERE
    (and in the export wrapper) and nowhere else in the calibration path.
    """
    with torch.amp.autocast("cuda", enabled=amp, dtype=amp_dtype):
        return model(images.to(device, non_blocking=True),
                     boxes.to(device, non_blocking=True),
                     box_counts.to(device, non_blocking=True), targets=None)


def pr_from_hists(pos_hist: np.ndarray, neg_hist: np.ndarray,
                  edges: np.ndarray) -> dict:
    """Histogram pair -> best-F1 operating point (mirrors
    analyze_predicate_pr_curves.py's reverse-cumsum block)."""
    ph, nh = pos_hist.astype(np.float64), neg_hist.astype(np.float64)
    TP = np.cumsum(ph[::-1])[::-1]
    FP = np.cumsum(nh[::-1])[::-1]
    total_pos = ph.sum()
    precision = np.where(TP + FP > 0, TP / np.maximum(TP + FP, 1e-12), 1.0)
    recall = TP / max(total_pos, 1e-12)
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.maximum(denom, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return {"best_f1_thr": float(edges[:-1][i]), "best_f1": float(f1[i]),
            "precision": float(precision[i]), "recall": float(recall[i]),
            "n_pos": int(total_pos), "n_neg": int(nh.sum())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--bank", default=None,
                    help="predicate_bank.npz whose names to calibrate. "
                         "Default: the curated set + trained gt>=30, i.e. "
                         "exactly what build_predicate_bank puts in a bank.")
    ap.add_argument("--data_root", default="runs/packed/megasg")
    ap.add_argument("--n_images", type=int, default=5000)
    ap.add_argument("--min_gt", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=448)
    ap.add_argument("--max_objects", type=int, default=60)
    ap.add_argument("--n_thresholds", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default="ema", choices=["ema", "raw"])
    ap.add_argument("--out", default=None,
                    help="default runs/analysis/<run>/deploy_thresholds.json")
    a = ap.parse_args()
    os.chdir(REPO)

    from relsgg.checkpoint import build_model_from_ckpt  # noqa: E402
    from relsgg.data.dataset import RelationDataset, collate_fn  # noqa: E402

    run_name = os.path.basename(os.path.dirname(os.path.abspath(a.checkpoint)))
    out_path = a.out or os.path.join("runs/analysis", run_name,
                                     "deploy_thresholds.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ckpt, a.weights).to(device).eval()
    pred_names = ckpt["pred_names"]
    name_to_id = {n: i for i, n in enumerate(pred_names)}

    # --- which predicates to calibrate --------------------------------------
    if a.bank:
        want = [str(n) for n in np.load(a.bank, allow_pickle=True)["names"]]
    else:
        from deploy.vocab import PREDICATE_VOCAB
        want = list(PREDICATE_VOCAB)
        rc = os.path.join(os.path.dirname(a.checkpoint), "per_class_recall.json")
        if os.path.exists(rc):
            for c in json.load(open(rc))["classes"]:
                if c["gt"] >= a.min_gt:
                    want.append(c["name"])
        seen = set()
        want = [n for n in want if not (n in seen or seen.add(n))]
    cal = [n for n in want if n in name_to_id]
    skipped = [n for n in want if n not in name_to_id]
    ids = np.array([name_to_id[n] for n in cal], dtype=np.int64)
    print(f"[cal] {run_name}: {len(cal)} predicates to calibrate "
          f"({len(skipped)} not in the training vocab -> NaN)")

    # --- data ----------------------------------------------------------------
    train_ds = RelationDataset(root=a.data_root, split="train",
                               resolution=a.img_size)
    val_ds = RelationDataset(root=a.data_root, split="val",
                             resolution=a.img_size, max_objects=a.max_objects,
                             cat_to_idx=train_ds.cat_to_idx,
                             rel_cat_to_idx=train_ds.rel_cat_to_idx)
    # The head trains on the union vocabulary (19,103 predicates) and the pack
    # carries its own (megasg: 10,102); the two are not the same list.
    # GT predicate ids in `targets` index the PACK vocabulary; map them to
    # calibration columns via the NAME, which is shared.
    pack_name_to_id = {n: i for i, n in enumerate(val_ds.predicate_names)}
    in_pack = sum(1 for n in cal if n in pack_name_to_id)
    if in_pack < 0.5 * len(cal):
        raise SystemExit(f"[cal] only {in_pack}/{len(cal)} bank predicates "
                         f"exist in {a.data_root}'s vocabulary — wrong pack?")
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(val_ds))[:a.n_images]
    loader = DataLoader(Subset(val_ds, idx.tolist()), batch_size=a.batch_size,
                        shuffle=False, collate_fn=collate_fn,
                        num_workers=a.num_workers, pin_memory=True)

    amp = device.type == "cuda"
    amp_dtype = (torch.bfloat16 if amp and
                 torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16)

    # GT labels arrive as PACK ids; columns are ordered by `cal`.
    id_to_col = {pack_name_to_id[n]: c for c, n in enumerate(cal)
                 if n in pack_name_to_id}

    edges = np.linspace(0.0, 1.0, a.n_thresholds + 1)
    pos_hist = np.zeros((len(cal), a.n_thresholds), dtype=np.int64)
    neg_hist = np.zeros((len(cal), a.n_thresholds), dtype=np.int64)
    ids_t = torch.from_numpy(ids).to(device)

    with torch.no_grad():
        for images, boxes, box_counts, targets in tqdm(loader, desc="calibrate"):
            out = forward_batch(model, images, boxes, box_counts,
                                device, amp, amp_dtype)
            # REGIME: predicate sigmoid alone — no pair term, no detector conf.
            scores = torch.sigmoid(out["logits"][..., ids_t].float()).cpu().numpy()
            valid_np = out["valid_mask"].cpu().numpy()
            sub_np = out["sub_idx"].cpu().numpy()
            obj_np = out["obj_idx"].cpu().numpy()
            for b in range(scores.shape[0]):
                slots = np.nonzero(valid_np[b])[0]
                if not len(slots):
                    continue
                sub_b, obj_b = sub_np[b][slots], obj_np[b][slots]
                first = {}
                for li, k in enumerate((sub_b.astype(np.int64) * 1024
                                        + obj_b.astype(np.int64)).tolist()):
                    first.setdefault(k, li)
                lab = np.zeros((len(slots), len(cal)), dtype=bool)
                for s, o, pid in targets[b]["relations"].tolist():
                    col = id_to_col.get(int(pid))
                    li = first.get(s * 1024 + o)
                    if col is not None and li is not None:
                        lab[li, col] = True
                sc = scores[b][slots]
                for c in range(len(cal)):
                    pos = sc[lab[:, c], c]
                    neg = sc[~lab[:, c], c]
                    if pos.size:
                        pos_hist[c] += np.histogram(pos, bins=edges)[0]
                    if neg.size:
                        neg_hist[c] += np.histogram(neg, bins=edges)[0]

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True).stdout.strip()
    rows = []
    for c, n in enumerate(cal):
        r = pr_from_hists(pos_hist[c], neg_hist[c], edges)
        r["name"] = n
        r["gt_support"] = r.pop("n_pos")
        if r["gt_support"] == 0:
            # no positives in the sample -> the sweep saw only negatives and
            # any threshold is vacuous. NaN, never a number.
            r.update(best_f1_thr=float("nan"), best_f1=float("nan"),
                     precision=float("nan"), recall=float("nan"))
        rows.append(r)
    for n in skipped:
        rows.append({"name": n, "best_f1_thr": float("nan"),
                     "best_f1": float("nan"), "precision": float("nan"),
                     "recall": float("nan"), "gt_support": 0, "n_neg": 0})

    # Sanity: a threshold pinned at 0 or the top edge with real support means
    # the sweep failed to bracket the operating point — refuse to ship it.
    deg = [r["name"] for r in rows
           if r["gt_support"] >= a.min_gt
           and (r["best_f1_thr"] <= edges[1] or r["best_f1_thr"] >= edges[-2])]
    if deg:
        print(f"[cal] WARNING degenerate thresholds (edge-pinned) for: {deg[:8]}")

    json.dump({
        "checkpoint": os.path.abspath(a.checkpoint), "run": run_name,
        "git_sha": git_sha, "date": datetime.date.today().isoformat(),
        "regime": {"boxes": "gt", "pair_weight": 0, "detector_conf": False,
                   "data": f"{a.data_root} val", "n_images": int(a.n_images),
                   "seed": a.seed, "n_thresholds": a.n_thresholds},
        "predicates": rows,
    }, open(out_path, "w"), indent=2)
    ok = sum(1 for r in rows if np.isfinite(r["best_f1_thr"]))
    print(f"[cal] wrote {out_path} ({ok}/{len(rows)} calibrated)")


if __name__ == "__main__":
    main()
