"""Fit a checkpoint's deployment calibration and write its `calibration.json`.

WHY THIS EXISTS. `relsgg/scoring.py` reads `calibration.json` from beside a
checkpoint, `deploy/pipeline.py` resolves it, and `deploy/render_operating_point.py`
now refuses to run without it — but until now NOTHING PRODUCED IT. The one file
on disk (the ViT-B full-recipe model's) was assembled by hand from an
`eval_deploy_metrics.py --fit_platt` console dump, so it could not be
reproduced, audited, or made for a second checkpoint. This is that producer.

WHY IT MUST BE REFIT PER CHECKPOINT. `vocab_head.logit_scale/logit_bias` got no
gradient under the ranking loss for eleven versions,
so the absolute score scale is an accident of initialisation and training
dynamics. Rank metrics never see it; a deployment THRESHOLD is nothing but it.
Carrying one model's (a, b) — or one model's tau — to another is therefore a
silent error, not an approximation.

WHAT IS BEING CALIBRATED, AND AGAINST WHAT. The fit is on Haystack's
ADJUDICATED cells: every cell a human explicitly marked true or false, joined
from the negatives sidecar. That makes the output mean

    P(a human adjudicator calls this relation true)

which is the deployment question. Fitting on PSG val emissions instead answers
"P(this triplet appears in a PSG annotation)", which bakes annotation
incompleteness into the number and reads 0.0255 where the truth-calibrated map
reads 0.3810 and the true rate is 0.1102.
Both are recorded; only the adjudicated one is installed.

TWO-FOLD CV IS REPORTED, NOT SHIPPED. The shipped (a, b) is the single fit over
all adjudicated cells; the folds exist to show the out-of-fold ECE, i.e. that
two numbers fitted on one half still calibrate the other. Platt is monotone for
a > 0, so R@K / mR@K / AP / AUC are bit-identical before and after — this buys
ADDRESSABILITY, never accuracy.

    python deploy/fit_calibration.py \
        --checkpoint runs/train/<run>/checkpoint_last.pth
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("HF_HOME", os.path.join(REPO, ".hf_cache"))


def _git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO, text=True).strip()
    except Exception:  # noqa: BLE001 — provenance is best-effort
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pack", default="runs/packed/haystack")
    p.add_argument("--split", default="test")
    p.add_argument("--negatives", default="runs/datamix/haystack_negatives.json")
    p.add_argument("--weights", default="ema")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    # 992 = PipelineConfig.final_budget = max_objects*(max_objects-1), i.e. the
    # EXHAUSTIVE setting the product actually ships, not the eval default of
    # 100. Calibration must be fitted under the emission regime it will be
    # applied in, and exhaustive costs 1.02x.
    p.add_argument("--eval_budget", type=int, default=992)
    p.add_argument("--folds", type=int, default=2)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing calibration.json")
    a = p.parse_args()

    out_path = os.path.join(os.path.dirname(a.checkpoint), "calibration.json")
    if os.path.exists(out_path) and not a.force:
        print(f"!! {out_path} exists — pass --force to overwrite")
        return 1

    import torch  # noqa: PLC0415 — after HF_HOME is set
    from torch.utils.data import DataLoader

    from benchmark import eval_deploy_metrics as EDM  # noqa: PLC0415
    from relsgg.data.dataset import RelationDataset, collate_fn  # noqa: PLC0415
    from relsgg.model.geometry import RelGeomEncoder  # noqa: PLC0415
    from relsgg.text.student import encode_texts_student  # noqa: PLC0415
    from relsgg.vocabulary import TRAIN_TEMPLATES  # noqa: PLC0415
    from relsgg.checkpoint import build_model_from_ckpt  # noqa: PLC0415

    # Identical load path to eval_deploy_metrics.py / eval_zeroshot.py, on
    # purpose: the vocabulary matrix must be encoded with the checkpoint's OWN
    # text student, or the logits being calibrated are not the logits deployed
    #.
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_ckpt(ck, a.weights).to(dev).eval()

    ds = RelationDataset(root=a.pack, split=a.split, resolution=448,
                         max_objects=40)
    names = ds.predicate_names
    ck_args = ck.get("args") or {}
    ck_args = ck_args if isinstance(ck_args, dict) else vars(ck_args)
    E = encode_texts_student(names, ck_args["text_student"],
                             templates=TRAIN_TEMPLATES, device=dev)
    model.vocab_head.set_vocabulary_matrix(names, E)
    model.reparameterize()
    # Same join as eval_haystack.py / eval_deploy_metrics.py: the pack assigns
    # predicate ids by first appearance, so BOTH remappings are mandatory.
    from pathlib import Path  # noqa: PLC0415
    side = json.load(open(a.negatives))
    pid_of = {n: i for i, n in enumerate(names)}
    row_of = {int(Path(f).stem.split("_")[-1]): i
              for i, f in enumerate(ds.file_names)}
    neg_by_index, n_neg = {}, 0
    for img_id, cells in side["by_image_id"].items():
        r = row_of.get(int(img_id))
        if r is None:
            continue
        keep = {(int(s), int(o), pid_of[q]) for s, o, q in cells if q in pid_of}
        if keep:
            neg_by_index[r] = keep
            n_neg += len(keep)
    print(f"federated: {n_neg:,} explicit negative cells over "
          f"{len(neg_by_index):,} images")

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=a.num_workers,
                        pin_memory=True)
    _rec, _flat, _gtc, cells = EDM.collect(
        model, loader, dev, eval_budget=a.eval_budget,
        neg_by_index=neg_by_index)

    # Fit on the ADJUDICATED, ACTUALLY-SCORED cells. Two exclusions, both
    # load-bearing:
    #   `seen == 0` are pairs the sampler never emitted; collect() fills their
    #   z terms with a -30.0 sentinel, and feeding those to a logistic fit
    #   would drag (a, b) toward separating a sentinel rather than calibrating
    #   the model. A triplet that is never emitted is also never thresholded,
    #   so they are out of scope by construction, not by convenience.
    # The score contract is sigmoid(a*(z_pred + z_pair) + b), so the quantity
    # being calibrated is the SUM — matching relsgg/scoring.py exactly rather
    # than re-deriving it here.
    seen = cells["seen"].astype(bool)
    logit = (cells["z_pred"][seen] + cells["z_pair"][seen]).astype(np.float64)
    tp = cells["tp"][seen].astype(np.int32)
    n_pos = int(tp.sum())
    print(f"adjudicated cells {len(cells['tp']):,} of which scored "
          f"{len(tp):,} ({seen.mean():.2%})  -> {n_pos:,} pos / "
          f"{len(tp) - n_pos:,} explicit neg")
    if n_pos < 100 or len(tp) - n_pos < 100:
        print("!! too few of one class to fit a threshold on")
        return 1

    ece_raw = EDM.reliability(1.0 / (1.0 + np.exp(-logit)), tp)[1]
    ab = EDM.fit_platt(logit, tp)

    # Folds are a REPORT on the shipped fit, not the fit itself.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(tp))
    ece_oof, auc_oof = [], []
    for k in range(a.folds):
        held = order[k::a.folds]
        train = np.setdiff1d(order, held, assume_unique=False)
        ab_k = EDM.fit_platt(logit[train], tp[train])
        s_k = EDM.apply_platt(logit[held], *ab_k)
        ece_oof.append(round(float(EDM.reliability(s_k, tp[held])[1]), 4))
        auc_oof.append(round(float(EDM.auc(s_k, tp[held])), 4))

    doc = {
        "a": round(float(ab[0]), 4),
        "b": round(float(ab[1]), 4),
        "score": "sigmoid(a*(pred_logit+pair_logit)+b)",
        "fit_on": (f"{a.pack} {a.split}, {len(tp):,} ADJUDICATED cells "
                   f"({n_pos:,} pos / {len(tp) - n_pos:,} explicit neg), "
                   f"{a.folds}-fold CV reported"),
        "means": ("P(a human adjudicator calls this relation true). This is "
                  "the deployment question and the analog of how a detector is "
                  "calibrated against exhaustively-labelled COCO negatives."),
        "ece_raw": round(float(ece_raw), 4),
        "ece_out_of_fold": ece_oof,
        "auc_out_of_fold": auc_oof,
        "provenance": {
            "checkpoint": a.checkpoint,
            "weights": a.weights,
            "eval_budget": a.eval_budget,
            "produced_by": "deploy/fit_calibration.py",
            "git": _git_rev(),
            "utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        },
    }
    with open(out_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"\nPlatt a={doc['a']} b={doc['b']}   "
          f"ECE {doc['ece_raw']} -> out-of-fold {ece_oof}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
