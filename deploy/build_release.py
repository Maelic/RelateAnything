"""Release driver: manifest entry -> deploy/dist/<model_id>/ artifact set.

Per model (deploy/release_manifest.json):
    1. export    — export_onnx.py --vocab-mode input --check
    2. calibrate — calibrate_thresholds.py (needs a GPU; --skip_calibrate
                   prints the command instead so the driver runs on a CPU)
    3. bank      — build_predicate_bank.py (thresholds folded in)
    4. assemble  deploy/dist/<model_id>/{relateanything.onnx,
                 relateanything.json, predicate_bank.npz, thresholds.json}

Idempotent: steps whose outputs exist are skipped unless --force. Everything
is a subprocess of the same scripts a human would run — the driver adds
ordering and provenance, not new behavior.

    python deploy/build_release.py --only relsgg-vitb16
    python deploy/build_release.py --skip_calibrate      # CPU-only pass
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def sh(cmd: list[str]) -> None:
    print("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=REPO)
    if r.returncode:
        raise SystemExit(f"[release] step failed ({r.returncode}): {cmd[0]}")


def build_one(m: dict, a) -> None:
    mid, run_dir = m["model_id"], m.get("run_dir")
    print(f"\n=== {mid} ({m.get('status')}) ===")
    if not run_dir:
        print("  no run_dir yet — skipping (train it first)")
        return
    # FINAL epoch ships: dev-FINAL predicts OOD better than dev-best and
    # calibration.json is fitted on it.
    ckpt = os.path.join(run_dir, a.checkpoint_name)
    if not os.path.exists(ckpt):
        print(f"  {ckpt} missing — skipping")
        return
    run_name = os.path.basename(run_dir)
    dist = os.path.join("deploy/dist", mid)
    os.makedirs(dist, exist_ok=True)

    # 1. ONNX export (+ parity check, recorded into the metadata json)
    onnx_path = os.path.join(dist, "relateanything.onnx")
    if a.force or not os.path.exists(onnx_path):
        sh([PY, "deploy/export_onnx.py", "--checkpoint", ckpt,
            "--vocab-mode", "input", "--out", onnx_path, "--check"])
    else:
        print(f"  export: {onnx_path} exists, skipping")

    # 2. Threshold calibration (GPU)
    thr = os.path.join("runs/analysis", run_name, "deploy_thresholds.json")
    if os.path.exists(thr) and not a.force:
        print(f"  calibrate: {thr} exists, skipping")
    elif a.skip_calibrate or not torch.cuda.is_available():
        print(f"  calibrate: SKIPPED (no GPU here). Run:\n"
              f"    sbatch --wrap '{PY} deploy/calibrate_thresholds.py "
              f"--checkpoint {ckpt}' "
              f"--gpus-per-node=A40:1 -t 1:00:00")
    else:
        sh([PY, "deploy/calibrate_thresholds.py", "--checkpoint", ckpt])

    # 3. Predicate bank (thresholds folded in when present; loud NaN warning
    #    otherwise — a release bank without calibration is not shippable)
    bank = os.path.join(dist, "predicate_bank.npz")
    if a.force or not os.path.exists(bank) or os.path.exists(thr):
        sh([PY, "deploy/build_predicate_bank.py", "--checkpoint", ckpt,
            "--out", bank])
    if os.path.exists(thr):
        shutil.copy2(thr, os.path.join(dist, "thresholds.json"))

    have = sorted(os.listdir(dist))
    print(f"  dist/{mid}: {have}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="deploy/release_manifest.json")
    ap.add_argument("--only", default=None, help="one model_id")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip_calibrate", action="store_true")
    ap.add_argument("--checkpoint_name", default="checkpoint_last.pth")
    a = ap.parse_args()
    os.chdir(REPO)
    man = json.load(open(a.manifest))
    for m in man["models"]:
        if a.only and m["model_id"] != a.only:
            continue
        # Unreleased arms (the -zeroshot siblings the gate measures against)
        # have no published bundle. Naming one with --only still builds it.
        if not a.only and m.get("status") != "final":
            print(f"\n=== {m['model_id']} ({m.get('status')}) === not released, skipping")
            continue
        build_one(m, a)


if __name__ == "__main__":
    main()
