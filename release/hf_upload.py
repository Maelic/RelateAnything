"""Upload the released models to the Hugging Face Hub, one repository per model.

Layout of ``<org>/<model_id>``:

    model.pth              stripped torch checkpoint (release/strip_checkpoint.py)
    text_student.pt        the distilled predicate text encoder the checkpoint
                           was trained with (+ CLIP tokenizer files)
    predicate_embeddings.npz  the training vocabulary encoded with that student,
                           so full_vocabulary=True is a download, not a re-encode
    relateanything.onnx    ONNX graph, numpy + onnxruntime only        (bundles)
    relateanything_fp16.*  OpenVINO IR for CPUs                        (bundles)
    relateanything.json    export metadata: parity, provenance
    predicate_bank.npz     vocabulary bank (embeddings, thresholds, types)
    thresholds.json        full per-predicate calibration record
    calibration.json       the Platt fit (a, b) behind every threshold
    README.md              GENERATED card (release/make_model_cards.py)

"bundles" marks files that exist only where deploy/build_release.py has run an
ONNX export; a model without one becomes a torch-only repository.

Only manifest entries with status ``final`` are published. The ``-zeroshot``
arms are status ``unreleased`` -- they exist so release_gate.py can measure each
released model against a sibling that never saw HICO-DET -- and are skipped.

Hard exclusions, enforced here rather than by convention:
    * anything matching ``*detector*`` or ``*.pt`` other than the text student.
      The demo detectors derive from ultralytics (AGPL-3.0) and are never
      redistributed (THIRD_PARTY_NOTICES.md); users rebuild them locally.
    * manifest rows whose status is not ``final``.

Idempotent: the hub skips unchanged files. Each upload is tagged with the git
SHA recorded in the export metadata, so a weights revision always names the
code that produced it.

    python release/hf_upload.py --org maelic --dry-run
    python release/hf_upload.py --org maelic --only relsgg-vits16plus
    python release/hf_upload.py --org maelic --weights-dir checkpoints_stripped --private
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EXCLUDE_PATTERNS = ["*detector*"]
ALLOWED_PT = {"text_student.pt"}
BUNDLE_FILES = ["relateanything.onnx", "relateanything_fp16.xml", "relateanything_fp16.bin"]
ALWAYS_FILES = ["relateanything.json", "predicate_bank.npz", "thresholds.json",
                "calibration.json", "README.md"]
#: Written beside model.pth by strip_checkpoint.py; without it every user of
#: full_vocabulary=True re-encodes 19k strings locally.
WEIGHTS_SIDECARS = ["text_student.pt", "predicate_embeddings.npz"]
TOKENIZER_FILES = ["tokenizer.json", "vocab.json", "merges.txt",
                   "tokenizer_config.json", "special_tokens_map.json"]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_weights(m: dict, weights_dir: str | None) -> str | None:
    """model.pth next to the bundle, or <weights_dir>/<run>/model.pth, or the
    stripped checkpoint_last.pth of that run."""
    dist = os.path.join("deploy/dist", m["model_id"])
    cands = [os.path.join(dist, "model.pth")]
    if weights_dir:
        run = os.path.basename(m["run_dir"])
        cands += [os.path.join(weights_dir, run, "model.pth"),
                  os.path.join(weights_dir, m["model_id"], "model.pth"),
                  os.path.join(weights_dir, run, m.get("checkpoint", "checkpoint_last.pth"))]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def gather(m: dict, weights_dir: str | None, torch_only: bool) -> dict:
    """Collect and validate the file set for one model. Raises on holes."""
    mid = m["model_id"]
    dist = os.path.join("deploy/dist", mid)
    files = {}
    w = find_weights(m, weights_dir)
    if not w:
        raise SystemExit(f"[upload] {mid}: no model.pth found. Run release/strip_checkpoint.py "
                         f"--checkpoint <run>/checkpoint_last.pth --out {dist}/model.pth")
    files["model.pth"] = w
    wdir = os.path.dirname(w)
    for fname in WEIGHTS_SIDECARS + TOKENIZER_FILES:
        for d in (wdir, dist):
            p = os.path.join(d, fname)
            if os.path.exists(p):
                files[fname] = p
                break
    if "text_student.pt" not in files:
        raise SystemExit(f"[upload] {mid}: text_student.pt missing next to {w}; "
                         "strip_checkpoint.py copies it there.")
    for fname in ALWAYS_FILES:
        p = os.path.join(dist, fname)
        if not os.path.exists(p):
            raise SystemExit(f"[upload] {mid}: missing {p}. build_release.py writes the "
                             "metadata; make_model_cards.py writes README.md.")
        files[fname] = p
    if not torch_only:
        for fname in BUNDLE_FILES:
            p = os.path.join(dist, fname)
            if os.path.exists(p):
                files[fname] = p
        if "relateanything.onnx" not in files:
            print(f"[upload] {mid}: no ONNX bundle on disk -> torch-only repository")
    for k in files:
        if any(fnmatch.fnmatch(k, pat) for pat in EXCLUDE_PATTERNS):
            raise SystemExit(f"[upload] {mid}: {k} matches an excluded pattern, refusing.")
        if k.endswith(".pt") and k not in ALLOWED_PT:
            raise SystemExit(f"[upload] {mid}: {k} is a.pt file that is not the text student, refusing.")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="deploy/release_manifest.json")
    ap.add_argument("--org", required=True, help="HF namespace, e.g. maelic")
    ap.add_argument("--only", default=None)
    ap.add_argument("--weights-dir", dest="weights_dir", default=None,
                    help="directory holding <run>/model.pth (or stripped checkpoint_last.pth)")
    ap.add_argument("--torch-only", dest="torch_only", action="store_true",
                    help="skip ONNX/OpenVINO files even when present")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    a = ap.parse_args()
    os.chdir(REPO)

    man = json.load(open(a.manifest))
    todo = [m for m in man["models"] if (not a.only or m["model_id"] == a.only)]
    for m in todo:
        mid = m["model_id"]
        if m.get("status") != "final":
            print(f"[upload] {mid}: status={m.get('status')!r} != 'final', skipped")
            continue
        files = gather(m, a.weights_dir, a.torch_only)
        git_sha = json.load(open(files["relateanything.json"])).get("git_sha", "")
        repo_id = m.get("hf_repo") or f"{a.org}/{mid}"
        if not repo_id.startswith(a.org + "/"):
            repo_id = f"{a.org}/{mid}"
        total = 0
        print(f"\n[upload] {repo_id} @ {git_sha[:12]}")
        for k, p in sorted(files.items()):
            sz = os.path.getsize(p); total += sz
            print(f"    {k:24s} {sz/1e6:9.2f} MB  sha256 {sha256(p)[:16]}  <- {p}")
        print(f"    {'total':24s} {total/1e6:9.2f} MB")
        if a.dry_run:
            continue
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id, private=a.private, exist_ok=True, repo_type="model")
        for k, p in files.items():
            api.upload_file(path_or_fileobj=p, path_in_repo=k,
                            repo_id=repo_id, repo_type="model")
        if git_sha:
            try:
                api.create_tag(repo_id, tag=f"git-{git_sha[:12]}", repo_type="model")
            except Exception as e:          # tag exists on re-run
                print(f"    (tag: {type(e).__name__})")
        print(f"[upload] {repo_id} done")


if __name__ == "__main__":
    main()
