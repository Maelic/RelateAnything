"""Upload the two dataset repositories the paper links.

``--ra4m``   -> <org>/RA-4M          the corpus: relation annotations, statistics,
                                     the image manifest, the extractor, the three
                                     generator prompts, and the training-ready packs
``--bench``  -> <org>/OV-SGG-Bench   the evaluation packs (no images), the
                                     negatives and cell files the scorers read, the
                                     text student and the training-side artifacts
                                     the shipped recipe names

Everything is read from ``$RA_RUNS`` (default ``runs/``), so run this from the
training checkout. Pack ``meta.json`` files carry the image root of the machine
that built them; the copies uploaded here have that root rewritten to
``datasets/<name>/<split>`` so a downloaded pack points at ``$RA_DATASETS``.

    python release/upload_datasets.py --ra4m --bench --dry-run
    python release/upload_datasets.py --org maelic --ra4m --private
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from relsgg.paths import RUNS, DATAMIX, PACKED  # noqa: E402

CARDS = REPO / "release" / "cards"

# ---- RA-4M -----------------------------------------------------------------
RA4M_FILES = {
    # in repo                              on disk
    "annotations/ra4m_train.jsonl.gz":     RUNS / "vllm_generate/megasg_sgg_train_full.jsonl",
    "annotations/ra4m_val.jsonl.gz":       RUNS / "vllm_generate/megasg_sgg_val_full.jsonl",
    "annotations/ra4m_train.stats.json":   RUNS / "vllm_generate/megasg_sgg_train_full.stats.json",
    "annotations/ra4m_val.stats.json":     RUNS / "vllm_generate/megasg_sgg_val_full.stats.json",
    "images/ra4m_image_manifest.tsv.gz":   REPO / "release/ra4m/ra4m_image_manifest.tsv.gz",
    "images/ra4m_image_filenames.txt.gz":  REPO / "release/ra4m/ra4m_image_filenames.txt.gz",
    "images/extract_from_mega1m.py":       REPO / "release/ra4m/extract_from_mega1m.py",
    "images/README.md":                    REPO / "release/ra4m/README.md",
    "prompts/iter_20.txt":                 REPO / "datagen/prompts/iter_20.txt",
    "prompts/sgg26b_grow_v1.txt":          REPO / "datagen/prompts/sgg26b_grow_v1.txt",
    "prompts/sgg26b_spatial_v3.txt":       REPO / "datagen/prompts/sgg26b_spatial_v3.txt",
    "README.md":                           CARDS / "RA-4M.md",
}
RA4M_PACKS = ["megasg_clean", "vg_raw", "hicodet", "megasg_proxy50k"]

# ---- OV-SGG-Bench ----------------------------------------------------------
BENCH_PACKS = ["vg150", "psg", "indoorvg", "hicodet", "haystack",
               "spatialsense", "spatialsense_test", "spatialsense_valid"]
BENCH_FILES = {
    "datamix/haystack_negatives.json":       DATAMIX / "haystack_negatives.json",
    "datamix/hicodet_negatives.json":        DATAMIX / "hicodet_negatives.json",
    "datamix/hicodet_negatives_train.json":  DATAMIX / "hicodet_negatives_train.json",
    "datamix/spatialsense_test_cells.json":  DATAMIX / "spatialsense_test_cells.json",
    "datamix/spatialsense_valid_cells.json": DATAMIX / "spatialsense_valid_cells.json",
    "datamix/indoorvg_holdout.json":         DATAMIX / "indoorvg_holdout.json",
    "datamix/registry.json":                 DATAMIX / "registry.json",
    "datamix/vg2coco.json":                  DATAMIX / "vg2coco.json",
    "datamix/psg2coco.json":                 DATAMIX / "psg2coco.json",
    "text_student_v2_512/student.pt":        PACKED / "text_student_v2_512/student.pt",
    "text_student_v2_512/student.history.json": PACKED / "text_student_v2_512/student.history.json",
    "text_student_v2_512/tau_calibration.json": PACKED / "text_student_v2_512/tau_calibration.json",
    "datamix_v22/pair_opportunity.npz":      PACKED / "datamix_v22/pair_opportunity.npz",
    "datamix_v22/text_space/soft_supervision.npz":  PACKED / "datamix_v22/text_space/soft_supervision.npz",
    "datamix_v22/text_space/syn_kernel_v2.npz":     PACKED / "datamix_v22/text_space/syn_kernel_v2.npz",
    "datamix_v22/text_space/pred_embeds_studentv2_512_photo.npz": PACKED / "datamix_v22/text_space/pred_embeds_studentv2_512_photo.npz",
    "datamix_v22/text_space/canonical_groups.json": PACKED / "datamix_v22/text_space/canonical_groups.json",
    "datamix_v22/text_space/union_predicates.json": PACKED / "datamix_v22/text_space/union_predicates.json",
    "datamix_v22/text_space/union_categories.json": PACKED / "datamix_v22/text_space/union_categories.json",
    "datamix_v22/text_space/union_meta.json":       PACKED / "datamix_v22/text_space/union_meta.json",
    "README.md":                             CARDS / "OV-SGG-Bench.md",
}

PRIVATE_MARKERS = ("/mimer/", "/proj/", "/home/", "/cephyr/", "/tmp/")


def clean_meta(meta: dict, name: str, split: str) -> dict:
    """Rewrite machine-local roots in a pack's meta.json."""
    meta = dict(meta)
    for k in ("img_dir", "ann_source"):
        v = meta.get(k)
        if isinstance(v, str) and any(m in v for m in PRIVATE_MARKERS):
            tail = v.split("DATASETS/")[-1] if "DATASETS/" in v else os.path.basename(v)
            meta[k] = f"datasets/{tail}"
    return meta


def stage_pack(name: str, dst_root: Path, staged: dict) -> None:
    src = PACKED / name
    if not src.is_dir():
        raise SystemExit(f"[datasets] pack not found: {src}")
    for split_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for f in sorted(split_dir.iterdir()):
            rel = f"packs/{name}/{split_dir.name}/{f.name}"
            if f.name == "meta.json":
                out = dst_root / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                json.dump(clean_meta(json.load(open(f)), name, split_dir.name), open(out, "w"))
                staged[rel] = out
            else:
                staged[rel] = f


def stage_files(files: dict, dst_root: Path, staged: dict) -> None:
    for rel, src in files.items():
        src = Path(src)
        if not src.exists():
            raise SystemExit(f"[datasets] missing: {src}  (wanted as {rel})")
        if rel.endswith(".gz") and not src.name.endswith(".gz"):
            out = dst_root / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                with open(src, "rb") as fi, gzip.open(out, "wb", compresslevel=6) as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
            staged[rel] = out
        else:
            staged[rel] = src


def push(repo_id: str, staged: dict, private: bool, dry_run: bool) -> None:
    total = 0
    print(f"\n[datasets] {repo_id}: {len(staged)} files")
    for rel in sorted(staged):
        sz = os.path.getsize(staged[rel]); total += sz
        print(f"    {rel:60s} {sz/1e6:9.2f} MB")
    print(f"    {'total':60s} {total/1e6:9.2f} MB")
    if dry_run:
        return
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True, repo_type="dataset")
    ops = [CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(p)) for rel, p in staged.items()]
    # one commit per ~40 files keeps each request small
    for i in range(0, len(ops), 40):
        api.create_commit(repo_id=repo_id, repo_type="dataset", operations=ops[i:i + 40],
                          commit_message=f"upload {i // 40 + 1}/{(len(ops) + 39) // 40}")
    print(f"[datasets] {repo_id} done")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="maelic")
    ap.add_argument("--ra4m", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true")
    ap.add_argument("--stage", default=None, help="staging dir for gzip/meta copies (default: temp)")
    a = ap.parse_args()
    if not (a.ra4m or a.bench):
        ap.error("pass --ra4m and/or --bench")
    stage_root = Path(a.stage) if a.stage else Path(tempfile.mkdtemp(prefix="ra_datasets_"))
    if a.ra4m:
        staged = {}
        stage_files(RA4M_FILES, stage_root / "ra4m", staged)
        for name in RA4M_PACKS:
            stage_pack(name, stage_root / "ra4m", staged)
        push(f"{a.org}/RA-4M", staged, a.private, a.dry_run)
    if a.bench:
        staged = {}
        stage_files(BENCH_FILES, stage_root / "bench", staged)
        for name in BENCH_PACKS:
            stage_pack(name, stage_root / "bench", staged)
        push(f"{a.org}/OV-SGG-Bench", staged, a.private, a.dry_run)


if __name__ == "__main__":
    main()
