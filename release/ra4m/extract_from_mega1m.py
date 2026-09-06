#!/usr/bin/env python3
"""Re-extract the RA-4M image set (500,000 MEGASG images) from the HF dataset JosephZ/mega_1m.

RA-4M annotations (DATASETS/MEGASG/{train,val}/_annotations.coco.json and the runs/packed/megasg*
packs) reference images by file_name. JosephZ/mega_1m holds the same images (identical file_name,
image bytes inline) in 444 parquet shards, ~221 GB total, 988,531 rows. This script streams the
shards one at a time, keeps the rows whose file_name is in ra4m_image_manifest.tsv(.gz), writes
them to <out>/<split>/<file_name>, and deletes each shard after use -- peak disk is one shard
(~0.5 GB) plus the extracted images (~41 GB).

Verified 2026-09-01 on shard 0: 2227/2227 file names matched the manifest; mega_1m image_id equals
our original_id; data_source in {object365, oi, coco}.

    pip install pyarrow huggingface_hub
    python extract_from_mega1m.py --out /data/MEGASG            # all shards
    python extract_from_mega1m.py --out /data/MEGASG --shards 0-99

Resumable: files already present in <out> are skipped. Rerun until the summary says 500000/500000.
Unset HF_HUB_OFFLINE if your environment sets it.
"""
import argparse, csv, gzip, os, sys, time

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "JosephZ/mega_1m"
NSHARDS = 444


def load_manifest(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", newline="") as f:
        return {row["file_name"]: row["split"] for row in csv.DictReader(f, delimiter="\t")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                       "ra4m_image_manifest.tsv.gz"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--shards", default=f"0-{NSHARDS - 1}", help="inclusive range a-b")
    ap.add_argument("--keep-shards", action="store_true", help="do not delete parquet shards after use")
    ap.add_argument("--cache", default=None, help="HF cache dir for shards")
    a = ap.parse_args()

    want = load_manifest(a.manifest)
    done = {fn for fn, split in want.items() if os.path.exists(os.path.join(a.out, split, fn))}
    print(f"manifest: {len(want)} images; already extracted: {len(done)}", flush=True)
    lo, hi = (int(x) for x in a.shards.split("-"))

    for s in range(lo, hi + 1):
        if len(done) == len(want):
            break
        t0 = time.time()
        p = hf_hub_download(REPO, f"data/train-{s:05d}-of-{NSHARDS:05d}.parquet",
                            repo_type="dataset", cache_dir=a.cache)
        n_hit = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=256, columns=["file_name", "image"]):
            for fn, im in zip(batch.column("file_name").to_pylist(), batch.column("image").to_pylist()):
                fn = os.path.basename(fn)
                if fn not in want or fn in done:
                    continue
                d = os.path.join(a.out, want[fn])
                os.makedirs(d, exist_ok=True)
                tmp = os.path.join(d, fn + ".part")
                with open(tmp, "wb") as f:
                    f.write(im["bytes"])
                os.replace(tmp, os.path.join(d, fn))
                done.add(fn)
                n_hit += 1
        if not a.keep_shards:
            for q in (os.path.realpath(p), p):  # blob first, then the cache symlink
                try:
                    os.remove(q)
                except OSError:
                    pass
        print(f"shard {s:03d}: +{n_hit}  total {len(done)}/{len(want)}  ({time.time() - t0:.0f}s)", flush=True)

    missing = len(want) - len(done)
    print(f"DONE: {len(done)}/{len(want)} extracted" + (f"; {missing} MISSING" if missing else ""))
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
