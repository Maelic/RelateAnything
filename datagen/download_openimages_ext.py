#!/usr/bin/env python3
"""Download the Open Images V6 relationship-annotated extension subset.

Fetches images from the public s3://open-images-dataset bucket (anonymous
HTTPS, no credentials needed) and re-saves them at MEGASG's exact JPEG
convention: same pixel dimensions as source (bucket already serves the
1024-max-side version, so no resize step), quality=75, no optimize flag,
metadata stripped. Verified byte-exact against 4 known MEGASG images.

Usage:
    python datagen/download_openimages_ext.py \
        --manifest <tsv with oi_split, path_split, image_id, n_relations> \
        --out DATASETS/MEGASG_OI_ext \
        --workers 32
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import shutil
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

BUCKET = "https://s3.amazonaws.com/open-images-dataset"
JPEG_QUALITY = 75
MIN_FREE_GB = 3.0  # abort remaining work if free space on --out's filesystem drops below this


def fetch_and_recompress(oi_split: str, path_split: str, image_id: str, out_dir: Path) -> tuple[str, bool, int, str]:
    url = f"{BUCKET}/{path_split}/{image_id}.jpg"
    dest = out_dir / oi_split / f"{image_id}.jpg"
    if dest.exists():
        return image_id, True, dest.stat().st_size, "cached"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read()
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=JPEG_QUALITY)
        data = buf.getvalue()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.rename(dest)
        return image_id, True, len(data), "ok"
    except Exception as e:
        return image_id, False, 0, f"{type(e).__name__}: {e}"


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="process only first N rows (smoke test)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    for s in ("train", "val", "test"):
        (out_dir / s).mkdir(parents=True, exist_ok=True)

    rows = []
    with open(args.manifest, newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            rows.append((row["oi_split"], row["path_split"], row["image_id"]))
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} images queued -> {out_dir}", flush=True)

    n_ok = n_fail = 0
    n_bytes = 0
    fail_log = open(out_dir / "download_failures.tsv", "a")
    manifest_out = open(out_dir / "downloaded_manifest.tsv", "a")
    t0 = time.time()
    aborted = False

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(fetch_and_recompress, oi_split, path_split, iid, out_dir): (oi_split, iid)
            for oi_split, path_split, iid in rows
        }
        for i, fut in enumerate(as_completed(futs), 1):
            oi_split, iid = futs[fut]
            image_id, ok, nbytes, status = fut.result()
            if ok:
                n_ok += 1
                n_bytes += nbytes
                manifest_out.write(f"{oi_split}\t{image_id}\t{nbytes}\n")
            else:
                n_fail += 1
                fail_log.write(f"{oi_split}\t{image_id}\t{status}\n")

            if i % 2000 == 0 or i == len(rows):
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                fg = free_gb(out_dir)
                print(f"[{i}/{len(rows)}] ok={n_ok} fail={n_fail} "
                      f"written={n_bytes/1e9:.2f}GB rate={rate:.1f} img/s free={fg:.1f}GB",
                      flush=True)
                manifest_out.flush()
                fail_log.flush()
                if fg < MIN_FREE_GB:
                    print(f"ABORT: free space {fg:.1f}GB < {MIN_FREE_GB}GB safety margin. "
                          f"Cancelling remaining tasks.", flush=True)
                    for f2 in futs:
                        f2.cancel()
                    aborted = True
                    break

    fail_log.close()
    manifest_out.close()
    elapsed = time.time() - t0
    print(f"DONE in {elapsed/60:.1f} min: ok={n_ok} fail={n_fail} written={n_bytes/1e9:.2f}GB "
          f"{'(ABORTED on low disk)' if aborted else ''}", flush=True)


if __name__ == "__main__":
    main()
