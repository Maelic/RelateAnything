"""Emit the IndoorVG val+test image ids so training packs can exclude them.

IndoorVG is drawn from Visual Genome, so its evaluation images collide with any
VG-derived training source. The collision is NOT visible as a filename match
against MEGASG, because MEGASG is keyed by zero-padded COCO ids while IndoorVG
is keyed by VG ids — the same photo appears under two names. This script emits
BOTH spellings for every held-out image, so a single stem-set catches the leak
in every pack regardless of which id space it was built in.

Measured 2026-07-26 (val 733 + test 4,403 = 5,136 images, disjoint splits):
    leak into runs/packed/vg_raw       506 imgs /  10,084 rels (1.33%)
    leak into runs/packed/megasg_clean  12 imgs /     172 rels (0.00%)
Only 4,613 of the 5,136 were already in registry protected_vg_ids, and NONE of
the 506 vg_raw offenders were — the existing registry does not cover IndoorVG.

    python training/build_indoorvg_holdout.py
    # -> runs/datamix/indoorvg_holdout.json  (pass to train.py --exclude_ids)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from relsgg.paths import DATASETS, DATAMIX, RUNS, REPO  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



IV = DATASETS / "IndoorVG_coco_format"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="IndoorVG splits to protect from training.")
    ap.add_argument("--out", default=str(DATAMIX / "indoorvg_holdout.json"))
    a = ap.parse_args()

    vg_ids: set[str] = set()
    for s in a.splits:
        d = IV / s
        got = {f[:-4] for f in os.listdir(d) if f.endswith(".jpg")}
        print(f"[indoorvg] {s:5s} {len(got):,} images")
        vg_ids |= got

    vg2coco = {k: int(v) for k, v in json.load(open(DATAMIX / "vg2coco.json")).items()}
    # MEGASG file names are zero-padded to 12 digits (COCO convention); pack
    # stems are compared literally, so emit the padded form.
    coco_ids = {f"{vg2coco[v]:012d}" for v in vg_ids if v in vg2coco}

    payload = {
        "source": "IndoorVG_coco_format",
        "splits": a.splits,
        "note": "Training-time exclusion: VG ids plus their zero-padded COCO "
                "twins, so one stem-set covers VG- and COCO-keyed packs alike.",
        "vg_ids": sorted(vg_ids),
        "coco_stems": sorted(coco_ids),
        "stems": sorted(vg_ids | coco_ids),
    }
    Path(a.out).write_text(json.dumps(payload))
    print(f"[indoorvg] {len(vg_ids):,} VG ids + {len(coco_ids):,} COCO twins "
          f"= {len(payload['stems']):,} stems -> {a.out}")


if __name__ == "__main__":
    main()
