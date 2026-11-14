"""Point packed datasets at the image root of this machine.

The loader opens images under ``meta.json["img_dir"]``, an absolute path
written when the pack was built. A pack built elsewhere therefore names the
other machine's directories. This rewrites ``img_dir`` of every split of every
pack: the part after the last ``datasets`` path component (any case) is kept
and re-rooted at ``--datasets`` (default ``$RA_DATASETS``), so
``/mimer/.../DATASETS/MEGASG/train`` becomes ``$RA_DATASETS/MEGASG/train``.
The previous value is kept as ``img_dir_built``.

    python training/relocate_packs.py runs/packed/megasg_clean runs/packed/vg_raw \\
        runs/packed/hicodet runs/packed/megasg runs/packed/psg
    python training/relocate_packs.py runs/packed/* --check     # every image present?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from relsgg.paths import relocated  # noqa: E402,F401  (re-exported: setup_data imports it from here)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+", help="pack directories (holding <split>/meta.json)")
    ap.add_argument("--datasets", default=os.environ.get("RA_DATASETS", "datasets"))
    ap.add_argument("--check", action="store_true",
                    help="only report missing images; rewrite nothing")
    a = ap.parse_args()
    root = os.path.abspath(a.datasets)

    missing_total = 0
    for pack in a.packs:
        for meta_path in sorted(Path(pack).glob("*/meta.json")):
            meta = json.load(open(meta_path))
            if a.check:
                if not (meta_path.parent / "file_names.json").exists():
                    # e.g. an evaluation pack's train/, kept for its vocabulary only
                    print(f"{meta_path.parent}: no file list, skipped")
                    continue
                names = json.load(open(meta_path.parent / "file_names.json"))
                d = Path(meta["img_dir"])
                missing = [n for n in names if not (d / n).is_file()]
                missing_total += len(missing)
                print(f"{meta_path.parent}: {len(names) - len(missing):,}/{len(names):,} images "
                      f"in {d}" + (f"  e.g. missing {missing[:2]}" if missing else ""))
                continue
            new = relocated(meta.get("img_dir_built", meta["img_dir"]), root)
            meta.setdefault("img_dir_built", meta["img_dir"])
            meta["img_dir"] = new
            # Packs may arrive as read-only symlinked files: replace, never edit in place.
            tmp = meta_path.with_suffix(".json.tmp")
            json.dump(meta, open(tmp, "w"))
            os.replace(tmp, meta_path)
            print(f"{meta_path.parent}: img_dir -> {new}")
    if a.check and missing_total:
        sys.exit(f"{missing_total:,} images missing")


if __name__ == "__main__":
    main()
