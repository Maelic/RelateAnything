"""Filesystem roots shared by every script in the repository.

Two environment variables relocate everything:

    RA_DATASETS   raw benchmark downloads (VG150, PSG, HICO-DET,...); default ``datasets/``
    RA_RUNS       packs, checkpoints and evaluation outputs;               default ``runs/``

Both default to directories under the current working directory, which is the
repository root in every documented command.

A pack's ``meta.json`` records ``img_dir``, the absolute image folder of the machine that built
it. ``relocated`` re-roots such a path at a local datasets folder by keeping the part after its
last ``datasets`` component (any case): ``/mimer/.../DATASETS/MEGASG/train`` becomes
``<root>/MEGASG/train``. ``resolve_img_dir`` applies that rule at load time when the recorded
folder does not exist and ``$RA_DATASETS`` is set. This module stays dependency-free: the OvSGTR
adapter imports it from OvSGTR's own environment.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]
DATASETS = Path(os.environ.get("RA_DATASETS", "datasets"))
RUNS = Path(os.environ.get("RA_RUNS", "runs"))
PACKED = RUNS / "packed"
DATAMIX = RUNS / "datamix"

__all__ = ["REPO", "DATASETS", "RUNS", "PACKED", "DATAMIX", "relocated", "resolve_img_dir"]


def relocated(img_dir: str, root: str) -> str:
    """``img_dir`` re-rooted at ``root``: the part after its last ``datasets`` component."""
    parts = PurePosixPath(str(img_dir)).parts
    hits = [i for i, p in enumerate(parts) if p.lower() == "datasets"]
    if not hits:
        raise ValueError(f"no 'datasets' component in {img_dir!r}")
    return str(Path(root, *parts[hits[-1] + 1:]))


def resolve_img_dir(meta: dict) -> Path:
    """The image folder of a pack split on this machine.

    Returns ``meta["img_dir"]`` when it exists. Otherwise, with ``$RA_DATASETS`` set, returns the
    recorded path re-rooted there (from ``img_dir_built`` when setup already rewrote the pack).
    When neither exists it warns, naming every path it tried, and returns the recorded one, so code
    that reads only the annotations keeps working.
    """
    recorded = Path(meta["img_dir"])
    if recorded.is_dir():
        return recorded
    tried = [str(recorded)]
    root = os.environ.get("RA_DATASETS")
    if root:
        try:
            cand = Path(relocated(meta.get("img_dir_built", meta["img_dir"]), root))
        except ValueError:
            cand = None
        if cand is not None:
            if cand.is_dir():
                return cand
            tried.append(str(cand))
    warnings.warn(
        "image folder not found; tried " + ", ".join(tried)
        + ("" if root else " ($RA_DATASETS is not set)")
        + ". Run benchmark/setup_data.py or training/relocate_packs.py.",
        stacklevel=2)
    return recorded
