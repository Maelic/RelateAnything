"""Filesystem roots shared by every script in the repository.

Two environment variables relocate everything:

    RA_DATASETS   raw benchmark downloads (VG150, PSG, HICO-DET,...); default ``datasets/``
    RA_RUNS       packs, checkpoints and evaluation outputs;               default ``runs/``

Both default to directories under the current working directory, which is the
repository root in every documented command.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATASETS = Path(os.environ.get("RA_DATASETS", "datasets"))
RUNS = Path(os.environ.get("RA_RUNS", "runs"))
PACKED = RUNS / "packed"
DATAMIX = RUNS / "datamix"

__all__ = ["REPO", "DATASETS", "RUNS", "PACKED", "DATAMIX"]
