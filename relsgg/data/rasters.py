"""Region rasters: a box or a mask as coverage of a g x g grid over the image.

A box rasterises to its rectangle and a mask to its shape, so the model sees
one representation for both. Both functions return the fraction of each cell
covered, in [0, 1].
"""
from __future__ import annotations

import numpy as np


def box_raster(x1: float, y1: float, x2: float, y2: float, g: int) -> np.ndarray:
    """Coverage of a normalised xyxy box over a g x g grid, ``[g, g]``."""
    e = np.arange(g + 1, dtype=np.float64) / g
    ix = np.clip(np.minimum(x2, e[1:]) - np.maximum(x1, e[:-1]), 0.0, None)
    iy = np.clip(np.minimum(y2, e[1:]) - np.maximum(y1, e[:-1]), 0.0, None)
    return (iy[:, None] * ix[None,:]) * (g * g)


def _bounds(n: int, g: int) -> np.ndarray:
    return (np.arange(g) * n) // g


def mask_raster(m: np.ndarray, g: int) -> np.ndarray:
    """Area-average a ``[H, W]`` binary mask down to ``[g, g]`` (exact box filter)."""
    m = np.asarray(m)
    if m.dtype != np.float64:
        m = m.astype(np.float64)
    H, W = m.shape
    yb, xb = _bounds(H, g), _bounds(W, g)
    rows = np.add.reduceat(m, yb, axis=0)
    acc = np.add.reduceat(rows, xb, axis=1)
    ycnt = np.diff(np.append(yb, H)).astype(np.float64)
    xcnt = np.diff(np.append(xb, W)).astype(np.float64)
    return acc / np.maximum(np.outer(ycnt, xcnt), 1.0)
