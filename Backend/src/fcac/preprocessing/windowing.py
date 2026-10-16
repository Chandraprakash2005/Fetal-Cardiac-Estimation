"""src/fcac/preprocessing/windowing.py

Implements: CLAUDE.md §2 "Window" -- 5000 samples (5 s @ 1 kHz), non-overlapping by
default; stride is a config param.
"""

from __future__ import annotations

import numpy as np


def segment(x: np.ndarray, win: int, stride: int) -> np.ndarray:
    """Slice (..., N) into (W, ..., win) windows of size `win`, hop `stride`.

    Works for a single channel (N,) -> (W, win), or stacked channels (C, N) -> (W, C, win).
    Trailing samples that don't fill a full window are dropped.
    """
    n = x.shape[-1]
    n_windows = 1 + (n - win) // stride if n >= win else 0
    windows = np.stack(
        [x[..., i * stride : i * stride + win] for i in range(n_windows)],
        axis=0,
    )
    return windows.astype(np.float32)
