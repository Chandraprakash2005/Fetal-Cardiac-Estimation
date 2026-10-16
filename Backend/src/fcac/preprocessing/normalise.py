"""src/fcac/preprocessing/normalise.py

Implements: CLAUDE.md §2 "Normalisation" -- per-channel z-score, per recording.
"""

from __future__ import annotations

import numpy as np


def zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel zero-mean, unit-variance normalisation.

    x : (..., N) -- mean/std computed over the last axis (time), independently per
    leading dimension (channel), per recording.
    """
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)
