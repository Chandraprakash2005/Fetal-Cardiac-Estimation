"""src/fcac/preprocessing/filters.py

Pure filtering/resampling functions. No dataset-specific knowledge lives here -- callers
(pipeline.py) pass explicit fs/band/order values read from Config.

Implements: CLAUDE.md §2 "Notch", "Resampling order"; §5 Recording -> filtered signal.
"""

from __future__ import annotations

from math import gcd

import numpy as np
from scipy import signal as sp_signal


def butter_bandpass(
    x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 4
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter along the last axis.

    x : (..., N) -- works for a single channel (N,) or stacked channels (C, N).
    """
    nyquist = fs / 2.0
    b, a = sp_signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    return sp_signal.filtfilt(b, a, x, axis=-1).astype(np.float32)


def notch_filter(x: np.ndarray, fs: float, freq_hz: float, quality_factor: float) -> np.ndarray:
    """Zero-phase IIR notch filter (e.g. 50 Hz mains), along the last axis."""
    nyquist = fs / 2.0
    b, a = sp_signal.iirnotch(freq_hz / nyquist, quality_factor)
    return sp_signal.filtfilt(b, a, x, axis=-1).astype(np.float32)


def resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Polyphase resample along the last axis from fs_in to fs_out."""
    g = gcd(int(fs_in), int(fs_out))
    up, down = int(fs_out) // g, int(fs_in) // g
    return sp_signal.resample_poly(x, up, down, axis=-1).astype(np.float32)


def trim_edges(x: np.ndarray, fs: float, trim_seconds: float) -> np.ndarray:
    """Drop `trim_seconds` from both ends of the last axis to remove filtfilt edge
    transients."""
    trim_samples = int(trim_seconds * fs)
    n = x.shape[-1]
    if n <= 2 * trim_samples:
        return x
    return x[..., trim_samples : n - trim_samples]
