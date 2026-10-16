"""src/fcac/evaluation/doppler_ref.py

Helpers for scoring extracted fetal ECG against NInFEA's Doppler-derived beat times
(src/fcac/io/ninfea_doppler.py). A Doppler peak is a flow event, not the R wave, so it trails
the R wave by an unknown, roughly constant lag; the lag is estimated from the data (the mode
of detected-R -> Doppler-peak time differences) and then applied as one global shift.

Implements: CLAUDE.md §2 "Ground truth" (NInFEA Doppler -> timing metrics only).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def lag_differences(
    detected_s: np.ndarray, reference_s: np.ndarray, half_window: float = 0.25
) -> np.ndarray:
    """For each detected beat, (nearest reference beat - detected beat) in seconds, kept when
    within +/- `half_window`. One pairing per detection, so a lag is unique modulo the heart
    period instead of aliasing onto the next beat; pick `half_window` below half the period."""
    if len(detected_s) == 0 or len(reference_s) == 0:
        return np.empty(0)
    diff = reference_s[None, :] - detected_s[:, None]
    nearest = diff[np.arange(len(detected_s)), np.abs(diff).argmin(axis=1)]
    return nearest[np.abs(nearest) <= half_window]


def estimate_lag(
    diffs: np.ndarray, half_window: float = 0.25, bin_s: float = 0.005, smooth_bins: int = 5
) -> Tuple[float, float]:
    """(lag seconds, concentration). The lag is the mode of the smoothed histogram of `diffs`
    over [-half_window, half_window]; concentration is the share of all differences within
    +/-25 ms of it. Unlocked (uniform) differences give about 0.05 / (2 * half_window) = 0.10;
    a beat-locked train is far higher."""
    if len(diffs) < 10:
        return float("nan"), float("nan")
    edges = np.arange(-half_window, half_window + bin_s, bin_s)
    hist, _ = np.histogram(diffs, bins=edges)
    smooth = np.convolve(hist, np.ones(smooth_bins) / smooth_bins, mode="same")
    lag = float(0.5 * (edges[smooth.argmax()] + edges[smooth.argmax() + 1]))
    return lag, float(np.mean(np.abs(diffs - lag) <= 0.025))


def heart_rate_bpm(beat_times_s: np.ndarray) -> float:
    """Median-interval heart rate; NaN with fewer than 3 beats."""
    if len(beat_times_s) < 3:
        return float("nan")
    return float(60.0 / np.median(np.diff(beat_times_s)))
