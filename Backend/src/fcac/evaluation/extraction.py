"""src/fcac/evaluation/extraction.py

fECG extraction-quality metrics (CLAUDE.md §2 "Evaluation" / "Ground truth"):
- MSE against ADFECGDB's Direct_1 scalp channel (ICC 2026 eq. 2; ADFECGDB only).
- Gain-matched SNR (sign- and scale-free) for extractors that are not trained to reproduce the
  reference amplitude.
- SNR against ADFECGDB's Direct_1 scalp channel (ADFECGDB only -- NInFEA has no
  amplitude ground truth).
- QRS detection F1 within +/-50 ms of a reference R-peak (either dataset, given a
  reference R-peak / beat-timing array).
- Bland-Altman R-R interval agreement.

Implements: CLAUDE.md §5 evaluation/extraction.py.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def mse(fecg_extracted: np.ndarray, fecg_reference: np.ndarray) -> float:
    """Mean squared error, mean((y - y_hat)^2), of an extracted fECG against the
    ADFECGDB Direct_1 scalp reference (ICC 2026 eq. 2). ADFECGDB only -- NInFEA and
    CinC 2013 have no amplitude reference.
    """
    return float(np.mean((fecg_reference - fecg_extracted) ** 2))


def snr_db(fecg_extracted: np.ndarray, fecg_reference: np.ndarray) -> float:
    """Signal-to-noise ratio (dB), 10*log10(sum(y^2) / sum((y - y_hat)^2)) (ICC 2026
    eq. 3), of an extracted fECG against the ADFECGDB Direct_1 scalp reference. ADFECGDB only -- CLAUDE.md §2 "Ground truth": NInFEA has no
    amplitude reference.
    """
    noise = fecg_extracted - fecg_reference
    signal_power = np.mean(fecg_reference**2)
    noise_power = np.mean(noise**2)
    if noise_power <= 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def gain_matched_snr_db(
    fecg_extracted: np.ndarray, fecg_reference: np.ndarray
) -> Tuple[float, float, float]:
    """SNR (dB) of an extracted fECG against the scalp reference after the best scalar gain.

    The ICC 2026 SNR (`snr_db`) is for a network trained to reproduce the reference amplitude. The
    PINN / BSS outputs are not: their scale is arbitrary and BSS also has an arbitrary sign. This
    fits x -> a*x by least squares (a may be negative) and returns
    (10*log10(sum(y^2) / sum((y - a*x)^2)), a, pearson r); the SNR equals -10*log10(1 - r^2), so it
    is always >= 0 dB and measures waveform agreement, not amplitude calibration.
    ADFECGDB only (scalp reference).
    """
    x = fecg_extracted.astype(np.float64) - fecg_extracted.mean()
    y = fecg_reference.astype(np.float64) - fecg_reference.mean()
    denom = float(np.dot(x, x))
    if denom <= 0 or float(np.dot(y, y)) <= 0:
        return float("nan"), float("nan"), float("nan")
    gain = float(np.dot(x, y) / denom)
    residual = float(np.sum((y - gain * x) ** 2))
    r = float(np.dot(x, y) / np.sqrt(denom * np.dot(y, y)))
    if residual <= 0:
        return float("inf"), gain, r
    return float(10 * np.log10(np.dot(y, y) / residual)), gain, r


def qrs_f1(
    detected_peaks: np.ndarray,
    reference_peaks: np.ndarray,
    fs: float,
    tolerance_ms: float = 50.0,
) -> Tuple[float, float, float]:
    """F1 (+ precision, recall) for detected R-peaks against a reference, matching
    within +/- tolerance_ms (CLAUDE.md §2 "Evaluation": 50 ms default).

    detected_peaks, reference_peaks : sample indices.
    Returns (f1, precision, recall).
    """
    tolerance_samples = tolerance_ms / 1000.0 * fs
    matched_ref = set()
    true_positives = 0
    for d in detected_peaks:
        candidates = [
            r for r in reference_peaks
            if abs(r - d) <= tolerance_samples and r not in matched_ref
        ]
        if candidates:
            closest = min(candidates, key=lambda r: abs(r - d))
            matched_ref.add(closest)
            true_positives += 1

    n_detected = len(detected_peaks)
    n_reference = len(reference_peaks)
    precision = true_positives / n_detected if n_detected else 0.0
    recall = true_positives / n_reference if n_reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, precision, recall


def bland_altman_rr(
    rr_detected_ms: np.ndarray, rr_reference_ms: np.ndarray
) -> Tuple[float, float, float]:
    """Bland-Altman agreement stats for R-R intervals (ms): (bias, loa_low, loa_high),
    where loa = bias +/- 1.96 * std(diff).

    rr_detected_ms, rr_reference_ms : matched, same-length R-R interval arrays.
    """
    diff = rr_detected_ms - rr_reference_ms
    bias = float(np.mean(diff))
    std_diff = float(np.std(diff))
    return bias, bias - 1.96 * std_diff, bias + 1.96 * std_diff
