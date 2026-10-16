"""src/fcac/features/fiducials.py

P/Q/R/S/T fiducial point detection on an isolated single-channel fetal ECG window.
Every morphological feature and the RR-derived dynamic features depend on this.

================================================================================
PARAMETER CHOICES NOT YET CONFIRMED BY THE SUPERVISOR (plan.md item 4, CLAUDE.md
section 7). The algorithm below is a standard R-anchored search-window delineator with
FETAL-scaled intervals; every window width is read from configs/default.yaml
(features.fiducials.*) so nothing is baked in. Confirm before reporting features.
================================================================================

Algorithm: Pan-Tompkins locates R peaks (reusing memd/diagnostics.py, so the detector is
the same one the extraction stage is scored with), then each other fiducial is the
extremum inside a fixed search window relative to its R peak:

    Q : minimum in [R - qs_window, R]
    S : minimum in [R, R + qs_window]
    P : maximum in [R - p_search_end, R - p_search_start]
    T : maximum in [R + t_search_start, R + t_search_end]

QRS onset/offset are taken as the first turning point outside Q/S where the slope
flattens, which is what the duration features need.

Why fetal-scaled windows matter: a fetal QRS is ~40-60 ms against an adult's ~80-120 ms,
and at 110-160 bpm the whole RR interval is only ~375-545 ms. Adult search windows would
run past the next beat and return garbage.

Implements: CLAUDE.md section 4 features/fiducials.py.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from fcac.config import Config, load_config
from fcac.memd.diagnostics import pan_tompkins_peaks


def _ms_to_samples(ms: float, fs: float) -> int:
    return max(1, int(round(ms / 1000.0 * fs)))


def _argmin_in(signal: np.ndarray, lo: int, hi: int) -> Optional[int]:
    lo, hi = max(0, lo), min(len(signal), hi)
    if hi - lo < 1:
        return None
    return lo + int(np.argmin(signal[lo:hi]))


def _argmax_in(signal: np.ndarray, lo: int, hi: int) -> Optional[int]:
    lo, hi = max(0, lo), min(len(signal), hi)
    if hi - lo < 1:
        return None
    return lo + int(np.argmax(signal[lo:hi]))


def _baseline(signal: np.ndarray) -> float:
    """Isoelectric level: the median is robust to the QRS spikes riding on it."""
    return float(np.median(signal))


def _wave_onset(
    signal: np.ndarray, peak: int, search_back: int, amplitude: float, frac: float
) -> Optional[int]:
    """Walk back from `peak` to where the wave returns to the isoelectric line.

    A wave boundary is where the deflection dies out, i.e. |signal - baseline| falls
    below `frac` of the wave's own amplitude. The earlier implementation instead looked
    for the slope flattening, which triggers at the peak itself (the derivative is zero
    at every extremum) and consequently overran the true boundary -- it measured a 50 ms
    synthetic QRS as 75 ms. Validated in tests/test_features.py against a template with
    known intervals.
    """
    lo = max(0, peak - search_back)
    if peak - lo < 2 or amplitude <= 0:
        return None
    base = _baseline(signal)
    threshold = frac * amplitude
    for i in range(peak, lo - 1, -1):
        if abs(signal[i] - base) <= threshold:
            return i
    return lo


def _wave_offset(
    signal: np.ndarray, peak: int, search_forward: int, amplitude: float, frac: float
) -> Optional[int]:
    """Walk forward from `peak` to where the wave returns to the isoelectric line."""
    hi = min(len(signal) - 1, peak + search_forward)
    if hi - peak < 2 or amplitude <= 0:
        return None
    base = _baseline(signal)
    threshold = frac * amplitude
    for i in range(peak, hi + 1):
        if abs(signal[i] - base) <= threshold:
            return i
    return hi


def detect_fiducials(
    fecg: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> Dict[str, np.ndarray]:
    """Detect P, Q, R, S, T sample indices in a single-channel fECG window.

    fecg : (N,) isolated fetal ECG (PINNOutput.fecg[w]).
    cfg  : supplies features.fiducials.* window widths; loaded from defaults if None.

    Returns a dict of equal-length arrays, one entry per detected beat, aligned by index
    so beat i is (P[i], Q[i], R[i], S[i], T[i]). Entries that could not be located are
    NaN, so callers must use nan-aware reductions rather than assuming every beat is
    complete. Also returns "qrs_onset" and "qrs_offset" (needed for QRS duration) and
    "t_offset" (needed for QT).
    """
    if cfg is None:
        cfg = load_config()
    params = cfg.features.fiducials

    r_peaks = pan_tompkins_peaks(fecg, fs)
    n = len(r_peaks)
    keys = ("P", "Q", "R", "S", "T", "qrs_onset", "qrs_offset", "t_offset",
            "p_onset", "p_offset")
    out = {k: np.full(n, np.nan) for k in keys}
    if n == 0:
        return {k: v.astype(np.float64) for k, v in out.items()}

    qs = _ms_to_samples(params.qs_window_ms, fs)
    p_start = _ms_to_samples(params.p_search_start_ms, fs)
    p_end = _ms_to_samples(params.p_search_end_ms, fs)
    t_start = _ms_to_samples(params.t_search_start_ms, fs)
    t_end = _ms_to_samples(params.t_search_end_ms, fs)
    onset_back = _ms_to_samples(params.onset_search_ms, fs)

    frac = params.boundary_amplitude_frac
    base = _baseline(fecg)

    r_peaks = refine_r_peaks(fecg, r_peaks, fs, params.r_refine_ms)
    out["R"] = r_peaks.astype(np.float64)
    for i, r in enumerate(r_peaks):
        q = _argmin_in(fecg, r - qs, r)
        s = _argmin_in(fecg, r + 1, r + qs)
        p = _argmax_in(fecg, r - p_end, r - p_start)
        t = _argmax_in(fecg, r + t_start, r + t_end)

        # QRS boundaries are referenced to the R peak's own amplitude, so a small Q or S
        # notch does not set the threshold.
        qrs_amplitude = abs(float(fecg[r]) - base)
        if q is not None:
            out["Q"][i] = q
            onset = _wave_onset(fecg, q, onset_back, qrs_amplitude, frac)
            if onset is not None:
                out["qrs_onset"][i] = onset
        if s is not None:
            out["S"][i] = s
            offset = _wave_offset(fecg, s, onset_back, qrs_amplitude, frac)
            if offset is not None:
                out["qrs_offset"][i] = offset

        if p is not None:
            out["P"][i] = p
            p_amplitude = abs(float(fecg[p]) - base)
            p_on = _wave_onset(fecg, p, p_end - p_start, p_amplitude, frac)
            p_off = _wave_offset(fecg, p, p_end - p_start, p_amplitude, frac)
            if p_on is not None:
                out["p_onset"][i] = p_on
            if p_off is not None:
                out["p_offset"][i] = p_off

        if t is not None:
            out["T"][i] = t
            t_amplitude = abs(float(fecg[t]) - base)
            t_off = _wave_offset(fecg, t, t_end - t_start, t_amplitude, frac)
            if t_off is not None:
                out["t_offset"][i] = t_off

    return out


def refine_r_peaks(
    signal: np.ndarray, r_peaks: np.ndarray, fs: float, refine_ms: float
) -> np.ndarray:
    """Snap Pan-Tompkins R peaks onto the true extremum of the RAW signal.

    Pan-Tompkins locates peaks in a squared, moving-window-INTEGRATED signal, so its
    output is systematically delayed by roughly half the integration window (~25 ms
    here) and sits off the actual R apex. That bias is invisible to a +/-50 ms QRS F1
    score but it wrecks anything measured relative to R: on a perfectly symmetric
    synthetic complex it made qrs_asymmetry read 0.515 instead of 1.0, because the
    "rise" (R - onset) absorbed the lag and the "fall" (offset - R) lost it.

    Each peak is moved to the largest absolute deviation from baseline within
    +/- refine_ms. Returns a sorted, de-duplicated array.
    """
    if len(r_peaks) == 0:
        return r_peaks
    half = _ms_to_samples(refine_ms, fs)
    base = _baseline(signal)

    refined = []
    for r in r_peaks:
        lo, hi = max(0, int(r) - half), min(len(signal), int(r) + half + 1)
        if hi - lo < 1:
            refined.append(int(r))
            continue
        refined.append(lo + int(np.argmax(np.abs(signal[lo:hi] - base))))
    return np.unique(np.asarray(refined, dtype=np.int64))


def filter_rr_artifacts(rr: np.ndarray, cfg: Config) -> np.ndarray:
    """Drop non-physiological RR intervals before any HRV measure sees them.

    R-peak detection on a real extracted fECG is not perfect: a missed beat produces an
    RR of roughly double, and a false detection produces two short ones. On ADFECGDB r08
    this left intervals from 97 ms (618 bpm) to 4320 ms (14 bpm) -- 9.8% of the series --
    and inflated the RR standard deviation to 180 ms where fetal RR SD is ~15 ms. Every
    dynamic feature is computed from this series, so unfiltered artefacts corrupt all
    seven of them (they drove symbolic dynamics to a saturated 96% and LF/HF to 34.6).

    Two standard stages:
      1. Range filter -- keep only physiologically possible fetal intervals.
      2. Successive-difference filter (Malik-style) -- drop any interval differing from
         the running median by more than `rr_max_relative_change`, which catches
         ectopics and detector slips that individually fall inside the range.

    Filtering breaks the series into fragments; the returned array concatenates the
    survivors. That is standard HRV practice and is acceptable for the distribution- and
    correlation-based measures used here.
    """
    p = cfg.features.dynamic
    if rr.size == 0:
        return rr

    keep = (rr >= p.rr_min_ms) & (rr <= p.rr_max_ms)
    ranged = rr[keep]
    if ranged.size < 3:
        return ranged

    reference = float(np.median(ranged))
    relative = np.abs(ranged - reference) / reference
    return ranged[relative <= p.rr_max_relative_change]


def rr_intervals(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> np.ndarray:
    """RR intervals in MILLISECONDS -- the input series for every dynamic feature.

    Returns an empty array if fewer than 2 R peaks were found.
    """
    if cfg is None:
        cfg = load_config()
    r_peaks = refine_r_peaks(
        fecg, pan_tompkins_peaks(fecg, fs), fs, cfg.features.fiducials.r_refine_ms
    )
    if len(r_peaks) < 2:
        return np.empty(0, dtype=np.float64)
    return filter_rr_artifacts(np.diff(r_peaks) / fs * 1000.0, cfg)
