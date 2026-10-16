"""src/fcac/features/morphological.py

Morphological cardiac biomarkers computed from fiducial points (features/fiducials.py):
QTc sigma, P-wave dispersion, and QRS duration mean/variance/asymmetry.

================================================================================
FORMULA CHOICES NOT YET CONFIRMED BY THE SUPERVISOR (plan.md item 4, CLAUDE.md
section 7). Each choice below is the most standard option and is named in config
(features.morphological.*), so it can be swapped without touching this file:

  * QTc correction  -> Bazett (QT / sqrt(RR_seconds)). The most widely reported; note it
    over-corrects at high heart rates, and fetal rates ARE high (110-160 bpm), so
    Fridericia (cube root) is the obvious alternative -- set
    features.morphological.qtc_correction: fridericia to switch.
  * QRS asymmetry   -> (S_offset - R) / (R - Q_onset), i.e. how far off-centre the R peak
    sits within its own complex. 1.0 is symmetric. The alternative reading is the
    statistical skewness of the QRS segment; set qrs_asymmetry_method: skewness.
================================================================================

All features return NaN rather than raising when a window has too few complete beats to
compute them -- a 5 s window holds only ~10 fetal beats, and some will be missing
fiducials. Callers must handle NaN (see features/extract.py).

Implements: CLAUDE.md section 4 features/morphological.py; 5 of the 12 FeatureVector
columns (see features/extract.py::FEATURE_NAMES).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import stats

from fcac.config import Config, load_config
from fcac.features.fiducials import detect_fiducials


def _min_beats(cfg: Config) -> int:
    return cfg.features.morphological.min_beats


def _durations_ms(start: np.ndarray, end: np.ndarray, fs: float) -> np.ndarray:
    """Positive (end - start) durations in ms, dropping incomplete/degenerate beats."""
    duration = (end - start) / fs * 1000.0
    return duration[np.isfinite(duration) & (duration > 0)]


def qtc_sigma(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Standard deviation (ms) of the rate-corrected QT interval across beats.

    QT is measured from QRS onset to T offset; RR is the interval to the preceding beat.
    Beat-to-beat QT variability is the biomarker here, not QT itself.
    """
    if cfg is None:
        cfg = load_config()
    f = detect_fiducials(fecg, fs, cfg)

    qt_ms = (f["t_offset"] - f["qrs_onset"]) / fs * 1000.0
    rr_ms = np.full_like(qt_ms, np.nan)
    rr_ms[1:] = np.diff(f["R"]) / fs * 1000.0  # RR preceding each beat

    valid = np.isfinite(qt_ms) & np.isfinite(rr_ms) & (qt_ms > 0) & (rr_ms > 0)
    if valid.sum() < _min_beats(cfg):
        return float("nan")

    qt, rr_s = qt_ms[valid], rr_ms[valid] / 1000.0
    correction = cfg.features.morphological.qtc_correction
    if correction == "bazett":
        qtc = qt / np.sqrt(rr_s)
    elif correction == "fridericia":
        qtc = qt / np.cbrt(rr_s)
    else:
        raise ValueError(f"unknown qtc_correction '{correction}' (bazett|fridericia)")
    return float(np.std(qtc))


def p_dispersion(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """P-wave dispersion (ms): max - min P-wave duration across beats.

    P onset/offset are the isoelectric-return points either side of the P peak, the same
    boundary rule the QRS delineation uses (features/fiducials.py).
    """
    if cfg is None:
        cfg = load_config()
    f = detect_fiducials(fecg, fs, cfg)

    # P-wave DURATION is p_onset -> p_offset. An earlier version measured P peak ->
    # QRS onset, which is the PR segment, not the P wave: it reported ~92 ms on r08
    # where a fetal P wave is ~30-50 ms, because it was measuring the wrong interval.
    durations = _durations_ms(f["p_onset"], f["p_offset"], fs)
    if durations.size < _min_beats(cfg):
        return float("nan")
    return float(durations.max() - durations.min())


def qrs_mean(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Mean QRS complex duration (ms) across beats."""
    if cfg is None:
        cfg = load_config()
    f = detect_fiducials(fecg, fs, cfg)
    durations = _durations_ms(f["qrs_onset"], f["qrs_offset"], fs)
    if durations.size < _min_beats(cfg):
        return float("nan")
    return float(durations.mean())


def qrs_var(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Variance (ms^2) of QRS complex duration across beats."""
    if cfg is None:
        cfg = load_config()
    f = detect_fiducials(fecg, fs, cfg)
    durations = _durations_ms(f["qrs_onset"], f["qrs_offset"], fs)
    if durations.size < _min_beats(cfg):
        return float("nan")
    return float(durations.var())


def qrs_asymmetry(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """QRS morphological asymmetry -- how off-centre the R peak sits in its complex.

    Ratio method (default): mean of (qrs_offset - R) / (R - qrs_onset) over beats;
    1.0 means a symmetric complex, > 1 means a longer downstroke than upstroke.
    Skewness method: mean statistical skewness of the QRS segment amplitudes.
    """
    if cfg is None:
        cfg = load_config()
    f = detect_fiducials(fecg, fs, cfg)
    method = cfg.features.morphological.qrs_asymmetry_method

    if method == "ratio":
        rise = f["R"] - f["qrs_onset"]
        fall = f["qrs_offset"] - f["R"]
        valid = np.isfinite(rise) & np.isfinite(fall) & (rise > 0) & (fall > 0)
        if valid.sum() < _min_beats(cfg):
            return float("nan")
        return float(np.mean(fall[valid] / rise[valid]))

    if method == "skewness":
        values = []
        for onset, offset in zip(f["qrs_onset"], f["qrs_offset"]):
            if not (np.isfinite(onset) and np.isfinite(offset)) or offset <= onset:
                continue
            segment = fecg[int(onset) : int(offset) + 1]
            if segment.size >= 3:
                values.append(stats.skew(segment))
        if len(values) < _min_beats(cfg):
            return float("nan")
        return float(np.mean(values))

    raise ValueError(f"unknown qrs_asymmetry_method '{method}' (ratio|skewness)")
