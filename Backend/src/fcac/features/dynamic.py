"""src/fcac/features/dynamic.py

Dynamic (nonlinear / complexity) cardiac biomarkers computed from the RR-interval series
derived from features/fiducials.py R-peaks: PRSA, multiscale entropy (MSE), approximate
entropy (ApEn), sample entropy (SampEn), DFA scaling exponent (alpha), LF/HF power ratio,
and symbolic dynamics.

================================================================================
PARAMETER CHOICES NOT YET CONFIRMED BY THE SUPERVISOR (plan.md item 4, CLAUDE.md
section 7). Every parameter is in configs/default.yaml under features.dynamic.*:

  * ApEn / SampEn  -> m=2, r=0.2*SD. The near-universal default (Pincus; Richman & Moorman).
  * MSE            -> coarse-grain scales 1..5, summarised by the mean SampEn over scales
                      (Costa et al.). Scale range is capped by series length at runtime.
  * DFA alpha      -> short-term alpha1, box sizes 4..16 beats (Peng et al.).
  * PRSA           -> deceleration capacity, Bauer et al.: anchors where RR increases,
                      window L, wavelet-like 2-point Haar summary.
  * Symbolic dyn.  -> Porta's 6-level uniform quantisation, word length 3, summarised as
                      0V% (fraction of words with no variation) -- the index most often
                      linked to sympathetic modulation.
  * LF/HF bands    -> FETAL bands 0.04-0.15 Hz (LF) and 0.4-1.5 Hz (HF). Adult bands
                      (0.15-0.4 HF) DO NOT apply: fetal heart rate is 110-160 bpm and
                      fetal respiratory/vagal modulation sits higher. plan.md flags this
                      explicitly as needing confirmation.

A HARD METHODOLOGICAL LIMIT, not a parameter choice: a 5 s window at 110-160 bpm holds
only ~9-13 beats, i.e. ~8-12 RR intervals. DFA, MSE and the entropies are normally
computed on hundreds to thousands of beats. Every function below returns NaN when the
series is shorter than its configured minimum rather than returning a number that would
look valid and be meaningless. See features/extract.py for what this implies.
================================================================================

Implements: CLAUDE.md section 4 features/dynamic.py; 7 of the 12 FeatureVector columns
(see features/extract.py::FEATURE_NAMES).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import interpolate, signal as sp_signal

from fcac.config import Config, load_config
from fcac.features.fiducials import rr_intervals


def _rr(fecg: np.ndarray, fs: float, cfg: Config) -> np.ndarray:
    return rr_intervals(fecg, fs, cfg)


# --------------------------------------------------------------------------- entropies


def _sample_entropy(series: np.ndarray, m: int, r: float) -> float:
    """Richman & Moorman SampEn. Returns NaN if no template matches (undefined)."""
    n = series.size
    if n <= m + 1:
        return float("nan")

    def _count(length: int) -> int:
        templates = np.lib.stride_tricks.sliding_window_view(series, length)
        total = 0
        for i in range(templates.shape[0] - 1):
            distance = np.max(np.abs(templates[i + 1 :] - templates[i]), axis=1)
            total += int(np.count_nonzero(distance <= r))
        return total

    b = _count(m)
    a = _count(m + 1)
    if b == 0 or a == 0:
        return float("nan")
    return float(-np.log(a / b))


def _approximate_entropy(series: np.ndarray, m: int, r: float) -> float:
    """Pincus ApEn (self-matches included, unlike SampEn)."""
    n = series.size
    if n <= m + 1:
        return float("nan")

    def _phi(length: int) -> float:
        templates = np.lib.stride_tricks.sliding_window_view(series, length)
        counts = [
            np.count_nonzero(np.max(np.abs(templates - template), axis=1) <= r)
            for template in templates
        ]
        return float(np.mean(np.log(np.asarray(counts) / templates.shape[0])))

    return float(_phi(m) - _phi(m + 1))


def _apen_from_rr(rr: np.ndarray, cfg: Config) -> float:
    """Approximate entropy of the RR series (m=2, r=0.2*SD by default)."""
    p = cfg.features.dynamic
    if rr.size < p.min_rr_entropy or np.std(rr) == 0:
        return float("nan")
    return _approximate_entropy(rr, p.entropy_m, p.entropy_r_factor * float(np.std(rr)))


def _sampen_from_rr(rr: np.ndarray, cfg: Config) -> float:
    """Sample entropy of the RR series (m=2, r=0.2*SD by default)."""
    p = cfg.features.dynamic
    if rr.size < p.min_rr_entropy or np.std(rr) == 0:
        return float("nan")
    return _sample_entropy(rr, p.entropy_m, p.entropy_r_factor * float(np.std(rr)))


def _mse_from_rr(rr: np.ndarray, cfg: Config) -> float:
    """Multiscale entropy: mean SampEn over coarse-grained scales 1..max_scale.

    Coarse-graining at scale s averages non-overlapping blocks of s samples, so scale s
    needs s times as many beats -- scales that would leave too short a series are skipped
    rather than reported from 2 or 3 points.
    """
    p = cfg.features.dynamic
    if rr.size < p.min_rr_entropy or np.std(rr) == 0:
        return float("nan")

    r = p.entropy_r_factor * float(np.std(rr))  # r fixed on the ORIGINAL series (Costa)
    values = []
    for scale in range(1, p.mse_max_scale + 1):
        n_blocks = rr.size // scale
        if n_blocks < p.min_rr_entropy:
            break
        grained = rr[: n_blocks * scale].reshape(n_blocks, scale).mean(axis=1)
        value = _sample_entropy(grained, p.entropy_m, r)
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


# -------------------------------------------------------------------------------- DFA


def _dfa_alpha_from_rr(rr: np.ndarray, cfg: Config) -> float:
    """Short-term DFA scaling exponent alpha1 over box sizes 4..16 beats."""
    p = cfg.features.dynamic
    if rr.size < p.min_rr_dfa:
        return float("nan")

    profile = np.cumsum(rr - rr.mean())
    sizes = [s for s in range(p.dfa_min_box, p.dfa_max_box + 1) if s <= rr.size // 2]
    if len(sizes) < 2:
        return float("nan")

    fluctuations = []
    for size in sizes:
        n_boxes = profile.size // size
        if n_boxes < 1:
            continue
        boxes = profile[: n_boxes * size].reshape(n_boxes, size)
        t = np.arange(size)
        residuals = [box - np.polyval(np.polyfit(t, box, 1), t) for box in boxes]
        fluctuations.append(np.sqrt(np.mean(np.concatenate(residuals) ** 2)))

    valid = [(s, f) for s, f in zip(sizes, fluctuations) if f > 0]
    if len(valid) < 2:
        return float("nan")
    log_s = np.log10([s for s, _ in valid])
    log_f = np.log10([f for _, f in valid])
    return float(np.polyfit(log_s, log_f, 1)[0])


# ------------------------------------------------------------------------- spectral


def _lf_hf_from_rr(rr: np.ndarray, cfg: Config) -> float:
    """LF/HF power ratio of the RR tachogram, using FETAL band edges.

    The tachogram is irregularly sampled (one value per beat), so it is cubic-spline
    resampled onto a uniform grid before the Welch estimate -- the standard HRV approach.
    """
    p = cfg.features.dynamic
    if rr.size < p.min_rr_spectral:
        return float("nan")

    times = np.cumsum(rr) / 1000.0  # beat times in seconds
    duration_s = float(times[-1] - times[0])

    # A spectrum cannot resolve a band it does not span: reading LF power at 0.04 Hz
    # needs at least `lf_cycles_required` full cycles, i.e. ~25 s of tachogram for one
    # cycle. A 5 s analysis window holds ~4.5 s and would report LF ~ 0 and hence a
    # ratio of exactly 0.00 -- a fabricated number that looks like a measurement.
    # Return NaN instead; widen preprocessing.window.size_samples to get a real value.
    min_duration_s = p.lf_cycles_required / p.lf_band_hz[0]
    if duration_s < min_duration_s:
        return float("nan")

    resample_fs = p.tachogram_resample_hz
    uniform_t = np.arange(times[0], times[-1], 1.0 / resample_fs)
    if uniform_t.size < 8:
        return float("nan")

    spline = interpolate.interp1d(times, rr, kind="cubic", fill_value="extrapolate")
    tachogram = spline(uniform_t)
    tachogram = tachogram - tachogram.mean()

    nperseg = min(tachogram.size, 256)
    freqs, psd = sp_signal.welch(tachogram, fs=resample_fs, nperseg=nperseg)

    lf = psd[(freqs >= p.lf_band_hz[0]) & (freqs < p.lf_band_hz[1])].sum()
    hf = psd[(freqs >= p.hf_band_hz[0]) & (freqs < p.hf_band_hz[1])].sum()
    if hf <= 0:
        return float("nan")
    return float(lf / hf)


# ------------------------------------------------------------------------------ PRSA


def _prsa_from_rr(rr: np.ndarray, cfg: Config) -> float:
    """Phase-rectified signal averaging -- deceleration capacity (ms), Bauer et al.

    Anchors are beats where RR increases (deceleration); windows of +/-L beats around
    each anchor are averaged, and the capacity is the Haar-like 2-point contrast at the
    centre of that average.
    """
    p = cfg.features.dynamic
    length = p.prsa_window
    if rr.size < 2 * length + 1:
        return float("nan")

    anchors = [
        i for i in range(length, rr.size - length) if rr[i] > rr[i - 1]
    ]  # deceleration anchors
    if not anchors:
        return float("nan")

    windows = np.stack([rr[i - length : i + length] for i in anchors])
    averaged = windows.mean(axis=0)

    # Bauer's DC = [X(0) + X(1) - X(-1) - X(-2)] / 4, centre index = `length`.
    c = length
    if c + 1 >= averaged.size or c - 2 < 0:
        return float("nan")
    return float((averaged[c] + averaged[c + 1] - averaged[c - 1] - averaged[c - 2]) / 4.0)


# ------------------------------------------------------------------- symbolic dynamics


def _symbolic_from_rr(rr: np.ndarray, cfg: Config) -> float:
    """Porta symbolic dynamics: 0V% -- the fraction of 3-beat words with no variation.

    The RR series is uniformly quantised into `symbol_levels` bins spanning its range;
    each consecutive triplet becomes a word, and 0V counts words whose three symbols are
    all equal (a stable pattern).
    """
    p = cfg.features.dynamic
    if rr.size < p.symbolic_word_length + 1:
        return float("nan")

    # Uniform-over-range is Porta's published quantisation and is the default. It is
    # only safe on an artefact-free series: before filter_rr_artifacts() a single 4320 ms
    # outlier stretched the range so 604/612 beats fell in one bin and 0V% saturated at
    # 96%. With artefacts removed the bins populate properly (measured on r01/r08), so the
    # fix belongs in the RR filter, not in the quantiser. `equiprobable` remains available
    # for recordings with very skewed RR spreads.
    method = p.symbolic_binning
    if method == "uniform":
        spread = rr.max() - rr.min()
        if spread <= 0:
            return float("nan")
        symbols = np.clip(
            ((rr - rr.min()) / spread * p.symbol_levels).astype(int), 0, p.symbol_levels - 1
        )
    elif method == "equiprobable":
        if np.unique(rr).size < p.symbol_levels:
            return float("nan")
        edges = np.quantile(rr, np.linspace(0.0, 1.0, p.symbol_levels + 1)[1:-1])
        symbols = np.digitize(rr, edges)
    else:
        raise ValueError(f"unknown symbolic_binning '{method}' (uniform|equiprobable)")

    words = np.lib.stride_tricks.sliding_window_view(symbols, p.symbolic_word_length)
    no_variation = np.all(words == words[:, :1], axis=1)
    return float(np.count_nonzero(no_variation) / words.shape[0] * 100.0)


# --------------------------------------------------------------------- public wrappers
#
# The public feature functions take a SIGNAL (one window) and derive its RR series; the
# `_*_from_rr` cores above take an RR series directly, so record-level extraction can
# pool RR intervals across windows. Concatenating the fECG signal instead does NOT work:
# pan_tompkins_peaks thresholds on the whole array's mean+SD, so a few high-amplitude
# windows raise the threshold above every other window's beats (measured: 2 detected
# intervals across a 295 s record vs ~600 expected).


def _from_signal(core, fecg: np.ndarray, fs: float, cfg: Optional[Config]) -> float:
    if cfg is None:
        cfg = load_config()
    return core(_rr(fecg, fs, cfg), cfg)


def apen(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Approximate entropy of the RR series (m=2, r=0.2*SD by default)."""
    return _from_signal(_apen_from_rr, fecg, fs, cfg)


def sampen(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Sample entropy of the RR series (m=2, r=0.2*SD by default)."""
    return _from_signal(_sampen_from_rr, fecg, fs, cfg)


def mse(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Multiscale entropy: mean SampEn over coarse-grained scales 1..max_scale."""
    return _from_signal(_mse_from_rr, fecg, fs, cfg)


def dfa_alpha(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Short-term DFA scaling exponent alpha1 over box sizes 4..16 beats."""
    return _from_signal(_dfa_alpha_from_rr, fecg, fs, cfg)


def lf_hf_ratio(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """LF/HF power ratio of the RR tachogram, using FETAL band edges."""
    return _from_signal(_lf_hf_from_rr, fecg, fs, cfg)


def prsa(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """PRSA deceleration capacity (ms), Bauer et al."""
    return _from_signal(_prsa_from_rr, fecg, fs, cfg)


def symbolic_dynamics(fecg: np.ndarray, fs: float, cfg: Optional[Config] = None) -> float:
    """Porta symbolic dynamics: 0V% -- fraction of 3-beat words with no variation."""
    return _from_signal(_symbolic_from_rr, fecg, fs, cfg)


DYNAMIC_FROM_RR = {
    "prsa": _prsa_from_rr,
    "mse": _mse_from_rr,
    "apen": _apen_from_rr,
    "sampen": _sampen_from_rr,
    "dfa_alpha": _dfa_alpha_from_rr,
    "lf_hf_ratio": _lf_hf_from_rr,
    "symbolic_dynamics": _symbolic_from_rr,
}
