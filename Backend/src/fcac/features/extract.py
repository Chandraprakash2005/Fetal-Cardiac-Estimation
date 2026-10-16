"""src/fcac/features/extract.py

Extracts the 12 interpretable cardiac biomarkers per window (CLAUDE.md §5
FeatureVector): 5 morphological + 7 dynamic features. FEATURE_NAMES fixes the column
order -- every other module (CVAE input, evaluation) must import this list rather than
re-deriving indices.

All 12 are implemented as of 2026-09-08. Their algorithm/parameter choices are standard
defaults documented in configs/default.yaml under `features.*` and in each module's
header -- they are NOT supervisor-confirmed (plan.md item 4), so treat the numbers as
provisional until signed off.

Features return NaN (never an exception) when a window has too few beats to support
them. This matters: a 5 s window at fetal rates holds only ~8-12 RR intervals, while
DFA/MSE/entropy normally need hundreds, so the dynamic columns are expected to be
sparsely populated at the default 5000-sample window. Widen
preprocessing.window.size_samples if the dynamic features are needed densely.

Implements: CLAUDE.md §8 item 7.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fcac.config import Config, load_config
from fcac.features import dynamic, morphological
from fcac.features.fiducials import filter_rr_artifacts
from fcac.memd.diagnostics import pan_tompkins_peaks

FEATURE_NAMES = [
    "qtc_sigma",
    "p_dispersion",
    "qrs_mean",
    "qrs_var",
    "qrs_asymmetry",
    "prsa",
    "mse",
    "apen",
    "sampen",
    "dfa_alpha",
    "lf_hf_ratio",
    "symbolic_dynamics",
]

assert len(FEATURE_NAMES) == 12
assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

_FEATURE_FUNCS = {
    "qtc_sigma": morphological.qtc_sigma,
    "p_dispersion": morphological.p_dispersion,
    "qrs_mean": morphological.qrs_mean,
    "qrs_var": morphological.qrs_var,
    "qrs_asymmetry": morphological.qrs_asymmetry,
    "prsa": dynamic.prsa,
    "mse": dynamic.mse,
    "apen": dynamic.apen,
    "sampen": dynamic.sampen,
    "dfa_alpha": dynamic.dfa_alpha,
    "lf_hf_ratio": dynamic.lf_hf_ratio,
    "symbolic_dynamics": dynamic.symbolic_dynamics,
}


def extract_features(
    fecg: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> np.ndarray:
    """fecg : (5000,) single-channel isolated fetal ECG window (PINNOutput.fecg[w]).

    Returns (12,) float32 vector in FEATURE_NAMES order. Individual entries are NaN when
    the window has too few beats for that feature -- see the module docstring.
    """
    if cfg is None:
        cfg = load_config()
    values = [_FEATURE_FUNCS[name](fecg, fs, cfg) for name in FEATURE_NAMES]
    return np.array(values, dtype=np.float32)


def extract_features_batch(
    fecg_windows: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> np.ndarray:
    """fecg_windows : (W, 5000) -> (W, 12)."""
    if cfg is None:
        cfg = load_config()
    return np.stack(
        [extract_features(fecg_windows[w], fs, cfg) for w in range(fecg_windows.shape[0])]
    )


def pooled_rr_intervals(
    fecg_windows: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> np.ndarray:
    """One RR series (ms) for a whole record, pooled across its windows.

    R peaks are detected PER WINDOW (where the detector's global threshold is valid) and
    then shifted into whole-record sample coordinates before differencing. Detecting on
    the concatenated signal instead fails badly -- see the note in features/dynamic.py.

    Windows are contiguous and non-overlapping at the default stride, so beat times
    compose exactly. If stride != window size the offsets below would need the stride,
    hence the explicit check.
    """
    if cfg is None:
        cfg = load_config()
    size = cfg.preprocessing.window.size_samples
    stride = cfg.preprocessing.window.stride_samples
    if stride != size:
        raise ValueError(
            "pooled_rr_intervals assumes non-overlapping windows "
            f"(stride {stride} != size {size}); overlapping windows would double-count beats."
        )

    peaks = [
        pan_tompkins_peaks(fecg_windows[w], fs) + w * size
        for w in range(fecg_windows.shape[0])
    ]
    all_peaks = np.concatenate(peaks) if peaks else np.empty(0)
    if all_peaks.size < 2:
        return np.empty(0, dtype=np.float64)
    return filter_rr_artifacts(np.diff(np.sort(all_peaks)) / fs * 1000.0, cfg)


def extract_features_record(
    fecg_windows: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> np.ndarray:
    """Record-level (12,) feature vector for a whole recording.

    Morphological features are averaged over the per-window values (they are beat-shape
    measures and are well defined within a 5 s window). Dynamic features are computed
    once on the POOLED RR series, which is the only way they are meaningful: a 5 s window
    holds ~10 RR intervals, while DFA/entropy need dozens and LF/HF needs >= 25 s to
    resolve its 0.04 Hz lower edge at all.
    """
    if cfg is None:
        cfg = load_config()

    per_window = extract_features_batch(fecg_windows, fs, cfg)  # (W, 12)
    out = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    for i, name in enumerate(FEATURE_NAMES):
        if name in dynamic.DYNAMIC_FROM_RR:
            continue
        column = per_window[:, i]
        finite = column[np.isfinite(column)]
        if finite.size:
            out[i] = finite.mean()

    rr = pooled_rr_intervals(fecg_windows, fs, cfg)
    if rr.size:
        for i, name in enumerate(FEATURE_NAMES):
            core = dynamic.DYNAMIC_FROM_RR.get(name)
            if core is not None:
                out[i] = core(rr, cfg)
    return out
