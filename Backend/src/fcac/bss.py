"""src/fcac/bss.py

Blind-source-separation fetal-ECG extractor: FastICA over the 4 abdominal channels, then an
annotation-free choice of which independent component is the fetal one. An established
alternative to the PINN (Andreotti et al. 2016 ranked BSS most accurate on simulated data).

FastICA unmixes statistically independent sources; the maternal and fetal hearts are
separate electrical sources, so they tend to land in different components. With only 4
channels (CLAUDE.md §2 "Channels") at most 4 sources, noise included, can be separated.

Implements: alternative to CLAUDE.md §5 stage "PINNOutput" (fecg (W, 5000)); not wired into
the default pipeline.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.decomposition import FastICA

from fcac.config import Config
from fcac.memd.diagnostics import pan_tompkins_peaks


def unmix_record(windows: np.ndarray, cfg: Config) -> np.ndarray:
    """FastICA fitted on all of a record's windows concatenated in time, then applied.

    windows : (W, C, N) -> components (W, C, N). Component order and sign are arbitrary.
    """
    w, c, n = windows.shape
    series = windows.transpose(1, 0, 2).reshape(c, w * n)  # (C, W*N)
    ica = FastICA(
        n_components=c,
        algorithm=cfg.bss.algorithm,
        fun=cfg.bss.fun,
        max_iter=cfg.bss.max_iter,
        tol=cfg.bss.tol,
        whiten="unit-variance",
        random_state=cfg.seed,
    )
    sources = ica.fit_transform(series.T).T  # (C, W*N)
    return sources.reshape(c, w, n).transpose(1, 0, 2).astype(np.float32)


def _rate_and_regularity(x: np.ndarray, fs: float, min_beats: int) -> Tuple[float, float]:
    """(heart rate bpm, RR coefficient of variation); (nan, inf) if too few beats."""
    peaks = pan_tompkins_peaks(x, fs)
    if len(peaks) < min_beats:
        return float("nan"), float("inf")
    rr = np.diff(peaks) / fs
    return float(60.0 / rr.mean()), float(rr.std() / rr.mean())


def select_fetal_component(components: np.ndarray, fs: float, cfg: Config) -> Tuple[int, float]:
    """Index of the fetal component of one window (C, N) and its rate (bpm), chosen without
    any reference annotation (see configs/default.yaml `bss`)."""
    lo, hi = cfg.quality.fetal_hr_min_bpm, cfg.quality.fetal_hr_max_bpm
    stats = [_rate_and_regularity(components[k], fs, cfg.bss.min_rr_beats) for k in range(len(components))]
    in_band = [k for k, (hr, _) in enumerate(stats) if np.isfinite(hr) and lo <= hr <= hi]
    if in_band:
        best = min(in_band, key=lambda k: stats[k][1])
        return best, stats[best][0]
    centre = 0.5 * (lo + hi)
    finite = [k for k, (hr, _) in enumerate(stats) if np.isfinite(hr)]
    if finite:
        best = min(finite, key=lambda k: abs(stats[k][0] - centre))
        return best, stats[best][0]
    return 0, float("nan")


def extract_fecg_bss(windows: np.ndarray, fs: float, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """(W, C, N) windows of one record -> single-channel fECG (W, N) and the chosen component
    index per window (W,)."""
    components = unmix_record(windows, cfg)
    chosen = np.array([select_fetal_component(components[w], fs, cfg)[0] for w in range(len(components))])
    fecg = np.stack([components[w, chosen[w]] for w in range(len(components))])
    return fecg.astype(np.float32), chosen
