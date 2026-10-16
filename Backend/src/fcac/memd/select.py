"""src/fcac/memd/select.py

Scores each IMF by spectral coherence x in-band power in the fetal band (20-70 Hz
default, CLAUDE.md §2 "MEMD role"), and selects the IMFs judged to carry fetal-band
content. "IMF 3-5" is a heuristic only -- selection is scored per recording, not a
fixed index range.

Note: Direct_1 (ADFECGDB scalp) may be used here as a scoring reference ONLY
(CLAUDE.md §6: "Never write code that touches Direct_1 outside io/, evaluation/, or
memd/select.py").

Implements: CLAUDE.md §5 MEMDOutput.scores / .selected.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal as sp_signal

from fcac.config import Config


def imf_in_band_power(imf: np.ndarray, fs: float, band_hz: Tuple[float, float]) -> float:
    """Fraction of an IMF's Welch PSD power falling inside `band_hz`.

    imf : (N,) or (C, N) -- averaged across channels if 2D.
    """
    if imf.ndim == 2:
        return float(np.mean([imf_in_band_power(ch, fs, band_hz) for ch in imf]))
    freqs, psd = sp_signal.welch(imf, fs=fs)
    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    total_power = psd.sum()
    if total_power <= 0:
        return 0.0
    return float(psd[in_band].sum() / total_power)


def imf_spectral_coherence(
    imf: np.ndarray, reference: np.ndarray, fs: float, band_hz: Tuple[float, float]
) -> float:
    """Mean magnitude-squared coherence between an IMF and a fetal-band reference
    signal, averaged over `band_hz`.

    Used when a same-length reference signal is available (e.g. ADFECGDB Direct_1, for
    scoring only -- see module docstring). Without a reference, callers should fall back
    to in-band power alone.

    imf, reference : (N,)
    """
    freqs, coherence = sp_signal.coherence(imf, reference, fs=fs)
    in_band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not np.any(in_band):
        return 0.0
    return float(coherence[in_band].mean())


def imf_scores(
    imfs: np.ndarray,
    fs: float,
    cfg: Config,
    reference: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Score each IMF: spectral_coherence x in_band_power (CLAUDE.md §2).

    imfs      : (K, C, N) or (K, N)
    reference : optional (N,) same-length reference (e.g. filtered Direct_1) for the
                coherence term. If None, coherence is fixed at 1.0 and the score
                reduces to in-band power alone.
    Returns   : (K,) scores.
    """
    band_hz = tuple(cfg.memd.fetal_band_hz)
    scores = np.zeros(imfs.shape[0], dtype=np.float32)
    for k in range(imfs.shape[0]):
        imf = imfs[k]
        power = imf_in_band_power(imf, fs, band_hz)
        if reference is not None and imf.ndim == 1:
            coherence = imf_spectral_coherence(imf, reference, fs, band_hz)
        else:
            coherence = 1.0
        scores[k] = coherence * power
    return scores


def select_fetal_band(
    scores: np.ndarray, cfg: Config, threshold: Optional[float] = None
) -> List[int]:
    """Select IMF indices judged to carry fetal-band content.

    Uses cfg.memd.score_top_k if set (fixed top-k), otherwise a score threshold
    (mean + 1 std by default, override via `threshold`) -- CLAUDE.md §2: selection is
    scored per recording, not a fixed "IMF 3-5" heuristic.
    """
    top_k = cfg.memd.score_top_k
    if top_k is not None:
        return [int(i) for i in np.argsort(scores)[::-1][:top_k]]

    cutoff = threshold if threshold is not None else scores.mean() + scores.std()
    return [int(i) for i in np.where(scores >= cutoff)[0]]


def reconstruct_fetal_band(imfs: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    """Sum the selected IMFs back into a single fetal-band signal.

    This is the MEMD -> PINN handoff (CLAUDE.md section 5 FetalBandWindows.x). MEMD's job
    stops here: what comes out still contains BOTH maternal and fetal complexes, with
    motion/baseline-drift/muscle artefact removed. Maternal-fetal separation is the
    PINN's job (CLAUDE.md section 2 "MEMD role") -- do not read this as a fetal ECG.

    imfs     : (K, C, N)
    selected : IMF indices from select_fetal_band()
    Returns  : (C, N). Falls back to the full reconstruction if nothing was selected,
               so a pathological window degrades to "unfiltered" rather than to silence.
    """
    if len(selected) == 0:
        return imfs.sum(axis=0).astype(np.float32)
    return imfs[list(selected)].sum(axis=0).astype(np.float32)


def build_fetal_band_windows(
    imfs_per_window: Sequence[np.ndarray],
    fs: float,
    cfg: Config,
    references: Optional[Sequence[np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Score, select and reconstruct every window of one recording.

    imfs_per_window : length W, each (K_w, C, N) -- K may differ per window.
    references      : optional length-W sequence of (N,) scoring references (ADFECGDB
                      filtered Direct_1 window). Scoring-only, never a model input
                      (CLAUDE.md section 6).

    Returns (x (W, C, N) fetal-band windows, scores (W, K_max) zero-padded, selected).
    """
    x_windows: List[np.ndarray] = []
    scores_per_window: List[np.ndarray] = []
    selected_per_window: List[List[int]] = []

    for w, imfs in enumerate(imfs_per_window):
        reference = references[w] if references is not None else None
        scores = imf_scores(imfs, fs, cfg, reference=reference)
        selected = select_fetal_band(scores, cfg)
        x_windows.append(reconstruct_fetal_band(imfs, selected))
        scores_per_window.append(scores)
        selected_per_window.append(selected)

    # K varies per window, so pad the score matrix to the widest one to keep the
    # MEMDOutput.scores (W, K) contract rectangular.
    k_max = max(s.shape[0] for s in scores_per_window)
    scores_padded = np.zeros((len(scores_per_window), k_max), dtype=np.float32)
    for w, s in enumerate(scores_per_window):
        scores_padded[w, : s.shape[0]] = s

    return np.stack(x_windows, axis=0).astype(np.float32), scores_padded, selected_per_window
