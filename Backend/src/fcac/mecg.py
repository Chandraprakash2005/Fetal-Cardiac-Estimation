"""src/fcac/mecg.py

Maternal ECG cancellation by template subtraction, applied between MEMD (stage 2) and
the PINN (stage 3).

Why this exists. The pipeline had no explicit maternal suppression: MEMD removes
motion/drift/muscle artefact only, and the PINN's sole separation mechanism is a soft
harmonic constraint. On recordings where the maternal complex dominates, the extracted
"fetal" ECG locks onto the maternal rate (measured: ADFECGDB r04/r07/r10 output 82-93 bpm
against a true fetal 125-131 bpm).

Why it is ADAPTIVE. Template subtraction is not free -- the maternal template overlaps
fetal complexes, so subtracting it on an already-clean recording destroys fetal signal.
Measured on ADFECGDB (QRS F1 vs .qrs annotations, PC1-projected):

    record   MEMD    + cancellation   delta
    r01      0.934        0.760       -0.174   <- clean: cancellation HURTS
    r08      0.959        0.722       -0.237   <- clean: cancellation HURTS
    r04      0.202        0.332       +0.130   <- maternal-locked: cancellation HELPS
    r07      0.190        0.453       +0.263
    r10      0.217        0.388       +0.171

The two representations are complementary, so `adaptive_cancel` picks between them per
window using an UNSUPERVISED criterion -- whether the resulting heart rate falls in the
fetal band -- never the ground truth.

Implements: CLAUDE.md section 2 "MEMD role" (maternal-fetal separation), supporting the
PINN stage.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from fcac.config import Config, load_config
from fcac.memd.diagnostics import pan_tompkins_bpm, pan_tompkins_peaks
from fcac.pinn.coords import project_to_single_channel


def maternal_template_subtract(
    x: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> np.ndarray:
    """Remove the maternal complex from a (C, N) window by template subtraction.

    Per channel: detect the dominant (maternal) R peaks, build a MEDIAN beat template
    over a window around them, then subtract that template -- amplitude-scaled per beat
    by least squares -- from every maternal beat.

    The median rather than the mean is what makes this work: fetal complexes fall at
    different phases of the maternal cycle, so they average into the template as a bias
    but survive a median. Per-beat amplitude scaling absorbs respiration-driven maternal
    amplitude drift.

    Returns a (C, N) array of the same shape; channels with too few detected beats are
    passed through untouched.
    """
    if cfg is None:
        cfg = load_config()
    params = cfg.mecg_cancellation

    half = max(1, int(round(params.template_half_ms / 1000.0 * fs)))
    out = np.asarray(x, dtype=np.float64).copy()

    for c in range(out.shape[0]):
        channel = out[c]
        peaks = [
            int(p)
            for p in pan_tompkins_peaks(channel, fs)
            if p - half >= 0 and p + half < channel.size
        ]
        if len(peaks) < params.min_beats:
            continue

        beats = np.stack([channel[p - half : p + half] for p in peaks])
        template = np.median(beats, axis=0)
        energy = float(template @ template)
        if energy <= 0:
            continue

        for p in peaks:
            segment = channel[p - half : p + half]
            scale = float(segment @ template) / energy
            out[c, p - half : p + half] -= scale * template

    return out.astype(np.float32)


def _is_fetal_rate(bpm: float, cfg: Config) -> bool:
    q = cfg.quality
    return bool(np.isfinite(bpm) and q.fetal_hr_min_bpm <= bpm <= q.fetal_hr_max_bpm)


def adaptive_cancel(
    x: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> Tuple[np.ndarray, bool]:
    """Return (window, cancellation_applied) choosing the better representation.

    Decision rule, entirely unsupervised -- it never sees a reference annotation:

      * If the window's own rate is already fetal (110-160 bpm), keep it. Cancellation
        would only damage a signal that is already fetal-dominated.
      * Otherwise the window is maternal-locked. Cancel, and keep the result if that
        moves the rate into the fetal band. If cancellation does not help either, keep
        it anyway -- the window has failed regardless, and `fcac.quality` will gate it.

    Modes (cfg.mecg_cancellation.mode): "adaptive" (default), "always", "never".
    """
    if cfg is None:
        cfg = load_config()
    mode = cfg.mecg_cancellation.mode

    if mode == "never" or not cfg.mecg_cancellation.enabled:
        return np.asarray(x, dtype=np.float32), False
    if mode == "always":
        return maternal_template_subtract(x, fs, cfg), True
    if mode != "adaptive":
        raise ValueError(f"unknown mecg_cancellation.mode '{mode}' (adaptive|always|never)")

    if _is_fetal_rate(pan_tompkins_bpm(project_to_single_channel(x), fs), cfg):
        return np.asarray(x, dtype=np.float32), False

    cancelled = maternal_template_subtract(x, fs, cfg)
    return cancelled, True


def adaptive_cancel_windows(
    windows: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply `adaptive_cancel` to every window of a (W, C, N) array.

    Returns (windows, applied) where `applied` is a (W,) bool array recording which
    windows were cancelled -- worth logging, since a record where most windows needed
    cancellation is a low-quality record.
    """
    if cfg is None:
        cfg = load_config()
    out = np.empty_like(windows, dtype=np.float32)
    applied = np.zeros(windows.shape[0], dtype=bool)
    for w in range(windows.shape[0]):
        out[w], applied[w] = adaptive_cancel(windows[w], fs, cfg)
    return out, applied
