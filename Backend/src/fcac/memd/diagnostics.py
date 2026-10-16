"""src/fcac/memd/diagnostics.py

Visual/diagnostic checks for MEMD output -- run these before any downstream stage
consumes IMFs (CLAUDE.md §6 "Plots").

Implements: CLAUDE.md §8 item 5 (MEMD diagnostics).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal


def plot_imf_welch(imfs: np.ndarray, fs: float, save_path: Path, channel: int = 0) -> None:
    """Time-domain + Welch spectrum for every IMF of one window, one channel.

    imfs : (K, C, N) or (K, N)
    """
    k = imfs.shape[0]
    fig, axes = plt.subplots(k, 2, figsize=(10, 2 * k), squeeze=False)
    for i in range(k):
        imf = imfs[i, channel] if imfs.ndim == 3 else imfs[i]
        axes[i, 0].plot(imf)
        axes[i, 0].set_ylabel(f"IMF {i}")
        freqs, psd = sp_signal.welch(imf, fs=fs)
        axes[i, 1].semilogy(freqs, psd)
    axes[0, 0].set_title("Time domain")
    axes[0, 1].set_title("Welch PSD")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def pan_tompkins_peaks(x: np.ndarray, fs: float) -> np.ndarray:
    """Simplified Pan-Tompkins QRS detection: R-peak sample indices.

    Stages: bandpass (5-35 Hz, the QRS energy band) -> derivative -> square ->
    moving-window integration -> thresholding with a refractory period.

    Sanity-check grade only -- the precision fiducial detector lives in
    features/fiducials.py. Used here to answer "does this look like a fetal heartbeat?"
    """
    nyquist = fs / 2.0
    b, a = sp_signal.butter(3, [5.0 / nyquist, min(35.0 / nyquist, 0.99)], btype="band")
    filtered = sp_signal.filtfilt(b, a, x)

    derivative = np.gradient(filtered)
    squared = derivative**2

    # ~50 ms integration window: long enough to merge one QRS into a single lobe,
    # short enough not to merge neighbouring fetal beats at 160 bpm.
    win = max(1, int(0.05 * fs))
    integrated = np.convolve(squared, np.ones(win) / win, mode="same")

    threshold = integrated.mean() + 0.5 * integrated.std()
    # Fetal HR tops out around 200 bpm -> beats are never closer than 300 ms.
    refractory = int(0.3 * fs)
    peaks, _ = sp_signal.find_peaks(integrated, height=threshold, distance=refractory)
    return peaks


def pan_tompkins_bpm(x: np.ndarray, fs: float) -> float:
    """Mean heart rate in bpm from `pan_tompkins_peaks`, or nan if under 2 beats found.

    A fetal-band reconstruction that scores well should land near 110-160 bpm; a value
    near 60-90 means the selected IMFs are still maternal-dominated.
    """
    peaks = pan_tompkins_peaks(x, fs)
    if peaks.size < 2:
        return float("nan")
    mean_rr_seconds = float(np.mean(np.diff(peaks))) / fs
    if mean_rr_seconds <= 0:
        return float("nan")
    return 60.0 / mean_rr_seconds


def plot_imf_scores(scores: np.ndarray, selected: Sequence[int], save_path: Path) -> None:
    """Bar chart of per-IMF scores, highlighting the selected (fetal-band) IMFs."""
    selected_set = set(selected)
    colors = ["tab:orange" if i in selected_set else "tab:gray" for i in range(len(scores))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(len(scores)), scores, color=colors)
    ax.set_xlabel("IMF index")
    ax.set_ylabel("score (coherence x in-band power)")
    ax.set_title("MEMD IMF fetal-band scores (orange = selected)")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
