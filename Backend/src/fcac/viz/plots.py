"""src/fcac/viz/plots.py

Shared matplotlib style and save-to-reports/figures helper, so every diagnostic and
result plot in this project looks consistent.

Implements: CLAUDE.md §6 "Plots".
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

REPORTS_FIGURES_DIR = Path("reports/figures")

STYLE = {
    "figure.figsize": (10, 5),
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
}


def apply_style() -> None:
    plt.rcParams.update(STYLE)


def save_figure(
    fig: plt.Figure, name: str, figures_dir: Path = REPORTS_FIGURES_DIR
) -> Path:
    """Save `fig` as `{figures_dir}/{name}.png`, creating the directory if needed."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{name}.png"
    fig.savefig(path, dpi=STYLE["figure.dpi"], bbox_inches="tight")
    return path


def plot_extraction_summary(
    abdominal: "np.ndarray",
    fetal_band: "np.ndarray",
    fecg: "np.ndarray",
    fs: float,
    title: str,
    reference: "Optional[np.ndarray]" = None,
    seconds: float = 5.0,
):
    """One-page visual check of the whole extraction chain for a single window.

    Four rows: the preprocessed abdominal channels, the MEMD fetal-band reconstruction,
    the PINN's isolated fECG (with the scalp reference overlaid when there is one), and
    the Welch spectra of the last two.

    abdominal  : (C, N) preprocessed window
    fetal_band : (C, N) sum of the selected IMFs
    fecg       : (N,)   PINN output
    reference  : (N,)   ADFECGDB scalp ground truth, or None (NInFEA has none)

    Returns the Figure; pass it to `save_figure`.
    """
    import numpy as np
    from scipy import signal as sp_signal

    apply_style()
    n = min(int(seconds * fs), abdominal.shape[1])
    t = np.arange(n) / fs

    fig, axes = plt.subplots(4, 1, figsize=(12, 11))

    for c in range(abdominal.shape[0]):
        # Stack the channels with a constant offset so all four stay readable.
        axes[0].plot(t, abdominal[c, :n] + 4.0 * c, lw=0.7, label=f"ch{c}")
    axes[0].set_title(f"{title}\n1. Preprocessed abdominal channels (filtered, z-scored)")
    axes[0].set_ylabel("z-score (offset)")
    axes[0].legend(loc="upper right", ncol=4, fontsize=8)

    for c in range(fetal_band.shape[0]):
        axes[1].plot(t, fetal_band[c, :n] + 4.0 * c, lw=0.7)
    axes[1].set_title(
        "2. MEMD fetal-band reconstruction -- artefact removed, "
        "STILL CONTAINS MATERNAL + FETAL"
    )
    axes[1].set_ylabel("a.u. (offset)")

    fecg_n = fecg[:n]
    axes[2].plot(t, fecg_n / (np.abs(fecg_n).max() + 1e-12), lw=0.9, label="PINN fECG")
    if reference is not None:
        ref_n = reference[:n]
        axes[2].plot(
            t,
            ref_n / (np.abs(ref_n).max() + 1e-12),
            lw=0.9,
            alpha=0.6,
            label="scalp reference (ground truth)",
        )
    axes[2].set_title("3. PINN-isolated fetal ECG (peak-normalised)")
    axes[2].set_ylabel("normalised")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(loc="upper right", fontsize=8)

    for label, sig in (("fetal-band ch0", fetal_band[0]), ("PINN fECG", fecg)):
        freqs, psd = sp_signal.welch(sig, fs=fs, nperseg=min(1024, sig.shape[0]))
        axes[3].semilogy(freqs, psd, lw=0.9, label=label)
    axes[3].set_xlim(0, min(150.0, fs / 2))
    axes[3].set_title("4. Welch spectra")
    axes[3].set_xlabel("frequency (Hz)")
    axes[3].set_ylabel("PSD")
    axes[3].legend(fontsize=8)

    fig.tight_layout()
    return fig
