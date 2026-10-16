#!/usr/bin/env python3
"""scripts/plot_report_figures.py

Regenerates the two figures the project spec asks for that no pipeline script saves:

  * MEMD IMF-score bar chart per dataset (spec section 7.1): value axis 0.0-1.0, the
    selected IMFs shaded. Reads data/processed/fetal_band/<dataset>/*.npz (written by
    scripts/02_memd.py). Scores are normalised per window to their maximum, then averaged
    over all windows of the dataset.
  * Feature distributions per dataset (spec section 12, deliverable 3). Reads
    data/processed/features/<dataset>/*.npz (written by scripts/05_features.py).
    Datasets with no feature files (CinC 2013 today) are skipped and said so.

Everything is saved to reports/figures/ via fcac.viz.plots.save_figure.

Usage:
    .venv/bin/python scripts/plot_report_figures.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fcac.features.extract import FEATURE_NAMES
from fcac.viz.plots import apply_style, save_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ("adfecgdb", "ninfea", "cinc2013")
FETAL_BAND_DIR = Path("data/processed/fetal_band")
FEATURE_DIR = Path("data/processed/features")


def plot_memd_scores(dataset: str) -> Path | None:
    """Spec 7.1 bar chart: mean per-window-normalised IMF score, selected IMFs shaded."""
    files = sorted((FETAL_BAND_DIR / dataset).glob("*.npz"))
    if not files:
        logger.warning("%s: no fetal-band files in %s, skipping", dataset, FETAL_BAND_DIR)
        return None

    normalised, selected_count, n_windows, max_k = [], None, 0, 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        scores = d["scores"].astype(np.float64)
        if scores.size == 0:
            continue
        max_k = max(max_k, scores.shape[1])
        peak = scores.max(axis=1, keepdims=True)
        peak[peak == 0] = 1.0
        normalised.append(scores / peak)
        n_windows += scores.shape[0]
        if selected_count is None:
            selected_count = np.zeros(0)
        counts = np.zeros(scores.shape[1])
        for sel in d["selected"]:
            for k in np.atleast_1d(sel).astype(int):
                counts[k] += 1
        selected_count = _add(selected_count, counts)

    mean_score = np.zeros(max_k)
    weight = np.zeros(max_k)
    for arr in normalised:
        mean_score[: arr.shape[1]] += arr.sum(axis=0)
        weight[: arr.shape[1]] += arr.shape[0]
    mean_score /= np.maximum(weight, 1)
    frac_selected = selected_count / max(n_windows, 1)
    shaded = np.where(frac_selected >= 0.5)[0]

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(max_k)
    ax.bar(x, mean_score, color="tab:gray")
    for k in shaded:
        ax.axvspan(k - 0.5, k + 0.5, color="tab:orange", alpha=0.3)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k + 1) for k in x])
    ax.set_xlabel("IMF index (1-based)")
    ax.set_ylabel("normalised score (max = 1)")
    ax.set_title(
        f"{dataset}: MEMD IMF fetal-band scores ({len(files)} records, {n_windows} windows)\n"
        "shaded = IMF selected in >= 50% of windows"
    )
    fig.tight_layout()
    path = save_figure(fig, f"memd_scores_{dataset}")
    plt.close(fig)
    logger.info("%s: wrote %s (shaded IMFs: %s)", dataset, path, (shaded + 1).tolist())
    return path


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = max(a.size, b.size)
    out = np.zeros(n)
    out[: a.size] += a
    out[: b.size] += b
    return out


def load_features(dataset: str) -> np.ndarray | None:
    files = sorted((FEATURE_DIR / dataset).glob("*.npz"))
    if not files:
        return None
    return np.concatenate([np.load(f, allow_pickle=True)["x"] for f in files], axis=0)


def plot_feature_distributions() -> Path | None:
    """One histogram panel per feature, one coloured histogram per dataset."""
    data = {}
    for dataset in DATASETS:
        x = load_features(dataset)
        if x is None:
            logger.warning("%s: no feature files in %s -- left out of the figure",
                           dataset, FEATURE_DIR / dataset)
            continue
        data[dataset] = x
    if not data:
        return None

    apply_style()
    n = len(FEATURE_NAMES)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows))
    axes = axes.ravel()
    colours = {"adfecgdb": "tab:blue", "ninfea": "tab:orange", "cinc2013": "tab:green"}
    for i, name in enumerate(FEATURE_NAMES):
        ax = axes[i]
        parts = []
        for dataset, x in data.items():
            v = x[:, i]
            parts.append((dataset, v[np.isfinite(v)], v.size))
        pooled = np.concatenate([p[1] for p in parts]) if parts else np.array([])
        if pooled.size:
            lo, hi = np.percentile(pooled, [1, 99])
            bins = np.linspace(lo, hi, 30) if hi > lo else 30
        else:
            bins = 30
        for dataset, v, total in parts:
            if v.size:
                ax.hist(v, bins=bins, density=True, alpha=0.5, color=colours[dataset],
                        label=f"{dataset} (n={v.size}/{total})")
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
        if not pooled.size:
            ax.text(0.5, 0.5, "all NaN", ha="center", va="center", transform=ax.transAxes)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = [], []
    for ax in axes[:n]:
        h, l = ax.get_legend_handles_labels()
        if len(l) > len(labels):
            handles, labels = h, l
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Feature distributions per window (n = finite / total windows; "
                 "x-range clipped to 1-99th percentile)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = save_figure(fig, "feature_distributions")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def main() -> None:
    for dataset in DATASETS:
        plot_memd_scores(dataset)
    plot_feature_distributions()


if __name__ == "__main__":
    main()
