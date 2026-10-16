#!/usr/bin/env python3
"""scripts/plot_preprocessing.py

Before/after figure for the preprocessing stage (spec stage 2, CLAUDE.md section 6: verify
preprocessing visually, time domain + Welch spectrum, before any downstream stage uses
it). One figure per dataset in reports/figures/preprocessing_<dataset>_<record>.png.

Runs the REAL pipeline stages from fcac.preprocessing.pipeline on the raw recording
(filter at native rate -> [resample] -> [kurtosis select] -> z-score -> window) and shows
one channel before and after, over the same 5 s of recording time (the window starts
after the edge trim). Band, notch and edge trim come from the config -- nothing is
hard-coded here, so the titles always state what was actually applied.

Usage:
    .venv/bin/python scripts/plot_preprocessing.py                 # all three datasets
    .venv/bin/python scripts/plot_preprocessing.py --dataset ninfea --record 10
"""

from __future__ import annotations

import argparse
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sp_signal

from fcac.config import load_config
from fcac.io.adfecgdb import load_adfecgdb_record
from fcac.io.cinc2013 import load_cinc2013_record
from fcac.io.ninfea import load_ninfea_record
from fcac.preprocessing.channels import kurtosis_select
from fcac.preprocessing.pipeline import _bandpass_band, _filter_stage, preprocess
from fcac.viz.plots import apply_style, save_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LOADERS = {
    "adfecgdb": (load_adfecgdb_record, "r01"),
    "ninfea": (load_ninfea_record, "10"),
    "cinc2013": (load_cinc2013_record, "a01"),
}


def _welch(x: np.ndarray, fs: float):
    return sp_signal.welch(x, fs=fs, nperseg=min(len(x), int(4 * fs)))


def plot_one(dataset: str, record_id: str) -> None:
    cfg = load_config(dataset=dataset)
    loader, _ = LOADERS[dataset]
    rec = loader(record_id, cfg)
    pre = preprocess(rec, cfg)
    fs_raw, fs_new = float(rec.fs_raw), pre.fs
    low, high = _bandpass_band(cfg)
    notch = cfg.preprocessing.notch_enabled
    trim = cfg.preprocessing.edge_trim_seconds
    win_s = cfg.preprocessing.window.size_samples / fs_new

    # Which raw channel is the one shown in the "after" trace? Same rule as the pipeline.
    channel = 0
    ch_name = f"channel {channel}"
    if rec.signal.shape[0] > cfg.preprocessing.n_channels:
        filt = _filter_stage(rec.signal, fs_raw, cfg)
        from fcac.preprocessing.filters import resample_to
        filt = resample_to(filt, fs_raw, fs_new) if fs_raw != fs_new else filt
        _, idx, _ = kurtosis_select(filt, cfg.preprocessing.kurtosis_selection.n_select)
        channel = int(idx[0])
        ch_name = f"channel {channel} (highest kurtosis of {rec.signal.shape[0]})"

    raw = rec.signal[channel].astype(np.float64)
    after = pre.windows[0, 0].astype(np.float64)
    i0 = int(trim * fs_raw)
    raw_seg = raw[i0 : i0 + int(win_s * fs_raw)]
    t_raw = np.arange(raw_seg.size) / fs_raw + trim
    t_new = np.arange(after.size) / fs_new + trim

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 7))
    axes[0, 0].plot(t_raw, raw_seg, color="tab:red", lw=0.8)
    axes[0, 0].set_title(f"Before filtering -- raw, {fs_raw:.0f} Hz, {ch_name}")
    axes[1, 0].plot(t_new, after, color="tab:blue", lw=0.8)
    steps = [f"bandpass {low:g}-{high:g} Hz"]
    if notch:
        steps.append(f"{cfg.preprocessing.notch.freq_hz:g} Hz notch")
    steps.append(f"trim {trim:g} s")
    if fs_raw != fs_new:
        steps.append(f"resample to {fs_new:.0f} Hz")
    steps.append("z-score")
    axes[1, 0].set_title("After: " + " + ".join(steps))
    axes[1, 0].set_xlabel("time in recording (s)")
    axes[0, 0].set_ylabel("amplitude (raw units)")
    axes[1, 0].set_ylabel("z-score")

    f0, p0 = _welch(raw - raw.mean(), fs_raw)
    f1, p1 = _welch(pre.windows[0, 0], fs_new)
    for ax, f, p, colour, title in (
        (axes[0, 1], f0, p0, "tab:red", "Welch spectrum before (whole recording, mean removed)"),
        (axes[1, 1], f1, p1, "tab:blue", "Welch spectrum after (window 0)"),
    ):
        ax.semilogy(f, p, color=colour, lw=0.9)
        ax.axvspan(low, high, color="tab:green", alpha=0.08, label=f"passband {low:g}-{high:g} Hz")
        if notch:
            ax.axvline(cfg.preprocessing.notch.freq_hz, color="k", ls=":", lw=0.8,
                       label=f"{cfg.preprocessing.notch.freq_hz:g} Hz")
        ax.set_xlim(0, 200)
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
    axes[1, 1].set_xlabel("frequency (Hz)")
    for ax in axes[:, 1]:
        ax.set_ylabel("PSD")

    fig.suptitle(f"Preprocessing check -- {dataset} / {record_id} (config preset "
                 f"'{cfg.preprocessing.bandpass_preset}')", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path = save_figure(fig, f"preprocessing_{dataset}_{record_id}")
    plt.close(fig)
    logger.info("wrote %s | %s", path, ", ".join(steps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", choices=list(LOADERS))
    parser.add_argument("--record")
    args = parser.parse_args()
    datasets = [args.dataset] if args.dataset else list(LOADERS)
    for ds in datasets:
        plot_one(ds, args.record or LOADERS[ds][1])


if __name__ == "__main__":
    main()
