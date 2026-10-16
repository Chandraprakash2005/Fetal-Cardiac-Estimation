#!/usr/bin/env python3
"""scripts/plot_separation.py

Shows, for one window, how the pipeline pulls the fetal ECG out of the maternal one:

  1. preprocessed abdominal mixture (PC1 of the 4 channels)
  2. the MATERNAL component the template cancellation removes (median beat template,
     per-beat scaled; src/fcac/mecg.py) with maternal R peaks marked
  3. what is left after removing it (maternal-suppressed MEMD band)
  4. the PINN output (the pipeline's actual input: adaptive cancellation, then PINN only,
     never the BSS fallback) with fetal R peaks marked
  5. the scalp reference (ADFECGDB only; ground truth, never an input) with the PINN
     peaks overlaid for comparison

Heart rates come from the simplified Pan-Tompkins detector in fcac.memd.diagnostics, so
they are a sanity check, not the evaluation metric. The maternal component in row 2 is
always the forced template subtraction, even when the pipeline's adaptive rule skipped
cancellation for this window (the figure says which happened).

Implements the visual check of CLAUDE.md section 2 "MEMD role" (maternal-fetal separation
is the PINN's job, supported by mecg.py).

Usage:
    .venv/bin/python scripts/plot_separation.py --record r04 --window 0
    .venv/bin/python scripts/plot_separation.py --dataset ninfea --record 10 --window 0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from fcac.config import load_config
from fcac.mecg import adaptive_cancel, maternal_template_subtract
from fcac.memd.diagnostics import pan_tompkins_bpm, pan_tompkins_peaks
from fcac.pinn.coords import project_to_single_channel
from fcac.pinn.infer import extract_fecg
from fcac.pinn.model import PINN
from fcac.viz.plots import apply_style, save_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MATERNAL, FETAL, TRUTH = "tab:red", "tab:blue", "tab:green"


def _norm(x: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(x))
    return x / peak if peak > 0 else x


def _mark(ax, peaks: np.ndarray, t: np.ndarray, colour: str, ymax: float = 1.0) -> None:
    for p in peaks:
        ax.axvline(t[p], color=colour, alpha=0.5, lw=0.9, ls="--", ymax=ymax)


def _ticks(ax, peaks: np.ndarray, t: np.ndarray) -> None:
    """True fetal beat positions (scalp reference) as green ticks along the top edge."""
    ax.plot(t[peaks], np.full(len(peaks), 1.02), marker="v", ls="none", color=TRUTH,
            ms=6, clip_on=False, transform=ax.get_xaxis_transform())


def plot_separation(dataset: str, record: str, window: int, checkpoint: str) -> Path:
    cfg = load_config(dataset=dataset)
    interim = np.load(Path(cfg.dataset.interim_dir) / f"{record}.npz", allow_pickle=True)
    band = np.load(Path("data/processed/fetal_band") / dataset / f"{record}.npz",
                   allow_pickle=True)["x"][window]
    fs = float(interim["fs"])
    mixture = interim["windows"][window]
    ref = interim["ref_windows"][window] if interim["ref_windows"].size else None

    # What the pipeline does: adaptive cancellation, then the frozen PINN.
    model = PINN(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    pipeline_in, applied = adaptive_cancel(band, fs, cfg)
    fecg = extract_fecg(model, pipeline_in[None], cfg).fecg[0]

    # For display: the maternal part is always the forced template subtraction.
    cancelled = maternal_template_subtract(band, fs, cfg)
    maternal = band - cancelled

    pc_mix = project_to_single_channel(mixture)
    pc_mat = project_to_single_channel(maternal)
    pc_res = project_to_single_channel(cancelled)
    pc_band = project_to_single_channel(band)

    t = np.arange(mixture.shape[1]) / fs
    p_mix = pan_tompkins_peaks(pc_band, fs)
    p_mat = pan_tompkins_peaks(pc_mat, fs)
    p_fet = pan_tompkins_peaks(fecg, fs)
    p_ref = pan_tompkins_peaks(ref, fs) if ref is not None else None
    bpm = lambda p: 60.0 / (np.mean(np.diff(p)) / fs) if len(p) > 1 else float("nan")  # noqa: E731

    apply_style()
    nrows = 5 if ref is not None else 4
    fig, axes = plt.subplots(nrows, 1, figsize=(13, 2.5 * nrows), sharex=True)

    ax = axes[0]
    ax.plot(t, _norm(pc_mix), color="k", lw=0.8)
    _mark(ax, p_mat, t, MATERNAL)
    if p_ref is not None:
        _ticks(ax, p_ref, t)
    ax.set_title(f"1. Abdominal mixture (PC1) -- dashed red = beats the cancellation locks onto "
                 f"({bpm(p_mat):.0f} bpm); green triangles = TRUE fetal beats (scalp)", pad=14)

    ax = axes[1]
    ax.plot(t, _norm(pc_mat), color=MATERNAL, lw=0.8)
    _mark(ax, p_mat, t, MATERNAL)
    ax.set_title("2. Component removed by template cancellation (median beat template, per-beat "
                 f"scaled) -- {'pipeline applied this' if applied else 'FORCED for display; the pipeline skipped it here'}")

    ax = axes[2]
    ax.plot(t, _norm(pc_res), color="0.3", lw=0.8)
    if p_ref is not None:
        _ticks(ax, p_ref, t)
    ax.set_title("3. After removing it (suppressed MEMD band)", pad=14)

    ax = axes[3]
    ax.plot(t, _norm(fecg), color=FETAL, lw=0.9)
    _mark(ax, p_fet, t, FETAL)
    if p_ref is not None:
        _ticks(ax, p_ref, t)
    ax.set_title(f"4. PINN output -- dashed blue = beats it detects ({bpm(p_fet):.0f} bpm); "
                 f"cancellation {'applied' if applied else 'skipped (window already fetal-rate)'}", pad=14)

    if ref is not None:
        ax = axes[4]
        ax.plot(t, _norm(ref), color=TRUTH, lw=0.9, label="scalp ECG (ground truth)")
        _mark(ax, p_fet, t, FETAL)
        ax.set_title(f"5. Scalp reference ({bpm(p_ref):.0f} bpm) with PINN fetal beats overlaid "
                     "(dashed blue)")
        ax.legend(loc="upper right", fontsize=8)

    for ax in axes:
        ax.set_ylabel("normalised")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Maternal vs fetal separation -- {dataset} / {record} / window {window}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path = save_figure(fig, f"separation_{dataset}_{record}_w{window}")
    plt.close(fig)
    logger.info("wrote %s | maternal %.0f bpm, PINN fetal %.0f bpm, cancellation %s",
                path, bpm(p_mat), bpm(p_fet), "applied" if applied else "skipped")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default="adfecgdb", choices=["adfecgdb", "ninfea", "cinc2013"])
    parser.add_argument("--record", required=True)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--checkpoint", default="data/processed/pinn/pinn.pt")
    args = parser.parse_args()
    plot_separation(args.dataset, args.record, args.window, args.checkpoint)


if __name__ == "__main__":
    main()
