#!/usr/bin/env python3
"""scripts/plot_extraction.py

Shows the result of the extraction chain (preprocess -> MEMD -> PINN) as a figure in
reports/figures/, plus a short numeric summary on stdout.

Not a numbered pipeline stage -- this is the "look at what came out" utility. CLAUDE.md
section 6 requires verifying preprocessing and MEMD output visually before any
downstream stage consumes them, and this is that check.

Two modes:

  --demo      Runs the whole chain on a synthetic maternal+fetal mixture, start to
              finish, with no PhysioNet data and no trained checkpoint required. Use it
              to confirm the plumbing works and to see what the output looks like.
              The "fetal ECG" here is synthetic -- it validates the code path, not the
              science.

  --dataset   Reads the real cached artefacts written by scripts 01/02/04:
              data/interim/<dataset>/, data/processed/fetal_band/<dataset>/ and
              data/processed/fecg/<dataset>/.

Usage:
    python scripts/plot_extraction.py --demo
    python scripts/plot_extraction.py --dataset adfecgdb --record r01 --window 0
    python scripts/plot_extraction.py --signal my_ecg.csv --fs 1000

  --signal    Runs the full chain on YOUR OWN recording: a 4-channel abdominal ECG in
              .npz / .npy / .csv, at any sampling rate (--fs). It is preprocessed,
              MEMD-decomposed and passed through the trained PINN exactly as a dataset
              record would be.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from fcac.config import Config, load_config
from fcac.memd.diagnostics import pan_tompkins_bpm
from fcac.memd.select import build_fetal_band_windows
from fcac.memd.sift import memd
from fcac.viz.plots import plot_extraction_summary, save_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _synthetic_window(cfg: Config, n_samples: int = 5000, fs: float = 1000.0):
    """A maternal + fetal mixture with the right spatial character.

    Maternal is far-field (near-uniform gains across the electrode array); fetal is
    near-field (sharply varying gains). That contrast is exactly what the PINN's
    harmonic constraint is meant to exploit -- see src/fcac/pinn/coords.py.
    """
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(0)

    def beats(bpm, width, amplitude, phase=0.0):
        s = np.zeros(n_samples)
        for centre in np.arange(phase, n_samples / fs, 60.0 / bpm):
            u = (t - centre) / width
            s += amplitude * np.exp(-0.5 * u**2) * (1 - u**2)
        return s

    maternal = beats(80, 0.012, 1.0)
    fetal = beats(140, 0.006, 0.15, phase=0.13)

    gains_maternal = np.array([[1.0], [0.95], [1.05], [1.0]])
    gains_fetal = np.array([[0.2], [1.0], [0.15], [0.5]])
    x = gains_maternal @ maternal[None, :] + gains_fetal @ fetal[None, :]
    x += 0.02 * rng.standard_normal((4, n_samples))
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x.astype(np.float32), fetal.astype(np.float32), fs


def run_demo(epochs: int) -> None:
    import torch

    from fcac.pinn.infer import extract_fecg
    from fcac.pinn.train import train_pinn_on_windows

    cfg = load_config()
    data = cfg.to_dict()
    data["pinn"]["epochs"] = epochs
    cfg = Config(data)

    abdominal, true_fetal, fs = _synthetic_window(cfg)

    logger.info("Running MEMD on the synthetic window (this takes a few seconds)...")
    imfs = memd(
        abdominal,
        n_directions=cfg.memd.directions,
        stopping_tolerance=cfg.memd.stopping_tolerance,
        max_sift_iterations=cfg.memd.max_iterations,
        abstain_fraction=cfg.memd.abstain_fraction,
    )
    fetal_band, _, selected = build_fetal_band_windows([imfs], fs, cfg)
    logger.info("MEMD produced K=%d IMFs; selected %s", imfs.shape[0], selected[0])

    logger.info("Training the PINN for %d epochs on this one window...", epochs)
    model = train_pinn_on_windows(cfg, fetal_band)
    fecg = extract_fecg(model, fetal_band, cfg).fecg[0]

    _report(abdominal, fetal_band[0], fecg, fs, reference=true_fetal)
    fig = plot_extraction_summary(
        abdominal,
        fetal_band[0],
        fecg,
        fs,
        title="DEMO -- synthetic maternal + fetal mixture (not real data)",
        reference=true_fetal,
    )
    path = save_figure(fig, "extraction_demo")
    logger.info("Wrote %s", path)


def run_dataset(dataset: str, record: str | None, window: int) -> None:
    cfg = load_config(dataset=dataset)

    interim = _pick(Path(cfg.dataset.interim_dir), record, "scripts/01_preprocess.py")
    band_path = _pick(
        Path("data/processed/fetal_band") / dataset, interim.stem, "scripts/02_memd.py"
    )
    fecg_path = _pick(
        Path("data/processed/fecg") / dataset, interim.stem, "scripts/04_extract_fecg.py"
    )

    interim_data = np.load(interim, allow_pickle=True)
    fs = float(interim_data["fs"])
    abdominal = interim_data["windows"][window]
    fetal_band = np.load(band_path, allow_pickle=True)["x"][window]
    fecg = np.load(fecg_path, allow_pickle=True)["fecg"][window]

    ref_windows = interim_data["ref_windows"]
    reference = ref_windows[window] if ref_windows.size > 0 else None

    _report(abdominal, fetal_band, fecg, fs, reference=reference)
    fig = plot_extraction_summary(
        abdominal,
        fetal_band,
        fecg,
        fs,
        title=f"{dataset} / {interim.stem} / window {window}",
        reference=reference,
    )
    path = save_figure(fig, f"extraction_{dataset}_{interim.stem}_w{window}")
    logger.info("Wrote %s", path)


def _pick(directory: Path, record: str | None, produced_by: str) -> Path:
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"Nothing in {directory}. Run {produced_by} first.")
    if record is None:
        return paths[0]
    for path in paths:
        if path.stem == record:
            return path
    raise FileNotFoundError(
        f"Record '{record}' not in {directory} (have: {', '.join(p.stem for p in paths)})"
    )


def _report(abdominal, fetal_band, fecg, fs, reference=None) -> None:
    """Numbers worth eyeballing next to the figure."""
    print("\n" + "=" * 68)
    print("EXTRACTION SUMMARY")
    print("=" * 68)
    print(f"  abdominal window   : {abdominal.shape}  @ {fs:.0f} Hz")
    print(f"  MEMD fetal band    : {fetal_band.shape}")
    print(f"  PINN fECG          : {fecg.shape}")

    # PC1, not channel 0 -- ch0 is the worst abdominal lead on every ADFECGDB record
    # (MEMD per-channel F1 0.22 / 0.92 / 0.91 / 0.96), so quoting it here would
    # understate the MEMD stage and contradict scripts/accuracy_table.py.
    from fcac.pinn.coords import project_to_single_channel

    bpm_band = pan_tompkins_bpm(project_to_single_channel(fetal_band), fs)
    bpm_fecg = pan_tompkins_bpm(fecg, fs)
    print("\n  Heart rate (Pan-Tompkins sanity check; fetal is ~110-160 bpm):")
    print(f"    MEMD fetal band  : {bpm_band:6.1f} bpm")
    print(f"    PINN fECG        : {bpm_fecg:6.1f} bpm")

    if reference is not None:
        def corr(a, b):
            a = a - a.mean()
            b = b - b.mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            return float(a @ b / denominator) if denominator else float("nan")

        print("\n  Correlation with the reference channel:")
        print(f"    raw abdominal    : {corr(project_to_single_channel(abdominal), reference):+.3f}")
        print(f"    MEMD fetal band  : {corr(project_to_single_channel(fetal_band), reference):+.3f}")
        print(f"    PINN fECG        : {corr(fecg, reference):+.3f}")
        print(
            "\n  (Correlation against a scalp reference is a rough guide only -- the"
            "\n   real extraction metrics are SNR / QRS F1 / Bland-Altman R-R in"
            "\n   src/fcac/evaluation/extraction.py.)"
        )
    print("=" * 68 + "\n")


def load_custom_signal(path: Path, fs: float, cfg: Config) -> np.ndarray:
    """Read a user-supplied multi-channel abdominal ECG as (C, N) float32.

    Accepts .npz (first array, or the key "signal"), .npy, or a delimited text file
    (.csv/.txt/.tsv). Either orientation works -- if the array looks like (N, C) with
    more rows than columns it is transposed, since real recordings always have far more
    samples than channels.
    """
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        key = "signal" if "signal" in data else data.files[0]
        array = data[key]
    elif suffix == ".npy":
        array = np.load(path)
    else:
        delimiter = "	" if suffix == ".tsv" else ","
        array = np.loadtxt(path, delimiter=delimiter)

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        raise ValueError(
            f"{path} holds a single channel; MEMD needs the multi-channel abdominal "
            "recording (4 channels) to separate anything."
        )
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D (C, N) array, got shape {array.shape}")
    if array.shape[0] > array.shape[1]:
        array = array.T  # (N, C) -> (C, N)

    n_channels = cfg.preprocessing.n_channels
    if array.shape[0] != n_channels:
        raise ValueError(
            f"{path}: got {array.shape[0]} channels, pipeline expects {n_channels}. "
            "Supply the 4 abdominal channels (or change preprocessing.n_channels)."
        )
    return array


def run_custom(path: Path, fs: float, checkpoint: str, window: int) -> None:
    """Full chain on a custom signal: preprocess -> MEMD -> PINN -> fetal ECG."""
    import torch

    from fcac.pinn.infer import extract_fecg
    from fcac.pinn.model import PINN
    from fcac.preprocessing.filters import butter_bandpass, notch_filter, resample_to, trim_edges
    from fcac.preprocessing.normalise import zscore
    from fcac.preprocessing.windowing import segment

    cfg = load_config(dataset="adfecgdb")  # the cohort the checkpoint was trained on
    raw = load_custom_signal(path, fs, cfg)
    logger.info("Loaded %s: %d channels x %d samples @ %.0f Hz", path, *raw.shape, fs)

    # Same preprocessing chain as scripts/01, applied at the signal's own rate.
    preset = cfg.preprocessing.bandpass_presets[cfg.preprocessing.bandpass_preset]
    x = butter_bandpass(raw, fs, preset.low_hz, preset.high_hz, cfg.preprocessing.bandpass_order)
    if cfg.preprocessing.notch_enabled:
        x = notch_filter(x, fs, cfg.preprocessing.notch.freq_hz,
                         cfg.preprocessing.notch.quality_factor)
    x = trim_edges(x, fs, cfg.preprocessing.edge_trim_seconds)

    target_fs = cfg.preprocessing.target_fs_hz
    if fs != target_fs:
        logger.info("Resampling %.0f Hz -> %d Hz", fs, target_fs)
        x = resample_to(x, fs, target_fs)
    x = zscore(x)

    size = cfg.preprocessing.window.size_samples
    windows = segment(x, size, cfg.preprocessing.window.stride_samples)
    if windows.shape[0] == 0:
        raise ValueError(
            f"Signal is too short: need at least {size} samples "
            f"({size / target_fs:.0f} s) after edge trimming, got {x.shape[1]}."
        )
    if window >= windows.shape[0]:
        raise ValueError(f"--window {window} out of range (got {windows.shape[0]} windows)")
    logger.info("Preprocessed into %d windows; running MEMD on window %d",
                windows.shape[0], window)

    imfs = memd(windows[window],
                n_directions=cfg.memd.directions,
                stopping_tolerance=cfg.memd.stopping_tolerance,
                max_sift_iterations=cfg.memd.max_iterations,
                abstain_fraction=cfg.memd.abstain_fraction)
    fetal_band, _, selected = build_fetal_band_windows([imfs], float(target_fs), cfg)
    logger.info("MEMD: K=%d IMFs, selected %s", imfs.shape[0], selected[0])

    model = PINN(cfg)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    fecg = extract_fecg(model, fetal_band, cfg).fecg[0]

    _report(windows[window], fetal_band[0], fecg, float(target_fs))
    fig = plot_extraction_summary(windows[window], fetal_band[0], fecg, float(target_fs),
                                  title=f"custom signal: {path.name} (window {window})")
    logger.info("Wrote %s", save_figure(fig, f"extraction_custom_{path.stem}_w{window}"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--demo", action="store_true", help="run on synthetic data")
    mode.add_argument("--dataset", choices=["adfecgdb", "ninfea", "cinc2013"])
    mode.add_argument("--signal", type=Path,
                      help="custom multi-channel abdominal ECG (.npz/.npy/.csv)")
    parser.add_argument("--fs", type=float, default=1000.0,
                        help="sampling rate of --signal in Hz (default 1000)")
    parser.add_argument("--checkpoint", default="data/processed/pinn/pinn.pt")
    parser.add_argument("--record", default=None, help="record id (default: first found)")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30, help="--demo only")
    args = parser.parse_args()

    if args.demo:
        run_demo(args.epochs)
    elif args.signal is not None:
        run_custom(args.signal, args.fs, args.checkpoint, args.window)
    else:
        run_dataset(args.dataset, args.record, args.window)


if __name__ == "__main__":
    main()
