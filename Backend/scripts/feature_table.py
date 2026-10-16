#!/usr/bin/env python3
"""scripts/feature_table.py

Prints the 12 interpretable cardiac biomarkers (CLAUDE.md section 5 FeatureVector) for
an extracted fetal ECG, as a table.

Sources, mutually exclusive:
  --signal  <file>            your own 4-channel abdominal ECG -- runs the whole chain
                              (preprocess -> MEMD -> PINN) then extracts features
  --dataset <name> [--record] a record already processed by scripts 01/02/04

Feature values are per-window; the table shows each window plus the mean/SD across them.
NaN is a real result, not a failure: a 5 s window holds only ~8-12 RR intervals, and the
entropy/DFA/spectral features need more than that. The summary at the bottom says how
many windows each feature was computable on.

Usage:
    python scripts/feature_table.py --dataset adfecgdb --record r08
    python scripts/feature_table.py --signal my_ecg.csv --fs 1000
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fcac.config import load_config
from fcac.features.extract import (
    FEATURE_NAMES,
    extract_features_batch,
    extract_features_record,
    pooled_rr_intervals,
)
from fcac.features.fiducials import rr_intervals
from fcac.memd.diagnostics import pan_tompkins_bpm
from fcac.quality import assess_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

UNITS = {
    "qtc_sigma": "ms",
    "p_dispersion": "ms",
    "qrs_mean": "ms",
    "qrs_var": "ms^2",
    "qrs_asymmetry": "ratio",
    "prsa": "ms",
    "mse": "nats",
    "apen": "nats",
    "sampen": "nats",
    "dfa_alpha": "-",
    "lf_hf_ratio": "-",
    "symbolic_dynamics": "% 0V",
}
KIND = {name: ("morphological" if i < 5 else "dynamic") for i, name in enumerate(FEATURE_NAMES)}


def fecg_from_dataset(dataset: str, record: str | None):
    cfg = load_config(dataset=dataset)
    directory = Path("data/processed/fecg") / dataset
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(
            f"No extracted fECG in {directory}. Run scripts 01, 02 and 04 first."
        )
    path = next((p for p in paths if p.stem == record), None) if record else paths[0]
    if path is None:
        raise FileNotFoundError(
            f"Record '{record}' not found (have: {', '.join(p.stem for p in paths)})"
        )
    data = np.load(path, allow_pickle=True)
    return data["fecg"], float(data["fs"]), f"{dataset}/{path.stem}", cfg


def fecg_from_signal(path: Path, fs_in: float, checkpoint: str):
    """Run the full chain on a custom recording, reusing scripts/plot_extraction.py."""
    import torch

    from fcac.memd.select import build_fetal_band_windows
    from fcac.memd.sift import memd
    from fcac.pinn.infer import extract_fecg
    from fcac.pinn.model import PINN
    from fcac.preprocessing.filters import butter_bandpass, notch_filter, resample_to, trim_edges
    from fcac.preprocessing.normalise import zscore
    from fcac.preprocessing.windowing import segment

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_extraction import load_custom_signal

    cfg = load_config(dataset="adfecgdb")
    raw = load_custom_signal(path, fs_in, cfg)

    preset = cfg.preprocessing.bandpass_presets[cfg.preprocessing.bandpass_preset]
    x = butter_bandpass(raw, fs_in, preset.low_hz, preset.high_hz, cfg.preprocessing.bandpass_order)
    if cfg.preprocessing.notch_enabled:
        x = notch_filter(x, fs_in, cfg.preprocessing.notch.freq_hz,
                         cfg.preprocessing.notch.quality_factor)
    x = trim_edges(x, fs_in, cfg.preprocessing.edge_trim_seconds)

    target_fs = float(cfg.preprocessing.target_fs_hz)
    if fs_in != target_fs:
        x = resample_to(x, fs_in, target_fs)
    x = zscore(x)

    windows = segment(x, cfg.preprocessing.window.size_samples,
                      cfg.preprocessing.window.stride_samples)
    if windows.shape[0] == 0:
        raise ValueError("Signal too short for even one window after edge trimming.")

    logger.info("Running MEMD on %d windows (a few seconds each)...", windows.shape[0])
    imfs = [memd(windows[w], n_directions=cfg.memd.directions,
                 stopping_tolerance=cfg.memd.stopping_tolerance,
                 max_sift_iterations=cfg.memd.max_iterations,
                 abstain_fraction=cfg.memd.abstain_fraction)
            for w in range(windows.shape[0])]
    fetal_band, _, _ = build_fetal_band_windows(imfs, target_fs, cfg)

    model = PINN(cfg)
    model.load_state_dict(__import__("torch").load(checkpoint, map_location="cpu",
                                                   weights_only=True))
    model.eval()
    return extract_fecg(model, fetal_band, cfg).fecg, target_fs, path.name, cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", choices=["adfecgdb", "ninfea", "cinc2013"])
    source.add_argument("--signal", type=Path)
    parser.add_argument("--record", default=None)
    parser.add_argument("--fs", type=float, default=1000.0, help="--signal sample rate")
    parser.add_argument("--checkpoint", default="data/processed/pinn/pinn.pt")
    parser.add_argument("--max-windows", type=int, default=8, help="columns to print")
    parser.add_argument("--no-gate", action="store_true",
                        help="compute features even on windows that fail the quality "
                             "gate (they will be maternal-locked and meaningless)")
    parser.add_argument("--level", choices=["window", "record"], default="record",
                        help="record (default): dynamic features on the RR series pooled "
                             "across all windows -- the only way they are meaningful. "
                             "window: per-window values, mostly NaN for the dynamic half.")
    args = parser.parse_args()

    if args.signal is not None:
        fecg, fs, label, cfg = fecg_from_signal(args.signal, args.fs, args.checkpoint)
    else:
        fecg, fs, label, cfg = fecg_from_dataset(args.dataset, args.record)

    # Quality gate FIRST: a maternal-locked window still yields 12 confident-looking
    # numbers, and nothing downstream can tell the difference (src/fcac/quality.py).
    quality = assess_record(fecg, fs, cfg)
    if not args.no_gate:
        if quality["n_acceptable"] == 0:
            print()
            print(
                f"{label}: NO windows pass the quality gate "
                f"(median {quality['median_heart_rate_bpm']:.0f} bpm). "
                "Extraction failed for this record -- any features would be "
                "computed from a maternal-locked signal. "
                "Re-run with --no-gate to see them anyway."
            )
            print()
            return
        fecg = fecg[quality["acceptable_mask"]]

    logger.info("Extracting 12 features from %d windows of %s", fecg.shape[0], label)
    features = extract_features_batch(fecg, fs, cfg)  # (W, 12)
    record_vector = extract_features_record(fecg, fs, cfg) if args.level == "record" else None
    pooled = pooled_rr_intervals(fecg, fs, cfg)

    n_show = min(args.max_windows, features.shape[0])
    beats = [len(rr_intervals(fecg[w], fs, cfg)) + 1 for w in range(features.shape[0])]
    rates = [pan_tompkins_bpm(fecg[w], fs) for w in range(features.shape[0])]

    print("\n" + "=" * 100)
    gate_note = ("all windows, gate DISABLED" if args.no_gate
                 else f"{quality['n_acceptable']}/{quality['n_windows']} windows passed the quality gate")
    print(f"12 CARDIAC BIOMARKERS -- {label}   ({features.shape[0]} windows @ {fs:.0f} Hz)")
    print(f"{gate_note}; median rate {quality['median_heart_rate_bpm']:.0f} bpm"
          + ("" if quality["usable"] else "   <- RECORD FLAGGED UNUSABLE"))
    print("=" * 100)
    print(f"{'#':<3}{'feature':<20}{'unit':<8}" + "".join(f"{'w'+str(w):>9}" for w in range(n_show))
          + f"{'mean':>10}{'SD':>9}{'n_ok':>6}")
    print("-" * 100)
    for i, name in enumerate(FEATURE_NAMES):
        column = features[:, i]
        finite = column[np.isfinite(column)]
        cells = "".join(
            (f"{column[w]:>9.2f}" if np.isfinite(column[w]) else f"{'nan':>9}")
            for w in range(n_show)
        )
        mean = f"{finite.mean():>10.2f}" if finite.size else f"{'--':>10}"
        sd = f"{finite.std():>9.2f}" if finite.size else f"{'--':>9}"
        print(f"{i:<3}{name:<20}{UNITS[name]:<8}{cells}{mean}{sd}{finite.size:>6}")
        if i == 4:
            print(f"{'':3}{'--- dynamic ---':<20}")
    print("-" * 100)
    if record_vector is not None:
        print(f"{'':3}{'RECORD-LEVEL':<20}{'':8}"
              + "".join(f"{'':>9}" for _ in range(n_show))
              + f"{'value':>10}{'':9}{'':>6}")
        for i, name in enumerate(FEATURE_NAMES):
            value = record_vector[i]
            cell = f"{value:>10.3f}" if np.isfinite(value) else f"{'nan':>10}"
            print(f"{i:<3}{name:<20}{UNITS[name]:<8}"
                  + "".join(f"{'':>9}" for _ in range(n_show)) + cell)
        n_ok = int(np.isfinite(record_vector).sum())
        print(f"{'':3}{'pooled RR intervals':<20}{'n':<8}"
              + "".join(f"{'':>9}" for _ in range(n_show)) + f"{pooled.size:>10}")
        if pooled.size:
            print(f"{'':3}{'median rate':<20}{'bpm':<8}"
                  + "".join(f"{'':>9}" for _ in range(n_show))
                  + f"{60000.0 / float(np.median(pooled)):>10.1f}")
        print(f"{'':3}record-level features computable: {n_ok}/12")
        print("-" * 100)
    print(f"{'':3}{'beats in window':<20}{'n':<8}" + "".join(f"{beats[w]:>9}" for w in range(n_show)))
    print(f"{'':3}{'heart rate':<20}{'bpm':<8}"
          + "".join((f"{rates[w]:>9.1f}" if np.isfinite(rates[w]) else f"{'nan':>9}")
                    for w in range(n_show)))
    print("=" * 100)
    n_full = int(np.isfinite(features).all(axis=1).sum())
    print(f"windows with all 12 features computable: {n_full}/{features.shape[0]}")
    print("NaN = too few beats in that window for that feature, not an error. Morphological")
    print("features need >=3 complete beats; DFA/entropy/spectral need 6-8 RR intervals.")
    print("Parameter choices are documented defaults, NOT supervisor-confirmed (plan.md item 4).")
    print("=" * 100 + "\n")

    out = Path("reports") / f"features_{label.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"source": label, "fs": fs, "feature_names": FEATURE_NAMES,
                   "units": UNITS, "kind": KIND,
                   "values": [[None if not np.isfinite(v) else float(v) for v in row]
                              for row in features],
                   "timestamp": datetime.now(timezone.utc).isoformat()}, fh, indent=2)
    logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
