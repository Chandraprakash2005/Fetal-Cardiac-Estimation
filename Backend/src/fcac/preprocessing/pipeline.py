"""src/fcac/preprocessing/pipeline.py

Orchestrates the full preprocessing chain in the LOCKED order (CLAUDE.md §2
"Resampling order"): filter at native rate -> resample to 1000 Hz -> [kurtosis-select
top-4 channels, if more than n_channels remain] -> per-channel z-score -> window into
(W, 4, 5000).

Implements: CLAUDE.md §5 Recording -> PreprocessedRecording.
"""

from __future__ import annotations

from typing import Tuple

from fcac.config import Config
from fcac.io.types import PreprocessedRecording, Recording
from fcac.preprocessing.channels import kurtosis_select
from fcac.preprocessing.filters import butter_bandpass, notch_filter, resample_to, trim_edges
from fcac.preprocessing.normalise import zscore
from fcac.preprocessing.windowing import segment


def _bandpass_band(cfg: Config) -> Tuple[float, float]:
    preset_name = cfg.preprocessing.bandpass_preset
    preset = cfg.preprocessing.bandpass_presets[preset_name]
    return preset.low_hz, preset.high_hz


def _filter_stage(x, fs: float, cfg: Config):
    low_hz, high_hz = _bandpass_band(cfg)
    x = butter_bandpass(x, fs, low_hz, high_hz, cfg.preprocessing.bandpass_order)
    if cfg.preprocessing.notch_enabled:
        x = notch_filter(
            x, fs, cfg.preprocessing.notch.freq_hz, cfg.preprocessing.notch.quality_factor
        )
    return trim_edges(x, fs, cfg.preprocessing.edge_trim_seconds)


def preprocess(recording: Recording, cfg: Config) -> PreprocessedRecording:
    """Run the full preprocessing chain on one Recording.

    `cfg` must already be merged with this recording's dataset config
    (fcac.config.load_config(dataset=...)) so cfg.preprocessing.notch_enabled and
    cfg.dataset.* reflect the right dataset (CLAUDE.md §2: notch is NInFEA-only).
    """
    target_fs = cfg.preprocessing.target_fs_hz

    # 1. Filter at native rate (+ notch if this dataset needs it, + edge trim).
    signal = _filter_stage(recording.signal, recording.fs_raw, cfg)

    # The scalp reference (ADFECGDB only) is a full-length signal aligned with
    # recording.signal and gets the same filtering. NInFEA's fetal_ref is a sparse
    # Doppler beat-timing array, not a signal, and is left untouched here.
    ref = recording.fetal_ref
    is_signal_shaped = (
        ref is not None and ref.ndim == 1 and ref.shape[-1] == recording.signal.shape[-1]
    )
    if is_signal_shaped:
        ref = _filter_stage(ref, recording.fs_raw, cfg)

    # 2. Resample to target fs (no-op for ADFECGDB, which is already 1000 Hz).
    if recording.fs_raw != target_fs:
        signal = resample_to(signal, recording.fs_raw, target_fs)
        if is_signal_shaped:
            ref = resample_to(ref, recording.fs_raw, target_fs)

    # 3. Kurtosis-select top-N channels if more than n_channels remain (NInFEA: 24 -> 4;
    #    ADFECGDB is already exactly n_channels wide and skips this).
    n_channels = cfg.preprocessing.n_channels
    if signal.shape[0] > n_channels:
        signal, _, _ = kurtosis_select(signal, cfg.preprocessing.kurtosis_selection.n_select)

    # 4. Per-channel z-score.
    signal = zscore(signal)
    if is_signal_shaped:
        ref = zscore(ref)

    # 5. Window.
    win = cfg.preprocessing.window.size_samples
    stride = cfg.preprocessing.window.stride_samples
    windows = segment(signal, win, stride)
    ref_windows = segment(ref, win, stride) if is_signal_shaped else None

    return PreprocessedRecording(
        windows=windows,
        fs=float(target_fs),
        ref_windows=ref_windows,
        ga_days=recording.ga_days,
        subject_id=recording.subject_id,
    )
