"""tests/test_preprocessing.py

Synthetic-signal tests for src/fcac/preprocessing/ (CLAUDE.md §6: small synthetic
signals, run in seconds, no PhysioNet data needed).

Implements: CLAUDE.md §8 item 9.
"""

import numpy as np

from fcac.config import load_config
from fcac.io.types import Recording
from fcac.preprocessing.channels import kurtosis_select
from fcac.preprocessing.filters import (
    butter_bandpass,
    notch_filter,
    resample_to,
    trim_edges,
)
from fcac.preprocessing.normalise import zscore
from fcac.preprocessing.pipeline import preprocess
from fcac.preprocessing.windowing import segment


def _synthetic_signal(n_channels, n_samples, fs):
    t = np.arange(n_samples) / fs
    base = np.sin(2 * np.pi * 5 * t)  # 5 Hz "cardiac-like" component
    drift = 0.5 * np.sin(2 * np.pi * 0.1 * t)  # slow baseline drift
    return np.stack([base + drift + 0.01 * i for i in range(n_channels)]).astype(np.float32)


def test_butter_bandpass_removes_low_frequency_drift():
    fs = 1000.0
    x = _synthetic_signal(1, 5000, fs)[0]
    filtered = butter_bandpass(x, fs, low_hz=1.0, high_hz=100.0)
    assert filtered.shape == x.shape
    # Bandpassing should reduce the very-low-frequency drift's contribution to variance.
    assert np.var(filtered) < np.var(x)


def test_notch_filter_shape_preserved():
    fs = 1000.0
    x = _synthetic_signal(1, 5000, fs)[0]
    filtered = notch_filter(x, fs, freq_hz=50.0, quality_factor=30.0)
    assert filtered.shape == x.shape


def test_resample_to_changes_length_by_expected_ratio():
    x = _synthetic_signal(4, 2048, 2048.0)
    resampled = resample_to(x, fs_in=2048.0, fs_out=1000.0)
    assert resampled.shape[0] == 4
    assert abs(resampled.shape[1] - 1000) < 5


def test_trim_edges_drops_expected_samples():
    x = np.zeros((4, 3000), dtype=np.float32)
    trimmed = trim_edges(x, fs=1000.0, trim_seconds=1.0)
    assert trimmed.shape[1] == 1000


def test_kurtosis_select_picks_high_kurtosis_channels():
    rng = np.random.default_rng(0)
    n_channels, n_samples = 6, 2000
    x = rng.normal(size=(n_channels, n_samples)).astype(np.float32)
    # Inject an impulse train (high kurtosis) into channel 2.
    x[2, ::100] += 20.0
    selected, indices, _kurt = kurtosis_select(x, n_select=1)
    assert indices[0] == 2
    assert selected.shape == (1, n_samples)


def test_kurtosis_select_never_picks_a_flat_channel_over_informative_ones():
    """Real-data bug, found running the real pipeline on NInFEA (recording 34's
    uni_abd9 electrode is dead/constant): scipy.stats.kurtosis on a flat channel
    returns NaN, and NaN sorts to the FRONT after np.argsort(...)[::-1] -- so a dead
    electrode was being selected as the #1 channel, ahead of every genuinely
    informative one."""
    rng = np.random.default_rng(0)
    n_channels, n_samples = 6, 2000
    x = rng.normal(size=(n_channels, n_samples)).astype(np.float32)
    x[2, ::100] += 20.0  # channel 2: genuinely informative (impulse train, high kurtosis)
    x[4, :] = 7.0  # channel 4: flat/dead electrode

    selected, indices, _kurt = kurtosis_select(x, n_select=2)
    assert 4 not in indices
    assert 2 in indices


def test_zscore_per_channel():
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], dtype=np.float32)
    z = zscore(x)
    assert np.allclose(z.mean(axis=-1), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=-1), 1.0, atol=1e-5)


def test_segment_non_overlapping_windows():
    x = np.arange(20).reshape(1, 20).astype(np.float32)
    windows = segment(x, win=5, stride=5)
    assert windows.shape == (4, 1, 5)


def test_full_pipeline_adfecgdb_shapes():
    cfg = load_config(dataset="adfecgdb")
    fs = cfg.dataset.native_fs_hz
    n_samples = int(30 * fs)
    signal = _synthetic_signal(cfg.preprocessing.n_channels, n_samples, fs)
    fetal_ref = _synthetic_signal(1, n_samples, fs)[0]

    recording = Recording(
        signal=signal, fs_raw=float(fs), fetal_ref=fetal_ref, ga_days=280.0, subject_id="r01"
    )
    result = preprocess(recording, cfg)

    win = cfg.preprocessing.window.size_samples
    assert result.windows.shape[1:] == (cfg.preprocessing.n_channels, win)
    assert result.ref_windows.shape[1:] == (win,)
    assert result.fs == cfg.preprocessing.target_fs_hz


def test_full_pipeline_ninfea_selects_four_channels():
    cfg = load_config(dataset="ninfea")
    fs = cfg.dataset.native_fs_hz
    n_samples = int(30 * fs)
    n_raw_channels = 24
    signal = _synthetic_signal(n_raw_channels, n_samples, fs)

    recording = Recording(
        signal=signal, fs_raw=float(fs), fetal_ref=None, ga_days=175.0, subject_id="1"
    )
    result = preprocess(recording, cfg)

    win = cfg.preprocessing.window.size_samples
    assert result.windows.shape[1] == cfg.preprocessing.n_channels
    assert result.windows.shape[2] == win
    assert result.ref_windows is None
