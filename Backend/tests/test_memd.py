"""tests/test_memd.py

Synthetic-signal tests for the MEMD package: the vendored decomposition (sift.py), the
fetal-band scoring/selection (select.py), and the diagnostics.

Everything runs on short summed-sinusoid / impulse-train signals so the suite stays in
seconds and needs no PhysioNet data (CLAUDE.md section 6 "Tests").

Implements: CLAUDE.md section 8 item 9.
"""

import numpy as np

from fcac.config import load_config
from fcac.memd.decompose import _run_memd
from fcac.memd.diagnostics import pan_tompkins_bpm
from fcac.memd.select import (
    build_fetal_band_windows,
    imf_in_band_power,
    imf_scores,
    reconstruct_fetal_band,
    select_fetal_band,
)
from fcac.memd.sift import direction_vectors, local_maxima, memd

FS = 1000.0


def _multivariate_signal(n_samples=1500, n_channels=4, seed=0):
    """4 channels mixing the same 5 / 40 / 120 Hz oscillations with different gains."""
    t = np.arange(n_samples) / FS
    rng = np.random.default_rng(seed)
    components = np.vstack([np.sin(2 * np.pi * f * t) for f in (5.0, 40.0, 120.0)])
    mixing = rng.uniform(0.5, 1.5, (n_channels, 3))
    return (mixing @ components).astype(np.float64)


# --------------------------------------------------------------------------- sift.py


def test_direction_vectors_are_unit_norm_and_spread():
    vectors = direction_vectors(64, 4)
    assert vectors.shape == (64, 4)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    # Quasi-uniform sampling should very nearly cancel out when averaged.
    assert np.linalg.norm(vectors.mean(axis=0)) < 0.15


def test_local_maxima_handles_plateaus():
    p = np.array([0.0, 1, 2, 2, 2, 1, 0, 1, 3, 1, 0])
    peaks = local_maxima(p)
    assert list(peaks) == [3, 8]  # plateau midpoint, then the sharp peak


def test_local_maxima_of_constant_signal_is_empty():
    assert local_maxima(np.ones(50)).size == 0


def test_memd_reconstructs_the_input_exactly():
    x = _multivariate_signal()
    imfs = memd(x, n_directions=16, max_sift_iterations=20)
    assert imfs.ndim == 3 and imfs.shape[1:] == x.shape
    # IMFs + residue must sum back to the original signal.
    assert np.abs(imfs.sum(axis=0) - x).max() < 1e-4


def test_memd_orders_imfs_high_to_low_frequency():
    from scipy import signal as sp_signal

    x = _multivariate_signal()
    imfs = memd(x, n_directions=16, max_sift_iterations=20)

    peak_freqs = []
    for k in range(imfs.shape[0]):
        freqs, psd = sp_signal.welch(imfs[k, 0], fs=FS, nperseg=512)
        peak_freqs.append(freqs[psd.argmax()])

    # EMD extracts fastest oscillations first; allow ties but no net upward trend.
    assert peak_freqs[0] > peak_freqs[-1]


def test_memd_imf_count_is_data_determined():
    """A signal with more oscillatory scales yields more IMFs than a simple one."""
    t = np.arange(1500) / FS
    simple = np.vstack([np.sin(2 * np.pi * 5 * t) * g for g in (1.0, 0.9, 1.1, 1.0)])
    rich = _multivariate_signal()

    k_simple = memd(simple, n_directions=8, max_sift_iterations=10).shape[0]
    k_rich = memd(rich, n_directions=8, max_sift_iterations=10).shape[0]
    assert k_rich > k_simple


def test_run_memd_uses_config():
    cfg = load_config()
    imfs = _run_memd(_multivariate_signal(n_samples=800), cfg)
    assert imfs.shape[1] == 4
    assert imfs.dtype == np.float32


# ------------------------------------------------------------------------- select.py


def test_imf_in_band_power_prefers_in_band_signal():
    t = np.arange(5000) / FS
    in_band = np.sin(2 * np.pi * 40 * t)  # inside the 20-70 Hz fetal band
    out_of_band = np.sin(2 * np.pi * 5 * t)

    assert imf_in_band_power(in_band, FS, (20.0, 70.0)) > imf_in_band_power(
        out_of_band, FS, (20.0, 70.0)
    )


def test_imf_scores_and_select_fetal_band():
    cfg = load_config()
    t = np.arange(5000) / FS
    imfs = np.stack(
        [
            np.sin(2 * np.pi * 40 * t),  # in-band -- should score high
            np.sin(2 * np.pi * 2 * t),  # out-of-band -- should score low
        ]
    ).astype(np.float32)

    scores = imf_scores(imfs, FS, cfg)
    assert scores[0] > scores[1]

    selected = select_fetal_band(scores, cfg, threshold=scores.mean())
    assert 0 in selected


def test_reconstruct_fetal_band_sums_selected_imfs_only():
    imfs = np.arange(3 * 4 * 10, dtype=np.float32).reshape(3, 4, 10)
    got = reconstruct_fetal_band(imfs, [0, 2])
    assert np.allclose(got, imfs[0] + imfs[2])


def test_reconstruct_fetal_band_falls_back_to_full_sum_when_nothing_selected():
    imfs = np.arange(3 * 4 * 10, dtype=np.float32).reshape(3, 4, 10)
    assert np.allclose(reconstruct_fetal_band(imfs, []), imfs.sum(axis=0))


def test_build_fetal_band_windows_pads_ragged_scores():
    cfg = load_config()
    # Two windows with different IMF counts -- the scores matrix must still be rectangular.
    imfs_per_window = [
        np.random.default_rng(0).standard_normal((5, 4, 500)).astype(np.float32),
        np.random.default_rng(1).standard_normal((8, 4, 500)).astype(np.float32),
    ]
    x, scores, selected = build_fetal_band_windows(imfs_per_window, FS, cfg)

    assert x.shape == (2, 4, 500)
    assert scores.shape == (2, 8)
    assert np.all(scores[0, 5:] == 0.0)  # zero-padded, not garbage
    assert len(selected) == 2


# ---------------------------------------------------------------------- diagnostics.py


def test_pan_tompkins_bpm_recovers_a_known_rate():
    bpm_true = 140.0
    n = 10000
    t = np.arange(n) / FS
    x = np.zeros(n)
    for centre in np.arange(0.2, n / FS, 60.0 / bpm_true):
        x += np.exp(-0.5 * ((t - centre) / 0.006) ** 2)

    assert abs(pan_tompkins_bpm(x, FS) - bpm_true) < 5.0


def test_pan_tompkins_bpm_is_nan_without_beats():
    assert np.isnan(pan_tompkins_bpm(np.zeros(2000), FS))
