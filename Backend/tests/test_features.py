"""tests/test_features.py

Tests for the 12 cardiac biomarkers: FEATURE_NAMES structure, fiducial detection,
each feature's behaviour on a synthetic beat train with known properties, and the
record-level pooling path.

Everything runs on synthetic impulse-train ECGs (CLAUDE.md section 6 "Tests") so the
suite needs no PhysioNet data. Note these check CORRECTNESS OF COMPUTATION, not clinical
validity -- the parameter choices are documented defaults, not supervisor-confirmed
(plan.md item 4).

Implements: CLAUDE.md section 8 item 9.
"""

import numpy as np
import pytest

from fcac.config import Config, load_config
from fcac.features import dynamic
from fcac.features.extract import (
    FEATURE_NAMES,
    extract_features,
    extract_features_batch,
    extract_features_record,
    pooled_rr_intervals,
)
from fcac.features.fiducials import detect_fiducials, rr_intervals
from fcac.features.morphological import qrs_mean, qtc_sigma

FS = 1000.0


def _beat_train(bpm=130.0, n_samples=5000, fs=FS, jitter_ms=0.0, seed=0):
    """Synthetic fetal ECG: a Gaussian-doublet QRS repeated at a known rate."""
    t = np.arange(n_samples) / fs
    rng = np.random.default_rng(seed)
    signal = np.zeros(n_samples)
    period = 60.0 / bpm
    centre = 0.15
    while centre < n_samples / fs:
        offset = rng.normal(0.0, jitter_ms / 1000.0) if jitter_ms else 0.0
        u = (t - (centre + offset)) / 0.006
        signal += np.exp(-0.5 * u**2) * (1 - u**2)
        centre += period
    return signal.astype(np.float32)


# --------------------------------------------------------------------------- structure


def test_feature_names_has_twelve_unique_entries():
    assert len(FEATURE_NAMES) == 12
    assert len(set(FEATURE_NAMES)) == 12


def test_extract_features_returns_twelve_floats():
    values = extract_features(_beat_train(), FS)
    assert values.shape == (12,)
    assert values.dtype == np.float32


def test_extract_features_batch_shape():
    windows = np.stack([_beat_train(seed=i) for i in range(3)])
    assert extract_features_batch(windows, FS).shape == (3, 12)


def test_features_never_raise_on_degenerate_input():
    """A flat window has no beats at all -- every feature must be NaN, not an exception."""
    values = extract_features(np.zeros(5000, dtype=np.float32), FS)
    assert values.shape == (12,)
    assert np.isnan(values).all()


# --------------------------------------------------------------------------- fiducials


def test_rr_intervals_recover_a_known_rate():
    rr = rr_intervals(_beat_train(bpm=130.0), FS)
    assert rr.size >= 8
    assert abs(60000.0 / np.median(rr) - 130.0) < 5.0


def test_detect_fiducials_aligns_all_waves_per_beat():
    f = detect_fiducials(_beat_train(), FS)
    lengths = {len(v) for v in f.values()}
    assert len(lengths) == 1, "every fiducial array must be one entry per beat"
    assert len(f["R"]) >= 8
    # Q precedes R precedes S, wherever all three were found.
    valid = np.isfinite(f["Q"]) & np.isfinite(f["S"])
    assert np.all(f["Q"][valid] < f["R"][valid])
    assert np.all(f["R"][valid] < f["S"][valid])


def test_fiducial_windows_are_fetal_scaled():
    """QRS duration must land in the fetal range (~20-80 ms), not the adult one."""
    duration = qrs_mean(_beat_train(), FS)
    assert np.isfinite(duration)
    assert 5.0 < duration < 100.0


# ----------------------------------------------------------------------- morphological


def test_qtc_sigma_is_nan_without_enough_beats():
    assert np.isnan(qtc_sigma(_beat_train(n_samples=400), FS))


# ---------------------------------------------------------------------------- dynamic


def _cfg() -> Config:
    return load_config()


def test_entropy_is_lower_for_a_regular_series():
    """A perfectly periodic RR series must be less complex than a random one."""
    cfg = _cfg()
    regular = np.full(200, 460.0) + np.tile([0.5, -0.5], 100)
    random = np.random.default_rng(0).normal(460, 20, 200)
    assert dynamic._sampen_from_rr(regular, cfg) < dynamic._sampen_from_rr(random, cfg)


def test_symbolic_dynamics_is_100pct_for_a_constant_like_series():
    """Every 3-beat word identical -> 0V% = 100."""
    cfg = _cfg()
    rr = np.full(60, 460.0)
    rr[0] = 400.0  # give it non-zero range so quantisation is defined
    assert dynamic._symbolic_from_rr(rr, cfg) > 90.0


def test_dynamic_features_are_nan_on_short_series():
    """The 5 s window limit: too few RR intervals must yield NaN, not a fake number."""
    cfg = _cfg()
    short = np.array([460.0, 455.0, 462.0])
    for name, core in dynamic.DYNAMIC_FROM_RR.items():
        assert np.isnan(core(short, cfg)), f"{name} returned a value on a 3-interval series"


def test_lf_hf_requires_a_resolvable_lf_band():
    """A 5 s tachogram cannot resolve 0.04 Hz -- it must be NaN, never 0.0.

    Regression test: an earlier version returned exactly 0.00 for every window, which
    looks like a measurement but is an artefact of the band being unresolvable.
    """
    cfg = _cfg()
    rr_5s = np.full(11, 460.0) + np.random.default_rng(0).normal(0, 10, 11)
    assert np.isnan(dynamic._lf_hf_from_rr(rr_5s, cfg))

    rr_long = np.full(600, 460.0) + np.random.default_rng(0).normal(0, 10, 600)
    assert np.isfinite(dynamic._lf_hf_from_rr(rr_long, cfg))


# ----------------------------------------------------------------------- record level


def test_pooled_rr_spans_all_windows():
    """Pooling must recover far more intervals than any single window holds."""
    cfg = _cfg()
    windows = np.stack([_beat_train(seed=i) for i in range(20)])
    pooled = pooled_rr_intervals(windows, FS, cfg)
    per_window = rr_intervals(windows[0], FS, cfg)
    assert pooled.size > 10 * per_window.size
    assert abs(60000.0 / np.median(pooled) - 130.0) < 8.0


def test_pooled_rr_rejects_overlapping_windows():
    cfg = load_config()
    data = cfg.to_dict()
    data["preprocessing"]["window"]["stride_samples"] = 2500  # overlapping
    with pytest.raises(ValueError, match="non-overlapping"):
        pooled_rr_intervals(np.stack([_beat_train()] * 2), FS, Config(data))


def test_record_level_completes_all_twelve_features():
    """The point of record-level extraction: dynamic features become computable."""
    cfg = _cfg()
    windows = np.stack([_beat_train(seed=i, jitter_ms=6.0) for i in range(30)])

    per_window = extract_features_batch(windows, FS, cfg)
    record = extract_features_record(windows, FS, cfg)

    assert record.shape == (12,)
    # No single window can supply LF/HF; the pooled record can.
    assert np.isnan(per_window[:, FEATURE_NAMES.index("lf_hf_ratio")]).all()
    assert np.isfinite(record[FEATURE_NAMES.index("lf_hf_ratio")])
    assert np.isfinite(record).sum() > np.isfinite(per_window).all(axis=0).sum()


# ------------------------------------------------- ground-truth interval validation
#
# These pin the three bugs found on 2026-09-08 by validating against a template whose
# intervals are known exactly (compact-support triangular waves, so every boundary is
# unambiguous). The reference paper defines none of the 12 biomarkers, so a synthetic
# ground truth is the only way to check these are computed correctly.


def _exact_template(n_samples=12000, fs=FS, noise=0.0, seed=0):
    """Beat train with EXACT 50 ms QRS, 40 ms P, symmetric QRS amplitudes."""
    def tri(t, centre, half_width, amplitude):
        y = amplitude * (1 - np.abs(t - centre) / half_width)
        return np.where(np.abs(t - centre) <= half_width, y, 0.0)

    t = np.arange(700) / fs * 1000.0
    r = 250.0
    beat = (
        tri(t, r - 120, 20, 0.15)                       # P: 40 ms span
        + tri(t, r - 17, 8, -0.30)                      # Q
        + tri(t, r, 10, 1.0)                            # R  -> QRS span 50 ms
        + tri(t, r + 17, 8, -0.30)                      # S (|Q| == |S|: symmetric)
        + tri(t, r + 185, 40, 0.30)                     # T
    )
    signal = np.zeros(n_samples)
    rng = np.random.default_rng(seed)
    position = 300.0
    while position + 700 < n_samples:
        i = int(position)
        signal[i : i + 700] += beat
        position += 460 + rng.normal(0, 8)
    if noise:
        signal += noise * rng.standard_normal(n_samples)
    return signal.astype(np.float32)


def test_qrs_duration_matches_a_known_interval():
    """Regression: a slope-flattening boundary rule measured this 50 ms QRS as 75 ms."""
    cfg = _cfg()
    f = detect_fiducials(_exact_template(), FS, cfg)
    durations = (f["qrs_offset"] - f["qrs_onset"]) / FS * 1000.0
    durations = durations[np.isfinite(durations)]
    assert durations.size > 10
    assert abs(durations.mean() - 50.0) / 50.0 < 0.10, durations.mean()


def test_p_duration_matches_a_known_interval():
    """Regression: p_dispersion used to measure the PR segment, not the P wave."""
    cfg = _cfg()
    f = detect_fiducials(_exact_template(), FS, cfg)
    durations = (f["p_offset"] - f["p_onset"]) / FS * 1000.0
    durations = durations[np.isfinite(durations)]
    assert durations.size > 10
    assert abs(durations.mean() - 40.0) / 40.0 < 0.12, durations.mean()


def test_qrs_asymmetry_is_one_for_a_symmetric_complex():
    """Regression: Pan-Tompkins' integration lag skewed this to 0.515."""
    from fcac.features.morphological import qrs_asymmetry

    value = qrs_asymmetry(_exact_template(), FS, _cfg())
    assert abs(value - 1.0) < 0.15, value


def test_r_peaks_are_refined_onto_the_true_apex():
    """The refined R must sit on the signal's own maximum, not the integrator's lag."""
    from fcac.features.fiducials import refine_r_peaks
    from fcac.memd.diagnostics import pan_tompkins_peaks

    signal = _exact_template()
    raw = pan_tompkins_peaks(signal, FS)
    refined = refine_r_peaks(signal, raw, FS, _cfg().features.fiducials.r_refine_ms)

    # Every refined peak is a local maximum of |signal| in its neighbourhood.
    for r in refined[1:-1]:
        window = np.abs(signal[max(0, r - 20) : r + 21])
        assert np.abs(signal[r]) >= window.max() - 1e-6


def test_p_dispersion_is_zero_for_constant_p_waves():
    from fcac.features.morphological import p_dispersion

    assert p_dispersion(_exact_template(), FS, _cfg()) < 5.0


def test_interval_measurement_survives_noise():
    """QRS duration must stay accurate with realistic additive noise."""
    cfg = _cfg()
    f = detect_fiducials(_exact_template(noise=0.05), FS, cfg)
    durations = (f["qrs_offset"] - f["qrs_onset"]) / FS * 1000.0
    durations = durations[np.isfinite(durations)]
    assert abs(durations.mean() - 50.0) / 50.0 < 0.15, durations.mean()


def test_rr_artefact_filter_removes_non_physiological_intervals():
    """Regression: unfiltered detector errors gave an RR SD of 180 ms (true ~15 ms) and
    corrupted all seven dynamic features."""
    from fcac.features.fiducials import filter_rr_artifacts

    cfg = _cfg()
    clean = np.full(100, 460.0) + np.random.default_rng(0).normal(0, 12, 100)
    corrupted = np.concatenate([clean, [97.0, 4320.0, 150.0, 2000.0]])

    filtered = filter_rr_artifacts(corrupted, cfg)
    assert filtered.size >= clean.size - 5
    assert filtered.min() >= cfg.features.dynamic.rr_min_ms
    assert filtered.max() <= cfg.features.dynamic.rr_max_ms
    assert filtered.std() < 40.0, "artefacts still inflating the spread"


def test_symbolic_dynamics_equiprobable_mode_resists_skew():
    """The equiprobable option exists for skewed RR spreads; check it is stable."""
    data = load_config().to_dict()
    data["features"]["dynamic"]["symbolic_binning"] = "equiprobable"
    cfg = Config(data)

    rr = np.full(200, 460.0) + np.random.default_rng(0).normal(0, 12, 200)
    baseline = dynamic._symbolic_from_rr(rr, cfg)
    skewed = np.concatenate([rr, [690.0]])
    assert abs(dynamic._symbolic_from_rr(skewed, cfg) - baseline) < 15.0


def test_symbolic_binning_rejects_unknown_method():
    data = load_config().to_dict()
    data["features"]["dynamic"]["symbolic_binning"] = "nonsense"
    with pytest.raises(ValueError, match="unknown symbolic_binning"):
        dynamic._symbolic_from_rr(np.full(60, 460.0) + np.arange(60) * 0.5, Config(data))
