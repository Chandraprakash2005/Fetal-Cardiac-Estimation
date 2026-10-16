"""tests/test_doppler_ref.py -- synthetic checks for fcac.evaluation.doppler_ref."""

import numpy as np

from fcac.evaluation.doppler_ref import estimate_lag, heart_rate_bpm, lag_differences


def test_estimate_lag_recovers_a_constant_offset_with_jitter():
    rng = np.random.default_rng(0)
    r_times = np.cumsum(rng.normal(0.43, 0.01, 200))
    doppler = r_times + 0.085 + rng.normal(0, 0.008, 200)
    lag, conc = estimate_lag(lag_differences(r_times, doppler))
    assert abs(lag - 0.085) < 0.01 and conc > 0.5


def test_unlocked_detections_have_low_concentration():
    rng = np.random.default_rng(1)
    doppler = np.cumsum(rng.normal(0.43, 0.01, 200))
    unrelated = np.cumsum(rng.normal(0.97, 0.05, 90))  # a maternal-rate train
    _, conc = estimate_lag(lag_differences(unrelated, doppler))
    assert conc < 0.3


def test_heart_rate_and_empty_inputs():
    assert abs(heart_rate_bpm(np.arange(0, 10, 0.4)) - 150) < 0.5
    assert np.isnan(heart_rate_bpm(np.array([0.0, 0.5])))
    assert lag_differences(np.array([]), np.array([1.0])).size == 0


def test_lag_is_unique_when_the_lag_is_close_to_half_a_period():
    """A true lag of -40 ms must not be read as +380 ms (the next beat)."""
    rng = np.random.default_rng(2)
    r_times = np.cumsum(rng.normal(0.42, 0.005, 200))
    doppler = r_times - 0.04 + rng.normal(0, 0.006, 200)
    lag, conc = estimate_lag(lag_differences(r_times, doppler))
    assert abs(lag + 0.04) < 0.01 and conc > 0.6
