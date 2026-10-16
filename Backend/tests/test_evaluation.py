"""tests/test_evaluation.py

Synthetic-data tests for src/fcac/evaluation/ (CLAUDE.md §8 item 9).
"""

import numpy as np
import pytest
import torch

from fcac.config import load_config
from fcac.evaluation.ageing import bootstrap_ci, mae, pearson_r, rmse
from fcac.evaluation.delta_age import delta_age_days
from fcac.evaluation.extraction import bland_altman_rr, mse, qrs_f1, snr_db
from fcac.evaluation.losocv import losocv_folds, run_losocv
from fcac.evaluation.verify_units import verify_mae_units


def test_mae_rmse_pearson_r_on_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    assert mae(y_true, y_pred) == 0.0
    assert rmse(y_true, y_pred) == 0.0
    assert pearson_r(y_true, y_pred) == pytest.approx(1.0)


def test_bootstrap_ci_contains_point_estimate():
    cfg = load_config()
    rng = np.random.default_rng(0)
    y_true = rng.normal(280, 5, size=30)
    y_pred = y_true + rng.normal(0, 2, size=30)

    point, lo, hi = bootstrap_ci(y_true, y_pred, mae, cfg)
    assert lo <= point <= hi


def test_delta_age_days():
    ca = np.array([285.0, 270.0])
    ga = np.array([280.0, 280.0])
    delta = delta_age_days(ca, ga)
    assert np.allclose(delta, [5.0, -10.0])


def test_snr_db_higher_for_cleaner_extraction():
    rng = np.random.default_rng(0)
    reference = np.sin(np.linspace(0, 20, 1000))
    clean = reference + rng.normal(0, 0.01, size=1000)
    noisy = reference + rng.normal(0, 1.0, size=1000)
    assert snr_db(clean, reference) > snr_db(noisy, reference)


def test_mse_matches_definition_and_snr_formula():
    reference = np.array([1.0, 2.0, 3.0, 4.0])
    extracted = np.array([1.0, 2.0, 3.0, 6.0])
    assert mse(extracted, reference) == pytest.approx(1.0)
    assert mse(reference, reference) == 0.0
    expected_snr = 10 * np.log10(np.sum(reference**2) / np.sum((reference - extracted) ** 2))
    assert snr_db(extracted, reference) == pytest.approx(expected_snr)


def test_qrs_f1_perfect_match():
    reference_peaks = np.array([100, 200, 300])
    detected_peaks = np.array([101, 199, 305])
    f1, _precision, _recall = qrs_f1(
        detected_peaks, reference_peaks, fs=1000.0, tolerance_ms=50.0
    )
    assert f1 == pytest.approx(1.0)


def test_bland_altman_rr_bias_zero_for_identical_series():
    rr = np.array([500.0, 510.0, 495.0])
    bias, lo, hi = bland_altman_rr(rr, rr)
    assert bias == 0.0
    assert lo == 0.0 and hi == 0.0


def test_losocv_folds_cover_every_subject_exactly_once_as_holdout():
    subject_ids = ["r01", "r01", "r04", "r07", "r08", "r10"]
    folds = losocv_folds(subject_ids)
    assert len(folds) == 5  # 5 unique subjects
    held_out = {f["held_out_subject"] for f in folds}
    assert held_out == {"r01", "r04", "r07", "r08", "r10"}


def test_run_losocv_calls_train_and_eval_per_fold():
    subject_ids = ["a", "b", "c"]

    def train_fn(train_idx):
        return "model"

    def eval_fn(state, test_idx):
        return {"mae_days": 1.0}

    results = run_losocv(subject_ids, train_fn, eval_fn)
    assert len(results) == 3
    assert all(r["mae_days"] == 1.0 for r in results)


def test_verify_mae_units_passes_for_plausible_values():
    y_true = np.array([280.0, 281.0, 279.0])
    y_pred = np.array([282.0, 280.0, 278.0])
    verify_mae_units(y_true, y_pred)  # should not raise


def test_verify_mae_units_flags_implausible_mae():
    y_true = np.array([280.0])
    y_pred = np.array([280.0 + 10000.0])
    with pytest.raises(AssertionError):
        verify_mae_units(y_true, y_pred)


def test_fit_ood_flags_far_points_and_keeps_near_points_in_distribution():
    from fcac.evaluation.ood import fit_ood, is_out_of_distribution

    rng = np.random.default_rng(0)
    train = torch.as_tensor(rng.normal(0, 1, size=(200, 4)), dtype=torch.float32)
    stats = fit_ood(train, percentile=95.0)

    near = train[:5]
    far = torch.full((3, 4), 50.0)
    assert not is_out_of_distribution(near, stats).any()
    assert is_out_of_distribution(far, stats).all()


def test_fit_ood_raises_when_samples_do_not_exceed_dimensions():
    """Final review: N <= D makes the sample covariance rank-deficient, so distances
    computed from it are numerically stable (the ridge term prevents a singular
    inverse) but statistically meaningless -- fit_ood must refuse rather than return
    an unreliable threshold silently."""
    from fcac.evaluation.ood import fit_ood

    z = torch.randn(10, 20)  # N=10 <= D=20
    with pytest.raises(ValueError):
        fit_ood(z)


def test_fit_ood_handles_zero_variance_dimension():
    """Final review: the original assertion only checked `mean`, which is finite
    regardless of the covariance -- strengthened to check the parts that actually
    depend on inverting the (would-be-singular, but for the ridge term) covariance."""
    from fcac.evaluation.ood import fit_ood

    z = torch.zeros(20, 3)
    z[:, 0] = torch.randn(20)
    stats = fit_ood(z)  # must not raise on the singular covariance
    assert torch.isfinite(stats.mean).all()
    assert torch.isfinite(stats.inv_cov).all()
    assert np.isfinite(stats.threshold)


def test_gain_matched_snr_is_scale_and_sign_free():
    from fcac.evaluation.extraction import gain_matched_snr_db

    rng = np.random.default_rng(0)
    y = np.sin(np.linspace(0, 40, 2000))
    x = -3.7 * y + rng.normal(0, 0.05, 2000)          # wrong scale AND wrong sign, small noise
    snr, gain, r = gain_matched_snr_db(x, y)
    assert snr > 20 and gain < 0 and r < -0.99
    assert gain_matched_snr_db(10 * x, y)[0] == pytest.approx(snr)   # scale-invariant
    assert gain_matched_snr_db(-x, y)[0] == pytest.approx(snr)       # sign-invariant


def test_gain_matched_snr_matches_the_correlation_identity_and_handles_degenerate_input():
    from fcac.evaluation.extraction import gain_matched_snr_db

    rng = np.random.default_rng(1)
    y, x = rng.normal(size=500), rng.normal(size=500)
    snr, _, r = gain_matched_snr_db(x, y)
    assert snr == pytest.approx(-10 * np.log10(1 - r**2))
    assert snr >= 0
    assert np.isnan(gain_matched_snr_db(np.zeros(10), y[:10])[0])
