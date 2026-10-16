"""tests/test_cvae_metrics.py -- synthetic checks for fcac.evaluation.cvae_metrics."""

import numpy as np
import pytest

from fcac.evaluation.cvae_metrics import (
    active_units, knn_regress, r2_from_sums, recon_sums, same_subject_nn,
)


def test_r2_is_one_for_perfect_and_zero_for_train_mean_prediction():
    x = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0], [6.0, 60.0]])
    sse, sst, n = recon_sums(x, x, x.mean(axis=0))
    assert r2_from_sums(sse, sst, n) == pytest.approx([1.0, 1.0])
    mean_pred = np.tile(x.mean(axis=0), (len(x), 1))
    sse, sst, n = recon_sums(x, mean_pred, x.mean(axis=0))
    assert r2_from_sums(sse, sst, n) == pytest.approx([0.0, 0.0])


def test_r2_ignores_nan_entries_and_is_nan_when_too_few_observed():
    x = np.full((8, 2), np.nan)
    x[:, 0] = np.arange(8.0)
    x[:2, 1] = [1.0, 2.0]  # only 2 observed
    sse, sst, n = recon_sums(x, np.zeros_like(x), np.zeros(2))
    r2 = r2_from_sums(sse, sst, n)
    assert np.isfinite(r2[0]) and np.isnan(r2[1])


def test_active_units_detects_collapse():
    rng = np.random.default_rng(0)
    mu = np.zeros((50, 8))
    mu[:, :3] = rng.normal(size=(50, 3))
    assert active_units(mu) == 3


def test_same_subject_nn_separates_clusters_and_reports_chance():
    rng = np.random.default_rng(0)
    centres = {"a": 0.0, "b": 10.0, "c": -10.0}
    z_ref = np.concatenate([rng.normal(c, 0.1, (10, 4)) for c in centres.values()])
    s_ref = np.repeat(list(centres), 10)
    z_q = np.concatenate([rng.normal(c, 0.1, (3, 4)) for c in centres.values()])
    s_q = np.repeat(list(centres), 3)
    acc, chance, n = same_subject_nn(z_q, s_q, z_ref, s_ref)
    assert acc == 1.0 and chance == pytest.approx(1 / 3) and n == 9


def test_same_subject_nn_skips_subjects_missing_from_reference():
    z = np.random.default_rng(0).normal(size=(4, 3))
    acc, chance, n = same_subject_nn(z, np.array(["x"] * 4), z, np.array(["y"] * 4))
    assert n == 0 and np.isnan(acc)


def test_knn_regress_recovers_linear_target():
    rng = np.random.default_rng(0)
    z_ref = rng.uniform(0, 1, (200, 1))
    y_ref = 100 * z_ref[:, 0]
    pred = knn_regress(np.array([[0.2], [0.8]]), z_ref, y_ref, k=5)
    assert pred == pytest.approx([20.0, 80.0], abs=5.0)
