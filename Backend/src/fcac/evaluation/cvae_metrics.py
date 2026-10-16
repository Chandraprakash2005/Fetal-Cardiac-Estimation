"""src/fcac/evaluation/cvae_metrics.py

Pure metric functions for validating the Contrastive VAE (CLAUDE.md §2 "CVAE"). All are
computed on HELD-OUT windows and against an explicit baseline, because a training loss
on its own says nothing about whether the latent is useful.

* reconstruction R^2 per feature, on OBSERVED entries only -- NaN features are
  median-imputed before training, and scoring the reconstruction of an imputed value is
  scoring how well the model recovers a constant we put there ourselves.
* active latent units -- a collapsed posterior has near-zero variance in most dimensions.
* same-subject nearest-neighbour retrieval -- the direct test of what the InfoNCE term
  optimises, against chance and against the same retrieval on raw standardised features.
* kNN regression -- does the latent carry gestational-age information?

Implements: CLAUDE.md §2 "CVAE" (validation), §10.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def recon_sums(
    x_true: np.ndarray, x_hat: np.ndarray, train_mean: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature (SSE, SST, n_observed) on the non-NaN entries of `x_true` (N, F).

    SST is taken about `train_mean` (the training-set mean per feature), so
    R^2 = 1 - SSE/SST measures the gain over the trivial "predict the training mean"
    baseline: 0 means no better than that, negative means worse.
    """
    observed = ~np.isnan(x_true)
    filled = np.where(observed, x_true, 0.0)
    sse = (((filled - x_hat) ** 2) * observed).sum(axis=0)
    sst = (((filled - train_mean) ** 2) * observed).sum(axis=0)
    return sse, sst, observed.sum(axis=0)


def r2_from_sums(sse: np.ndarray, sst: np.ndarray, n: np.ndarray, min_obs: int = 5) -> np.ndarray:
    """R^2 per feature from accumulated sums; NaN where fewer than `min_obs` observed or
    the held-out values have no variance about the train mean."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - sse / sst
    r2[(n < min_obs) | (sst <= 0)] = np.nan
    return r2


def active_units(mu: np.ndarray, threshold: float = 0.01) -> int:
    """Latent dimensions whose posterior mean varies across windows by more than
    `threshold` (variance). Few active units = posterior collapse."""
    return int((mu.var(axis=0) > threshold).sum())


def _standardise(ref: np.ndarray, *others: np.ndarray):
    mean, std = ref.mean(axis=0), ref.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return [(a - mean) / std for a in (ref, *others)]


def same_subject_nn(
    z_query: np.ndarray, subj_query: np.ndarray, z_ref: np.ndarray, subj_ref: np.ndarray
) -> Tuple[float, float, int]:
    """Fraction of query windows whose Euclidean nearest reference window (after
    standardising with the reference statistics) is from the same subject, plus the
    chance level for the same queries (the share of reference windows that are the same
    subject, averaged over queries) and the number of queries scored. Queries whose
    subject is absent from the reference are skipped -- they cannot be retrieved."""
    ref_s, query_s = _standardise(z_ref, z_query)
    present = np.isin(subj_query, subj_ref)
    if not present.any():
        return float("nan"), float("nan"), 0
    q, sq = query_s[present], subj_query[present]
    dist = ((q[:, None, :] - ref_s[None, :, :]) ** 2).sum(-1)
    nearest = subj_ref[dist.argmin(axis=1)]
    accuracy = float((nearest == sq).mean())
    chance = float(np.mean([(subj_ref == s).mean() for s in sq]))
    return accuracy, chance, int(present.sum())


def knn_regress(z_query: np.ndarray, z_ref: np.ndarray, y_ref: np.ndarray, k: int = 5) -> np.ndarray:
    """Mean target of the k nearest reference windows (Euclidean, after standardising
    with the reference statistics) for each query window."""
    ref_s, query_s = _standardise(z_ref, z_query)
    k = min(k, len(ref_s))
    dist = ((query_s[:, None, :] - ref_s[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(dist, axis=1)[:, :k]
    return y_ref[idx].mean(axis=1)
