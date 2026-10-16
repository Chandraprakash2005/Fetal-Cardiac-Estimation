"""src/fcac/evaluation/ood.py

Mahalanobis out-of-distribution (OOD) detector for latent-space representations.
Used to flag NInFEA zero-shot windows/recordings that fall far outside the ADFECGDB
training distribution in CVAE latent space (CLAUDE.md §3 "GA coverage gap": training
is 38-41 wks, validation is 21-27 wks -- downstream code must never assume overlapping
GA ranges, so an explicit OOD flag on the zero-shot side is required, not optional).

Implements: CLAUDE.md §3 "GA coverage gap" note; additive diagnostic, not a locked
architecture change.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch


class OODStats(NamedTuple):
    mean: torch.Tensor      # (D,)
    inv_cov: torch.Tensor   # (D, D)
    threshold: float        # percentile of training distances


def fit_ood(z_train: torch.Tensor, percentile: float = 99.0, ridge: float = 1e-3) -> OODStats:
    """Fit a Mahalanobis OOD detector on training latents.

    z_train    : (N, D) training-distribution latent vectors (e.g. ADFECGDB CVAE means).
    percentile : threshold is this percentile of the training distances themselves.
    ridge      : diagonal regularisation so inversion is stable even when a latent
                 dimension has near-zero variance in the training set.

    Raises ValueError when N <= D: the ridge term keeps `torch.linalg.inv` from
    raising on a singular covariance, but a sample covariance from N <= D points is
    rank-deficient regardless -- the resulting distances would be numerically finite
    but statistically meaningless (final review, 2026-09-29: fitting on 5
    recording-level means in a 64-dim latent space flagged every in-distribution
    point as OOD). Callers with too few aggregate points (e.g. per-recording means)
    should fit on the finer-grained points instead (e.g. per-window latents).
    """
    if z_train.shape[0] <= z_train.shape[1]:
        raise ValueError(
            f"fit_ood needs more samples than dimensions to estimate a covariance "
            f"(got N={z_train.shape[0]} samples, D={z_train.shape[1]} dims); fit on "
            f"finer-grained points (e.g. per-window latents, not per-recording means)"
        )
    mean = z_train.mean(0)
    cov = torch.cov(z_train.T) + ridge * torch.eye(z_train.shape[1], dtype=z_train.dtype)
    inv_cov = torch.linalg.inv(cov)
    provisional = OODStats(mean=mean, inv_cov=inv_cov, threshold=0.0)
    train_distances = mahalanobis_distance(z_train, provisional)
    threshold = float(np.percentile(train_distances.numpy(), percentile))
    return OODStats(mean=mean, inv_cov=inv_cov, threshold=threshold)


def mahalanobis_distance(z: torch.Tensor, stats: OODStats) -> torch.Tensor:
    """z : (N, D) -> (N,) Mahalanobis distance from the fitted training distribution."""
    diff = z - stats.mean
    return torch.sqrt(((diff @ stats.inv_cov) * diff).sum(dim=-1).clamp_min(0))


def is_out_of_distribution(z: torch.Tensor, stats: OODStats) -> torch.Tensor:
    """z : (N, D) -> (N,) bool, True where the distance exceeds the fitted threshold."""
    return mahalanobis_distance(z, stats) > stats.threshold
