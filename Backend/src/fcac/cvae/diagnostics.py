"""src/fcac/cvae/diagnostics.py

Sanity checks for a trained ContrastiveVAE's latent space, run after training (not
part of the loss): does latent distance track |ΔGA| at all (it should, if the
contrastive objective is doing anything), and can a trivial ridge regression from the
latent recover GA under leave-one-recording-out (a floor the Transformer stage should
comfortably beat).

Implements: CLAUDE.md §2 "CVAE" -- diagnostic support for reports/engineering_log.md
decisions, not a locked architecture requirement.
"""

from __future__ import annotations

import numpy as np
import torch


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation without a scipy dependency."""
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def latent_distance_vs_ga_gap(
    z: torch.Tensor, ga_days: torch.Tensor, max_pairs: int = 2000, seed: int = 0
) -> float:
    """Spearman correlation between pairwise latent distance and pairwise |ΔGA| over
    a random sample of window pairs. Should be > 0 if the contrastive loss is
    separating windows by GA at all; NaN if every sampled window shares the same GA."""
    generator = torch.Generator().manual_seed(seed)
    if len(z) > max_pairs:
        idx = torch.randperm(len(z), generator=generator)[:max_pairs]
        z, ga_days = z[idx], ga_days[idx]
    i, j = torch.triu_indices(len(z), len(z), offset=1)
    latent_dist = torch.cdist(z, z)[i, j].numpy()
    ga_gap = (ga_days[:, None] - ga_days[None, :]).abs()[i, j].numpy()
    if ga_gap.std() == 0:
        return float("nan")
    return spearman_corr(latent_dist, ga_gap)


def ridge_regression_loro_mae_days(
    z: torch.Tensor, ga_days: torch.Tensor, subject_idx: torch.Tensor, alpha: float = 1.0
) -> float:
    """Leave-one-recording-out ridge regression z -> GA. Returns MAE in days -- a
    linear floor the Transformer stage should beat by a wide margin."""
    errors = []
    for subject in torch.unique(subject_idx):
        test_mask, train_mask = subject_idx == subject, subject_idx != subject
        design = torch.cat([z[train_mask], torch.ones(train_mask.sum(), 1)], dim=1)
        weights = torch.linalg.solve(
            design.T @ design + alpha * torch.eye(design.shape[1]),
            design.T @ ga_days[train_mask],
        )
        test_design = torch.cat([z[test_mask], torch.ones(test_mask.sum(), 1)], dim=1)
        prediction = (test_design @ weights).mean()
        errors.append((prediction - ga_days[test_mask][0]).abs().item())
    return float(np.mean(errors))
