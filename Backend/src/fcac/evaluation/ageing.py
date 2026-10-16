"""src/fcac/evaluation/ageing.py

Standard regression metrics for Cardiac Age evaluation, plus bootstrap confidence
intervals (CLAUDE.md §2 "Evaluation": MAE in days, 95% bootstrap CI, 1000 resamples,
2.5-97.5 pct).

Implements: CLAUDE.md §5 evaluation metrics; §8 item 8.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from fcac.config import Config


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    cfg: Config,
    seed: Optional[int] = None,
) -> Tuple[float, float, float]:
    """Bootstrap CI for `metric_fn(y_true, y_pred)` (CLAUDE.md §2 "Evaluation":
    1000 resamples, 2.5-97.5 pct).

    Returns (point_estimate, ci_low, ci_high).
    """
    rng = np.random.default_rng(seed if seed is not None else cfg.seed)
    n = len(y_true)
    n_resamples = cfg.evaluation.bootstrap.n_resamples

    point_estimate = metric_fn(y_true, y_pred)
    resampled = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled[i] = metric_fn(y_true[idx], y_pred[idx])

    ci_low = float(np.percentile(resampled, cfg.evaluation.bootstrap.ci_low_pct))
    ci_high = float(np.percentile(resampled, cfg.evaluation.bootstrap.ci_high_pct))
    return point_estimate, ci_low, ci_high
