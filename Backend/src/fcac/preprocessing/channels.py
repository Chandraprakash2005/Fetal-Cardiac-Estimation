"""src/fcac/preprocessing/channels.py

Channel exclusion and kurtosis-based selection (NInFEA only -- ADFECGDB's 4 abdominal
channels are used as-is).

Implements: CLAUDE.md §2 "Kurtosis channel selection".
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
from scipy import stats


def exclude_nonphys(
    x: np.ndarray, channel_names: Sequence[str], exclude: Sequence[str]
) -> Tuple[np.ndarray, List[str]]:
    """Drop non-physiological channels (dc1-4, saw, sync) by name.

    x : (C, N). Returns (x_kept, kept_names).
    """
    exclude_set = set(exclude)
    keep_indices = [i for i, name in enumerate(channel_names) if name not in exclude_set]
    kept_names = [channel_names[i] for i in keep_indices]
    return x[keep_indices, :], kept_names


def kurtosis_select(
    x: np.ndarray, n_select: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the top-`n_select` channels by kurtosis.

    IMPORTANT (CLAUDE.md §2): call this on the ALREADY BANDPASS-FILTERED signal, not
    raw -- kurtosis on a drift-dominated raw signal reflects drift shape, not heartbeat
    quality.

    A dead/constant electrode (real NInFEA recordings have these -- e.g. recording 34's
    uni_abd9) has zero variance, so scipy.stats.kurtosis returns NaN for it. NaN sorts
    to the END of `np.argsort` ascending, which `[::-1]` then puts FIRST -- silently
    ranking the least informative channel as the best one. Non-finite kurtosis values
    are treated as the lowest possible rank instead, so a dead channel is never picked
    ahead of a channel that actually has signal.

    x : (C, N). Returns (x_selected (n_select, N), selected_indices, kurtosis_per_channel).
    """
    kurtosis_per_channel = stats.kurtosis(x, axis=-1)
    rank_key = np.where(np.isfinite(kurtosis_per_channel), kurtosis_per_channel, -np.inf)
    selected_indices = np.argsort(rank_key)[::-1][:n_select]
    return x[selected_indices, :], selected_indices, kurtosis_per_channel
