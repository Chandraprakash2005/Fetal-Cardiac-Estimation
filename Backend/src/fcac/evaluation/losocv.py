"""src/fcac/evaluation/losocv.py

Leave-one-subject-out cross-validation driver (CLAUDE.md §2 "Evaluation": ADFECGDB,
5 folds -- one per recording, since each of the 5 ADFECGDB recordings is its own
subject).

Implements: CLAUDE.md §8 item 8.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np


def losocv_folds(subject_ids: Sequence[str]) -> List[Dict[str, Any]]:
    """Build LOSOCV fold index arrays for a flat array of per-window subject ids.

    Returns a list of {"held_out_subject", "train_idx", "test_idx"} dicts, one per
    unique subject.
    """
    subject_ids_arr = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids_arr)
    folds = []
    for held_out in unique_subjects:
        test_idx = np.where(subject_ids_arr == held_out)[0]
        train_idx = np.where(subject_ids_arr != held_out)[0]
        folds.append(
            {"held_out_subject": held_out, "train_idx": train_idx, "test_idx": test_idx}
        )
    return folds


def run_losocv(
    subject_ids: Sequence[str],
    train_fn: Callable[[np.ndarray], Any],
    eval_fn: Callable[[Any, np.ndarray], Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Generic LOSOCV driver: for each held-out subject, call train_fn(train_idx) to
    get a fitted model/state, then eval_fn(state, test_idx) to get that fold's metrics.

    train_fn and eval_fn close over whatever data (features, targets) they need --
    this driver only owns the fold indexing.
    """
    results = []
    for fold in losocv_folds(subject_ids):
        state = train_fn(fold["train_idx"])
        metrics = eval_fn(state, fold["test_idx"])
        results.append({"held_out_subject": fold["held_out_subject"], **metrics})
    return results
