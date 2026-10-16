"""src/fcac/evaluation/verify_units.py

Mandatory unit-verification check (CLAUDE.md §2 "Evaluation": "Mandatory MAE
unit-verification script before any number is reported"). Run this and confirm its
output before writing any MAE/RMSE number into a report, table, or figure.

Implements: CLAUDE.md §8 item 8.
"""

from __future__ import annotations

import numpy as np

from fcac.evaluation.ageing import mae

DAYS_PER_WEEK = 7.0


def verify_mae_units(y_true_days: np.ndarray, y_pred_days: np.ndarray) -> None:
    """Sanity-check that an MAE computed in days is internally consistent with the
    same MAE computed in weeks (mae_days / 7 == mae_weeks), and that it is not
    implausible for a gestational-age-scale quantity, before any number from this run
    is reported.

    Raises AssertionError on failure -- do not catch this and report a number anyway.
    """
    mae_days = mae(y_true_days, y_pred_days)
    mae_weeks = mae(y_true_days / DAYS_PER_WEEK, y_pred_days / DAYS_PER_WEEK)

    assert np.isclose(mae_days / DAYS_PER_WEEK, mae_weeks, rtol=1e-6), (
        f"Unit mismatch: mae_days/7={mae_days / DAYS_PER_WEEK} != mae_weeks={mae_weeks}"
    )
    assert 0 <= mae_days <= 365, (
        f"MAE of {mae_days} days is implausible for a gestational-age-scale quantity "
        "-- check for a unit bug before reporting this number."
    )
    print(f"MAE unit check passed: {mae_days:.3f} days == {mae_weeks:.3f} weeks")


if __name__ == "__main__":
    raise SystemExit(
        "TODO(bhuvanesh): call verify_mae_units(y_true_days, y_pred_days) with real "
        "evaluation outputs once scripts/08_evaluate.py produces them."
    )
