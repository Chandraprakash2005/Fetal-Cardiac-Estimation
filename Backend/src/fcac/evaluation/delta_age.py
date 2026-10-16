"""src/fcac/evaluation/delta_age.py

Delta-Age = CA - GA (CLAUDE.md §1, §2) -- candidate IUGR screening index. GA is the
externally supplied LMP-dated gestational age (never a model output); CA is the
transformer's predicted Cardiac Age. Both in days (CLAUDE.md §5).

Implements: CLAUDE.md §5 CAPrediction (+ external GA) -> Delta-Age.
"""

from __future__ import annotations

import numpy as np


def delta_age_days(ca_days: np.ndarray, ga_days: np.ndarray) -> np.ndarray:
    """Delta-Age = CA - GA, both in days. Positive = cardiac maturity ahead of the
    LMP-dated GA."""
    return ca_days - ga_days
