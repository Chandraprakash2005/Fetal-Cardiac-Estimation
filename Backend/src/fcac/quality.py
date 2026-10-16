"""src/fcac/quality.py

Signal-quality assessment and gating for extracted fetal ECG.

Why this exists, and why it is a CORRECTNESS issue rather than an accuracy one: the
extraction chain fails silently. On ADFECGDB r04/r07/r10 the output locks onto the
maternal rate, and the 12-biomarker extractor will happily compute twelve
confident-looking numbers from that maternal signal. Nothing downstream can tell the
difference. Gating marks those windows and records so they are EXCLUDED from ageing
rather than silently averaged in.

The reference paper does the same thing: ICC 2026 pairs its 1D-CNN extractor with an
Earth Mover's Distance signal-quality framework and reports quality per record
(r01/r04/r08 high, r07 low). This module is the equivalent gate for this pipeline,
built from criteria the pipeline already measures.

Three criteria, all unsupervised (never reference annotations):
  1. Heart rate in the fetal band (110-160 bpm). A window reading 60-100 bpm has locked
     onto the mother -- this is the observed failure signature.
  2. RR artefact rate below a threshold. Detector slips produce impossible intervals;
     a high rate means the R peaks are unreliable (r08 measured 9.8%).
  3. Enough beats to measure anything at all.

Implements: CLAUDE.md section 2 "Evaluation"; the quality-gating role of ICC 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from fcac.config import Config, load_config
from fcac.memd.diagnostics import pan_tompkins_bpm, pan_tompkins_peaks


@dataclass(frozen=True)
class QualityReport:
    """Per-window quality verdict.

    acceptable : whether this window should reach the feature/ageing stages at all.
    reasons    : human-readable failures; empty when acceptable.
    """

    acceptable: bool
    heart_rate_bpm: float
    n_beats: int
    rr_artifact_rate: float
    reasons: List[str]


def assess_window(
    fecg: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> QualityReport:
    """Assess one extracted single-channel fetal ECG window (N,)."""
    if cfg is None:
        cfg = load_config()
    q = cfg.quality

    peaks = pan_tompkins_peaks(fecg, fs)
    n_beats = int(len(peaks))
    heart_rate = pan_tompkins_bpm(fecg, fs)

    # Artefact rate is measured BEFORE filtering, on the raw detected intervals -- that
    # is what says whether the detector is trustworthy here. features/ shows the filtered
    # view; this shows how much had to be thrown away.
    if n_beats >= 2:
        rr = np.diff(peaks) / fs * 1000.0
        bad = (rr < cfg.features.dynamic.rr_min_ms) | (rr > cfg.features.dynamic.rr_max_ms)
        artifact_rate = float(np.count_nonzero(bad) / rr.size)
    else:
        artifact_rate = 1.0

    reasons: List[str] = []
    if n_beats < q.min_beats:
        reasons.append(f"only {n_beats} beats (need {q.min_beats})")
    if not np.isfinite(heart_rate):
        reasons.append("heart rate undetectable")
    elif not (q.fetal_hr_min_bpm <= heart_rate <= q.fetal_hr_max_bpm):
        reasons.append(
            f"heart rate {heart_rate:.0f} bpm outside fetal band "
            f"{q.fetal_hr_min_bpm:.0f}-{q.fetal_hr_max_bpm:.0f} "
            "(likely locked onto the maternal rhythm)"
        )
    if artifact_rate > q.max_rr_artifact_rate:
        reasons.append(
            f"RR artefact rate {artifact_rate:.0%} above {q.max_rr_artifact_rate:.0%}"
        )

    return QualityReport(
        acceptable=not reasons,
        heart_rate_bpm=float(heart_rate),
        n_beats=n_beats,
        rr_artifact_rate=artifact_rate,
        reasons=reasons,
    )


def assess_record(
    fecg_windows: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> dict:
    """Assess a whole record's (W, N) extracted fECG.

    A record is usable when at least `quality.min_acceptable_window_fraction` of its
    windows pass. Returns the per-window mask so callers can drop individual bad windows
    even inside an otherwise usable record.
    """
    if cfg is None:
        cfg = load_config()

    reports = [assess_window(fecg_windows[w], fs, cfg) for w in range(fecg_windows.shape[0])]
    mask = np.array([r.acceptable for r in reports], dtype=bool)
    fraction = float(mask.mean()) if mask.size else 0.0

    rates = np.array([r.heart_rate_bpm for r in reports], dtype=float)
    finite = rates[np.isfinite(rates)]

    return {
        "n_windows": int(fecg_windows.shape[0]),
        "n_acceptable": int(mask.sum()),
        "acceptable_fraction": fraction,
        "usable": fraction >= cfg.quality.min_acceptable_window_fraction,
        "median_heart_rate_bpm": float(np.median(finite)) if finite.size else float("nan"),
        "acceptable_mask": mask,
        "reports": reports,
    }


def filter_acceptable(
    fecg_windows: np.ndarray, fs: float, cfg: Optional[Config] = None
) -> np.ndarray:
    """Return only the windows that pass the gate -- the array to feed downstream.

    Use this before feature extraction so no biomarker is ever computed from a
    maternal-locked window.
    """
    summary = assess_record(fecg_windows, fs, cfg)
    return fecg_windows[summary["acceptable_mask"]]
