"""src/fcac/io/cinc2013.py

Loads PhysioNet/CinC Challenge 2013 set-a records into the Recording contract
(CLAUDE.md section 5). Loading only -- filtering, normalisation and windowing happen in
src/fcac/preprocessing/ so every dataset arrives downstream in the same shape.

This is the SECOND zero-shot validation cohort (train ADFECGDB -> validate NInFEA +
CinC 2013). Nothing here is ever trained on.

Dataset facts, verified against the files on 2026-09-08:
  * 25 records a01..a25, 60 s each, WFDB format.
  * 4 abdominal channels AECG1..AECG4 at 1000 Hz -- already exactly the 4 channels at
    the target rate the pipeline wants, so neither kurtosis selection nor resampling
    applies here (unlike NInFEA's 24 channels at 2048 Hz).
  * `.fqrs` annotations give fetal R-peak sample indices. Like NInFEA this is a TIMING
    reference only -- there is no scalp ECG, so no amplitude ground truth and no SNR
    metric (CLAUDE.md section 2 "Ground truth").
  * 13 of 25 records carry strong 50 Hz mains, so the notch is enabled in
    configs/datasets/cinc2013.yaml.

Implements: CLAUDE.md section 5 Recording contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import wfdb

from fcac.config import Config
from fcac.io.types import Recording

logger = logging.getLogger(__name__)


def load_cinc2013_record(record_id: str, cfg: Config) -> Recording:
    """Load one CinC 2013 record's 4 abdominal channels at native 1000 Hz."""
    record = wfdb.rdrecord(str(Path(cfg.dataset.raw_dir) / record_id))

    prefix = cfg.dataset.abdominal_channel_prefix
    abdo_indices = [i for i, name in enumerate(record.sig_name) if name.startswith(prefix)]
    if len(abdo_indices) != cfg.preprocessing.n_channels:
        raise ValueError(
            f"{record_id}: expected {cfg.preprocessing.n_channels} channels with prefix "
            f"'{prefix}', found {len(abdo_indices)} in {record.sig_name}"
        )

    signal = record.p_signal[:, abdo_indices].T.astype(np.float32)  # (C, N)

    # CinC 2013 records contain occasional NaN gaps; leaving them in would poison the
    # whole recording once filtfilt smears them across the signal.
    if not np.isfinite(signal).all():
        n_bad = int((~np.isfinite(signal)).sum())
        logger.warning(
            "%s: %d non-finite samples, interpolating over them before preprocessing",
            record_id,
            n_bad,
        )
        signal = _interpolate_non_finite(signal)

    return Recording(
        signal=signal,
        fs_raw=float(record.fs),
        fetal_ref=load_cinc2013_fqrs(record_id, cfg),
        ga_days=_ga_days(record_id, cfg),
        subject_id=record_id,
    )


def _ga_days(record_id: str, cfg: Config) -> Optional[float]:
    ga_days = cfg.dataset.ga_days
    return ga_days[record_id] if record_id in ga_days else None


def _interpolate_non_finite(signal: np.ndarray) -> np.ndarray:
    """Linearly interpolate over NaN/inf runs, per channel."""
    out = signal.copy()
    for c in range(out.shape[0]):
        channel = out[c]
        bad = ~np.isfinite(channel)
        if not bad.any():
            continue
        if bad.all():
            out[c] = 0.0
            continue
        good_idx = np.flatnonzero(~bad)
        channel[bad] = np.interp(np.flatnonzero(bad), good_idx, channel[good_idx])
    return out


def load_cinc2013_fqrs(record_id: str, cfg: Config) -> Optional[np.ndarray]:
    """Fetal R-peak sample indices from the `.fqrs` annotation file.

    Timing reference for evaluation (QRS F1, Bland-Altman R-R) -- never a model input,
    and never used as an amplitude target. Returns None if the file is missing so a
    record without references still loads.
    """
    record_path = Path(cfg.dataset.raw_dir) / record_id
    try:
        annotation = wfdb.rdann(str(record_path), cfg.dataset.fqrs_extension)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("%s: no %s annotations (%s)", record_id, cfg.dataset.fqrs_extension, exc)
        return None
    return np.asarray(annotation.sample, dtype=np.int64)
