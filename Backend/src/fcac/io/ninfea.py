"""src/fcac/io/ninfea.py

Loads NInFEA WFDB recordings into the Recording contract (CLAUDE.md §5). Loading only --
stays at native 2048 Hz and returns all non-excluded abdominal channels; filtering,
resampling to 1000 Hz, kurtosis channel selection, normalisation, and windowing all
happen in src/fcac/preprocessing/, in the LOCKED order (CLAUDE.md §2 "Resampling order"):
filter at native rate -> resample -> normalise -> window, with kurtosis selection scored
on the filtered signal.

Ports the working logic from the pre-scaffold src/preprocessing/ninfea.py, minus the
filtering/downsampling/channel-selection it used to do inline (moved to preprocessing/,
with the order-of-operations bug fixed: the old code downsampled *before* filtering).

Implements: CLAUDE.md §3 "NInFEA", §5 Recording contract.
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


def load_ninfea_record(record_id: str, cfg: Config) -> Recording:
    """Load one NInFEA record's non-excluded abdominal channels at native fs.

    Channel selection here only drops the non-physiological channels
    (cfg.dataset.non_physiological_channels); the top-4-by-kurtosis selection happens
    later in preprocessing/channels.py, after filtering.
    """
    record = wfdb.rdrecord(str(Path(cfg.dataset.raw_dir) / record_id))

    prefix = cfg.dataset.abdominal_channel_prefix
    exclude = set(cfg.dataset.non_physiological_channels)
    abdo_indices = [
        i for i, name in enumerate(record.sig_name)
        if name.startswith(prefix) and name not in exclude
    ]

    signal = record.p_signal[:, abdo_indices].T.astype(np.float32)  # (C, N)
    fetal_ref = load_ninfea_doppler_beats(record_id, cfg)
    ga_days = cfg.dataset.ga_days[record_id] if record_id in cfg.dataset.ga_days else None

    return Recording(
        signal=signal,
        fs_raw=float(record.fs),
        fetal_ref=fetal_ref,
        ga_days=ga_days,
        subject_id=record_id,
    )


def load_ninfea_doppler_beats(record_id: str, cfg: Config) -> Optional[np.ndarray]:
    """Fetal beat times (seconds, ECG time base) derived from the pulsed-wave Doppler bitmaps.

    NInFEA has no scalp ECG (CLAUDE.md §2 "Ground truth") -- this is a timing-only
    reference for F1/R-R metrics in src/fcac/evaluation/extraction.py, never an
    amplitude target, and it is NOT part of PreprocessedRecording.ref_windows (which is
    always None for NInFEA). Doppler peaks lag the ECG R wave by an unknown constant offset
    (see src/fcac/io/ninfea_doppler.py).

    Reads the cache written by scripts/ninfea_doppler_beats.py; returns None (with a warning)
    when it has not been generated, so the abdominal-signal loading path stays usable alone.
    """
    cache = Path(cfg.dataset.doppler.cache_dir) / f"{record_id}.npz"
    if not cache.exists():
        logger.warning(
            "load_ninfea_doppler_beats(%s): no cache at %s -- run scripts/ninfea_doppler_beats.py",
            record_id, cache,
        )
        return None
    return np.load(cache)["beat_times_s"]
