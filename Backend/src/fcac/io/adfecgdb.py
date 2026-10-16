"""src/fcac/io/adfecgdb.py

Loads ADFECGDB EDF recordings into the Recording contract (CLAUDE.md §5). Loading only --
no filtering, resampling, or windowing here; that all happens in
src/fcac/preprocessing/pipeline.py so it is identical in shape for every dataset.

Ports the working logic from the pre-scaffold src/preprocessing/adfecgdb.py, minus the
filtering it used to do inline (moved to preprocessing/filters.py, with the fix that
ADFECGDB does NOT get the 50 Hz notch -- CLAUDE.md §2 "Notch").

Implements: CLAUDE.md §3 "ADFECGDB", §5 Recording contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyedflib
import wfdb

from fcac.config import Config
from fcac.io.types import Recording


def load_adfecgdb_record(record_id: str, cfg: Config) -> Recording:
    """Load one ADFECGDB record's abdominal channels + Direct_1 scalp reference.

    record_id must be one of cfg.dataset.record_ids (e.g. "r01").
    """
    edf_path = Path(cfg.dataset.raw_dir) / f"{record_id}.edf"

    reader = pyedflib.EdfReader(str(edf_path))
    try:
        channel_names = reader.getSignalLabels()
        fs_raw = reader.getSampleFrequency(0)
        n_samples = reader.getNSamples()[0]

        all_signals = np.zeros((reader.signals_in_file, n_samples), dtype=np.float32)
        for i in range(reader.signals_in_file):
            all_signals[i, :] = reader.readSignal(i)
    finally:
        reader.close()

    scalp_index = channel_names.index(cfg.dataset.scalp_channel)
    abdo_indices = [
        i for i, name in enumerate(channel_names)
        if name.startswith(cfg.dataset.abdominal_channel_prefix)
    ]
    if len(abdo_indices) != cfg.preprocessing.n_channels:
        raise ValueError(
            f"{record_id}: expected {cfg.preprocessing.n_channels} abdominal channels "
            f"with prefix '{cfg.dataset.abdominal_channel_prefix}', found {len(abdo_indices)}"
        )

    signal = all_signals[abdo_indices, :]
    fetal_ref = all_signals[scalp_index, :]
    ga_days = cfg.dataset.ga_days[record_id] if record_id in cfg.dataset.ga_days else None

    return Recording(
        signal=signal,
        fs_raw=float(fs_raw),
        fetal_ref=fetal_ref,
        ga_days=ga_days,
        subject_id=record_id,
    )


def load_adfecgdb_qrs_annotations(record_id: str, cfg: Config) -> np.ndarray:
    """Load fetal R-peak sample positions from the .qrs annotation file.

    Evaluation-only (QRS F1 in src/fcac/evaluation/extraction.py) -- never a model input.

    The annotation files ship as `<record>.edf.qrs`, so the wfdb extension is
    "edf.qrs", not "qrs" (wfdb splits the record name on the last dot). All beats carry
    the symbol "N"; verified against r01, which yields 644 beats over 300 s = 128.8 bpm,
    consistent with the Direct_1 scalp channel's measured rate.

    Returns: (n_beats,) int sample indices at the record's native 1000 Hz.
    """
    record_path = Path(cfg.dataset.raw_dir) / record_id
    annotation = wfdb.rdann(str(record_path), cfg.dataset.qrs_extension)
    return np.asarray(annotation.sample, dtype=np.int64)
