"""src/fcac/io/types.py

Data contracts for every stage boundary in the pipeline. Shapes here are the source of
truth (CLAUDE.md §5) -- if a module needs a different shape, change the contract here
first, then update every producer/consumer.

All array fields are float32 numpy arrays unless noted. GA/CA are stored in days
everywhere internally; convert to weeks only at display time.

Implements: CLAUDE.md §8 item 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


@dataclass(frozen=True)
class Recording:
    """Raw, loaded-from-disk recording before any preprocessing.

    signal      : (C_raw, N_raw) -- abdominal channels only (never includes Direct_1
                  for ADFECGDB; scalp/Doppler reference lives in `fetal_ref` instead).
    fs_raw      : native sampling rate in Hz (1000 for ADFECGDB, 2048 for NInFEA).
    fetal_ref   : ground-truth / timing reference. ADFECGDB -> 1D scalp ECG array
                  (Direct_1), same length as signal's time axis. NInFEA -> 1D array of
                  fetal beat timestamps (seconds) derived from Doppler; no amplitude
                  reference exists for NInFEA (CLAUDE.md §2 "Ground truth").
    ga_days     : gestational age in days from LMP dating (external label, never a
                  model output).
    subject_id  : stable identifier for LOSOCV grouping (record id for ADFECGDB,
                  subject id for NInFEA where multiple recordings share a subject).
    """

    signal: np.ndarray
    fs_raw: float
    fetal_ref: Optional[np.ndarray]
    ga_days: Optional[float]
    subject_id: str


@dataclass(frozen=True)
class PreprocessedRecording:
    """Output of src/fcac/preprocessing/pipeline.py::preprocess().

    windows     : (W, 4, 5000) -- filtered, resampled, z-scored, windowed abdominal signal.
    fs          : always 1000 (Hz) by the time preprocessing is done.
    ref_windows : (W, 5000) matching scalp windows for ADFECGDB, or None for NInFEA
                  (which has no amplitude reference -- use fetal_ref timing instead).
    ga_days     : gestational age in days, carried through from Recording.
    subject_id  : carried through from Recording.
    """

    windows: np.ndarray
    fs: float
    ref_windows: Optional[np.ndarray]
    ga_days: Optional[float]
    subject_id: str


@dataclass(frozen=True)
class MEMDOutput:
    """Output of src/fcac/memd/decompose.py + select.py.

    imfs     : (W, K, 4, 5000) -- K intrinsic mode functions per window (K is
               data-determined, typically ~10-14).
    scores   : (W, K) -- per-IMF score (spectral coherence x in-band power in the
               fetal band, CLAUDE.md §2).
    selected : per-window list of IMF indices judged to carry fetal-band content.
               Length W; each element is a list[int] of length <= K.
    """

    imfs: np.ndarray
    scores: np.ndarray
    selected: List[List[int]]


@dataclass(frozen=True)
class FetalBandWindows:
    """Sum of each window's selected IMFs -- the PINN's input.

    x : (W, 4, 5000)
    """

    x: np.ndarray


@dataclass(frozen=True)
class PINNOutput:
    """Single-channel fetal ECG isolated by the PINN.

    fecg : (W, 5000)
    """

    fecg: np.ndarray


@dataclass(frozen=True)
class FeatureVector:
    """12 interpretable cardiac biomarkers per window.

    f : (W, 12) -- column order fixed by src/fcac/features/extract.py::FEATURE_NAMES.
    """

    f: np.ndarray


@dataclass(frozen=True)
class Latent:
    """CVAE latent code per window.

    z : (W, 64)
    """

    z: np.ndarray


@dataclass(frozen=True)
class CAPrediction:
    """Transformer output: predicted Cardiac Age.

    ca_days : (W,) per-window predictions, or (R,) per-recording aggregates
              (e.g. mean over a recording's windows) -- `aggregated` distinguishes them.
    """

    ca_days: np.ndarray
    aggregated: bool = False


FetalReference = Union[np.ndarray, None]
