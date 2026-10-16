"""src/fcac/extraction.py

Stage-4 fetal-ECG extraction with a switchable method (`extraction.method` in default.yaml):

  pinn    maternal cancellation (mecg) -> frozen PINN. The original pipeline.
  hybrid  per window: the cancel+PINN output when it passes the quality gate, otherwise the
          FastICA BSS output (src/fcac/bss.py) of the preprocessed abdominal windows.

The choice is annotation-free (the quality gate only), so it is usable at inference. Measured
2026-10-02/03 (reports/engineering_log.md): hybrid is the best extractor on ADFECGDB (QRS F1
0.637, PINN in-sample), CinC 2013 (0.542, zero-shot) and NInFEA (right heart rate in 54% of
windows against 5% for cancel+PINN).

Implements: CLAUDE.md §5 stage "PINNOutput" (fecg (W, 5000)).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from fcac.bss import extract_fecg_bss
from fcac.config import Config
from fcac.mecg import adaptive_cancel_windows
from fcac.pinn.infer import extract_fecg
from fcac.pinn.model import PINN
from fcac.quality import assess_window


def extract_with_method(
    band_windows: np.ndarray,
    preprocessed_windows: np.ndarray,
    fs: float,
    cfg: Config,
    model: PINN,
    method: str | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-channel fECG for one record.

    band_windows         : (W, 4, N) MEMD fetal-band windows (PINN input).
    preprocessed_windows : (W, 4, N) preprocessed abdominal windows (BSS input).
    Returns (fecg (W, N), from_pinn (W,) bool, cancelled (W,) bool) -- `from_pinn` marks windows
    taken from the PINN path (all True for method "pinn"), `cancelled` the windows the maternal
    cancellation altered.
    """
    method = method or cfg.extraction.method
    if method not in ("pinn", "hybrid"):
        raise ValueError(f"unknown extraction.method {method!r}; use 'pinn' or 'hybrid'")
    if preprocessed_windows.shape[0] != band_windows.shape[0]:
        raise ValueError("band and preprocessed windows must have the same number of windows")

    cancelled_windows, cancelled = adaptive_cancel_windows(band_windows, fs, cfg)
    pinn_fecg = extract_fecg(model, cancelled_windows, cfg).fecg
    if method == "pinn":
        return pinn_fecg, np.ones(len(pinn_fecg), dtype=bool), cancelled

    from_pinn = np.array([assess_window(pinn_fecg[w], fs, cfg).acceptable for w in range(len(pinn_fecg))])
    if from_pinn.all():
        return pinn_fecg, from_pinn, cancelled
    bss_fecg, _ = extract_fecg_bss(preprocessed_windows, fs, cfg)
    return np.where(from_pinn[:, None], pinn_fecg, bss_fecg).astype(np.float32), from_pinn, cancelled
