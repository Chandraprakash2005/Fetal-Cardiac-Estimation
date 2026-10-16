"""src/fcac/memd/decompose.py

Wraps the vendored Multivariate EMD implementation (src/fcac/memd/sift.py), with disk
caching -- MEMD costs ~1.5 s per (4, 5000) window, so a 5-minute ADFECGDB record is
minutes of compute and must never be recomputed silently.
Cache key: data/memd/{dataset}/{record_id}/win{idx}.npz (CLAUDE.md section 6 "Caching").

Implements: CLAUDE.md section 5 PreprocessedRecording -> MEMDOutput.imfs (scoring and
selection live in select.py).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from fcac.config import Config
from fcac.memd.sift import memd

logger = logging.getLogger(__name__)


def _cache_path(dataset: str, record_id: str, window_idx: int, cache_dir: Path) -> Path:
    return cache_dir / dataset / record_id / f"win{window_idx}.npz"


def _run_memd(window: np.ndarray, cfg: Config) -> np.ndarray:
    """Run MEMD on a single (4, 5000) window, returning (K, 4, 5000) IMFs.

    K is data-determined (~10-14 per CLAUDE.md section 2) -- `max_imfs` stays None so
    nothing pins it to a fixed count.
    """
    return memd(
        window,
        n_directions=cfg.memd.directions,
        stopping_tolerance=cfg.memd.stopping_tolerance,
        max_sift_iterations=cfg.memd.max_iterations,
        max_imfs=None,
        abstain_fraction=cfg.memd.abstain_fraction,
    )


def decompose_window(
    window: np.ndarray,
    dataset: str,
    record_id: str,
    window_idx: int,
    cfg: Config,
    cache_dir: Path = Path("data/memd"),
) -> np.ndarray:
    """Return (K, 4, 5000) IMFs for one window, using the on-disk cache if present.

    window : (4, 5000)
    """
    path = _cache_path(dataset, record_id, window_idx, cache_dir)
    if path.exists():
        return np.load(path)["imfs"]

    imfs = _run_memd(window, cfg)

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, imfs=imfs)
    return imfs


def decompose_recording(
    windows: np.ndarray,
    dataset: str,
    record_id: str,
    cfg: Config,
    cache_dir: Path = Path("data/memd"),
) -> List[np.ndarray]:
    """Return a list of length W of (K_w, 4, 5000) IMF arrays (K can vary per window)."""
    out = []
    for w in range(windows.shape[0]):
        imfs = decompose_window(windows[w], dataset, record_id, w, cfg, cache_dir)
        logger.debug("%s/%s window %d: K=%d", dataset, record_id, w, imfs.shape[0])
        out.append(imfs)
    return out
