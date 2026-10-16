"""src/fcac/transformer/latents.py

Real-data glue between scripts/05_features.py and the transformer: loads per-record
window features, fills NaNs with the per-column medians scripts/06_train_cvae.py used
(always the TRAINING set's medians, also for zero-shot cohorts), and encodes them to
CVAE posterior-mean latents.

Implements: CLAUDE.md §5 FeatureVector -> Latent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from fcac.config import Config
from fcac.cvae.model import ContrastiveVAE
from fcac.features.extract import FEATURE_NAMES

FEATURES_DIR = Path("data/processed/features")
CVAE_CKPT = Path("data/processed/cvae/cvae.pt")
CVAE_REPORT = Path("reports/06_train_cvae.json")


def load_cvae_and_fill_values(cfg: Config) -> tuple[ContrastiveVAE, np.ndarray]:
    """Trained CVAE (eval mode) and the (12,) per-feature imputation fill values."""
    cvae = ContrastiveVAE(cfg)
    cvae.load_state_dict(torch.load(CVAE_CKPT, map_location="cpu"))
    cvae.eval()
    with open(CVAE_REPORT) as fh:
        imputation = json.load(fh)["imputation"]
    fill = np.array([imputation[n]["fill_value"] for n in FEATURE_NAMES], dtype=np.float32)
    return cvae, fill


def load_recording_latents(
    dataset: str,
    cvae: ContrastiveVAE,
    fill_values: np.ndarray,
    record_ids: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """record_id -> (W, latent_dim) CVAE posterior means. record_ids=None takes every
    features file for the dataset."""
    in_dir = FEATURES_DIR / dataset
    if record_ids is None:
        record_ids = sorted(p.stem for p in in_dir.glob("*.npz"))
    latents = {}
    for record_id in record_ids:
        path = in_dir / f"{record_id}.npz"
        if not path.exists():
            continue
        x = np.load(path, allow_pickle=True)["x"].astype(np.float32)
        x = np.where(np.isnan(x), fill_values, x).astype(np.float32)
        with torch.no_grad():
            latents[record_id] = cvae(cvae.standardise(torch.from_numpy(x)))["mu"]
    return latents


def load_pooled_recordings(
    cfgs: Dict[str, Config],
    cvae: ContrastiveVAE,
    fill_values: np.ndarray,
) -> List[dict]:
    """Every gated recording across `cfgs` (dataset name -> that dataset's Config) as
    {"key", "dataset", "record_id", "subject", "ga_days", "z"} dicts. `key` and
    `subject` are namespaced by dataset so ids never collide. A dataset's `ga_days` map
    gives GA (None -> recording skipped with a warning); NInFEA subjects come from
    `dataset.subject_of_record`, other datasets use one subject per record (CLAUDE.md §3)."""
    import logging

    logger = logging.getLogger(__name__)
    pooled = []
    for dataset, cfg in cfgs.items():
        ids = list(cfg.dataset.record_ids) if "record_ids" in cfg.dataset else None
        latents = load_recording_latents(dataset, cvae, fill_values, ids)
        for record_id, z in latents.items():
            ga = cfg.dataset.ga_days[str(record_id)] if str(record_id) in cfg.dataset.ga_days else None
            if ga is None:
                logger.warning("%s:%s has no GA label, skipping", dataset, record_id)
                continue
            if "subject_of_record" in cfg.dataset:
                subject = f"{dataset}:S{cfg.dataset.subject_of_record[str(record_id)]}"
            else:
                subject = f"{dataset}:{record_id}"
            pooled.append(
                {"key": f"{dataset}:{record_id}", "dataset": dataset, "record_id": str(record_id),
                 "subject": subject, "ga_days": float(ga), "z": z}
            )
    return pooled


def impute_missing(features: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-column median imputation from the real observed values in `features`.
    A column with zero real values is imputed with 0.0. Returns (imputed, report)."""
    imputed = features.copy()
    report = {}
    for j, name in enumerate(FEATURE_NAMES):
        column = features[:, j]
        missing = np.isnan(column)
        n_real = int((~missing).sum())
        fill_value = float(np.median(column[~missing])) if n_real > 0 else 0.0
        imputed[missing, j] = fill_value
        report[name] = {
            "n_real": n_real,
            "n_imputed": int(missing.sum()),
            "fill_value": fill_value,
        }
    return imputed, report


def load_recording_features(
    dataset: str, record_ids: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """record_id -> raw (W, 12) gated window features, NaNs intact (unlike
    `load_recording_latents`, which imputes and encodes)."""
    in_dir = FEATURES_DIR / dataset
    if record_ids is None:
        record_ids = sorted(p.stem for p in in_dir.glob("*.npz"))
    out = {}
    for record_id in record_ids:
        path = in_dir / f"{record_id}.npz"
        if path.exists():
            out[record_id] = np.load(path, allow_pickle=True)["x"].astype(np.float32)
    return out


@torch.no_grad()
def encode_features(cvae: ContrastiveVAE, x: np.ndarray, fill_values: np.ndarray) -> torch.Tensor:
    """(W, 12) raw features -> (W, latent_dim) CVAE posterior means, NaNs filled with
    `fill_values`."""
    cvae.eval()
    x = np.where(np.isnan(x), fill_values, x).astype(np.float32)
    return cvae(cvae.standardise(torch.from_numpy(x)))["mu"]
