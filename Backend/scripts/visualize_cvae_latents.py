#!/usr/bin/env python3
"""scripts/visualize_cvae_latents.py

Visual check that the trained CVAE (scripts/06_train_cvae.py) learned real structure
from real ADFECGDB biomarkers -- not a claim about GA prediction (GA is unavailable
for ADFECGDB, see plan.md item 1), just: does the encoder separate windows in a way
that isn't random?

Projects the 64-dim latent mean (mu, deterministic -- eval mode) down to 2D via PCA
(SVD, consistent with src/fcac/pinn/coords.py's project_to_single_channel) and colours
each point by its source recording. Two real, checkable things this shows:
  1. Windows from the same recording should cluster more than windows from different
     recordings, IF the CVAE learned per-subject structure (the InfoNCE positive rule
     here is same-subject-only, since GA is unavailable -- see scripts/06's docstring).
  2. The two records with a real, physiologically correct fetal extraction (r01, r08 --
     see reports/accuracy_adfecgdb.json) versus the three maternal-locked ones (r04,
     r07, r10) is a real, independently-known split; whether the latent space respects
     that split is an honest, unrehearsed check, not something tuned to look good.

Usage: python scripts/visualize_cvae_latents.py

Implements: ad-hoc diagnostic, not part of the numbered pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module

train_cvae_script = import_module("06_train_cvae")

from fcac.config import load_config
from fcac.cvae.model import ContrastiveVAE
from fcac.viz.plots import apply_style, save_figure

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Known-good vs. maternal-locked, from reports/accuracy_adfecgdb.json's quality gate --
# an independent fact about these recordings, not derived from the CVAE.
KNOWN_GOOD = {"r01", "r08"}


def pca_2d(z: np.ndarray) -> np.ndarray:
    """(N, D) -> (N, 2) via the top 2 principal components (SVD, no sklearn needed)."""
    centred = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


def main() -> None:
    cfg = load_config(dataset="adfecgdb")

    features, subject_ids, _ga_days = train_cvae_script.load_real_features({"adfecgdb": cfg})
    imputed, _report = train_cvae_script.impute_missing(features)

    model = ContrastiveVAE(cfg)
    model.load_state_dict(torch.load("data/processed/cvae/cvae.pt", map_location="cpu", weights_only=True))
    model.eval()

    with torch.no_grad():
        out = model(model.standardise(torch.from_numpy(imputed.astype(np.float32))))
    mu = out["mu"].numpy()  # (N, 64)
    logger.info("Encoded %d real windows -> (%d, %d) latent means", len(mu), *mu.shape)

    coords_2d = pca_2d(mu)

    apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, record_id in enumerate(sorted(set(subject_ids))):
        mask = subject_ids == record_id
        marker = "o" if record_id in KNOWN_GOOD else "x"
        label = f"{record_id}" + (" (real fetal HR)" if record_id in KNOWN_GOOD else " (maternal-locked)")
        ax.scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            label=label, color=colors[i], marker=marker, s=50, alpha=0.8,
        )

    ax.set_title(
        "CVAE latent space (PCA of mu), 153 real ADFECGDB windows\n"
        "same-subject-only positive pairing -- GA unavailable, see plan.md item 1"
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    path = save_figure(fig, "cvae_latent_space_adfecgdb")
    logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
