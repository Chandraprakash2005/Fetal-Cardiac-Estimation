"""src/fcac/pinn/train.py

Training loop for the PINN. LOCKED (CLAUDE.md section 2 "PINN"): train on ADFECGDB only,
then freeze and run zero-shot on NInFEA -- never fine-tune on NInFEA (see pinn/infer.py).

Two entry points:
  * `train_pinn`            -- the low-level loop over an explicit (coords, target) pair.
  * `train_pinn_on_windows` -- the real one: minibatches sampled across every MEMD
                               fetal-band window of the training set.

Implements: CLAUDE.md section 5 FetalBandWindows -> PINNOutput (training half).
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np
import torch

from fcac.config import Config
from fcac.pinn.coords import expected_input_dim, sample_collocation, window_to_coords
from fcac.pinn.losses import hybrid_loss
from fcac.pinn.model import PINN
from fcac.seed import set_seed

logger = logging.getLogger(__name__)


def _resolve_device(device_setting: str) -> torch.device:
    """"auto" picks CUDA if available, else CPU. Otherwise passed straight to
    torch.device (e.g. "cpu", "cuda", "cuda:0")."""
    if device_setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_setting)


def _check_input_dim(cfg: Config) -> None:
    expected = expected_input_dim(cfg)
    if cfg.pinn.input_dim != expected:
        raise ValueError(
            f"pinn.input_dim is {cfg.pinn.input_dim} but the coordinate scheme in "
            f"src/fcac/pinn/coords.py produces {expected} "
            f"(3 spatial + 1 time + {cfg.preprocessing.n_channels} observation-context "
            "dims). Fix configs/default.yaml."
        )


def train_pinn(cfg: Config, coords: torch.Tensor, target: torch.Tensor) -> PINN:
    """Train a PINN on one explicit (coords, target) pair, full batch.

    coords : (N, cfg.pinn.input_dim) -- given requires_grad=True internally (the physics
             term differentiates through it).
    target : (N, cfg.pinn.output_dim)

    The physics residual is evaluated at `coords` here rather than at interior
    collocation points; `train_pinn_on_windows` is the version that does it properly.
    """
    set_seed(cfg.seed)
    device = _resolve_device(cfg.pinn.device)
    model = PINN(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pinn.learning_rate)

    coords = coords.clone().detach().to(device).requires_grad_(True)
    target = target.to(device)

    for epoch in range(cfg.pinn.epochs):
        optimizer.zero_grad()
        v_pred = model(coords)
        loss = hybrid_loss(v_pred, target, coords, cfg)
        loss.backward()
        optimizer.step()
        if epoch % max(1, cfg.pinn.epochs // 10) == 0:
            logger.info("epoch %d: loss=%.6f", epoch, loss.item())

    return model


def train_pinn_on_windows(
    cfg: Config,
    windows: np.ndarray,
    model: Optional[PINN] = None,
) -> PINN:
    """Train the PINN across every MEMD fetal-band window of the training set.

    windows : (W, C, N) -- FetalBandWindows.x for ADFECGDB, concatenated over records.

    One epoch = one pass over the windows in random order. Per window the data term uses
    a random subset of that window's C*N electrode/time points (cfg.pinn.batch_size
    counts *windows*, and cfg.pinn.points_per_window caps points per window -- a full
    (4, 5000) window is 20k points, which is wasteful to use in full every step), and
    the physics term uses freshly drawn interior collocation points.
    """
    _check_input_dim(cfg)
    set_seed(cfg.seed)
    device = _resolve_device(cfg.pinn.device)

    if model is None:
        model = PINN(cfg)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pinn.learning_rate)

    n_windows = windows.shape[0]
    points_per_window = cfg.pinn.points_per_window
    n_collocation = cfg.pinn.collocation.n_points
    rng = np.random.default_rng(cfg.seed)

    for epoch in range(cfg.pinn.epochs):
        order = rng.permutation(n_windows)
        epoch_loss = 0.0

        for w in order:
            window = windows[w]
            coords, target = window_to_coords(window, cfg)
            coords, target = coords.to(device), target.to(device)

            if points_per_window < coords.shape[0]:
                # Index with the plain numpy array rather than a torch tensor: a
                # torch index tensor must live on the same device as `coords`
                # (device-mismatch error on CUDA), a numpy array works either way.
                pick = rng.choice(coords.shape[0], points_per_window, replace=False)
                coords, target = coords[pick], target[pick]

            collocation = sample_collocation(window, n_collocation, cfg).to(device)
            collocation = collocation.clone().detach().requires_grad_(True)

            optimizer.zero_grad()
            v_pred = model(coords)
            v_collocation = model(collocation)
            loss = hybrid_loss(
                v_pred,
                target,
                coords,
                cfg,
                v_collocation=v_collocation,
                collocation_coords=collocation,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % max(1, cfg.pinn.epochs // 10) == 0:
            logger.info("epoch %d: mean loss=%.6f", epoch, epoch_loss / max(n_windows, 1))

    return model


def concat_fetal_band_windows(per_record: Sequence[np.ndarray]) -> np.ndarray:
    """Stack each record's (W_r, C, N) fetal-band windows into one (sum W_r, C, N)."""
    return np.concatenate(list(per_record), axis=0).astype(np.float32)
