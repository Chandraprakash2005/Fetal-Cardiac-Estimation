from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from fcac.config import Config


def expected_input_dim(cfg: Config) -> int:
    """3 spatial + 1 time + C observation-context dims."""
    return 4 + cfg.preprocessing.n_channels


def electrode_positions(cfg: Config) -> np.ndarray:
    positions = np.asarray([list(p) for p in cfg.pinn.electrodes], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"pinn.electrodes must be (C, 3), got shape {positions.shape}")
    if positions.shape[0] != cfg.preprocessing.n_channels:
        raise ValueError(
            f"pinn.electrodes has {positions.shape[0]} rows but preprocessing.n_channels "
            f"is {cfg.preprocessing.n_channels}"
        )
    return positions


def window_to_coords(
    window: np.ndarray, cfg: Config
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One (C, N) fetal-band window -> (coords (C*N, input_dim), target (C*N, 1)).

    Row ordering is channel-major -- channel 0's N samples, then channel 1's, ... -- so
    `readout` can reshape the network output straight back to (C, N).
    """
    positions = electrode_positions(cfg)  # (C, 3)
    n_channels, n_samples = window.shape
    input_dim = expected_input_dim(cfg)

    t_norm = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)  # (N,)
    context = window.T.astype(np.float32)  # (N, C) -- obs vector at each instant

    coords = np.empty((n_channels * n_samples, input_dim), dtype=np.float32)
    for c in range(n_channels):
        lo, hi = c * n_samples, (c + 1) * n_samples
        coords[lo:hi, :3] = positions[c]
        coords[lo:hi, 3] = t_norm
        coords[lo:hi, 4:] = context

    target = window.reshape(-1, 1).astype(np.float32)
    return torch.from_numpy(coords), torch.from_numpy(target)


def sample_collocation(
    window: np.ndarray,
    n_points: int,
    cfg: Config,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """(n_points, input_dim) interior points for the physics residual.

    Spatial coordinates are drawn uniformly from the electrode array's bounding box,
    inflated by cfg.pinn.collocation.box_margin so the domain has interior volume even
    though the electrodes are near-coplanar. Each point borrows a real time instant and
    its observation context from the window, so the physics is enforced on the same
    conditional field the data term sees -- not on an arbitrary one.
    """
    positions = electrode_positions(cfg)
    margin = cfg.pinn.collocation.box_margin
    low = torch.from_numpy(positions.min(axis=0) - margin)
    high = torch.from_numpy(positions.max(axis=0) + margin)

    n_samples = window.shape[1]
    idx = torch.randint(0, n_samples, (n_points,), generator=generator)

    spatial = low + torch.rand(n_points, 3, generator=generator) * (high - low)
    t = torch.from_numpy(np.linspace(0.0, 1.0, n_samples, dtype=np.float32))[idx]
    context = torch.from_numpy(window.T.astype(np.float32))[idx]  # (n_points, C)

    return torch.cat([spatial, t.unsqueeze(1), context], dim=1)


def readout(window: np.ndarray, v_pred: torch.Tensor, cfg: Config) -> np.ndarray:
    """Collapse the PINN output over one window into a single-channel fetal ECG (N,).

    window : (C, N) fetal-band input this V was fit to.
    v_pred : (C*N, 1) network output at the matching coords, channel-major.
    """
    n_channels, n_samples = window.shape
    v = v_pred.detach().cpu().numpy().reshape(n_channels, n_samples)

    mode = cfg.pinn.readout
    if mode == "residual":
        signal = window - v
    elif mode == "potential":
        signal = v
    else:
        raise ValueError(f"unknown pinn.readout '{mode}' (expected residual|potential)")

    return project_to_single_channel(signal)


def project_to_single_channel(signal: np.ndarray) -> np.ndarray:
    """(C, N) -> (N,) via the first principal component across channels.

    PC1 is the maximum-variance spatial mixture, which for a residual dominated by one
    localised source approximates that source's lead-field direction. A PCA component's
    sign is arbitrary, so it is fixed by making the largest-magnitude deflection
    positive -- otherwise R peaks would invert at random between windows and break
    downstream fiducial detection.
    """
    centred = signal - signal.mean(axis=1, keepdims=True)
    # SVD rather than an explicit covariance eigendecomposition: C is tiny and this
    # avoids squaring the condition number.
    _, _, vt = np.linalg.svd(centred.T, full_matrices=False)
    component = centred.T @ vt[0]  # (N,)

    if component[np.argmax(np.abs(component))] < 0:
        component = -component
    return component.astype(np.float32)
