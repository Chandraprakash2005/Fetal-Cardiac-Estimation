"""src/fcac/pinn/losses.py

Hybrid PINN loss: L = 0.7 * L_data + 0.3 * L_physics (CLAUDE.md section 2 "PINN";
weights in configs/default.yaml pinn.loss_weights).

L_data fits the scalar potential V at the four electrode coordinates to the observed
MEMD fetal-band signal there. L_physics penalises departure from Laplace's equation,
div grad V = 0, over the SPATIAL coordinates only, evaluated at interior collocation
points rather than at the electrodes (see src/fcac/pinn/coords.py -- a Laplacian is not
identifiable from four near-coplanar samples).

Implements: CLAUDE.md section 2 "PINN" loss.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from fcac.config import Config
from fcac.pinn.physics import laplacian


def data_loss(v_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L_data: MSE between V at the electrode coords and the observed fetal-band signal."""
    return nn.functional.mse_loss(v_pred, target)


def physics_loss(
    v: torch.Tensor, coords: torch.Tensor, cfg: Optional[Config] = None
) -> torch.Tensor:
    """L_physics: mean squared Laplace residual.

    `cfg` supplies cfg.pinn.spatial_dims; when omitted the Laplacian is taken over every
    coordinate column (used by the architecture-level tests).
    """
    dims = tuple(cfg.pinn.spatial_dims) if cfg is not None else None
    lap = laplacian(v, coords, dims=dims)
    return torch.mean(lap**2)


def hybrid_loss(
    v_pred: torch.Tensor,
    target: torch.Tensor,
    coords: torch.Tensor,
    cfg: Config,
    v_collocation: Optional[torch.Tensor] = None,
    collocation_coords: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """L = w_data * L_data + w_physics * L_physics.

    v_pred / coords                      : model output and inputs at the electrodes.
    v_collocation / collocation_coords   : model output and inputs at interior points.
        When given, the physics residual is evaluated there (the correct formulation --
        see module docstring). When omitted it falls back to the electrode coords, which
        keeps the simple `hybrid_loss(v, target, coords, cfg)` signature working.
    """
    l_data = data_loss(v_pred, target)

    if v_collocation is not None and collocation_coords is not None:
        l_physics = physics_loss(v_collocation, collocation_coords, cfg)
    else:
        l_physics = physics_loss(v_pred, coords, cfg)

    return cfg.pinn.loss_weights.data * l_data + cfg.pinn.loss_weights.physics * l_physics
