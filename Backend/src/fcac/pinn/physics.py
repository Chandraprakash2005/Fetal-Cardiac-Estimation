"""src/fcac/pinn/physics.py

Physics residual for the PINN: enforce Laplace's equation ∇²V = 0 (CLAUDE.md §2 "PINN")
via double autograd, treating the model's scalar potential output V(coords) as a
harmonic function of its input coordinates.

Implements: CLAUDE.md §2 "PINN" physics constraint.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


def laplacian(
    v: torch.Tensor, coords: torch.Tensor, dims: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """Compute the Laplacian of scalar field v with respect to `coords` via double
    autograd (sum of unmixed second partial derivatives).

    v      : (N, 1) -- model output; its graph must trace back to `coords`.
    coords : (N, D) -- input coordinates, requires_grad=True.
    dims   : which coordinate columns to differentiate over. None (default) uses all D.
             The fECG model passes the SPATIAL columns only (cfg.pinn.spatial_dims):
             Laplace's equation is spatial, and the quasi-static approximation is exactly
             the statement that time enters V as a parameter, not a differentiated
             coordinate -- summing d2V/dt2 into the residual would wrongly penalise the
             signal for varying in time at all.
    Returns: (N, 1) laplacian value per sample.
    """
    grad_v = torch.autograd.grad(
        v, coords, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True
    )[0]  # (N, D)

    if not grad_v.requires_grad:
        # grad_v has no residual graph dependency on coords at all (v is exactly
        # linear here, e.g. a fresh/degenerate network), so the second derivative --
        # and hence the Laplacian -- is exactly zero; there is nothing further to
        # differentiate.
        return torch.zeros_like(v)

    laplacian_sum = torch.zeros_like(v)
    for d in range(coords.shape[1]) if dims is None else dims:
        grad2 = torch.autograd.grad(
            grad_v[:, d],
            coords,
            grad_outputs=torch.ones_like(grad_v[:, d]),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]  # (N, D), or None if grad_v[:, d] has no dependency on coords (e.g. v is
        # locally linear, so its second derivative is genuinely zero there).
        if grad2 is not None:
            laplacian_sum = laplacian_sum + grad2[:, d : d + 1]

    return laplacian_sum
