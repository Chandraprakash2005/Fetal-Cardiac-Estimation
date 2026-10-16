"""src/fcac/cvae/augment.py

Latent-space augmentation: sample neighbours of a window's latent code and decode them
back to feature space, replacing the dropped WGAN (CLAUDE.md §2 "WGAN": unstable at
~90 recordings; augmentation comes from the CVAE latent space instead. Do not scaffold
a WGAN module).

Implements: CLAUDE.md §2 "CVAE" augment.py role.
"""

from __future__ import annotations

import torch

from fcac.cvae.model import ContrastiveVAE


@torch.no_grad()
def sample_latent_neighbours(
    model: ContrastiveVAE, z: torch.Tensor, n_samples: int, noise_std: float = 0.1
) -> torch.Tensor:
    """Sample `n_samples` latent-space neighbours per input latent and decode them.

    z : (N, latent_dim) -- e.g. encoder means for a batch of real windows.
    Returns (N, n_samples, input_dim) augmented feature vectors.
    """
    model.eval()
    n = z.shape[0]
    z_expanded = z.unsqueeze(1).expand(n, n_samples, -1)  # (N, n_samples, D)
    noise = torch.randn_like(z_expanded) * noise_std
    z_neighbours = (z_expanded + noise).reshape(n * n_samples, -1)
    x_aug = model.decoder(z_neighbours)
    return x_aug.reshape(n, n_samples, -1)
