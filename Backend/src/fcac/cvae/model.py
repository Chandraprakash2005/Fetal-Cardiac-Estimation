"""src/fcac/cvae/model.py

Contrastive VAE (CLAUDE.md §2 "CVAE" -- Contrastive, not Conditional): encodes a
12-feature window vector to a 64-dim latent, trained with reconstruction + KL +
InfoNCE contrastive loss (cvae/losses.py). GA is never fed as input (circular).

The InfoNCE term operates on a separate projection of `z` (`p`), not `z` itself --
optimising the contrastive objective directly against the latent used for
reconstruction and downstream (CVAE -> Transformer) tasks distorts both; a small
projection head is the standard SimCLR-style fix and costs nothing else.

Implements: CLAUDE.md §2 "CVAE" architecture.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from fcac.config import Config


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU(), nn.Dropout(dropout)
    )


class Encoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        dims = [cfg.cvae.input_dim] + list(cfg.cvae.encoder_hidden)
        self.backbone = nn.Sequential(
            *[_mlp_block(a, b, cfg.cvae.dropout) for a, b in zip(dims[:-1], dims[1:])]
        )
        self.mu = nn.Linear(dims[-1], cfg.cvae.latent_dim)
        self.logvar = nn.Linear(dims[-1], cfg.cvae.latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.mu(h), self.logvar(h).clamp(-8.0, 8.0)


class Decoder(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        dims = [cfg.cvae.latent_dim] + list(cfg.cvae.decoder_hidden)
        blocks = [_mlp_block(a, b, cfg.cvae.dropout) for a, b in zip(dims[:-1], dims[1:])]
        self.net = nn.Sequential(*blocks, nn.Linear(dims[-1], cfg.cvae.input_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Projector(nn.Module):
    """Maps the VAE latent to the space the InfoNCE loss actually operates in."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.cvae.latent_dim, cfg.cvae.latent_dim),
            nn.GELU(),
            nn.Linear(cfg.cvae.latent_dim, cfg.cvae.proj_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1)


class ContrastiveVAE(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.projector = Projector(cfg)
        # Feature scaler (identity until train_cvae fits it); saved in the state dict, so a loaded
        # checkpoint standardises exactly like the model it was trained as.
        self.register_buffer("feat_mean", torch.zeros(cfg.cvae.input_dim))
        self.register_buffer("feat_std", torch.ones(cfg.cvae.input_dim))

    def standardise(self, x: torch.Tensor) -> torch.Tensor:
        """Raw (N, 12) features -> the space the encoder/decoder work in."""
        return (x - self.feat_mean) / self.feat_std

    def destandardise(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of `standardise` (e.g. to read x_hat back in raw feature units)."""
        return x * self.feat_std + self.feat_mean

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x : (N, 12) -> {x_hat (N,12), mu/logvar/z (N,64), p (N, proj_dim)}."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return {
            "x_hat": self.decoder(z),
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "p": self.projector(z),
        }
