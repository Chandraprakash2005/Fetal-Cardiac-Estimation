"""src/fcac/cvae/losses.py

Contrastive VAE loss: reconstruction + KL + InfoNCE (CLAUDE.md §2 "CVAE", tau=0.07).
Positive pairs are same-subject windows, or windows within `ga_window_days` of each
other -- CLAUDE.md leaves "same-GA" undefined, and a hard equality match discards
almost every pair in practice (GA is effectively continuous per-window), so a
day-tolerance window is the only version of "same-GA" that produces any positives at
all. InfoNCE runs on the encoder's projection head output (`p`), not the VAE latent
(`z`) itself, so the contrastive objective doesn't fight the reconstruction /
downstream-task objective for the same embedding space.

TODO(tejasree): CLAUDE.md does not lock a relative weighting between recon and KL
(unlike the PINN's explicit 0.7/0.3) -- recon weight is implicitly 1.0 below, KL is
warmed up to `beta_kl` (0.1) over `kl_warmup_epochs`, InfoNCE weight is implicitly 1.0.
Flag to the supervisor if a different weighting is wanted.

Implements: CLAUDE.md §2 "CVAE" loss.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from fcac.config import Config


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def kl_warmup_beta(epoch: int, cfg: Config) -> float:
    """Linear KL warmup 0 -> cfg.cvae.beta_kl over cfg.cvae.kl_warmup_epochs, so the
    reconstruction term gets to shape the latent before the KL term starts pulling it
    toward the prior (avoids posterior collapse)."""
    warmup_epochs = max(1, cfg.cvae.kl_warmup_epochs)
    return cfg.cvae.beta_kl * min(1.0, epoch / warmup_epochs)


def info_nce_loss(
    p: torch.Tensor, positive_mask: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Multi-positive InfoNCE over a batch of projections.

    p             : (N, D), already L2-normalised (Projector normalises internally).
    positive_mask : (N, N) bool; positive_mask[i, j] True iff j is a positive for i;
                    diagonal is ignored regardless of its value.
    temperature   : tau (0.07, CLAUDE.md §2).
    """
    n = p.shape[0]
    sim = p @ p.T / temperature
    self_mask = torch.eye(n, dtype=torch.bool, device=p.device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    log_prob = sim - torch.logsumexp(sim, dim=-1, keepdim=True)

    positive_mask = positive_mask & ~self_mask
    n_positives = positive_mask.sum(dim=-1).clamp(min=1)
    # log_prob is -inf on the self-diagonal; log_prob * positive_mask would compute
    # -inf * False = NaN there (IEEE754) even though masked out, so use torch.where.
    masked_log_prob = torch.where(positive_mask, log_prob, torch.zeros_like(log_prob))
    loss_per_sample = -masked_log_prob.sum(dim=-1) / n_positives

    has_positive = positive_mask.sum(dim=-1) > 0
    if not torch.any(has_positive):
        return p.sum() * 0.0
    return loss_per_sample[has_positive].mean()


def hybrid_loss(
    outputs: dict,
    x: torch.Tensor,
    positive_mask: torch.Tensor,
    epoch: int,
    cfg: Config,
) -> torch.Tensor:
    """outputs is a ContrastiveVAE.forward() dict: {x_hat, mu, logvar, z, p}."""
    recon = reconstruction_loss(outputs["x_hat"], x)
    kl = kl_divergence(outputs["mu"], outputs["logvar"])
    contrastive = info_nce_loss(outputs["p"], positive_mask, cfg.cvae.contrastive.temperature)
    beta = kl_warmup_beta(epoch, cfg)
    return recon + beta * kl + contrastive
