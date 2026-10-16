"""src/fcac/transformer/infer.py

Inference for the trained Cardiac Age transformer.

Implements: CLAUDE.md §5 Latent -> CAPrediction.
"""

from __future__ import annotations

import torch

from fcac.io.types import CAPrediction
from fcac.transformer.model import CardiacAgeTransformer


@torch.no_grad()
def infer(model: CardiacAgeTransformer, z_seq: torch.Tensor) -> CAPrediction:
    """z_seq : (N, L, latent_dim) -> per-sequence CA prediction (days)."""
    model.eval()
    ca_days = model(z_seq).cpu().numpy()
    return CAPrediction(ca_days=ca_days, aggregated=False)
