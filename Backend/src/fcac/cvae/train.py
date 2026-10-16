"""src/fcac/cvae/train.py

Training loop for the Contrastive VAE (CLAUDE.md §2 "CVAE"). Uses AdamW with a cosine
learning-rate schedule and gradient clipping over mini-batches, rather than one
full-batch Adam step per epoch -- with as few as ~90 recordings' worth of windows,
mini-batching is what makes the InfoNCE term see different positive/negative
arrangements across epochs instead of the same single batch every time.

Implements: CLAUDE.md §8 item 6 (train/infer loops that run end-to-end on random tensors).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fcac.config import Config
from fcac.device import resolve_device
from fcac.cvae.losses import hybrid_loss
from fcac.cvae.model import ContrastiveVAE
from fcac.seed import set_seed

logger = logging.getLogger(__name__)


def build_positive_mask(
    subject_ids: np.ndarray, ga_days: np.ndarray, ga_window_days: float
) -> torch.Tensor:
    """positive_mask[i, j] = True iff window j is a positive for window i: same
    subject, OR GA within `ga_window_days` days of each other (CLAUDE.md §2 "CVAE"
    positive_rule).

    subject_ids : (N,) array of hashable ids.
    ga_days     : (N,) array of GA in days.
    """
    same_subject = subject_ids[:, None] == subject_ids[None, :]
    ga = torch.as_tensor(ga_days, dtype=torch.float32)
    close_ga = (ga[:, None] - ga[None, :]).abs() <= ga_window_days
    return torch.from_numpy(same_subject) | close_ga


def train_cvae(
    cfg: Config, features: torch.Tensor, subject_ids: np.ndarray, ga_days: np.ndarray
) -> ContrastiveVAE:
    """features : (N, 12) feature vectors (FeatureVector.f, CLAUDE.md §5)."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.cvae.device)
    model = ContrastiveVAE(cfg).to(device)
    if cfg.cvae.standardise_inputs:
        # Training-set statistics; a constant feature (std 0, e.g. an all-imputed column) keeps std 1.
        mean = features.mean(dim=0)
        std = features.std(dim=0, unbiased=False)
        model.feat_mean.copy_(mean)
        model.feat_std.copy_(torch.where(std > 1e-8, std, torch.ones_like(std)))
        features = model.standardise(features.to(device)).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.cvae.learning_rate, weight_decay=cfg.cvae.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.cvae.epochs)

    positive_mask_full = build_positive_mask(
        subject_ids, ga_days, cfg.cvae.contrastive.ga_window_days
    )
    batch_size = min(cfg.cvae.batch_size, len(features))
    dataset = TensorDataset(features, torch.arange(len(features)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(cfg.cvae.epochs):
        model.train()
        epoch_loss = 0.0
        for x_batch, idx_batch in loader:
            positive_mask = positive_mask_full[idx_batch][:, idx_batch].to(device)
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            loss = hybrid_loss(outputs, x_batch, positive_mask, epoch, cfg)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.cvae.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % max(1, cfg.cvae.epochs // 10) == 0:
            logger.info("epoch %d: loss=%.6f", epoch, epoch_loss / len(loader))

    return model.cpu()
