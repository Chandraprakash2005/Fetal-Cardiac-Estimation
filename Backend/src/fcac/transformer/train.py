"""src/fcac/transformer/train.py

Training for the Cardiac Age transformer: MAE loss, AdamW, warmup + cosine decay
schedule, gradient clipping (CLAUDE.md §2 "Transformer").

Two entry points:
- `train_transformer` takes one fixed-shape batch and trains for `cfg.transformer.epochs`
  steps -- the CLAUDE.md §8 item 6 "runs end-to-end on random tensors" path, used by
  `scripts/07_train_transformer.py` and unit tests.
- `train_transformer_on_recordings` takes a list of variable-length per-recording
  latent sequences and trains with real epoch x mini-batch iteration over
  `RecordingDataset` -- the path `evaluate.py`'s LOSOCV/zero-shot driver actually uses.

Implements: CLAUDE.md §8 item 6.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from fcac.config import Config
from fcac.device import resolve_device
from fcac.seed import set_seed
from fcac.transformer.dataset import RecordingDataset, collate
from fcac.transformer.model import CardiacAgeTransformer

logger = logging.getLogger(__name__)


def _warmup_cosine_lr(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train_transformer(
    cfg: Config,
    z_seq: torch.Tensor,
    ca_days_target: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> CardiacAgeTransformer:
    """z_seq : (N, L, latent_dim), ca_days_target : (N,), pad_mask : (N, L) bool or None."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.transformer.device)
    model = CardiacAgeTransformer(cfg).to(device)
    z_seq, ca_days_target = z_seq.to(device), ca_days_target.to(device)
    pad_mask = None if pad_mask is None else pad_mask.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.transformer.learning_rate,
        weight_decay=cfg.transformer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _warmup_cosine_lr(
            epoch, cfg.transformer.warmup_epochs, cfg.transformer.epochs
        ),
    )
    loss_fn = nn.L1Loss()  # MAE

    for epoch in range(cfg.transformer.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(z_seq, pad_mask), ca_days_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.transformer.grad_clip)
        optimizer.step()
        scheduler.step()
        if epoch % max(1, cfg.transformer.epochs // 10) == 0:
            logger.info("epoch %d: MAE=%.4f days", epoch, loss.item())

    return model.cpu()


def train_transformer_on_recordings(
    cfg: Config, z_by_recording: List[torch.Tensor], ca_days_target: torch.Tensor
) -> CardiacAgeTransformer:
    """Real training path for variable-length per-recording sequences: crops, pads
    and mini-batches recordings across `cfg.transformer.epochs`, unlike
    `train_transformer`, which trains on one fixed-shape batch.

    Standardises `ca_days_target` (real GA is ~266-301 days) before training and
    stores the mean/std on the returned model (`model.target_mean`/`target_std`) so
    `predict_recordings` can un-scale -- a freshly-initialised head starts near 0 and
    cannot close a ~280-day gap within a normal epoch budget at `learning_rate=1e-4`
    (see model.py docstring)."""
    set_seed(cfg.seed)
    device = resolve_device(cfg.transformer.device)
    model = CardiacAgeTransformer(cfg).to(device)
    target_mean = ca_days_target.mean()
    # unbiased=False: a single-recording training fold gives torch's default
    # (unbiased, N-1 denominator) std() a NaN with N=1, which would silently
    # NaN-poison every scaled target and every loss from then on. The population std
    # (unbiased=False) is exactly 0.0 for N=1, which clamp_min turns into 1.0 --
    # correctly falling back to "no scaling" rather than propagating a NaN.
    target_std = ca_days_target.std(unbiased=False).clamp_min(1.0)
    model.target_mean.fill_(target_mean)
    model.target_std.fill_(target_std)
    scaled_target = (ca_days_target - target_mean) / target_std

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.transformer.learning_rate,
        weight_decay=cfg.transformer.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _warmup_cosine_lr(
            epoch, cfg.transformer.warmup_epochs, cfg.transformer.epochs
        ),
    )
    loss_fn = nn.L1Loss()
    dataset = RecordingDataset(
        z_by_recording, scaled_target, cfg.transformer.crop_len, cfg.transformer.repeats_per_epoch
    )
    loader = DataLoader(
        dataset,
        batch_size=min(cfg.transformer.batch_size, len(dataset)),
        shuffle=True,
        collate_fn=collate,
    )

    last_loss = torch.tensor(float("nan"))
    for epoch in range(cfg.transformer.epochs):
        model.train()
        for z_batch, mask_batch, y_batch in loader:
            z_batch, mask_batch, y_batch = z_batch.to(device), mask_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            last_loss = loss_fn(model(z_batch, mask_batch), y_batch)
            last_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.transformer.grad_clip)
            optimizer.step()
        scheduler.step()
        if epoch % max(1, cfg.transformer.epochs // 10) == 0:
            logger.info("epoch %d: MAE(last batch)=%.4f days", epoch, last_loss.item())

    return model.cpu()
