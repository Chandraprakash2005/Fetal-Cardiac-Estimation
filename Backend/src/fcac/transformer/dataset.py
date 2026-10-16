"""src/fcac/transformer/dataset.py

Batching utilities for variable-length per-recording sequences of CVAE latents. Each
ADFECGDB/NInFEA recording produces a different number of 5 s windows (CLAUDE.md §5
Latent.z is (W, 64) per recording, W varies by recording length), so batches need
padding plus a mask, and long recordings are randomly cropped per epoch so every
window in a recording gets seen across training rather than only the first
`crop_len` of them.

Implements: CLAUDE.md §5 Latent -> Transformer input; CLAUDE.md §8 item 6.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
from torch.utils.data import Dataset


def pad_batch(sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """List of (T_i, D) -> padded (B, T_max, D) and bool mask (B, T_max), True = pad."""
    max_len = max(seq.shape[0] for seq in sequences)
    dim = sequences[0].shape[1]
    padded = torch.zeros(len(sequences), max_len, dim)
    pad_mask = torch.ones(len(sequences), max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        padded[i, : len(seq)] = seq
        pad_mask[i, : len(seq)] = False
    return padded, pad_mask


class RecordingDataset(Dataset):
    """One item = a random crop of one recording's window-latent sequence, so a long
    recording contributes different windows on different epochs instead of always the
    same first `crop_len`."""

    def __init__(
        self,
        z_by_recording: List[torch.Tensor],
        ca_days_target: torch.Tensor,
        crop_len: int,
        repeats_per_epoch: int = 1,
    ):
        self.z_by_recording = z_by_recording
        self.ca_days_target = ca_days_target
        self.crop_len = crop_len
        self.repeats_per_epoch = repeats_per_epoch

    def __len__(self) -> int:
        return len(self.z_by_recording) * self.repeats_per_epoch

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        recording_idx = index % len(self.z_by_recording)
        z = self.z_by_recording[recording_idx]
        if len(z) > self.crop_len:
            start = torch.randint(0, len(z) - self.crop_len + 1, (1,)).item()
            z = z[start : start + self.crop_len]
        return z, self.ca_days_target[recording_idx]


def collate(
    batch: List[Tuple[torch.Tensor, torch.Tensor]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z_padded, pad_mask = pad_batch([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])
    return z_padded, pad_mask, targets
