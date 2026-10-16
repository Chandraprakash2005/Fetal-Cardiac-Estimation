"""src/fcac/device.py -- torch device selection shared by the trainers.

`auto` picks CUDA when available, otherwise CPU; any other value goes straight to torch.device
(e.g. "cpu", "cuda", "cuda:0"). Implements: CLAUDE.md §6 (settings come from Config).
"""

from __future__ import annotations

import torch


def resolve_device(setting: str) -> torch.device:
    if setting == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(setting)
