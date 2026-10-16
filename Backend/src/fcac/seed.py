"""src/fcac/seed.py

Single place to fix every source of randomness. Every training script calls
set_seed(cfg.seed) before doing anything else (CLAUDE.md §6).

Implements: CLAUDE.md §11 (reproducibility).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices), and force
    deterministic torch algorithms where available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
