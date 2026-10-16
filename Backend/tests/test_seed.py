"""tests/test_seed.py

Implements: CLAUDE.md §8 item 9 -- seed.py reproducibility.
"""

import numpy as np
import torch

from fcac.seed import set_seed


def test_set_seed_reproduces_numpy_and_torch():
    set_seed(123)
    a_np = np.random.rand(5)
    a_torch = torch.rand(5)

    set_seed(123)
    b_np = np.random.rand(5)
    b_torch = torch.rand(5)

    assert np.allclose(a_np, b_np)
    assert torch.allclose(a_torch, b_torch)
