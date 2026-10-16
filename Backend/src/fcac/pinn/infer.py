"""src/fcac/pinn/infer.py

Frozen inference for the trained PINN (CLAUDE.md section 2 "PINN": train on ADFECGDB,
freeze, zero-shot on NInFEA, no fine-tuning ever).

Implements: CLAUDE.md section 5 FetalBandWindows -> PINNOutput (inference half).
"""

from __future__ import annotations

import numpy as np
import torch

from fcac.config import Config
from fcac.io.types import PINNOutput
from fcac.pinn.coords import readout, window_to_coords
from fcac.pinn.model import PINN


@torch.no_grad()
def infer(model: PINN, coords: torch.Tensor) -> torch.Tensor:
    """Run the frozen PINN forward pass.

    Callers must not call model.train() or update parameters afterwards -- CLAUDE.md
    section 2: no fine-tuning on NInFEA, ever.
    """
    model.eval()
    device = next(model.parameters()).device
    return model(coords.to(device))


def extract_fecg(model: PINN, windows: np.ndarray, cfg: Config) -> PINNOutput:
    """Isolate the fetal ECG from MEMD fetal-band windows.

    windows : (W, C, N) FetalBandWindows.x
    Returns : PINNOutput with .fecg (W, N)

    Note the readout is a residual by default (observed minus the harmonic potential),
    so it is computed OUTSIDE the no_grad forward pass in src/fcac/pinn/coords.py --
    see that module's header for the assumption this rests on.
    """
    model.eval()
    fecg = np.empty((windows.shape[0], windows.shape[2]), dtype=np.float32)

    for w in range(windows.shape[0]):
        window = windows[w]
        coords, _ = window_to_coords(window, cfg)
        v_pred = infer(model, coords)
        fecg[w] = readout(window, v_pred, cfg)

    return PINNOutput(fecg=fecg)
