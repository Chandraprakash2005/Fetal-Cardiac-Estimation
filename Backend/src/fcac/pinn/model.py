"""src/fcac/pinn/model.py

Fully-connected network predicting a scalar potential V from input coordinates.
Architecture is LOCKED (CLAUDE.md §2 "PINN"): 5 layers x 512 units, Tanh on layers 1-3,
ReLU + Dropout(0.2) on layers 4-5.

TODO(bhuvanesh): the exact input coordinate scheme (spatial x,y,z + time? electrode
position + time? something else) is not specified in CLAUDE.md §2 or the spec excerpt
available here -- `input_dim`/`output_dim` are placeholders (configs/default.yaml
pinn.input_dim/output_dim) pending confirmation. Ask before training on real data.

Implements: CLAUDE.md §2 "PINN" architecture.
"""

from __future__ import annotations

import torch
from torch import nn

from fcac.config import Config

_ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


class PINN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        hidden_layers = list(cfg.pinn.hidden_layers)
        activations = list(cfg.pinn.activations)
        dropout = cfg.pinn.dropout

        if len(hidden_layers) != len(activations):
            raise ValueError(
                "pinn.hidden_layers and pinn.activations must be the same length"
            )

        layers = []
        in_dim = cfg.pinn.input_dim
        for units, act_name in zip(hidden_layers, activations):
            layers.append(nn.Linear(in_dim, units))
            layers.append(_ACTIVATIONS[act_name]())
            if act_name == "relu":
                layers.append(nn.Dropout(dropout))
            in_dim = units
        layers.append(nn.Linear(in_dim, cfg.pinn.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords : (N, input_dim) -> V : (N, output_dim)"""
        return self.net(coords)
