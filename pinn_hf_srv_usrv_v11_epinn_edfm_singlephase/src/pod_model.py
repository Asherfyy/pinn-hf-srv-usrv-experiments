"""Small MLP used to predict POD modal coefficients from time features."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


class PODMLP(nn.Module):
    """Map two time/BHP features to a POD coefficient vector."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Iterable[int],
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_dims:
            hidden_dim = int(hidden)
            if hidden_dim <= 0:
                raise ValueError("PODMLP hidden dimensions must be positive.")
            layers.append(nn.Linear(previous, hidden_dim))
            layers.append(_activation(activation))
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            previous = hidden_dim
        layers.append(nn.Linear(previous, int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def _activation(name: str) -> nn.Module:
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU()
    if key == "tanh":
        return nn.Tanh()
    if key == "silu":
        return nn.SiLU()
    raise ValueError("PODMLP activation must be one of: relu, tanh, silu.")

