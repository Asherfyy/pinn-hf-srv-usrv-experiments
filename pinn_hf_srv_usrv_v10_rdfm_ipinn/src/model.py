"""Global MLP used by the v10 RDFM/I-PINN solver."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str) -> None:
        super().__init__()
        activation_name = activation.lower()
        if activation_name == "relu":
            act_factory: type[nn.Module] = nn.ReLU
        elif activation_name == "tanh":
            act_factory = nn.Tanh
        elif activation_name == "silu":
            act_factory = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_units)))
            layers.append(act_factory())
            in_dim = int(hidden_units)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        # Start exactly from the uniform initial pressure in dimensionless space.
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PINNModel(nn.Module):
    """Single global network: physical xy -> normalized pressure [u12, u13]."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        self.config = config
        self.net = MLP(
            input_dim=int(model_cfg["input_dim"]),
            output_dim=int(model_cfg["output_dim"]),
            hidden_layers=int(model_cfg["hidden_layers"]),
            hidden_units=int(model_cfg["hidden_units"]),
            activation=str(model_cfg["activation"]),
        )

    def normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must have shape [N, 2], got {tuple(xy.shape)}.")
        geom = self.config["geometry"]
        x_hat = (xy[:, 0:1] - float(geom["x_min"])) / (float(geom["x_max"]) - float(geom["x_min"]))
        y_hat = (xy[:, 1:2] - float(geom["y_min"])) / (float(geom["y_max"]) - float(geom["y_min"]))
        return torch.cat([x_hat, y_hat], dim=1)

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.net(z)
        return 1.0 + raw

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize_xy(xy))
