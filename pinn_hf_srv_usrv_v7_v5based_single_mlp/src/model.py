"""Single-MLP PINN model for the HF/SRV/USRV problem."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .geometry import ReservoirGeometry


class MLP(nn.Module):
    """Fully connected network with Xavier initialization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str) -> None:
        super().__init__()
        if activation.lower() == "tanh":
            act_factory = nn.Tanh
        elif activation.lower() == "silu":
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
        # Start from the exact constant initial-pressure field.
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PINNModel(nn.Module):
    """One shared MLP over the whole domain.

    v7 intentionally removes the HF/SRV/USRV partitioned subnets from v5.
    Region names are still accepted by compatibility methods because the
    existing PDE and interface losses pass the material region separately for
    coefficients. The neural network itself is shared everywhere.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        if int(model_cfg["input_dim"]) != 3:
            raise ValueError("The public model input must be x_hat/y_hat/t_hat with dimension 3.")
        self.constraint_mode = str(model_cfg["constraint_mode"]).lower()
        if self.constraint_mode not in {"ic_hard", "ic_base_correction"}:
            raise ValueError("The v7 single-MLP model supports constraint_mode='ic_hard' or 'ic_base_correction'.")

        self.config = config
        self.geometry = ReservoirGeometry(config["geometry"])
        self.network_input_dim = int(model_cfg.get("network_input_dim", model_cfg["input_dim"]))
        if self.network_input_dim != 3:
            raise ValueError("v7 single-MLP uses only global normalized coordinates [x_hat,y_hat,t_hat].")
        self.base_time_lag_days = float(model_cfg.get("base_time_lag_days", 0.0))
        self.correction_envelope_power = float(model_cfg.get("correction_envelope_power", 1.0))
        self.__dict__["_base_model_ref"] = None

        self.network = MLP(
            self.network_input_dim,
            int(model_cfg["output_dim"]),
            int(model_cfg["hidden_layers"]),
            int(model_cfg["hidden_units"]),
            str(model_cfg["activation"]),
        )

    def attach_base_model(self, base_model: nn.Module | None) -> None:
        if base_model is None:
            self.__dict__["_base_model_ref"] = None
            return
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad_(False)
        self.__dict__["_base_model_ref"] = base_model

    def normalize_xyt(self, xyt: torch.Tensor) -> torch.Tensor:
        if xyt.ndim != 2 or xyt.shape[1] != 3:
            raise ValueError(f"xyt must have shape [N,3], got {tuple(xyt.shape)}.")
        geom = self.config["geometry"]
        sampler = self.config["sampler"]
        x_hat = (xyt[:, 0:1] - float(geom["x_min"])) / (float(geom["x_max"]) - float(geom["x_min"]))
        y_hat = (xyt[:, 1:2] - float(geom["y_min"])) / (float(geom["y_max"]) - float(geom["y_min"]))
        t_hat = (xyt[:, 2:3] - float(sampler["t_min"])) / (float(sampler["t_max"]) - float(sampler["t_min"]))
        return torch.cat([x_hat, y_hat, t_hat], dim=1)

    def denormalize_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != 3:
            raise ValueError(f"z must have shape [N,3], got {tuple(z.shape)}.")
        geom = self.config["geometry"]
        sampler = self.config["sampler"]
        x = float(geom["x_min"]) + z[:, 0:1] * (float(geom["x_max"]) - float(geom["x_min"]))
        y = float(geom["y_min"]) + z[:, 1:2] * (float(geom["y_max"]) - float(geom["y_min"]))
        t = float(sampler["t_min"]) + z[:, 2:3] * (float(sampler["t_max"]) - float(sampler["t_min"]))
        return x, y, t

    def features(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def features_for_region(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        _ = region_name
        return self.features(z)

    def _time_envelope(self, z: torch.Tensor) -> torch.Tensor:
        _x, _y, t = self.denormalize_z(z)
        decay = float(self.config["boundary"]["decay_rate"])
        return 1.0 - torch.exp(-decay * t)

    def _base_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.base_time_lag_days <= 0.0:
            return z
        x, y, t = self.denormalize_z(z)
        sampler = self.config["sampler"]
        t_min = float(sampler["t_min"])
        t_base = torch.clamp(t - float(self.base_time_lag_days), min=t_min)
        xyt_base = torch.cat([x, y, t_base], dim=1)
        return self.normalize_xyt(xyt_base)

    def _correction_envelope(self, z: torch.Tensor) -> torch.Tensor:
        beta = self._time_envelope(z)
        power = max(float(self.correction_envelope_power), 1.0e-12)
        return torch.clamp(beta, min=0.0) ** power

    def _base_field(self, z: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        base_model = self.__dict__.get("_base_model_ref")
        if base_model is None:
            return torch.ones_like(correction)
        z_base = self._base_z(z)
        base = base_model.forward_normalized(z_base)
        return base.to(device=correction.device, dtype=correction.dtype)

    def _apply_initial_condition(self, raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        beta = self._time_envelope(z)
        return 1.0 + beta * raw

    def _apply_base_correction(self, correction: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        base = self._base_field(z, correction)
        envelope = self._correction_envelope(z)
        return base + envelope * correction

    def forward_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        _ = region_name
        return self.forward_normalized(z)

    def forward_raw_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        _ = region_name
        return self.forward_raw_normalized(z)

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.network(self.features(z))
        if self.constraint_mode == "ic_hard":
            return self._apply_initial_condition(raw, z)
        return self._apply_base_correction(raw, z)

    def forward_raw_normalized(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(self.features(z))

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize_xyt(xyt))
