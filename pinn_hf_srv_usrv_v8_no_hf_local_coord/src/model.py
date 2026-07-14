"""Partitioned MLP PINN model for HF/SRV/USRV regions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, Rect, ReservoirGeometry


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
        # Start from the exact constant initial-pressure field and avoid large
        # random PDE curvature before the boundary signal has been learned.
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PINNModel(nn.Module):
    """Three-subnet PINN: one MLP for HF, SRV, and USRV each.

    The public coordinate normalization remains z=[x_hat, y_hat, t_hat].
    Each region subnet receives local geometry-aware features:
    [x_local, y_local, x_hat, y_hat, t_hat]. HF PDE points are sampled in the
    full fracture rectangles, but the thin-aperture local feature is fixed at
    0.5 to avoid amplifying derivatives across 0.01 m apertures.
    """

    REGION_KEY_BY_ID = {
        REGION_HF: "HF",
        REGION_SRV: "SRV",
        REGION_USRV: "USRV",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        if int(model_cfg["input_dim"]) != 3:
            raise ValueError("The public model input must be x_hat/y_hat/t_hat with dimension 3.")
        if str(model_cfg["constraint_mode"]).lower() != "ic_hard":
            raise ValueError("This partitioned model currently supports constraint_mode='ic_hard'.")

        self.config = config
        self.geometry = ReservoirGeometry(config["geometry"])
        self.subnet_input_dim = int(model_cfg.get("subnet_input_dim", 5))
        if self.subnet_input_dim not in {3, 5}:
            raise ValueError("model.subnet_input_dim must be 3 or 5.")

        activation = str(model_cfg["activation"])
        output_dim = int(model_cfg["output_dim"])
        default_layers = int(model_cfg["hidden_layers"])
        default_units = int(model_cfg["hidden_units"])
        region_layers = model_cfg.get("region_hidden_layers", {})
        region_units = model_cfg.get("region_hidden_units", {})
        self.subnets = nn.ModuleDict(
            {
                region: MLP(
                    self.subnet_input_dim,
                    output_dim,
                    int(region_layers.get(region, default_layers)),
                    int(region_units.get(region, default_units)),
                    activation,
                )
                for region in ["HF", "SRV", "USRV"]
            }
        )

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

    @staticmethod
    def _inside_rect(x: torch.Tensor, y: torch.Tensor, rect: Rect) -> torch.Tensor:
        return (x >= rect.x_min) & (x <= rect.x_max) & (y >= rect.y_min) & (y <= rect.y_max)

    @staticmethod
    def _local_xy(x: torch.Tensor, y: torch.Tensor, rect: Rect) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(float(rect.width), 1.0e-12)
        height = max(float(rect.height), 1.0e-12)
        return (x - rect.x_min) / width, (y - rect.y_min) / height

    def _hf_local_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build HF local coordinates with the thin aperture coordinate fixed."""

        x_local = torch.zeros_like(x)
        y_local = torch.zeros_like(y)
        assigned = torch.zeros_like(x, dtype=torch.bool)
        for rect in self.geometry.hf_rects:
            mask = self._inside_rect(x, y, rect) & (~assigned)
            rx, ry = self._local_xy(x, y, rect)
            if rect.width >= rect.height:
                # Horizontal main fracture: keep length coordinate only.
                ry = torch.full_like(ry, 0.5)
            else:
                # Vertical secondary fracture: keep length coordinate only.
                rx = torch.full_like(rx, 0.5)
            x_local = torch.where(mask, rx, x_local)
            y_local = torch.where(mask, ry, y_local)
            assigned = assigned | mask
        fallback_x, fallback_y = self._local_xy(x, y, self.geometry.srv_bg)
        x_local = torch.where(assigned, x_local, fallback_x)
        y_local = torch.where(assigned, y_local, fallback_y)
        return x_local, y_local

    def features_for_region(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        if self.subnet_input_dim == 3:
            return z

        x, y, _t = self.denormalize_z(z)
        region = region_name.upper()
        if region == "HF":
            x_local, y_local = self._hf_local_xy(x, y)
        elif region == "SRV":
            x_local, y_local = self._local_xy(x, y, self.geometry.srv_bg)
        elif region == "USRV":
            x_local, y_local = self._local_xy(x, y, self.geometry.domain)
        else:
            raise ValueError(f"Unknown region: {region_name}")
        return torch.cat([x_local, y_local, z[:, 0:1], z[:, 1:2], z[:, 2:3]], dim=1)

    def _time_envelope(self, z: torch.Tensor) -> torch.Tensor:
        _x, _y, t = self.denormalize_z(z)
        decay = float(self.config["boundary"]["decay_rate"])
        return 1.0 - torch.exp(-decay * t)

    def _apply_initial_condition(self, raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        beta = self._time_envelope(z)
        return 1.0 + beta * raw

    def forward_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        region = region_name.upper()
        raw = self.subnets[region](self.features_for_region(z, region))
        return self._apply_initial_condition(raw, z)

    def forward_raw_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        region = region_name.upper()
        return self.subnets[region](self.features_for_region(z, region))

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        x, y, _t = self.denormalize_z(z)
        region_id = self.geometry.region_id_torch(x, y)
        outputs = torch.zeros((z.shape[0], int(self.config["model"]["output_dim"])), dtype=z.dtype, device=z.device)
        for region_id_value, region_name in self.REGION_KEY_BY_ID.items():
            mask = (region_id == int(region_id_value)).view(-1)
            if torch.any(mask):
                outputs[mask] = self.forward_region_normalized(z[mask], region_name)
        outside = (region_id < 0).view(-1)
        if torch.any(outside):
            outputs[outside] = self.forward_region_normalized(z[outside], "USRV")
        return outputs

    def forward_raw_normalized(self, z: torch.Tensor) -> torch.Tensor:
        x, y, _t = self.denormalize_z(z)
        region_id = self.geometry.region_id_torch(x, y)
        outputs = torch.zeros((z.shape[0], int(self.config["model"]["output_dim"])), dtype=z.dtype, device=z.device)
        for region_id_value, region_name in self.REGION_KEY_BY_ID.items():
            mask = (region_id == int(region_id_value)).view(-1)
            if torch.any(mask):
                outputs[mask] = self.forward_raw_region_normalized(z[mask], region_name)
        outside = (region_id < 0).view(-1)
        if torch.any(outside):
            outputs[outside] = self.forward_raw_region_normalized(z[outside], "USRV")
        return outputs

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize_xyt(xyt))
