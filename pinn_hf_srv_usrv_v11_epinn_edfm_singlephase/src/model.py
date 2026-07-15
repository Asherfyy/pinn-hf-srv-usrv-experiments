"""Sparse E-PINN pressure-vector model for v11."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class EPINNModel(nn.Module):
    """Map normalized pressure at time t to normalized pressure at time t+dt.

    This sparse variant keeps the pressure-vector E-PINN idea but replaces the
    dense N x N adjacency-location block with local graph message passing over
    the EDFM/FVM connection graph.
    """

    def __init__(self, num_cells: int, edge_index: torch.Tensor, edge_weight: torch.Tensor, config: dict[str, Any]) -> None:
        super().__init__()
        self.num_cells = int(num_cells)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}.")
        if edge_weight.ndim != 1 or edge_weight.numel() != edge_index.shape[1]:
            raise ValueError("edge_weight must be one-dimensional and match edge_index.shape[1].")
        if edge_index.numel() > 0 and (int(torch.min(edge_index)) < 0 or int(torch.max(edge_index)) >= self.num_cells):
            raise ValueError("edge_index contains a cell index outside [0, num_cells).")

        model_cfg = config["model"]
        hidden_dim = int(model_cfg.get("hidden_dim", 64))
        output_dim = int(model_cfg.get("output_dim", len(config.get("pressure", {}).get("components", [1]))))
        message_passing_steps = int(model_cfg.get("message_passing_steps", 2))
        if hidden_dim <= 0:
            raise ValueError("model.hidden_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("model.output_dim must be positive.")
        if message_passing_steps <= 0:
            raise ValueError("model.message_passing_steps must be positive.")

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.message_passing_steps = message_passing_steps
        self.register_buffer("edge_index", edge_index.to(dtype=torch.long))
        self.register_buffer("edge_weight", edge_weight.to(dtype=torch.float32))
        self.input_layer = nn.Linear(output_dim, hidden_dim)
        self.batch_norms = nn.ModuleList(nn.BatchNorm1d(hidden_dim) for _ in range(message_passing_steps))
        self.hidden_layers = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(message_passing_steps))
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.cell_update_bias = nn.Parameter(torch.zeros(self.num_cells, output_dim, dtype=torch.float32))
        self.anchor_weight = nn.Parameter(torch.tensor(float(model_cfg["anchor_weight_init"]), dtype=torch.float32))
        self.gate_weight = nn.Parameter(torch.tensor(float(model_cfg["gate_weight_init"]), dtype=torch.float32))
        self.adaptive_alpha = nn.Parameter(torch.tensor(float(model_cfg["adaptive_alpha_init"]), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for layer in self.hidden_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        nn.init.zeros_(self.cell_update_bias)

    def forward(self, pressure_hat_t: torch.Tensor) -> torch.Tensor:
        if pressure_hat_t.ndim == 2 and pressure_hat_t.shape[0] == 1:
            pressure_hat_t = pressure_hat_t.view(-1)
        if pressure_hat_t.ndim == 1 and self.output_dim == 1:
            pressure_hat_t = pressure_hat_t.view(self.num_cells, 1)
        if pressure_hat_t.ndim != 2 or pressure_hat_t.shape != (self.num_cells, self.output_dim):
            raise ValueError(f"pressure_hat_t must have shape [{self.num_cells}, {self.output_dim}], got {tuple(pressure_hat_t.shape)}.")

        hidden = self.input_layer(pressure_hat_t)
        for batch_norm, layer in zip(self.batch_norms, self.hidden_layers):
            message = self._aggregate(hidden)
            anchored = hidden + self.anchor_weight.to(dtype=hidden.dtype, device=hidden.device) * message
            hidden = torch.relu(layer(batch_norm(anchored)) * self.adaptive_alpha.to(dtype=hidden.dtype, device=hidden.device))

        update = self.output_layer(hidden) + self.cell_update_bias.to(device=pressure_hat_t.device, dtype=pressure_hat_t.dtype)
        gate = torch.clamp(self.gate_weight.to(dtype=pressure_hat_t.dtype, device=pressure_hat_t.device), 0.0, 1.0)
        return pressure_hat_t + (1.0 - gate) * update

    def _aggregate(self, node_features: torch.Tensor) -> torch.Tensor:
        source = self.edge_index[0].to(device=node_features.device)
        target = self.edge_index[1].to(device=node_features.device)
        weight = self.edge_weight.to(device=node_features.device, dtype=node_features.dtype)
        out = torch.zeros_like(node_features)
        out.index_add_(0, target, node_features[source] * weight.unsqueeze(1))
        return out
