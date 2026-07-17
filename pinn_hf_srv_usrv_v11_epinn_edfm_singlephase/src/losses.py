"""Single-phase FVM residual loss for v11 E-PINN."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .edfm_grid import EdfmGrid, connection_transmissibility_matrix
from .model import EPINNModel
from .utils import pressure_hat_to_mpa


@dataclass(frozen=True)
class FvmOperators:
    conn_i: torch.Tensor
    conn_j: torch.Tensor
    ff_conn_i: torch.Tensor
    ff_conn_j: torch.Tensor
    ff_transmissibility: torch.Tensor
    transmissibility: torch.Tensor
    storage: torch.Tensor
    row_scale: torch.Tensor
    residual_weight: torch.Tensor
    well_cell: int
    well_cells: torch.Tensor
    free_cells: torch.Tensor


def operators_from_grid(grid: EdfmGrid, config: dict[str, Any], device: torch.device, dtype: torch.dtype) -> FvmOperators:
    conn_i = torch.as_tensor([conn.i for conn in grid.connections], dtype=torch.long, device=device)
    conn_j = torch.as_tensor([conn.j for conn in grid.connections], dtype=torch.long, device=device)
    component_count = len(config["pressure"]["components"])
    transmissibility = torch.as_tensor(connection_transmissibility_matrix(grid.connections, component_count), dtype=dtype, device=device)
    ff_indices = [idx for idx, conn in enumerate(grid.connections) if conn.kind == "ff"]
    ff_conn_i = torch.as_tensor([grid.connections[idx].i for idx in ff_indices], dtype=torch.long, device=device)
    ff_conn_j = torch.as_tensor([grid.connections[idx].j for idx in ff_indices], dtype=torch.long, device=device)
    ff_transmissibility = transmissibility[ff_indices] if ff_indices else torch.empty((0, component_count), dtype=dtype, device=device)
    storage = torch.as_tensor(grid.cell_storage, dtype=dtype, device=device)
    row_scale = torch.zeros((grid.num_cells, component_count), dtype=dtype, device=device)
    if transmissibility.numel() > 0:
        row_scale.index_add_(0, conn_i, transmissibility)
        row_scale.index_add_(0, conn_j, transmissibility)
    residual_weight_np = np.ones((grid.num_cells,), dtype=np.float64)
    residual_weight_np[grid.cell_region.astype(str) == "HF"] = float(config["training"].get("fracture_residual_weight", 1.0))
    residual_weight = torch.as_tensor(residual_weight_np, dtype=dtype, device=device)
    well_cells = torch.as_tensor(grid.well_cells, dtype=torch.long, device=device)
    all_cells = np.arange(grid.num_cells, dtype=np.int64)
    free_np = np.setdiff1d(all_cells, grid.well_cells, assume_unique=False)
    free_cells = torch.as_tensor(free_np, dtype=torch.long, device=device)
    return FvmOperators(
        conn_i=conn_i,
        conn_j=conn_j,
        ff_conn_i=ff_conn_i,
        ff_conn_j=ff_conn_j,
        ff_transmissibility=ff_transmissibility,
        transmissibility=transmissibility,
        storage=storage,
        row_scale=row_scale,
        residual_weight=residual_weight,
        well_cell=int(grid.well_cell),
        well_cells=well_cells,
        free_cells=free_cells,
    )


def apply_bhp_pressure(pressure_mpa: torch.Tensor, well_cell: int, bhp_mpa: float) -> torch.Tensor:
    constrained = pressure_mpa.clone()
    constrained[int(well_cell)] = torch.as_tensor(float(bhp_mpa), dtype=pressure_mpa.dtype, device=pressure_mpa.device)
    return constrained


def apply_bhp_pressure_to_cells(pressure_mpa: torch.Tensor, well_cells: torch.Tensor, bhp_mpa: float) -> torch.Tensor:
    constrained = pressure_mpa.clone()
    if well_cells.numel() > 0:
        target = torch.as_tensor(bhp_mpa, dtype=pressure_mpa.dtype, device=pressure_mpa.device)
        constrained[well_cells] = target
    return constrained


def singlephase_residual(
    pressure_pred_mpa: torch.Tensor,
    pressure_prev_mpa: torch.Tensor,
    operators: FvmOperators,
    dt_days: float,
) -> torch.Tensor:
    if dt_days <= 0.0:
        raise ValueError(f"dt_days must be positive, got {dt_days:g}.")
    squeezed = False
    if pressure_pred_mpa.ndim == 1:
        pressure_pred_mpa = pressure_pred_mpa.view(-1, 1)
        pressure_prev_mpa = pressure_prev_mpa.view(-1, 1)
        squeezed = True
    residual = operators.storage.view(-1, 1) * (pressure_pred_mpa - pressure_prev_mpa) / float(dt_days)
    if operators.transmissibility.numel() > 0:
        delta = pressure_pred_mpa[operators.conn_j] - pressure_pred_mpa[operators.conn_i]
        transmissibility = operators.transmissibility.to(device=delta.device, dtype=delta.dtype)
        if transmissibility.shape[1] != delta.shape[1]:
            raise ValueError(f"connection transmissibility component count {transmissibility.shape[1]} does not match pressure components {delta.shape[1]}.")
        flux = transmissibility * delta
        residual.index_add_(0, operators.conn_i, -flux)
        residual.index_add_(0, operators.conn_j, flux)
    return residual.view(-1) if squeezed else residual


def compute_step_loss(
    model: EPINNModel,
    pressure_prev_hat: torch.Tensor,
    pressure_prev_mpa: torch.Tensor,
    operators: FvmOperators,
    config: dict[str, Any],
    bhp_mpa: float,
    dt_days: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    pred_hat_raw = model(pressure_prev_hat)
    pred_mpa_raw = pressure_hat_to_mpa(pred_hat_raw, config)
    pred_mpa = apply_bhp_pressure_to_cells(pred_mpa_raw, operators.well_cells, bhp_mpa)
    residual = singlephase_residual(pred_mpa, pressure_prev_mpa, operators, dt_days)
    free_residual = residual[operators.free_cells]
    implicit_scale = operators.storage.view(-1, 1) / float(dt_days) + operators.row_scale.to(device=pred_mpa.device, dtype=pred_mpa.dtype)
    free_scale = implicit_scale[operators.free_cells].clamp_min(torch.finfo(implicit_scale.dtype).eps)
    free_residual_scaled = free_residual / free_scale
    free_weight = operators.residual_weight[operators.free_cells].view(-1, 1)
    loss = torch.sum(free_weight * free_residual_scaled**2) / (torch.sum(free_weight) * free_residual_scaled.shape[1]).clamp_min(torch.finfo(free_weight.dtype).eps)
    fracture_flux_loss = torch.zeros((), dtype=loss.dtype, device=loss.device)
    fracture_flux_weight = float(config["training"].get("fracture_flux_weight", 0.0))
    if fracture_flux_weight > 0.0 and operators.ff_transmissibility.numel() > 0:
        pressure_scale = float(config["well"]["initial_pressure_mpa"]) - float(config["well"]["final_pressure_mpa"])
        ff_delta = (pred_mpa[operators.ff_conn_i] - pred_mpa[operators.ff_conn_j]) / max(pressure_scale, 1.0e-12)
        ff_weight = operators.ff_transmissibility / operators.ff_transmissibility.mean().clamp_min(torch.finfo(operators.ff_transmissibility.dtype).eps)
        fracture_flux_loss = torch.mean(ff_weight * ff_delta**2)
        loss = loss + fracture_flux_weight * fracture_flux_loss
    diagnostics = {
        "loss_total": loss,
        "loss_fracture_flux": fracture_flux_loss,
        "residual_rms": torch.sqrt(torch.mean(free_residual**2).clamp_min(0.0)),
        "residual_scaled_rms": torch.sqrt(torch.mean(free_residual_scaled**2).clamp_min(0.0)),
        "residual_mae": torch.mean(torch.abs(free_residual)),
        "residual_max_abs": torch.max(torch.abs(free_residual)),
        "pressure_min_mpa": torch.min(pred_mpa),
        "pressure_max_mpa": torch.max(pred_mpa),
    }
    for component, name in enumerate(config["pressure"]["components"]):
        diagnostics[f"{name}_min_mpa"] = torch.min(pred_mpa[:, component])
        diagnostics[f"{name}_max_mpa"] = torch.max(pred_mpa[:, component])
        diagnostics[f"residual_scaled_rms_{name}"] = torch.sqrt(torch.mean(free_residual_scaled[:, component] ** 2).clamp_min(0.0))
    return loss, diagnostics, pred_mpa
