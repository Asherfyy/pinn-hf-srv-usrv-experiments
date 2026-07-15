"""I-PINN loss based on the RDFM finite-element residual."""

from __future__ import annotations

from typing import Any

import torch

from .rdfm_assembly import RdfmOperators


def apply_dirichlet_values(u: torch.Tensor, dirichlet_nodes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Hard-impose production Dirichlet values on a cloned nodal field."""

    if target.ndim == 1:
        target = target.view(1, -1)
    if target.shape[1] != u.shape[1]:
        raise ValueError(f"Dirichlet target has incompatible shape {tuple(target.shape)} for u {tuple(u.shape)}.")
    constrained = u.clone()
    if dirichlet_nodes.numel() > 0:
        constrained[dirichlet_nodes] = target.to(device=u.device, dtype=u.dtype)
    return constrained


def compute_ipinn_step_loss(
    model: torch.nn.Module,
    node_xy: torch.Tensor,
    operators: RdfmOperators,
    u_prev: torch.Tensor,
    free_nodes: torch.Tensor,
    dirichlet_nodes: torch.Tensor,
    dirichlet_target: torch.Tensor,
    dt_seconds: float,
    fracture_free_nodes: torch.Tensor | None = None,
    fracture_residual_weight: float = 0.0,
    main_fracture_free_nodes: torch.Tensor | None = None,
    main_fracture_residual_weight: float = 0.0,
    normalize_residual_by_row_sum: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Return loss, diagnostics, and hard-constrained current prediction."""

    if dt_seconds <= 0.0:
        raise ValueError(f"dt_seconds must be positive, got {dt_seconds:g}.")
    u_raw = model(node_xy)
    u_pred = apply_dirichlet_values(u_raw, dirichlet_nodes, dirichlet_target)
    residuals: list[torch.Tensor] = []
    diagnostics: dict[str, torch.Tensor] = {}
    for component, name in [(0, "u12"), (1, "u13")]:
        ops = operators.for_component(component)
        pred_col = u_pred[:, component : component + 1]
        prev_col = u_prev[:, component : component + 1]
        mass_delta = torch.sparse.mm(ops.mass, pred_col - prev_col) / float(dt_seconds)
        stiffness_term = torch.sparse.mm(ops.stiffness, pred_col)
        residual_raw = mass_delta + stiffness_term
        if normalize_residual_by_row_sum:
            row_scale = ops.mass_row_abs.view(-1, 1) / float(dt_seconds) + ops.stiffness_row_abs.view(-1, 1)
            residual = residual_raw / row_scale.clamp_min(1.0e-30)
        else:
            residual = residual_raw
        residual_free = residual[free_nodes]
        free_mae = torch.mean(torch.abs(residual_free))
        rms = torch.sqrt(torch.mean(residual_free**2).clamp_min(0.0))
        component_loss = free_mae
        diagnostics[f"loss_free_{name}"] = free_mae
        diagnostics[f"rms_{name}"] = rms
        diagnostics[f"max_abs_{name}"] = torch.max(torch.abs(residual_free))
        if fracture_free_nodes is not None and fracture_free_nodes.numel() > 0:
            residual_fracture = residual[fracture_free_nodes]
            fracture_mae = torch.mean(torch.abs(residual_fracture))
            diagnostics[f"loss_fracture_{name}"] = fracture_mae
            diagnostics[f"max_abs_fracture_{name}"] = torch.max(torch.abs(residual_fracture))
            if fracture_residual_weight > 0.0:
                component_loss = component_loss + float(fracture_residual_weight) * fracture_mae
        if main_fracture_free_nodes is not None and main_fracture_free_nodes.numel() > 0:
            residual_main_fracture = residual[main_fracture_free_nodes]
            main_fracture_mae = torch.mean(torch.abs(residual_main_fracture))
            diagnostics[f"loss_main_fracture_{name}"] = main_fracture_mae
            diagnostics[f"max_abs_main_fracture_{name}"] = torch.max(torch.abs(residual_main_fracture))
            if main_fracture_residual_weight > 0.0:
                component_loss = component_loss + float(main_fracture_residual_weight) * main_fracture_mae
        diagnostics[f"loss_{name}"] = component_loss
        residuals.append(component_loss)
    total = torch.stack(residuals).mean()
    diagnostics["loss_total"] = total
    return total, diagnostics, u_pred


def snapshot_to_pressure_diagnostics(u: torch.Tensor) -> dict[str, float]:
    detached = u.detach()
    return {
        "u_min": float(torch.min(detached).cpu()),
        "u_max": float(torch.max(detached).cpu()),
        "nonfinite_u_points": float(torch.sum(~torch.isfinite(detached)).cpu()),
    }
