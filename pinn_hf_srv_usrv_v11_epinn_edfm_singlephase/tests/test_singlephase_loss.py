from __future__ import annotations

import copy

import torch

from src.config import load_config
from src.edfm_grid import build_edfm_grid
from src.geometry import ReservoirGeometry
from src.losses import apply_bhp_pressure, apply_bhp_pressure_to_cells, compute_step_loss, operators_from_grid, singlephase_residual
from src.model import EPINNModel
from src.utils import bhp_component_target_mpa, pressure_component_affine_parameters, pressure_mpa_to_hat


def _context():
    config = copy.deepcopy(load_config("config/default.yaml"))
    config["grid"]["nx"] = 6
    config["grid"]["ny"] = 5
    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    operators = operators_from_grid(grid, config, torch.device("cpu"), torch.float32)
    model = EPINNModel(grid.num_cells, torch.as_tensor(grid.edge_index), torch.as_tensor(grid.edge_weight), config)
    return config, grid, operators, model


def test_constant_pressure_has_zero_residual_without_boundary_change() -> None:
    config, grid, operators, _model = _context()
    initial = torch.as_tensor(pressure_component_affine_parameters(config)["initial_mpa"], dtype=torch.float32)
    pressure = initial.view(1, -1).repeat(grid.num_cells, 1)
    residual = singlephase_residual(pressure, pressure, operators, dt_days=1.0)
    assert float(torch.max(torch.abs(residual))) < 1.0e-6


def test_bhp_cell_is_hard_constrained_and_training_step_backpropagates() -> None:
    config, grid, operators, model = _context()
    initial = torch.as_tensor(pressure_component_affine_parameters(config)["initial_mpa"], dtype=torch.float32)
    pressure_prev = initial.view(1, -1).repeat(grid.num_cells, 1)
    pressure_hat = torch.as_tensor(pressure_mpa_to_hat(pressure_prev, config), dtype=torch.float32)
    bhp = torch.as_tensor(bhp_component_target_mpa(1000.0, config), dtype=torch.float32)
    loss, _diagnostics, pressure_pred = compute_step_loss(model, pressure_hat, pressure_prev, operators, config, bhp_mpa=bhp, dt_days=1.0)
    assert torch.allclose(pressure_pred[operators.well_cells], bhp.view(1, -1).repeat(operators.well_cells.numel(), 1))
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
    covered = apply_bhp_pressure(pressure_prev, grid.well_cell, 7.0)
    assert torch.allclose(covered[grid.well_cell], torch.full_like(covered[grid.well_cell], 7.0))
    target_many = torch.as_tensor([8.0, 0.08], dtype=torch.float32)
    covered_many = apply_bhp_pressure_to_cells(pressure_prev, operators.well_cells, target_many)
    assert torch.allclose(covered_many[operators.well_cells], target_many.view(1, -1).repeat(operators.well_cells.numel(), 1))
