from __future__ import annotations

import copy

import torch
import pytest

from src.config import load_config
from src.edfm_grid import build_edfm_grid
from src.geometry import ReservoirGeometry
from src.model import EPINNModel


def _config():
    config = copy.deepcopy(load_config("config/default.yaml"))
    config["grid"]["nx"] = 6
    config["grid"]["ny"] = 5
    return config


def test_epinn_output_shape_and_initial_parameters() -> None:
    config = _config()
    grid = build_edfm_grid(ReservoirGeometry(config["geometry"]), config)
    model = EPINNModel(grid.num_cells, torch.as_tensor(grid.edge_index), torch.as_tensor(grid.edge_weight), config)
    p = torch.ones((grid.num_cells, len(config["pressure"]["components"])), dtype=torch.float32)
    out = model(p)
    assert out.shape == p.shape
    assert torch.isfinite(out).all()
    assert float(model.anchor_weight.detach()) == pytest.approx(0.01)
    assert float(model.gate_weight.detach()) == pytest.approx(0.5)
    assert float(model.adaptive_alpha.detach()) == pytest.approx(0.1)
    assert model.edge_index.shape[0] == 2
    assert model.edge_weight.numel() == model.edge_index.shape[1]


def test_sparse_epinn_supports_more_than_dense_adjacency_limit() -> None:
    config = copy.deepcopy(load_config("config/default.yaml"))
    config["grid"]["nx"] = 80
    config["grid"]["ny"] = 70
    grid = build_edfm_grid(ReservoirGeometry(config["geometry"]), config)
    assert grid.num_cells > int(config["edfm"]["max_dense_elements"])
    assert grid.adjacency is None
    model = EPINNModel(grid.num_cells, torch.as_tensor(grid.edge_index), torch.as_tensor(grid.edge_weight), config)
    p = torch.ones((grid.num_cells, len(config["pressure"]["components"])), dtype=torch.float32)
    out = model(p)
    assert out.shape == p.shape
    assert torch.isfinite(out).all()
