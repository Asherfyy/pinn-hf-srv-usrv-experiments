from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.config import load_config
from src.geometry import ReservoirGeometry
from src.model import PINNModel
from src.utils import get_torch_dtype, set_seed


def test_dirichlet_hard_constraint_matches_boundary_value() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    set_seed(2026)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    model = PINNModel(geom, config).to(dtype=dtype)

    n = 32
    y = torch.linspace(74.995, 75.005, n, dtype=dtype).view(-1, 1)
    x = torch.full_like(y, 360.0)
    t = torch.linspace(0.0, 1000.0, n, dtype=dtype).view(-1, 1)
    xyt = torch.cat([x, y, t], dim=1)

    pred = model(xyt)
    target = model.boundary_value_hat(t)
    assert torch.max(torch.abs(pred - target)).item() < 1.0e-4


def test_tanh_adf_is_zero_on_dirichlet_and_recovers_locally() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")

    x_boundary = torch.tensor([[360.0]], dtype=dtype)
    y_boundary = torch.tensor([[75.0]], dtype=dtype)
    b_boundary = geom.adf_dirichlet_torch(x_boundary, y_boundary)
    assert torch.abs(b_boundary).item() < 1.0e-7

    x_1m = torch.tensor([[359.0]], dtype=dtype)
    b_1m = geom.adf_dirichlet_torch(x_1m, y_boundary)
    assert 0.5 < b_1m.item() < 0.9

    x_far = torch.tensor([[356.0]], dtype=dtype)
    b_far = geom.adf_dirichlet_torch(x_far, y_boundary)
    assert 0.99 < b_far.item() < 1.0


def test_dirichlet_distance_feature_still_uses_global_normalization() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")

    x = torch.tensor([[359.0]], dtype=dtype)
    y = torch.tensor([[75.0]], dtype=dtype)
    dist_feature = geom.distance_to_dirichlet_torch(x, y)
    assert abs(dist_feature.item() - 1.0 / geom.l_ref) < 1.0e-6


def test_ic_hard_constraint_matches_initial_value_at_t0() -> None:
    root = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(load_config(root / "config" / "default.yaml"))
    config["model"]["constraint_mode"] = "ic_hard"
    set_seed(2026)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    model = PINNModel(geom, config).to(dtype=dtype)

    x = torch.tensor([[220.0], [260.0], [20.0]], dtype=dtype)
    y = torch.tensor([[75.0], [90.0], [20.0]], dtype=dtype)
    t = torch.zeros_like(x)
    xyt = torch.cat([x, y, t], dim=1)

    pred = model(xyt)
    target = model.initial_value_hat(x, y)
    assert torch.max(torch.abs(pred - target)).item() < 1.0e-6


def test_direct_mode_returns_raw_network_pressure() -> None:
    root = Path(__file__).resolve().parents[1]
    config = copy.deepcopy(load_config(root / "config" / "default.yaml"))
    config["model"]["constraint_mode"] = "direct"
    set_seed(2026)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    model = PINNModel(geom, config).to(dtype=dtype)

    xyt = torch.tensor(
        [
            [220.0, 75.0, 0.0],
            [260.0, 90.0, 100.0],
            [20.0, 20.0, 1000.0],
        ],
        dtype=dtype,
    )

    pred = model(xyt)
    raw = model.forward_raw(xyt)
    assert torch.max(torch.abs(pred - raw)).item() < 1.0e-6
