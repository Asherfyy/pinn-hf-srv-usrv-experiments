from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.utils import dirichlet_target_hat, get_torch_dtype, pressure_hat_to_mpa, pressure_mpa_to_hat


def test_pressure_round_trip_single_p12() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    pressure = torch.tensor([[25.0], [12.0], [3.0]], dtype=dtype)
    u = pressure_mpa_to_hat(pressure, config["boundary"])
    restored = pressure_hat_to_mpa(u, config["boundary"])
    assert torch.max(torch.abs(restored - pressure)).item() < 1.0e-10


def test_dirichlet_target_scale() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    t0 = torch.zeros((4, 1), dtype=dtype)
    assert torch.max(torch.abs(dirichlet_target_hat(t0, config["boundary"]) - 1.0)).item() < 1.0e-12
    t_large = torch.full((4, 1), 1000.0, dtype=dtype)
    target = dirichlet_target_hat(t_large, config["boundary"])
    assert target.shape == (4, 1)
    assert torch.max(torch.abs(target)).item() < 1.0e-6
