from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.utils import dirichlet_target_hat, get_torch_dtype, pressure_hat_to_mpa, pressure_mpa_to_hat


def test_pressure_round_trip() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    p12 = torch.tensor([[24.7304361684], [12.0], [2.96765234021]], dtype=dtype)
    p13 = torch.tensor([[0.269563831592], [0.15], [0.0323476597911]], dtype=dtype)
    u12, u13 = pressure_mpa_to_hat(p12, p13, config["boundary"])
    restored = pressure_hat_to_mpa(torch.cat([u12, u13], dim=1), config["boundary"])
    assert torch.max(torch.abs(restored[:, 0:1] - p12)).item() < 1.0e-10
    assert torch.max(torch.abs(restored[:, 1:2] - p13)).item() < 1.0e-10


def test_dirichlet_target_scale() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    t0 = torch.zeros((4, 1), dtype=dtype)
    assert torch.max(torch.abs(dirichlet_target_hat(t0, config["boundary"]) - 1.0)).item() < 1.0e-12
    t_large = torch.full((4, 1), 1000.0, dtype=dtype)
    target = dirichlet_target_hat(t_large, config["boundary"])
    assert torch.max(torch.abs(target[:, 0:1] - target[:, 1:2])).item() < 1.0e-12
    assert torch.max(torch.abs(target)).item() < 1.0e-6
