from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.model import PINNModel
from src.physics import dimensionless_pde_coefficients, effective_diffusion, normal_flux_scale, pde_residual
from src.utils import get_torch_dtype


class ConstantICModel(PINNModel):
    """返回常数 1 的测试模型，用于验证零残差。"""

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        one = z[:, 0:1] * 0.0 + 1.0
        return torch.cat([one, one], dim=1)

    def forward_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        return self.forward_normalized(z)


def test_effective_diffusion_and_dimensionless_coefficients() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    assert abs(effective_diffusion(config["physics"], "u12", "HF") - 10.0) < 1.0e-12
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    coeff = dimensionless_pde_coefficients(config["physics"], config, "u12", "HF", torch.device("cpu"), dtype)
    expected = 10.0 * (1000.0 * 86400.0) / (360.0 * 360.0)
    assert abs(coeff["kappa_x"].item() - expected) / expected < 1.0e-5
    assert coeff["residual_scale"].item() > 0.0
    assert torch.isfinite(coeff["residual_scale"]).item()


def test_pde_residual_shapes_and_constant_solution() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = ConstantICModel(config).to(dtype=dtype)
    xyt = torch.tensor([[220.0, 75.0, 1.0], [260.0, 90.0, 10.0]], dtype=dtype)
    r12, r13 = pde_residual(model, xyt, "SRV", config["physics"], config)
    assert r12.shape == (2, 1)
    assert r13.shape == (2, 1)
    assert torch.max(torch.abs(r12)).item() < 1.0e-6
    assert torch.max(torch.abs(r13)).item() < 1.0e-6


def test_normal_flux_scale_uses_interface_normal_direction() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    coeff = dimensionless_pde_coefficients(config["physics"], config, "u12", "HF", torch.device("cpu"), dtype)
    normal_x = torch.tensor([[1.0, 0.0], [-1.0, 0.0]], dtype=dtype)
    normal_y = torch.tensor([[0.0, 1.0], [0.0, -1.0]], dtype=dtype)

    scale_x = normal_flux_scale(config["physics"], config, "u12", "HF", normal_x)
    scale_y = normal_flux_scale(config["physics"], config, "u12", "HF", normal_y)
    expected_x = torch.maximum(torch.ones_like(scale_x), torch.abs(coeff["kappa_x"]).expand_as(scale_x))
    expected_y = torch.maximum(torch.ones_like(scale_y), torch.abs(coeff["kappa_y"]).expand_as(scale_y))

    assert torch.allclose(scale_x, expected_x)
    assert torch.allclose(scale_y, expected_y)
    assert torch.all(scale_y > scale_x)
