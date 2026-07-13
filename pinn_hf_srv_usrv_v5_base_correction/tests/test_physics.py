from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.model import PINNModel
from src.physics import dimensionless_pde_coefficients, effective_diffusion, pde_residual
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
    assert abs(coeff["kappa_x"].item() - expected) / expected < 1.0e-12
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
    assert torch.max(torch.abs(r12)).item() < 1.0e-12
    assert torch.max(torch.abs(r13)).item() < 1.0e-12
