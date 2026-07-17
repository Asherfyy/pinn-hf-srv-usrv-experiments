from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.model import PINNModel
from src.physics import dimensionless_pde_coefficients, effective_diffusion, line_pde_residual, pde_residual
from src.utils import get_torch_dtype


class ConstantICModel(PINNModel):
    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, 0:1] * 0.0 + 1.0

    def forward_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        return self.forward_normalized(z)


def test_effective_diffusion_and_dimensionless_coefficients() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    assert abs(effective_diffusion(config["physics"], "HF") - 10.0) < 1.0e-12
    assert effective_diffusion(config["physics"], "HF") > effective_diffusion(config["physics"], "SRV")
    assert effective_diffusion(config["physics"], "SRV") > effective_diffusion(config["physics"], "USRV")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    coeff = dimensionless_pde_coefficients(config["physics"], config, "HF", torch.device("cpu"), dtype)
    expected = 10.0 * (1000.0 * 86400.0) / (360.0 * 360.0)
    assert abs(coeff["kappa_x"].item() - expected) / expected < 1.0e-12
    assert torch.isfinite(coeff["residual_scale"]).item()


def test_pde_residual_shape_and_constant_solution() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = ConstantICModel(config).to(dtype=dtype)
    xyt = torch.tensor([[220.0, 75.0, 1.0], [260.0, 90.0, 10.0]], dtype=dtype)
    residual = pde_residual(model, xyt, "SRV", config["physics"], config)
    assert residual.shape == (2, 1)
    assert torch.max(torch.abs(residual)).item() < 1.0e-12


def test_line_fracture_pde_residual_accepts_tangents() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = ConstantICModel(config).to(dtype=dtype)
    xyt = torch.tensor([[220.0, 75.0, 1.0], [306.124, 90.0, 10.0]], dtype=dtype)
    tangent = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype)
    residual = line_pde_residual(model, xyt, tangent, config["physics"], config)
    assert residual.shape == (2, 1)
    assert torch.max(torch.abs(residual)).item() < 1.0e-12
