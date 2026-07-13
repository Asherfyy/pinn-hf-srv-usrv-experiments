from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.config import load_config
from src.geometry import ReservoirGeometry
from src.losses import compute_pde_loss
from src.physics import get_dimensionless_pde_coefficients
from src.utils import get_torch_dtype, save_loss_history


class ZeroResidualModel(torch.nn.Module):
    """输出恒零但仍依赖 xyt 的模型，用于验证零残差归一化后仍为零。"""

    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(()))

    def forward(self, xyt: torch.Tensor, detach_distance_features: bool = False) -> torch.Tensor:
        zero = xyt[:, 0:1] * 0.0 + self.dummy * 0.0
        return torch.cat([zero, zero], dim=1)


class QuadraticModel(torch.nn.Module):
    """构造非零二阶导的简单模型，使六个 PDE 分量产生可检测 loss。"""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, xyt: torch.Tensor, detach_distance_features: bool = False) -> torch.Tensor:
        x = xyt[:, 0:1] / 360.0
        y = xyt[:, 1:2] / 150.0
        t = xyt[:, 2:3] / 1000.0
        u12 = self.scale * (x * x + 0.1 * t)
        u13 = self.scale * (y * y + 0.2 * t)
        return torch.cat([u12, u13], dim=1)


def make_context() -> tuple[dict, ReservoirGeometry, torch.dtype]:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    return config, geometry, dtype


def make_pde_points(dtype: torch.dtype) -> dict[str, torch.Tensor]:
    """为 HF/SRV/USRV 各构造少量确定性 PDE 点，避免测试依赖随机采样。"""

    return {
        "hf": torch.tensor([[250.0, 75.0, 1.0], [330.0, 75.0, 10.0]], dtype=dtype),
        "srv": torch.tensor([[220.0, 60.0, 10.0], [300.0, 90.0, 100.0]], dtype=dtype),
        "usrv": torch.tensor([[40.0, 30.0, 10.0], [120.0, 120.0, 100.0]], dtype=dtype),
    }


def test_dimensionless_pde_scales_are_positive_and_finite() -> None:
    config, _geometry, dtype = make_context()
    for variable in ["u12", "u13"]:
        for region in ["HF", "SRV", "USRV"]:
            coeff_info = get_dimensionless_pde_coefficients(
                config["physics"],
                config,
                variable,
                region,
                device=torch.device("cpu"),
                dtype=dtype,
            )
            scale = coeff_info["residual_scale"]
            assert torch.isfinite(scale).item()
            assert scale.item() > 0.0


def test_zero_pde_residual_remains_zero_after_normalization() -> None:
    config, geometry, dtype = make_context()
    model = ZeroResidualModel().to(dtype=dtype)
    loss, diagnostics = compute_pde_loss(model, make_pde_points(dtype), geometry, config["physics"], config)
    assert loss.item() == 0.0
    for key, value in diagnostics.items():
        assert key.startswith(("loss_pde_", "rms_raw_", "rms_normalized_"))
        assert value.item() == 0.0


def test_pde_loss_is_split_into_six_weighted_components() -> None:
    config, geometry, dtype = make_context()
    config = copy.deepcopy(config)
    config["pde_residual_normalization"]["component_weights"].update(
        {
            "u12_HF": 1.0,
            "u13_HF": 2.0,
            "u12_SRV": 3.0,
            "u13_SRV": 4.0,
            "u12_USRV": 5.0,
            "u13_USRV": 6.0,
        }
    )
    model = QuadraticModel().to(dtype=dtype)
    loss, diagnostics = compute_pde_loss(model, make_pde_points(dtype), geometry, config["physics"], config)

    required = [
        "loss_pde_u12_hf",
        "loss_pde_u13_hf",
        "loss_pde_u12_srv",
        "loss_pde_u13_srv",
        "loss_pde_u12_usrv",
        "loss_pde_u13_usrv",
    ]
    for key in required:
        assert key in diagnostics
        assert torch.isfinite(diagnostics[key]).item()

    expected = (
        1.0 * diagnostics["loss_pde_u12_hf"]
        + 2.0 * diagnostics["loss_pde_u13_hf"]
        + 3.0 * diagnostics["loss_pde_u12_srv"]
        + 4.0 * diagnostics["loss_pde_u13_srv"]
        + 5.0 * diagnostics["loss_pde_u12_usrv"]
        + 6.0 * diagnostics["loss_pde_u13_usrv"]
    )
    assert torch.allclose(loss, expected, rtol=1.0e-6, atol=1.0e-8)


def test_loss_history_saves_dynamic_diagnostic_columns(tmp_path: Path) -> None:
    out_path = tmp_path / "loss_history.csv"
    rows = [
        {"epoch": 1.0, "loss_total": 1.0, "loss_pde_u12_hf": 0.1},
        {"epoch": 2.0, "loss_total": 0.5, "rms_normalized_u13_usrv": 0.2},
    ]
    save_loss_history(rows, out_path)
    text = out_path.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert header.startswith("epoch,")
    assert "loss_pde_u12_hf" in header
    assert "rms_normalized_u13_usrv" in header
