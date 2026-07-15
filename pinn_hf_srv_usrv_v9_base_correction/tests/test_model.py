from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.model import PINNModel
from src.utils import get_torch_dtype


def test_model_input_output_and_ic_hard() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    xyt = torch.tensor([[200.0, 75.0, 0.0], [250.0, 75.0, 10.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)
    assert z.shape[1] == 3
    pred = model(xyt)
    assert pred.shape == (2, 2)
    assert torch.max(torch.abs(pred[0:1] - torch.ones_like(pred[0:1]))).item() < 1.0e-6


def test_base_correction_applies_base_field_and_envelope() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)

    class ConstantBase(torch.nn.Module):
        def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
            return torch.full((z.shape[0], 2), 0.8, dtype=z.dtype, device=z.device)

    model.attach_base_model(ConstantBase().to(dtype=dtype))
    with torch.no_grad():
        model.subnets["HF"].net[-1].bias.copy_(torch.tensor([0.25, -0.10], dtype=dtype))

    xyt = torch.tensor([[250.0, 75.0, 10.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)
    envelope = model._correction_envelope(z)
    expected = torch.tensor([[0.8, 0.8]], dtype=dtype) + envelope * torch.tensor([[0.25, -0.10]], dtype=dtype)
    assert torch.allclose(model(xyt), expected)


def test_model_uses_partitioned_subnets_and_local_features() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    model = PINNModel(config)
    assert set(model.subnets.keys()) == {"HF", "SRV", "USRV"}
    assert model.subnets["HF"].net[0].in_features == int(config["model"]["subnet_input_dim"])
    assert hasattr(model, "geometry")
    assert not hasattr(model, "adf_dirichlet_torch")


def test_hf_local_features_fix_short_axis_coordinate() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["model"]["subnet_input_dim"] = 5
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)

    # Horizontal main fracture: aperture direction is y_local and should be
    # fixed even though physical PDE points still cover the full aperture.
    xyt = torch.tensor(
        [
            [250.0, 74.995, 10.0],
            [250.0, 75.000, 10.0],
            [250.0, 75.005, 10.0],
        ],
        dtype=dtype,
    )
    z = model.normalize_xyt(xyt)
    features = model.features_for_region(z, "HF")
    assert torch.allclose(features[:, 1], torch.full((3,), 0.5, dtype=dtype), atol=1.0e-6)

    # Vertical secondary fracture: use a point away from y=75.0 because the
    # secondary fracture intersects the horizontal main fracture there, and the
    # HF rectangle matcher assigns overlapping points to the main fracture first.
    xyt_vertical = torch.tensor(
        [
            [333.0545, 80.0, 10.0],
            [333.0595, 80.0, 10.0],
            [333.0645, 80.0, 10.0],
        ],
        dtype=dtype,
    )
    z_vertical = model.normalize_xyt(xyt_vertical)
    features_vertical = model.features_for_region(z_vertical, "HF")
    assert torch.allclose(features_vertical[:, 0], torch.full((3,), 0.5, dtype=dtype), atol=1.0e-6)
