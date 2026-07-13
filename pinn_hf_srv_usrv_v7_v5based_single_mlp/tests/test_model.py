from __future__ import annotations

import copy
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
    assert torch.max(torch.abs(pred[0:1] - torch.ones_like(pred[0:1]))).item() < 1.0e-12


def test_model_uses_single_shared_network() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    model = PINNModel(config)
    assert hasattr(model, "network")
    assert not hasattr(model, "subnets")
    assert model.network.net[0].in_features == int(config["model"]["network_input_dim"])
    assert hasattr(model, "geometry")
    assert not hasattr(model, "adf_dirichlet_torch")


def test_base_correction_uses_attached_previous_time_field() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    base_config = copy.deepcopy(config)
    base_config["model"]["constraint_mode"] = "ic_hard"
    base_model = PINNModel(base_config).to(dtype=dtype)
    with torch.no_grad():
        base_model.network.net[-1].bias[:] = torch.tensor([0.25, -0.10], dtype=dtype)
    model.attach_base_model(base_model)

    xyt = torch.tensor([[250.0, 75.0, 10.0], [300.0, 75.0, 100.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)
    expected = base_model.forward_normalized(model._base_z(z))
    pred = model(xyt)
    assert torch.max(torch.abs(pred - expected)).item() < 1.0e-12


def test_single_mlp_features_use_global_normalized_coordinates() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
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
    assert torch.allclose(features, z, atol=1.0e-12)
