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


def test_model_uses_partitioned_subnets_and_default_coordinate_features() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config)
    assert set(model.subnets.keys()) == {"HF", "SRV", "USRV"}
    assert int(config["model"]["subnet_input_dim"]) == 3
    assert model.subnets["HF"].net[0].in_features == 3
    assert hasattr(model, "geometry")
    assert not hasattr(model, "adf_dirichlet_torch")
    xyt = torch.tensor([[250.0, 75.0, 10.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)
    features = model.features_for_region(z, "HF")
    assert torch.allclose(features, z)


def test_optional_5d_hf_local_features_use_length_axis_for_thin_fractures() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["subnet_input_dim"] = 5
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
    assert features.shape[1] == 5
    assert torch.allclose(features[:, 1], torch.full((3,), 0.5, dtype=dtype), atol=1.0e-10)
