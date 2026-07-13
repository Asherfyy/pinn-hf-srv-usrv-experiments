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
    assert torch.max(torch.abs(pred[0:1] - torch.ones_like(pred[0:1]))).item() < 1.0e-12


def test_model_has_no_onehot_distance_or_adf_logic() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    model = PINNModel(config)
    assert model.mlp.net[0].in_features == 3
    assert not hasattr(model, "geometry")
    assert not hasattr(model, "adf_dirichlet_torch")
