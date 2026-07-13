from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import torch

from src.config import load_config
from src.evaluate import load_trained_model
from src.geometry import ReservoirGeometry
from src.model import PINNModel
from src.train import load_optional_data_batch
from src.utils import (
    get_torch_dtype,
    pressure_hat_to_mpa,
    pressure_mpa_to_hat,
    pressure_normalization_mode,
)


def load_default() -> dict:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "config" / "default.yaml")


def test_component_affine_pressure_round_trip() -> None:
    config = load_default()
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    p12 = torch.tensor([[24.7304361684], [12.0], [2.96765234021]], dtype=dtype)
    p13 = torch.tensor([[0.269563831592], [0.15], [0.0323476597911]], dtype=dtype)
    u12, u13 = pressure_mpa_to_hat(p12, p13, config["boundary"])
    restored = pressure_hat_to_mpa(torch.cat([u12, u13], dim=1), config["boundary"])
    assert torch.max(torch.abs(restored[:, 0:1] - p12)).item() < 2.0e-6
    assert torch.max(torch.abs(restored[:, 1:2] - p13)).item() < 2.0e-6


def test_boundary_values_decay_equally_for_both_components() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_default()
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    model = PINNModel(geom, config).to(dtype=dtype)

    at_zero = model.boundary_value_hat(torch.zeros((3, 1), dtype=dtype))
    assert torch.max(torch.abs(at_zero - torch.ones_like(at_zero))).item() < 1.0e-6

    at_large = model.boundary_value_hat(torch.full((3, 1), 1000.0, dtype=dtype))
    assert torch.max(torch.abs(at_large[:, 0:1] - at_large[:, 1:2])).item() < 1.0e-6
    assert torch.max(torch.abs(at_large)).item() < 1.0e-6


def test_initial_component_values_have_similar_scale() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_default()
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    model = PINNModel(geom, config).to(dtype=dtype)

    x = torch.tensor([[220.0], [260.0], [20.0]], dtype=dtype)
    y = torch.tensor([[75.0], [90.0], [20.0]], dtype=dtype)
    initial = model.initial_value_hat(x, y)
    assert torch.max(torch.abs(initial - torch.ones_like(initial))).item() < 1.0e-5


def test_legacy_shared_mode_still_runs_but_is_not_default() -> None:
    config = load_default()
    assert pressure_normalization_mode(config["boundary"]) == "component_affine"

    legacy = copy.deepcopy(config["boundary"])
    legacy["pressure_normalization"] = "legacy_shared"
    p12 = torch.tensor([[24.7304361684]], dtype=torch.float32)
    p13 = torch.tensor([[0.269563831592]], dtype=torch.float32)
    u12, u13 = pressure_mpa_to_hat(p12, p13, legacy)
    restored = pressure_hat_to_mpa(torch.cat([u12, u13], dim=1), legacy)
    assert torch.max(torch.abs(restored[:, 0:1] - p12)).item() < 1.0e-6
    assert torch.max(torch.abs(restored[:, 1:2] - p13)).item() < 1.0e-6


def test_comsol_data_batch_uses_same_component_affine_conversion(tmp_path: Path) -> None:
    config = load_default()
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    config = copy.deepcopy(config)
    config["paths"]["data"] = str(tmp_path)
    config["loss_weights"]["data"] = 1.0
    df = pd.DataFrame(
        [
            {
                "x": 220.0,
                "y": 75.0,
                "t": 0.0,
                "Pg1": config["boundary"]["P1_t0"],
                "Pg2": config["boundary"]["P2_t0"],
            }
        ]
    )
    df.to_csv(tmp_path / "comsol_snapshots.csv", index=False)

    batch = load_optional_data_batch(config, torch.device("cpu"), dtype)
    assert batch is not None
    restored = pressure_hat_to_mpa(batch["target_hat"], config["boundary"])
    assert torch.max(torch.abs(restored[0, 0] - torch.tensor(config["boundary"]["P1_t0"], dtype=dtype))).item() < 2.0e-6
    assert torch.max(torch.abs(restored[0, 1] - torch.tensor(config["boundary"]["P2_t0"], dtype=dtype))).item() < 2.0e-6


def test_checkpoint_pressure_normalization_mismatch_raises(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_default()
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    model = PINNModel(geom, config).to(dtype=dtype)

    checkpoint_config = copy.deepcopy(config)
    checkpoint_config["boundary"]["pressure_normalization"] = "legacy_shared"
    checkpoint_path = tmp_path / "legacy_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": checkpoint_config}, checkpoint_path)

    with pytest.raises(ValueError, match="压力无量纲化模式"):
        load_trained_model(root / "config" / "default.yaml", checkpoint_path)
