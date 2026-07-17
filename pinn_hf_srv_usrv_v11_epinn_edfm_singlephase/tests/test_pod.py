from __future__ import annotations

import copy

import numpy as np
import torch

from src.config import load_config
from src.pod_decomposition import create_time_split, fit_pod_basis, reconstruct, project
from src.pod_model import PODMLP
from src.pod_predict import assemble_full_pressure_mpa, default_prediction_times, save_prediction_archive
from src.pod_utils import make_pod_snapshot_times
from src.utils import bhp_component_target_mpa


def _config() -> dict:
    return copy.deepcopy(load_config("config/default.yaml"))


def test_pod_time_grid() -> None:
    config = _config()
    times = make_pod_snapshot_times(config)
    assert times[0] == 0.0
    assert times[-1] == config["pod"]["snapshot_generation"]["final_time_days"]
    assert np.all(np.diff(times) > 0.0)
    assert times.size == 161


def test_pod_synthetic_low_rank_reconstruction() -> None:
    config = _config()
    true_rank = 3
    config["pod"]["decomposition"]["rank_mode"] = "fixed"
    config["pod"]["decomposition"]["fixed_rank"] = true_rank
    rng = np.random.default_rng(123)
    temporal = rng.normal(size=(20, true_rank))
    spatial_raw = rng.normal(size=(12, true_rank))
    spatial, _ = np.linalg.qr(spatial_raw)
    mean = rng.normal(size=(12,))
    states = mean.reshape(1, -1) + temporal @ spatial.T
    train = np.arange(states.shape[0], dtype=np.int64)
    empty = np.empty((0,), dtype=np.int64)
    basis = fit_pod_basis(
        states=states,
        config=config,
        free_cells=np.arange(6, dtype=np.int64),
        well_cells=np.empty((0,), dtype=np.int64),
        component_names=["P12", "P13"],
        train_indices=train,
        validation_indices=empty,
        test_indices=empty,
        feature_shape=(6, 2),
        source_times_days=np.arange(states.shape[0], dtype=np.float64),
        training_t_max=float(states.shape[0] - 1),
        source_snapshot_file="synthetic.npz",
    )
    coefficients = project(states, basis)
    reconstructed = reconstruct(coefficients, basis)
    error = np.linalg.norm(reconstructed - states) / max(np.linalg.norm(states), 1.0e-12)
    assert error < 1.0e-5


def test_pod_split() -> None:
    split_a = create_time_split(20, 0.15, 0.15, 2026, True)
    split_b = create_time_split(20, 0.15, 0.15, 2026, True)
    train, validation, test = split_a
    assert np.array_equal(train, split_b[0])
    assert np.array_equal(validation, split_b[1])
    assert np.array_equal(test, split_b[2])
    assert not (set(train.tolist()) & set(validation.tolist()))
    assert not (set(train.tolist()) & set(test.tolist()))
    assert not (set(validation.tolist()) & set(test.tolist()))
    assert sorted(set(train.tolist()) | set(validation.tolist()) | set(test.tolist())) == list(range(20))
    assert 0 in train
    assert 19 in train


def test_pod_basis_orthogonality() -> None:
    config = _config()
    config["pod"]["decomposition"]["rank_mode"] = "fixed"
    config["pod"]["decomposition"]["fixed_rank"] = 4
    rng = np.random.default_rng(456)
    states = rng.normal(size=(16, 18))
    train = np.arange(16, dtype=np.int64)
    empty = np.empty((0,), dtype=np.int64)
    basis = fit_pod_basis(
        states=states,
        config=config,
        free_cells=np.arange(9, dtype=np.int64),
        well_cells=np.empty((0,), dtype=np.int64),
        component_names=["P12", "P13"],
        train_indices=train,
        validation_indices=empty,
        test_indices=empty,
        feature_shape=(9, 2),
        source_times_days=np.arange(16, dtype=np.float64),
        training_t_max=15.0,
        source_snapshot_file="synthetic.npz",
    )
    error = np.linalg.norm(basis.basis.T @ basis.basis - np.eye(basis.selected_rank))
    assert error < 1.0e-4


def test_default_prediction_times_clip_to_pod_training_range() -> None:
    config = _config()
    config["pod"]["decomposition"]["rank_mode"] = "fixed"
    config["pod"]["decomposition"]["fixed_rank"] = 2
    states = np.eye(4, dtype=np.float64)
    train = np.arange(states.shape[0], dtype=np.int64)
    empty = np.empty((0,), dtype=np.int64)
    basis = fit_pod_basis(
        states=states,
        config=config,
        free_cells=np.arange(2, dtype=np.int64),
        well_cells=np.empty((0,), dtype=np.int64),
        component_names=["P12", "P13"],
        train_indices=train,
        validation_indices=empty,
        test_indices=empty,
        feature_shape=(2, 2),
        source_times_days=np.asarray([0.0, 1.0, 3.0, 999.0], dtype=np.float64),
        training_t_max=999.0,
        source_snapshot_file="synthetic.npz",
    )

    times = default_prediction_times(config, basis)

    assert times[0] == 0.0
    assert times[-1] == 999.0
    assert 1000.0 not in times


def test_pod_mlp_shapes() -> None:
    model = PODMLP(input_dim=2, output_dim=5, hidden_dims=[8, 8], activation="silu", dropout=0.0)
    output = model(torch.zeros((7, 2), dtype=torch.float32))
    assert output.shape == (7, 5)


def test_pod_bhp_hard_enforcement() -> None:
    config = _config()
    times = np.asarray([0.0, 2.5, 7.0], dtype=np.float64)
    free_cells = np.asarray([0, 2], dtype=np.int64)
    well_cells = np.asarray([1], dtype=np.int64)
    free_pressure = np.ones((times.size, free_cells.size, 2), dtype=np.float32)
    full = assemble_full_pressure_mpa(times, free_pressure, free_cells, well_cells, 3, config)
    for row, time_value in enumerate(times):
        target = bhp_component_target_mpa(float(time_value), config)
        assert np.max(np.abs(full[row, well_cells, :] - target.reshape(1, -1))) < 1.0e-6


def test_prediction_archive_schema(tmp_path) -> None:
    times = np.asarray([1.0, 2.0], dtype=np.float64)
    pressure = np.ones((2, 3, 2), dtype=np.float32)
    source = {
        "cell_xy": np.zeros((3, 2), dtype=np.float64),
        "matrix_cell_count": np.asarray(3, dtype=np.int64),
        "well_cells": np.asarray([2], dtype=np.int64),
        "component_names": np.asarray(["P12", "P13"]),
    }
    path = tmp_path / "predictions.npz"
    save_prediction_archive(path, source, times, pressure)
    with np.load(path, allow_pickle=True) as data:
        assert "times_days" in data.files
        assert "pressure_mpa" in data.files
        assert "cell_xy" in data.files
        assert "matrix_cell_count" in data.files
        assert "well_cells" in data.files
        assert "component_names" in data.files
        assert data["pressure_mpa"].shape == (2, 3, 2)
