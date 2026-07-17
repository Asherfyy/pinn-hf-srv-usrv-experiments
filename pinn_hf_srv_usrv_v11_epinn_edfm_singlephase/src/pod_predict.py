"""Predict full pressure fields at arbitrary times with POD-MLP."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch

from .config import load_config
from .pod_decomposition import PODBasis, load_pod_basis, reconstruct
from .pod_model import PODMLP
from .pod_utils import (
    POD_VERSION,
    as_string_list,
    assert_finite,
    build_time_features,
    get_pod_directories,
    load_snapshot_archive,
    reshape_flat_free_state,
    save_snapshot_archive,
)
from .utils import bhp_component_target_mpa, ensure_output_dirs, pressure_hat_to_mpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict pressure fields with a trained POD-MLP model.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--times", type=float, nargs="+", default=None)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    pod_dirs = get_pod_directories(config)
    files = config["pod"]["files"]
    basis_path = pod_dirs["outputs"] / str(files.get("basis_file", "pod_basis.npz"))
    checkpoint_path = pod_dirs["checkpoints"] / str(files.get("checkpoint_file", "pod_mlp.pt"))
    basis = load_pod_basis(basis_path)
    checkpoint = load_pod_checkpoint(checkpoint_path)
    source_snapshot = resolve_prediction_source(config, checkpoint, basis)
    source = load_snapshot_archive(source_snapshot)
    validate_prediction_inputs(basis, checkpoint, source)

    raw_times = default_prediction_times(config, basis) if args.times is None else np.asarray(args.times, dtype=np.float64)
    times = validate_prediction_times(
        raw_times,
        basis,
        allow_extrapolation=bool(args.allow_extrapolation or config["pod"]["inference"].get("allow_extrapolation", False)),
    )
    pressure = predict_pressure_mpa_from_loaded(
        times_days=times,
        config=config,
        basis=basis,
        checkpoint=checkpoint,
        source_snapshots=source,
        batch_size=int(args.batch_size),
        allow_extrapolation=True,
        clip_normalized_pressure=bool(config["pod"]["inference"].get("clip_normalized_pressure", False)),
    )
    output_name = args.output_name or str(files.get("prediction_file", "pod_predictions.npz"))
    output_path = pod_dirs["outputs"] / output_name
    save_prediction_archive(output_path, source, times, pressure)
    print(f"prediction count: {times.size}")
    print(f"output path: {output_path}")


def load_pod_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"POD-MLP checkpoint does not exist: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("POD-MLP checkpoint must contain a dictionary.")
    return checkpoint


def build_model_from_checkpoint(checkpoint: dict[str, Any]) -> PODMLP:
    model = PODMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint["output_dim"]),
        hidden_dims=[int(value) for value in checkpoint["hidden_dims"]],
        activation=str(checkpoint["activation"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device=torch.device("cpu"), dtype=torch.float32)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_coefficients(
    times_days: np.ndarray,
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    batch_size: int = 256,
) -> np.ndarray:
    model = build_model_from_checkpoint(checkpoint)
    features = build_time_features(np.asarray(times_days, dtype=np.float64), config, float(checkpoint["training_t_max"]))
    coefficient_mean = np.asarray(checkpoint["coefficient_mean"], dtype=np.float64)
    coefficient_std = np.asarray(checkpoint["coefficient_std"], dtype=np.float64)
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, features.shape[0], max(1, int(batch_size))):
            batch = torch.as_tensor(features[start : start + int(batch_size)], dtype=torch.float32)
            standardized = model(batch).detach().cpu().numpy().astype(np.float64)
            coefficients = standardized * coefficient_std.reshape(1, -1) + coefficient_mean.reshape(1, -1)
            outputs.append(coefficients)
    result = np.vstack(outputs) if outputs else np.empty((0, int(checkpoint["output_dim"])), dtype=np.float64)
    assert_finite("predicted POD coefficients", result)
    return result


def predict_pressure_mpa_from_loaded(
    times_days: np.ndarray,
    config: dict[str, Any],
    basis: PODBasis,
    checkpoint: dict[str, Any],
    source_snapshots: dict[str, np.ndarray],
    batch_size: int = 256,
    allow_extrapolation: bool = False,
    clip_normalized_pressure: bool = False,
) -> np.ndarray:
    times = validate_prediction_times(np.asarray(times_days, dtype=np.float64), basis, allow_extrapolation)
    coefficients = predict_coefficients(times, config, checkpoint, batch_size=batch_size)
    return pressure_from_coefficients(
        times_days=times,
        coefficients=coefficients,
        config=config,
        basis=basis,
        source_snapshots=source_snapshots,
        batch_size=batch_size,
        clip_normalized_pressure=clip_normalized_pressure,
    )


def pressure_from_coefficients(
    times_days: np.ndarray,
    coefficients: np.ndarray,
    config: dict[str, Any],
    basis: PODBasis,
    source_snapshots: dict[str, np.ndarray],
    batch_size: int = 256,
    clip_normalized_pressure: bool = False,
) -> np.ndarray:
    total_cell_count = int(np.asarray(source_snapshots["cell_xy"]).shape[0])
    component_count = len(basis.component_names)
    pressure_batches: list[np.ndarray] = []
    for start in range(0, coefficients.shape[0], max(1, int(batch_size))):
        coeff_batch = coefficients[start : start + int(batch_size)]
        times_batch = np.asarray(times_days[start : start + int(batch_size)], dtype=np.float64)
        free_hat_flat = reconstruct(coeff_batch, basis)
        if clip_normalized_pressure:
            free_hat_flat = np.clip(free_hat_flat, 0.0, 1.0)
        free_hat = reshape_flat_free_state(free_hat_flat, basis.free_cells.size, component_count)
        free_mpa = normalized_free_pressure_to_mpa(free_hat, config)
        full = assemble_full_pressure_mpa(
            times_days=times_batch,
            free_pressure_mpa=free_mpa,
            free_cells=basis.free_cells,
            well_cells=basis.well_cells,
            total_cell_count=total_cell_count,
            config=config,
        )
        pressure_batches.append(full)
    pressure = np.concatenate(pressure_batches, axis=0) if pressure_batches else np.empty((0, total_cell_count, component_count), dtype=np.float32)
    assert_finite("predicted pressure", pressure)
    return pressure.astype(np.float32)


def normalized_free_pressure_to_mpa(free_hat: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    pressure_hat = torch.as_tensor(np.asarray(free_hat, dtype=np.float32), dtype=torch.float32)
    pressure_mpa = pressure_hat_to_mpa(pressure_hat, config).detach().cpu().numpy()
    assert_finite("free pressure MPa", pressure_mpa)
    return pressure_mpa.astype(np.float32)


def assemble_full_pressure_mpa(
    times_days: np.ndarray,
    free_pressure_mpa: np.ndarray,
    free_cells: np.ndarray,
    well_cells: np.ndarray,
    total_cell_count: int,
    config: dict[str, Any],
) -> np.ndarray:
    times = np.asarray(times_days, dtype=np.float64).reshape(-1)
    free_pressure = np.asarray(free_pressure_mpa, dtype=np.float32)
    free = np.asarray(free_cells, dtype=np.int64)
    well = np.asarray(well_cells, dtype=np.int64).reshape(-1)
    if free_pressure.ndim != 3:
        raise ValueError("free_pressure_mpa must have shape [T, N_free, C].")
    if free_pressure.shape[0] != times.size:
        raise ValueError("free_pressure_mpa time dimension does not match times_days.")
    component_count = int(free_pressure.shape[2])
    full = np.full((times.size, int(total_cell_count), component_count), np.nan, dtype=np.float32)
    full[:, free, :] = free_pressure
    for row, time_value in enumerate(times):
        full[row, well, :] = np.asarray(bhp_component_target_mpa(float(time_value), config), dtype=np.float32).reshape(1, component_count)
    assert_finite("assembled full pressure", full)
    return full


def validate_prediction_inputs(
    basis: PODBasis,
    checkpoint: dict[str, Any],
    source_snapshots: dict[str, np.ndarray],
) -> None:
    if int(checkpoint["selected_rank"]) != int(basis.selected_rank):
        raise ValueError("POD basis rank does not match checkpoint rank.")
    if int(checkpoint["output_dim"]) != int(basis.selected_rank):
        raise ValueError("Checkpoint output_dim does not match selected rank.")
    if as_string_list(checkpoint["component_names"]) != list(basis.component_names):
        raise ValueError("Checkpoint component names do not match POD basis component names.")
    if as_string_list(source_snapshots["component_names"]) != list(basis.component_names):
        raise ValueError("Source snapshot component names do not match POD basis component names.")
    total_cell_count = int(np.asarray(source_snapshots["cell_xy"]).shape[0])
    if int(checkpoint["total_cell_count"]) != total_cell_count:
        raise ValueError("Checkpoint total cell count does not match source snapshots.")
    if int(checkpoint["free_cell_count"]) != int(basis.free_cells.size):
        raise ValueError("Checkpoint free cell count does not match POD basis.")
    expected_features = int(basis.free_cells.size) * len(basis.component_names)
    if basis.mean_state.size != expected_features or basis.basis.shape[0] != expected_features:
        raise ValueError("POD basis feature dimension is inconsistent with free cells and components.")
    source_well = np.asarray(source_snapshots["well_cells"], dtype=np.int64).reshape(-1)
    if not np.array_equal(np.sort(source_well), np.sort(basis.well_cells)):
        raise ValueError("Source snapshot well cells do not match POD basis.")
    if str(checkpoint.get("pod_version", "")) != POD_VERSION or str(basis.pod_version) != POD_VERSION:
        raise ValueError("POD version mismatch.")


def validate_prediction_times(times_days: np.ndarray, basis: PODBasis, allow_extrapolation: bool) -> np.ndarray:
    times = np.asarray(times_days, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("At least one prediction time is required.")
    if not np.all(np.isfinite(times)):
        raise ValueError("Prediction times contain NaN or Inf.")
    if np.any(times < 0.0):
        raise ValueError("Prediction times must be non-negative.")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Prediction times must be strictly increasing.")
    train_min = float(np.min(basis.source_times_days))
    train_max = float(basis.training_t_max)
    outside = np.any(times < train_min - 1.0e-12) or np.any(times > train_max + 1.0e-12)
    if outside and not allow_extrapolation:
        raise ValueError(f"Prediction times must be inside [{train_min:g}, {train_max:g}] days unless extrapolation is allowed.")
    if outside:
        print(f"Warning: extrapolating outside [{train_min:g}, {train_max:g}] days.")
    return times


def default_prediction_times(config: dict[str, Any], basis: PODBasis) -> np.ndarray:
    times = np.asarray(config["evaluation"].get("times", []), dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("No default prediction times found in evaluation.times.")
    if not np.all(np.isfinite(times)):
        raise ValueError("Default prediction times contain NaN or Inf.")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("Default prediction times must be strictly increasing.")

    train_min = float(np.min(basis.source_times_days))
    train_max = float(basis.training_t_max)
    inside = times[(times >= train_min - 1.0e-12) & (times <= train_max + 1.0e-12)]
    if inside.size == 0:
        raise ValueError(f"No evaluation.times entries fall inside POD training range [{train_min:g}, {train_max:g}] days.")
    if inside.size != times.size:
        dropped = times.size - inside.size
        print(f"Default prediction times clipped to POD training range [{train_min:g}, {train_max:g}] days; dropped {dropped} time(s).")
    if not np.isclose(inside[-1], train_max, rtol=1.0e-12, atol=1.0e-12) and train_max > inside[-1]:
        inside = np.append(inside, train_max)
    return inside.astype(np.float64)


def resolve_prediction_source(config: dict[str, Any], checkpoint: dict[str, Any], basis: PODBasis) -> Path:
    name = str(checkpoint.get("source_snapshot_file", basis.source_snapshot_file))
    path = Path(name)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Source snapshot file does not exist: {path}")
    candidates = [
        Path.cwd() / path,
        get_pod_directories(config)["outputs"] / path,
        Path(config["paths"]["outputs"]) / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not locate source snapshot file {name}. Searched: {searched}")


def save_prediction_archive(
    output_path: str | Path,
    source_snapshots: dict[str, np.ndarray],
    times_days: np.ndarray,
    pressure_mpa: np.ndarray,
) -> None:
    save_snapshot_archive(
        output_path,
        source_metadata=source_snapshots,
        times_days=times_days,
        pressure_mpa=pressure_mpa,
        solver="pod_mlp",
        project_version=POD_VERSION,
    )


if __name__ == "__main__":
    main()
