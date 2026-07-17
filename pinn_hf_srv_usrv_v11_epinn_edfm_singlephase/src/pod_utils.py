"""Utilities shared by the POD-MLP reduced-order workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .utils import bhp_target_mpa, pressure_mpa_to_hat


POD_VERSION = "pod_mlp_v1"
FLATTEN_ORDER = "cell_major_component_last_C_order"
SNAPSHOT_PRESSURE_KEY = "pressure_mpa"
SNAPSHOT_TIME_KEY = "times_days"
REQUIRED_SNAPSHOT_KEYS = {
    "times_days",
    "pressure_mpa",
    "cell_xy",
    "matrix_cell_count",
    "well_cells",
    "component_names",
}


def get_pod_directories(config: dict[str, Any]) -> dict[str, Path]:
    """Return POD output directories and create them when needed."""

    dirs = {
        "outputs": Path(config["paths"]["outputs"]) / "pod",
        "checkpoints": Path(config["paths"]["checkpoints"]) / "pod",
        "figures": Path(config["paths"]["figures"]) / "pod",
        "logs": Path(config["paths"]["logs"]) / "pod",
        "tables": Path(config["paths"]["tables"]) / "pod",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def make_pod_snapshot_times(config: dict[str, Any]) -> np.ndarray:
    """Build the default POD snapshot time grid."""

    snapshot_cfg = config["pod"]["snapshot_generation"]
    early_end = float(snapshot_cfg["early_end_days"])
    middle_end = float(snapshot_cfg["middle_end_days"])
    final_time = float(snapshot_cfg["final_time_days"])
    early_points = int(snapshot_cfg["early_points"])
    middle_points = int(snapshot_cfg["middle_points"])
    late_points = int(snapshot_cfg["late_points"])
    if not (final_time > middle_end > early_end >= 0.0):
        raise ValueError("POD snapshot time limits must satisfy final_time_days > middle_end_days > early_end_days >= 0.")
    if min(early_points, middle_points, late_points) < 2:
        raise ValueError("POD snapshot point counts must each be at least 2.")

    early = np.expm1(np.linspace(np.log1p(0.0), np.log1p(early_end), early_points, dtype=np.float64))
    middle = np.linspace(early_end, middle_end, middle_points, dtype=np.float64)[1:]
    late = np.linspace(middle_end, final_time, late_points, dtype=np.float64)[1:]
    times = np.concatenate([early, middle, late])
    times = np.unique(np.round(times, decimals=12))
    if times.size == 0 or abs(float(times[0])) > 1.0e-12:
        raise ValueError("POD snapshot time grid must start at 0.")
    times[0] = 0.0
    if abs(float(times[-1]) - final_time) > 1.0e-10:
        raise ValueError("POD snapshot time grid must end at final_time_days.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("POD snapshot time grid must be strictly increasing.")
    return times.astype(np.float64)


def load_snapshot_archive(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate a snapshots.npz-compatible archive."""

    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Snapshot archive does not exist: {archive_path}")
    with np.load(archive_path, allow_pickle=True) as data:
        snapshots = {key: data[key] for key in data.files}

    missing = sorted(REQUIRED_SNAPSHOT_KEYS.difference(snapshots))
    if missing:
        raise KeyError(f"Snapshot archive missing required keys: {missing}")

    times = np.asarray(snapshots["times_days"], dtype=np.float64)
    pressure = np.asarray(snapshots["pressure_mpa"])
    cell_xy = np.asarray(snapshots["cell_xy"])
    component_names = as_string_list(snapshots["component_names"])
    if pressure.ndim != 3:
        raise ValueError(f"pressure_mpa must have shape [T, N, C], got {pressure.shape}.")
    if not np.all(np.isfinite(times)):
        raise ValueError("times_days contains NaN or Inf.")
    if not np.all(np.isfinite(pressure)):
        raise ValueError("pressure_mpa contains NaN or Inf.")
    if pressure.shape[0] != times.shape[0]:
        raise ValueError("pressure_mpa time dimension does not match times_days.")
    if pressure.shape[1] != cell_xy.shape[0]:
        raise ValueError("pressure_mpa cell dimension does not match cell_xy.")
    if pressure.shape[2] != len(component_names):
        raise ValueError("pressure_mpa component dimension does not match component_names.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_days must be strictly increasing.")
    return snapshots


def copy_snapshot_metadata(snapshots: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Copy all non-pressure and non-time metadata from a snapshot archive."""

    excluded = {SNAPSHOT_PRESSURE_KEY, SNAPSHOT_TIME_KEY}
    return {key: value for key, value in snapshots.items() if key not in excluded}


def build_normalized_free_state(
    pressure_mpa: np.ndarray,
    free_cells: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Normalize pressure on free cells and flatten as cell-major data."""

    pressure = np.asarray(pressure_mpa)
    cells = np.asarray(free_cells, dtype=np.int64)
    if pressure.ndim != 3:
        raise ValueError(f"pressure_mpa must have shape [T, N, C], got {pressure.shape}.")
    if np.any(cells < 0) or np.any(cells >= pressure.shape[1]):
        raise ValueError("free_cells contains indices outside the pressure cell range.")
    pressure_hat = pressure_mpa_to_hat(pressure[:, cells, :], config)
    pressure_hat = np.asarray(pressure_hat, dtype=np.float64)
    assert_finite("normalized free pressure", pressure_hat)
    return pressure_hat.reshape(pressure.shape[0], cells.size * pressure.shape[2], order="C")


def reshape_flat_free_state(flat_state: np.ndarray, free_cell_count: int, component_count: int) -> np.ndarray:
    """Reverse the POD flattening rule."""

    flat = np.asarray(flat_state)
    if flat.ndim == 1:
        flat = flat.reshape(1, -1)
    expected = int(free_cell_count) * int(component_count)
    if flat.shape[1] != expected:
        raise ValueError(f"Flat state has {flat.shape[1]} features, expected {expected}.")
    return flat.reshape(flat.shape[0], int(free_cell_count), int(component_count), order="C")


def build_time_features(times_days: np.ndarray, config: dict[str, Any], training_t_max: float) -> np.ndarray:
    """Build POD-MLP input features from time and total BHP."""

    times = np.asarray(times_days, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(times)):
        raise ValueError("times_days contains NaN or Inf.")
    if np.any(times < 0.0):
        raise ValueError("times_days must be non-negative.")
    t_max = float(training_t_max)
    if t_max <= 0.0:
        raise ValueError("training_t_max must be positive.")
    log_denom = max(float(np.log1p(t_max)), 1.0e-12)
    well_cfg = config["well"]
    p_initial = float(well_cfg["initial_pressure_mpa"])
    p_final = float(well_cfg["final_pressure_mpa"])
    p_scale = max(p_initial - p_final, 1.0e-12)
    bhp = np.asarray([bhp_target_mpa(float(time_value), well_cfg) for time_value in times], dtype=np.float64)
    features = np.empty((times.size, 2), dtype=np.float32)
    features[:, 0] = (np.log1p(times) / log_denom).astype(np.float32)
    features[:, 1] = ((bhp - p_final) / p_scale).astype(np.float32)
    assert_finite("time features", features)
    return features


def as_string_list(values: Any) -> list[str]:
    """Convert numpy string-like arrays to a plain list of Python strings."""

    array = np.asarray(values).reshape(-1)
    result: list[str] = []
    for value in array:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def assert_finite(name: str, values: np.ndarray) -> None:
    """Raise a clear error when an array contains NaN or Inf."""

    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        bad_count = int(np.sum(~np.isfinite(array)))
        raise ValueError(f"{name} contains {bad_count} NaN or Inf values.")


def validate_free_well_cells(total_cell_count: int, free_cells: np.ndarray, well_cells: np.ndarray) -> None:
    """Validate the free/well partition used by the default POD path."""

    free = np.asarray(free_cells, dtype=np.int64)
    well = np.asarray(well_cells, dtype=np.int64)
    if np.intersect1d(free, well).size:
        raise ValueError("free_cells and well_cells must not overlap when well cells are excluded from POD.")
    merged = np.union1d(free, well)
    expected = np.arange(int(total_cell_count), dtype=np.int64)
    if merged.shape != expected.shape or not np.array_equal(merged, expected):
        raise ValueError("free_cells and well_cells must cover every cell exactly once.")


def save_snapshot_archive(
    path: str | Path,
    source_metadata: dict[str, np.ndarray],
    times_days: np.ndarray,
    pressure_mpa: np.ndarray,
    solver: str,
    project_version: str = POD_VERSION,
) -> None:
    """Save a snapshots.npz-compatible archive."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pressure = np.asarray(pressure_mpa, dtype=np.float32)
    times = np.asarray(times_days, dtype=np.float64)
    assert_finite("snapshot pressure", pressure)
    payload = copy_snapshot_metadata(source_metadata)
    payload.update(
        {
            "times_days": times,
            "pressure_mpa": pressure,
            "solver": np.asarray(str(solver)),
            "project_version": np.asarray(str(project_version)),
        }
    )
    np.savez_compressed(out, **payload)

