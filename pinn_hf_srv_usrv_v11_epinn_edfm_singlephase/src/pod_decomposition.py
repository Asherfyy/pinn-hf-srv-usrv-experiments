"""Snapshot POD decomposition utilities for the POD-MLP workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pod_utils import FLATTEN_ORDER, POD_VERSION, as_string_list, assert_finite


@dataclass
class PODBasis:
    mean_state: np.ndarray
    basis: np.ndarray
    singular_values: np.ndarray
    cumulative_energy: np.ndarray
    selected_rank: int
    retained_energy: float
    free_cells: np.ndarray
    well_cells: np.ndarray
    component_names: list[str]
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    feature_shape: np.ndarray
    flatten_order: str
    source_times_days: np.ndarray
    training_t_max: float
    source_snapshot_file: str
    pod_version: str = POD_VERSION


def create_time_split(
    num_snapshots: int,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    keep_endpoints_in_train: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a deterministic train/validation/test split over time indices."""

    count = int(num_snapshots)
    if count < 10:
        raise ValueError("At least 10 snapshots are required for a POD train/validation/test split.")
    validation_fraction = float(validation_fraction)
    test_fraction = float(test_fraction)
    if validation_fraction < 0.0 or validation_fraction >= 0.5:
        raise ValueError("validation_fraction must be in [0, 0.5).")
    if test_fraction < 0.0 or test_fraction >= 0.5:
        raise ValueError("test_fraction must be in [0, 0.5).")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("validation_fraction + test_fraction must be less than 0.8.")

    all_indices = np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    if keep_endpoints_in_train:
        interior = np.arange(1, count - 1, dtype=np.int64)
    else:
        interior = all_indices.copy()
    shuffled = interior.copy()
    rng.shuffle(shuffled)

    validation_count = _split_count(interior.size, validation_fraction)
    test_count = _split_count(interior.size, test_fraction)
    if validation_count + test_count > interior.size:
        raise ValueError("Validation and test splits exceed available interior snapshots.")
    validation = np.sort(shuffled[:validation_count])
    test = np.sort(shuffled[validation_count : validation_count + test_count])
    held_out = set(validation.tolist()) | set(test.tolist())
    train = np.asarray([idx for idx in all_indices.tolist() if idx not in held_out], dtype=np.int64)
    if keep_endpoints_in_train:
        if 0 not in set(train.tolist()) or count - 1 not in set(train.tolist()):
            raise RuntimeError("Endpoint snapshots were not kept in the training split.")
    _validate_split(count, train, validation, test)
    return np.sort(train), validation, test


def fit_pod_basis(
    states: np.ndarray,
    config: dict[str, Any],
    free_cells: np.ndarray,
    well_cells: np.ndarray,
    component_names: list[str],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    feature_shape: tuple[int, int],
    source_times_days: np.ndarray,
    training_t_max: float,
    source_snapshot_file: str,
) -> PODBasis:
    """Fit a POD basis from training snapshots only."""

    states_arr = np.asarray(states, dtype=np.float64)
    if states_arr.ndim != 2:
        raise ValueError(f"states must have shape [T, F], got {states_arr.shape}.")
    assert_finite("POD input states", states_arr)
    train = np.asarray(train_indices, dtype=np.int64)
    if train.size < 2:
        raise ValueError("At least two training snapshots are required for POD.")
    x_train = states_arr[train]
    mean_state = np.mean(x_train, axis=0)
    x_centered = x_train - mean_state
    gram = x_centered @ x_centered.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = np.sqrt(eigenvalues)
    if singular_values.size == 0 or float(singular_values[0]) <= 0.0:
        raise ValueError("POD training states have no nonzero variance.")

    decomp_cfg = config["pod"]["decomposition"]
    min_ratio = float(decomp_cfg.get("min_singular_ratio", 1.0e-12))
    keep = singular_values / max(float(singular_values[0]), 1.0e-30) >= min_ratio
    singular_values = singular_values[keep]
    eigenvectors = eigenvectors[:, keep]
    if singular_values.size == 0:
        raise ValueError("No POD modes remain after min_singular_ratio filtering.")

    modal_energy = singular_values**2
    total_energy = max(float(np.sum(modal_energy)), 1.0e-30)
    cumulative_energy = np.cumsum(modal_energy) / total_energy
    selected_rank = select_rank(cumulative_energy, decomp_cfg)

    basis_full = x_centered.T @ eigenvectors
    basis_full = basis_full / np.maximum(singular_values.reshape(1, -1), 1.0e-30)
    basis_selected = basis_full[:, :selected_rank]
    basis_q, _ = np.linalg.qr(basis_selected, mode="reduced")
    orthogonality_error = float(np.linalg.norm(basis_q.T @ basis_q - np.eye(selected_rank)))
    if orthogonality_error > 1.0e-4:
        raise ValueError(f"POD basis orthogonality check failed: {orthogonality_error:.3e}")
    assert_finite("POD basis", basis_q)

    return PODBasis(
        mean_state=mean_state.astype(np.float64),
        basis=basis_q.astype(np.float64),
        singular_values=singular_values.astype(np.float64),
        cumulative_energy=cumulative_energy.astype(np.float64),
        selected_rank=int(selected_rank),
        retained_energy=float(cumulative_energy[selected_rank - 1]),
        free_cells=np.asarray(free_cells, dtype=np.int64),
        well_cells=np.asarray(well_cells, dtype=np.int64),
        component_names=list(component_names),
        train_indices=np.asarray(train_indices, dtype=np.int64),
        validation_indices=np.asarray(validation_indices, dtype=np.int64),
        test_indices=np.asarray(test_indices, dtype=np.int64),
        feature_shape=np.asarray(feature_shape, dtype=np.int64),
        flatten_order=FLATTEN_ORDER,
        source_times_days=np.asarray(source_times_days, dtype=np.float64),
        training_t_max=float(training_t_max),
        source_snapshot_file=str(source_snapshot_file),
        pod_version=POD_VERSION,
    )


def select_rank(cumulative_energy: np.ndarray, decomp_cfg: dict[str, Any]) -> int:
    """Select a POD rank from fixed or energy mode settings."""

    available_rank = int(np.asarray(cumulative_energy).size)
    if available_rank <= 0:
        raise ValueError("available POD rank must be positive.")
    rank_mode = str(decomp_cfg.get("rank_mode", "energy")).lower()
    fixed_rank = int(decomp_cfg.get("fixed_rank", available_rank))
    max_rank = int(decomp_cfg.get("max_rank", available_rank))
    if fixed_rank <= 0 or max_rank <= 0:
        raise ValueError("fixed_rank and max_rank must be positive.")
    if rank_mode == "fixed":
        return max(1, min(fixed_rank, available_rank))
    if rank_mode != "energy":
        raise ValueError("pod.decomposition.rank_mode must be 'fixed' or 'energy'.")
    threshold = float(decomp_cfg.get("energy_threshold", 0.99999))
    if threshold <= 0.0 or threshold > 1.0:
        raise ValueError("energy_threshold must be in (0, 1].")
    needed = int(np.searchsorted(cumulative_energy, threshold, side="left") + 1)
    limited = min(max_rank, available_rank)
    if needed > limited:
        retained = float(cumulative_energy[limited - 1])
        print(
            "Warning: POD energy threshold was not reached within max_rank; "
            f"using rank={limited} with retained_energy={retained:.12f}."
        )
        return max(1, limited)
    return max(1, min(needed, available_rank))


def project(states: np.ndarray, pod_basis: PODBasis) -> np.ndarray:
    """Project normalized states onto the POD basis."""

    states_arr = np.asarray(states, dtype=np.float64)
    coefficients = (states_arr - pod_basis.mean_state.reshape(1, -1)) @ pod_basis.basis
    assert_finite("POD coefficients", coefficients)
    return coefficients


def reconstruct(coefficients: np.ndarray, pod_basis: PODBasis) -> np.ndarray:
    """Reconstruct normalized states from POD coefficients."""

    coeff = np.asarray(coefficients, dtype=np.float64)
    states = pod_basis.mean_state.reshape(1, -1) + coeff @ pod_basis.basis.T
    assert_finite("POD reconstruction", states)
    return states


def save_pod_basis(pod_basis: PODBasis, path: str | Path) -> None:
    """Save POD basis data outside the PyTorch checkpoint."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        mean_state=pod_basis.mean_state.astype(np.float32),
        basis=pod_basis.basis.astype(np.float32),
        singular_values=pod_basis.singular_values.astype(np.float64),
        cumulative_energy=pod_basis.cumulative_energy.astype(np.float64),
        selected_rank=np.asarray(pod_basis.selected_rank, dtype=np.int64),
        retained_energy=np.asarray(pod_basis.retained_energy, dtype=np.float64),
        free_cells=pod_basis.free_cells.astype(np.int64),
        well_cells=pod_basis.well_cells.astype(np.int64),
        component_names=np.asarray(pod_basis.component_names),
        train_indices=pod_basis.train_indices.astype(np.int64),
        validation_indices=pod_basis.validation_indices.astype(np.int64),
        test_indices=pod_basis.test_indices.astype(np.int64),
        feature_shape=pod_basis.feature_shape.astype(np.int64),
        flatten_order=np.asarray(pod_basis.flatten_order),
        source_times_days=pod_basis.source_times_days.astype(np.float64),
        training_t_max=np.asarray(pod_basis.training_t_max, dtype=np.float64),
        source_snapshot_file=np.asarray(pod_basis.source_snapshot_file),
        pod_version=np.asarray(pod_basis.pod_version),
    )


def load_pod_basis(path: str | Path) -> PODBasis:
    """Load a saved POD basis."""

    basis_path = Path(path)
    if not basis_path.exists():
        raise FileNotFoundError(f"POD basis file does not exist: {basis_path}")
    with np.load(basis_path, allow_pickle=True) as data:
        values = {key: data[key] for key in data.files}
    return PODBasis(
        mean_state=np.asarray(values["mean_state"], dtype=np.float64),
        basis=np.asarray(values["basis"], dtype=np.float64),
        singular_values=np.asarray(values["singular_values"], dtype=np.float64),
        cumulative_energy=np.asarray(values["cumulative_energy"], dtype=np.float64),
        selected_rank=int(np.asarray(values["selected_rank"]).item()),
        retained_energy=float(np.asarray(values["retained_energy"]).item()),
        free_cells=np.asarray(values["free_cells"], dtype=np.int64),
        well_cells=np.asarray(values["well_cells"], dtype=np.int64),
        component_names=as_string_list(values["component_names"]),
        train_indices=np.asarray(values["train_indices"], dtype=np.int64),
        validation_indices=np.asarray(values["validation_indices"], dtype=np.int64),
        test_indices=np.asarray(values["test_indices"], dtype=np.int64),
        feature_shape=np.asarray(values["feature_shape"], dtype=np.int64),
        flatten_order=str(np.asarray(values["flatten_order"]).item()),
        source_times_days=np.asarray(values["source_times_days"], dtype=np.float64),
        training_t_max=float(np.asarray(values["training_t_max"]).item()),
        source_snapshot_file=str(np.asarray(values["source_snapshot_file"]).item()),
        pod_version=str(np.asarray(values["pod_version"]).item()),
    )


def _split_count(interior_count: int, fraction: float) -> int:
    if fraction <= 0.0:
        return 0
    return max(1, int(np.floor(float(interior_count) * float(fraction))))


def _validate_split(count: int, train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> None:
    sets = [set(values.tolist()) for values in (train, validation, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise RuntimeError("POD splits overlap.")
    covered = sorted((sets[0] | sets[1] | sets[2]))
    if covered != list(range(count)):
        raise RuntimeError("POD splits do not cover all snapshots.")

