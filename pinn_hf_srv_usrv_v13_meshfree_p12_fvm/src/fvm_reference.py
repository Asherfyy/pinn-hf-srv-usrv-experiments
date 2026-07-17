"""FVM/EDFM reference snapshots for offline comparison only."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .edfm_grid import EdfmGrid, build_edfm_grid
from .fvm_solve import assert_fvm_solution_trustworthy, diagnose_fvm_solution, solve_direct
from .geometry import REGION_HF, ReservoirGeometry
from .utils import PROJECT_VERSION, save_csv, pressure_mpa_to_hat


@dataclass(frozen=True)
class FVMReference:
    times_days: np.ndarray
    cell_xy: np.ndarray
    cell_region: np.ndarray
    pressure_mpa: np.ndarray
    matrix_cell_count: int
    nx: int
    ny: int
    x_edges: np.ndarray
    y_edges: np.ndarray
    fracture_cell_ids: np.ndarray

    def pressure_mpa_at(self, xyt: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
        device = xyt.device
        dtype = xyt.dtype
        arr = xyt.detach().cpu().numpy().astype(np.float64)
        cell_ids = self._cell_ids_for_xy(arr[:, 0], arr[:, 1], config)
        pressure_t = self._interpolate_time(arr[:, 2], cell_ids)
        return torch.as_tensor(pressure_t.reshape(-1, 1), dtype=dtype, device=device)

    def pressure_hat_at(self, xyt: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
        pressure = self.pressure_mpa_at(xyt, config)
        pressure_hat = pressure_mpa_to_hat(pressure, config["boundary"])
        return pressure_hat.to(device=xyt.device, dtype=xyt.dtype)

    def _cell_ids_for_xy(self, x: np.ndarray, y: np.ndarray, config: dict[str, Any]) -> np.ndarray:
        geom = ReservoirGeometry(config["geometry"])
        region = geom.region_id_np(x, y)
        ids = np.empty_like(x, dtype=np.int64)

        matrix_mask = region != REGION_HF
        if np.any(matrix_mask):
            i = np.searchsorted(self.x_edges, x[matrix_mask], side="right") - 1
            j = np.searchsorted(self.y_edges, y[matrix_mask], side="right") - 1
            i = np.clip(i, 0, self.nx - 1)
            j = np.clip(j, 0, self.ny - 1)
            ids[matrix_mask] = j * self.nx + i

        hf_mask = ~matrix_mask
        if np.any(hf_mask):
            if self.fracture_cell_ids.size == 0:
                ids[hf_mask] = self._nearest_cells(x[hf_mask], y[hf_mask], np.arange(self.matrix_cell_count, dtype=np.int64))
            else:
                ids[hf_mask] = self._nearest_cells(x[hf_mask], y[hf_mask], self.fracture_cell_ids)
        return ids

    def _nearest_cells(self, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        points = np.column_stack([x, y])
        centers = self.cell_xy[candidates]
        delta = points[:, None, :] - centers[None, :, :]
        nearest = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        return candidates[nearest]

    def _interpolate_time(self, t: np.ndarray, cell_ids: np.ndarray) -> np.ndarray:
        t_clipped = np.clip(t, float(self.times_days[0]), float(self.times_days[-1]))
        right = np.searchsorted(self.times_days, t_clipped, side="right")
        right = np.clip(right, 1, self.times_days.size - 1)
        left = right - 1
        t0 = self.times_days[left]
        t1 = self.times_days[right]
        w = np.where(t1 > t0, (t_clipped - t0) / (t1 - t0), 0.0)
        p0 = self.pressure_mpa[left, cell_ids, 0]
        p1 = self.pressure_mpa[right, cell_ids, 0]
        return (1.0 - w) * p0 + w * p1


def build_or_load_fvm_reference(config: dict[str, Any], geometry: ReservoirGeometry) -> FVMReference | None:
    ref_cfg = config.get("fvm_reference", {})
    if not bool(ref_cfg.get("enabled", False)):
        return None
    path = Path(config["paths"]["outputs"]) / str(ref_cfg.get("snapshot_name", "fvm_reference_snapshots.npz"))
    if path.exists() and not bool(ref_cfg.get("rebuild", False)):
        return load_fvm_reference(path)

    grid = build_edfm_grid(geometry, config)
    times = [float(value) for value in config["time_grid"]["times_days"]]
    snapshots, _well_history = solve_direct(grid, config, times)
    diagnostics = diagnose_fvm_solution(grid, config, times, snapshots)
    assert_fvm_solution_trustworthy(diagnostics, config)
    save_csv(diagnostics, Path(config["paths"]["tables"]) / "fvm_diagnostics.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times_days=np.asarray(times, dtype=np.float64),
        cell_xy=grid.cell_xy.astype(np.float64),
        cell_region=grid.cell_region.astype(str),
        pressure_mpa=snapshots.astype(np.float64),
        matrix_cell_count=np.asarray(grid.matrix_cell_count, dtype=np.int64),
        nx=np.asarray(grid.nx, dtype=np.int64),
        ny=np.asarray(grid.ny, dtype=np.int64),
        x_edges=grid.x_edges.astype(np.float64),
        y_edges=grid.y_edges.astype(np.float64),
        fracture_cell_ids=np.asarray([segment.cell_index for segment in grid.fracture_segments], dtype=np.int64),
        solver=np.asarray("direct_implicit_fvm_edfm"),
        project_version=np.asarray(PROJECT_VERSION),
    )
    return reference_from_grid(grid, np.asarray(times, dtype=np.float64), snapshots)


def build_trusted_fvm_reference(
    config: dict[str, Any],
    geometry: ReservoirGeometry,
    *,
    snapshot_name: str | None = None,
    rebuild: bool = False,
) -> tuple[FVMReference, list[dict[str, float | str | int]]]:
    """Build or load an FVM reference and verify it against the current config."""

    ref_cfg = config.setdefault("fvm_reference", {})
    if snapshot_name is not None:
        ref_cfg["snapshot_name"] = str(snapshot_name)
    ref_cfg["enabled"] = True
    ref_cfg["rebuild"] = bool(rebuild)
    path = Path(config["paths"]["outputs"]) / str(ref_cfg.get("snapshot_name", "fvm_reference_snapshots.npz"))
    grid = build_edfm_grid(geometry, config)
    times = [float(value) for value in config["time_grid"]["times_days"]]

    reference: FVMReference | None = None
    if path.exists() and not bool(rebuild):
        try:
            loaded = load_fvm_reference(path)
            if _reference_matches_grid(loaded, grid, times):
                reference = loaded
        except (KeyError, ValueError, OSError):
            reference = None

    if reference is None:
        snapshots, _well_history = solve_direct(grid, config, times)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            times_days=np.asarray(times, dtype=np.float64),
            cell_xy=grid.cell_xy.astype(np.float64),
            cell_region=grid.cell_region.astype(str),
            pressure_mpa=snapshots.astype(np.float64),
            matrix_cell_count=np.asarray(grid.matrix_cell_count, dtype=np.int64),
            nx=np.asarray(grid.nx, dtype=np.int64),
            ny=np.asarray(grid.ny, dtype=np.int64),
            x_edges=grid.x_edges.astype(np.float64),
            y_edges=grid.y_edges.astype(np.float64),
            fracture_cell_ids=np.asarray([segment.cell_index for segment in grid.fracture_segments], dtype=np.int64),
            solver=np.asarray("direct_implicit_fvm_edfm"),
            project_version=np.asarray(PROJECT_VERSION),
        )
        reference = reference_from_grid(grid, np.asarray(times, dtype=np.float64), snapshots)

    diagnostics = diagnose_fvm_solution(grid, config, times, reference.pressure_mpa)
    assert_fvm_solution_trustworthy(diagnostics, config)
    save_csv(diagnostics, Path(config["paths"]["tables"]) / "fvm_diagnostics.csv")
    return reference, diagnostics


def load_fvm_reference(path: str | Path) -> FVMReference:
    data = np.load(path, allow_pickle=True)
    return FVMReference(
        times_days=np.asarray(data["times_days"], dtype=np.float64),
        cell_xy=np.asarray(data["cell_xy"], dtype=np.float64),
        cell_region=np.asarray(data["cell_region"]),
        pressure_mpa=np.asarray(data["pressure_mpa"], dtype=np.float64),
        matrix_cell_count=int(np.asarray(data["matrix_cell_count"]).item()),
        nx=int(np.asarray(data["nx"]).item()),
        ny=int(np.asarray(data["ny"]).item()),
        x_edges=np.asarray(data["x_edges"], dtype=np.float64),
        y_edges=np.asarray(data["y_edges"], dtype=np.float64),
        fracture_cell_ids=np.asarray(data["fracture_cell_ids"], dtype=np.int64),
    )


def reference_from_grid(grid: EdfmGrid, times: np.ndarray, pressure_mpa: np.ndarray) -> FVMReference:
    return FVMReference(
        times_days=np.asarray(times, dtype=np.float64),
        cell_xy=grid.cell_xy.astype(np.float64),
        cell_region=grid.cell_region.astype(str),
        pressure_mpa=np.asarray(pressure_mpa, dtype=np.float64),
        matrix_cell_count=int(grid.matrix_cell_count),
        nx=int(grid.nx),
        ny=int(grid.ny),
        x_edges=grid.x_edges.astype(np.float64),
        y_edges=grid.y_edges.astype(np.float64),
        fracture_cell_ids=np.asarray([segment.cell_index for segment in grid.fracture_segments], dtype=np.int64),
    )


def _reference_matches_grid(reference: FVMReference, grid: EdfmGrid, times: list[float]) -> bool:
    if int(reference.nx) != int(grid.nx) or int(reference.ny) != int(grid.ny):
        return False
    if int(reference.matrix_cell_count) != int(grid.matrix_cell_count):
        return False
    if reference.pressure_mpa.shape[1] != grid.num_cells:
        return False
    if not np.allclose(reference.times_days, np.asarray(times, dtype=np.float64), rtol=0.0, atol=1.0e-12):
        return False
    if reference.cell_xy.shape != grid.cell_xy.shape:
        return False
    return bool(np.allclose(reference.cell_xy, grid.cell_xy, rtol=0.0, atol=1.0e-10))
