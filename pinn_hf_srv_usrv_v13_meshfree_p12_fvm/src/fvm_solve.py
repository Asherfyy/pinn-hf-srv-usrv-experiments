"""Direct implicit FVM/EDFM reference solver for v13 offline comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config
from .edfm_grid import EdfmGrid, build_edfm_grid, connection_transmissibility_matrix
from .geometry import ReservoirGeometry
from .utils import PROJECT_VERSION, bhp_component_target_mpa, bhp_target_mpa, ensure_output_dirs, pressure_component_affine_parameters, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the v13 single-pressure EDFM/FVM reference system directly.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output-name", type=str, default="snapshots.npz", help="Snapshot file name under paths.outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    times = [float(value) for value in config["time_grid"]["times_days"]]
    snapshots, well_history = solve_direct(grid, config, times)
    diagnostics = diagnose_fvm_solution(grid, config, times, snapshots)
    assert_fvm_solution_trustworthy(diagnostics, config)
    _save_snapshots(config, grid, times, snapshots, args.output_name)
    save_csv(diagnostics, Path(config["paths"]["tables"]) / "fvm_diagnostics.csv")
    save_csv(well_history, Path(config["paths"]["tables"]) / "well_history.csv")


def solve_direct(grid: EdfmGrid, config: dict[str, Any], times: list[float]) -> tuple[np.ndarray, list[dict[str, float]]]:
    if len(times) < 2:
        raise ValueError("At least two time values are required.")
    storage = grid.cell_storage
    well_cells = np.asarray(grid.well_cells, dtype=np.int64)
    free_cells = np.setdiff1d(np.arange(grid.num_cells, dtype=np.int64), well_cells, assume_unique=False)
    free_pos = np.full((grid.num_cells,), -1, dtype=np.int64)
    free_pos[free_cells] = np.arange(free_cells.size, dtype=np.int64)
    conn_i = np.asarray([connection.i for connection in grid.connections], dtype=np.int64)
    conn_j = np.asarray([connection.j for connection in grid.connections], dtype=np.int64)
    pressure_params = pressure_component_affine_parameters(config)
    component_count = len(config["pressure"]["components"])
    transmissibility = connection_transmissibility_matrix(grid.connections, component_count)
    row_scale = np.zeros((grid.num_cells, component_count), dtype=np.float64)
    if transmissibility.size > 0:
        np.add.at(row_scale, conn_i, transmissibility)
        np.add.at(row_scale, conn_j, transmissibility)
    pressure = np.tile(pressure_params["initial_mpa"].reshape(1, component_count), (grid.num_cells, 1)).astype(np.float64)
    pressure[well_cells] = bhp_component_target_mpa(times[0], config)
    snapshots = [pressure.copy()]
    well_history = [_well_row(grid, times[0], pressure, config)]
    solver_cfg = config.get("direct_solver", {})
    cg_rtol = float(solver_cfg.get("cg_rtol", 1.0e-9))
    cg_atol = float(solver_cfg.get("cg_atol", 1.0e-10))
    cg_max_iter = int(solver_cfg.get("cg_max_iter", 3000))

    for time_prev, time_next in zip(times, times[1:]):
        dt_days = float(time_next - time_prev)
        if dt_days <= 0.0:
            raise ValueError(f"time_grid must be strictly increasing, got dt={dt_days:g}.")
        bhp = bhp_component_target_mpa(time_next, config)
        storage_over_dt = storage / dt_days
        pressure_free_components: list[np.ndarray] = []
        iteration_max = 0
        residual_max = 0.0
        for component in range(component_count):
            transmissibility_component = transmissibility[:, component]
            row_scale_component = row_scale[:, component]
            rhs = _reduced_rhs(storage_over_dt, pressure[:, component], free_cells, free_pos, well_cells, conn_i, conn_j, transmissibility_component, float(bhp[component]))
            pressure_free, iterations, residual_norm = _pcg_solve(
                x0=pressure[free_cells, component],
                rhs=rhs,
                free_cells=free_cells,
                storage_over_dt=storage_over_dt,
                row_scale=row_scale_component,
                conn_i=conn_i,
                conn_j=conn_j,
                transmissibility=transmissibility_component,
                num_cells=grid.num_cells,
                rtol=cg_rtol,
                atol=cg_atol,
                max_iter=cg_max_iter,
            )
            pressure_free_components.append(pressure_free)
            iteration_max = max(iteration_max, iterations)
            residual_max = max(residual_max, residual_norm)
        pressure = pressure.copy()
        for component, pressure_free in enumerate(pressure_free_components):
            pressure[free_cells, component] = pressure_free
        pressure[well_cells] = bhp
        print(f"direct solve t={time_next:g} day cg_iter_max={iteration_max} residual_max={residual_max:.3e}")
        snapshots.append(pressure.copy())
        well_history.append(_well_row(grid, time_next, pressure, config))

    return np.stack(snapshots, axis=0), well_history


def diagnose_fvm_solution(grid: EdfmGrid, config: dict[str, Any], times: list[float], snapshots: np.ndarray) -> list[dict[str, float | str | int]]:
    """Compute residual and consistency diagnostics for a completed FVM solve."""

    times_arr = np.asarray(times, dtype=np.float64)
    pressure = np.asarray(snapshots, dtype=np.float64)
    if pressure.shape[0] != times_arr.size:
        raise ValueError(f"snapshots time dimension {pressure.shape[0]} does not match {times_arr.size} times.")
    if pressure.shape[1] != grid.num_cells:
        raise ValueError(f"snapshots cell dimension {pressure.shape[1]} does not match grid cells {grid.num_cells}.")
    component_count = len(config["pressure"]["components"])
    if pressure.shape[2] != component_count:
        raise ValueError(f"snapshots component dimension {pressure.shape[2]} does not match config components {component_count}.")

    rows: list[dict[str, float | str | int]] = []
    rows.extend(_diagnose_fvm_grid(grid))
    pressure_params = pressure_component_affine_parameters(config)
    p_min_allowed = float(np.min(pressure_params["out_mpa"])) - 1.0e-7
    p_max_allowed = float(np.max(pressure_params["initial_mpa"])) + 1.0e-7
    rows.append({"metric": "fvm_pressure_min_mpa", "value": float(np.min(pressure))})
    rows.append({"metric": "fvm_pressure_max_mpa", "value": float(np.max(pressure))})
    rows.append({"metric": "fvm_pressure_bounds_ok", "value": float(np.min(pressure) >= p_min_allowed and np.max(pressure) <= p_max_allowed)})
    rows.append({"metric": "fvm_nonfinite_pressure_count", "value": int(np.sum(~np.isfinite(pressure)))})

    storage = grid.cell_storage
    well_cells = np.asarray(grid.well_cells, dtype=np.int64)
    free_cells = np.setdiff1d(np.arange(grid.num_cells, dtype=np.int64), well_cells, assume_unique=False)
    free_pos = np.full((grid.num_cells,), -1, dtype=np.int64)
    free_pos[free_cells] = np.arange(free_cells.size, dtype=np.int64)
    conn_i = np.asarray([connection.i for connection in grid.connections], dtype=np.int64)
    conn_j = np.asarray([connection.j for connection in grid.connections], dtype=np.int64)
    transmissibility = connection_transmissibility_matrix(grid.connections, component_count)

    residual_rel_max = 0.0
    residual_abs_max = 0.0
    for step, (time_prev, time_next) in enumerate(zip(times_arr[:-1], times_arr[1:]), start=1):
        dt_days = float(time_next - time_prev)
        if dt_days <= 0.0:
            raise ValueError("times must be strictly increasing for FVM diagnostics.")
        storage_over_dt = storage / dt_days
        bhp = bhp_component_target_mpa(float(time_next), config)
        for component, component_name in enumerate(config["pressure"]["components"]):
            trans = transmissibility[:, component]
            rhs = _reduced_rhs(storage_over_dt, pressure[step - 1, :, component], free_cells, free_pos, well_cells, conn_i, conn_j, trans, float(bhp[component]))
            residual = rhs - _apply_system_free(pressure[step, free_cells, component], free_cells, storage_over_dt, conn_i, conn_j, trans, grid.num_cells)
            abs_norm = float(np.linalg.norm(residual))
            rel_norm = abs_norm / max(float(np.linalg.norm(rhs)), 1.0)
            residual_abs_max = max(residual_abs_max, abs_norm)
            residual_rel_max = max(residual_rel_max, rel_norm)
            rows.append(
                {
                    "metric": "fvm_step_residual",
                    "time_days": float(time_next),
                    "component": str(component_name),
                    "residual_abs_l2": abs_norm,
                    "residual_rel_l2": rel_norm,
                }
            )
    rows.append({"metric": "fvm_residual_abs_l2_max", "value": residual_abs_max})
    rows.append({"metric": "fvm_residual_rel_l2_max", "value": residual_rel_max})
    return rows


def assert_fvm_solution_trustworthy(diagnostics: list[dict[str, float | str | int]], config: dict[str, Any]) -> None:
    """Raise if FVM diagnostics violate the direct-solver trust criteria."""

    solver_cfg = config.get("direct_solver", {})
    residual_limit = float(solver_cfg.get("diagnostic_residual_rel_max", 1.0e-7))
    pressure_bounds_ok = _diagnostic_value(diagnostics, "fvm_pressure_bounds_ok", default=0.0)
    nonfinite_count = _diagnostic_value(diagnostics, "fvm_nonfinite_pressure_count", default=1.0)
    residual_rel = _diagnostic_value(diagnostics, "fvm_residual_rel_l2_max", default=float("inf"))
    bad_connections = _diagnostic_value(diagnostics, "fvm_bad_connection_count", default=1.0)
    bad_storage = _diagnostic_value(diagnostics, "fvm_bad_storage_count", default=1.0)
    if pressure_bounds_ok < 0.5:
        raise RuntimeError("FVM diagnostic failed: pressure violates maximum-principle bounds.")
    if nonfinite_count > 0:
        raise RuntimeError(f"FVM diagnostic failed: nonfinite pressure count is {nonfinite_count:g}.")
    if residual_rel > residual_limit:
        raise RuntimeError(f"FVM diagnostic failed: max relative implicit residual {residual_rel:.3e} exceeds {residual_limit:.3e}.")
    if bad_connections > 0:
        raise RuntimeError(f"FVM diagnostic failed: bad connection count is {bad_connections:g}.")
    if bad_storage > 0:
        raise RuntimeError(f"FVM diagnostic failed: bad storage count is {bad_storage:g}.")


def _diagnose_fvm_grid(grid: EdfmGrid) -> list[dict[str, float | str | int]]:
    transmissibility = np.asarray([connection.transmissibility for connection in grid.connections], dtype=np.float64)
    bad_conn = int(np.sum(~np.isfinite(transmissibility)) + np.sum(transmissibility < 0.0))
    storage = np.asarray(grid.cell_storage, dtype=np.float64)
    bad_storage = int(np.sum(~np.isfinite(storage)) + np.sum(storage <= 0.0))
    rows: list[dict[str, float | str | int]] = [
        {"metric": "fvm_num_cells", "value": int(grid.num_cells)},
        {"metric": "fvm_matrix_cell_count", "value": int(grid.matrix_cell_count)},
        {"metric": "fvm_fracture_segment_count", "value": int(len(grid.fracture_segments))},
        {"metric": "fvm_connection_count", "value": int(len(grid.connections))},
        {"metric": "fvm_well_cell_count", "value": int(grid.well_cells.size)},
        {"metric": "fvm_bad_connection_count", "value": bad_conn},
        {"metric": "fvm_bad_storage_count", "value": bad_storage},
        {"metric": "fvm_storage_min", "value": float(np.min(storage))},
        {"metric": "fvm_storage_max", "value": float(np.max(storage))},
    ]
    for kind in ["mm", "mf", "ff"]:
        rows.append({"metric": f"fvm_connection_count_{kind}", "value": int(sum(1 for connection in grid.connections if connection.kind == kind))})
    if transmissibility.size > 0:
        rows.append({"metric": "fvm_transmissibility_min", "value": float(np.min(transmissibility))})
        rows.append({"metric": "fvm_transmissibility_max", "value": float(np.max(transmissibility))})
    return rows


def _diagnostic_value(diagnostics: list[dict[str, float | str | int]], metric: str, default: float) -> float:
    for row in diagnostics:
        if row.get("metric") == metric and "value" in row:
            return float(row["value"])
    return float(default)


def _reduced_rhs(
    storage_over_dt: np.ndarray,
    pressure_prev: np.ndarray,
    free_cells: np.ndarray,
    free_pos: np.ndarray,
    well_cells: np.ndarray,
    conn_i: np.ndarray,
    conn_j: np.ndarray,
    transmissibility: np.ndarray,
    bhp: float,
) -> np.ndarray:
    rhs = (storage_over_dt * pressure_prev)[free_cells].copy()
    well_mask = np.zeros((free_pos.size,), dtype=bool)
    well_mask[well_cells] = True
    i_free_j_well = (free_pos[conn_i] >= 0) & well_mask[conn_j]
    j_free_i_well = (free_pos[conn_j] >= 0) & well_mask[conn_i]
    np.add.at(rhs, free_pos[conn_i[i_free_j_well]], transmissibility[i_free_j_well] * float(bhp))
    np.add.at(rhs, free_pos[conn_j[j_free_i_well]], transmissibility[j_free_i_well] * float(bhp))
    return rhs


def _pcg_solve(
    x0: np.ndarray,
    rhs: np.ndarray,
    free_cells: np.ndarray,
    storage_over_dt: np.ndarray,
    row_scale: np.ndarray,
    conn_i: np.ndarray,
    conn_j: np.ndarray,
    transmissibility: np.ndarray,
    num_cells: int,
    rtol: float,
    atol: float,
    max_iter: int,
) -> tuple[np.ndarray, int, float]:
    x = np.asarray(x0, dtype=np.float64).copy()
    diag = (storage_over_dt + row_scale)[free_cells]
    inv_diag = 1.0 / np.maximum(diag, 1.0e-30)
    rhs_norm = float(np.linalg.norm(rhs))
    tolerance = max(float(atol), float(rtol) * max(rhs_norm, 1.0))
    residual = rhs - _apply_system_free(x, free_cells, storage_over_dt, conn_i, conn_j, transmissibility, num_cells)
    residual_norm = float(np.linalg.norm(residual))
    if residual_norm <= tolerance:
        return x, 0, residual_norm
    z = inv_diag * residual
    direction = z.copy()
    rz_old = float(np.dot(residual, z))
    for iteration in range(1, int(max_iter) + 1):
        matvec = _apply_system_free(direction, free_cells, storage_over_dt, conn_i, conn_j, transmissibility, num_cells)
        denom = float(np.dot(direction, matvec))
        if abs(denom) <= 1.0e-30:
            break
        alpha = rz_old / denom
        x += alpha * direction
        residual -= alpha * matvec
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance:
            return x, iteration, residual_norm
        z = inv_diag * residual
        rz_new = float(np.dot(residual, z))
        if abs(rz_old) <= 1.0e-30:
            break
        beta = rz_new / rz_old
        direction = z + beta * direction
        rz_old = rz_new
    return x, int(max_iter), residual_norm


def _apply_system_free(
    x_free: np.ndarray,
    free_cells: np.ndarray,
    storage_over_dt: np.ndarray,
    conn_i: np.ndarray,
    conn_j: np.ndarray,
    transmissibility: np.ndarray,
    num_cells: int,
) -> np.ndarray:
    pressure = np.zeros((num_cells,), dtype=np.float64)
    pressure[free_cells] = x_free
    result = storage_over_dt * pressure
    delta = pressure[conn_i] - pressure[conn_j]
    np.add.at(result, conn_i, transmissibility * delta)
    np.add.at(result, conn_j, -transmissibility * delta)
    return result[free_cells]


def _save_snapshots(config: dict[str, Any], grid: EdfmGrid, times: list[float], snapshots: np.ndarray, output_name: str) -> None:
    out = Path(config["paths"]["outputs"]) / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times_days=np.asarray(times, dtype=np.float64),
        cell_xy=grid.cell_xy.astype(np.float64),
        cell_region=grid.cell_region.astype(str),
        pressure_mpa=snapshots.astype(np.float64),
        matrix_cell_count=np.asarray(grid.matrix_cell_count, dtype=np.int64),
        nx=np.asarray(grid.nx, dtype=np.int64),
        ny=np.asarray(grid.ny, dtype=np.int64),
        x_edges=grid.x_edges.astype(np.float64),
        y_edges=grid.y_edges.astype(np.float64),
        well_cell=np.asarray(grid.well_cell, dtype=np.int64),
        well_cells=grid.well_cells.astype(np.int64),
        component_names=np.asarray(config["pressure"]["components"]),
        fracture_cell_ids=np.asarray([segment.cell_index for segment in grid.fracture_segments], dtype=np.int64),
        fracture_start=np.asarray([segment.start for segment in grid.fracture_segments], dtype=np.float64),
        fracture_end=np.asarray([segment.end for segment in grid.fracture_segments], dtype=np.float64),
        fracture_name=np.asarray([segment.name for segment in grid.fracture_segments]),
        solver=np.asarray("direct_fvm_edfm"),
        project_version=np.asarray(PROJECT_VERSION),
    )
    print(f"Direct FVM snapshots saved: {out}")


def _well_row(grid: EdfmGrid, time_days: float, pressure: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    well_cells = {int(value) for value in grid.well_cells.tolist()}
    rate = np.zeros((pressure.shape[1],), dtype=np.float64)
    for connection in grid.connections:
        transmissibility = np.asarray(connection.component_transmissibility or (connection.transmissibility,) * rate.size, dtype=np.float64)
        if connection.i in well_cells and connection.j not in well_cells:
            rate += transmissibility * (pressure[connection.j] - pressure[connection.i])
        elif connection.j in well_cells and connection.i not in well_cells:
            rate += transmissibility * (pressure[connection.i] - pressure[connection.j])
    row = {
        "time_days": float(time_days),
        "bhp_target_mpa": float(bhp_target_mpa(time_days, config["well"])),
        "well_pressure_mpa": float(np.mean(np.sum(pressure[list(well_cells)], axis=1))),
        "estimated_rate": float(np.sum(rate)),
    }
    for idx, name in enumerate(config["pressure"]["components"]):
        row[f"{name}_well_pressure_mpa"] = float(np.mean(pressure[list(well_cells), idx]))
        row[f"{name}_estimated_rate"] = float(rate[idx])
    return row


if __name__ == "__main__":
    main()
