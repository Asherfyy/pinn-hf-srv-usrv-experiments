"""Generate source snapshots for POD-MLP training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from .config import load_config, validate_config
from .edfm_grid import EdfmGrid, build_edfm_grid
from .fvm_solve import solve_direct
from .gas_metrics import build_snapshot_gas_metadata
from .geometry import ReservoirGeometry
from .pod_utils import get_pod_directories, load_snapshot_archive, make_pod_snapshot_times, save_snapshot_archive
from .utils import PROJECT_VERSION, ensure_output_dirs, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or import POD source snapshots.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--grid-nx", type=int, default=None)
    parser.add_argument("--grid-ny", type=int, default=None)
    parser.add_argument("--early-points", type=int, default=None)
    parser.add_argument("--middle-points", type=int, default=None)
    parser.add_argument("--late-points", type=int, default=None)
    parser.add_argument("--final-time", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    pod_dirs = get_pod_directories(config)
    source = _snapshot_source(config)

    if source == "direct_fvm":
        _apply_overrides(config, args)
        validate_config(config)
        _generate_direct_snapshots(config, pod_dirs)
    elif source == "pinn":
        _warn_ignored_overrides(args)
        _generate_pinn_snapshots(config, pod_dirs)
    else:
        raise ValueError(f"Unsupported POD snapshot source: {source}")


def _snapshot_source(config: dict[str, Any]) -> str:
    return str(config["pod"]["snapshot_generation"].get("source", "direct_fvm")).lower()


def _generate_direct_snapshots(config: dict[str, Any], pod_dirs: dict[str, Path]) -> None:
    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    times = make_pod_snapshot_times(config)
    snapshots, well_history = solve_direct(grid, config, times.tolist())

    output_name = str(config["pod"]["snapshot_generation"].get("snapshot_file", "direct_snapshots.npz"))
    output_path = pod_dirs["outputs"] / output_name
    _save_pod_direct_snapshots(config, grid, times, snapshots, output_path)
    save_csv(well_history, pod_dirs["tables"] / "source_well_history.csv")
    print(f"snapshot source: direct_fvm")
    print(f"snapshot count: {times.size}")
    print(f"time min/max days: {float(times[0]):g} / {float(times[-1]):g}")
    print(f"cell count: {grid.num_cells}")
    print(f"component count: {snapshots.shape[2]}")
    print(f"output path: {output_path}")


def _generate_pinn_snapshots(config: dict[str, Any], pod_dirs: dict[str, Path]) -> None:
    input_path = _resolve_pinn_snapshot_path(config)
    source_snapshots = load_snapshot_archive(input_path)
    times = np.asarray(source_snapshots["times_days"], dtype=np.float64)
    pressure = np.asarray(source_snapshots["pressure_mpa"], dtype=np.float32)

    output_name = str(config["pod"]["snapshot_generation"].get("snapshot_file", "pinn_snapshots.npz"))
    output_path = pod_dirs["outputs"] / output_name
    save_snapshot_archive(
        output_path,
        source_metadata=source_snapshots,
        times_days=times,
        pressure_mpa=pressure,
        solver="pinn_epinn_train_pod_dataset",
        project_version=PROJECT_VERSION,
    )
    print(f"snapshot source: pinn")
    print(f"input path: {input_path}")
    print(f"snapshot count: {times.size}")
    print(f"time min/max days: {float(times[0]):g} / {float(times[-1]):g}")
    print(f"cell count: {pressure.shape[1]}")
    print(f"component count: {pressure.shape[2]}")
    print(f"output path: {output_path}")
    if times.size < 10:
        print("Warning: POD training needs at least 10 snapshots; run main.py train with more time steps before POD training.")


def _resolve_pinn_snapshot_path(config: dict[str, Any]) -> Path:
    snapshot = config["pod"]["snapshot_generation"]
    raw_path = str(snapshot.get("pinn_snapshot_file", "snapshots.npz"))
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return Path(config["paths"]["outputs"]) / path
    return path


def _warn_ignored_overrides(args: argparse.Namespace) -> None:
    ignored = [
        args.grid_nx,
        args.grid_ny,
        args.early_points,
        args.middle_points,
        args.late_points,
        args.final_time,
    ]
    if any(value is not None for value in ignored):
        print("POD snapshot source is pinn; grid/time generation overrides are ignored.")


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.grid_nx is not None:
        config["grid"]["nx"] = int(args.grid_nx)
    if args.grid_ny is not None:
        config["grid"]["ny"] = int(args.grid_ny)
    snapshot = config["pod"]["snapshot_generation"]
    if args.early_points is not None:
        snapshot["early_points"] = int(args.early_points)
    if args.middle_points is not None:
        snapshot["middle_points"] = int(args.middle_points)
    if args.late_points is not None:
        snapshot["late_points"] = int(args.late_points)
    if args.final_time is not None:
        final_time = float(args.final_time)
        if final_time <= 0.0:
            raise ValueError("--final-time must be positive.")
        snapshot["final_time_days"] = final_time
        if not (final_time > float(snapshot["middle_end_days"]) > float(snapshot["early_end_days"]) >= 0.0):
            snapshot["early_end_days"] = min(float(snapshot["early_end_days"]), 0.03 * final_time)
            snapshot["middle_end_days"] = min(float(snapshot["middle_end_days"]), 0.2 * final_time)
            if not (final_time > float(snapshot["middle_end_days"]) > float(snapshot["early_end_days"]) >= 0.0):
                snapshot["early_end_days"] = final_time / 30.0
                snapshot["middle_end_days"] = final_time / 5.0


def _save_pod_direct_snapshots(
    config: dict[str, Any],
    grid: EdfmGrid,
    times: np.ndarray,
    snapshots: np.ndarray,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times_days=np.asarray(times, dtype=np.float64),
        cell_xy=grid.cell_xy.astype(np.float64),
        cell_region=grid.cell_region.astype(str),
        pressure_mpa=np.asarray(snapshots, dtype=np.float32),
        matrix_cell_count=np.asarray(grid.matrix_cell_count, dtype=np.int64),
        nx=np.asarray(grid.nx, dtype=np.int64),
        ny=np.asarray(grid.ny, dtype=np.int64),
        x_edges=grid.x_edges.astype(np.float64),
        y_edges=grid.y_edges.astype(np.float64),
        well_cell=np.asarray(grid.well_cell, dtype=np.int64),
        well_cells=grid.well_cells.astype(np.int64),
        component_names=np.asarray(config["pressure"]["components"]),
        fracture_start=np.asarray([segment.start for segment in grid.fracture_segments], dtype=np.float64),
        fracture_end=np.asarray([segment.end for segment in grid.fracture_segments], dtype=np.float64),
        fracture_name=np.asarray([segment.name for segment in grid.fracture_segments]),
        **build_snapshot_gas_metadata(config, grid),
        solver=np.asarray("direct_fvm_edfm_pod_dataset"),
        project_version=np.asarray(PROJECT_VERSION),
    )


if __name__ == "__main__":
    main()
