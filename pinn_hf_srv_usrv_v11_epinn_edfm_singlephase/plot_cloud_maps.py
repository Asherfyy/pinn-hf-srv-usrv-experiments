"""Plot PINN/POD, FVM, and error pressure cloud maps.

Examples:

    python plot_cloud_maps.py --source pinn --time 999 --component Ptotal
    python plot_cloud_maps.py --source pod --time 500 --component P12
    python plot_cloud_maps.py --source pod --times 100 500 999 --fvm-file outputs/direct_snapshots.npz

The comparison requires the selected PINN/POD archive and FVM archive to share
the same matrix grid. The script does not interpolate between different grids.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# IDE direct-run defaults.
IDE_SOURCE = "pinn"  # "pinn" or "pod"
IDE_TIMES_DAYS: list[float] | None = None  # None means the last source snapshot time.
IDE_COMPONENT = "P12"  # "P12", "P13", or "Ptotal"
IDE_PINN_FILE: str | None = None
IDE_POD_FILE: str | None = None
IDE_FVM_FILE: str | None = None
IDE_OUTPUT_DIR: str | None = None
IDE_AUTO_FVM = True
IDE_AUTO_FVM_MAX_MATRIX_CELLS = 5000


def relaunch_with_venv_if_needed() -> None:
    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v11 PINN/POD vs FVM pressure cloud-map comparisons.")
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--source", choices=["pinn", "pod"], default=IDE_SOURCE, help="Left panel source: PINN training snapshots or POD predictions.")
    parser.add_argument("--time", type=float, default=None, help="Single requested plotting time in days.")
    parser.add_argument("--times", type=float, nargs="+", default=None, help="One or more requested plotting times in days.")
    parser.add_argument("--component", type=str, default=IDE_COMPONENT, help="Pressure component to plot: P12, P13, Ptotal, or another archive component name.")
    parser.add_argument("--pinn-file", type=str, default=IDE_PINN_FILE, help="PINN snapshot archive. Default: outputs/snapshots.npz.")
    parser.add_argument("--pod-file", type=str, default=IDE_POD_FILE, help="POD prediction archive. Default: outputs/pod/pod_predictions.npz.")
    parser.add_argument("--fvm-file", type=str, default=IDE_FVM_FILE, help="FVM snapshot archive. Default search prefers an existing same-grid FVM archive.")
    parser.add_argument("--output-dir", type=str, default=IDE_OUTPUT_DIR, help="Output directory. Default: outputs/figures/comparison.")
    parser.add_argument("--no-auto-fvm", action="store_true", default=not IDE_AUTO_FVM, help="Do not auto-build a same-grid FVM reference when none exists.")
    parser.add_argument("--auto-fvm-max-cells", type=int, default=IDE_AUTO_FVM_MAX_MATRIX_CELLS, help="Maximum matrix cells allowed for automatic FVM reference generation.")
    return parser.parse_args()


def main() -> None:
    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)

    from src.config import load_config
    from src.pod_utils import load_snapshot_archive
    from src.utils import ensure_output_dirs

    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    source_path = resolve_source_snapshot(config, args.source, args.pinn_file, args.pod_file)
    source_snapshots = load_snapshot_archive(source_path)
    times = _requested_times(args, source_snapshots)
    fvm_path = resolve_fvm_snapshot(
        config,
        args.fvm_file,
        source_snapshots,
        source_path,
        auto_build=not bool(args.no_auto_fvm),
        auto_max_matrix_cells=int(args.auto_fvm_max_cells),
    )
    fvm_snapshots = load_snapshot_archive(fvm_path)
    validate_same_matrix_grid(source_snapshots, fvm_snapshots, source_path, fvm_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(config["paths"]["figures"]) / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    for requested_time in times:
        result = build_comparison(source_snapshots, fvm_snapshots, args.component, float(requested_time))
        label = "PINN" if args.source == "pinn" else "POD"
        out = output_dir / f"{args.source}_fvm_error_{args.component}_t{requested_time:g}.png"
        plot_comparison(result, source_snapshots, label, args.component, out)
        print(
            f"Saved: {out} | requested={requested_time:g} day "
            f"{label}_time={result['source_time']:g} FVM_time={result['fvm_time']:g} "
            f"MAE={result['mae']:.6g} MPa RMSE={result['rmse']:.6g} MPa MaxAbs={result['max_abs']:.6g} MPa"
        )


def _requested_times(args: argparse.Namespace, source_snapshots: dict[str, np.ndarray]) -> list[float]:
    if args.times is not None and len(args.times) > 0:
        return [float(value) for value in args.times]
    if args.time is not None:
        return [float(args.time)]
    if IDE_TIMES_DAYS is not None:
        return [float(value) for value in IDE_TIMES_DAYS]
    times = np.asarray(source_snapshots["times_days"], dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("Source snapshot archive contains no time values.")
    return [float(times[-1])]


def resolve_source_snapshot(config: dict[str, Any], source: str, pinn_file: str | None, pod_file: str | None) -> Path:
    if source == "pinn":
        return resolve_existing_path(config, pinn_file or str(Path(config["paths"]["outputs"]) / "snapshots.npz"))
    pod_name = pod_file or str(Path(config["paths"]["outputs"]) / "pod" / str(config["pod"]["files"].get("prediction_file", "pod_predictions.npz")))
    return resolve_existing_path(config, pod_name)


def resolve_fvm_snapshot(
    config: dict[str, Any],
    fvm_file: str | None,
    source_snapshots: dict[str, np.ndarray],
    source_path: Path,
    auto_build: bool,
    auto_max_matrix_cells: int,
) -> Path:
    if fvm_file is not None:
        return resolve_existing_path(config, fvm_file)
    candidates = [
        Path(config["paths"]["outputs"]) / "direct_snapshots.npz",
        Path(config["paths"]["outputs"]) / "pod" / "direct_snapshots.npz",
    ]
    mismatched: list[str] = []
    for candidate in candidates:
        path = resolve_candidate(candidate)
        if not path.exists() or not _archive_solver_contains(path, "direct_fvm"):
            continue
        with np.load(path, allow_pickle=True) as data:
            candidate_snapshots = {key: data[key] for key in data.files}
        try:
            validate_same_matrix_grid(source_snapshots, candidate_snapshots, source_path, path)
        except ValueError as exc:
            mismatched.append(f"{path} ({exc})")
            continue
        return path
    if auto_build:
        matrix_count = int(np.asarray(source_snapshots["matrix_cell_count"]).item())
        if matrix_count <= int(auto_max_matrix_cells):
            return build_matching_fvm_snapshot(config, source_snapshots)
        searched = ", ".join(str(resolve_candidate(candidate)) for candidate in candidates)
        raise FileNotFoundError(
            "No same-grid FVM reference archive was found, and automatic FVM generation was skipped because "
            f"matrix_cell_count={matrix_count} exceeds --auto-fvm-max-cells={int(auto_max_matrix_cells)}. "
            "Generate a same-grid FVM archive explicitly, then pass it with --fvm-file. "
            f"Searched: {searched}"
        )
    searched = ", ".join(str(resolve_candidate(candidate)) for candidate in candidates)
    details = " Mismatched candidates: " + " | ".join(mismatched) if mismatched else ""
    raise FileNotFoundError(
        "No same-grid FVM reference archive was found. Provide --fvm-file, or generate one with "
        "`python -m src.fvm_solve --config config/default.yaml --output-name direct_snapshots.npz`. "
        f"Searched: {searched}.{details}"
    )


def build_matching_fvm_snapshot(config: dict[str, Any], source_snapshots: dict[str, np.ndarray]) -> Path:
    from src.edfm_grid import build_edfm_grid
    from src.fvm_solve import _save_snapshots, solve_direct
    from src.geometry import ReservoirGeometry

    fvm_config = copy.deepcopy(config)
    nx = int(np.asarray(source_snapshots["nx"]).item())
    ny = int(np.asarray(source_snapshots["ny"]).item())
    times = [float(value) for value in np.asarray(source_snapshots["times_days"], dtype=np.float64).reshape(-1)]
    fvm_config["grid"]["nx"] = nx
    fvm_config["grid"]["ny"] = ny
    fvm_config["time_grid"]["times_days"] = times
    geometry = ReservoirGeometry(fvm_config["geometry"])
    grid = build_edfm_grid(geometry, fvm_config)
    snapshots, _well_history = solve_direct(grid, fvm_config, times)
    output_name = f"comparison_fvm/fvm_{nx}x{ny}_{len(times)}t.npz"
    _save_snapshots(fvm_config, grid, times, snapshots, output_name)
    out = Path(fvm_config["paths"]["outputs"]) / output_name
    print(f"Auto-built same-grid FVM reference: {out}")
    return resolve_candidate(out)


def resolve_existing_path(config: dict[str, Any], value: str | Path) -> Path:
    raw = Path(value)
    candidates = [raw] if raw.is_absolute() else [
        PROJECT_ROOT / raw,
        Path(config["paths"]["outputs"]) / raw,
        Path(config["paths"]["outputs"]) / "pod" / raw,
    ]
    for candidate in candidates:
        path = resolve_candidate(candidate)
        if path.exists():
            return path
    searched = ", ".join(str(resolve_candidate(candidate)) for candidate in candidates)
    raise FileNotFoundError(f"Snapshot archive does not exist. Searched: {searched}")


def resolve_candidate(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _archive_solver_contains(path: Path, token: str) -> bool:
    try:
        with np.load(path, allow_pickle=True) as data:
            if "solver" not in data.files:
                return False
            solver = str(np.asarray(data["solver"]).item())
            return token in solver
    except Exception:
        return False


def validate_same_matrix_grid(source: dict[str, np.ndarray], fvm: dict[str, np.ndarray], source_path: Path | None = None, fvm_path: Path | None = None) -> None:
    source_matrix_count = int(np.asarray(source["matrix_cell_count"]).item())
    fvm_matrix_count = int(np.asarray(fvm["matrix_cell_count"]).item())
    if source_matrix_count != fvm_matrix_count:
        raise ValueError(_grid_error("matrix_cell_count differs", source_path, fvm_path))
    for key in ["nx", "ny"]:
        if int(np.asarray(source.get(key, -1)).item()) != int(np.asarray(fvm.get(key, -2)).item()):
            raise ValueError(_grid_error(f"{key} differs", source_path, fvm_path))
    for key in ["x_edges", "y_edges"]:
        if key in source and key in fvm and not np.allclose(np.asarray(source[key]), np.asarray(fvm[key]), rtol=1.0e-10, atol=1.0e-10):
            raise ValueError(_grid_error(f"{key} differs", source_path, fvm_path))
    if not np.allclose(np.asarray(source["cell_xy"])[:source_matrix_count], np.asarray(fvm["cell_xy"])[:fvm_matrix_count], rtol=1.0e-10, atol=1.0e-10):
        raise ValueError(_grid_error("matrix cell centers differ", source_path, fvm_path))


def _grid_error(reason: str, source_path: Path | None, fvm_path: Path | None) -> str:
    left = str(source_path) if source_path is not None else "source archive"
    right = str(fvm_path) if fvm_path is not None else "FVM archive"
    return (
        f"Cannot compare archives because their matrix grids do not match: {reason}. "
        f"Source: {left}; FVM: {right}. Generate or choose FVM snapshots on the same grid."
    )


def build_comparison(source: dict[str, np.ndarray], fvm: dict[str, np.ndarray], component: str, requested_time: float) -> dict[str, Any]:
    source_index = nearest_time_index(source["times_days"], requested_time)
    fvm_index = nearest_time_index(fvm["times_days"], requested_time)
    source_values = extract_variable(source, source_index, component)
    fvm_values = extract_variable(fvm, fvm_index, component)
    matrix_count = int(np.asarray(source["matrix_cell_count"]).item())
    source_matrix = source_values[:matrix_count].astype(np.float64)
    fvm_matrix = fvm_values[:matrix_count].astype(np.float64)
    error = source_matrix - fvm_matrix
    return {
        "requested_time": float(requested_time),
        "source_time": float(np.asarray(source["times_days"], dtype=np.float64)[source_index]),
        "fvm_time": float(np.asarray(fvm["times_days"], dtype=np.float64)[fvm_index]),
        "source": source_matrix,
        "fvm": fvm_matrix,
        "error": error,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs": float(np.max(np.abs(error))),
    }


def nearest_time_index(times_days: np.ndarray, requested_time: float) -> int:
    times = np.asarray(times_days, dtype=np.float64).reshape(-1)
    if times.size == 0:
        raise ValueError("Snapshot archive contains no time values.")
    return int(np.argmin(np.abs(times - float(requested_time))))


def extract_variable(snapshots: dict[str, np.ndarray], time_index: int, component: str) -> np.ndarray:
    pressure = np.asarray(snapshots["pressure_mpa"], dtype=np.float64)
    if pressure.ndim != 3:
        raise ValueError(f"pressure_mpa must have shape [T, N, C], got {pressure.shape}.")
    name = str(component)
    if name.lower() == "ptotal":
        return np.sum(pressure[int(time_index)], axis=1)
    names = [str(value) for value in np.asarray(snapshots["component_names"]).reshape(-1)]
    if name not in names:
        raise ValueError(f"Unknown component {name!r}. Available components: {names + ['Ptotal']}.")
    return pressure[int(time_index), :, names.index(name)]


def plot_comparison(result: dict[str, Any], snapshots: dict[str, np.ndarray], source_label: str, component: str, output_path: str | Path) -> None:
    matrix_count = int(np.asarray(snapshots["matrix_cell_count"]).item())
    nx = int(np.asarray(snapshots["nx"]).item())
    ny = int(np.asarray(snapshots["ny"]).item())
    x_edges = np.asarray(snapshots["x_edges"], dtype=np.float64)
    y_edges = np.asarray(snapshots["y_edges"], dtype=np.float64)
    source_field = np.asarray(result["source"], dtype=np.float64).reshape(ny, nx)
    fvm_field = np.asarray(result["fvm"], dtype=np.float64).reshape(ny, nx)
    error_field = np.asarray(result["error"], dtype=np.float64).reshape(ny, nx)
    pressure_min = float(np.nanmin([np.nanmin(source_field), np.nanmin(fvm_field)]))
    pressure_max = float(np.nanmax([np.nanmax(source_field), np.nanmax(fvm_field)]))
    if np.isclose(pressure_min, pressure_max):
        pad = max(abs(pressure_min), 1.0) * 1.0e-6
        pressure_min -= pad
        pressure_max += pad
    error_abs = max(float(np.nanmax(np.abs(error_field))), 1.0e-12)
    panels = [
        (f"{source_label} {component}", source_field, "rainbow", pressure_min, pressure_max, "MPa"),
        (f"FVM {component}", fvm_field, "rainbow", pressure_min, pressure_max, "MPa"),
        (f"{source_label} - FVM", error_field, "coolwarm", -error_abs, error_abs, "MPa"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), constrained_layout=True)
    for ax, (title, values, cmap, vmin, vmax, label) in zip(axes, panels):
        mesh = ax.pcolormesh(x_edges, y_edges, values, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        draw_fracture_overlay(ax, snapshots)
        ax.set_title(title)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        fig.colorbar(mesh, ax=ax, label=label)
    fig.suptitle(
        f"{component} comparison at requested t={result['requested_time']:g} day "
        f"({source_label} t={result['source_time']:g}, FVM t={result['fvm_time']:g}); "
        f"MAE={result['mae']:.4g} MPa, RMSE={result['rmse']:.4g} MPa, MaxAbs={result['max_abs']:.4g} MPa"
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def draw_fracture_overlay(ax: Any, snapshots: dict[str, np.ndarray]) -> None:
    if "fracture_start" not in snapshots or "fracture_end" not in snapshots:
        return
    for start, end in zip(np.asarray(snapshots["fracture_start"]), np.asarray(snapshots["fracture_end"])):
        ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=0.75)


if __name__ == "__main__":
    main()
