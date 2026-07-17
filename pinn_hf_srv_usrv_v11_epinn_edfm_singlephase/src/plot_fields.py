"""Plot pressure fields from v11 cell-centered snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .evaluate import load_snapshots
from .geometry import ReservoirGeometry
from .utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v11 pressure fields.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--snapshot-file", type=str, default=None)
    return parser.parse_args()


def _time_indices(times_available: np.ndarray, requested: list[float]) -> list[int]:
    indices: list[int] = []
    for time_value in requested:
        idx = int(np.argmin(np.abs(times_available - float(time_value))))
        if idx not in indices:
            indices.append(idx)
    return indices


def _configured_times(config: dict[str, Any]) -> list[float]:
    plotting = config.get("plotting", {})
    values = plotting.get("field_times_days", plotting.get("times_days", config["evaluation"]["times"]))
    return [float(value) for value in values]


def _select_variables(variables: dict[str, np.ndarray], requested: list[str] | None) -> dict[str, np.ndarray]:
    if not requested or any(str(value).lower() == "all" for value in requested):
        return variables
    selected: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for raw_name in requested:
        name = str(raw_name)
        if name in variables:
            selected[name] = variables[name]
        else:
            missing.append(name)
    if missing:
        available = ", ".join(variables.keys())
        raise ValueError(f"Unknown plot variable(s): {missing}. Available: {available}.")
    return selected


def plot_field(
    config: dict[str, Any],
    geometry: ReservoirGeometry,
    snapshots: dict[str, np.ndarray],
    time_index: int,
    values: np.ndarray,
    variable_name: str,
    output_dir: Path | None = None,
) -> None:
    matrix_count = int(np.asarray(snapshots["matrix_cell_count"]).item())
    nx = int(np.asarray(snapshots.get("nx", config["grid"]["nx"])).item())
    ny = int(np.asarray(snapshots.get("ny", config["grid"]["ny"])).item())
    pressure = values[:matrix_count].reshape(ny, nx)
    time_value = float(snapshots["times_days"][time_index])
    x_edges = snapshots.get("x_edges", np.linspace(float(config["geometry"]["x_min"]), float(config["geometry"]["x_max"]), nx + 1))
    y_edges = snapshots.get("y_edges", np.linspace(float(config["geometry"]["y_min"]), float(config["geometry"]["y_max"]), ny + 1))

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(x_edges, y_edges, pressure, cmap="rainbow", shading="auto")
    _draw_edfm_overlay(ax, snapshots)
    geometry.draw_overlay(ax)
    ax.set_title(f"{variable_name} at {time_value:g} day")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(mesh, ax=ax, label="MPa")
    out_dir = output_dir if output_dir is not None else Path(config["paths"]["figures"])
    out = out_dir / f"field_{variable_name}_t{time_value:g}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


def _draw_edfm_overlay(ax: Any, snapshots: dict[str, np.ndarray]) -> None:
    for start, end in zip(snapshots["fracture_start"], snapshots["fracture_end"]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color="black", linewidth=0.8)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"])
    snapshots = load_snapshots(config, args.snapshot_file)
    output_dir = Path(config["paths"]["figures"]) if args.snapshot_file is None else Path(config["paths"]["figures"]) / "pod"
    output_dir.mkdir(parents=True, exist_ok=True)
    pressure = snapshots["pressure_mpa"]
    if pressure.ndim == 3:
        component_names = [str(value) for value in snapshots.get("component_names", np.asarray(config["pressure"]["components"]))]
        variables = {name: pressure[:, :, idx] for idx, name in enumerate(component_names)}
        variables["Ptotal"] = np.sum(pressure, axis=2)
    else:
        variables = {"pressure": pressure}
    requested = _configured_times(config)
    variables = _select_variables(variables, config.get("plotting", {}).get("field_variables"))
    for idx in _time_indices(snapshots["times_days"], requested):
        for variable_name, values_by_time in variables.items():
            plot_field(config, geometry, snapshots, idx, values_by_time[idx], variable_name, output_dir)


if __name__ == "__main__":
    main()
