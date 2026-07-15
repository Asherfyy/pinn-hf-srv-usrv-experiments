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
    return parser.parse_args()


def _time_indices(times_available: np.ndarray, requested: list[float]) -> list[int]:
    indices: list[int] = []
    for time_value in requested:
        idx = int(np.argmin(np.abs(times_available - float(time_value))))
        if idx not in indices:
            indices.append(idx)
    return indices


def plot_field(
    config: dict[str, Any],
    geometry: ReservoirGeometry,
    snapshots: dict[str, np.ndarray],
    time_index: int,
    values: np.ndarray,
    variable_name: str,
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
    out = Path(config["paths"]["figures"]) / f"field_{variable_name}_t{time_value:g}.png"
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
    snapshots = load_snapshots(config)
    pressure = snapshots["pressure_mpa"]
    if pressure.ndim == 3:
        component_names = [str(value) for value in snapshots.get("component_names", np.asarray(config["pressure"]["components"]))]
        variables = {name: pressure[:, :, idx] for idx, name in enumerate(component_names)}
        variables["Ptotal"] = np.sum(pressure, axis=2)
    else:
        variables = {"pressure": pressure}
    requested = [float(value) for value in config["evaluation"]["times"]]
    for idx in _time_indices(snapshots["times_days"], requested):
        for variable_name, values_by_time in variables.items():
            plot_field(config, geometry, snapshots, idx, values_by_time[idx], variable_name)


if __name__ == "__main__":
    main()
