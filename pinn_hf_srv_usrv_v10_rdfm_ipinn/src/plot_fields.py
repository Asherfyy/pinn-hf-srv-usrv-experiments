"""Plot pressure fields from v10 RDFM/I-PINN snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from .config import load_config
from .evaluate import load_snapshots
from .geometry import ReservoirGeometry
from .utils import ensure_output_dirs, pressure_hat_to_mpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v10 RDFM/I-PINN pressure fields.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def pressure_snapshots_mpa(u: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    flat = torch.as_tensor(u.reshape(-1, 2), dtype=torch.float32)
    pressure = pressure_hat_to_mpa(flat, config["boundary"]).numpy()
    return pressure.reshape(u.shape)


def _time_indices(times_available: np.ndarray, requested: list[float]) -> list[int]:
    indices: list[int] = []
    for time_value in requested:
        idx = int(np.argmin(np.abs(times_available - float(time_value))))
        if idx not in indices:
            indices.append(idx)
    return indices


def plot_field(
    geometry: ReservoirGeometry,
    xy: np.ndarray,
    triangles: np.ndarray,
    values: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    tri = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    contour = ax.tricontourf(tri, values, levels=40, cmap="rainbow")
    geometry.draw_overlay(ax)
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label="MPa")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"])
    snapshots = load_snapshots(config)
    xy = snapshots["node_xy"]
    triangles = snapshots["triangles"]
    times = snapshots["times_days"]
    pressure = pressure_snapshots_mpa(snapshots["u"], config)
    variables = {
        "P12": pressure[:, :, 0],
        "P13": pressure[:, :, 1],
        "Ptotal": pressure[:, :, 0] + pressure[:, :, 1],
    }
    requested = [float(value) for value in config["evaluation"]["times"]]
    for idx in _time_indices(times, requested):
        time_value = float(times[idx])
        for name, values in variables.items():
            out = Path(config["paths"]["figures"]) / f"field_{name}_t{time_value:g}.png"
            plot_field(geometry, xy, triangles, values[idx], f"{name} at {time_value:g} day", out)
            print(f"Saved: {out}")


if __name__ == "__main__":
    main()
