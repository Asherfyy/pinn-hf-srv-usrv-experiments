"""Plot fracture pressure profiles from v10 snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .evaluate import load_snapshots, pressure_snapshots_mpa
from .geometry import ReservoirGeometry
from .rdfm_fractures import fractures_from_geometry
from .utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v10 RDFM/I-PINN fracture profiles.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def nearest_values(node_xy: np.ndarray, values: np.ndarray, query_xy: np.ndarray) -> np.ndarray:
    result = np.empty((query_xy.shape[0],), dtype=np.float64)
    for idx, point in enumerate(query_xy):
        nearest = int(np.argmin(np.sum((node_xy - point[None, :]) ** 2, axis=1)))
        result[idx] = values[nearest]
    return result


def plot_profile(distance: np.ndarray, values_by_time: dict[float, np.ndarray], title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for time_value, values in values_by_time.items():
        ax.plot(distance, values, label=f"{time_value:g} day", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("distance along fracture (m)")
    ax.set_ylabel("Ptotal (MPa)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"])
    fractures = fractures_from_geometry(geometry)
    snapshots = load_snapshots(config)
    node_xy = snapshots["node_xy"]
    times = snapshots["times_days"]
    pressure = pressure_snapshots_mpa(snapshots["u"], config)
    ptotal = pressure[:, :, 0] + pressure[:, :, 1]
    requested = [float(value) for value in config["evaluation"]["times"]]
    time_indices = [int(np.argmin(np.abs(times - value))) for value in requested]

    for fracture in [fractures[0], fractures[1] if len(fractures) > 1 else fractures[0]]:
        points = np.linspace(fracture.start, fracture.end, 300)
        distance = np.linalg.norm(points - points[0], axis=1)
        values_by_time = {
            float(times[idx]): nearest_values(node_xy, ptotal[idx], points)
            for idx in dict.fromkeys(time_indices)
        }
        safe_name = fracture.name.replace(" ", "_")
        out = Path(config["paths"]["figures"]) / f"profile_{safe_name}_Ptotal.png"
        plot_profile(distance, values_by_time, f"{fracture.name} Ptotal profile", out)


if __name__ == "__main__":
    main()
