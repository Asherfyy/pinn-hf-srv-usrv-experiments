"""Plot EDFM fracture pressure profiles from v11 snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .config import load_config
from .evaluate import load_snapshots
from .utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v11 fracture pressure profiles.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    snapshots = load_snapshots(config)
    names = snapshots["fracture_name"].astype(str)
    main_ids = np.where(names == "main_frac")[0]
    if main_ids.size == 0:
        print("No main_frac segments found; skipping main fracture profile.")
        return
    matrix_count = int(np.asarray(snapshots["matrix_cell_count"]).item())
    starts = snapshots["fracture_start"][main_ids]
    ends = snapshots["fracture_end"][main_ids]
    centers = 0.5 * (starts + ends)
    order = np.argsort(centers[:, 0])
    cell_ids = matrix_count + main_ids[order]
    distance = centers[order, 0] - float(np.min(centers[order, 0]))
    requested = [float(value) for value in config["evaluation"]["times"]]
    time_indices = [int(np.argmin(np.abs(snapshots["times_days"] - value))) for value in requested]

    pressure = snapshots["pressure_mpa"]
    if pressure.ndim == 3:
        component_names = [str(value) for value in snapshots.get("component_names", np.asarray(config["pressure"]["components"]))]
        variables = {name: pressure[:, :, component] for component, name in enumerate(component_names)}
        variables["Ptotal"] = np.sum(pressure, axis=2)
    else:
        variables = {"pressure": pressure}

    fig, axes = plt.subplots(len(variables), 1, figsize=(8, 3.2 * len(variables)), constrained_layout=True, sharex=True)
    axes_arr = np.atleast_1d(axes)
    for ax, (variable_name, values_by_time) in zip(axes_arr, variables.items()):
        for idx in dict.fromkeys(time_indices):
            ax.plot(distance, values_by_time[idx, cell_ids], linewidth=1.6, label=f"{snapshots['times_days'][idx]:g} day")
        ax.set_title(f"main_frac {variable_name} profile")
        ax.set_ylabel("MPa")
        ax.grid(True, alpha=0.25)
        ax.legend()
    axes_arr[-1].set_xlabel("distance along fracture (m)")
    out = Path(config["paths"]["figures"]) / "main_fracture_profile.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
