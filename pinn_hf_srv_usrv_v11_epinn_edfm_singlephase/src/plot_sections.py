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
    parser.add_argument("--snapshot-file", type=str, default=None)
    return parser.parse_args()


def _configured_times(config: dict) -> list[float]:
    plotting = config.get("plotting", {})
    values = plotting.get("profile_times_days", plotting.get("times_days", config["evaluation"]["times"]))
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
        raise ValueError(f"Unknown profile variable(s): {missing}. Available: {available}.")
    return selected


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    snapshots = load_snapshots(config, args.snapshot_file)
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
    requested = _configured_times(config)
    time_indices = [int(np.argmin(np.abs(snapshots["times_days"] - value))) for value in requested]

    pressure = snapshots["pressure_mpa"]
    if pressure.ndim == 3:
        component_names = [str(value) for value in snapshots.get("component_names", np.asarray(config["pressure"]["components"]))]
        variables = {name: pressure[:, :, component] for component, name in enumerate(component_names)}
        variables["Ptotal"] = np.sum(pressure, axis=2)
    else:
        variables = {"pressure": pressure}
    variables = _select_variables(variables, config.get("plotting", {}).get("profile_variables"))

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
    output_dir = Path(config["paths"]["figures"]) if args.snapshot_file is None else Path(config["paths"]["figures"]) / "pod"
    out = output_dir / "main_fracture_profile.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
