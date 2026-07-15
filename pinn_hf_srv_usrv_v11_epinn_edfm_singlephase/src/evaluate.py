"""Evaluate v11 E-PINN/EDFM pressure snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config
from .utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostics for v11 snapshots.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def load_snapshots(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(config["paths"]["outputs"]) / "snapshots.npz"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file does not exist: {path}. Run training first.")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def compute_diagnostics(snapshots: dict[str, np.ndarray]) -> pd.DataFrame:
    pressure = snapshots["pressure_mpa"]
    times = snapshots["times_days"]
    component_names = [str(value) for value in snapshots.get("component_names", np.asarray(["pressure"]))]
    rows: list[dict[str, float | str]] = [
        {"metric": "snapshot_count", "value": float(len(times))},
        {"metric": "cell_count", "value": float(snapshots["cell_xy"].shape[0])},
        {"metric": "matrix_cell_count", "value": float(np.asarray(snapshots["matrix_cell_count"]).item())},
        {"metric": "fracture_segment_count", "value": float(snapshots["fracture_start"].shape[0])},
        {"metric": "nonfinite_pressure_points", "value": float(np.sum(~np.isfinite(pressure)))},
        {"metric": "negative_pressure_points", "value": float(np.sum(pressure < 0.0))},
    ]
    for idx, time_value in enumerate(times):
        if pressure.ndim == 3:
            total = np.sum(pressure[idx], axis=1)
            for component_index, component_name in enumerate(component_names):
                values = pressure[idx, :, component_index]
                rows.extend(
                    [
                        {"metric": f"{component_name}_min_t{time_value:g}", "value": float(np.nanmin(values))},
                        {"metric": f"{component_name}_max_t{time_value:g}", "value": float(np.nanmax(values))},
                        {"metric": f"{component_name}_mean_t{time_value:g}", "value": float(np.nanmean(values))},
                    ]
                )
            rows.extend(
                [
                    {"metric": f"Ptotal_min_t{time_value:g}", "value": float(np.nanmin(total))},
                    {"metric": f"Ptotal_max_t{time_value:g}", "value": float(np.nanmax(total))},
                    {"metric": f"Ptotal_mean_t{time_value:g}", "value": float(np.nanmean(total))},
                ]
            )
        else:
            rows.extend(
                [
                    {"metric": f"pressure_min_t{time_value:g}", "value": float(np.nanmin(pressure[idx]))},
                    {"metric": f"pressure_max_t{time_value:g}", "value": float(np.nanmax(pressure[idx]))},
                    {"metric": f"pressure_mean_t{time_value:g}", "value": float(np.nanmean(pressure[idx]))},
                ]
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    diagnostics = compute_diagnostics(load_snapshots(config))
    out_path = Path(config["paths"]["tables"]) / "diagnostics.csv"
    diagnostics.to_csv(out_path, index=False)
    print(f"Diagnostics saved: {out_path}")


if __name__ == "__main__":
    main()
