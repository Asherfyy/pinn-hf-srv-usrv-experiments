"""Evaluate v10 RDFM/I-PINN node snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .utils import ensure_output_dirs, pressure_hat_to_mpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diagnostics for v10 RDFM/I-PINN snapshots.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def load_snapshots(config: dict[str, Any]) -> dict[str, np.ndarray]:
    path = Path(config["paths"]["outputs"]) / "snapshots.npz"
    if not path.exists():
        raise FileNotFoundError(f"Snapshot file does not exist: {path}. Run training first.")
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def pressure_snapshots_mpa(u: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    flat = torch.as_tensor(u.reshape(-1, 2), dtype=torch.float32)
    pressure = pressure_hat_to_mpa(flat, config["boundary"]).numpy()
    return pressure.reshape(u.shape)


def compute_diagnostics(config: dict[str, Any], snapshots: dict[str, np.ndarray]) -> pd.DataFrame:
    times = snapshots["times_days"]
    u = snapshots["u"]
    pressure = pressure_snapshots_mpa(u, config)
    ptotal = pressure[:, :, 0] + pressure[:, :, 1]
    rows: list[dict[str, float | str]] = []

    rows.append({"metric": "snapshot_count", "value": float(len(times))})
    rows.append({"metric": "node_count", "value": float(snapshots["node_xy"].shape[0])})
    rows.append({"metric": "nonfinite_pressure_points", "value": float(np.sum(~np.isfinite(pressure)))})
    rows.append({"metric": "negative_pressure_points", "value": float(np.sum(pressure < 0.0))})

    for idx, time_value in enumerate(times):
        rows.extend(
            [
                {"metric": f"P12_min_t{time_value:g}", "value": float(np.nanmin(pressure[idx, :, 0]))},
                {"metric": f"P12_max_t{time_value:g}", "value": float(np.nanmax(pressure[idx, :, 0]))},
                {"metric": f"P13_min_t{time_value:g}", "value": float(np.nanmin(pressure[idx, :, 1]))},
                {"metric": f"P13_max_t{time_value:g}", "value": float(np.nanmax(pressure[idx, :, 1]))},
                {"metric": f"Ptotal_min_t{time_value:g}", "value": float(np.nanmin(ptotal[idx]))},
                {"metric": f"Ptotal_max_t{time_value:g}", "value": float(np.nanmax(ptotal[idx]))},
            ]
        )

    loss_path = Path(config["paths"]["logs"]) / "loss_history.csv"
    if loss_path.exists():
        history = pd.read_csv(loss_path)
        if not history.empty and "loss_total" in history.columns:
            rows.append({"metric": "final_loss_total", "value": float(history["loss_total"].iloc[-1])})
            rows.append({"metric": "best_loss_total", "value": float(history["loss_total"].min())})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    snapshots = load_snapshots(config)
    diagnostics = compute_diagnostics(config, snapshots)
    out_path = Path(config["paths"]["tables"]) / "diagnostics.csv"
    diagnostics.to_csv(out_path, index=False)
    print(f"Diagnostics saved: {out_path}")


if __name__ == "__main__":
    main()
