"""Configuration loading and validation for v11 E-PINN/EDFM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config file must parse to a dictionary.")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "runtime",
        "geometry",
        "grid",
        "edfm",
        "physics",
        "rock",
        "fluid",
        "well",
        "pressure",
        "time_grid",
        "model",
        "training",
        "evaluation",
        "paths",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise KeyError(f"Config missing required sections: {missing}")

    runtime = config["runtime"]
    if str(runtime.get("device", "cpu")).lower() != "cpu":
        raise ValueError("v11 is CPU-only by default; set runtime.device='cpu'.")
    if str(runtime.get("dtype", "float32")).lower() != "float32":
        raise ValueError("v11 currently supports runtime.dtype='float32'.")

    if str(config["physics"].get("mode", "")).lower() != "epinn_edfm_singlephase":
        raise ValueError("v11 requires physics.mode='epinn_edfm_singlephase'.")

    grid = config["grid"]
    if int(grid.get("nx", 0)) <= 0 or int(grid.get("ny", 0)) <= 0:
        raise ValueError("grid.nx and grid.ny must be positive integers.")
    if float(grid.get("thickness_m", 0.0)) <= 0.0:
        raise ValueError("grid.thickness_m must be positive.")

    times = [float(value) for value in config["time_grid"].get("times_days", [])]
    if len(times) < 2:
        raise ValueError("time_grid.times_days must contain at least two values.")
    if times[0] != 0.0:
        raise ValueError("time_grid.times_days must start at 0.0.")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("time_grid.times_days must be strictly increasing.")

    if int(config["edfm"].get("max_dense_elements", 0)) <= 0:
        raise ValueError("edfm.max_dense_elements must be positive.")
    if float(config["rock"].get("total_compressibility_per_MPa", 0.0)) <= 0.0:
        raise ValueError("rock.total_compressibility_per_MPa must be positive.")
    if float(config["fluid"].get("viscosity_cP", 0.0)) <= 0.0:
        raise ValueError("fluid.viscosity_cP must be positive.")

    pressure = config["pressure"]
    components = list(pressure.get("components", []))
    if len(components) != int(config["model"].get("output_dim", len(components))):
        raise ValueError("pressure.components length must match model.output_dim.")
    if len(components) < 1:
        raise ValueError("pressure.components must contain at least one component.")
    if float(pressure.get("C13_C12", 1.0)) <= 0.0 and len(components) == 2:
        raise ValueError("pressure.C13_C12 must be positive for two pressure components.")
    multipliers = list(pressure.get("transmissibility_multipliers", [1.0] * len(components)))
    if len(multipliers) != len(components):
        raise ValueError("pressure.transmissibility_multipliers length must match pressure.components.")
    if any(float(value) <= 0.0 for value in multipliers):
        raise ValueError("pressure.transmissibility_multipliers must all be positive.")
