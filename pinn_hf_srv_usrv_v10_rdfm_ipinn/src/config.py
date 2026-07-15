"""Configuration loading and validation for v10 RDFM/I-PINN."""

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
        "mesh",
        "time_grid",
        "physics",
        "boundary",
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
        raise ValueError("v10 is CPU-only; set runtime.device='cpu'.")
    if str(runtime.get("dtype", "float32")).lower() != "float32":
        raise ValueError("v10 currently supports runtime.dtype='float32'.")

    physics_mode = str(config["physics"].get("mode", "")).lower()
    if physics_mode != "rdfm_ipinn":
        raise ValueError("v10 requires physics.mode='rdfm_ipinn'.")

    mesh = config["mesh"]
    if int(mesh.get("nx", 0)) <= 0 or int(mesh.get("ny", 0)) <= 0:
        raise ValueError("mesh.nx and mesh.ny must be positive integers.")

    times = [float(value) for value in config["time_grid"].get("times_days", [])]
    if len(times) < 2:
        raise ValueError("time_grid.times_days must contain at least two values.")
    if times[0] != 0.0:
        raise ValueError("time_grid.times_days must start at 0.0.")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("time_grid.times_days must be strictly increasing.")

    model = config["model"]
    if int(model.get("input_dim", -1)) != 2:
        raise ValueError("v10 model.input_dim must be 2 for [x_hat, y_hat].")
    if int(model.get("output_dim", -1)) != 2:
        raise ValueError("v10 model.output_dim must be 2 for [u12, u13].")
    if str(model.get("activation", "")).lower() not in {"relu", "tanh", "silu"}:
        raise ValueError("model.activation must be one of relu, tanh, or silu.")

    training = config["training"]
    if int(training.get("epochs_per_step", 0)) <= 0:
        raise ValueError("training.epochs_per_step must be positive.")
    if float(training.get("learning_rate", 0.0)) <= 0.0:
        raise ValueError("training.learning_rate must be positive.")
