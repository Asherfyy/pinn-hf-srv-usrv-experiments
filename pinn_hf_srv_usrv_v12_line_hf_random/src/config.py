"""Configuration loading and lightweight validation for v12."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and run minimal consistency checks."""

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
    """Check the v12 assumptions that other modules rely on."""

    required = [
        "runtime",
        "geometry",
        "physics",
        "boundary",
        "sampler",
        "model",
        "training",
        "loss_weights",
        "evaluation",
        "paths",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise KeyError(f"Config is missing required sections: {missing}")

    runtime_cfg = config["runtime"]
    if str(runtime_cfg.get("device", "cpu")).lower() != "cpu":
        raise ValueError("v12 is CPU-only; set runtime.device to 'cpu'.")
    if str(runtime_cfg.get("dtype", "float64")).lower() != "float64":
        raise ValueError("v12 expects runtime.dtype='float64' for the current extreme coefficient scales.")

    model_cfg = config["model"]
    if int(model_cfg.get("input_dim", -1)) != 3:
        raise ValueError("model.input_dim must be 3 for x_hat/y_hat/t_hat.")
    if int(model_cfg.get("subnet_input_dim", 3)) not in {3, 5}:
        raise ValueError("model.subnet_input_dim must be 3 or 5.")
    constraint_mode = str(model_cfg.get("constraint_mode", "")).lower()
    if constraint_mode not in {"ic_hard", "ic_base_correction"}:
        raise ValueError("v12 supports model.constraint_mode='ic_hard' or 'ic_base_correction'.")
    if float(model_cfg.get("base_time_lag_days", 0.0)) < 0.0:
        raise ValueError("model.base_time_lag_days must be non-negative.")
    if float(model_cfg.get("correction_envelope_power", 1.0)) <= 0.0:
        raise ValueError("model.correction_envelope_power must be positive.")

    physics_mode = str(config["physics"].get("mode", "")).lower()
    if physics_mode != "effective_diffusion":
        raise ValueError("v12 requires physics.mode='effective_diffusion'.")

    sampler_cfg = config["sampler"]
    sampling_mode = str(sampler_cfg.get("sampling_mode", "random")).lower()
    if sampling_mode not in {"random", "uniform"}:
        raise ValueError("sampler.sampling_mode must be 'random' or 'uniform'.")
    time_sampling_mode = str(sampler_cfg.get("time_sampling_mode", sampling_mode)).lower()
    if time_sampling_mode not in {"random", "uniform"}:
        raise ValueError("sampler.time_sampling_mode must be 'random' or 'uniform'.")
    time_pairing_mode = str(sampler_cfg.get("time_pairing_mode", "paired")).lower()
    if time_pairing_mode not in {"paired", "cartesian"}:
        raise ValueError("sampler.time_pairing_mode must be 'paired' or 'cartesian'.")
    if time_pairing_mode == "cartesian":
        for key in ["n_time_pde", "n_time_boundary", "n_time_interface", "n_time_link"]:
            if int(sampler_cfg.get(key, sampler_cfg.get("n_time_collocation", 1))) <= 0:
                raise ValueError(f"sampler.{key} must be positive when time_pairing_mode is 'cartesian'.")
