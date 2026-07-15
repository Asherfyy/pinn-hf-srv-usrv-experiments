"""Shared utilities for the v11 E-PINN/EDFM project."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_VERSION = "epinn_edfm_singlephase_v11"


def force_cpu(cpu_threads: int) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(cpu_threads))
    return torch.device("cpu")


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if str(dtype_name).lower() != "float32":
        raise ValueError(f"v11 only supports float32, got dtype={dtype_name}.")
    return torch.float32


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ["outputs", "checkpoints", "figures", "logs", "tables"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def pressure_affine_parameters(well_cfg: dict[str, Any], eps: float = 1.0e-12) -> dict[str, float]:
    p_initial = float(well_cfg["initial_pressure_mpa"])
    p_final = float(well_cfg["final_pressure_mpa"])
    scale = p_initial - p_final
    if scale <= eps:
        raise ValueError("well.initial_pressure_mpa must be larger than well.final_pressure_mpa.")
    return {"offset_mpa": p_final, "scale_mpa": scale, "initial_mpa": p_initial, "final_mpa": p_final}


def pressure_component_names(config: dict[str, Any]) -> list[str]:
    return [str(value) for value in config.get("pressure", {}).get("components", ["pressure"])]


def pressure_component_affine_parameters(config: dict[str, Any], eps: float = 1.0e-12) -> dict[str, np.ndarray]:
    names = pressure_component_names(config)
    well_cfg = config["well"]
    p_initial = float(well_cfg["initial_pressure_mpa"])
    p_final = float(well_cfg["final_pressure_mpa"])
    if p_initial <= p_final:
        raise ValueError("well.initial_pressure_mpa must be larger than well.final_pressure_mpa.")

    if len(names) == 1:
        initial = np.asarray([p_initial], dtype=np.float64)
        final = np.asarray([p_final], dtype=np.float64)
    elif len(names) == 2:
        ratio = float(config["pressure"]["C13_C12"])
        denominator = 1.0 + ratio
        initial = np.asarray([p_initial / denominator, p_initial * ratio / denominator], dtype=np.float64)
        final = np.asarray([p_final / denominator, p_final * ratio / denominator], dtype=np.float64)
    else:
        fractions = np.asarray(config["pressure"].get("component_fractions", []), dtype=np.float64)
        if fractions.size != len(names) or np.any(fractions <= 0.0):
            raise ValueError("pressure.component_fractions must be positive and match pressure.components for more than two components.")
        fractions = fractions / np.sum(fractions)
        initial = p_initial * fractions
        final = p_final * fractions

    scale = initial - final
    if np.any(scale <= eps):
        raise ValueError(f"component pressure scales must be positive, got {scale}.")
    return {"initial_mpa": initial, "final_mpa": final, "offset_mpa": final, "scale_mpa": scale}


def pressure_mpa_to_hat(pressure_mpa: np.ndarray | torch.Tensor | float, config_or_well: dict[str, Any]) -> np.ndarray | torch.Tensor | float:
    if "pressure" not in config_or_well:
        params = pressure_affine_parameters(config_or_well)
        return (pressure_mpa - params["offset_mpa"]) / params["scale_mpa"]
    params = pressure_component_affine_parameters(config_or_well)
    offset = params["offset_mpa"]
    scale = params["scale_mpa"]
    if isinstance(pressure_mpa, torch.Tensor):
        offset_t = torch.as_tensor(offset, dtype=pressure_mpa.dtype, device=pressure_mpa.device)
        scale_t = torch.as_tensor(scale, dtype=pressure_mpa.dtype, device=pressure_mpa.device)
        return (pressure_mpa - offset_t) / scale_t
    return (pressure_mpa - offset) / scale


def pressure_hat_to_mpa(pressure_hat: torch.Tensor, config_or_well: dict[str, Any]) -> torch.Tensor:
    if "pressure" not in config_or_well:
        params = pressure_affine_parameters(config_or_well)
        return pressure_hat * params["scale_mpa"] + params["offset_mpa"]
    params = pressure_component_affine_parameters(config_or_well)
    offset = torch.as_tensor(params["offset_mpa"], dtype=pressure_hat.dtype, device=pressure_hat.device)
    scale = torch.as_tensor(params["scale_mpa"], dtype=pressure_hat.dtype, device=pressure_hat.device)
    return pressure_hat * scale + offset


def bhp_target_mpa(time_days: float, well_cfg: dict[str, Any]) -> float:
    mode = str(well_cfg.get("target_mode", "exponential")).lower()
    p_initial = float(well_cfg["initial_pressure_mpa"])
    p_final = float(well_cfg["final_pressure_mpa"])
    if mode == "constant":
        return p_final
    if mode == "exponential":
        decay = float(well_cfg.get("decay_rate_per_day", 0.0))
        return p_final + (p_initial - p_final) * float(np.exp(-decay * float(time_days)))
    raise ValueError(f"Unsupported well.target_mode: {mode}")


def bhp_component_target_mpa(time_days: float, config: dict[str, Any]) -> np.ndarray:
    params = pressure_component_affine_parameters(config)
    total_hat = pressure_mpa_to_hat(bhp_target_mpa(time_days, config["well"]), config["well"])
    return params["offset_mpa"] + total_hat * params["scale_mpa"]


def save_csv(rows: Iterable[dict[str, float | str]], path: str | Path) -> None:
    row_list = list(rows)
    keys: set[str] = set()
    for row in row_list:
        keys.update(row.keys())
    fieldnames = sorted(keys)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)
