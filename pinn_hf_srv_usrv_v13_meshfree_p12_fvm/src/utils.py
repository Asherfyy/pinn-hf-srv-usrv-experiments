"""Shared utilities for the v13 mesh-free P12 PINN."""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_VERSION = "meshfree_p12_line_hf_v13"


def force_cpu(cpu_threads: int) -> torch.device:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(cpu_threads))
    return torch.device("cpu")


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    if str(dtype_name).lower() != "float64":
        raise ValueError(f"v13 expects runtime.dtype='float64', got {dtype_name!r}.")
    return torch.float64


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in ["outputs", "checkpoints", "figures", "logs", "tables"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def tensor_from_numpy(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def pressure_affine_parameters(boundary_cfg: dict[str, Any], eps: float = 1.0e-14) -> dict[str, float]:
    p_initial = float(boundary_cfg["P_t0"])
    p_out = float(boundary_cfg["P_out"])
    if p_initial <= p_out:
        raise ValueError(f"P_t0 must be larger than P_out, got {p_initial:g} <= {p_out:g}.")
    scale = p_initial - p_out
    if scale <= eps:
        raise ValueError(f"pressure scale must be positive, got {scale:g}.")
    return {
        "offset_mpa": p_out,
        "scale_mpa": scale,
        "initial_mpa": p_initial,
        "out_mpa": p_out,
    }


def pressure_component_affine_parameters(config: dict[str, Any]) -> dict[str, np.ndarray]:
    params = pressure_affine_parameters(config["boundary"])
    return {
        "offset_mpa": np.asarray([params["offset_mpa"]], dtype=np.float64),
        "scale_mpa": np.asarray([params["scale_mpa"]], dtype=np.float64),
        "initial_mpa": np.asarray([params["initial_mpa"]], dtype=np.float64),
        "out_mpa": np.asarray([params["out_mpa"]], dtype=np.float64),
    }


def pressure_mpa_to_hat(pressure_mpa: float | np.ndarray | torch.Tensor, boundary_cfg: dict[str, Any]) -> float | np.ndarray | torch.Tensor:
    params = pressure_affine_parameters(boundary_cfg)
    return (pressure_mpa - params["offset_mpa"]) / params["scale_mpa"]


def pressure_hat_to_mpa(u: torch.Tensor, boundary_cfg: dict[str, Any]) -> torch.Tensor:
    if u.ndim != 2 or u.shape[1] != 1:
        raise ValueError(f"pressure_hat_to_mpa expects [N,1], got {tuple(u.shape)}.")
    params = pressure_affine_parameters(boundary_cfg)
    return u * params["scale_mpa"] + params["offset_mpa"]


def dirichlet_target_hat(t: torch.Tensor, boundary_cfg: dict[str, Any]) -> torch.Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1)
    target = torch.exp(-float(boundary_cfg["decay_rate"]) * t)
    return target


def bhp_target_mpa(time_days: float, well_cfg: dict[str, Any]) -> float:
    initial = float(well_cfg["initial_pressure_mpa"])
    final = float(well_cfg["final_pressure_mpa"])
    decay = float(well_cfg["decay_rate_per_day"])
    return final + (initial - final) * np.exp(-decay * float(time_days))


def bhp_component_target_mpa(time_days: float, config: dict[str, Any]) -> np.ndarray:
    return np.asarray([bhp_target_mpa(time_days, config["well"])], dtype=np.float64)


def save_loss_history(rows: Iterable[dict[str, float]], path: str | Path) -> None:
    row_list = list(rows)
    keys: set[str] = set()
    for row in row_list:
        keys.update(row.keys())
    fieldnames = ["epoch", *sorted(k for k in keys if k != "epoch")]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def save_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    row_list = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not row_list:
        path.write_text("", encoding="utf-8")
        return
    keys: set[str] = set()
    for row in row_list:
        keys.update(row.keys())
    fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)
