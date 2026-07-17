"""PINN/FVM comparison fields and figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch

from .evaluate import build_adaptive_plot_points, predict_pressure_mpa
from .fvm_reference import FVMReference
from .model import PINNModel


def build_comparison_field(
    config: dict[str, Any],
    geometry: Any,
    model: PINNModel,
    reference: FVMReference,
    time_days: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    """Evaluate PINN, FVM, and PINN-FVM error on one adaptive point cloud."""

    xy = build_adaptive_plot_points(geometry, config)
    t_col = np.full((xy.shape[0], 1), float(time_days), dtype=np.float64)
    xyt = torch.as_tensor(np.column_stack([xy, t_col]), dtype=dtype, device=device)
    pinn = predict_pressure_mpa(model, xyt, config).numpy()[:, 0]
    with torch.no_grad():
        fvm = reference.pressure_mpa_at(xyt, config).detach().cpu().numpy()[:, 0]
    region = geometry.region_id_np(xy[:, 0], xy[:, 1])
    return {
        "x": xy[:, 0],
        "y": xy[:, 1],
        "time_days": np.full((xy.shape[0],), float(time_days), dtype=np.float64),
        "P12_PINN": pinn,
        "P12_FVM": fvm,
        "Error_PINN_minus_FVM": pinn - fvm,
        "Abs_Error": np.abs(pinn - fvm),
        "region": region,
    }


def error_summary(field: dict[str, np.ndarray]) -> dict[str, float]:
    err = np.asarray(field["Error_PINN_minus_FVM"], dtype=np.float64)
    pinn = np.asarray(field["P12_PINN"], dtype=np.float64)
    fvm = np.asarray(field["P12_FVM"], dtype=np.float64)
    finite = np.isfinite(err) & np.isfinite(pinn) & np.isfinite(fvm)
    if not np.any(finite):
        return {"count": 0.0, "mae_mpa": float("nan"), "rmse_mpa": float("nan"), "max_abs_mpa": float("nan")}
    values = err[finite]
    return {
        "count": float(values.size),
        "mae_mpa": float(np.mean(np.abs(values))),
        "rmse_mpa": float(np.sqrt(np.mean(values * values))),
        "max_abs_mpa": float(np.max(np.abs(values))),
        "pinn_min_mpa": float(np.min(pinn[finite])),
        "pinn_max_mpa": float(np.max(pinn[finite])),
        "fvm_min_mpa": float(np.min(fvm[finite])),
        "fvm_max_mpa": float(np.max(fvm[finite])),
    }


def save_comparison_figure(field: dict[str, np.ndarray], geometry: Any, time_days: float, path: str | Path) -> None:
    """Save a three-panel PINN/FVM/error contour figure."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), dpi=170, constrained_layout=True)
    tri = mtri.Triangulation(field["x"], field["y"])
    values = [
        ("PINN / MPa", field["P12_PINN"], "rainbow", None),
        ("FVM / MPa", field["P12_FVM"], "rainbow", None),
        ("PINN - FVM / MPa", field["Error_PINN_minus_FVM"], "coolwarm", "symmetric"),
    ]
    for ax, (title, data, cmap, mode) in zip(axes, values):
        levels = _levels(np.asarray(data, dtype=np.float64), mode)
        contour = None
        if levels is not None and np.sum(np.isfinite(data)) >= 3:
            contour = ax.tricontourf(tri, data, levels=levels, cmap=cmap)
        geometry.draw_overlay(ax)
        ax.set_xlim(geometry.domain.x_min, geometry.domain.x_max)
        ax.set_ylim(geometry.domain.y_min, geometry.domain.y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title(title)
        if contour is not None:
            fig.colorbar(contour, ax=ax)
    summary = error_summary(field)
    fig.suptitle(
        f"PINN/FVM comparison at t={float(time_days):g} d, "
        f"RMSE={summary['rmse_mpa']:.3g} MPa, MaxAbs={summary['max_abs_mpa']:.3g} MPa"
    )
    fig.savefig(path)
    plt.close(fig)


def _levels(values: np.ndarray, mode: str | None, n_levels: int = 80) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    if mode == "symmetric":
        vmax = max(float(np.max(np.abs(finite))), 1.0e-12)
        return np.linspace(-vmax, vmax, int(n_levels))
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    pad = max(abs(v_max - v_min), 1.0) * 1.0e-6
    v_min -= pad
    v_max += pad
    if np.isclose(v_min, v_max):
        pad = max(abs(v_min), 1.0) * 1.0e-6
        v_min -= pad
        v_max += pad
    return np.linspace(v_min, v_max, int(n_levels))
