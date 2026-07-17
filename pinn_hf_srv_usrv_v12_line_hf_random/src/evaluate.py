"""评价、诊断与自适应点云工具。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import torch

from .base_model import attach_base_model_from_config
from .config import load_config
from .geometry import REGION_HF, REGION_NAMES, REGION_SRV, REGION_USRV, Rect, ReservoirGeometry
from .losses import compute_dirichlet_loss, compute_hf_junction_loss, compute_hf_main_link_loss, compute_hf_secondary_link_loss, compute_interface_loss, compute_pde_loss
from .model import PINNModel
from .physics import neumann_normal_derivative
from .sampler import ReservoirSampler
from .utils import PROJECT_VERSION, ensure_output_dirs, force_cpu, get_torch_dtype, pressure_hat_to_mpa, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate v12 line-HF partitioned-MLP diagnostics.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/final.pt")
    return parser.parse_args()


def load_trained_model(config_path: str | Path, checkpoint_path: str | Path) -> tuple[dict[str, Any], ReservoirGeometry, PINNModel, torch.device, torch.dtype]:
    """加载配置、几何和 checkpoint。"""

    config = load_config(config_path)
    runtime = config["runtime"]
    set_seed(int(runtime["seed"]))
    device = force_cpu(int(runtime["cpu_threads"]))
    dtype = get_torch_dtype(str(runtime["dtype"]))
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"])
    model = PINNModel(config).to(device=device, dtype=dtype)
    attach_base_model_from_config(model, config, device, dtype, config_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("project_version") != PROJECT_VERSION:
        raise ValueError(f"checkpoint project_version 不匹配: {checkpoint.get('project_version')} != {PROJECT_VERSION}")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return config, geometry, model, device, dtype


def predict_pressure_mpa(model: PINNModel, xyt: torch.Tensor, config: dict[str, Any], batch_size: int = 32768) -> torch.Tensor:
    """分批预测物理压力 MPa。"""

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, xyt.shape[0], batch_size):
            u = model(xyt[start : start + batch_size])
            outputs.append(pressure_hat_to_mpa(u, config["boundary"]).cpu())
    return torch.cat(outputs, dim=0)


def _mesh_points(x_min: float, x_max: float, y_min: float, y_max: float, nx: int, ny: int) -> np.ndarray:
    """生成矩形规则点云。"""

    xs = np.linspace(float(x_min), float(x_max), int(nx))
    ys = np.linspace(float(y_min), float(y_max), int(ny))
    x_grid, y_grid = np.meshgrid(xs, ys)
    return np.column_stack([x_grid.ravel(), y_grid.ravel()]).astype(np.float64)


def _unique_points(points: np.ndarray) -> np.ndarray:
    rounded = np.round(points.astype(np.float64), decimals=8)
    _unique, idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(idx)]


def build_adaptive_plot_points(geometry: ReservoirGeometry, config: dict[str, Any]) -> np.ndarray:
    """构造全域粗网格 + SRV/HF/过渡带加密点云。"""

    eval_cfg = config["evaluation"]
    plot_cfg = eval_cfg["adaptive_plot"]
    points = [
        _mesh_points(geometry.domain.x_min, geometry.domain.x_max, geometry.domain.y_min, geometry.domain.y_max, int(plot_cfg["coarse_nx"]), int(plot_cfg["coarse_ny"])),
        _mesh_points(geometry.srv_bg.x_min, geometry.srv_bg.x_max, geometry.srv_bg.y_min, geometry.srv_bg.y_max, int(plot_cfg["srv_nx"]), int(plot_cfg["srv_ny"])),
    ]
    hf_long = int(plot_cfg["hf_long_axis_points"])
    hf_short = int(plot_cfg["hf_short_axis_points"])
    padding = float(plot_cfg["hf_padding_m"])
    for rect in geometry.hf_rects:
        expanded = rect.expanded(padding, geometry.domain)
        nx, ny = (hf_long, hf_short) if rect.width >= rect.height else (hf_short, hf_long)
        points.append(_mesh_points(expanded.x_min, expanded.x_max, expanded.y_min, expanded.y_max, nx, ny))
    band_points = int(plot_cfg["transition_band_points"])
    for rect in geometry.srv_usrv_band_rects(float(config["sampler"]["srv_usrv_band_width_m"])):
        points.append(_mesh_points(max(rect.x_min, geometry.domain.x_min), min(rect.x_max, geometry.domain.x_max), max(rect.y_min, geometry.domain.y_min), min(rect.y_max, geometry.domain.y_max), band_points, max(5, band_points // 4)))
    all_points = _unique_points(np.vstack(points))
    inside = geometry.inside_domain_np(all_points[:, 0], all_points[:, 1])
    return all_points[inside]


def predict_field(model: PINNModel, geometry: ReservoirGeometry, config: dict[str, Any], time_value: float, device: torch.device, dtype: torch.dtype) -> dict[str, np.ndarray]:
    """在自适应点云上预测 P12/P13/Ptotal。"""

    xy = build_adaptive_plot_points(geometry, config)
    t_col = np.full((xy.shape[0], 1), float(time_value), dtype=np.float64)
    xyt = torch.as_tensor(np.column_stack([xy, t_col]), dtype=dtype, device=device)
    pressure = predict_pressure_mpa(model, xyt, config).numpy()
    region = geometry.region_id_np(xy[:, 0], xy[:, 1])
    return {
        "x": xy[:, 0],
        "y": xy[:, 1],
        "P12": pressure[:, 0],
        "P13": pressure[:, 1],
        "Ptotal": pressure[:, 0] + pressure[:, 1],
        "region": region,
    }


def triangulation_for_region(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    region: np.ndarray,
    target_region: int,
    config: dict[str, Any],
) -> tuple[mtri.Triangulation, np.ndarray] | None:
    """按区域创建三角剖分，并 mask 超长边三角形。"""

    if target_region == REGION_HF:
        return None
    mask = region == int(target_region)
    if int(np.sum(mask)) < 3:
        return None
    x_sub = x[mask]
    y_sub = y[mask]
    values_sub = values[mask]
    tri = mtri.Triangulation(x_sub, y_sub)
    tri_cfg = config["evaluation"]["triangulation"]
    if target_region == REGION_HF:
        max_edge = float(tri_cfg["max_edge_length_hf_m"])
    elif target_region == REGION_SRV:
        max_edge = float(tri_cfg["max_edge_length_srv_m"])
    else:
        max_edge = float(tri_cfg["max_edge_length_global_m"])
    triangles = tri.triangles
    pts = np.column_stack([x_sub, y_sub])
    p0 = pts[triangles[:, 0]]
    p1 = pts[triangles[:, 1]]
    p2 = pts[triangles[:, 2]]
    max_len = np.maximum.reduce([np.linalg.norm(p0 - p1, axis=1), np.linalg.norm(p1 - p2, axis=1), np.linalg.norm(p2 - p0, axis=1)])
    tri.set_mask(max_len > max_edge)
    return tri, values_sub


def compute_diagnostics(config: dict[str, Any], geometry: ReservoirGeometry, model: PINNModel, device: torch.device, dtype: torch.dtype) -> pd.DataFrame:
    """生成训练后诊断表。

    这里的界面跳跃如果后续需要可以扩展为观察指标，但绝不会进入训练 loss。
    """

    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=int(config["runtime"]["seed"]) + 99)
    samples = sampler.sample_all()
    loss_pde, pde_diag = compute_pde_loss(model, samples["pde"], config["physics"], config)
    loss_interface_pressure, loss_interface_flux, interface_diag = compute_interface_loss(model, samples, geometry, config["physics"], config)
    loss_hf_main_link, hf_main_diag = compute_hf_main_link_loss(model, samples, config)
    loss_hf_secondary_link, hf_secondary_diag = compute_hf_secondary_link_loss(model, samples, config)
    loss_hf_junction, hf_junction_diag = compute_hf_junction_loss(model, samples, config)
    loss_dir = compute_dirichlet_loss(model, samples["dirichlet"], config)
    dn12, dn13 = neumann_normal_derivative(model, samples["neumann"]["xyt"], samples["neumann"]["normal"])
    rows: list[dict[str, float | str | int]] = [
        {"metric": "loss_pde", "value": float(loss_pde.detach().cpu())},
        {"metric": "loss_interface_pressure", "value": float(loss_interface_pressure.detach().cpu())},
        {"metric": "loss_interface_flux", "value": float(loss_interface_flux.detach().cpu())},
        {"metric": "loss_hf_main_link", "value": float(loss_hf_main_link.detach().cpu())},
        {"metric": "loss_hf_secondary_link", "value": float(loss_hf_secondary_link.detach().cpu())},
        {"metric": "loss_hf_junction", "value": float(loss_hf_junction.detach().cpu())},
        {"metric": "dirichlet_rmse", "value": float(torch.sqrt(loss_dir.detach()).cpu())},
        {"metric": "neumann_du12_rms", "value": float(torch.sqrt(torch.mean(dn12.detach() ** 2)).cpu())},
        {"metric": "neumann_du13_rms", "value": float(torch.sqrt(torch.mean(dn13.detach() ** 2)).cpu())},
    ]
    for key, value in pde_diag.items():
        if key.startswith("rms_"):
            rows.append({"metric": key, "value": float(value.detach().cpu())})
    for key, value in interface_diag.items():
        if key.startswith("rms_") or key.startswith("n_interface_valid"):
            rows.append({"metric": key, "value": float(value.detach().cpu())})
    for key, value in hf_main_diag.items():
        rows.append({"metric": key, "value": float(value.detach().cpu())})
    for key, value in hf_secondary_diag.items():
        rows.append({"metric": key, "value": float(value.detach().cpu())})
    for key, value in hf_junction_diag.items():
        rows.append({"metric": key, "value": float(value.detach().cpu())})

    field = predict_field(model, geometry, config, float(config["evaluation"]["times"][-1]), device, dtype)
    finite_mask = np.isfinite(field["P12"]) & np.isfinite(field["P13"]) & np.isfinite(field["Ptotal"])
    rows.append({"metric": "negative_pressure_points", "value": int(np.sum((field["P12"] < 0.0) | (field["P13"] < 0.0) | (field["Ptotal"] < 0.0)))})
    rows.append({"metric": "nonfinite_pressure_points", "value": int(np.sum(~finite_mask))})
    for region_id, region_name in [(REGION_HF, "HF"), (REGION_SRV, "SRV"), (REGION_USRV, "USRV")]:
        mask = field["region"] == region_id
        if not np.any(mask):
            continue
        for variable in ["P12", "P13", "Ptotal"]:
            rows.append({"metric": f"{region_name}_{variable}_min", "value": float(np.nanmin(field[variable][mask]))})
            rows.append({"metric": f"{region_name}_{variable}_max", "value": float(np.nanmax(field[variable][mask]))})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    diagnostics = compute_diagnostics(config, geometry, model, device, dtype)
    out_path = Path(config["paths"]["tables"]) / "diagnostics.csv"
    diagnostics.to_csv(out_path, index=False)
    print(f"诊断表已保存: {out_path}")


if __name__ == "__main__":
    main()
