"""模型评价脚本。

运行方式：
    python -m src.evaluate --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .config import load_config
from .geometry import REGION_NAMES, ReservoirGeometry
from .model import PINNModel
from .utils import (
    ensure_output_dirs,
    force_cpu,
    get_torch_dtype,
    pressure_hat_to_mpa,
    pressure_normalization_mode,
    relative_l2,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="评价 HF/SRV/USRV PINN 模型。")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/final.pt")
    return parser.parse_args()


def _hard_constraint_probe_points(
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    n_x: int = 128,
) -> torch.Tensor:
    """构造评价阶段 hard constraint 诊断点。"""

    geom_cfg = config["geometry"]
    x = torch.linspace(
        float(geom_cfg["main_frac_x_min"]),
        float(geom_cfg["main_frac_x_max"]),
        int(n_x),
        dtype=dtype,
        device=device,
    ).view(-1, 1)
    y_value = 0.5 * (float(geom_cfg["main_frac_y_min"]) + float(geom_cfg["main_frac_y_max"]))
    chunks = []
    for time_value in config.get("evaluation", {}).get("times", [1.0, 10.0, 100.0, 1000.0]):
        y = torch.full_like(x, y_value)
        t = torch.full_like(x, float(time_value))
        chunks.append(torch.cat([x, y, t], dim=1))
    return torch.cat(chunks, dim=0)


def print_hard_constraint_raw_summary(
    model: PINNModel,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """打印 hard constraint 中 raw 与 B_D*raw 的最大绝对值诊断。"""

    if model.constraint_mode() != "dirichlet_hard":
        return
    xyt = _hard_constraint_probe_points(config, device, dtype)
    with torch.no_grad():
        components = model.hard_constraint_components(xyt)
        raw = components["raw"]
        correction = components["correction"]
    print(
        "评价 hard constraint 诊断: "
        f"max_abs_raw_u12={torch.max(torch.abs(raw[:, 0])).item():.4e}, "
        f"max_abs_raw_u13={torch.max(torch.abs(raw[:, 1])).item():.4e}, "
        f"max_abs_BD_raw_u12={torch.max(torch.abs(correction[:, 0])).item():.4e}, "
        f"max_abs_BD_raw_u13={torch.max(torch.abs(correction[:, 1])).item():.4e}",
        flush=True,
    )


def load_trained_model(
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[dict[str, Any], ReservoirGeometry, PINNModel, torch.device, torch.dtype]:
    """加载配置、几何对象和训练好的模型。"""

    config = load_config(config_path)
    runtime_cfg = config["runtime"]
    device = force_cpu(int(runtime_cfg["cpu_threads"]))
    dtype = get_torch_dtype(str(runtime_cfg["dtype"]))
    set_seed(int(runtime_cfg["seed"]))
    ensure_output_dirs(config)

    geometry = ReservoirGeometry(config["geometry"], data_dir=config["paths"]["data"])
    model = PINNModel(geometry, config).to(device=device, dtype=dtype)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # 旧版 PyTorch 没有 weights_only 参数，保留回退以支持更宽的本地环境。
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config")
    if isinstance(checkpoint_config, dict) and "boundary" in checkpoint_config:
        current_mode = pressure_normalization_mode(config["boundary"])
        checkpoint_mode = pressure_normalization_mode(checkpoint_config["boundary"])
        if checkpoint_mode != current_mode:
            raise ValueError(
                "checkpoint 的压力无量纲化模式与当前配置不一致，不能继续预测："
                f"checkpoint={checkpoint_mode}, current={current_mode}。"
                "更改 pressure_normalization 后旧 checkpoint 与新模型输出尺度不兼容，请重新训练。"
            )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print_hard_constraint_raw_summary(model, config, device, dtype)
    return config, geometry, model, device, dtype


def predict_pressure_mpa(
    model: PINNModel,
    xyt: torch.Tensor,
    config: dict[str, Any],
    batch_size: int = 32768,
) -> torch.Tensor:
    """批量预测压力 MPa。

    规则网格点数可能较多，分批预测可以控制 CPU 内存占用。
    """

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, xyt.shape[0], batch_size):
            u = model(xyt[start : start + batch_size])
            outputs.append(pressure_hat_to_mpa(u, config["boundary"]).cpu())
    return torch.cat(outputs, dim=0)


def _mesh_points(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    nx: int,
    ny: int,
) -> np.ndarray:
    """生成矩形区域规则点云。

    后处理使用点云而不是单一结构网格，是为了能把全域、SRV 和极窄 HF 裂缝附近的
    多个不同尺度网格拼在一起，避免 0.01 m 裂缝被 1 m 量级绘图网格完全跳过。
    """

    xs = np.linspace(x_min, x_max, int(nx))
    ys = np.linspace(y_min, y_max, int(ny))
    x_grid, y_grid = np.meshgrid(xs, ys)
    return np.column_stack([x_grid.ravel(), y_grid.ravel()]).astype(np.float32)


def _unique_points(points: np.ndarray) -> np.ndarray:
    """去除重复点，减少三角剖分和预测的冗余计算。"""

    rounded = np.round(points.astype(np.float64), decimals=8)
    _unique, idx = np.unique(rounded, axis=0, return_index=True)
    return points[np.sort(idx)]


def build_adaptive_plot_points(geometry: ReservoirGeometry, config: dict[str, Any]) -> np.ndarray:
    """构造全域粗网格 + SRV/HF 局部加密点云。

    COMSOL 图能显示裂缝，是因为网格对裂缝附近做了几何解析。当前裂缝宽度只有 0.01 m，
    若继续用 241x121 的全域规则网格，基本采不到 HF 内部点。这里保留全域背景点，
    额外叠加 SRV 区和每条裂缝附近的细网格，以便云图反映局部压降结构。
    """

    eval_cfg = config["evaluation"]
    refine_cfg = eval_cfg.get("adaptive_plot", {})
    geom_cfg = config["geometry"]

    coarse_nx = int(refine_cfg.get("coarse_nx", eval_cfg["grid_nx"]))
    coarse_ny = int(refine_cfg.get("coarse_ny", eval_cfg["grid_ny"]))
    points = [
        _mesh_points(
            float(geom_cfg["x_min"]),
            float(geom_cfg["x_max"]),
            float(geom_cfg["y_min"]),
            float(geom_cfg["y_max"]),
            coarse_nx,
            coarse_ny,
        )
    ]

    srv_nx = int(refine_cfg.get("srv_nx", 361))
    srv_ny = int(refine_cfg.get("srv_ny", 181))
    points.append(
        _mesh_points(
            geometry.srv_bg.x_min,
            geometry.srv_bg.x_max,
            geometry.srv_bg.y_min,
            geometry.srv_bg.y_max,
            srv_nx,
            srv_ny,
        )
    )

    hf_long = int(refine_cfg.get("hf_long_axis_points", 600))
    hf_short = int(refine_cfg.get("hf_short_axis_points", 11))
    hf_padding = float(refine_cfg.get("hf_padding_m", 0.05))
    for rect in geometry.hf_rects:
        x0 = max(geometry.domain.x_min, rect.x_min - hf_padding)
        x1 = min(geometry.domain.x_max, rect.x_max + hf_padding)
        y0 = max(geometry.domain.y_min, rect.y_min - hf_padding)
        y1 = min(geometry.domain.y_max, rect.y_max + hf_padding)
        if rect.width >= rect.height:
            nx, ny = hf_long, hf_short
        else:
            nx, ny = hf_short, hf_long
        points.append(_mesh_points(x0, x1, y0, y1, nx, ny))

    all_points = _unique_points(np.vstack(points))
    # 只保留总计算域内部点，避免三角剖分跨越域外空白区域。
    inside = geometry.inside_domain_np(all_points[:, 0], all_points[:, 1])
    return all_points[inside]


def predict_grid_fields(
    model: PINNModel,
    geometry: ReservoirGeometry,
    config: dict[str, Any],
    time_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, np.ndarray]:
    """在自适应点云上预测 P12/P13 并生成区域 mask。

    函数名保留为 `predict_grid_fields`，以兼容已有脚本；内部已经从单一规则网格改为
    局部加密点云。返回的一维 `x/y/P12/P13` 将由 `tricontourf` 绘制。
    """

    xy = build_adaptive_plot_points(geometry, config)
    t_col = np.full((xy.shape[0], 1), float(time_value), dtype=np.float32)
    xyt_np = np.column_stack([xy, t_col]).astype(np.float32)
    xyt = torch.as_tensor(xyt_np, dtype=dtype, device=device)
    pressure = predict_pressure_mpa(model, xyt, config).numpy()
    region = geometry.region_id_np(xy[:, 0], xy[:, 1])
    return {
        "x": xy[:, 0],
        "y": xy[:, 1],
        "P12": pressure[:, 0],
        "P13": pressure[:, 1],
        "region": region,
        "adaptive": np.asarray([1], dtype=np.int8),
    }


def draw_geometry_overlay(ax: plt.Axes, geometry: ReservoirGeometry) -> None:
    """在云图上叠加 SRV 与 HF 边界线。"""

    srv = geometry.srv_bg
    ax.plot(
        [srv.x_min, srv.x_max, srv.x_max, srv.x_min, srv.x_min],
        [srv.y_min, srv.y_min, srv.y_max, srv.y_max, srv.y_min],
        color="white",
        linewidth=1.0,
        linestyle="--",
    )
    for rect in geometry.hf_rects:
        ax.plot(
            [rect.x_min, rect.x_max, rect.x_max, rect.x_min, rect.x_min],
            [rect.y_min, rect.y_min, rect.y_max, rect.y_max, rect.y_min],
            color="black",
            linewidth=0.8,
        )


def save_field_figure(
    field_data: dict[str, np.ndarray],
    geometry: ReservoirGeometry,
    variable: str,
    time_value: float,
    path: str | Path,
) -> None:
    """保存单张二维压力云图。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    x = field_data["x"]
    y = field_data["y"]
    z = field_data[variable]
    if np.ndim(z) == 2:
        contour = ax.contourf(x, y, z, levels=60, cmap="rainbow")
    else:
        # 自适应点云不是矩形数组，使用三角剖分云图。这样 HF/SRV 局部加密点会真实参与
        # 色场插值，比把细裂缝强行落到粗规则网格上更接近 COMSOL 的局部网格思想。
        contour = ax.tricontourf(x, y, z, levels=80, cmap="rainbow")
    draw_geometry_overlay(ax, geometry)
    ax.set_xlim(geometry.domain.x_min, geometry.domain.x_max)
    ax.set_ylim(geometry.domain.y_min, geometry.domain.y_max)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"{variable} at t={time_value:g} d")
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(contour, ax=ax, label="Pressure / MPa")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _region_labels_from_geometry(df: pd.DataFrame, geometry: ReservoirGeometry) -> pd.Series:
    """补充或规范 COMSOL 数据中的 region 列。"""

    if "region" in df.columns:
        raw = df["region"].astype(str).str.upper()
        mapping = {
            "0": "HF",
            "1": "SRV",
            "2": "USRV",
            "REGION_HF": "HF",
            "REGION_SRV": "SRV",
            "REGION_USRV": "USRV",
        }
        return raw.replace(mapping)
    region_id = geometry.region_id_np(df["x"].to_numpy(), df["y"].to_numpy())
    names = np.array(["OUTSIDE", *REGION_NAMES], dtype=object)
    mapped = np.where(region_id < 0, "OUTSIDE", np.asarray(REGION_NAMES, dtype=object)[region_id])
    return pd.Series(mapped, index=df.index)


def compute_comsol_error_table(
    model: PINNModel,
    geometry: ReservoirGeometry,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> pd.DataFrame:
    """如果存在 COMSOL 快照数据，则计算相对 L2 误差表。"""

    path = Path(config["paths"]["data"]) / "comsol_snapshots.csv"
    columns = ["region", "E12_relative", "E13_relative", "n_points"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame([{"region": "ALL", "E12_relative": np.nan, "E13_relative": np.nan, "n_points": 0}])
    df = pd.read_csv(path)
    required = {"x", "y", "t", "Pg1", "Pg2"}
    if len(df) == 0 or not required.issubset(df.columns):
        return pd.DataFrame([{"region": "ALL", "E12_relative": np.nan, "E13_relative": np.nan, "n_points": 0}])

    xyt = torch.as_tensor(df[["x", "y", "t"]].to_numpy(), dtype=dtype, device=device)
    pred = predict_pressure_mpa(model, xyt, config).numpy()
    ref12 = df["Pg1"].to_numpy(dtype=float)
    ref13 = df["Pg2"].to_numpy(dtype=float)
    labels = _region_labels_from_geometry(df, geometry)

    rows: list[dict[str, float | str | int]] = []
    masks = {"ALL": np.ones(len(df), dtype=bool)}
    for name in REGION_NAMES:
        masks[name] = labels.to_numpy() == name
    for region, mask in masks.items():
        if not np.any(mask):
            rows.append({"region": region, "E12_relative": np.nan, "E13_relative": np.nan, "n_points": 0})
            continue
        rows.append(
            {
                "region": region,
                "E12_relative": relative_l2(pred[mask, 0], ref12[mask]),
                "E13_relative": relative_l2(pred[mask, 1], ref13[mask]),
                "n_points": int(np.sum(mask)),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def save_region_mask_table(field_data: dict[str, np.ndarray], path: str | Path) -> None:
    """保存规则网格区域 mask，便于后处理检查分区。"""

    df = pd.DataFrame(
        {
            "x": field_data["x"].ravel(),
            "y": field_data["y"].ravel(),
            "region_id": field_data["region"].ravel(),
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    """评价主流程。"""

    args = parse_args()
    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)

    first_field: dict[str, np.ndarray] | None = None
    for time_value in config["evaluation"]["times"]:
        field = predict_grid_fields(model, geometry, config, float(time_value), device, dtype)
        if first_field is None:
            first_field = field
        for variable in ["P12", "P13"]:
            save_field_figure(
                field,
                geometry,
                variable,
                float(time_value),
                Path(config["paths"]["figures"]) / f"field_{variable}_t{float(time_value):g}.png",
            )

    if first_field is not None:
        save_region_mask_table(first_field, Path(config["paths"]["tables"]) / "region_mask.csv")

    error_df = compute_comsol_error_table(model, geometry, config, device, dtype)
    error_path = Path(config["paths"]["tables"]) / "error_metrics.csv"
    error_df.to_csv(error_path, index=False)
    print(f"评价完成，误差表已保存到 {error_path}")


if __name__ == "__main__":
    main()
