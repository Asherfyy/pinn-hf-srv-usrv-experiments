"""IDE 直接运行：计算并绘制 HF/SRV/USRV 各区域累产气量随时间曲线。

本脚本基于当前 PINN 模型输出的两种气体分压 P12/P13 进行后处理。由于项目
配置中尚未包含厚度、孔隙度、气体状态方程和地面体积换算系数，这里计算的是
“储量亏空等效累产量”：

    G_i,R(t) = C * ∫_R A_i,R * max(P_i,initial,R - P_i(x,y,t), 0) dA

其中 i 为 P12/P13 两个组分，R 为 HF/SRV/USRV 区域，A 为当前 PDE 中的储容
系数，C 默认为 1。若后续给出真实 PVT/厚度/孔隙度换算关系，可通过
`--conversion-factor` 统一乘上物理换算系数。

输出：
    outputs/tables/regional_cumulative_gas.csv
    outputs/figures/regional_cumulative_gas.png
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "final.pt"
DEFAULT_TABLE = PROJECT_ROOT / "outputs" / "tables" / "regional_cumulative_gas.csv"
DEFAULT_FIGURE = PROJECT_ROOT / "outputs" / "figures" / "regional_cumulative_gas.png"


@dataclass
class QuadratureRegion:
    """区域积分点与面积权重。"""

    name: str
    xy: "np.ndarray"
    weight: "np.ndarray"


def relaunch_with_venv_if_needed() -> None:
    """如果 IDE 没有选中项目虚拟环境，则自动切换到 `.venv` 后重启脚本。"""

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    """解析命令行参数；IDE 直接运行时使用默认配置。"""

    parser = argparse.ArgumentParser(description="计算并绘制各区域累产气量随时间曲线。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML 配置文件路径。")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="训练好的模型 checkpoint。")
    parser.add_argument("--output-table", type=Path, default=DEFAULT_TABLE, help="累产曲线 CSV 输出路径。")
    parser.add_argument("--output-figure", type=Path, default=DEFAULT_FIGURE, help="累产曲线图片输出路径。")
    parser.add_argument("--n-times", type=int, default=101, help="时间采样点数。")
    parser.add_argument("--time-min", type=float, default=None, help="最小时间，默认使用 sampler.t_min。")
    parser.add_argument("--time-max", type=float, default=None, help="最大时间，默认使用 sampler.t_max。")
    parser.add_argument("--region-nx", type=int, default=260, help="SRV/USRV 面积分网 x 方向点数。")
    parser.add_argument("--region-ny", type=int, default=140, help="SRV/USRV 面积分网 y 方向点数。")
    parser.add_argument("--hf-long-axis-points", type=int, default=500, help="HF 裂缝长轴方向积分点数。")
    parser.add_argument("--hf-short-axis-points", type=int, default=8, help="HF 裂缝短轴方向积分点数。")
    parser.add_argument("--batch-size", type=int, default=32768, help="模型预测批大小。")
    parser.add_argument("--conversion-factor", type=float, default=1.0, help="累产量统一换算系数，默认 1。")
    parser.add_argument("--no-clamp-depletion", action="store_true", help="不截断负亏空，保留模型压力回升造成的负贡献。")
    parser.add_argument("--show", action="store_true", help="保存后弹出图窗显示。")
    return parser.parse_args()


def configure_matplotlib() -> None:
    """设置 Matplotlib 中文字体。"""

    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def midpoint_axis(v_min: float, v_max: float, n: int):
    """生成一维中点积分坐标。"""

    import numpy as np

    edges = np.linspace(float(v_min), float(v_max), int(n) + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def rect_midpoint_quadrature(x_min: float, x_max: float, y_min: float, y_max: float, nx: int, ny: int):
    """生成矩形区域中点积分点和等面积权重。"""

    import numpy as np

    xs = midpoint_axis(x_min, x_max, nx)
    ys = midpoint_axis(y_min, y_max, ny)
    x_grid, y_grid = np.meshgrid(xs, ys)
    xy = np.column_stack([x_grid.ravel(), y_grid.ravel()]).astype(np.float32)
    cell_area = (float(x_max) - float(x_min)) * (float(y_max) - float(y_min)) / (int(nx) * int(ny))
    weight = np.full(xy.shape[0], cell_area, dtype=np.float64)
    return xy, weight


def filter_region(geometry, xy, weight, region_id: int):
    """按几何分区筛选积分点。"""

    mask = geometry.region_id_np(xy[:, 0], xy[:, 1]) == int(region_id)
    return xy[mask], weight[mask]


def build_quadrature_regions(geometry, args: argparse.Namespace) -> dict[str, QuadratureRegion]:
    """构造 HF/SRV/USRV 三个区域的面积积分点。"""

    import numpy as np

    from src.geometry import REGION_HF, REGION_SRV, REGION_USRV

    hf_xy_parts = []
    hf_weight_parts = []
    for rect in geometry.hf_rects:
        if rect.width >= rect.height:
            nx, ny = int(args.hf_long_axis_points), int(args.hf_short_axis_points)
        else:
            nx, ny = int(args.hf_short_axis_points), int(args.hf_long_axis_points)
        xy, weight = rect_midpoint_quadrature(rect.x_min, rect.x_max, rect.y_min, rect.y_max, nx, ny)
        xy, weight = filter_region(geometry, xy, weight, REGION_HF)
        hf_xy_parts.append(xy)
        hf_weight_parts.append(weight)

    hf_xy = np.vstack(hf_xy_parts) if hf_xy_parts else np.empty((0, 2), dtype=np.float32)
    hf_weight = np.concatenate(hf_weight_parts) if hf_weight_parts else np.empty((0,), dtype=np.float64)

    srv_xy_all, srv_weight_all = rect_midpoint_quadrature(
        geometry.srv_bg.x_min,
        geometry.srv_bg.x_max,
        geometry.srv_bg.y_min,
        geometry.srv_bg.y_max,
        int(args.region_nx),
        int(args.region_ny),
    )
    srv_xy, srv_weight = filter_region(geometry, srv_xy_all, srv_weight_all, REGION_SRV)

    domain = geometry.domain
    usrv_xy_all, usrv_weight_all = rect_midpoint_quadrature(
        domain.x_min,
        domain.x_max,
        domain.y_min,
        domain.y_max,
        int(args.region_nx),
        int(args.region_ny),
    )
    usrv_xy, usrv_weight = filter_region(geometry, usrv_xy_all, usrv_weight_all, REGION_USRV)

    return {
        "HF": QuadratureRegion("HF", hf_xy, hf_weight),
        "SRV": QuadratureRegion("SRV", srv_xy, srv_weight),
        "USRV": QuadratureRegion("USRV", usrv_xy, usrv_weight),
    }


def pressure_to_hat_array(p12_mpa, p13_mpa, boundary_cfg: dict[str, Any]):
    """把物理压力 MPa 转为当前模型使用的无量纲压力。"""

    from src.utils import pressure_mpa_to_hat

    # 统一调用公共转换函数，避免累产后处理继续沿用旧的共享 p_scale 逻辑。
    return pressure_mpa_to_hat(p12_mpa, p13_mpa, boundary_cfg)


def initial_pair_for_region(config: dict[str, Any], region_name: str) -> tuple[float, float]:
    """返回指定区域的 P12/P13 初始压力，优先使用 COMSOL 参数名。"""

    from src.utils import initial_pressure_pair_mpa

    return initial_pressure_pair_mpa(config, region_name.lower())


def storage_coeff_for_region(physics_cfg: dict[str, Any], region_name: str) -> float:
    """返回指定区域的 COMSOL Fai 储容系数，兼容旧 A12 写法。"""

    key = f"Fai_{region_name}"
    if key in physics_cfg:
        return float(physics_cfg[key])
    return float(physics_cfg["A12"][region_name])


def predict_region_pressure(model, region: QuadratureRegion, time_value: float, config, device, dtype, batch_size: int):
    """预测单个区域、单个时刻的 P12/P13 物理压力 MPa。"""

    import numpy as np
    import torch

    from src.evaluate import predict_pressure_mpa

    t_col = np.full((region.xy.shape[0], 1), float(time_value), dtype=np.float32)
    xyt_np = np.column_stack([region.xy, t_col]).astype(np.float32)
    xyt = torch.as_tensor(xyt_np, dtype=dtype, device=device)
    return predict_pressure_mpa(model, xyt, config, batch_size=batch_size).numpy()


def compute_cumulative_table(config, geometry, model, device, dtype, args: argparse.Namespace):
    """计算各区域、各时刻的等效累产量表。"""

    import numpy as np
    import pandas as pd

    regions = build_quadrature_regions(geometry, args)
    time_min = float(config["sampler"]["t_min"] if args.time_min is None else args.time_min)
    time_max = float(config["sampler"]["t_max"] if args.time_max is None else args.time_max)
    times = np.linspace(time_min, time_max, int(args.n_times))

    rows = []
    boundary_cfg = config["boundary"]
    physics_cfg = config["physics"]
    clamp_depletion = not bool(args.no_clamp_depletion)

    for time_value in times:
        print(f"计算 t={time_value:g} d ...", flush=True)
        for region_name, region in regions.items():
            if region.xy.shape[0] == 0:
                continue
            pressure = predict_region_pressure(
                model,
                region,
                float(time_value),
                config,
                device,
                dtype,
                int(args.batch_size),
            )

            p12_initial, p13_initial = initial_pair_for_region(config, region_name)
            depletion12_mpa = p12_initial - pressure[:, 0]
            depletion13_mpa = p13_initial - pressure[:, 1]
            if clamp_depletion:
                depletion12_mpa = np.maximum(depletion12_mpa, 0.0)
                depletion13_mpa = np.maximum(depletion13_mpa, 0.0)

            a12 = storage_coeff_for_region(physics_cfg, region_name)
            a13 = storage_coeff_for_region(physics_cfg, region_name)
            cumulative_p12 = float(args.conversion_factor) * float(np.sum(a12 * depletion12_mpa * region.weight))
            cumulative_p13 = float(args.conversion_factor) * float(np.sum(a13 * depletion13_mpa * region.weight))
            cumulative_total = cumulative_p12 + cumulative_p13

            u12_initial, u13_initial = pressure_to_hat_array(p12_initial, p13_initial, boundary_cfg)
            p_hat = pressure_to_hat_array(pressure[:, 0], pressure[:, 1], boundary_cfg)
            depletion12_hat = u12_initial - p_hat[0]
            depletion13_hat = u13_initial - p_hat[1]
            if clamp_depletion:
                depletion12_hat = np.maximum(depletion12_hat, 0.0)
                depletion13_hat = np.maximum(depletion13_hat, 0.0)
            cumulative_p12_hat = float(np.sum(a12 * depletion12_hat * region.weight))
            cumulative_p13_hat = float(np.sum(a13 * depletion13_hat * region.weight))

            rows.append(
                {
                    "time_d": float(time_value),
                    "region": region_name,
                    "area_m2": float(np.sum(region.weight)),
                    "mean_P12_mpa": float(np.average(pressure[:, 0], weights=region.weight)),
                    "mean_P13_mpa": float(np.average(pressure[:, 1], weights=region.weight)),
                    "cumulative_P12_equiv": cumulative_p12,
                    "cumulative_P13_equiv": cumulative_p13,
                    "cumulative_total_equiv": cumulative_total,
                    "cumulative_P12_hat_area": cumulative_p12_hat,
                    "cumulative_P13_hat_area": cumulative_p13_hat,
                }
            )

    return pd.DataFrame(rows)


def plot_cumulative_curves(df, output_path: Path, show: bool) -> None:
    """绘制各区域累产量曲线。"""

    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    fig.suptitle("Regional Cumulative Gas Production Equivalent", fontsize=15, fontweight="bold")

    plot_specs = [
        ("cumulative_total_equiv", "Total"),
        ("cumulative_P12_equiv", "P12 component"),
        ("cumulative_P13_equiv", "P13 component"),
    ]
    colors = {"HF": "#1f77b4", "SRV": "#2ca02c", "USRV": "#d62728"}

    for ax, (column, title) in zip(axes, plot_specs):
        for region_name in ["HF", "SRV", "USRV"]:
            sub = df[df["region"] == region_name]
            if sub.empty:
                continue
            ax.plot(
                sub["time_d"],
                sub[column],
                linewidth=1.8,
                color=colors.get(region_name),
                label=region_name,
            )
        ax.set_ylabel("Cumulative equivalent")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")

    axes[-1].set_xlabel("Time / d")
    fig.savefig(output_path, dpi=220)
    print(f"已保存累产气量曲线: {output_path}", flush=True)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    """脚本入口。"""

    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)
    configure_matplotlib()
    args = parse_args()

    from src.evaluate import load_trained_model

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"未找到模型 checkpoint: {args.checkpoint}\n"
            "请先运行 main.py 完成训练，或通过 --checkpoint 指定模型文件。"
        )

    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    df = compute_cumulative_table(config, geometry, model, device, dtype, args)

    args.output_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_table, index=False, encoding="utf-8-sig")
    print(f"已保存累产气量数据表: {args.output_table}", flush=True)

    plot_cumulative_curves(df, args.output_figure, bool(args.show))


if __name__ == "__main__":
    main()
