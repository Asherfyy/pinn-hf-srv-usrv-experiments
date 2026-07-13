"""诊断 Dirichlet hard constraint 的 ADF 与网络扰动项。

脚本沿主裂缝中心线生成诊断表和曲线，用来判断新的局部 tanh ADF 是否让
`B_D * raw` 在离开生产边界后恢复到合理量级，同时确认 Dirichlet 边界处
`B_D=0`、最终无量纲压力由参考场精确给出。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "final.pt"


def relaunch_with_venv_if_needed() -> None:
    """若 IDE 运行按钮没有选择项目虚拟环境，则切换到 `.venv` 后重启。"""

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    """解析命令行参数，IDE 直接运行时使用默认配置。"""

    parser = argparse.ArgumentParser(description="诊断 Dirichlet hard constraint 的 ADF 与 raw 输出。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML 配置文件路径。")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="训练好的模型 checkpoint。")
    parser.add_argument("--n-x", type=int, default=400, help="主裂缝中心线 x 方向诊断点数。")
    return parser.parse_args()


def configure_matplotlib() -> None:
    """设置 Matplotlib 后端和基础字体。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def build_diagnostic_table(config, geometry, model, device, dtype, n_x: int):
    """沿主裂缝中心线计算 hard constraint 分解量。"""

    import numpy as np
    import pandas as pd
    import torch

    geom_cfg = config["geometry"]
    x = torch.linspace(
        float(geom_cfg["main_frac_x_min"]),
        float(geom_cfg["main_frac_x_max"]),
        int(n_x),
        dtype=dtype,
        device=device,
    ).view(-1, 1)
    y_value = 0.5 * (float(geom_cfg["main_frac_y_min"]) + float(geom_cfg["main_frac_y_max"]))
    y = torch.full_like(x, y_value)
    rows = []
    with torch.no_grad():
        d_dirichlet_m = geometry.distance_to_dirichlet_torch(x, y) * geometry.l_ref
        b_d = geometry.adf_dirichlet_torch(x, y)
        for time_value in config["evaluation"]["times"]:
            t = torch.full_like(x, float(time_value))
            xyt = torch.cat([x, y, t], dim=1)
            components = model.hard_constraint_components(xyt)
            raw = components["raw"]
            reference = components["reference"]
            correction = components["correction"]
            final = components["final"]
            block = pd.DataFrame(
                {
                    "x": x.detach().cpu().numpy().ravel(),
                    "t": np.full(int(n_x), float(time_value)),
                    "d_dirichlet_m": d_dirichlet_m.detach().cpu().numpy().ravel(),
                    "B_D": b_d.detach().cpu().numpy().ravel(),
                    "u_ref_12": reference[:, 0].detach().cpu().numpy(),
                    "u_ref_13": reference[:, 1].detach().cpu().numpy(),
                    "raw_12": raw[:, 0].detach().cpu().numpy(),
                    "raw_13": raw[:, 1].detach().cpu().numpy(),
                    "correction_12": correction[:, 0].detach().cpu().numpy(),
                    "correction_13": correction[:, 1].detach().cpu().numpy(),
                    "u_final_12": final[:, 0].detach().cpu().numpy(),
                    "u_final_13": final[:, 1].detach().cpu().numpy(),
                }
            )
            rows.append(block)
    return pd.concat(rows, ignore_index=True)


def save_plots(df, figures_dir: Path) -> None:
    """保存 ADF、raw 输出和分解项曲线图。"""

    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.0), dpi=180)
    first_time = float(df["t"].min())
    sub = df[df["t"] == first_time]
    ax.plot(sub["x"], sub["B_D"], color="#111827", linewidth=2.0)
    ax.set_xlabel("x / m")
    ax.set_ylabel("B_D")
    ax.set_title("Dirichlet ADF along main fracture centerline")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "hard_constraint_BD.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True, dpi=180)
    for time_value, sub in df.groupby("t"):
        axes[0].plot(sub["x"], sub["raw_12"], linewidth=1.2, label=f"t={time_value:g} d")
        axes[1].plot(sub["x"], sub["raw_13"], linewidth=1.2, label=f"t={time_value:g} d")
    axes[0].set_ylabel("raw_12")
    axes[1].set_ylabel("raw_13")
    axes[1].set_xlabel("x / m")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "hard_constraint_raw.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.5), sharex=True, dpi=180)
    for time_value, sub in df.groupby("t"):
        axes[0].plot(sub["x"], sub["u_ref_12"], linestyle="--", linewidth=1.0, label=f"ref t={time_value:g}")
        axes[0].plot(sub["x"], sub["u_final_12"], linewidth=1.4, label=f"final t={time_value:g}")
        axes[1].plot(sub["x"], sub["u_ref_13"], linestyle="--", linewidth=1.0, label=f"ref t={time_value:g}")
        axes[1].plot(sub["x"], sub["u_final_13"], linewidth=1.4, label=f"final t={time_value:g}")
    axes[0].set_ylabel("u12 components")
    axes[1].set_ylabel("u13 components")
    axes[1].set_xlabel("x / m")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "hard_constraint_components.png")
    plt.close(fig)


def main() -> None:
    """脚本入口。"""

    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)
    configure_matplotlib()
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"未找到模型 checkpoint: {args.checkpoint}。请先训练模型。")

    from src.evaluate import load_trained_model

    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    df = build_diagnostic_table(config, geometry, model, device, dtype, int(args.n_x))

    tables_dir = Path(config["paths"]["tables"])
    figures_dir = Path(config["paths"]["figures"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "hard_constraint_diagnostics.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_plots(df, figures_dir)
    print(f"已保存 hard constraint 诊断表: {out_csv}", flush=True)
    print(f"已保存 hard constraint 诊断图: {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
