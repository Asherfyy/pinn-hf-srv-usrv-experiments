"""IDE 直接运行：绘制 PINN 训练采样点分布。

本脚本不训练模型，只根据 `config/default.yaml` 初始化几何和采样器，然后把采样点
保存成图片。它适合检查三件事：

1. HF 裂缝点是否真的落在极窄裂缝内；
2. SRV/USRV 点是否按分区正确分布；
3. Dirichlet、Neumann 和界面点是否覆盖了预期边界。

在 Cursor/IDE 中打开本文件后，点击右上角三角形运行即可。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def relaunch_with_venv_if_needed() -> None:
    """若 IDE 没有选中项目虚拟环境，则自动切换到 `.venv` 后重启脚本。"""

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    """解析可选参数。IDE 直接运行时使用默认值。"""

    parser = argparse.ArgumentParser(description="绘制 HF/SRV/USRV PINN 采样点分布。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径。")
    parser.add_argument("--seed", type=int, default=2026, help="采样随机种子。")
    parser.add_argument("--max-points-per-cloud", type=int, default=3000, help="单类点云最大绘制数量。")
    return parser.parse_args()


def maybe_subsample(array: Any, max_points: int, rng: Any) -> Any:
    """为了图像清晰，点数过多时随机下采样。"""

    if len(array) <= max_points:
        return array
    idx = rng.choice(len(array), size=max_points, replace=False)
    return array[idx]


def run() -> None:
    """采样并绘制所有采样点分布图。"""

    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import torch

    from src.config import load_config
    from src.evaluate import draw_geometry_overlay
    from src.geometry import ReservoirGeometry
    from src.sampler import ReservoirSampler
    from src.utils import ensure_output_dirs, force_cpu, get_torch_dtype, set_seed

    args = parse_args()
    config = load_config(args.config)
    config["runtime"]["seed"] = int(args.seed)

    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(str(config["runtime"]["dtype"]))
    set_seed(int(args.seed))
    ensure_output_dirs(config)

    geometry = ReservoirGeometry(config["geometry"], data_dir=config["paths"]["data"])
    sampler = ReservoirSampler(geometry, config["sampler"], device=device, dtype=dtype, seed=int(args.seed))
    samples = sampler.sample_all()
    rng = np.random.default_rng(int(args.seed))

    figures_dir = Path(config["paths"]["figures"])
    tables_dir = Path(config["paths"]["tables"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    def to_np(tensor: torch.Tensor) -> np.ndarray:
        """把 CPU/GPU 张量安全转换为 NumPy 数组。"""

        return tensor.detach().cpu().numpy()

    def base_axes(title: str) -> tuple[Any, Any]:
        """创建带几何边界的基础坐标轴。"""

        fig, ax = plt.subplots(figsize=(9, 4.6), dpi=170)
        draw_geometry_overlay(ax, geometry)
        ax.set_xlim(float(config["geometry"]["x_min"]), float(config["geometry"]["x_max"]))
        ax.set_ylim(float(config["geometry"]["y_min"]), float(config["geometry"]["y_max"]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
        ax.set_title(title)
        ax.grid(True, linewidth=0.35, alpha=0.35)
        return fig, ax

    def save_xy_cloud(path: Path, title: str, clouds: list[tuple[str, np.ndarray, str, float]]) -> None:
        """保存多个二维点云的叠加图。"""

        fig, ax = base_axes(title)
        for label, points, color, alpha in clouds:
            points = maybe_subsample(points, int(args.max_points_per_cloud), rng)
            ax.scatter(points[:, 0], points[:, 1], s=5, c=color, alpha=alpha, label=f"{label} ({len(points)})")
        ax.legend(loc="upper right", fontsize=7, frameon=True)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        print(f"已保存 {path}", flush=True)

    pde_hf = to_np(samples["pde"]["hf"])
    pde_srv = to_np(samples["pde"]["srv"])
    pde_usrv = to_np(samples["pde"]["usrv"])
    save_xy_cloud(
        figures_dir / "sampling_pde_points.png",
        "PDE collocation points",
        [
            ("HF", pde_hf, "#d62728", 0.9),
            ("SRV", pde_srv, "#1f77b4", 0.45),
            ("USRV", pde_usrv, "#2ca02c", 0.35),
        ],
    )

    ic_hf = to_np(samples["initial"]["hf"])
    ic_srv = to_np(samples["initial"]["srv"])
    ic_usrv = to_np(samples["initial"]["usrv"])
    save_xy_cloud(
        figures_dir / "sampling_initial_points.png",
        "Initial condition points, t=0",
        [
            ("HF IC", ic_hf, "#d62728", 0.9),
            ("SRV IC", ic_srv, "#1f77b4", 0.45),
            ("USRV IC", ic_usrv, "#2ca02c", 0.35),
        ],
    )

    dirichlet = to_np(samples["dirichlet"]["xyt"])
    neumann = to_np(samples["neumann"]["xyt"])
    hf_srv = to_np(samples["interface_hf_srv"]["xyt"])
    srv_usrv = to_np(samples["interface_srv_usrv"]["xyt"])
    save_xy_cloud(
        figures_dir / "sampling_boundary_interface_points.png",
        "Boundary and interface points",
        [
            ("Dirichlet", dirichlet, "#e377c2", 1.0),
            ("Neumann", neumann, "#9467bd", 0.7),
            ("HF-SRV interface", hf_srv, "#ff7f0e", 0.8),
            ("SRV-USRV interface", srv_usrv, "#17becf", 0.8),
        ],
    )

    all_clouds = [
        ("PDE HF", pde_hf, "#d62728", 0.85),
        ("PDE SRV", pde_srv, "#1f77b4", 0.35),
        ("PDE USRV", pde_usrv, "#2ca02c", 0.25),
        ("Dirichlet", dirichlet, "#e377c2", 0.95),
        ("Neumann", neumann, "#9467bd", 0.55),
        ("Interfaces", np.vstack([hf_srv, srv_usrv]), "#ff7f0e", 0.65),
    ]
    save_xy_cloud(figures_dir / "sampling_all_points.png", "All sampled points", all_clouds)

    # HF 裂缝非常窄，单独放大主裂缝和次级裂缝区域，便于确认采样点确实落在裂缝内。
    fig, ax = base_axes("Zoomed HF fracture samples")
    ax.scatter(pde_hf[:, 0], pde_hf[:, 1], s=7, c="#d62728", alpha=0.85, label=f"HF PDE ({len(pde_hf)})")
    ax.set_xlim(190.0, 360.0)
    ax.set_ylim(40.0, 110.0)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    zoom_path = figures_dir / "sampling_hf_zoom.png"
    fig.savefig(zoom_path)
    plt.close(fig)
    print(f"已保存 {zoom_path}", flush=True)

    # 时间采样分布图：检查 early/late mixed 策略是否按预期覆盖早期和长期。
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=170)
    bins = np.r_[np.linspace(0, 10, 16), np.linspace(20, 1000, 25)]
    ax.hist(pde_hf[:, 2], bins=bins, alpha=0.65, label="HF", color="#d62728")
    ax.hist(pde_srv[:, 2], bins=bins, alpha=0.45, label="SRV", color="#1f77b4")
    ax.hist(pde_usrv[:, 2], bins=bins, alpha=0.35, label="USRV", color="#2ca02c")
    ax.set_xlabel("t / day")
    ax.set_ylabel("count")
    ax.set_title("PDE time sampling histogram")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=8)
    fig.tight_layout()
    time_path = figures_dir / "sampling_time_histogram.png"
    fig.savefig(time_path)
    plt.close(fig)
    print(f"已保存 {time_path}", flush=True)

    counts = pd.DataFrame(
        [
            {"name": "pde_hf", "count": len(pde_hf)},
            {"name": "pde_srv", "count": len(pde_srv)},
            {"name": "pde_usrv", "count": len(pde_usrv)},
            {"name": "initial_hf", "count": len(ic_hf)},
            {"name": "initial_srv", "count": len(ic_srv)},
            {"name": "initial_usrv", "count": len(ic_usrv)},
            {"name": "dirichlet", "count": len(dirichlet)},
            {"name": "neumann", "count": len(neumann)},
            {"name": "interface_hf_srv", "count": len(hf_srv)},
            {"name": "interface_srv_usrv", "count": len(srv_usrv)},
        ]
    )
    counts_path = tables_dir / "sampling_point_counts.csv"
    counts.to_csv(counts_path, index=False)
    print(f"已保存 {counts_path}", flush=True)

    print("\n采样点分布图生成完成。", flush=True)


if __name__ == "__main__":
    run()
