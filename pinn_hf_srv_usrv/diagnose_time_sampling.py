"""诊断 `log1p_uniform` 时间采样分布。

脚本复用 `ReservoirSampler.sample_time()`，生成大量时间样本并同时绘制物理时间 t
和 log1p(t) 空间的直方图，便于确认 0~1、1~10、10~100、100~1000 d 均被覆盖。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


def relaunch_with_venv_if_needed() -> None:
    """若 IDE 使用系统 Python，则切换到项目 `.venv` 后重启脚本。"""

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    """解析时间采样诊断参数。"""

    parser = argparse.ArgumentParser(description="诊断 PINN 时间采样分布。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML 配置文件路径。")
    parser.add_argument("--n", type=int, default=100000, help="采样数量。")
    parser.add_argument("--seed", type=int, default=2026, help="采样随机种子。")
    return parser.parse_args()


def configure_matplotlib() -> None:
    """设置非交互绘图后端。"""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def compute_time_statistics(samples):
    """计算时间样本的关键区间占比和基础统计量。"""

    import numpy as np
    import pandas as pd

    t = samples.reshape(-1)
    stats = {
        "min": float(np.min(t)),
        "max": float(np.max(t)),
        "median": float(np.median(t)),
        "ratio_0_1d": float(np.mean((t >= 0.0) & (t < 1.0))),
        "ratio_1_10d": float(np.mean((t >= 1.0) & (t < 10.0))),
        "ratio_10_100d": float(np.mean((t >= 10.0) & (t < 100.0))),
        "ratio_100_1000d": float(np.mean((t >= 100.0) & (t <= 1000.0))),
    }
    return pd.DataFrame([stats])


def save_time_histograms(samples, figures_dir: Path) -> None:
    """保存物理时间和 log1p 时间空间的直方图。"""

    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir.mkdir(parents=True, exist_ok=True)
    t = samples.reshape(-1)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=180)
    ax.hist(t, bins=80, color="#2563eb", alpha=0.78)
    ax.set_xlabel("t / d")
    ax.set_ylabel("count")
    ax.set_title("Time sampling histogram in physical space")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "time_sampling_physical.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=180)
    ax.hist(np.log1p(t), bins=80, color="#059669", alpha=0.78)
    ax.set_xlabel("log1p(t)")
    ax.set_ylabel("count")
    ax.set_title("Time sampling histogram in log1p space")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "time_sampling_log_space.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2), dpi=180)
    axes[0].hist(t, bins=80, color="#2563eb", alpha=0.78)
    axes[0].set_xlabel("t / d")
    axes[0].set_title("Physical time")
    axes[1].hist(np.log1p(t), bins=80, color="#059669", alpha=0.78)
    axes[1].set_xlabel("log1p(t)")
    axes[1].set_title("Log space")
    for ax in axes:
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "time_sampling_log1p.png")
    plt.close(fig)


def main() -> None:
    """脚本入口。"""

    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)
    configure_matplotlib()
    args = parse_args()

    import torch

    from src.config import load_config
    from src.geometry import ReservoirGeometry
    from src.sampler import ReservoirSampler
    from src.utils import ensure_output_dirs, get_torch_dtype

    config = load_config(args.config)
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"], data_dir=config["paths"]["data"])
    sampler = ReservoirSampler(
        geometry,
        config["sampler"],
        device=torch.device("cpu"),
        dtype=get_torch_dtype(config["runtime"]["dtype"]),
        seed=int(args.seed),
    )
    samples = sampler.sample_time(int(args.n))
    stats = compute_time_statistics(samples)

    tables_dir = Path(config["paths"]["tables"])
    figures_dir = Path(config["paths"]["figures"])
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_csv = tables_dir / "time_sampling_log1p_stats.csv"
    stats.to_csv(out_csv, index=False, encoding="utf-8-sig")
    save_time_histograms(samples, figures_dir)
    print(stats.to_string(index=False), flush=True)
    print(f"已保存时间采样统计: {out_csv}", flush=True)
    print(f"已保存时间采样直方图: {figures_dir}", flush=True)


if __name__ == "__main__":
    main()
