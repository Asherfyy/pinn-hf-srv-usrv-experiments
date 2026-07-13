"""IDE 直接运行：绘制训练损失曲线。

打开本文件后点击 Cursor/IDE 右上角三角形运行即可。脚本会读取
`outputs/logs/loss_history.csv`，绘制总损失和各分项损失，并保存到
`outputs/figures/loss_history.png`。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_HISTORY = PROJECT_ROOT / "outputs" / "logs" / "loss_history.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "figures" / "loss_history.png"


def relaunch_with_venv_if_needed() -> None:
    """如果 IDE 没有选中项目虚拟环境，则自动切换到 `.venv` 后重启脚本。"""

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    """解析命令行参数；IDE 直接运行时使用默认路径。"""

    parser = argparse.ArgumentParser(description="绘制 PINN 训练损失曲线。")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="loss_history.csv 路径。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出图片路径。")
    parser.add_argument("--linear", action="store_true", help="使用线性 y 轴；默认使用对数 y 轴。")
    parser.add_argument("--show", action="store_true", help="保存后弹出图窗显示。")
    return parser.parse_args()


def configure_matplotlib() -> None:
    """设置 Matplotlib 字体和样式。"""

    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def positive_for_log_scale(series):
    """对数坐标下把非正值替换为 NaN，避免 0 值导致曲线异常。"""

    return series.where(series > 0)


def plot_loss_history(history_path: Path, output_path: Path, use_log_scale: bool, show: bool) -> None:
    """读取 CSV 并绘制训练损失曲线。"""

    import matplotlib.pyplot as plt
    import pandas as pd

    if not history_path.exists():
        raise FileNotFoundError(
            f"未找到训练损失文件: {history_path}\n"
            "请先运行 main.py 完成至少一个 epoch 的训练。"
        )

    df = pd.read_csv(history_path)
    if "epoch" not in df.columns:
        raise ValueError(f"{history_path} 中缺少 epoch 列。")

    loss_columns = [col for col in df.columns if col.startswith("loss_")]
    if not loss_columns:
        raise ValueError(f"{history_path} 中没有 loss_ 开头的损失列。")

    df = df[["epoch", *loss_columns]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["epoch"])
    if df.empty:
        raise ValueError(f"{history_path} 中没有可绘制的数据。")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    fig.suptitle("Training Loss History", fontsize=15, fontweight="bold")

    epoch = df["epoch"]
    total_col = "loss_total" if "loss_total" in df.columns else loss_columns[0]
    total_y = positive_for_log_scale(df[total_col]) if use_log_scale else df[total_col]
    axes[0].plot(epoch, total_y, color="#111827", linewidth=2.0, label=total_col)
    axes[0].set_ylabel("Total Loss")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(loc="best")

    component_columns = [col for col in loss_columns if col != total_col]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, col in enumerate(component_columns):
        y = positive_for_log_scale(df[col]) if use_log_scale else df[col]
        color = color_cycle[idx % len(color_cycle)] if color_cycle else None
        axes[1].plot(epoch, y, linewidth=1.3, label=col, color=color)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Component Loss")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend(loc="best", ncol=2, fontsize=8)

    if use_log_scale:
        axes[0].set_yscale("log")
        axes[1].set_yscale("log")

    suffix = "log scale" if use_log_scale else "linear scale"
    axes[0].set_title(f"{total_col} ({suffix})")
    axes[1].set_title(f"Loss components ({suffix})")

    fig.savefig(output_path, dpi=220)
    print(f"已保存训练损失曲线: {output_path}")

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
    plot_loss_history(
        history_path=args.history,
        output_path=args.output,
        use_log_scale=not args.linear,
        show=bool(args.show),
    )


if __name__ == "__main__":
    main()
