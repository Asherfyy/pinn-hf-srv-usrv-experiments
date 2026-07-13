"""诊断绘图工具。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss_history(history_path: str | Path, output_path: str | Path, use_log_scale: bool = True) -> None:
    """绘制训练 loss 曲线。

    训练日志包含大量 PDE 分量和 RMS 诊断列；这里只自动选择 `loss_` 开头的列绘制，
    使新增诊断不会破坏 loss 曲线脚本。
    """

    history = Path(history_path)
    if not history.exists():
        raise FileNotFoundError(f"未找到 loss history: {history}")
    df = pd.read_csv(history)
    if "epoch" not in df.columns:
        raise ValueError("loss_history.csv 缺少 epoch 列。")
    loss_cols = [col for col in df.columns if col.startswith("loss_")]
    if not loss_cols:
        raise ValueError("loss_history.csv 没有 loss_ 列。")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, dpi=180)
    axes[0].plot(df["epoch"], df["loss_total"], color="#111827", linewidth=2.0, label="loss_total")
    for col in [c for c in loss_cols if c != "loss_total"]:
        axes[1].plot(df["epoch"], df[col], linewidth=1.0, label=col)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        if use_log_scale:
            ax.set_yscale("log")
    axes[0].set_ylabel("total")
    axes[1].set_ylabel("components")
    axes[1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"loss 曲线已保存: {out}")
