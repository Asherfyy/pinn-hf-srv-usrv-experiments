"""Plot v11 training loss history."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
history_path = ROOT / "outputs" / "logs" / "loss_history.csv"
figure_path = ROOT / "outputs" / "figures" / "loss_history.png"


def main() -> None:
    if not history_path.exists():
        raise FileNotFoundError(f"Loss history does not exist: {history_path}")
    history = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.semilogy(history["epoch"], history["loss_total"], label="loss_total")
    if "residual_rms" in history.columns:
        ax.semilogy(history["epoch"], history["residual_rms"], label="residual_rms")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(f"loss curve saved: {figure_path}")


if __name__ == "__main__":
    main()
