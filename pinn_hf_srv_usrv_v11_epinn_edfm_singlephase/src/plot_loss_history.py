"""Plot v11 training loss history."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .config import load_config
from .utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v11 training loss history.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--history-file", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    history_path = Path(args.history_file) if args.history_file else Path(config["paths"]["logs"]) / "loss_history.csv"
    figure_path = Path(args.output_file) if args.output_file else Path(config["paths"]["figures"]) / "loss_history.png"
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
