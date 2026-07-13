"""绘制主裂缝、次级裂缝和跨分区剖面。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .evaluate import load_trained_model, predict_pressure_mpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 v5 base-correction partitioned-MLP 压力剖面。")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/final.pt")
    parser.add_argument("--n", type=int, default=800)
    return parser.parse_args()


def _predict_profile(model, config: dict, device: torch.device, dtype: torch.dtype, x: np.ndarray, y: np.ndarray, t: float) -> pd.DataFrame:
    """对一条剖面线预测 P12/P13/Ptotal。"""

    t_col = np.full_like(x, float(t), dtype=np.float64)
    xyt = torch.as_tensor(np.column_stack([x, y, t_col]), dtype=dtype, device=device)
    pressure = predict_pressure_mpa(model, xyt, config).numpy()
    return pd.DataFrame({"x": x, "y": y, "t": t_col, "P12": pressure[:, 0], "P13": pressure[:, 1], "Ptotal": pressure[:, 0] + pressure[:, 1]})


def build_profiles(config: dict, geometry, model, device: torch.device, dtype: torch.dtype, n: int) -> dict[str, pd.DataFrame]:
    """生成三类剖面数据。"""

    profiles: dict[str, pd.DataFrame] = {}
    times = [float(value) for value in config["evaluation"]["times"]]
    for time_value in times:
        x = np.linspace(geometry.main_frac.x_min, geometry.main_frac.x_max, int(n))
        y = np.full_like(x, 0.5 * (geometry.main_frac.y_min + geometry.main_frac.y_max))
        profiles[f"main_fracture_t{time_value:g}"] = _predict_profile(model, config, device, dtype, x, y, time_value)

        rect = geometry.secondary_fractures[0]
        y2 = np.linspace(rect.y_min, rect.y_max, int(n))
        x2 = np.full_like(y2, 0.5 * (rect.x_min + rect.x_max))
        profiles[f"secondary_1_t{time_value:g}"] = _predict_profile(model, config, device, dtype, x2, y2, time_value)

        x3 = np.linspace(geometry.domain.x_min, geometry.domain.x_max, int(n))
        y3 = np.full_like(x3, 75.0)
        profiles[f"cross_region_y75_t{time_value:g}"] = _predict_profile(model, config, device, dtype, x3, y3, time_value)
    return profiles


def save_profile_outputs(profiles: dict[str, pd.DataFrame], config: dict) -> None:
    """保存剖面 CSV 和图片。"""

    tables = Path(config["paths"]["tables"])
    figures = Path(config["paths"]["figures"])
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, df in profiles.items():
        tmp = df.copy()
        tmp.insert(0, "profile", name)
        all_rows.append(tmp)
    all_df = pd.concat(all_rows, ignore_index=True)
    csv_path = tables / "section_profiles.csv"
    all_df.to_csv(csv_path, index=False)

    for prefix in ["main_fracture", "secondary_1", "cross_region_y75"]:
        fig, axes = plt.subplots(3, 1, figsize=(8.5, 8.2), sharex=True, dpi=170)
        for name, df in profiles.items():
            if not name.startswith(prefix):
                continue
            label = name.split("_t")[-1] + " d"
            s = df["x"] if prefix != "secondary_1" else df["y"]
            axes[0].plot(s, df["P12"], label=label)
            axes[1].plot(s, df["P13"], label=label)
            axes[2].plot(s, df["Ptotal"], label=label)
        axes[0].set_ylabel("P12 / MPa")
        axes[1].set_ylabel("P13 / MPa")
        axes[2].set_ylabel("Ptotal / MPa")
        axes[2].set_xlabel("x / m" if prefix != "secondary_1" else "y / m")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(prefix)
        fig.tight_layout()
        fig.savefig(figures / f"section_{prefix}.png")
        plt.close(fig)
    print(f"剖面数据已保存: {csv_path}")


def main() -> None:
    args = parse_args()
    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    profiles = build_profiles(config, geometry, model, device, dtype, int(args.n))
    save_profile_outputs(profiles, config)


if __name__ == "__main__":
    main()
