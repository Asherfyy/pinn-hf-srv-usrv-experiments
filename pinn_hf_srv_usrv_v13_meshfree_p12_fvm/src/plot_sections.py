"""Plot P12 profiles along fracture and cross-region lines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .evaluate import load_trained_model, predict_pressure_mpa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v13 mesh-free P12 pressure sections.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/final.pt")
    parser.add_argument("--n", type=int, default=800)
    return parser.parse_args()


def _predict_profile(model, config: dict, device: torch.device, dtype: torch.dtype, x: np.ndarray, y: np.ndarray, t: float) -> pd.DataFrame:
    t_col = np.full_like(x, float(t), dtype=np.float64)
    xyt = torch.as_tensor(np.column_stack([x, y, t_col]), dtype=dtype, device=device)
    pressure = predict_pressure_mpa(model, xyt, config).numpy()
    return pd.DataFrame({"x": x, "y": y, "t": t_col, "P12": pressure[:, 0]})


def build_profiles(config: dict, geometry, model, device: torch.device, dtype: torch.dtype, n: int) -> dict[str, pd.DataFrame]:
    profiles: dict[str, pd.DataFrame] = {}
    for time_value in [float(value) for value in config["evaluation"]["times"]]:
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


def build_srv_usrv_jump_profile(config: dict, geometry, model, device: torch.device, dtype: torch.dtype, n: int) -> pd.DataFrame:
    time_value = 1000.0
    eps = float(config["sampler"]["eps_srv_usrv"])
    margin = max(1.0, 4.0 * eps)
    rows: list[pd.DataFrame] = []
    s = geometry.srv_bg
    for side in ["left", "bottom", "top"]:
        if side == "left":
            coord = np.linspace(s.y_min + margin, s.y_max - margin, int(n), dtype=np.float64)
            srv = np.column_stack([np.full_like(coord, s.x_min + eps), coord, np.full_like(coord, time_value)])
            usrv = np.column_stack([np.full_like(coord, s.x_min - eps), coord, np.full_like(coord, time_value)])
        elif side == "bottom":
            coord = np.linspace(s.x_min + margin, s.x_max - margin, int(n), dtype=np.float64)
            srv = np.column_stack([coord, np.full_like(coord, s.y_min + eps), np.full_like(coord, time_value)])
            usrv = np.column_stack([coord, np.full_like(coord, s.y_min - eps), np.full_like(coord, time_value)])
        else:
            coord = np.linspace(s.x_min + margin, s.x_max - margin, int(n), dtype=np.float64)
            srv = np.column_stack([coord, np.full_like(coord, s.y_max - eps), np.full_like(coord, time_value)])
            usrv = np.column_stack([coord, np.full_like(coord, s.y_max + eps), np.full_like(coord, time_value)])
        xyt = torch.as_tensor(np.vstack([srv, usrv]), dtype=dtype, device=device)
        pressure = predict_pressure_mpa(model, xyt, config).numpy().reshape(2, -1)
        rows.append(
            pd.DataFrame(
                {
                    "side": side,
                    "coord": coord,
                    "t": np.full_like(coord, time_value),
                    "P12_srv": pressure[0],
                    "P12_usrv": pressure[1],
                    "jump_mpa": pressure[0] - pressure[1],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def save_profile_outputs(profiles: dict[str, pd.DataFrame], config: dict) -> None:
    tables = Path(config["paths"]["tables"])
    figures = Path(config["paths"]["figures"])
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    all_df = pd.concat([df.assign(profile=name) for name, df in profiles.items()], ignore_index=True)
    csv_path = tables / "section_profiles.csv"
    all_df.to_csv(csv_path, index=False)

    for prefix in ["main_fracture", "secondary_1", "cross_region_y75"]:
        fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=170)
        for name, df in profiles.items():
            if not name.startswith(prefix):
                continue
            label = name.split("_t")[-1] + " d"
            s = df["x"] if prefix != "secondary_1" else df["y"]
            ax.plot(s, df["P12"], label=label)
        ax.set_ylabel("P12 / MPa")
        ax.set_xlabel("x / m" if prefix != "secondary_1" else "y / m")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.suptitle(prefix)
        fig.tight_layout()
        fig.savefig(figures / f"section_{prefix}.png")
        plt.close(fig)
    print(f"saved {csv_path}")


def save_srv_usrv_jump_outputs(profile: pd.DataFrame, config: dict) -> None:
    tables = Path(config["paths"]["tables"])
    figures = Path(config["paths"]["figures"])
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    csv_path = tables / "srv_usrv_jump_profiles.csv"
    profile.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=170)
    for side, df in profile.groupby("side"):
        ax.plot(df["coord"], df["jump_mpa"], label=side)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("P12 SRV - USRV / MPa")
    ax.set_xlabel("interface coordinate / m")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.suptitle("SRV-USRV pressure jump at t=1000 d")
    fig.tight_layout()
    fig.savefig(figures / "section_srv_usrv_jump_t1000.png")
    plt.close(fig)
    print(f"saved {csv_path}")


def main() -> None:
    args = parse_args()
    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    profiles = build_profiles(config, geometry, model, device, dtype, int(args.n))
    save_profile_outputs(profiles, config)
    jump_profile = build_srv_usrv_jump_profile(config, geometry, model, device, dtype, int(args.n))
    save_srv_usrv_jump_outputs(jump_profile, config)


if __name__ == "__main__":
    main()
