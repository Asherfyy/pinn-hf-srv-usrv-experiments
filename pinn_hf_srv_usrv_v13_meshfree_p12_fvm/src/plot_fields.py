"""Plot P12 adaptive point-cloud maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from .evaluate import load_trained_model, predict_field


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v13 mesh-free P12 pressure fields.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/final.pt")
    return parser.parse_args()


def _color_levels(values: np.ndarray, n_levels: int = 80) -> np.ndarray | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    pad = max(abs(v_max - v_min), 1.0) * 1.0e-6
    v_min -= pad
    v_max += pad
    if np.isclose(v_min, v_max):
        pad = max(abs(v_min), 1.0) * 1.0e-6
        v_min -= pad
        v_max += pad
    return np.linspace(v_min, v_max, int(n_levels))


def _global_triangulation(field: dict, config: dict) -> mtri.Triangulation:
    _ = config
    return mtri.Triangulation(field["x"], field["y"])


def save_field_figure(field: dict, geometry, config: dict, variable: str, time_value: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=170)
    values = field[variable]
    levels = _color_levels(values)
    contour = None
    if levels is not None and np.sum(np.isfinite(values)) >= 3:
        contour = ax.tricontourf(_global_triangulation(field, config), values, levels=levels, cmap="rainbow")
    geometry.draw_overlay(ax)
    ax.set_xlim(geometry.domain.x_min, geometry.domain.x_max)
    ax.set_ylim(geometry.domain.y_min, geometry.domain.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"P12 at t={time_value:g} d")
    if contour is not None:
        fig.colorbar(contour, ax=ax, label="P12 / MPa")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    for time_value in config["evaluation"]["times"]:
        field = predict_field(model, geometry, config, float(time_value), device, dtype)
        out = Path(config["paths"]["figures"]) / f"field_P12_t{float(time_value):g}.png"
        save_field_figure(field, geometry, config, "P12", float(time_value), out)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
