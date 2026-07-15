"""Plot the v11 Cartesian matrix grid and EDFM fracture segments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from .config import load_config
from .edfm_grid import build_edfm_grid
from .geometry import ReservoirGeometry
from .utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v11 EDFM grid.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    for x in grid.x_edges:
        ax.plot([x, x], [grid.y_edges[0], grid.y_edges[-1]], color="0.85", linewidth=0.35)
    for y in grid.y_edges:
        ax.plot([grid.x_edges[0], grid.x_edges[-1]], [y, y], color="0.85", linewidth=0.35)
    for segment in grid.fracture_segments:
        ax.plot([segment.start[0], segment.end[0]], [segment.start[1], segment.end[1]], color="black", linewidth=1.0)
    well_xy = grid.cell_xy[grid.well_cells]
    ax.scatter(well_xy[:, 0], well_xy[:, 1], s=40, color="magenta", label="BHP cells", zorder=5)
    geometry.draw_overlay(ax)
    ax.set_title("v11 EDFM grid")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    out = Path(config["paths"]["figures"]) / "mesh_edfm.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
