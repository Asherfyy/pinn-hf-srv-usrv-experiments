"""绘制 P12/P13/Ptotal 自适应点云云图。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .evaluate import load_trained_model, predict_field, triangulation_for_region
from .geometry import REGION_SRV, REGION_USRV


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 v5 base-correction partitioned-MLP 压力云图。")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/final.pt")
    return parser.parse_args()


def _shared_color_levels(region_results: list[tuple[object, np.ndarray]], n_levels: int = 80) -> np.ndarray | None:
    """Return one finite color scale shared by all plotted regions."""

    finite_chunks = [values[np.isfinite(values)] for _tri, values in region_results]
    finite_chunks = [values for values in finite_chunks if values.size > 0]
    if not finite_chunks:
        return None

    all_values = np.concatenate(finite_chunks)
    v_min = float(np.min(all_values))
    v_max = float(np.max(all_values))
    if np.isclose(v_min, v_max):
        pad = max(abs(v_min), 1.0) * 1.0e-6
        v_min -= pad
        v_max += pad
    return np.linspace(v_min, v_max, int(n_levels))


def _draw_interpretation_overlay(ax, geometry) -> None:
    """Draw SRV outline, HF centerlines, and producer marker.

    The HF apertures are only 0.01 m, far below full-domain pixel resolution.
    Centerlines are therefore used as location markers, not as true aperture
    visualization.
    """

    srv = geometry.srv_bg
    ax.plot(
        [srv.x_min, srv.x_max, srv.x_max, srv.x_min, srv.x_min],
        [srv.y_min, srv.y_min, srv.y_max, srv.y_max, srv.y_min],
        color="white",
        linestyle="--",
        linewidth=1.0,
        zorder=20,
    )
    for rect in geometry.hf_rects:
        if rect.width >= rect.height:
            y_center = 0.5 * (rect.y_min + rect.y_max)
            ax.plot([rect.x_min, rect.x_max], [y_center, y_center], color="black", linewidth=1.1, zorder=21)
        else:
            x_center = 0.5 * (rect.x_min + rect.x_max)
            ax.plot([x_center, x_center], [rect.y_min, rect.y_max], color="black", linewidth=1.1, zorder=21)
    seg = geometry.dirichlet_segment
    ax.plot(
        [float(seg["x0"]), float(seg["x1"])],
        [float(seg["y0"]), float(seg["y1"])],
        color="magenta",
        linewidth=2.2,
        zorder=22,
    )


def _draw_region_contours(ax, field: dict, config: dict, variable: str, region_ids: list[int]):
    region_results = []
    for region_id in region_ids:
        result = triangulation_for_region(field["x"], field["y"], field[variable], field["region"], region_id, config)
        if result is not None:
            region_results.append(result)

    levels = _shared_color_levels(region_results)
    contour = None
    for tri, values in region_results:
        if levels is None or np.sum(np.isfinite(values)) < 3:
            continue
        contour = ax.tricontourf(tri, values, levels=levels, cmap="rainbow")
    return contour


def save_field_figure(field: dict, geometry, config: dict, variable: str, time_value: float, path: Path) -> None:
    """按区域三角剖分并叠加绘制，避免跨区域插值。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=170)
    contour = _draw_region_contours(ax, field, config, variable, [REGION_USRV, REGION_SRV])
    _draw_interpretation_overlay(ax, geometry)
    ax.set_xlim(geometry.domain.x_min, geometry.domain.x_max)
    ax.set_ylim(geometry.domain.y_min, geometry.domain.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"{variable} at t={time_value:g} d, SRV/USRV field with HF centerlines")
    if contour is not None:
        fig.colorbar(contour, ax=ax, label="Pressure / MPa")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_srv_funnel_figure(field: dict, geometry, config: dict, variable: str, time_value: float, path: Path) -> None:
    """Save a SRV-local pressure map for inspecting the funnel around HF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=180)
    contour = _draw_region_contours(ax, field, config, variable, [REGION_SRV])
    _draw_interpretation_overlay(ax, geometry)
    srv = geometry.srv_bg
    ax.set_xlim(srv.x_min, srv.x_max)
    ax.set_ylim(srv.y_min, srv.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title(f"{variable} at t={time_value:g} d, SRV pressure funnel around HF")
    if contour is not None:
        fig.colorbar(contour, ax=ax, label="Pressure / MPa")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    for time_value in config["evaluation"]["times"]:
        field = predict_field(model, geometry, config, float(time_value), device, dtype)
        for variable in ["P12", "P13", "Ptotal"]:
            out = Path(config["paths"]["figures"]) / f"field_{variable}_t{float(time_value):g}.png"
            save_field_figure(field, geometry, config, variable, float(time_value), out)
            srv_out = Path(config["paths"]["figures"]) / f"field_{variable}_t{float(time_value):g}_srv_funnel.png"
            save_srv_funnel_figure(field, geometry, config, variable, float(time_value), srv_out)
            print(f"已保存 {out}")


if __name__ == "__main__":
    main()
