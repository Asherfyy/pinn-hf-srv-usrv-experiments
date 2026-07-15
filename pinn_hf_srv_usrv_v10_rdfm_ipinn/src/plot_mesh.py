"""Plot the structured FEM mesh used by the v10 RDFM/I-PINN solver."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from .config import load_config
from .geometry import ReservoirGeometry
from .rdfm_fractures import RdfmFracture, fractures_from_geometry
from .rdfm_mesh import REGION_SRV_NAME, RdfmMesh, build_structured_mesh
from .utils import ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the v10 structured FEM mesh and RDFM fractures.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output", type=str, default=None, help="Overview mesh figure path.")
    parser.add_argument("--zoom-output", type=str, default=None, help="SRV/fracture zoom figure path.")
    parser.add_argument("--show-all-nodes", action="store_true", help="Scatter all mesh nodes. Disabled by default for readability.")
    return parser.parse_args()


def _domain_limits(geometry: ReservoirGeometry, padding: float = 0.0) -> tuple[tuple[float, float], tuple[float, float]]:
    domain = geometry.domain
    return (domain.x_min - padding, domain.x_max + padding), (domain.y_min - padding, domain.y_max + padding)


def _srv_zoom_limits(geometry: ReservoirGeometry, padding: float = 15.0) -> tuple[tuple[float, float], tuple[float, float]]:
    srv = geometry.srv_bg
    domain = geometry.domain
    xlim = (max(domain.x_min, srv.x_min - padding), min(domain.x_max, srv.x_max + padding))
    ylim = (max(domain.y_min, srv.y_min - padding), min(domain.y_max, srv.y_max + padding))
    return xlim, ylim


def _grid_segments(mesh: RdfmMesh, xlim: tuple[float, float], ylim: tuple[float, float]) -> list[list[tuple[float, float]]]:
    xmin, xmax = xlim
    ymin, ymax = ylim
    x_coords = mesh.x_coords[(mesh.x_coords >= xmin) & (mesh.x_coords <= xmax)]
    y_coords = mesh.y_coords[(mesh.y_coords >= ymin) & (mesh.y_coords <= ymax)]
    segments: list[list[tuple[float, float]]] = []
    segments.extend([[(float(x), ymin), (float(x), ymax)] for x in x_coords])
    segments.extend([[(xmin, float(y)), (xmax, float(y))] for y in y_coords])
    return segments


def _draw_region_background(ax: Any, mesh: RdfmMesh) -> None:
    region = np.asarray([1 if name == REGION_SRV_NAME else 0 for name in mesh.cell_region], dtype=np.float64).reshape(mesh.ny, mesh.nx)
    cmap = ListedColormap(["#f8fafc", "#dff3e3"])
    ax.pcolormesh(mesh.x_coords, mesh.y_coords, region, cmap=cmap, shading="flat", alpha=0.75, zorder=0)


def _draw_grid(ax: Any, mesh: RdfmMesh, xlim: tuple[float, float], ylim: tuple[float, float], zoom: bool) -> None:
    segments = _grid_segments(mesh, xlim, ylim)
    collection = LineCollection(
        segments,
        colors="#6b7280",
        linewidths=0.28 if zoom else 0.18,
        alpha=0.45 if zoom else 0.28,
        zorder=2,
    )
    ax.add_collection(collection)


def _draw_geometry_overlay(ax: Any, geometry: ReservoirGeometry, fractures: list[RdfmFracture], mesh: RdfmMesh, show_all_nodes: bool) -> None:
    domain = geometry.domain
    srv = geometry.srv_bg
    ax.add_patch(
        Rectangle(
            (domain.x_min, domain.y_min),
            domain.width,
            domain.height,
            fill=False,
            edgecolor="#111827",
            linewidth=1.0,
            zorder=5,
            label="domain",
        )
    )
    ax.add_patch(
        Rectangle(
            (srv.x_min, srv.y_min),
            srv.width,
            srv.height,
            fill=False,
            edgecolor="#15803d",
            linestyle="--",
            linewidth=1.3,
            zorder=6,
            label="SRV boundary",
        )
    )
    for rect in geometry.hf_rects:
        ax.add_patch(
            Rectangle(
                (rect.x_min, rect.y_min),
                rect.width,
                rect.height,
                fill=False,
                edgecolor="#9ca3af",
                linestyle=":",
                linewidth=0.8,
                zorder=7,
            )
        )
    for fracture in fractures:
        ax.plot(
            [fracture.start[0], fracture.end[0]],
            [fracture.start[1], fracture.end[1]],
            color="#111827",
            linewidth=1.4,
            zorder=8,
        )
    if fractures:
        ax.plot([], [], color="#111827", linewidth=1.4, label="RDFM centerline")

    seg = geometry.dirichlet_segment
    ax.plot(
        [float(seg["x0"]), float(seg["x1"])],
        [float(seg["y0"]), float(seg["y1"])],
        color="#d946ef",
        linewidth=2.4,
        zorder=9,
        label="Dirichlet segment",
    )
    dirichlet_xy = mesh.node_xy[mesh.dirichlet_nodes]
    ax.scatter(dirichlet_xy[:, 0], dirichlet_xy[:, 1], s=28, color="#dc2626", edgecolors="white", linewidths=0.4, zorder=10, label="Dirichlet nodes")
    if show_all_nodes:
        ax.scatter(mesh.node_xy[:, 0], mesh.node_xy[:, 1], s=2.0, color="#2563eb", alpha=0.28, linewidths=0.0, zorder=4, label="mesh nodes")


def _format_axes(ax: Any, title: str, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)


def plot_mesh_figure(
    geometry: ReservoirGeometry,
    mesh: RdfmMesh,
    fractures: list[RdfmFracture],
    output_path: Path,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_all_nodes: bool,
    zoom: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 5.2), constrained_layout=True)
    _draw_region_background(ax, mesh)
    _draw_grid(ax, mesh, xlim, ylim, zoom=zoom)
    _draw_geometry_overlay(ax, geometry, fractures, mesh, show_all_nodes=show_all_nodes)
    _format_axes(ax, title, xlim, ylim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved: {output_path}")


def default_output_paths(config: dict[str, Any], output: str | None, zoom_output: str | None) -> tuple[Path, Path]:
    figures_dir = Path(config["paths"]["figures"])
    overview = Path(output) if output is not None else figures_dir / "mesh_overview.png"
    zoom = Path(zoom_output) if zoom_output is not None else figures_dir / "mesh_srv_zoom.png"
    return overview, zoom


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    fractures = fractures_from_geometry(geometry)
    overview_path, zoom_path = default_output_paths(config, args.output, args.zoom_output)

    domain_xlim, domain_ylim = _domain_limits(geometry)
    plot_mesh_figure(
        geometry=geometry,
        mesh=mesh,
        fractures=fractures,
        output_path=overview_path,
        title=f"v10 structured Q1 FEM mesh ({mesh.nx} x {mesh.ny} cells)",
        xlim=domain_xlim,
        ylim=domain_ylim,
        show_all_nodes=args.show_all_nodes,
        zoom=False,
    )

    zoom_xlim, zoom_ylim = _srv_zoom_limits(geometry)
    plot_mesh_figure(
        geometry=geometry,
        mesh=mesh,
        fractures=fractures,
        output_path=zoom_path,
        title="v10 SRV/RDFM fracture mesh zoom",
        xlim=zoom_xlim,
        ylim=zoom_ylim,
        show_all_nodes=True,
        zoom=True,
    )


if __name__ == "__main__":
    main()
