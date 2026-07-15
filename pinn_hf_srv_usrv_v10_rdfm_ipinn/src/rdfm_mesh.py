"""Structured Q1 finite-element mesh for the v10 RDFM/I-PINN solver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .geometry import ReservoirGeometry


REGION_SRV_NAME = "SRV"
REGION_USRV_NAME = "USRV"


@dataclass(frozen=True)
class RdfmMesh:
    node_xy: np.ndarray
    elements: np.ndarray
    triangles: np.ndarray
    cell_region: np.ndarray
    dirichlet_nodes: np.ndarray
    free_nodes: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    dx: float
    dy: float
    nx: int
    ny: int

    @property
    def num_nodes(self) -> int:
        return int(self.node_xy.shape[0])

    @property
    def num_elements(self) -> int:
        return int(self.elements.shape[0])

    def node_xy_torch(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(self.node_xy, dtype=dtype, device=device)


def node_id(i: int, j: int, nx: int) -> int:
    return int(j * (nx + 1) + i)


def build_structured_mesh(geometry: ReservoirGeometry, mesh_cfg: dict[str, Any]) -> RdfmMesh:
    nx = int(mesh_cfg["nx"])
    ny = int(mesh_cfg["ny"])
    if nx <= 0 or ny <= 0:
        raise ValueError("mesh.nx and mesh.ny must be positive.")

    x_coords = np.linspace(geometry.domain.x_min, geometry.domain.x_max, nx + 1, dtype=np.float64)
    y_coords = np.linspace(geometry.domain.y_min, geometry.domain.y_max, ny + 1, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(x_coords, y_coords)
    node_xy = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    dx = float(x_coords[1] - x_coords[0])
    dy = float(y_coords[1] - y_coords[0])

    elements: list[list[int]] = []
    triangles: list[list[int]] = []
    region: list[str] = []
    for j in range(ny):
        for i in range(nx):
            bl = node_id(i, j, nx)
            br = node_id(i + 1, j, nx)
            tr = node_id(i + 1, j + 1, nx)
            tl = node_id(i, j + 1, nx)
            elements.append([bl, br, tr, tl])
            triangles.append([bl, br, tr])
            triangles.append([bl, tr, tl])
            cx = 0.5 * (x_coords[i] + x_coords[i + 1])
            cy = 0.5 * (y_coords[j] + y_coords[j + 1])
            in_srv = bool(geometry.inside_rect_np(cx, cy, geometry.srv_bg))
            region.append(REGION_SRV_NAME if in_srv else REGION_USRV_NAME)

    dirichlet_nodes = _find_dirichlet_nodes(geometry, node_xy, dx, dy)
    all_nodes = np.arange(node_xy.shape[0], dtype=np.int64)
    free_nodes = np.setdiff1d(all_nodes, dirichlet_nodes, assume_unique=False)

    return RdfmMesh(
        node_xy=node_xy.astype(np.float64),
        elements=np.asarray(elements, dtype=np.int64),
        triangles=np.asarray(triangles, dtype=np.int64),
        cell_region=np.asarray(region, dtype=object),
        dirichlet_nodes=dirichlet_nodes.astype(np.int64),
        free_nodes=free_nodes.astype(np.int64),
        x_coords=x_coords,
        y_coords=y_coords,
        dx=dx,
        dy=dy,
        nx=nx,
        ny=ny,
    )


def _find_dirichlet_nodes(geometry: ReservoirGeometry, node_xy: np.ndarray, dx: float, dy: float) -> np.ndarray:
    seg = geometry.dirichlet_segment
    x0 = float(seg["x0"])
    x1 = float(seg["x1"])
    y0 = min(float(seg["y0"]), float(seg["y1"]))
    y1 = max(float(seg["y0"]), float(seg["y1"]))
    x = node_xy[:, 0]
    y = node_xy[:, 1]
    x_tol = max(1.0e-8, 1.0e-6 * dx)
    y_tol = max(1.0e-8, 1.0e-6 * dy)
    if abs(x0 - x1) <= x_tol:
        mask = np.isclose(x, x0, atol=x_tol) & (y >= y0 - y_tol) & (y <= y1 + y_tol)
    else:
        raise NotImplementedError("v10 currently supports vertical Dirichlet production segments.")
    nodes = np.where(mask)[0]
    if nodes.size == 0:
        # Fallback to the nearest node to the segment midpoint. This keeps the
        # hard production boundary well-defined even when the segment is thinner
        # than the grid spacing.
        midpoint = np.asarray([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float64)
        nodes = np.asarray([int(np.argmin(np.linalg.norm(node_xy - midpoint[None, :], axis=1)))], dtype=np.int64)
    return np.unique(nodes)
