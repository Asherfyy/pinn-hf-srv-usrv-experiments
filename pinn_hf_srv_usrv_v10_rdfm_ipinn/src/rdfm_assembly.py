"""Finite-element assembly for the v10 RDFM/I-PINN residual."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .rdfm_fractures import RdfmFracture, fracture_subsegments
from .rdfm_mesh import REGION_SRV_NAME, REGION_USRV_NAME, RdfmMesh, node_id


@dataclass(frozen=True)
class ComponentOperators:
    mass: torch.Tensor
    stiffness: torch.Tensor
    mass_row_abs: torch.Tensor
    stiffness_row_abs: torch.Tensor


@dataclass(frozen=True)
class RdfmOperators:
    u12: ComponentOperators
    u13: ComponentOperators

    def for_component(self, component: int) -> ComponentOperators:
        if component == 0:
            return self.u12
        if component == 1:
            return self.u13
        raise ValueError(f"Unknown component index: {component}")


def assemble_rdfm_operators(
    mesh: RdfmMesh,
    fractures: list[RdfmFracture],
    physics_cfg: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> RdfmOperators:
    mass_entries = _assemble_mass_entries(mesh, fractures, physics_cfg)
    stiff12_entries = _assemble_stiffness_entries(mesh, fractures, physics_cfg, variable="u12")
    stiff13_entries = _assemble_stiffness_entries(mesh, fractures, physics_cfg, variable="u13")
    shape = (mesh.num_nodes, mesh.num_nodes)
    mass = _entries_to_sparse(mass_entries, shape, device, dtype)
    stiff12 = _entries_to_sparse(stiff12_entries, shape, device, dtype)
    stiff13 = _entries_to_sparse(stiff13_entries, shape, device, dtype)
    return RdfmOperators(
        u12=ComponentOperators(mass=mass, stiffness=stiff12, mass_row_abs=_sparse_abs_row_sum(mass), stiffness_row_abs=_sparse_abs_row_sum(stiff12)),
        u13=ComponentOperators(mass=mass, stiffness=stiff13, mass_row_abs=_sparse_abs_row_sum(mass), stiffness_row_abs=_sparse_abs_row_sum(stiff13)),
    )


def fracture_coupled_nodes(mesh: RdfmMesh, fractures: list[RdfmFracture]) -> np.ndarray:
    """Return matrix nodes touched by RDFM fracture line-integral terms."""

    nodes: set[int] = set()
    gauss = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    for fracture in fractures:
        if fracture.aperture <= 0.0:
            continue
        for p0, p1 in fracture_subsegments(fracture, mesh):
            segment = p1 - p0
            if float(np.linalg.norm(segment)) <= 0.0:
                continue
            for xi in gauss:
                point = 0.5 * (p0 + p1) + 0.5 * float(xi) * segment
                nodes.update(int(node) for node in _element_nodes_for_point(mesh, float(point[0]), float(point[1])))
    return np.asarray(sorted(nodes), dtype=np.int64)


def _assemble_mass_entries(mesh: RdfmMesh, fractures: list[RdfmFracture], physics_cfg: dict[str, Any]) -> list[tuple[int, int, float]]:
    entries: list[tuple[int, int, float]] = []
    _assemble_matrix_terms(mesh, physics_cfg, variable="u12", entries_mass=entries, entries_stiffness=None)
    _assemble_fracture_terms(mesh, fractures, physics_cfg, variable="u12", entries_mass=entries, entries_stiffness=None)
    return entries


def _assemble_stiffness_entries(
    mesh: RdfmMesh,
    fractures: list[RdfmFracture],
    physics_cfg: dict[str, Any],
    variable: str,
) -> list[tuple[int, int, float]]:
    entries: list[tuple[int, int, float]] = []
    _assemble_matrix_terms(mesh, physics_cfg, variable=variable, entries_mass=None, entries_stiffness=entries)
    _assemble_fracture_terms(mesh, fractures, physics_cfg, variable=variable, entries_mass=None, entries_stiffness=entries)
    return entries


def _assemble_matrix_terms(
    mesh: RdfmMesh,
    physics_cfg: dict[str, Any],
    variable: str,
    entries_mass: list[tuple[int, int, float]] | None,
    entries_stiffness: list[tuple[int, int, float]] | None,
) -> None:
    gauss = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    det_j = mesh.dx * mesh.dy / 4.0
    variable_key = "D1" if variable == "u12" else "D2"
    alpha = float(physics_cfg["anisotropy_y_1"] if variable == "u12" else physics_cfg["anisotropy_y_2"])
    for elem_idx, nodes in enumerate(mesh.elements):
        region = str(mesh.cell_region[elem_idx])
        if region not in {REGION_SRV_NAME, REGION_USRV_NAME}:
            raise ValueError(f"Unexpected matrix cell region: {region}")
        fai = float(physics_cfg["Fai"][region])
        diffusion_x = float(physics_cfg[variable_key][region])
        diffusion_y = alpha * diffusion_x
        local_mass = np.zeros((4, 4), dtype=np.float64)
        local_stiffness = np.zeros((4, 4), dtype=np.float64)
        for xi in gauss:
            for eta in gauss:
                n, dndx, dndy = _q1_shape_and_grad(float(xi), float(eta), mesh.dx, mesh.dy)
                local_mass += fai * np.outer(n, n) * det_j
                local_stiffness += (diffusion_x * np.outer(dndx, dndx) + diffusion_y * np.outer(dndy, dndy)) * det_j
        for a, node_a in enumerate(nodes):
            for b, node_b in enumerate(nodes):
                if entries_mass is not None:
                    entries_mass.append((int(node_a), int(node_b), float(local_mass[a, b])))
                if entries_stiffness is not None:
                    entries_stiffness.append((int(node_a), int(node_b), float(local_stiffness[a, b])))


def _assemble_fracture_terms(
    mesh: RdfmMesh,
    fractures: list[RdfmFracture],
    physics_cfg: dict[str, Any],
    variable: str,
    entries_mass: list[tuple[int, int, float]] | None,
    entries_stiffness: list[tuple[int, int, float]] | None,
) -> None:
    gauss = [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]
    variable_key = "D1" if variable == "u12" else "D2"
    fai = float(physics_cfg["Fai"]["HF"])
    diffusion = float(physics_cfg[variable_key]["HF"])
    for fracture in fractures:
        if fracture.aperture <= 0.0:
            continue
        for p0, p1 in fracture_subsegments(fracture, mesh):
            segment = p1 - p0
            length = float(np.linalg.norm(segment))
            if length <= 0.0:
                continue
            for xi in gauss:
                point = 0.5 * (p0 + p1) + 0.5 * float(xi) * segment
                weight = 0.5 * length
                nodes = _element_nodes_for_point(mesh, float(point[0]), float(point[1]))
                local_xi, local_eta = _local_coords(mesh, float(point[0]), float(point[1]))
                n, dndx, dndy = _q1_shape_and_grad(local_xi, local_eta, mesh.dx, mesh.dy)
                dtau = fracture.tangent[0] * dndx + fracture.tangent[1] * dndy
                local_mass = fracture.aperture * fai * np.outer(n, n) * weight
                local_stiffness = fracture.aperture * diffusion * np.outer(dtau, dtau) * weight
                for a, node_a in enumerate(nodes):
                    for b, node_b in enumerate(nodes):
                        if entries_mass is not None:
                            entries_mass.append((int(node_a), int(node_b), float(local_mass[a, b])))
                        if entries_stiffness is not None:
                            entries_stiffness.append((int(node_a), int(node_b), float(local_stiffness[a, b])))


def _q1_shape_and_grad(xi: float, eta: float, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(
        [
            0.25 * (1.0 - xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 - eta),
            0.25 * (1.0 + xi) * (1.0 + eta),
            0.25 * (1.0 - xi) * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    dndxi = np.asarray(
        [
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta),
        ],
        dtype=np.float64,
    )
    dndeta = np.asarray(
        [
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi),
        ],
        dtype=np.float64,
    )
    return n, dndxi * (2.0 / dx), dndeta * (2.0 / dy)


def _element_indices_for_point(mesh: RdfmMesh, x: float, y: float) -> tuple[int, int]:
    i = int(np.searchsorted(mesh.x_coords, x, side="right") - 1)
    j = int(np.searchsorted(mesh.y_coords, y, side="right") - 1)
    return max(0, min(mesh.nx - 1, i)), max(0, min(mesh.ny - 1, j))


def _element_nodes_for_point(mesh: RdfmMesh, x: float, y: float) -> np.ndarray:
    i, j = _element_indices_for_point(mesh, x, y)
    return np.asarray(
        [
            node_id(i, j, mesh.nx),
            node_id(i + 1, j, mesh.nx),
            node_id(i + 1, j + 1, mesh.nx),
            node_id(i, j + 1, mesh.nx),
        ],
        dtype=np.int64,
    )


def _local_coords(mesh: RdfmMesh, x: float, y: float) -> tuple[float, float]:
    i, j = _element_indices_for_point(mesh, x, y)
    x0 = float(mesh.x_coords[i])
    y0 = float(mesh.y_coords[j])
    xi = 2.0 * (x - x0) / mesh.dx - 1.0
    eta = 2.0 * (y - y0) / mesh.dy - 1.0
    return float(np.clip(xi, -1.0, 1.0)), float(np.clip(eta, -1.0, 1.0))


def _entries_to_sparse(
    entries: list[tuple[int, int, float]],
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not entries:
        indices = torch.zeros((2, 0), dtype=torch.long, device=device)
        values = torch.zeros((0,), dtype=dtype, device=device)
        with torch.sparse.check_sparse_tensor_invariants(False):
            return torch.sparse_coo_tensor(
                indices,
                values,
                size=shape,
                device=device,
                dtype=dtype,
                check_invariants=False,
                is_coalesced=True,
            )
    rows = np.asarray([row for row, _col, _value in entries], dtype=np.int64)
    cols = np.asarray([col for _row, col, _value in entries], dtype=np.int64)
    values_np = np.asarray([value for _row, _col, value in entries], dtype=np.float64)
    indices = torch.as_tensor(np.vstack([rows, cols]), dtype=torch.long, device=device)
    values = torch.as_tensor(values_np, dtype=dtype, device=device)
    with torch.sparse.check_sparse_tensor_invariants(False):
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=shape,
            device=device,
            dtype=dtype,
            check_invariants=False,
            is_coalesced=False,
        ).coalesce()


def _sparse_abs_row_sum(matrix: torch.Tensor) -> torch.Tensor:
    coalesced = matrix.coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    result = torch.zeros((matrix.shape[0],), dtype=values.dtype, device=values.device)
    result.index_add_(0, indices[0], torch.abs(values))
    return result
