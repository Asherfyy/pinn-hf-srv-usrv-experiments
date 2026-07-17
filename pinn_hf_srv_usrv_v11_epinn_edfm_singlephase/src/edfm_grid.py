"""Cell-centered FVM grid and lightweight EDFM connections for v11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import ReservoirGeometry, Rect


REGION_SRV = "SRV"
REGION_USRV = "USRV"
REGION_HF = "HF"


@dataclass(frozen=True)
class FractureLine:
    name: str
    start: np.ndarray
    end: np.ndarray
    aperture: float
    tangent: np.ndarray

    @property
    def is_horizontal(self) -> bool:
        return abs(float(self.tangent[0])) >= abs(float(self.tangent[1]))


@dataclass(frozen=True)
class FractureSegment:
    name: str
    start: np.ndarray
    end: np.ndarray
    cell_index: int
    aperture: float

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.start + self.end)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def is_horizontal(self) -> bool:
        return abs(float(self.end[0] - self.start[0])) >= abs(float(self.end[1] - self.start[1]))


@dataclass(frozen=True)
class Connection:
    i: int
    j: int
    transmissibility: float
    kind: str
    component_transmissibility: tuple[float, ...] = ()


@dataclass(frozen=True)
class EdfmGrid:
    cell_xy: np.ndarray
    cell_region: np.ndarray
    cell_volume: np.ndarray
    cell_phi: np.ndarray
    cell_representative_conductivity: np.ndarray
    cell_conductivity: np.ndarray
    cell_storage: np.ndarray
    matrix_cell_count: int
    fracture_segments: list[FractureSegment]
    connections: list[Connection]
    adjacency: np.ndarray | None
    edge_index: np.ndarray
    edge_weight: np.ndarray
    well_cell: int
    well_cells: np.ndarray
    nx: int
    ny: int
    dx: float
    dy: float
    x_edges: np.ndarray
    y_edges: np.ndarray

    @property
    def num_cells(self) -> int:
        return int(self.cell_xy.shape[0])

    @property
    def free_cells(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.num_cells, dtype=np.int64), self.well_cells, assume_unique=False)


def matrix_cell_id(i: int, j: int, nx: int) -> int:
    return int(j * nx + i)


def build_edfm_grid(geometry: ReservoirGeometry, config: dict[str, Any]) -> EdfmGrid:
    grid_cfg = config["grid"]
    nx = int(grid_cfg["nx"])
    ny = int(grid_cfg["ny"])
    thickness = float(grid_cfg["thickness_m"])
    x_edges = np.linspace(geometry.domain.x_min, geometry.domain.x_max, nx + 1, dtype=np.float64)
    y_edges = np.linspace(geometry.domain.y_min, geometry.domain.y_max, ny + 1, dtype=np.float64)
    dx = float(x_edges[1] - x_edges[0])
    dy = float(y_edges[1] - y_edges[0])

    component_count = len(config["pressure"]["components"])
    cell_xy, cell_region, cell_volume, cell_phi, cell_repr_conductivity, cell_conductivity = _build_matrix_cells(geometry, config, x_edges, y_edges, thickness, component_count)
    matrix_count = int(cell_xy.shape[0])
    fracture_lines = [_fracture_line_from_rect(rect) for rect in geometry.hf_rects]
    fracture_segments = _build_fracture_segments(fracture_lines, x_edges, y_edges, matrix_count, _region_storage_phi(config, REGION_HF), thickness)

    if fracture_segments:
        frac_xy = np.asarray([segment.center for segment in fracture_segments], dtype=np.float64)
        frac_volume = np.asarray([segment.length * segment.aperture * thickness for segment in fracture_segments], dtype=np.float64)
        frac_phi = np.full((len(fracture_segments),), _region_storage_phi(config, REGION_HF), dtype=np.float64)
        frac_repr_conductivity = np.full((len(fracture_segments),), _region_representative_conductivity(config, REGION_HF), dtype=np.float64)
        frac_conductivity = np.tile(_region_component_conductivity(config, REGION_HF, component_count).reshape(1, -1), (len(fracture_segments), 1))
        cell_xy = np.vstack([cell_xy, frac_xy])
        cell_region = np.concatenate([cell_region, np.full((len(fracture_segments),), REGION_HF, dtype=object)])
        cell_volume = np.concatenate([cell_volume, frac_volume])
        cell_phi = np.concatenate([cell_phi, frac_phi])
        cell_repr_conductivity = np.concatenate([cell_repr_conductivity, frac_repr_conductivity])
        cell_conductivity = np.vstack([cell_conductivity, frac_conductivity])

    cell_storage = _cell_storage(cell_phi, cell_volume)
    connections = _build_connections(config, x_edges, y_edges, cell_conductivity, fracture_segments, matrix_count, nx, ny, dx, dy, thickness)
    edge_index, edge_weight = _build_sparse_adjacency(cell_xy.shape[0], connections)
    adjacency = _build_adjacency_if_small(cell_xy.shape[0], connections, int(config["edfm"].get("max_dense_elements", 0)))
    well_cell = _nearest_matrix_cell(cell_xy[:matrix_count], float(config["well"]["x"]), float(config["well"]["y"]))
    well_cells = _well_cells(cell_xy, fracture_segments, well_cell, config)
    return EdfmGrid(
        cell_xy=cell_xy,
        cell_region=cell_region,
        cell_volume=cell_volume,
        cell_phi=cell_phi,
        cell_representative_conductivity=cell_repr_conductivity,
        cell_conductivity=cell_conductivity,
        cell_storage=cell_storage,
        matrix_cell_count=matrix_count,
        fracture_segments=fracture_segments,
        connections=connections,
        adjacency=adjacency,
        edge_index=edge_index,
        edge_weight=edge_weight,
        well_cell=well_cell,
        well_cells=well_cells,
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        x_edges=x_edges,
        y_edges=y_edges,
    )


def _build_matrix_cells(
    geometry: ReservoirGeometry,
    config: dict[str, Any],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    thickness: float,
    component_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float64)
    in_srv = geometry.inside_rect_np(xy[:, 0], xy[:, 1], geometry.srv_bg)
    region = np.where(in_srv, REGION_SRV, REGION_USRV).astype(object)
    area = float((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]) * thickness)
    volume = np.full((xy.shape[0],), area, dtype=np.float64)
    phi = np.where(in_srv, _region_storage_phi(config, REGION_SRV), _region_storage_phi(config, REGION_USRV)).astype(np.float64)
    representative_conductivity = np.where(in_srv, _region_representative_conductivity(config, REGION_SRV), _region_representative_conductivity(config, REGION_USRV)).astype(np.float64)
    srv_conductivity = _region_component_conductivity(config, REGION_SRV, component_count)
    usrv_conductivity = _region_component_conductivity(config, REGION_USRV, component_count)
    conductivity = np.where(in_srv.reshape(-1, 1), srv_conductivity.reshape(1, -1), usrv_conductivity.reshape(1, -1)).astype(np.float64)
    return xy, region, volume, phi, representative_conductivity, conductivity


def _fracture_line_from_rect(rect: Rect) -> FractureLine:
    cx = 0.5 * (rect.x_min + rect.x_max)
    cy = 0.5 * (rect.y_min + rect.y_max)
    if rect.width >= rect.height:
        return FractureLine(rect.name, np.asarray([rect.x_min, cy]), np.asarray([rect.x_max, cy]), float(rect.height), np.asarray([1.0, 0.0]))
    return FractureLine(rect.name, np.asarray([cx, rect.y_min]), np.asarray([cx, rect.y_max]), float(rect.width), np.asarray([0.0, 1.0]))


def _build_fracture_segments(
    lines: list[FractureLine],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    matrix_count: int,
    _phi_hf: float,
    _thickness: float,
) -> list[FractureSegment]:
    segments: list[FractureSegment] = []
    next_cell = matrix_count
    for line in lines:
        for start, end in _split_line_by_grid(line, x_edges, y_edges):
            if float(np.linalg.norm(end - start)) <= 0.0:
                continue
            segments.append(FractureSegment(line.name, start, end, next_cell, line.aperture))
            next_cell += 1
    return segments


def _split_line_by_grid(line: FractureLine, x_edges: np.ndarray, y_edges: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    if line.is_horizontal:
        lo, hi = sorted([float(line.start[0]), float(line.end[0])])
        cuts = [lo, hi, *[float(x) for x in x_edges if lo < float(x) < hi]]
        y = float(line.start[1])
        unique = _unique_sorted(cuts)
        return [(np.asarray([a, y], dtype=np.float64), np.asarray([b, y], dtype=np.float64)) for a, b in zip(unique, unique[1:]) if b > a]
    lo, hi = sorted([float(line.start[1]), float(line.end[1])])
    cuts = [lo, hi, *[float(y) for y in y_edges if lo < float(y) < hi]]
    x = float(line.start[0])
    unique = _unique_sorted(cuts)
    return [(np.asarray([x, a], dtype=np.float64), np.asarray([x, b], dtype=np.float64)) for a, b in zip(unique, unique[1:]) if b > a]


def _build_connections(
    config: dict[str, Any],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    conductivity: np.ndarray,
    fracture_segments: list[FractureSegment],
    matrix_count: int,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    thickness: float,
) -> list[Connection]:
    scale = _transmissibility_scale(config)
    connections: list[Connection] = []
    for j in range(ny):
        for i in range(nx):
            cell = matrix_cell_id(i, j, nx)
            if i + 1 < nx:
                other = matrix_cell_id(i + 1, j, nx)
                area = dy * thickness
                connections.append(_connection(cell, other, scale * _harmonic_vector(conductivity[cell], conductivity[other]) * area / dx, "mm"))
            if j + 1 < ny:
                other = matrix_cell_id(i, j + 1, nx)
                area = dx * thickness
                connections.append(_connection(cell, other, scale * _harmonic_vector(conductivity[cell], conductivity[other]) * area / dy, "mm"))

    distance = max(1.0e-12, float(config["edfm"].get("matrix_fracture_distance_factor", 0.25)) * min(dx, dy))
    for segment in fracture_segments:
        matrix_cells = _matrix_cells_for_point(segment.center[0], segment.center[1], x_edges, y_edges, nx, ny)
        for matrix_cell in matrix_cells:
            area = segment.length * thickness
            t_mf = scale * _harmonic_vector(conductivity[matrix_cell], conductivity[segment.cell_index]) * area / (distance * len(matrix_cells))
            connections.append(_connection(matrix_cell, segment.cell_index, t_mf, "mf"))

    by_name: dict[str, list[FractureSegment]] = {}
    for segment in fracture_segments:
        by_name.setdefault(segment.name, []).append(segment)
    for segments in by_name.values():
        ordered = sorted(segments, key=lambda seg: (float(seg.center[0]), float(seg.center[1])))
        for first, second in zip(ordered, ordered[1:]):
            distance_ff = max(1.0e-12, float(np.linalg.norm(second.center - first.center)))
            area = min(first.aperture, second.aperture) * thickness
            multiplier = float(config["edfm"].get("fracture_tangential_multiplier", 1.0))
            connections.append(_connection(first.cell_index, second.cell_index, multiplier * scale * np.minimum(conductivity[first.cell_index], conductivity[second.cell_index]) * area / distance_ff, "ff"))

    for idx, first in enumerate(fracture_segments):
        for second in fracture_segments[idx + 1 :]:
            if first.name == second.name:
                continue
            if _axis_aligned_intersection(first, second) is None:
                continue
            distance_ff = max(1.0e-12, 0.5 * first.length + 0.5 * second.length)
            area = min(first.aperture, second.aperture) * thickness
            multiplier = float(config["edfm"].get("fracture_tangential_multiplier", 1.0))
            connections.append(_connection(first.cell_index, second.cell_index, multiplier * scale * np.minimum(conductivity[first.cell_index], conductivity[second.cell_index]) * area / distance_ff, "ff"))
    return [conn for conn in connections if np.isfinite(conn.transmissibility) and conn.transmissibility >= 0.0]


def _legacy_diffusivity_keys(config: dict[str, Any], component_count: int) -> list[str]:
    default = [f"D{idx + 1}" for idx in range(component_count)]
    return [str(value) for value in config["physics"].get("diffusivity_keys", default)]


def _region_storage_phi(config: dict[str, Any], region: str) -> float:
    return float(config["physics"]["Fai"][region])


def _region_component_conductivity(config: dict[str, Any], region: str, component_count: int) -> np.ndarray:
    keys = _legacy_diffusivity_keys(config, component_count)
    return np.asarray([float(config["physics"][key][region]) for key in keys], dtype=np.float64)


def _region_representative_conductivity(config: dict[str, Any], region: str) -> float:
    component_count = len(config["pressure"]["components"])
    values = _region_component_conductivity(config, region, component_count)
    return float(np.mean(values))


def _cell_storage(cell_phi: np.ndarray, cell_volume: np.ndarray) -> np.ndarray:
    return cell_phi * cell_volume


def _transmissibility_scale(config: dict[str, Any]) -> float:
    base = float(config["physics"].get("transmissibility_scale", 1.0))
    return base * float(config["physics"]["seconds_per_day"])


def _connection(i: int, j: int, component_transmissibility: np.ndarray, kind: str) -> Connection:
    values = np.asarray(component_transmissibility, dtype=np.float64)
    return Connection(int(i), int(j), float(np.mean(values)), kind, tuple(float(value) for value in values))


def connection_transmissibility_matrix(connections: list[Connection], component_count: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for connection in connections:
        if connection.component_transmissibility:
            values = np.asarray(connection.component_transmissibility, dtype=np.float64)
        else:
            values = np.full((component_count,), float(connection.transmissibility), dtype=np.float64)
        if values.size != component_count:
            raise ValueError(f"Connection {connection.kind} {connection.i}->{connection.j} has {values.size} transmissibilities, expected {component_count}.")
        rows.append(values)
    return np.vstack(rows) if rows else np.empty((0, component_count), dtype=np.float64)


def _matrix_cells_for_point(x: float, y: float, x_edges: np.ndarray, y_edges: np.ndarray, nx: int, ny: int) -> list[int]:
    i = int(np.searchsorted(x_edges, x, side="right") - 1)
    j = int(np.searchsorted(y_edges, y, side="right") - 1)
    i_candidates = {max(0, min(nx - 1, i))}
    j_candidates = {max(0, min(ny - 1, j))}
    tol = 1.0e-10
    edge_i = np.where(np.isclose(x_edges, x, atol=tol))[0]
    edge_j = np.where(np.isclose(y_edges, y, atol=tol))[0]
    for edge in edge_i:
        if 0 < edge < nx:
            i_candidates.update([int(edge - 1), int(edge)])
    for edge in edge_j:
        if 0 < edge < ny:
            j_candidates.update([int(edge - 1), int(edge)])
    return sorted(matrix_cell_id(ii, jj, nx) for jj in j_candidates for ii in i_candidates if 0 <= ii < nx and 0 <= jj < ny)


def _axis_aligned_intersection(first: FractureSegment, second: FractureSegment) -> np.ndarray | None:
    if first.is_horizontal == second.is_horizontal:
        return None
    horizontal = first if first.is_horizontal else second
    vertical = second if first.is_horizontal else first
    x = float(vertical.start[0])
    y = float(horizontal.start[1])
    hx0, hx1 = sorted([float(horizontal.start[0]), float(horizontal.end[0])])
    vy0, vy1 = sorted([float(vertical.start[1]), float(vertical.end[1])])
    if hx0 - 1.0e-10 <= x <= hx1 + 1.0e-10 and vy0 - 1.0e-10 <= y <= vy1 + 1.0e-10:
        return np.asarray([x, y], dtype=np.float64)
    return None


def _build_adjacency_if_small(num_cells: int, connections: list[Connection], max_dense_elements: int) -> np.ndarray | None:
    if int(num_cells) > int(max_dense_elements):
        return None
    adjacency = np.eye(num_cells, dtype=np.float32)
    for conn in connections:
        adjacency[conn.i, conn.j] = 1.0
        adjacency[conn.j, conn.i] = 1.0
    return adjacency


def _build_sparse_adjacency(num_cells: int, connections: list[Connection]) -> tuple[np.ndarray, np.ndarray]:
    sources = [np.arange(num_cells, dtype=np.int64)]
    targets = [np.arange(num_cells, dtype=np.int64)]
    if connections:
        conn_i = np.asarray([conn.i for conn in connections], dtype=np.int64)
        conn_j = np.asarray([conn.j for conn in connections], dtype=np.int64)
        sources.extend([conn_i, conn_j])
        targets.extend([conn_j, conn_i])
    source = np.concatenate(sources)
    target = np.concatenate(targets)
    degree = np.bincount(target, minlength=num_cells).astype(np.float32)
    edge_weight = 1.0 / np.maximum(degree[target], 1.0)
    edge_index = np.vstack([source, target]).astype(np.int64)
    return edge_index, edge_weight.astype(np.float32)


def _nearest_matrix_cell(cell_xy: np.ndarray, x: float, y: float) -> int:
    point = np.asarray([x, y], dtype=np.float64)
    return int(np.argmin(np.sum((cell_xy - point[None, :]) ** 2, axis=1)))


def _well_cells(cell_xy: np.ndarray, fracture_segments: list[FractureSegment], matrix_well_cell: int, config: dict[str, Any]) -> np.ndarray:
    cells = {int(matrix_well_cell)}
    if bool(config["well"].get("constrain_connected_fracture", True)):
        point = np.asarray([float(config["well"]["x"]), float(config["well"]["y"])], dtype=np.float64)
        if fracture_segments:
            distances = np.asarray([_distance_point_to_segment(point, segment.start, segment.end) for segment in fracture_segments], dtype=np.float64)
            min_distance = float(np.min(distances))
            tolerance = max(1.0e-8, min_distance + 1.0e-8)
            for segment, distance in zip(fracture_segments, distances):
                if float(distance) <= tolerance:
                    cells.add(int(segment.cell_index))
    return np.asarray(sorted(cells), dtype=np.int64)


def _distance_point_to_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 0.0:
        return float(np.linalg.norm(point - start))
    alpha = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    closest = start + alpha * segment
    return float(np.linalg.norm(point - closest))


def _harmonic_vector(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_arr = np.asarray(a, dtype=np.float64)
    b_arr = np.asarray(b, dtype=np.float64)
    out = np.zeros_like(a_arr, dtype=np.float64)
    mask = (a_arr > 0.0) & (b_arr > 0.0)
    out[mask] = 2.0 * a_arr[mask] * b_arr[mask] / (a_arr[mask] + b_arr[mask])
    return out


def _unique_sorted(values: list[float], tolerance: float = 1.0e-10) -> list[float]:
    ordered = sorted(float(value) for value in values)
    unique: list[float] = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return unique
