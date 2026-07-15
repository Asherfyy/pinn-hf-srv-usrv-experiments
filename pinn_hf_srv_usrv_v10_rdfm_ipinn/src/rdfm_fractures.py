"""RDFM fracture centerlines derived from the existing thin HF rectangles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Rect, ReservoirGeometry
from .rdfm_mesh import RdfmMesh


@dataclass(frozen=True)
class RdfmFracture:
    name: str
    start: np.ndarray
    end: np.ndarray
    aperture: float
    tangent: np.ndarray

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def is_horizontal(self) -> bool:
        return abs(float(self.tangent[0])) >= abs(float(self.tangent[1]))


def fractures_from_geometry(geometry: ReservoirGeometry) -> list[RdfmFracture]:
    return [_fracture_from_rect(rect) for rect in geometry.hf_rects]


def _fracture_from_rect(rect: Rect) -> RdfmFracture:
    cx = 0.5 * (rect.x_min + rect.x_max)
    cy = 0.5 * (rect.y_min + rect.y_max)
    if rect.width >= rect.height:
        start = np.asarray([rect.x_min, cy], dtype=np.float64)
        end = np.asarray([rect.x_max, cy], dtype=np.float64)
        aperture = float(rect.height)
        tangent = np.asarray([1.0, 0.0], dtype=np.float64)
    else:
        start = np.asarray([cx, rect.y_min], dtype=np.float64)
        end = np.asarray([cx, rect.y_max], dtype=np.float64)
        aperture = float(rect.width)
        tangent = np.asarray([0.0, 1.0], dtype=np.float64)
    return RdfmFracture(rect.name, start, end, aperture, tangent)


def fracture_intersections(fractures: list[RdfmFracture], tolerance: float = 1.0e-9) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for idx, first in enumerate(fractures):
        for second in fractures[idx + 1 :]:
            point = _orthogonal_intersection(first, second, tolerance)
            if point is not None:
                points.append(point)
    return points


def _orthogonal_intersection(first: RdfmFracture, second: RdfmFracture, tolerance: float) -> np.ndarray | None:
    if first.is_horizontal == second.is_horizontal:
        return None
    horizontal = first if first.is_horizontal else second
    vertical = second if first.is_horizontal else first
    x = float(vertical.start[0])
    y = float(horizontal.start[1])
    h_x0, h_x1 = sorted([float(horizontal.start[0]), float(horizontal.end[0])])
    v_y0, v_y1 = sorted([float(vertical.start[1]), float(vertical.end[1])])
    if h_x0 - tolerance <= x <= h_x1 + tolerance and v_y0 - tolerance <= y <= v_y1 + tolerance:
        return np.asarray([x, y], dtype=np.float64)
    return None


def fracture_subsegments(fracture: RdfmFracture, mesh: RdfmMesh) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split a centerline by crossed grid lines for stable quadrature."""

    if fracture.length <= 0.0:
        return []
    if fracture.is_horizontal:
        lo, hi = sorted([float(fracture.start[0]), float(fracture.end[0])])
        cuts = [lo, hi]
        cuts.extend(float(x) for x in mesh.x_coords if lo < float(x) < hi)
        y = float(fracture.start[1])
        unique = _unique_sorted(cuts)
        return [(np.asarray([a, y]), np.asarray([b, y])) for a, b in zip(unique, unique[1:]) if b > a]
    lo, hi = sorted([float(fracture.start[1]), float(fracture.end[1])])
    cuts = [lo, hi]
    cuts.extend(float(y) for y in mesh.y_coords if lo < float(y) < hi)
    x = float(fracture.start[0])
    unique = _unique_sorted(cuts)
    return [(np.asarray([x, a]), np.asarray([x, b])) for a, b in zip(unique, unique[1:]) if b > a]


def _unique_sorted(values: list[float], tolerance: float = 1.0e-10) -> list[float]:
    ordered = sorted(float(value) for value in values)
    unique: list[float] = []
    for value in ordered:
        if not unique or abs(value - unique[-1]) > tolerance:
            unique.append(value)
    return unique
