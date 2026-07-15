"""Analytic HF/SRV/USRV geometry for v12.

The v12 geometry differs from v3 in one important way: HF fractures are no
longer two-dimensional 0.01 m rectangles. The rectangle entries in the config
are kept only as source data. They are converted to centerline segments, and
HF is classified only on those lines. The surrounding area remains SRV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


REGION_OUTSIDE = -1
REGION_HF = 0
REGION_SRV = 1
REGION_USRV = 2
REGION_NAMES = ["HF", "SRV", "USRV"]


@dataclass(frozen=True)
class Rect:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    name: str = ""

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def expanded(self, padding: float, bounds: "Rect") -> "Rect":
        return Rect(
            max(bounds.x_min, self.x_min - padding),
            min(bounds.x_max, self.x_max + padding),
            max(bounds.y_min, self.y_min - padding),
            min(bounds.y_max, self.y_max + padding),
            f"{self.name}_expanded",
        )


@dataclass(frozen=True)
class FractureLine:
    x0: float
    y0: float
    x1: float
    y1: float
    aperture: float
    name: str = ""

    @property
    def dx(self) -> float:
        return self.x1 - self.x0

    @property
    def dy(self) -> float:
        return self.y1 - self.y0

    @property
    def length(self) -> float:
        return float(np.hypot(self.dx, self.dy))

    @property
    def is_horizontal(self) -> bool:
        return abs(self.dy) <= abs(self.dx)

    @property
    def tangent(self) -> tuple[float, float]:
        length = max(self.length, 1.0e-12)
        return self.dx / length, self.dy / length

    @property
    def normal_pair(self) -> tuple[tuple[float, float], tuple[float, float]]:
        tx, ty = self.tangent
        n0 = (-ty, tx)
        n1 = (ty, -tx)
        return n0, n1


class ReservoirGeometry:
    """HF/SRV/USRV geometry with line-based hydraulic fractures."""

    def __init__(self, geometry_cfg: dict[str, Any]) -> None:
        self.cfg = geometry_cfg
        self.hf_representation = str(geometry_cfg.get("hf_representation", "line")).lower()
        self.line_tol = float(geometry_cfg.get("line_region_tolerance_m", 1.0e-8))
        self.domain = Rect(float(geometry_cfg["x_min"]), float(geometry_cfg["x_max"]), float(geometry_cfg["y_min"]), float(geometry_cfg["y_max"]), "domain")
        self.srv_bg = Rect(float(geometry_cfg["srv_x_min"]), float(geometry_cfg["srv_x_max"]), float(geometry_cfg["srv_y_min"]), float(geometry_cfg["srv_y_max"]), "srv")
        self.main_frac = Rect(
            float(geometry_cfg["main_frac_x_min"]),
            float(geometry_cfg["main_frac_x_max"]),
            float(geometry_cfg["main_frac_y_min"]),
            float(geometry_cfg["main_frac_y_max"]),
            "main_frac",
        )
        self.secondary_fractures = [
            Rect(float(x0), float(x1), float(y0), float(y1), f"secondary_{idx + 1}")
            for idx, (x0, x1, y0, y1) in enumerate(geometry_cfg["secondary_fractures"])
        ]
        self.hf_rects = [self.main_frac, *self.secondary_fractures]
        self.hf_lines = [self._rect_to_centerline(rect) for rect in self.hf_rects]
        self.dirichlet_segment = geometry_cfg["dirichlet_segment"]

    @staticmethod
    def _rect_to_centerline(rect: Rect) -> FractureLine:
        if rect.width >= rect.height:
            y = 0.5 * (rect.y_min + rect.y_max)
            return FractureLine(rect.x_min, y, rect.x_max, y, aperture=max(rect.height, 0.0), name=rect.name)
        x = 0.5 * (rect.x_min + rect.x_max)
        return FractureLine(x, rect.y_min, x, rect.y_max, aperture=max(rect.width, 0.0), name=rect.name)

    @property
    def hf_as_line(self) -> bool:
        return self.hf_representation == "line"

    @staticmethod
    def inside_rect_np(x: np.ndarray | float, y: np.ndarray | float, rect: Rect) -> np.ndarray:
        x_arr, y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        return (x_arr >= rect.x_min) & (x_arr <= rect.x_max) & (y_arr >= rect.y_min) & (y_arr <= rect.y_max)

    def inside_domain_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        return self.inside_rect_np(x, y, self.domain)

    def inside_hf_line_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_arr, y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        mask = np.zeros_like(x_arr, dtype=bool)
        tol = max(self.line_tol, 0.0)
        for line in self.hf_lines:
            if line.is_horizontal:
                x0, x1 = sorted([line.x0, line.x1])
                mask |= (x_arr >= x0 - tol) & (x_arr <= x1 + tol) & (np.abs(y_arr - line.y0) <= tol)
            else:
                y0, y1 = sorted([line.y0, line.y1])
                mask |= (y_arr >= y0 - tol) & (y_arr <= y1 + tol) & (np.abs(x_arr - line.x0) <= tol)
        return mask

    def inside_hf_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        if self.hf_as_line:
            return self.inside_hf_line_np(x, y)
        x_arr, _ = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        mask = np.zeros_like(x_arr, dtype=bool)
        for rect in self.hf_rects:
            mask |= self.inside_rect_np(x, y, rect)
        return mask

    def region_id_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        x_arr, _ = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        region = np.full(x_arr.shape, REGION_OUTSIDE, dtype=np.int64)
        in_domain = self.inside_domain_np(x, y)
        in_srv = self.inside_rect_np(x, y, self.srv_bg)
        in_hf = self.inside_hf_np(x, y)
        region[in_domain] = REGION_USRV
        region[in_domain & in_srv] = REGION_SRV
        region[in_domain & in_hf] = REGION_HF
        return region

    def region_id_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_col = x.view(-1, 1)
        y_col = y.view(-1, 1)
        in_domain = self._inside_rect_torch(x_col, y_col, self.domain)
        in_srv = self._inside_rect_torch(x_col, y_col, self.srv_bg)
        in_hf = self._inside_hf_line_torch(x_col, y_col) if self.hf_as_line else torch.zeros_like(x_col, dtype=torch.bool)
        if not self.hf_as_line:
            for rect in self.hf_rects:
                in_hf = in_hf | self._inside_rect_torch(x_col, y_col, rect)
        region = torch.full_like(x_col, REGION_OUTSIDE, dtype=torch.long)
        region = torch.where(in_domain, torch.full_like(region, REGION_USRV), region)
        region = torch.where(in_domain & in_srv, torch.full_like(region, REGION_SRV), region)
        region = torch.where(in_domain & in_hf, torch.full_like(region, REGION_HF), region)
        return region

    @staticmethod
    def _inside_rect_torch(x: torch.Tensor, y: torch.Tensor, rect: Rect) -> torch.Tensor:
        return (x >= rect.x_min) & (x <= rect.x_max) & (y >= rect.y_min) & (y <= rect.y_max)

    def _inside_hf_line_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        tol = max(self.line_tol, 0.0)
        mask = torch.zeros_like(x, dtype=torch.bool)
        for line in self.hf_lines:
            if line.is_horizontal:
                x0, x1 = sorted([line.x0, line.x1])
                mask = mask | ((x >= x0 - tol) & (x <= x1 + tol) & (torch.abs(y - line.y0) <= tol))
            else:
                y0, y1 = sorted([line.y0, line.y1])
                mask = mask | ((y >= y0 - tol) & (y <= y1 + tol) & (torch.abs(x - line.x0) <= tol))
        return mask

    def draw_overlay(self, ax: Any) -> None:
        srv = self.srv_bg
        ax.plot([srv.x_min, srv.x_max, srv.x_max, srv.x_min, srv.x_min], [srv.y_min, srv.y_min, srv.y_max, srv.y_max, srv.y_min], color="white", linestyle="--", linewidth=1.0)
        for line in self.hf_lines:
            ax.plot([line.x0, line.x1], [line.y0, line.y1], color="black", linewidth=0.9)
        xw, yw = self.dirichlet_point()
        ax.plot([xw], [yw], marker="o", color="magenta", markersize=3.0)

    def dirichlet_point(self) -> tuple[float, float]:
        seg = self.dirichlet_segment
        return 0.5 * (float(seg["x0"]) + float(seg["x1"])), 0.5 * (float(seg["y0"]) + float(seg["y1"]))

    def srv_usrv_band_rects(self, width_m: float) -> list[Rect]:
        w = float(width_m)
        s = self.srv_bg
        return [
            Rect(s.x_min - w, s.x_min + w, s.y_min, s.y_max, "srv_left_band"),
            Rect(s.x_min, s.x_max, s.y_min - w, s.y_min + w, "srv_bottom_band"),
            Rect(s.x_min, s.x_max, s.y_max - w, s.y_max + w, "srv_top_band"),
        ]

    @staticmethod
    def rect_boundary_segments(rect: Rect) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        return [
            ((rect.x_min, rect.y_min), (rect.x_min, rect.y_max), (-1.0, 0.0)),
            ((rect.x_max, rect.y_min), (rect.x_max, rect.y_max), (1.0, 0.0)),
            ((rect.x_min, rect.y_min), (rect.x_max, rect.y_min), (0.0, -1.0)),
            ((rect.x_min, rect.y_max), (rect.x_max, rect.y_max), (0.0, 1.0)),
        ]

    def hf_srv_interface_segments(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        if not self.hf_as_line:
            segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
            for rect in self.hf_rects:
                segments.extend(self.rect_boundary_segments(rect))
            return segments

        segments = []
        for line in self.hf_lines:
            p0 = (line.x0, line.y0)
            p1 = (line.x1, line.y1)
            for normal in line.normal_pair:
                segments.append((p0, p1, normal))
        return segments

    def srv_usrv_interface_segments(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        s = self.srv_bg
        return [
            ((s.x_min, s.y_min), (s.x_min, s.y_max), (-1.0, 0.0)),
            ((s.x_min, s.y_min), (s.x_max, s.y_min), (0.0, -1.0)),
            ((s.x_min, s.y_max), (s.x_max, s.y_max), (0.0, 1.0)),
        ]
