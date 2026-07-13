"""解析矩形 CSG 几何。

v2 的几何模块故意保持“笨但透明”：只做矩形区域判断、绘图边界和采样辅助。不实现
SDF/ADF、不提供距离特征、不构造区域 one-hot 输入。这样网络学习的函数只依赖
`x_hat/y_hat/t_hat`，几何只通过采样位置和 PDE 系数选择进入训练。
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
    """二维轴对齐矩形。"""

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
        """在给定总域内外扩矩形。"""

        return Rect(
            max(bounds.x_min, self.x_min - padding),
            min(bounds.x_max, self.x_max + padding),
            max(bounds.y_min, self.y_min - padding),
            min(bounds.y_max, self.y_max + padding),
            f"{self.name}_expanded",
        )


class ReservoirGeometry:
    """HF/SRV/USRV 解析几何对象。"""

    def __init__(self, geometry_cfg: dict[str, Any]) -> None:
        self.cfg = geometry_cfg
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
        self.dirichlet_segment = geometry_cfg["dirichlet_segment"]

    @staticmethod
    def inside_rect_np(x: np.ndarray | float, y: np.ndarray | float, rect: Rect) -> np.ndarray:
        """NumPy 矩形内点判断，边界点算作内部。"""

        x_arr, y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        return (x_arr >= rect.x_min) & (x_arr <= rect.x_max) & (y_arr >= rect.y_min) & (y_arr <= rect.y_max)

    def inside_domain_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        return self.inside_rect_np(x, y, self.domain)

    def inside_hf_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """判断点是否位于任意 HF 裂缝矩形内。"""

        x_arr, _ = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        mask = np.zeros_like(x_arr, dtype=bool)
        for rect in self.hf_rects:
            mask = mask | self.inside_rect_np(x, y, rect)
        return mask

    def region_id_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """返回区域编号，优先级为 HF > SRV > USRV > OUTSIDE。"""

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
        """Torch 区域编号，仅用于诊断和测试，不参与网络输入。"""

        x_col = x.view(-1, 1)
        y_col = y.view(-1, 1)
        in_domain = self._inside_rect_torch(x_col, y_col, self.domain)
        in_srv = self._inside_rect_torch(x_col, y_col, self.srv_bg)
        in_hf = torch.zeros_like(x_col, dtype=torch.bool)
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

    def draw_overlay(self, ax: Any) -> None:
        """在 Matplotlib 坐标轴上叠加 SRV、HF 和生产边界。"""

        srv = self.srv_bg
        ax.plot([srv.x_min, srv.x_max, srv.x_max, srv.x_min, srv.x_min], [srv.y_min, srv.y_min, srv.y_max, srv.y_max, srv.y_min], color="white", linestyle="--", linewidth=1.0)
        for rect in self.hf_rects:
            ax.plot([rect.x_min, rect.x_max, rect.x_max, rect.x_min, rect.x_min], [rect.y_min, rect.y_min, rect.y_max, rect.y_max, rect.y_min], color="black", linewidth=0.8)
        seg = self.dirichlet_segment
        ax.plot([float(seg["x0"]), float(seg["x1"])], [float(seg["y0"]), float(seg["y1"])], color="magenta", linewidth=2.0)

    def srv_usrv_band_rects(self, width_m: float) -> list[Rect]:
        """返回 SRV-USRV 三条内部界面附近的窄带矩形。"""

        w = float(width_m)
        s = self.srv_bg
        return [
            Rect(s.x_min - w, s.x_min + w, s.y_min, s.y_max, "srv_left_band"),
            Rect(s.x_min, s.x_max, s.y_min - w, s.y_min + w, "srv_bottom_band"),
            Rect(s.x_min, s.x_max, s.y_max - w, s.y_max + w, "srv_top_band"),
        ]

    @staticmethod
    def rect_boundary_segments(rect: Rect) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """返回矩形四条边及从矩形内部指向外部的法向。

        界面连续 loss 需要在材料界面两侧取偏移点。这里仅提供解析线段和法向，不向网络
        注入任何区域编码或距离特征。
        """

        return [
            ((rect.x_min, rect.y_min), (rect.x_min, rect.y_max), (-1.0, 0.0)),
            ((rect.x_max, rect.y_min), (rect.x_max, rect.y_max), (1.0, 0.0)),
            ((rect.x_min, rect.y_min), (rect.x_max, rect.y_min), (0.0, -1.0)),
            ((rect.x_min, rect.y_max), (rect.x_max, rect.y_max), (0.0, 1.0)),
        ]

    def hf_srv_interface_segments(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """返回 HF 裂缝边界候选线段。

        裂缝交叉处或生产端外边界会在后续偏移区域检查中被过滤掉；这里保持线段生成简单，
        让真正的“是否跨 HF-SRV 界面”由 region_id 判断。
        """

        segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
        for rect in self.hf_rects:
            segments.extend(self.rect_boundary_segments(rect))
        return segments

    def srv_usrv_interface_segments(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """返回 SRV-USRV 内部界面线段，法向从 SRV 指向 USRV。"""

        s = self.srv_bg
        return [
            ((s.x_min, s.y_min), (s.x_min, s.y_max), (-1.0, 0.0)),
            ((s.x_min, s.y_min), (s.x_max, s.y_min), (0.0, -1.0)),
            ((s.x_min, s.y_max), (s.x_max, s.y_max), (0.0, 1.0)),
        ]
