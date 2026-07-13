"""解析 CSG 几何与 SDF/ADF 距离特征。

本模块只使用矩形布尔逻辑描述复杂几何，不依赖 COMSOL API。这样做的好处是：
第一版可以稳定跑通 PINN 训练闭环；后续若从 COMSOL 导入更复杂边界，也可以把
这里的距离函数替换为更高阶几何表示，而不影响模型、损失和训练脚本。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


REGION_OUTSIDE = -1
REGION_HF = 0
REGION_SRV = 1
REGION_USRV = 2
REGION_NAMES = ["HF", "SRV", "USRV"]


@dataclass
class Rect:
    """二维轴对齐矩形。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    name: str = ""

    @property
    def width(self) -> float:
        """矩形宽度。"""

        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        """矩形高度。"""

        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        """矩形面积，用于 HF 裂缝采样点数分配。"""

        return max(self.width, 0.0) * max(self.height, 0.0)


class ReservoirGeometry:
    """HF/SRV/USRV 储层几何对象。

    功能：
    1. 判断点是否位于总区域内；
    2. 判断点属于 HF、SRV 还是 USRV；
    3. 计算点到外边界、裂缝、Dirichlet 边界的距离；
    4. 生成界面边界段；
    5. 为 PINN 构造几何嵌入特征。

    区域判断采用固定优先级：裂缝 HF > SRV 背景矩形 > USRV 背景。这个优先级
    非常重要，因为主裂缝和次级裂缝都嵌在 SRV 背景区内，如果先判断 SRV，
    裂缝点会被误分到 SRV，导致分区系数和界面损失全部偏离物理含义。
    """

    def __init__(self, geometry_cfg: dict[str, Any], data_dir: str | Path | None = None) -> None:
        self.cfg = geometry_cfg
        self.domain = Rect(
            float(geometry_cfg["x_min"]),
            float(geometry_cfg["x_max"]),
            float(geometry_cfg["y_min"]),
            float(geometry_cfg["y_max"]),
            "domain",
        )
        self.srv_bg = Rect(
            float(geometry_cfg["srv_x_min"]),
            float(geometry_cfg["srv_x_max"]),
            float(geometry_cfg["srv_y_min"]),
            float(geometry_cfg["srv_y_max"]),
            "srv_bg",
        )
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
        self.dirichlet_points_np = self._load_dirichlet_points(data_dir)

        lx = self.domain.x_max - self.domain.x_min
        ly = self.domain.y_max - self.domain.y_min
        self.l_ref = max(lx, ly)

    def _load_dirichlet_points(self, data_dir: str | Path | None) -> np.ndarray | None:
        """预留 CSV 形式 Dirichlet 点云接口。

        当前默认仍使用解析线段作为硬约束主定义，因为线段距离可以保证任意线段点
        上的 ADF 精确为 0。若后续 COMSOL 导出离散边界点云，可在这里读入并与解析
        线段距离取最小值，逐步过渡到点云边界。
        """

        if data_dir is None:
            return None
        path = Path(data_dir) / "dirichlet_boundary.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return None
        if {"x", "y"}.issubset(df.columns) and len(df) > 0:
            return df[["x", "y"]].to_numpy(dtype=float)
        return None

    @staticmethod
    def _as_column_torch(value: torch.Tensor) -> torch.Tensor:
        """把一维张量整理为 `[N, 1]`，便于后续按列拼接特征。"""

        if value.ndim == 1:
            return value.view(-1, 1)
        return value

    def _smooth_abs_torch(self, value: torch.Tensor, eps_m: float) -> torch.Tensor:
        """Smooth absolute value used only by distance features."""

        eps = torch.as_tensor(max(float(eps_m), 1.0e-12), dtype=value.dtype, device=value.device)
        return torch.sqrt(value * value + eps * eps)

    @staticmethod
    def _weighted_smooth_min_torch(values: torch.Tensor, beta: float) -> torch.Tensor:
        """Differentiable min approximation without the log-sum-exp bias."""

        beta_value = max(float(beta), 1.0e-12)
        shifted = values - torch.min(values, dim=1, keepdim=True).values
        weights = torch.softmax(-beta_value * shifted, dim=1)
        return torch.sum(weights * values, dim=1, keepdim=True)

    @staticmethod
    def inside_rect_np(x: np.ndarray | float, y: np.ndarray | float, rect: Rect) -> np.ndarray:
        """NumPy 版本矩形内点判断，边界点视为区域内部。"""

        x_arr, y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        return (
            (x_arr >= rect.x_min)
            & (x_arr <= rect.x_max)
            & (y_arr >= rect.y_min)
            & (y_arr <= rect.y_max)
        )

    def inside_domain_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """判断点是否位于总计算区域内。"""

        return self.inside_rect_np(x, y, self.domain)

    def inside_hf_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """判断点是否位于任意裂缝矩形内。"""

        x_arr, _y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        mask = np.zeros_like(x_arr, dtype=bool)
        for rect in self.hf_rects:
            mask = mask | self.inside_rect_np(x, y, rect)
        return mask

    def inside_srv_bg_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """判断点是否位于 SRV 背景矩形内。"""

        return self.inside_rect_np(x, y, self.srv_bg)

    def region_id_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """返回区域编号：HF=0, SRV=1, USRV=2, 区域外=-1。

        该函数会被采样器反复调用，因此使用向量化布尔 mask，避免逐点 Python 循环。
        """

        x_arr, _y_arr = np.broadcast_arrays(np.asarray(x), np.asarray(y))
        shape = x_arr.shape
        region = np.full(shape, REGION_OUTSIDE, dtype=np.int64)

        in_domain = self.inside_domain_np(x, y)
        in_hf = self.inside_hf_np(x, y)
        in_srv_bg = self.inside_srv_bg_np(x, y)

        region[in_domain] = REGION_USRV
        region[in_domain & in_srv_bg] = REGION_SRV
        region[in_domain & in_hf] = REGION_HF
        return region

    def region_onehot_np(self, x: np.ndarray | float, y: np.ndarray | float) -> np.ndarray:
        """返回固定顺序 one-hot: HF, SRV, USRV。"""

        region = self.region_id_np(x, y)
        flat_region = region.reshape(-1)
        onehot = np.zeros((flat_region.size, 3), dtype=np.float32)
        for idx in (REGION_HF, REGION_SRV, REGION_USRV):
            onehot[:, idx] = flat_region == idx
        return onehot.reshape((*region.shape, 3))

    @staticmethod
    def inside_rect_torch(x: torch.Tensor, y: torch.Tensor, rect: Rect) -> torch.Tensor:
        """Torch 版本矩形内点判断。

        返回布尔张量，不参与梯度传播；几何分类本身是离散操作，只用于选择分区系数。
        """

        x = ReservoirGeometry._as_column_torch(x)
        y = ReservoirGeometry._as_column_torch(y)
        return (
            (x >= rect.x_min)
            & (x <= rect.x_max)
            & (y >= rect.y_min)
            & (y <= rect.y_max)
        )

    def region_onehot_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Torch 版本区域 one-hot，顺序固定为 HF/SRV/USRV。

        one-hot 用张量运算构造，后续可直接与系数张量相乘选择分区常数系数。
        """

        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        in_domain = self.inside_rect_torch(x, y, self.domain)
        in_srv_bg = self.inside_rect_torch(x, y, self.srv_bg)

        in_hf = torch.zeros_like(x, dtype=torch.bool)
        for rect in self.hf_rects:
            in_hf = in_hf | self.inside_rect_torch(x, y, rect)

        hf = in_domain & in_hf
        srv = in_domain & (~in_hf) & in_srv_bg
        usrv = in_domain & (~in_hf) & (~in_srv_bg)
        return torch.cat(
            [
                hf.to(dtype=x.dtype),
                srv.to(dtype=x.dtype),
                usrv.to(dtype=x.dtype),
            ],
            dim=1,
        )

    def region_id_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Torch 版本区域编号：HF=0, SRV=1, USRV=2, 区域外=-1。

        界面偏移检查需要知道偏移点是否真的落在目标区域，而 one-hot 对区域外点会给出
        `[0,0,0]`，不适合直接 `argmax`。因此这里显式保留 `-1` 区域外编号。
        """

        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        in_domain = self.inside_rect_torch(x, y, self.domain)
        in_srv_bg = self.inside_rect_torch(x, y, self.srv_bg)
        in_hf = torch.zeros_like(x, dtype=torch.bool)
        for rect in self.hf_rects:
            in_hf = in_hf | self.inside_rect_torch(x, y, rect)

        region = torch.full_like(x, REGION_OUTSIDE, dtype=torch.long)
        region = torch.where(in_domain, torch.full_like(region, REGION_USRV), region)
        region = torch.where(in_domain & in_srv_bg, torch.full_like(region, REGION_SRV), region)
        region = torch.where(in_domain & in_hf, torch.full_like(region, REGION_HF), region)
        return region

    def signed_distance_to_rect_torch(self, x: torch.Tensor, y: torch.Tensor, rect: Rect) -> torch.Tensor:
        """返回轴对齐矩形的近似 signed distance。

        约定：矩形内部为负，边界为 0，外部为正。相比旧版“内部一律为 0”的无符号距离，
        SDF 能同时表达点在几何内外的位置和到边界的尺度，对 PINN 的几何嵌入更有信息量。
        """

        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        cx = 0.5 * (rect.x_min + rect.x_max)
        cy = 0.5 * (rect.y_min + rect.y_max)
        hx = 0.5 * (rect.x_max - rect.x_min)
        hy = 0.5 * (rect.y_max - rect.y_min)

        qx = torch.abs(x - cx) - hx
        qy = torch.abs(y - cy) - hy
        ox = torch.clamp(qx, min=0.0)
        oy = torch.clamp(qy, min=0.0)
        outside2 = ox * ox + oy * oy
        outside = torch.where(outside2 > 0.0, torch.sqrt(outside2 + 1.0e-24), torch.zeros_like(outside2))
        inside = torch.minimum(torch.maximum(qx, qy), torch.zeros_like(qx))
        return outside + inside

    def distance_to_rect_torch(self, x: torch.Tensor, y: torch.Tensor, rect: Rect) -> torch.Tensor:
        """返回点到矩形实体的归一化无符号距离。

        这是 SDF 的正部：矩形内部为 0，外部为到矩形的欧氏距离。该函数保留给“点到
        某个实体”的传统距离查询；真正用于几何嵌入的 HF 距离见 `distance_to_hf_torch`。
        """

        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        sdf = self.signed_distance_to_rect_torch(x, y, rect)
        dist = torch.clamp(sdf, min=0.0)
        return torch.clamp(dist / self.l_ref, min=0.0, max=1.0)

    def distance_to_outer_boundary_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """返回点到总区域外边界的归一化最近距离。

        PINN 在复杂几何中容易把边界层学得较粗糙。把外边界距离作为输入特征，可以让
        网络更容易表达边界附近与内部不同的变化尺度。
        """

        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        # 对总矩形使用 SDF 后取绝对值，内部点得到到外边界的距离，外部点得到到总域的距离。
        # 这比逐边 min 更接近标准 SDF 表达，在角点外侧也能给出欧氏距离。
        dist = torch.abs(self.signed_distance_to_rect_torch(x, y, self.domain))
        return torch.clamp(dist / self.l_ref, min=0.0, max=1.0)

    def distance_to_hf_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """返回点到 HF 裂缝边界的归一化最近距离。

        旧版在 HF 内部全部返回 0，会让网络无法区分裂缝中心和裂缝边界。这里改为
        `min(abs(SDF_i))`，因此无论在 HF 内侧还是外侧，都表达“离最近裂缝边界多远”。
        """

        smooth_eps_m = float(self.cfg.get("hf_distance_smooth_eps_m", 5.0e-4))
        smooth_beta = float(self.cfg.get("hf_distance_smooth_beta", 500.0))
        dists = [
            self._smooth_abs_torch(self.signed_distance_to_rect_torch(x, y, rect), smooth_eps_m) / self.l_ref
            for rect in self.hf_rects
        ]
        dist = self._weighted_smooth_min_torch(torch.cat(dists, dim=1), smooth_beta)
        return torch.clamp(dist, min=0.0, max=1.0)

    def _distance_to_dirichlet_segment_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算点到默认 Dirichlet 线段的归一化距离。"""

        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        seg = self.dirichlet_segment
        ax = torch.as_tensor(float(seg["x0"]), dtype=x.dtype, device=x.device)
        ay = torch.as_tensor(float(seg["y0"]), dtype=x.dtype, device=x.device)
        bx = torch.as_tensor(float(seg["x1"]), dtype=x.dtype, device=x.device)
        by = torch.as_tensor(float(seg["y1"]), dtype=x.dtype, device=x.device)

        vx = bx - ax
        vy = by - ay
        wx = x - ax
        wy = y - ay
        denom = vx * vx + vy * vy
        if float(denom.detach().cpu()) == 0.0:
            closest_x = ax
            closest_y = ay
        else:
            tau = torch.clamp((wx * vx + wy * vy) / denom, min=0.0, max=1.0)
            closest_x = ax + tau * vx
            closest_y = ay + tau * vy
        dist2 = torch.clamp((x - closest_x) ** 2 + (y - closest_y) ** 2, min=0.0)
        dist = torch.where(dist2 > 0.0, torch.sqrt(dist2 + 1.0e-24), torch.zeros_like(dist2))
        return torch.clamp(dist / self.l_ref, min=0.0, max=1.0)

    def _distance_to_dirichlet_points_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
        """计算点到 CSV Dirichlet 点云的归一化最近距离。"""

        if self.dirichlet_points_np is None:
            return None
        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        points = torch.as_tensor(self.dirichlet_points_np, dtype=x.dtype, device=x.device)
        px = points[:, 0].view(1, -1)
        py = points[:, 1].view(1, -1)
        dist2 = (x - px) ** 2 + (y - py) ** 2
        min_dist2 = torch.clamp(torch.min(dist2, dim=1, keepdim=True).values, min=0.0)
        dist = torch.where(min_dist2 > 0.0, torch.sqrt(min_dist2 + 1.0e-24), torch.zeros_like(min_dist2))
        return torch.clamp(dist / self.l_ref, min=0.0, max=1.0)

    def distance_to_dirichlet_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """返回点到 Dirichlet 边界的归一化距离。

        解析线段距离保证硬约束在整条边界段上精确成立；CSV 点云距离作为预留接口，
        只会进一步缩小距离，不会破坏默认线段约束。
        """

        d_segment = self._distance_to_dirichlet_segment_torch(x, y)
        d_points = self._distance_to_dirichlet_points_torch(x, y)
        if d_points is None:
            return d_segment
        return torch.minimum(d_segment, d_points)

    def adf_dirichlet_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """返回 Dirichlet 硬约束因子 B_D(x,y)。

        B_D 在 Dirichlet 边界上为 0，内部为正。模型最终输出写成
        `u_ref + B_D N_theta`，因此无论网络扰动项是多少，边界上都会退化为给定值。
        注意：网络输入中的 Dirichlet 距离特征仍保持 `d_D/l_ref` 形式；这里只把该距离
        恢复为物理距离后用于 hard constraint，避免全域尺度 360 m 让 B_D 增长过慢。
        """

        d_hat = self.distance_to_dirichlet_torch(x, y)
        power = float(self.cfg.get("dirichlet_adf_power", 1.0))
        if power <= 0.0:
            raise ValueError(f"dirichlet_adf_power 必须为正数，当前为 {power:g}。")

        adf_type = str(self.cfg.get("dirichlet_adf_type", "legacy_linear")).lower()
        if adf_type == "legacy_linear":
            # 兼容旧实验：旧版直接使用归一化距离 d_hat^p。该模式会让边界外数米处的
            # B_D 仍然很小，因此 default.yaml 不再使用它。
            if abs(power - 1.0) < 1.0e-12:
                return d_hat
            return torch.pow(torch.clamp(d_hat, min=0.0), power)
        if adf_type != "tanh":
            raise ValueError(
                f"不支持的 dirichlet_adf_type: {adf_type}。"
                "请使用 'tanh' 或 'legacy_linear'。"
            )

        length_m = float(self.cfg.get("dirichlet_adf_length_m", 1.0))
        if length_m <= 0.0:
            raise ValueError(f"dirichlet_adf_length_m 必须为正数，当前为 {length_m:g} m。")

        d_m = d_hat * self.l_ref
        scaled = torch.clamp(d_m / length_m, min=0.0)
        # tanh(z) 在 z=0 时严格为 0，且对正 z 单调增加并小于 1；这能让网络扰动项
        # 在离开生产边界约几个局部长度后恢复 O(1) 自由度。
        return torch.tanh(torch.pow(scaled, power))

    def build_distance_features_torch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """构造三个距离特征: d_outer, d_HF, d_Dirichlet。"""

        x = self._as_column_torch(x)
        y = self._as_column_torch(y)
        return torch.cat(
            [
                self.distance_to_outer_boundary_torch(x, y),
                self.distance_to_hf_torch(x, y),
                self.distance_to_dirichlet_torch(x, y),
            ],
            dim=1,
        )

    def rect_boundary_segments(self, rect: Rect) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """返回矩形四条边及其外法向，用于界面采样和绘图。"""

        return [
            ((rect.x_min, rect.y_min), (rect.x_min, rect.y_max), (-1.0, 0.0)),
            ((rect.x_max, rect.y_min), (rect.x_max, rect.y_max), (1.0, 0.0)),
            ((rect.x_min, rect.y_min), (rect.x_max, rect.y_min), (0.0, -1.0)),
            ((rect.x_min, rect.y_max), (rect.x_max, rect.y_max), (0.0, 1.0)),
        ]

    def hf_srv_interface_segments(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """返回 HF 裂缝边界段。

        法向定义为从 HF 指向外侧。界面损失沿该法向两侧微小偏移，从而获得 HF 侧
        与周围 SRV/USRV 侧的压力和通量。主裂缝右端 x=360 同时是 Dirichlet 边界，
        第一版将其交给 hard constraint 处理，不再作为材料界面重复施加通量连续。
        """

        segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
        for rect in self.hf_rects:
            for segment in self.rect_boundary_segments(rect):
                p0, p1, normal = segment
                on_right_outer = (
                    abs(p0[0] - self.domain.x_max) < 1.0e-12
                    and abs(p1[0] - self.domain.x_max) < 1.0e-12
                    and normal == (1.0, 0.0)
                )
                if not on_right_outer:
                    segments.append(segment)
        return segments

    def srv_usrv_interface_segments(self) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
        """返回 SRV-USRV 内部界面段。

        第一版排除 x=360 这条外边界，因为它同时是总区域外边界，容易与 Neumann/Dirichlet
        边界条件混在一起；只保留 x=180、y=37.5、y=112.5 三条内部界面。
        """

        r = self.srv_bg
        return [
            ((r.x_min, r.y_min), (r.x_min, r.y_max), (-1.0, 0.0)),
            ((r.x_min, r.y_min), (r.x_max, r.y_min), (0.0, -1.0)),
            ((r.x_min, r.y_max), (r.x_max, r.y_max), (0.0, 1.0)),
        ]
