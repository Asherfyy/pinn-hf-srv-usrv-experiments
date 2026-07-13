"""训练点采样器。

采样器保留 HF/SRV/USRV 分区采样、近界面 PDE 加密，并额外生成界面连续 loss 使用
的候选点。近界面 PDE 点仍只是普通 collocation 点；界面 loss 点会在 physics 层沿
法向做两侧偏移并过滤。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, Rect, ReservoirGeometry
from .utils import tensor_from_numpy


class ReservoirSampler:
    """HF/SRV/USRV 训练点采样器。"""

    def __init__(
        self,
        geometry: ReservoirGeometry,
        sampler_cfg: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> None:
        self.geometry = geometry
        self.cfg = sampler_cfg
        self.device = device
        self.dtype = dtype
        self.rng = np.random.default_rng(int(seed))

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return tensor_from_numpy(array, self.device, self.dtype)

    def sample_time(self, n: int) -> np.ndarray:
        """使用 log1p 均匀策略采样时间。

        压力衰减常常集中在早期和中期。如果直接在物理时间上均匀采样，1000 d 区间会
        把大量点浪费在长期尾部；log1p 空间均匀能同时覆盖 0 附近和 10~100 d 区间。
        """

        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        t_min = float(self.cfg["t_min"])
        t_max = float(self.cfg["t_max"])
        if t_min < 0.0:
            raise ValueError(f"t_min 必须 >= 0，当前 {t_min:g}。")
        if t_max <= t_min:
            raise ValueError(f"t_max 必须大于 t_min，当前 {t_max:g} <= {t_min:g}。")
        strategy = str(self.cfg.get("time_strategy", "log1p_uniform")).lower()
        if strategy != "log1p_uniform":
            raise ValueError("v2 只实现 log1p_uniform 时间采样。")
        s = self.rng.uniform(np.log1p(t_min), np.log1p(t_max), size=(n, 1))
        t = np.expm1(s)
        t = np.clip(t, t_min, t_max).astype(np.float64)
        if not np.all(np.isfinite(t)):
            raise RuntimeError("时间采样出现 NaN 或 Inf。")
        self.rng.shuffle(t, axis=0)
        return t

    def _allocate_counts(self, n: int, weights: list[float]) -> list[int]:
        """按面积或长度权重分配点数，并保证总数严格等于 n。"""

        if n <= 0:
            return [0 for _ in weights]
        w = np.asarray(weights, dtype=float)
        if np.sum(w) <= 0.0:
            w = np.ones_like(w)
        raw = n * w / np.sum(w)
        counts = np.floor(raw).astype(int)
        remainder = n - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for idx in order[:remainder]:
                counts[idx] += 1
        return counts.tolist()

    def _sample_rect_xy(self, rect: Rect, n: int) -> np.ndarray:
        """在矩形内均匀采样 x/y。"""

        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        x = self.rng.uniform(rect.x_min, rect.x_max, size=(n, 1))
        y = self.rng.uniform(rect.y_min, rect.y_max, size=(n, 1))
        return np.hstack([x, y]).astype(np.float64)

    def _sample_hf_xy(self, n: int) -> np.ndarray:
        """在所有 HF 裂缝矩形内采样。"""

        counts = self._allocate_counts(n, [rect.area for rect in self.geometry.hf_rects])
        chunks = [self._sample_rect_xy(rect, count) for rect, count in zip(self.geometry.hf_rects, counts) if count > 0]
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)
        self.rng.shuffle(xy, axis=0)
        return xy[:n]

    def _sample_region_xy(self, n: int, rect: Rect, target_region: int) -> np.ndarray:
        """在背景矩形内拒绝采样指定区域。"""

        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        accepted: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < n and attempts < 300:
            attempts += 1
            batch_n = max(128, (n - total) * 3)
            xy = self._sample_rect_xy(rect, batch_n)
            region = self.geometry.region_id_np(xy[:, 0], xy[:, 1])
            keep = xy[region == target_region]
            if keep.size > 0:
                accepted.append(keep)
                total += keep.shape[0]
        if total < n:
            raise RuntimeError(f"区域采样失败，target_region={target_region}, 得到 {total}/{n}。")
        return np.vstack(accepted)[:n].astype(np.float64)

    def _sample_from_rects_for_region(self, rects: list[Rect], n: int, target_region: int) -> np.ndarray:
        """从多个窄带矩形中采样指定区域点。"""

        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        counts = self._allocate_counts(n, [max(rect.area, 1.0e-12) for rect in rects])
        chunks: list[np.ndarray] = []
        for rect, count in zip(rects, counts):
            if count <= 0:
                continue
            chunks.append(self._sample_region_xy(count, rect, target_region))
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)
        self.rng.shuffle(xy, axis=0)
        return xy[:n]

    def sample_near_hf_srv_points(self) -> dict[str, np.ndarray]:
        """采样 HF 外侧、属于 SRV 的普通 PDE 加密点。"""

        n = int(self.cfg["n_near_hf_srv"])
        width = float(self.cfg["hf_srv_band_width_m"])
        rects = [rect.expanded(width, self.geometry.domain) for rect in self.geometry.hf_rects]
        return {"srv": self._sample_from_rects_for_region(rects, n, REGION_SRV)}

    def sample_near_srv_usrv_points(self) -> dict[str, np.ndarray]:
        """采样 SRV-USRV 内部界面窄带两侧普通 PDE 点。"""

        n = int(self.cfg["n_near_srv_usrv"])
        width = float(self.cfg["srv_usrv_band_width_m"])
        rects = self.geometry.srv_usrv_band_rects(width)
        n_srv = n // 2
        n_usrv = n - n_srv
        return {
            "srv": self._sample_from_rects_for_region(rects, n_srv, REGION_SRV),
            "usrv": self._sample_from_rects_for_region(rects, n_usrv, REGION_USRV),
        }

    def _make_xyt(self, xy: np.ndarray) -> torch.Tensor:
        t = self.sample_time(xy.shape[0])
        return self._to_tensor(np.hstack([xy, t]).astype(np.float64))

    def sample_pde_points(self) -> dict[str, torch.Tensor]:
        """采样三大区域 PDE 点，并合并近界面加密点。"""

        hf_xy = self._sample_hf_xy(int(self.cfg["n_pde_hf"]))
        srv_xy = self._sample_region_xy(int(self.cfg["n_pde_srv"]), self.geometry.srv_bg, REGION_SRV)
        usrv_xy = self._sample_region_xy(int(self.cfg["n_pde_usrv"]), self.geometry.domain, REGION_USRV)
        near_hf = self.sample_near_hf_srv_points()
        near_srv_usrv = self.sample_near_srv_usrv_points()
        srv_xy = np.vstack([srv_xy, near_hf["srv"], near_srv_usrv["srv"]])
        usrv_xy = np.vstack([usrv_xy, near_srv_usrv["usrv"]])
        return {"hf": self._make_xyt(hf_xy), "srv": self._make_xyt(srv_xy), "usrv": self._make_xyt(usrv_xy)}

    def sample_dirichlet_boundary_points(self) -> dict[str, torch.Tensor]:
        """采样生产端 Dirichlet soft loss 点。"""

        n = int(self.cfg["n_dirichlet"])
        seg = self.geometry.dirichlet_segment
        x = np.full((n, 1), float(seg["x0"]), dtype=np.float64)
        y = self.rng.uniform(float(seg["y0"]), float(seg["y1"]), size=(n, 1)).astype(np.float64)
        t = self.sample_time(n)
        return {"xyt": self._to_tensor(np.hstack([x, y, t]).astype(np.float64))}

    def _sample_right_neumann_y(self, n: int) -> np.ndarray:
        """右边界采样时排除生产边界小线段及邻域。"""

        seg = self.geometry.dirichlet_segment
        gap = 0.05
        low = float(seg["y0"]) - gap
        high = float(seg["y1"]) + gap
        values: list[np.ndarray] = []
        total = 0
        while total < n:
            y = self.rng.uniform(self.geometry.domain.y_min, self.geometry.domain.y_max, size=(max(64, 2 * (n - total)), 1))
            keep = y[(y[:, 0] < low) | (y[:, 0] > high)]
            if keep.size > 0:
                values.append(keep.reshape(-1, 1))
                total += keep.size
        return np.vstack(values)[:n].astype(np.float64)

    def sample_neumann_boundary_points(self) -> dict[str, torch.Tensor]:
        """采样外边界无流点；normal 使用归一化坐标方向。"""

        n = int(self.cfg["n_neumann"])
        d = self.geometry.domain
        counts = self._allocate_counts(n, [d.height, d.height, d.width, d.width])
        chunks: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        if counts[0] > 0:
            x = np.full((counts[0], 1), d.x_min)
            y = self.rng.uniform(d.y_min, d.y_max, size=(counts[0], 1))
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[-1.0, 0.0]], (counts[0], 1)))
        if counts[1] > 0:
            x = np.full((counts[1], 1), d.x_max)
            y = self._sample_right_neumann_y(counts[1])
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[1.0, 0.0]], (counts[1], 1)))
        if counts[2] > 0:
            x = self.rng.uniform(d.x_min, d.x_max, size=(counts[2], 1))
            y = np.full((counts[2], 1), d.y_min)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, -1.0]], (counts[2], 1)))
        if counts[3] > 0:
            x = self.rng.uniform(d.x_min, d.x_max, size=(counts[3], 1))
            y = np.full((counts[3], 1), d.y_max)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, 1.0]], (counts[3], 1)))
        xy = np.vstack(chunks).astype(np.float64)
        normal = np.vstack(normals).astype(np.float64)
        return {"xyt": self._make_xyt(xy), "normal": self._to_tensor(normal)}

    def _sample_segments(
        self,
        n: int,
        segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    ) -> dict[str, torch.Tensor]:
        """按线段长度采样界面点和法向。

        这里采到的是几何界面中心点；真正用于 loss 的两侧材料点会在 physics 中沿法向
        做小偏移并过滤，避免裂缝交叉或外边界误入训练。
        """

        if n <= 0:
            empty_xyt = self._to_tensor(np.empty((0, 3), dtype=np.float64))
            empty_normal = self._to_tensor(np.empty((0, 2), dtype=np.float64))
            return {"xyt": empty_xyt, "normal": empty_normal}
        lengths = [float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) for p0, p1, _normal in segments]
        counts = self._allocate_counts(n, lengths)
        xy_chunks: list[np.ndarray] = []
        normal_chunks: list[np.ndarray] = []
        for (p0, p1, normal), count in zip(segments, counts):
            if count <= 0:
                continue
            tau = self.rng.uniform(0.0, 1.0, size=(count, 1))
            x = p0[0] + tau * (p1[0] - p0[0])
            y = p0[1] + tau * (p1[1] - p0[1])
            xy_chunks.append(np.hstack([x, y]).astype(np.float64))
            normal_chunks.append(np.tile(np.asarray(normal, dtype=np.float64).reshape(1, 2), (count, 1)))
        xy = np.vstack(xy_chunks)
        normal = np.vstack(normal_chunks)
        in_domain = self.geometry.inside_domain_np(xy[:, 0], xy[:, 1])
        xy = xy[in_domain]
        normal = normal[in_domain]
        if xy.shape[0] > n:
            xy = xy[:n]
            normal = normal[:n]
        return {"xyt": self._make_xyt(xy), "normal": self._to_tensor(normal)}

    def sample_hf_srv_interface_points(self) -> dict[str, torch.Tensor]:
        """采样 HF-SRV 界面候选点。"""

        return self._sample_segments(int(self.cfg["n_interface_hf_srv"]), self.geometry.hf_srv_interface_segments())

    def sample_srv_usrv_interface_points(self) -> dict[str, torch.Tensor]:
        """采样 SRV-USRV 界面候选点。"""

        return self._sample_segments(int(self.cfg["n_interface_srv_usrv"]), self.geometry.srv_usrv_interface_segments())

    def sample_all(self) -> dict[str, Any]:
        """一次性采样全部训练点；默认训练会固定重复使用这套点。"""

        return {
            "pde": self.sample_pde_points(),
            "dirichlet": self.sample_dirichlet_boundary_points(),
            "neumann": self.sample_neumann_boundary_points(),
            "interface_hf_srv": self.sample_hf_srv_interface_points(),
            "interface_srv_usrv": self.sample_srv_usrv_interface_points(),
        }
