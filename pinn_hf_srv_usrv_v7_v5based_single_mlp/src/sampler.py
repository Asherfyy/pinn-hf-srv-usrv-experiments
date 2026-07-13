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
        self.sampling_mode = str(self.cfg.get("sampling_mode", "random")).lower()
        if self.sampling_mode not in {"random", "uniform"}:
            raise ValueError(f"sampler.sampling_mode must be 'random' or 'uniform', got {self.sampling_mode!r}.")
        self.time_sampling_mode = str(self.cfg.get("time_sampling_mode", self.sampling_mode)).lower()
        if self.time_sampling_mode not in {"random", "uniform"}:
            raise ValueError(f"sampler.time_sampling_mode must be 'random' or 'uniform', got {self.time_sampling_mode!r}.")

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return tensor_from_numpy(array, self.device, self.dtype)

    @staticmethod
    def _uniform_interval(low: float, high: float, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        if n == 1:
            return np.asarray([[0.5 * (float(low) + float(high))]], dtype=np.float64)
        step = (float(high) - float(low)) / float(n)
        values = float(low) + (np.arange(n, dtype=np.float64).reshape(-1, 1) + 0.5) * step
        return values.astype(np.float64)

    @staticmethod
    def _take_evenly(points: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return points[:0]
        if points.shape[0] <= n:
            return points
        idx = np.linspace(0, points.shape[0] - 1, n, dtype=np.int64)
        return points[idx]

    def _uniform_rect_xy(self, rect: Rect, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        width = max(float(rect.width), 1.0e-12)
        height = max(float(rect.height), 1.0e-12)
        if width >= height:
            ny = max(1, int(np.ceil(np.sqrt(n * height / width))))
            nx = int(np.ceil(n / ny))
        else:
            nx = max(1, int(np.ceil(np.sqrt(n * width / height))))
            ny = int(np.ceil(n / nx))
        xs = self._uniform_interval(rect.x_min, rect.x_max, nx).reshape(-1)
        ys = self._uniform_interval(rect.y_min, rect.y_max, ny).reshape(-1)
        x_grid, y_grid = np.meshgrid(xs, ys)
        points = np.column_stack([x_grid.ravel(), y_grid.ravel()]).astype(np.float64)
        return self._take_evenly(points, n)

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
            raise ValueError("v7 只实现 log1p_uniform 时间采样。")
        if self.time_sampling_mode == "uniform":
            s = self._uniform_interval(np.log1p(t_min), np.log1p(t_max), n)
        else:
            s = self.rng.uniform(np.log1p(t_min), np.log1p(t_max), size=(n, 1))
        t = np.expm1(s)
        t = np.clip(t, t_min, t_max).astype(np.float64)
        if not np.all(np.isfinite(t)):
            raise RuntimeError("时间采样出现 NaN 或 Inf。")
        if self.time_sampling_mode == "random":
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
        if self.sampling_mode == "uniform":
            return self._uniform_rect_xy(rect, n)
        x = self.rng.uniform(rect.x_min, rect.x_max, size=(n, 1))
        y = self.rng.uniform(rect.y_min, rect.y_max, size=(n, 1))
        return np.hstack([x, y]).astype(np.float64)

    def _sample_hf_xy(self, n: int) -> np.ndarray:
        """在所有 HF 裂缝矩形内采样。"""

        counts = self._allocate_counts(n, [rect.area for rect in self.geometry.hf_rects])
        chunks = [self._sample_rect_xy(rect, count) for rect, count in zip(self.geometry.hf_rects, counts) if count > 0]
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "random":
            self.rng.shuffle(xy, axis=0)
        return xy[:n]

    def _sample_region_xy(self, n: int, rect: Rect, target_region: int) -> np.ndarray:
        """在背景矩形内拒绝采样指定区域。"""

        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "uniform":
            batch_n = max(128, n * 2)
            keep = np.empty((0, 2), dtype=np.float64)
            for _attempt in range(30):
                xy = self._uniform_rect_xy(rect, batch_n)
                region = self.geometry.region_id_np(xy[:, 0], xy[:, 1])
                keep = xy[region == target_region]
                if keep.shape[0] >= n:
                    return self._take_evenly(keep, n).astype(np.float64)
                batch_n *= 2
            raise RuntimeError(f"Uniform region sampling failed, target_region={target_region}, got {keep.shape[0]}/{n}.")
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
        if self.sampling_mode == "random":
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
        if self.sampling_mode == "uniform":
            y = self._uniform_interval(float(seg["y0"]), float(seg["y1"]), n)
        else:
            y = self.rng.uniform(float(seg["y0"]), float(seg["y1"]), size=(n, 1)).astype(np.float64)
        t = self.sample_time(n)
        return {"xyt": self._to_tensor(np.hstack([x, y, t]).astype(np.float64))}

    def sample_hf_secondary_link_points(self) -> dict[str, torch.Tensor]:
        """Sample secondary-fracture centerline points tied to their main-fracture junction."""

        n = int(self.cfg.get("n_hf_secondary_link", 0))
        empty = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        if n <= 0 or not self.geometry.secondary_fractures:
            return {"xyt": empty, "junction_xyt": empty}

        main = self.geometry.main_frac
        y_junction = 0.5 * (main.y_min + main.y_max)
        counts = self._allocate_counts(n, [rect.height for rect in self.geometry.secondary_fractures])
        xyt_chunks: list[np.ndarray] = []
        junction_chunks: list[np.ndarray] = []
        for rect, count in zip(self.geometry.secondary_fractures, counts):
            if count <= 0:
                continue
            x_center = 0.5 * (rect.x_min + rect.x_max)
            x = np.full((count, 1), x_center, dtype=np.float64)
            lower_min = rect.y_min
            lower_max = min(main.y_min, rect.y_max)
            upper_min = max(main.y_max, rect.y_min)
            upper_max = rect.y_max
            y_parts: list[np.ndarray] = []
            segment_counts = self._allocate_counts(count, [max(lower_max - lower_min, 0.0), max(upper_max - upper_min, 0.0)])
            if segment_counts[0] > 0:
                y_parts.append(self._uniform_interval(lower_min, lower_max, segment_counts[0]))
            if segment_counts[1] > 0:
                y_parts.append(self._uniform_interval(upper_min, upper_max, segment_counts[1]))
            y = np.vstack(y_parts) if y_parts else self._uniform_interval(rect.y_min, rect.y_max, count)
            t = self.sample_time(count)
            xyt_chunks.append(np.hstack([x, y, t]).astype(np.float64))
            y_ref = np.full((count, 1), y_junction, dtype=np.float64)
            junction_chunks.append(np.hstack([x, y_ref, t]).astype(np.float64))

        xyt = np.vstack(xyt_chunks) if xyt_chunks else np.empty((0, 3), dtype=np.float64)
        junction_xyt = np.vstack(junction_chunks) if junction_chunks else np.empty((0, 3), dtype=np.float64)
        order = self.rng.permutation(xyt.shape[0]) if self.sampling_mode == "random" else np.arange(xyt.shape[0])
        return {"xyt": self._to_tensor(xyt[order]), "junction_xyt": self._to_tensor(junction_xyt[order])}

    def sample_hf_junction_pair_points(self) -> dict[str, torch.Tensor]:
        """Sample paired points around main-secondary intersections for strong pressure coupling."""

        n = int(self.cfg.get("n_hf_junction", 0))
        empty = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        if n <= 0 or not self.geometry.secondary_fractures:
            return {"main_xyt": empty, "secondary_xyt": empty}

        main = self.geometry.main_frac
        y_center = 0.5 * (main.y_min + main.y_max)
        half_main_thickness = 0.5 * main.height
        offset = float(self.cfg.get("junction_offset_m", 1.0e-3))
        counts = self._allocate_counts(n, [rect.height for rect in self.geometry.secondary_fractures])
        main_chunks: list[np.ndarray] = []
        secondary_chunks: list[np.ndarray] = []
        for rect, count in zip(self.geometry.secondary_fractures, counts):
            if count <= 0:
                continue
            x_center = 0.5 * (rect.x_min + rect.x_max)
            half_secondary_width = 0.5 * rect.width
            t = self.sample_time(count)

            if self.sampling_mode == "uniform":
                idx = np.arange(count, dtype=np.int64).reshape(-1, 1)
                x_sign = np.where(idx % 2 == 0, -1.0, 1.0).astype(np.float64)
                y_sign = np.where((idx // 2) % 2 == 0, -1.0, 1.0).astype(np.float64)
            else:
                x_sign = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(count, 1))
                y_sign = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(count, 1))
            main_x = x_center + x_sign * (half_secondary_width + offset)
            main_y = np.full((count, 1), y_center, dtype=np.float64)
            secondary_x = np.full((count, 1), x_center, dtype=np.float64)
            secondary_y = y_center + y_sign * (half_main_thickness + offset)

            main_chunks.append(np.hstack([main_x, main_y, t]).astype(np.float64))
            secondary_chunks.append(np.hstack([secondary_x, secondary_y, t]).astype(np.float64))

        main_xyt = np.vstack(main_chunks) if main_chunks else np.empty((0, 3), dtype=np.float64)
        secondary_xyt = np.vstack(secondary_chunks) if secondary_chunks else np.empty((0, 3), dtype=np.float64)
        order = self.rng.permutation(main_xyt.shape[0]) if self.sampling_mode == "random" else np.arange(main_xyt.shape[0])
        return {"main_xyt": self._to_tensor(main_xyt[order]), "secondary_xyt": self._to_tensor(secondary_xyt[order])}

    def _sample_right_neumann_y(self, n: int) -> np.ndarray:
        """右边界采样时排除生产边界小线段及邻域。"""

        seg = self.geometry.dirichlet_segment
        gap = 0.05
        low = float(seg["y0"]) - gap
        high = float(seg["y1"]) + gap
        if self.sampling_mode == "uniform":
            d = self.geometry.domain
            counts = self._allocate_counts(n, [max(low - d.y_min, 0.0), max(d.y_max - high, 0.0)])
            chunks: list[np.ndarray] = []
            if counts[0] > 0:
                chunks.append(self._uniform_interval(d.y_min, low, counts[0]))
            if counts[1] > 0:
                chunks.append(self._uniform_interval(high, d.y_max, counts[1]))
            return np.vstack(chunks)[:n].astype(np.float64)
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
            if self.sampling_mode == "uniform":
                y = self._uniform_interval(d.y_min, d.y_max, counts[0])
            else:
                y = self.rng.uniform(d.y_min, d.y_max, size=(counts[0], 1))
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[-1.0, 0.0]], (counts[0], 1)))
        if counts[1] > 0:
            x = np.full((counts[1], 1), d.x_max)
            y = self._sample_right_neumann_y(counts[1])
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[1.0, 0.0]], (counts[1], 1)))
        if counts[2] > 0:
            if self.sampling_mode == "uniform":
                x = self._uniform_interval(d.x_min, d.x_max, counts[2])
            else:
                x = self.rng.uniform(d.x_min, d.x_max, size=(counts[2], 1))
            y = np.full((counts[2], 1), d.y_min)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, -1.0]], (counts[2], 1)))
        if counts[3] > 0:
            if self.sampling_mode == "uniform":
                x = self._uniform_interval(d.x_min, d.x_max, counts[3])
            else:
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
            if self.sampling_mode == "uniform":
                tau = self._uniform_interval(0.0, 1.0, count)
            else:
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

    def sample_hf_main_link_points(self) -> dict[str, torch.Tensor]:
        """Sample centerline points on the high-conductivity main fracture."""

        n = int(self.cfg.get("n_hf_main_link", 0))
        if n <= 0:
            return {"xyt": self._to_tensor(np.empty((0, 3), dtype=np.float64))}
        rect = self.geometry.main_frac
        x = self._uniform_interval(rect.x_min, rect.x_max, n)
        y_center = 0.5 * (rect.y_min + rect.y_max)
        y = np.full((n, 1), y_center, dtype=np.float64)
        t = self.sample_time(n)
        return {"xyt": self._to_tensor(np.hstack([x, y, t]).astype(np.float64))}

    def sample_all(self) -> dict[str, Any]:
        """一次性采样全部训练点；默认训练会固定重复使用这套点。"""

        return {
            "pde": self.sample_pde_points(),
            "dirichlet": self.sample_dirichlet_boundary_points(),
            "neumann": self.sample_neumann_boundary_points(),
            "interface_hf_srv": self.sample_hf_srv_interface_points(),
            "interface_srv_usrv": self.sample_srv_usrv_interface_points(),
            "hf_main_link": self.sample_hf_main_link_points(),
            "hf_secondary_link": self.sample_hf_secondary_link_points(),
            "hf_junction": self.sample_hf_junction_pair_points(),
        }
