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
        aliases = {
            "lhs": "latin_hypercube",
            "latin-hypercube": "latin_hypercube",
        }
        raw_sampling_mode = str(self.cfg.get("sampling_mode", "random")).lower()
        self.sampling_mode = aliases.get(raw_sampling_mode, raw_sampling_mode)
        valid_modes = {"random", "uniform", "latin_hypercube"}
        if self.sampling_mode not in valid_modes:
            raise ValueError(
                "sampler.sampling_mode must be 'random', 'uniform', "
                f"or 'latin_hypercube', got {self.sampling_mode!r}."
            )
        raw_time_mode = str(self.cfg.get("time_sampling_mode", self.sampling_mode)).lower()
        self.time_sampling_mode = aliases.get(raw_time_mode, raw_time_mode)
        if self.time_sampling_mode not in valid_modes:
            raise ValueError(
                "sampler.time_sampling_mode must be 'random', 'uniform', "
                f"or 'latin_hypercube', got {self.time_sampling_mode!r}."
            )

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        return tensor_from_numpy(array, self.device, self.dtype)

    def _latin_hypercube_unit(self, n: int, dim: int) -> np.ndarray:
        """Generate Latin hypercube samples in [0, 1]^dim."""

        if n <= 0:
            return np.empty((0, dim), dtype=np.float64)
        if dim <= 0:
            raise ValueError(f"Latin hypercube dimension must be positive, got {dim}.")
        strata = np.arange(n, dtype=np.float64).reshape(n, 1)
        unit = (strata + self.rng.random((n, dim))) / float(n)
        for axis in range(dim):
            self.rng.shuffle(unit[:, axis])
        return unit.astype(np.float64)

    def _latin_hypercube_interval(self, low: float, high: float, n: int) -> np.ndarray:
        """Generate one-dimensional Latin hypercube samples in [low, high]."""

        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        unit = self._latin_hypercube_unit(n, dim=1)
        values = float(low) + unit * (float(high) - float(low))
        return values.astype(np.float64)

    def _latin_hypercube_rect(self, rect: Rect, n: int) -> np.ndarray:
        """Generate two-dimensional Latin hypercube samples inside a rectangle."""

        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        unit = self._latin_hypercube_unit(n, dim=2)
        x = rect.x_min + unit[:, 0:1] * rect.width
        y = rect.y_min + unit[:, 1:2] * rect.height
        return np.hstack([x, y]).astype(np.float64)

    def _sample_interval(self, low: float, high: float, n: int, mode: str | None = None) -> np.ndarray:
        """Sample a one-dimensional interval with the requested mode."""

        selected_mode = self.sampling_mode if mode is None else str(mode).lower()
        if selected_mode == "uniform":
            return self._uniform_interval(low, high, n)
        if selected_mode == "latin_hypercube":
            return self._latin_hypercube_interval(low, high, n)
        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        return self.rng.uniform(float(low), float(high), size=(n, 1)).astype(np.float64)

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
        """采样时间点，支持连续 log1p 采样和固定时间切片混合采样。"""

        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        t_min = float(self.cfg["t_min"])
        t_max = float(self.cfg["t_max"])
        if t_min < 0.0:
            raise ValueError(f"t_min 必须 >= 0，当前 {t_min:g}。")
        if t_max <= t_min:
            raise ValueError(f"t_max 必须大于 t_min，当前 {t_max:g} <= {t_min:g}。")
        strategy = str(self.cfg.get("time_strategy", "log1p_uniform")).lower()
        if strategy == "log1p_uniform":
            return self._sample_log1p_time(n, t_min, t_max)
        if strategy in {"hybrid_log1p_fixed", "log1p_fixed_slices"}:
            return self._sample_hybrid_time(n, t_min, t_max)
        raise ValueError(f"Unsupported time_strategy: {strategy!r}.")

    def _sample_log1p_time(self, n: int, t_min: float, t_max: float) -> np.ndarray:
        """在 log(1+t) 空间中连续采样时间点。"""

        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        log_min = float(np.log1p(t_min))
        log_max = float(np.log1p(t_max))
        log_time = self._sample_interval(log_min, log_max, n, mode=self.time_sampling_mode)
        t = np.expm1(log_time)
        t = np.clip(t, t_min, t_max).astype(np.float64)
        if not np.all(np.isfinite(t)):
            raise RuntimeError("时间采样出现 NaN 或 Inf。")
        return t

    def _time_fixed_slices(self) -> list[tuple[float, float]]:
        """读取固定时间切片，返回 (time_days, fraction)。"""

        default_slices = [
            {"time": 1.0, "fraction": 0.1},
            {"time": 100.0, "fraction": 0.1},
            {"time": 500.0, "fraction": 0.1},
            {"time": 1000.0, "fraction": 0.2},
        ]
        raw_slices = self.cfg.get("time_fixed_slices", default_slices)
        parsed: list[tuple[float, float]] = []
        for item in raw_slices:
            if isinstance(item, dict):
                time_value = item["time"] if "time" in item else item["time_days"]
                fraction = item["fraction"]
            else:
                time_value, fraction = item
            parsed.append((float(time_value), float(fraction)))
        return parsed

    def _sample_hybrid_time(self, n: int, t_min: float, t_max: float) -> np.ndarray:
        """连续 log1p 时间采样与固定时间切片混合。"""

        continuous_fraction = float(self.cfg.get("time_continuous_fraction", 0.5))
        fixed_slices = self._time_fixed_slices()
        fractions = [continuous_fraction, *[fraction for _time_value, fraction in fixed_slices]]
        if any(fraction < 0.0 for fraction in fractions):
            raise ValueError("time sampling fractions must be non-negative.")
        if sum(fractions) <= 0.0:
            raise ValueError("time sampling fractions must have a positive sum.")

        counts = self._allocate_counts(n, fractions)
        chunks: list[np.ndarray] = []
        if counts[0] > 0:
            chunks.append(self._sample_log1p_time(counts[0], t_min, t_max))
        for count, (time_value, _fraction) in zip(counts[1:], fixed_slices):
            if count <= 0:
                continue
            if time_value < t_min or time_value > t_max:
                raise ValueError(f"fixed time slice {time_value:g} is outside [{t_min:g}, {t_max:g}].")
            chunks.append(np.full((count, 1), time_value, dtype=np.float64))

        t = np.vstack(chunks).astype(np.float64) if chunks else np.empty((0, 1), dtype=np.float64)
        if self.time_sampling_mode != "uniform" and t.shape[0] > 1:
            self.rng.shuffle(t, axis=0)
        return t[:n]

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

    def _allocate_counts_with_minimum(self, n: int, weights: list[float], min_count: int) -> list[int]:
        """先满足每段最低点数，再按权重分配剩余点数。"""

        if n <= 0:
            return [0 for _ in weights]
        if not weights:
            return []
        min_count = max(0, int(min_count))
        if min_count <= 0:
            return self._allocate_counts(n, weights)

        segment_count = len(weights)
        min_total = min_count * segment_count
        if n >= min_total:
            counts = np.full(segment_count, min_count, dtype=int)
            counts += np.asarray(self._allocate_counts(n - min_total, weights), dtype=int)
            return counts.tolist()

        # 总点数不足以满足最低值时，退化为尽量均匀分配，仍保证总数等于 n。
        counts = np.full(segment_count, n // segment_count, dtype=int)
        remainder = n - int(np.sum(counts))
        if remainder > 0:
            w = np.asarray(weights, dtype=float)
            if np.sum(w) <= 0.0:
                w = np.ones_like(w)
            for idx in np.argsort(-w)[:remainder]:
                counts[idx] += 1
        return counts.tolist()

    def _sample_rect_xy(self, rect: Rect, n: int) -> np.ndarray:
        """在矩形内采样 x/y。"""

        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "uniform":
            return self._uniform_rect_xy(rect, n)
        if self.sampling_mode == "latin_hypercube":
            return self._latin_hypercube_rect(rect, n)
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
        """在背景矩形中采样指定区域。"""

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
        while total < n and attempts < 200:
            attempts += 1
            remaining = n - total
            batch_n = max(512, int(np.ceil(remaining * 1.5)))
            if self.sampling_mode == "latin_hypercube":
                xy = self._latin_hypercube_rect(rect, batch_n)
            else:
                xy = self._sample_rect_xy(rect, batch_n)
            region = self.geometry.region_id_np(xy[:, 0], xy[:, 1])
            keep = xy[region == target_region]
            if keep.size > 0:
                accepted.append(keep)
                total += keep.shape[0]
        if total < n:
            raise RuntimeError(f"区域采样失败，target_region={target_region}, 得到 {total}/{n}。")
        points = np.vstack(accepted)[:n].astype(np.float64)
        self.rng.shuffle(points, axis=0)
        return points

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
        y = self._sample_interval(float(seg["y0"]), float(seg["y1"]), n)
        t = self.sample_time(n)
        return {"xyt": self._to_tensor(np.hstack([x, y, t]).astype(np.float64))}

    def _sample_right_neumann_y(self, n: int) -> np.ndarray:
        """右边界采样时排除生产端线段及其邻域。"""

        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        seg = self.geometry.dirichlet_segment
        gap = 0.05
        low = float(seg["y0"]) - gap
        high = float(seg["y1"]) + gap
        domain = self.geometry.domain
        counts = self._allocate_counts(n, [max(low - domain.y_min, 0.0), max(domain.y_max - high, 0.0)])
        chunks: list[np.ndarray] = []
        if counts[0] > 0:
            chunks.append(self._sample_interval(domain.y_min, low, counts[0]))
        if counts[1] > 0:
            chunks.append(self._sample_interval(high, domain.y_max, counts[1]))
        if not chunks:
            raise RuntimeError("生产边界排除区覆盖了整个右边界。")
        values = np.vstack(chunks)[:n].astype(np.float64)
        self.rng.shuffle(values, axis=0)
        return values

    def sample_neumann_boundary_points(self) -> dict[str, torch.Tensor]:
        """采样外边界无流点；normal 使用归一化坐标方向。"""

        n = int(self.cfg["n_neumann"])
        d = self.geometry.domain
        counts = self._allocate_counts(n, [d.height, d.height, d.width, d.width])
        chunks: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        if counts[0] > 0:
            x = np.full((counts[0], 1), d.x_min)
            y = self._sample_interval(d.y_min, d.y_max, counts[0])
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[-1.0, 0.0]], (counts[0], 1)))
        if counts[1] > 0:
            x = np.full((counts[1], 1), d.x_max)
            y = self._sample_right_neumann_y(counts[1])
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[1.0, 0.0]], (counts[1], 1)))
        if counts[2] > 0:
            x = self._sample_interval(d.x_min, d.x_max, counts[2])
            y = np.full((counts[2], 1), d.y_min)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, -1.0]], (counts[2], 1)))
        if counts[3] > 0:
            x = self._sample_interval(d.x_min, d.x_max, counts[3])
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
        min_points_per_segment: int = 0,
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
        counts = self._allocate_counts_with_minimum(n, lengths, min_points_per_segment)
        xy_chunks: list[np.ndarray] = []
        normal_chunks: list[np.ndarray] = []
        for (p0, p1, normal), count in zip(segments, counts):
            if count <= 0:
                continue
            tau = self._sample_interval(0.0, 1.0, count)
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

        return self._sample_segments(
            int(self.cfg["n_interface_hf_srv"]),
            self.geometry.hf_srv_interface_segments(),
            min_points_per_segment=int(self.cfg.get("min_points_per_hf_srv_interface_segment", 0)),
        )

    def sample_srv_usrv_interface_points(self) -> dict[str, torch.Tensor]:
        """采样 SRV-USRV 界面候选点。"""

        return self._sample_segments(
            int(self.cfg["n_interface_srv_usrv"]),
            self.geometry.srv_usrv_interface_segments(),
            min_points_per_segment=int(self.cfg.get("min_points_per_srv_usrv_interface_segment", 0)),
        )

    def sample_all(self) -> dict[str, Any]:
        """一次性采样全部训练点；默认训练会固定重复使用这套点。"""

        return {
            "pde": self.sample_pde_points(),
            "dirichlet": self.sample_dirichlet_boundary_points(),
            "neumann": self.sample_neumann_boundary_points(),
            "interface_hf_srv": self.sample_hf_srv_interface_points(),
            "interface_srv_usrv": self.sample_srv_usrv_interface_points(),
        }
