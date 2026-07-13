"""PINN 训练点分区采样器。

裂缝宽度只有 0.01 m，如果在总储层矩形中普通随机采样，几乎采不到 HF 点。
因此本模块按 HF/SRV/USRV 分区分别采样：HF 直接从裂缝矩形采样，SRV/USRV
只在相应背景区域中做轻量拒绝采样。这样训练 batch 中始终包含足够的裂缝信息。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, Rect, ReservoirGeometry
from .utils import tensor_from_numpy


class ReservoirSampler:
    """PINN 训练点采样器。

    负责生成：
    1. HF/SRV/USRV 内部 PDE 点；
    2. 初始条件点；
    3. Dirichlet 边界点；
    4. Neumann 无流边界点；
    5. HF-SRV 界面点；
    6. SRV-USRV 界面点。
    """

    def __init__(
        self,
        geometry: ReservoirGeometry,
        sampler_cfg: dict[str, Any],
        device: torch.device,
        dtype: torch.dtype,
        seed: int = 2026,
    ) -> None:
        self.geometry = geometry
        self.cfg = sampler_cfg
        self.device = device
        self.dtype = dtype
        self.rng = np.random.default_rng(seed)

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """统一把采样结果转为 CPU 张量。"""

        return tensor_from_numpy(array, self.device, self.dtype)

    def sample_time(self, n: int) -> np.ndarray:
        """采样时间 t。

        默认策略为 `log1p_uniform`：在 log1p(t) 空间均匀采样，再用 expm1 还原到
        物理时间。这样既能覆盖 t=0 附近，也会在 1~100 d 的主要衰减阶段给出更密的点。
        初始条件采样不会调用该函数，而是显式固定 t=0。
        """

        if n <= 0:
            return np.empty((0, 1), dtype=np.float32)
        t_min = float(self.cfg["t_min"])
        t_max = float(self.cfg["t_max"])
        if t_min < 0.0:
            raise ValueError(f"sampler.t_min 必须 >= 0，当前为 {t_min:g}。")
        if t_max <= t_min:
            raise ValueError(f"sampler.t_max 必须大于 t_min，当前为 {t_max:g} <= {t_min:g}。")

        strategy = str(self.cfg.get("time_strategy", "early_late_mixed")).lower()
        if strategy == "log1p_uniform":
            s = self.rng.uniform(np.log1p(t_min), np.log1p(t_max), size=(n, 1))
            t = np.expm1(s)
            # 浮点反变换后做轻微裁剪，保证严格落在配置区间内。
            t = np.clip(t, t_min, t_max)
        elif strategy == "early_late_mixed":
            t_early = float(self.cfg["t_early"])
            if not (t_min <= t_early <= t_max):
                raise ValueError(
                    f"sampler.t_early 必须位于 [t_min,t_max]，当前 t_early={t_early:g}, "
                    f"区间=[{t_min:g},{t_max:g}]。"
                )
            n_early = n // 2
            n_late = n - n_early
            early = self.rng.uniform(t_min, t_early, size=(n_early, 1))
            late = self.rng.uniform(t_early, t_max, size=(n_late, 1))
            t = np.vstack([early, late])
        elif strategy == "uniform":
            t = self.rng.uniform(t_min, t_max, size=(n, 1))
        else:
            raise ValueError(
                f"不支持的 time_strategy: {strategy}。"
                "请使用 'log1p_uniform'、'early_late_mixed' 或 'uniform'。"
            )
        t = t.astype(np.float32, copy=False)
        self.rng.shuffle(t, axis=0)
        return t

    def _allocate_counts_by_weights(self, n: int, weights: list[float]) -> list[int]:
        """按长度或面积权重分配采样数量，并保证总数严格等于 n。"""

        if n <= 0:
            return [0 for _ in weights]
        w = np.asarray(weights, dtype=float)
        if np.all(w <= 0.0):
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
        """在给定矩形中均匀采样 x,y。"""

        x = self.rng.uniform(rect.x_min, rect.x_max, size=(n, 1))
        y = self.rng.uniform(rect.y_min, rect.y_max, size=(n, 1))
        return np.hstack([x, y]).astype(np.float32)

    def _distance_to_dirichlet_segment_np(self, xy: np.ndarray) -> np.ndarray:
        """计算 NumPy 点云到默认 Dirichlet 线段的物理距离。

        该函数只用于采样器的 PDE 内点过滤。网络输入距离特征和 hard constraint 仍由
        `ReservoirGeometry.distance_to_dirichlet_torch()` 统一计算，避免训练图中混入 NumPy。
        """

        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy 期望形状为 [N,2]，当前为 {xy.shape}。")
        seg = self.geometry.dirichlet_segment
        ax = float(seg["x0"])
        ay = float(seg["y0"])
        bx = float(seg["x1"])
        by = float(seg["y1"])
        vx = bx - ax
        vy = by - ay
        denom = vx * vx + vy * vy
        if denom <= 0.0:
            return np.hypot(xy[:, 0] - ax, xy[:, 1] - ay)
        tau = ((xy[:, 0] - ax) * vx + (xy[:, 1] - ay) * vy) / denom
        tau = np.clip(tau, 0.0, 1.0)
        closest_x = ax + tau * vx
        closest_y = ay + tau * vy
        return np.hypot(xy[:, 0] - closest_x, xy[:, 1] - closest_y)

    def _sample_rect_xy_with_dirichlet_exclusion(self, rect: Rect, n: int, exclusion_m: float) -> np.ndarray:
        """在矩形中采样，并可排除 Dirichlet 边界附近 PDE 内点。

        新的 tanh ADF 在生产端几米内有较强二阶导。该过渡层由 hard constraint 和
        Dirichlet 边界点负责，不适合作为 PDE 内点强行求二阶导，因此 PDE 采样可配置
        一个很小的物理排除距离。初始条件采样不会使用该过滤。
        """

        if n <= 0:
            return np.empty((0, 2), dtype=np.float32)
        if exclusion_m <= 0.0:
            return self._sample_rect_xy(rect, n)

        accepted: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < n and attempts < 200:
            attempts += 1
            batch_n = max(64, (n - total) * 3)
            xy = self._sample_rect_xy(rect, batch_n)
            dist = self._distance_to_dirichlet_segment_np(xy)
            keep = xy[dist >= float(exclusion_m)]
            if keep.size > 0:
                accepted.append(keep)
                total += keep.shape[0]
        if total < n:
            raise RuntimeError(
                f"HF PDE 采样失败：排除 Dirichlet {exclusion_m:g} m 后只得到 {total}/{n} 个点。"
            )
        return np.vstack(accepted)[:n].astype(np.float32)

    def _sample_hf_xy(self, n: int, pde_dirichlet_exclusion_m: float = 0.0) -> np.ndarray:
        """直接在主裂缝和五条次级裂缝矩形中采样 HF 点。

        `pde_dirichlet_exclusion_m` 只供 PDE 内点使用，用于避开 hard constraint ADF
        过渡层；初始条件和其他采样仍传 0，保持全几何覆盖。
        """

        exclusion = float(pde_dirichlet_exclusion_m)
        if exclusion < 0.0:
            raise ValueError(f"pde_dirichlet_exclusion_m 必须 >= 0，当前为 {exclusion:g}。")
        counts = self._allocate_counts_by_weights(n, [rect.area for rect in self.geometry.hf_rects])
        chunks = [
            self._sample_rect_xy_with_dirichlet_exclusion(rect, count, exclusion)
            for rect, count in zip(self.geometry.hf_rects, counts)
            if count > 0
        ]
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float32)
        self.rng.shuffle(xy, axis=0)
        return xy[:n]

    def _sample_region_xy(self, n: int, rect: Rect, target_region: int) -> np.ndarray:
        """在指定背景矩形内拒绝采样目标区域。

        这里的拒绝采样只发生在已经很接近目标区域的背景矩形内，所以效率远高于从总域
        盲采。例如 SRV 只从 SRV 背景区采样后排除 HF；USRV 只需从总域排除 SRV 背景。
        """

        if n <= 0:
            return np.empty((0, 2), dtype=np.float32)
        accepted: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < n and attempts < 200:
            attempts += 1
            batch_n = max(64, (n - total) * 3)
            xy = self._sample_rect_xy(rect, batch_n)
            region = self.geometry.region_id_np(xy[:, 0], xy[:, 1])
            keep = xy[region == target_region]
            if keep.size > 0:
                accepted.append(keep)
                total += keep.shape[0]
        if total < n:
            raise RuntimeError(f"区域采样失败，target_region={target_region}, 只得到 {total}/{n} 个点。")
        return np.vstack(accepted)[:n].astype(np.float32)

    def _make_xyt(self, xy: np.ndarray, t: np.ndarray) -> torch.Tensor:
        """拼接 x,y,t 并转换为张量。"""

        return self._to_tensor(np.hstack([xy, t]).astype(np.float32))

    def sample_pde_points(self) -> dict[str, torch.Tensor]:
        """采样 HF/SRV/USRV 内部 PDE 点。"""

        n_hf = int(self.cfg["n_pde_hf"])
        n_srv = int(self.cfg["n_pde_srv"])
        n_usrv = int(self.cfg["n_pde_usrv"])

        hf_xy = self._sample_hf_xy(n_hf, float(self.cfg.get("pde_dirichlet_exclusion_m", 0.0)))
        srv_xy = self._sample_region_xy(n_srv, self.geometry.srv_bg, REGION_SRV)
        usrv_xy = self._sample_region_xy(n_usrv, self.geometry.domain, REGION_USRV)

        return {
            "hf": self._make_xyt(hf_xy, self.sample_time(n_hf)),
            "srv": self._make_xyt(srv_xy, self.sample_time(n_srv)),
            "usrv": self._make_xyt(usrv_xy, self.sample_time(n_usrv)),
        }

    def sample_initial_points(self) -> dict[str, torch.Tensor]:
        """采样初始条件点，所有点 t=0。"""

        n_hf = int(self.cfg["n_initial_hf"])
        n_srv = int(self.cfg["n_initial_srv"])
        n_usrv = int(self.cfg["n_initial_usrv"])

        hf_xy = self._sample_hf_xy(n_hf)
        srv_xy = self._sample_region_xy(n_srv, self.geometry.srv_bg, REGION_SRV)
        usrv_xy = self._sample_region_xy(n_usrv, self.geometry.domain, REGION_USRV)

        return {
            "hf": self._make_xyt(hf_xy, np.zeros((n_hf, 1), dtype=np.float32)),
            "srv": self._make_xyt(srv_xy, np.zeros((n_srv, 1), dtype=np.float32)),
            "usrv": self._make_xyt(usrv_xy, np.zeros((n_usrv, 1), dtype=np.float32)),
        }

    def sample_dirichlet_boundary_points(self) -> dict[str, torch.Tensor]:
        """采样默认 Dirichlet 线段 x=360, y 属于主裂缝右端。

        在 dirichlet_hard 模式下，这些点用于诊断硬约束误差；在 direct/ic_hard 模式下，
        这些点直接参与 Dirichlet soft loss，用来约束生产端 Pb1/Pb2 压力。
        """

        n = int(self.cfg["n_dirichlet"])
        seg = self.geometry.dirichlet_segment
        y0 = float(seg["y0"])
        y1 = float(seg["y1"])
        x = np.full((n, 1), float(seg["x0"]), dtype=np.float32)
        y = self.rng.uniform(y0, y1, size=(n, 1)).astype(np.float32)
        t = self.sample_time(n)
        xyt = self._to_tensor(np.hstack([x, y, t]).astype(np.float32))
        normal = self._to_tensor(np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (n, 1)))
        return {"xyt": xyt, "normal": normal}

    def _sample_right_neumann_y(self, n: int) -> np.ndarray:
        """在右边界采样 y，并排除 Dirichlet 小线段附近。"""

        y_min = self.geometry.domain.y_min
        y_max = self.geometry.domain.y_max
        seg = self.geometry.dirichlet_segment
        gap = 0.05
        low = float(seg["y0"]) - gap
        high = float(seg["y1"]) + gap
        values: list[np.ndarray] = []
        total = 0
        while total < n:
            y = self.rng.uniform(y_min, y_max, size=(max(64, 2 * (n - total)), 1))
            keep = y[(y[:, 0] < low) | (y[:, 0] > high)]
            if keep.size:
                values.append(keep.reshape(-1, 1))
                total += keep.size
        return np.vstack(values)[:n].astype(np.float32)

    def sample_neumann_boundary_points(self) -> dict[str, torch.Tensor]:
        """采样外边界无流 Neumann 点及对应外法向。"""

        n = int(self.cfg["n_neumann"])
        counts = self._allocate_counts_by_weights(
            n,
            [
                self.geometry.domain.height,
                self.geometry.domain.height,
                self.geometry.domain.width,
                self.geometry.domain.width,
            ],
        )
        chunks: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        d = self.geometry.domain

        # 左边界 x=0
        c = counts[0]
        if c:
            x = np.full((c, 1), d.x_min)
            y = self.rng.uniform(d.y_min, d.y_max, size=(c, 1))
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[-1.0, 0.0]], (c, 1)))

        # 右边界 x=360，排除 Dirichlet 线段邻域
        c = counts[1]
        if c:
            x = np.full((c, 1), d.x_max)
            y = self._sample_right_neumann_y(c)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[1.0, 0.0]], (c, 1)))

        # 下边界 y=0
        c = counts[2]
        if c:
            x = self.rng.uniform(d.x_min, d.x_max, size=(c, 1))
            y = np.full((c, 1), d.y_min)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, -1.0]], (c, 1)))

        # 上边界 y=y_max
        c = counts[3]
        if c:
            x = self.rng.uniform(d.x_min, d.x_max, size=(c, 1))
            y = np.full((c, 1), d.y_max)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, 1.0]], (c, 1)))

        xy = np.vstack(chunks).astype(np.float32)
        normal = np.vstack(normals).astype(np.float32)
        t = self.sample_time(xy.shape[0])
        return {"xyt": self._make_xyt(xy, t), "normal": self._to_tensor(normal)}

    def _sample_segments(
        self,
        n: int,
        segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    ) -> dict[str, torch.Tensor]:
        """按线段长度采样界面点及法向。"""

        lengths = [
            float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
            for p0, p1, _normal in segments
        ]
        counts = self._allocate_counts_by_weights(n, lengths)
        xy_chunks: list[np.ndarray] = []
        normal_chunks: list[np.ndarray] = []
        for (p0, p1, normal), count in zip(segments, counts):
            if count <= 0:
                continue
            tau = self.rng.uniform(0.0, 1.0, size=(count, 1))
            x = p0[0] + tau * (p1[0] - p0[0])
            y = p0[1] + tau * (p1[1] - p0[1])
            xy_chunks.append(np.hstack([x, y]).astype(np.float32))
            normal_chunks.append(np.tile(np.asarray(normal, dtype=np.float32).reshape(1, 2), (count, 1)))

        xy = np.vstack(xy_chunks)
        normal = np.vstack(normal_chunks)
        in_domain = self.geometry.inside_domain_np(xy[:, 0], xy[:, 1])
        xy = xy[in_domain]
        normal = normal[in_domain]
        if xy.shape[0] > n:
            xy = xy[:n]
            normal = normal[:n]
        t = self.sample_time(xy.shape[0])
        return {"xyt": self._make_xyt(xy, t), "normal": self._to_tensor(normal.astype(np.float32))}

    def sample_hf_srv_interface_points(self) -> dict[str, torch.Tensor]:
        """采样 HF-SRV/HF-USRV 裂缝边界界面点。"""

        return self._sample_segments(
            int(self.cfg["n_interface_hf_srv"]),
            self.geometry.hf_srv_interface_segments(),
        )

    def sample_srv_usrv_interface_points(self) -> dict[str, torch.Tensor]:
        """采样 SRV-USRV 内部界面点。"""

        return self._sample_segments(
            int(self.cfg["n_interface_srv_usrv"]),
            self.geometry.srv_usrv_interface_segments(),
        )

    def sample_all(self) -> dict[str, Any]:
        """一次性生成训练所需的所有点集。"""

        return {
            "pde": self.sample_pde_points(),
            "initial": self.sample_initial_points(),
            "dirichlet": self.sample_dirichlet_boundary_points(),
            "neumann": self.sample_neumann_boundary_points(),
            "interface_hf_srv": self.sample_hf_srv_interface_points(),
            "interface_srv_usrv": self.sample_srv_usrv_interface_points(),
        }
