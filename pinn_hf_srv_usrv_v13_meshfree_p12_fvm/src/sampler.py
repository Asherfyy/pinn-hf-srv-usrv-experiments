"""Random mesh-free collocation sampler for v13 line-fracture PINN training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .geometry import REGION_SRV, REGION_USRV, FractureLine, Rect, ReservoirGeometry
from .utils import tensor_from_numpy


class ReservoirSampler:
    """Sample PDE, boundary, interface, and fracture-link collocation points."""

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
        self.time_pairing_mode = str(self.cfg.get("time_pairing_mode", "paired")).lower()
        if self.time_pairing_mode not in {"paired", "cartesian"}:
            raise ValueError(f"sampler.time_pairing_mode must be 'paired' or 'cartesian', got {self.time_pairing_mode!r}.")

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

    def _allocate_counts(self, n: int, weights: list[float]) -> list[int]:
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

    def _sample_rect_xy(self, rect: Rect, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "uniform":
            return self._uniform_rect_xy(rect, n)
        x = self.rng.uniform(rect.x_min, rect.x_max, size=(n, 1))
        y = self.rng.uniform(rect.y_min, rect.y_max, size=(n, 1))
        return np.hstack([x, y]).astype(np.float64)

    def _sample_line_xy(self, line: FractureLine, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "uniform":
            tau = self._uniform_interval(0.0, 1.0, n)
        else:
            tau = self.rng.uniform(0.0, 1.0, size=(n, 1))
        x = line.x0 + tau * (line.x1 - line.x0)
        y = line.y0 + tau * (line.y1 - line.y0)
        return np.hstack([x, y]).astype(np.float64)

    def _sample_line_xy_interior(self, line: FractureLine, n: int, endpoint_margin_m: float) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        length = max(float(line.length), 1.0e-12)
        margin = min(max(float(endpoint_margin_m), 0.0) / length, 0.49)
        if self.sampling_mode == "uniform":
            tau = self._uniform_interval(margin, 1.0 - margin, n)
        else:
            tau = self.rng.uniform(margin, 1.0 - margin, size=(n, 1))
        x = line.x0 + tau * (line.x1 - line.x0)
        y = line.y0 + tau * (line.y1 - line.y0)
        return np.hstack([x, y]).astype(np.float64)

    def _sample_hf_xy(self, n: int) -> np.ndarray:
        """Sample HF collocation points on fracture centerlines only."""

        counts = self._allocate_counts(n, [line.length for line in self.geometry.hf_lines])
        chunks = [self._sample_line_xy(line, count) for line, count in zip(self.geometry.hf_lines, counts) if count > 0]
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "random" and xy.shape[0] > 0:
            self.rng.shuffle(xy, axis=0)
        return xy[:n]

    def _sample_hf_xy_with_tangent_raw(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        counts = self._allocate_counts(n, [line.length for line in self.geometry.hf_lines])
        xy_chunks: list[np.ndarray] = []
        tangent_chunks: list[np.ndarray] = []
        for line, count in zip(self.geometry.hf_lines, counts):
            if count <= 0:
                continue
            xy = self._sample_line_xy(line, count)
            dx = float(line.x1 - line.x0)
            dy = float(line.y1 - line.y0)
            length = max(float(np.hypot(dx, dy)), 1.0e-12)
            tangent = np.tile(np.asarray([[dx / length, dy / length]], dtype=np.float64), (xy.shape[0], 1))
            xy_chunks.append(xy)
            tangent_chunks.append(tangent)
        xy_all = np.vstack(xy_chunks) if xy_chunks else np.empty((0, 2), dtype=np.float64)
        tangent_all = np.vstack(tangent_chunks) if tangent_chunks else np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "random" and xy_all.shape[0] > 0:
            order = self.rng.permutation(xy_all.shape[0])
            xy_all = xy_all[order]
            tangent_all = tangent_all[order]
        return xy_all[:n].astype(np.float64), tangent_all[:n].astype(np.float64)

    def _sample_hf_xy_with_tangent_and_aperture_raw(
        self,
        n: int,
        endpoint_margin_m: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n <= 0:
            return (
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 2), dtype=np.float64),
                np.empty((0, 1), dtype=np.float64),
            )
        counts = self._allocate_counts(n, [line.length for line in self.geometry.hf_lines])
        xy_chunks: list[np.ndarray] = []
        tangent_chunks: list[np.ndarray] = []
        aperture_chunks: list[np.ndarray] = []
        for line, count in zip(self.geometry.hf_lines, counts):
            if count <= 0:
                continue
            xy = self._sample_line_xy_interior(line, count, endpoint_margin_m)
            dx = float(line.x1 - line.x0)
            dy = float(line.y1 - line.y0)
            length = max(float(np.hypot(dx, dy)), 1.0e-12)
            tangent = np.tile(np.asarray([[dx / length, dy / length]], dtype=np.float64), (xy.shape[0], 1))
            aperture = np.full((xy.shape[0], 1), max(float(line.aperture), 1.0e-12), dtype=np.float64)
            xy_chunks.append(xy)
            tangent_chunks.append(tangent)
            aperture_chunks.append(aperture)
        xy_all = np.vstack(xy_chunks) if xy_chunks else np.empty((0, 2), dtype=np.float64)
        tangent_all = np.vstack(tangent_chunks) if tangent_chunks else np.empty((0, 2), dtype=np.float64)
        aperture_all = np.vstack(aperture_chunks) if aperture_chunks else np.empty((0, 1), dtype=np.float64)
        if self.sampling_mode == "random" and xy_all.shape[0] > 0:
            order = self.rng.permutation(xy_all.shape[0])
            xy_all = xy_all[order]
            tangent_all = tangent_all[order]
            aperture_all = aperture_all[order]
        return xy_all[:n].astype(np.float64), tangent_all[:n].astype(np.float64), aperture_all[:n].astype(np.float64)

    def _sample_hf_segment_conservation_raw(
        self,
        n: int,
    ) -> dict[str, np.ndarray]:
        empty = {
            "left_xy": np.empty((0, 2), dtype=np.float64),
            "right_xy": np.empty((0, 2), dtype=np.float64),
            "gauss_xy": np.empty((0, 2, 2), dtype=np.float64),
            "tangent": np.empty((0, 2), dtype=np.float64),
            "aperture": np.empty((0, 1), dtype=np.float64),
            "half_length": np.empty((0, 1), dtype=np.float64),
        }
        if n <= 0:
            return empty
        target_n = int(n)
        candidate_n = max(target_n, int(np.ceil(1.5 * target_n)) + len(self.geometry.hf_lines))
        min_half = float(self.cfg.get("hf_segment_min_half_length_m", 0.5))
        max_half = float(self.cfg.get("hf_segment_max_half_length_m", 5.0))
        endpoint_margin = float(self.cfg.get("hf_segment_endpoint_margin_m", self.cfg.get("hf_leakoff_endpoint_margin_m", 0.05)))
        min_half = max(min_half, 1.0e-12)
        max_half = max(max_half, min_half)
        counts = self._allocate_counts(candidate_n, [line.length for line in self.geometry.hf_lines])
        left_chunks: list[np.ndarray] = []
        right_chunks: list[np.ndarray] = []
        gauss_chunks: list[np.ndarray] = []
        tangent_chunks: list[np.ndarray] = []
        aperture_chunks: list[np.ndarray] = []
        half_chunks: list[np.ndarray] = []
        gauss_unit = 1.0 / np.sqrt(3.0)
        gauss_offsets = np.asarray([-gauss_unit, gauss_unit], dtype=np.float64)
        for line, count in zip(self.geometry.hf_lines, counts):
            if count <= 0:
                continue
            dx = float(line.x1 - line.x0)
            dy = float(line.y1 - line.y0)
            length = max(float(np.hypot(dx, dy)), 1.0e-12)
            tangent = np.asarray([[dx / length, dy / length]], dtype=np.float64)
            margin = min(max(float(endpoint_margin), 0.0), 0.49 * length)
            available_half = max(0.5 * length - margin, min_half)
            high = min(max_half, available_half)
            low = min(min_half, high)
            if self.sampling_mode == "uniform":
                center_s = self._uniform_interval(margin + high, length - margin - high, count).reshape(-1, 1)
                half_length = np.full((count, 1), 0.5 * (low + high), dtype=np.float64)
            else:
                center_s = self.rng.uniform(margin + high, length - margin - high, size=(count, 1))
                if high > low:
                    log_low = np.log(low)
                    log_high = np.log(high)
                    half_length = np.exp(self.rng.uniform(log_low, log_high, size=(count, 1))).astype(np.float64)
                else:
                    half_length = np.full((count, 1), low, dtype=np.float64)
            origin = np.asarray([[float(line.x0), float(line.y0)]], dtype=np.float64)
            center_xy = origin + center_s * tangent
            left_xy = center_xy - half_length * tangent
            right_xy = center_xy + half_length * tangent
            gauss_xy = center_xy[:, None, :] + half_length[:, None, :] * gauss_offsets.reshape(1, 2, 1) * tangent.reshape(1, 1, 2)
            keep = self._outside_well_exclusion(left_xy) & self._outside_well_exclusion(right_xy)
            if not np.any(keep):
                continue
            n_keep = int(np.sum(keep))
            left_chunks.append(left_xy[keep])
            right_chunks.append(right_xy[keep])
            gauss_chunks.append(gauss_xy[keep])
            tangent_chunks.append(np.tile(tangent, (n_keep, 1)))
            aperture_chunks.append(np.full((n_keep, 1), max(float(line.aperture), 1.0e-12), dtype=np.float64))
            half_chunks.append(half_length[keep])

        if not left_chunks:
            return empty
        left_all = np.vstack(left_chunks)
        right_all = np.vstack(right_chunks)
        gauss_all = np.vstack(gauss_chunks)
        tangent_all = np.vstack(tangent_chunks)
        aperture_all = np.vstack(aperture_chunks)
        half_all = np.vstack(half_chunks)
        if self.sampling_mode == "random" and left_all.shape[0] > 0:
            order = self.rng.permutation(left_all.shape[0])
            left_all = left_all[order]
            right_all = right_all[order]
            gauss_all = gauss_all[order]
            tangent_all = tangent_all[order]
            aperture_all = aperture_all[order]
            half_all = half_all[order]
        if left_all.shape[0] < target_n:
            raise RuntimeError(f"HF segment conservation sampling failed after producer exclusion, got {left_all.shape[0]}/{target_n}.")
        return {
            "left_xy": left_all[:target_n].astype(np.float64),
            "right_xy": right_all[:target_n].astype(np.float64),
            "gauss_xy": gauss_all[:target_n].astype(np.float64),
            "tangent": tangent_all[:target_n].astype(np.float64),
            "aperture": aperture_all[:target_n].astype(np.float64),
            "half_length": half_all[:target_n].astype(np.float64),
        }

    def _outside_well_exclusion(self, xy: np.ndarray) -> np.ndarray:
        radius = float(self.cfg.get("well_pde_exclusion_radius_m", 0.0))
        if radius <= 0.0 or xy.shape[0] == 0:
            return np.ones((xy.shape[0],), dtype=bool)
        xw, yw = self.geometry.dirichlet_point()
        distance = np.hypot(xy[:, 0] - float(xw), xy[:, 1] - float(yw))
        return distance >= radius

    def _sample_hf_xy_with_tangent(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample HF line PDE points away from the producer Dirichlet point."""

        if n <= 0:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
        xy_parts: list[np.ndarray] = []
        tangent_parts: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < n and attempts < 20:
            attempts += 1
            xy, tangent = self._sample_hf_xy_with_tangent_raw(max(n, 2 * (n - total)))
            keep = self._outside_well_exclusion(xy)
            if np.any(keep):
                xy_parts.append(xy[keep])
                tangent_parts.append(tangent[keep])
                total += int(np.sum(keep))
        if total < n:
            raise RuntimeError(f"HF line PDE sampling failed after producer exclusion, got {total}/{n}.")
        xy_all = np.vstack(xy_parts)[:n]
        tangent_all = np.vstack(tangent_parts)[:n]
        return xy_all.astype(np.float64), tangent_all.astype(np.float64)

    def _sample_secondary_link_xy(self, line: FractureLine, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        main = self.geometry.main_frac
        if abs(line.x1 - line.x0) > abs(line.y1 - line.y0):
            return self._sample_line_xy(line, n)

        x = float(line.x0)
        y_low = min(line.y0, line.y1)
        y_high = max(line.y0, line.y1)
        sub_lines: list[FractureLine] = []
        if y_low < main.y_min:
            sub_lines.append(FractureLine(x, y_low, x, min(main.y_min, y_high), line.aperture, f"{line.name}_lower"))
        if y_high > main.y_max:
            sub_lines.append(FractureLine(x, max(main.y_max, y_low), x, y_high, line.aperture, f"{line.name}_upper"))
        if not sub_lines:
            return self._sample_line_xy(line, n)

        counts = self._allocate_counts(n, [segment.length for segment in sub_lines])
        chunks = [self._sample_line_xy(segment, count) for segment, count in zip(sub_lines, counts) if count > 0]
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "random" and xy.shape[0] > 0:
            self.rng.shuffle(xy, axis=0)
        return xy[:n]

    def sample_time(self, n: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 1), dtype=np.float64)
        t_min = float(self.cfg["t_min"])
        t_max = float(self.cfg["t_max"])
        if t_min < 0.0:
            raise ValueError(f"t_min must be non-negative, got {t_min:g}.")
        if t_max <= t_min:
            raise ValueError(f"t_max must be larger than t_min, got {t_max:g} <= {t_min:g}.")
        strategy = str(self.cfg.get("time_strategy", "log1p_uniform")).lower()
        if strategy != "log1p_uniform":
            raise ValueError("v13 currently supports only log1p_uniform time sampling.")
        anchors = np.asarray(self.cfg.get("time_anchor_days", []), dtype=np.float64).reshape(-1, 1)
        if anchors.size > 0:
            anchors = anchors[(anchors[:, 0] >= t_min) & (anchors[:, 0] <= t_max)].reshape(-1, 1)
            anchors = np.unique(anchors.reshape(-1)).reshape(-1, 1)
        anchor_fraction = float(self.cfg.get("time_anchor_fraction", 0.0))
        n_anchor = 0
        if anchors.shape[0] > 0 and anchor_fraction > 0.0:
            n_anchor = min(n, max(1, int(round(n * min(anchor_fraction, 1.0)))))
            if bool(self.cfg.get("time_anchor_include_all_when_possible", True)) and n >= anchors.shape[0]:
                n_anchor = max(n_anchor, anchors.shape[0])
            n_anchor = min(n, n_anchor)
        if n_anchor > 0:
            if n_anchor <= anchors.shape[0]:
                idx = np.linspace(0, anchors.shape[0] - 1, n_anchor, dtype=np.int64)
                anchor_values = anchors[idx]
            else:
                repeat = int(np.ceil(n_anchor / anchors.shape[0]))
                anchor_values = np.tile(anchors, (repeat, 1))[:n_anchor]
        else:
            anchor_values = np.empty((0, 1), dtype=np.float64)
        n_random = n - anchor_values.shape[0]
        if self.time_sampling_mode == "uniform":
            s = self._uniform_interval(np.log1p(t_min), np.log1p(t_max), n_random)
        else:
            s = self.rng.uniform(np.log1p(t_min), np.log1p(t_max), size=(n_random, 1))
        random_t = np.expm1(s) if n_random > 0 else np.empty((0, 1), dtype=np.float64)
        t = np.vstack([anchor_values, random_t])
        t = np.clip(t, t_min, t_max).astype(np.float64)
        if self.time_sampling_mode == "random":
            self.rng.shuffle(t, axis=0)
        return t

    def _time_count(self, key: str) -> int:
        value = int(self.cfg.get(key, self.cfg.get("n_time_collocation", 1)))
        if value <= 0:
            raise ValueError(f"sampler.{key} must be positive when time_pairing_mode='cartesian'.")
        return value

    def _xyt_from_xy_and_time(self, xy: np.ndarray, t: np.ndarray) -> np.ndarray:
        if xy.shape[0] == 0 or t.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        xy_rep = np.repeat(xy.astype(np.float64), t.shape[0], axis=0)
        t_tile = np.tile(t.astype(np.float64), (xy.shape[0], 1))
        return np.hstack([xy_rep, t_tile]).astype(np.float64)

    def _repeat_per_time(self, values: np.ndarray, n_time: int) -> np.ndarray:
        if values.shape[0] == 0 or n_time <= 0:
            return values[:0]
        return np.repeat(values, n_time, axis=0).astype(np.float64)

    def _make_xyt(self, xy: np.ndarray, time_count_key: str = "n_time_collocation") -> torch.Tensor:
        if self.time_pairing_mode == "paired":
            t = self.sample_time(xy.shape[0])
            return self._to_tensor(np.hstack([xy, t]).astype(np.float64))
        t = self.sample_time(self._time_count(time_count_key))
        return self._to_tensor(self._xyt_from_xy_and_time(xy, t))

    def _make_xyt_with_normal(self, xy: np.ndarray, normal: np.ndarray, time_count_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_pairing_mode == "paired":
            return self._make_xyt(xy, time_count_key), self._to_tensor(normal)
        n_time = self._time_count(time_count_key)
        t = self.sample_time(n_time)
        xyt = self._xyt_from_xy_and_time(xy, t)
        normal_rep = self._repeat_per_time(normal, n_time)
        return self._to_tensor(xyt), self._to_tensor(normal_rep)

    def _make_xyt_with_vector(self, xy: np.ndarray, vector: np.ndarray, time_count_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.time_pairing_mode == "paired":
            return self._make_xyt(xy, time_count_key), self._to_tensor(vector)
        n_time = self._time_count(time_count_key)
        t = self.sample_time(n_time)
        xyt = self._xyt_from_xy_and_time(xy, t)
        vector_rep = self._repeat_per_time(vector, n_time)
        return self._to_tensor(xyt), self._to_tensor(vector_rep)

    def _make_xyt_with_tangent_and_aperture(
        self,
        xy: np.ndarray,
        tangent: np.ndarray,
        aperture: np.ndarray,
        time_count_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.time_pairing_mode == "paired":
            return self._make_xyt(xy, time_count_key), self._to_tensor(tangent), self._to_tensor(aperture)
        n_time = self._time_count(time_count_key)
        t = self.sample_time(n_time)
        xyt = self._xyt_from_xy_and_time(xy, t)
        tangent_rep = self._repeat_per_time(tangent, n_time)
        aperture_rep = self._repeat_per_time(aperture, n_time)
        return self._to_tensor(xyt), self._to_tensor(tangent_rep), self._to_tensor(aperture_rep)

    def _make_hf_segment_conservation_xyt(self, raw: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        left_xy = raw["left_xy"]
        right_xy = raw["right_xy"]
        gauss_xy = raw["gauss_xy"]
        tangent = raw["tangent"]
        aperture = raw["aperture"]
        half_length = raw["half_length"]
        empty_xyt = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        empty_vec = self._to_tensor(np.empty((0, 2), dtype=np.float64))
        empty_col = self._to_tensor(np.empty((0, 1), dtype=np.float64))
        if left_xy.shape[0] == 0:
            return {
                "left_xyt": empty_xyt,
                "right_xyt": empty_xyt,
                "gauss_xyt": empty_xyt,
                "tangent": empty_vec,
                "gauss_tangent": empty_vec,
                "aperture": empty_col,
                "gauss_aperture": empty_col,
                "half_length": empty_col,
            }
        if self.time_pairing_mode == "paired":
            t = self.sample_time(left_xy.shape[0])
            left_xyt = np.hstack([left_xy, t]).astype(np.float64)
            right_xyt = np.hstack([right_xy, t]).astype(np.float64)
            gauss_xyt = np.hstack([gauss_xy.reshape(-1, 2), np.repeat(t, 2, axis=0)]).astype(np.float64)
            return {
                "left_xyt": self._to_tensor(left_xyt),
                "right_xyt": self._to_tensor(right_xyt),
                "gauss_xyt": self._to_tensor(gauss_xyt),
                "tangent": self._to_tensor(tangent),
                "gauss_tangent": self._to_tensor(np.repeat(tangent, 2, axis=0)),
                "aperture": self._to_tensor(aperture),
                "gauss_aperture": self._to_tensor(np.repeat(aperture, 2, axis=0)),
                "half_length": self._to_tensor(half_length),
            }
        n_time = int(self.cfg.get("n_time_hf_segment_conservation", self.cfg.get("n_time_link", self.cfg.get("n_time_collocation", 1))))
        if n_time <= 0:
            raise ValueError("sampler.n_time_hf_segment_conservation must be positive when time_pairing_mode='cartesian'.")
        t = self.sample_time(n_time)
        left_xyt = self._xyt_from_xy_and_time(left_xy, t)
        right_xyt = self._xyt_from_xy_and_time(right_xy, t)
        gauss_chunks: list[np.ndarray] = []
        for idx in range(left_xy.shape[0]):
            for time_value in t.reshape(-1):
                t_pair = np.full((2, 1), float(time_value), dtype=np.float64)
                gauss_chunks.append(np.hstack([gauss_xy[idx], t_pair]).astype(np.float64))
        gauss_xyt = np.vstack(gauss_chunks).astype(np.float64)
        return {
            "left_xyt": self._to_tensor(left_xyt),
            "right_xyt": self._to_tensor(right_xyt),
            "gauss_xyt": self._to_tensor(gauss_xyt),
            "tangent": self._to_tensor(self._repeat_per_time(tangent, n_time)),
            "gauss_tangent": self._to_tensor(np.repeat(self._repeat_per_time(tangent, n_time), 2, axis=0)),
            "aperture": self._to_tensor(self._repeat_per_time(aperture, n_time)),
            "gauss_aperture": self._to_tensor(np.repeat(self._repeat_per_time(aperture, n_time), 2, axis=0)),
            "half_length": self._to_tensor(self._repeat_per_time(half_length, n_time)),
        }

    def _make_hf_junction_flux_xyt(
        self,
        branch_xy: np.ndarray,
        direction: np.ndarray,
        aperture: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        empty_xyt = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        empty_vec = self._to_tensor(np.empty((0, 2), dtype=np.float64))
        empty_col = self._to_tensor(np.empty((0, 1), dtype=np.float64))
        if branch_xy.shape[0] == 0:
            return {"xyt": empty_xyt, "direction": empty_vec, "aperture": empty_col}

        chunks: list[np.ndarray] = []
        direction_chunks: list[np.ndarray] = []
        aperture_chunks: list[np.ndarray] = []
        if self.time_pairing_mode == "paired":
            times = self.sample_time(branch_xy.shape[0]).reshape(-1)
            for idx, time_value in enumerate(times):
                t_col = np.full((4, 1), float(time_value), dtype=np.float64)
                chunks.append(np.hstack([branch_xy[idx], t_col]).astype(np.float64))
                direction_chunks.append(direction[idx])
                aperture_chunks.append(aperture[idx])
        else:
            n_time = int(self.cfg.get("n_time_hf_junction_flux", self.cfg.get("n_time_link", self.cfg.get("n_time_collocation", 1))))
            if n_time <= 0:
                raise ValueError("sampler.n_time_hf_junction_flux must be positive when time_pairing_mode='cartesian'.")
            times = self.sample_time(n_time).reshape(-1)
            for idx in range(branch_xy.shape[0]):
                for time_value in times:
                    t_col = np.full((4, 1), float(time_value), dtype=np.float64)
                    chunks.append(np.hstack([branch_xy[idx], t_col]).astype(np.float64))
                    direction_chunks.append(direction[idx])
                    aperture_chunks.append(aperture[idx])
        return {
            "xyt": self._to_tensor(np.vstack(chunks).astype(np.float64)),
            "direction": self._to_tensor(np.vstack(direction_chunks).astype(np.float64)),
            "aperture": self._to_tensor(np.vstack(aperture_chunks).astype(np.float64)),
        }

    def _make_paired_xyt(
        self,
        first_xy: np.ndarray,
        second_xy: np.ndarray,
        time_count_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first_xy.shape != second_xy.shape:
            raise ValueError("Paired spatial arrays must have the same shape.")
        if self.time_pairing_mode == "paired":
            t = self.sample_time(first_xy.shape[0])
            return (
                self._to_tensor(np.hstack([first_xy, t]).astype(np.float64)),
                self._to_tensor(np.hstack([second_xy, t]).astype(np.float64)),
            )
        t = self.sample_time(self._time_count(time_count_key))
        return (
            self._to_tensor(self._xyt_from_xy_and_time(first_xy, t)),
            self._to_tensor(self._xyt_from_xy_and_time(second_xy, t)),
        )

    def _reflect_y(self, xy: np.ndarray) -> np.ndarray:
        reflected = xy.copy().astype(np.float64)
        reflected[:, 1] = float(self.geometry.domain.y_min) + float(self.geometry.domain.y_max) - reflected[:, 1]
        return reflected

    def _sample_region_xy(self, n: int, rect: Rect, target_region: int) -> np.ndarray:
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
            raise RuntimeError(f"Region sampling failed, target_region={target_region}, got {total}/{n}.")
        return np.vstack(accepted)[:n].astype(np.float64)

    def _sample_from_rects_for_region(self, rects: list[Rect], n: int, target_region: int) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        counts = self._allocate_counts(n, [max(rect.area, 1.0e-12) for rect in rects])
        chunks = [self._sample_region_xy(count, rect, target_region) for rect, count in zip(rects, counts) if count > 0]
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "random" and xy.shape[0] > 0:
            self.rng.shuffle(xy, axis=0)
        return xy[:n]

    def _rectangle_corners(self, center: np.ndarray, half_size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cx = center[:, 0:1]
        cy = center[:, 1:2]
        hx = half_size[:, 0:1]
        hy = half_size[:, 1:2]
        x = np.hstack([cx - hx, cx - hx, cx + hx, cx + hx, cx, cx, cx - hx, cx + hx])
        y = np.hstack([cy - hy, cy + hy, cy - hy, cy + hy, cy - hy, cy + hy, cy, cy])
        return x, y

    def _sample_log_half_size(
        self,
        n: int,
        min_key: str = "conservation_min_half_size_m",
        max_key: str = "conservation_max_half_size_m",
    ) -> np.ndarray:
        min_h = float(self.cfg.get(min_key, self.cfg.get("conservation_min_half_size_m", 1.0)))
        max_h = float(self.cfg.get(max_key, self.cfg.get("conservation_max_half_size_m", 12.0)))
        if min_h <= 0.0 or max_h < min_h:
            raise ValueError("sampler conservation half sizes must satisfy 0 < min <= max.")
        if self.sampling_mode == "uniform":
            values = self._uniform_interval(np.log(min_h), np.log(max_h), max(n, 1)).reshape(-1)
            hx = np.exp(values[:n])
            hy = np.exp(values[::-1][:n])
            return np.column_stack([hx, hy]).astype(np.float64)
        return np.exp(self.rng.uniform(np.log(min_h), np.log(max_h), size=(n, 2))).astype(np.float64)

    def _sample_conservation_rectangles_for_region(self, n: int, rect: Rect, target_region: int) -> dict[str, torch.Tensor]:
        if n <= 0:
            empty_2 = self._to_tensor(np.empty((0, 2), dtype=np.float64))
            empty_1 = self._to_tensor(np.empty((0, 1), dtype=np.float64))
            return {"center": empty_2, "half_size": empty_2, "t": empty_1}

        margin = float(self.cfg.get("conservation_boundary_margin_m", 1.0e-6))
        accepted_center: list[np.ndarray] = []
        accepted_half: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < n and attempts < 200:
            attempts += 1
            batch_n = max(128, (n - total) * 5)
            center = self._sample_region_xy(batch_n, rect, target_region)
            half_size = self._sample_log_half_size(center.shape[0])
            cx = center[:, 0:1]
            cy = center[:, 1:2]
            hx = half_size[:, 0:1]
            hy = half_size[:, 1:2]
            inside_bounds = (
                (cx - hx >= self.geometry.domain.x_min + margin)
                & (cx + hx <= self.geometry.domain.x_max - margin)
                & (cy - hy >= self.geometry.domain.y_min + margin)
                & (cy + hy <= self.geometry.domain.y_max - margin)
            ).reshape(-1)
            corner_x, corner_y = self._rectangle_corners(center, half_size)
            corner_region = self.geometry.region_id_np(corner_x, corner_y)
            same_region = np.all(corner_region == target_region, axis=1)
            keep = inside_bounds & same_region
            if np.any(keep):
                accepted_center.append(center[keep])
                accepted_half.append(half_size[keep])
                total += int(np.sum(keep))
        if total < n:
            raise RuntimeError(f"Local conservation rectangle sampling failed, target_region={target_region}, got {total}/{n}.")
        center_all = np.vstack(accepted_center)[:n].astype(np.float64)
        half_all = np.vstack(accepted_half)[:n].astype(np.float64)
        if self.sampling_mode == "random":
            order = self.rng.permutation(center_all.shape[0])
            center_all = center_all[order]
            half_all = half_all[order]
        return {
            "center": self._to_tensor(center_all),
            "half_size": self._to_tensor(half_all),
            "t": self._to_tensor(self.sample_time(n)),
        }

    def _sample_conservation_rectangles_near_hf_tips(self, n: int) -> dict[str, torch.Tensor]:
        if n <= 0:
            empty_2 = self._to_tensor(np.empty((0, 2), dtype=np.float64))
            empty_1 = self._to_tensor(np.empty((0, 1), dtype=np.float64))
            return {"center": empty_2, "half_size": empty_2, "t": empty_1}

        radius = float(self.cfg.get("conservation_hf_tip_radius_m", self.cfg.get("hf_tip_band_radius_m", 3.0)))
        tips = self._hf_dead_end_tip_xy()
        counts = self._allocate_counts(n, [1.0 for _ in range(tips.shape[0])])
        accepted_center: list[np.ndarray] = []
        accepted_half: list[np.ndarray] = []
        total = 0
        margin = float(self.cfg.get("conservation_boundary_margin_m", 1.0e-6))
        for tip, count in zip(tips, counts):
            if count <= 0:
                continue
            attempts = 0
            local_total = 0
            while local_total < count and attempts < 100:
                attempts += 1
                batch_n = max(64, (count - local_total) * 8)
                center = self._sample_tip_halo_xy(tip, batch_n, radius)
                half_size = self._sample_log_half_size(
                    center.shape[0],
                    "conservation_hf_tip_min_half_size_m",
                    "conservation_hf_tip_max_half_size_m",
                )
                cx = center[:, 0:1]
                cy = center[:, 1:2]
                hx = half_size[:, 0:1]
                hy = half_size[:, 1:2]
                inside_bounds = (
                    (cx - hx >= self.geometry.domain.x_min + margin)
                    & (cx + hx <= self.geometry.domain.x_max - margin)
                    & (cy - hy >= self.geometry.domain.y_min + margin)
                    & (cy + hy <= self.geometry.domain.y_max - margin)
                ).reshape(-1)
                corner_x, corner_y = self._rectangle_corners(center, half_size)
                corner_region = self.geometry.region_id_np(corner_x, corner_y)
                same_region = np.all(corner_region == REGION_SRV, axis=1)
                keep = inside_bounds & same_region
                if np.any(keep):
                    accepted_center.append(center[keep])
                    accepted_half.append(half_size[keep])
                    local_total += int(np.sum(keep))
                    total += int(np.sum(keep))
            if local_total < count:
                raise RuntimeError(f"HF tip local conservation sampling failed near {tip.tolist()}, got {local_total}/{count}.")
        if total < n:
            raise RuntimeError(f"HF tip local conservation sampling failed, got {total}/{n}.")
        center_all = np.vstack(accepted_center)[:n].astype(np.float64)
        half_all = np.vstack(accepted_half)[:n].astype(np.float64)
        if self.sampling_mode == "random":
            order = self.rng.permutation(center_all.shape[0])
            center_all = center_all[order]
            half_all = half_all[order]
        return {
            "center": self._to_tensor(center_all),
            "half_size": self._to_tensor(half_all),
            "t": self._to_tensor(self.sample_time(n)),
        }

    @staticmethod
    def _concat_rect_samples(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: torch.cat([first[key], second[key]], dim=0) for key in first.keys()}

    def sample_local_conservation_rectangles(self) -> dict[str, dict[str, torch.Tensor]]:
        srv = self._sample_conservation_rectangles_for_region(int(self.cfg.get("n_conservation_srv", 0)), self.geometry.srv_bg, REGION_SRV)
        tip_srv = self._sample_conservation_rectangles_near_hf_tips(int(self.cfg.get("n_conservation_hf_tip_srv", 0)))
        if tip_srv["center"].shape[0] > 0:
            srv = self._concat_rect_samples(srv, tip_srv)
        return {
            "srv": srv,
            "usrv": self._sample_conservation_rectangles_for_region(int(self.cfg.get("n_conservation_usrv", 0)), self.geometry.domain, REGION_USRV),
        }

    def sample_local_conservation_candidate_rectangles(self, scale: int) -> dict[str, dict[str, torch.Tensor]]:
        scale = max(int(scale), 1)
        srv = self._sample_conservation_rectangles_for_region(int(self.cfg.get("n_conservation_srv", 0)) * scale, self.geometry.srv_bg, REGION_SRV)
        tip_srv = self._sample_conservation_rectangles_near_hf_tips(int(self.cfg.get("n_conservation_hf_tip_srv", 0)) * scale)
        if tip_srv["center"].shape[0] > 0:
            srv = self._concat_rect_samples(srv, tip_srv)
        return {
            "srv": srv,
            "usrv": self._sample_conservation_rectangles_for_region(int(self.cfg.get("n_conservation_usrv", 0)) * scale, self.geometry.domain, REGION_USRV),
        }

    def sample_near_hf_srv_points(self) -> dict[str, np.ndarray]:
        n = int(self.cfg["n_near_hf_srv"])
        width = float(self.cfg["hf_srv_band_width_m"])
        rects = [rect.expanded(width, self.geometry.domain) for rect in self.geometry.hf_rects]
        return {"srv": self._sample_from_rects_for_region(rects, n, REGION_SRV)}

    def _hf_dead_end_tip_xy(self) -> np.ndarray:
        points: list[tuple[float, float]] = []
        main = self.geometry.hf_lines[0]
        points.append((float(main.x0), float(main.y0)))
        for line in self.geometry.hf_lines[1:]:
            points.append((float(line.x0), float(line.y0)))
            points.append((float(line.x1), float(line.y1)))
        return np.asarray(points, dtype=np.float64)

    def _sample_tip_halo_xy(self, tip: np.ndarray, n: int, radius: float) -> np.ndarray:
        if n <= 0:
            return np.empty((0, 2), dtype=np.float64)
        radius = max(float(radius), 1.0e-12)
        rect = Rect(
            max(self.geometry.domain.x_min, float(tip[0]) - radius),
            min(self.geometry.domain.x_max, float(tip[0]) + radius),
            max(self.geometry.domain.y_min, float(tip[1]) - radius),
            min(self.geometry.domain.y_max, float(tip[1]) + radius),
            "hf_tip_halo",
        )
        accepted: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < n and attempts < 100:
            attempts += 1
            batch_n = max(64, (n - total) * 5)
            xy = self._sample_rect_xy(rect, batch_n)
            distance = np.hypot(xy[:, 0] - float(tip[0]), xy[:, 1] - float(tip[1]))
            region = self.geometry.region_id_np(xy[:, 0], xy[:, 1])
            keep = (distance <= radius) & (region == REGION_SRV)
            if np.any(keep):
                accepted.append(xy[keep])
                total += int(np.sum(keep))
        if total < n:
            raise RuntimeError(f"HF tip halo sampling failed near {tip.tolist()}, got {total}/{n}.")
        xy_all = np.vstack(accepted)[:n].astype(np.float64)
        if self.sampling_mode == "random":
            self.rng.shuffle(xy_all, axis=0)
        return xy_all

    def sample_near_hf_tip_srv_points(self) -> dict[str, np.ndarray]:
        n = int(self.cfg.get("n_near_hf_tip_srv", 0))
        if n <= 0:
            return {"srv": np.empty((0, 2), dtype=np.float64)}
        radius = float(self.cfg.get("hf_tip_band_radius_m", 3.0))
        tips = self._hf_dead_end_tip_xy()
        counts = self._allocate_counts(n, [1.0 for _ in range(tips.shape[0])])
        chunks = [self._sample_tip_halo_xy(tip, count, radius) for tip, count in zip(tips, counts) if count > 0]
        xy = np.vstack(chunks) if chunks else np.empty((0, 2), dtype=np.float64)
        if self.sampling_mode == "random" and xy.shape[0] > 0:
            self.rng.shuffle(xy, axis=0)
        return {"srv": xy[:n].astype(np.float64)}

    def sample_near_srv_usrv_points(self) -> dict[str, np.ndarray]:
        n = int(self.cfg["n_near_srv_usrv"])
        width = float(self.cfg["srv_usrv_band_width_m"])
        rects = self.geometry.srv_usrv_band_rects(width)
        n_srv = n // 2
        n_usrv = n - n_srv
        return {
            "srv": self._sample_from_rects_for_region(rects, n_srv, REGION_SRV),
            "usrv": self._sample_from_rects_for_region(rects, n_usrv, REGION_USRV),
        }

    def _sample_pde_points_scaled(self, scale: int = 1) -> dict[str, Any]:
        scale = max(int(scale), 1)
        hf_xy, hf_tangent = self._sample_hf_xy_with_tangent(int(self.cfg["n_pde_hf"]) * scale)
        srv_xy = self._sample_region_xy(int(self.cfg["n_pde_srv"]) * scale, self.geometry.srv_bg, REGION_SRV)
        usrv_xy = self._sample_region_xy(int(self.cfg["n_pde_usrv"]) * scale, self.geometry.domain, REGION_USRV)
        near_hf = self.sample_near_hf_srv_points()
        near_hf_tip = self.sample_near_hf_tip_srv_points()
        near_srv_usrv = self.sample_near_srv_usrv_points()
        srv_xy = np.vstack([srv_xy, near_hf["srv"], near_hf_tip["srv"], near_srv_usrv["srv"]])
        usrv_xy = np.vstack([usrv_xy, near_srv_usrv["usrv"]])
        hf_xyt, hf_tangent_tensor = self._make_xyt_with_vector(hf_xy, hf_tangent, "n_time_pde")
        return {
            "hf": {"xyt": hf_xyt, "tangent": hf_tangent_tensor},
            "srv": self._make_xyt(srv_xy, "n_time_pde"),
            "usrv": self._make_xyt(usrv_xy, "n_time_pde"),
        }

    def sample_pde_points(self) -> dict[str, Any]:
        return self._sample_pde_points_scaled(1)

    def sample_pde_candidate_points(self, scale: int) -> dict[str, Any]:
        return self._sample_pde_points_scaled(scale)

    def sample_dirichlet_boundary_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg["n_dirichlet"])
        xw, yw = self.geometry.dirichlet_point()
        xy = np.column_stack([np.full(n, xw, dtype=np.float64), np.full(n, yw, dtype=np.float64)])
        return {"xyt": self._make_xyt(xy, "n_time_boundary")}

    def sample_hf_main_link_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg.get("n_hf_main_link", 0))
        if n <= 0:
            return {"xyt": self._to_tensor(np.empty((0, 3), dtype=np.float64))}
        line = self.geometry.hf_lines[0]
        radius = float(self.cfg.get("well_pde_exclusion_radius_m", 0.0))
        if radius > 0.0 and abs(line.y1 - line.y0) <= abs(line.x1 - line.x0):
            x_low = min(line.x0, line.x1)
            x_high = max(line.x0, line.x1) - radius
            line = FractureLine(x_low, line.y0, max(x_low, x_high), line.y1, line.aperture, line.name)
        xy = self._sample_line_xy(line, n)
        return {"xyt": self._make_xyt(xy, "n_time_link")}

    def sample_hf_secondary_link_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg.get("n_hf_secondary_link", 0))
        empty = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        secondary_lines = self.geometry.hf_lines[1:]
        if n <= 0 or not secondary_lines:
            return {"xyt": empty, "junction_xyt": empty}

        main = self.geometry.hf_lines[0]
        y_junction = main.y0
        counts = self._allocate_counts(n, [line.length for line in secondary_lines])
        xy_chunks: list[np.ndarray] = []
        junction_xy_chunks: list[np.ndarray] = []
        for line, count in zip(secondary_lines, counts):
            if count <= 0:
                continue
            xy = self._sample_secondary_link_xy(line, count)
            junction_xy = np.column_stack([np.full(count, line.x0, dtype=np.float64), np.full(count, y_junction, dtype=np.float64)])
            xy_chunks.append(xy.astype(np.float64))
            junction_xy_chunks.append(junction_xy.astype(np.float64))

        xy_all = np.vstack(xy_chunks) if xy_chunks else np.empty((0, 2), dtype=np.float64)
        junction_xy_all = np.vstack(junction_xy_chunks) if junction_xy_chunks else np.empty((0, 2), dtype=np.float64)
        order = self.rng.permutation(xy_all.shape[0]) if self.sampling_mode == "random" and xy_all.shape[0] > 0 else np.arange(xy_all.shape[0])
        xyt, junction_xyt = self._make_paired_xyt(xy_all[order], junction_xy_all[order], "n_time_link")
        return {"xyt": xyt, "junction_xyt": junction_xyt}

    def sample_hf_junction_pair_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg.get("n_hf_junction", 0))
        empty = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        secondary_lines = self.geometry.hf_lines[1:]
        if n <= 0 or not secondary_lines:
            return {"main_xyt": empty, "secondary_xyt": empty}

        main = self.geometry.hf_lines[0]
        y_center = main.y0
        offset = float(self.cfg.get("junction_offset_m", 0.05))
        counts = self._allocate_counts(n, [line.length for line in secondary_lines])
        main_xy_chunks: list[np.ndarray] = []
        secondary_xy_chunks: list[np.ndarray] = []
        for line, count in zip(secondary_lines, counts):
            if count <= 0:
                continue
            if self.sampling_mode == "uniform":
                idx = np.arange(count, dtype=np.int64).reshape(-1, 1)
                x_sign = np.where(idx % 2 == 0, -1.0, 1.0).astype(np.float64)
                y_sign = np.where((idx // 2) % 2 == 0, -1.0, 1.0).astype(np.float64)
            else:
                x_sign = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(count, 1))
                y_sign = self.rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(count, 1))

            x_center = float(line.x0)
            main_x = np.clip(x_center + x_sign * offset, min(main.x0, main.x1), max(main.x0, main.x1))
            main_y = np.full((count, 1), y_center, dtype=np.float64)
            sec_x = np.full((count, 1), x_center, dtype=np.float64)
            sec_y = np.clip(y_center + y_sign * offset, min(line.y0, line.y1), max(line.y0, line.y1))
            main_xy_chunks.append(np.hstack([main_x, main_y]).astype(np.float64))
            secondary_xy_chunks.append(np.hstack([sec_x, sec_y]).astype(np.float64))

        main_xy = np.vstack(main_xy_chunks) if main_xy_chunks else np.empty((0, 2), dtype=np.float64)
        secondary_xy = np.vstack(secondary_xy_chunks) if secondary_xy_chunks else np.empty((0, 2), dtype=np.float64)
        order = self.rng.permutation(main_xy.shape[0]) if self.sampling_mode == "random" and main_xy.shape[0] > 0 else np.arange(main_xy.shape[0])
        main_xyt, secondary_xyt = self._make_paired_xyt(main_xy[order], secondary_xy[order], "n_time_link")
        return {"main_xyt": main_xyt, "secondary_xyt": secondary_xyt}

    def sample_hf_junction_flux_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg.get("n_hf_junction_flux", 0))
        secondary_lines = self.geometry.hf_lines[1:]
        if n <= 0 or not secondary_lines:
            empty = np.empty((0, 4, 2), dtype=np.float64)
            empty_col = np.empty((0, 4, 1), dtype=np.float64)
            return self._make_hf_junction_flux_xyt(empty, empty, empty_col)

        main = self.geometry.hf_lines[0]
        y_center = float(main.y0)
        offset_cfg = float(self.cfg.get("hf_junction_flux_offset_m", self.cfg.get("junction_offset_m", 0.05)))
        counts = self._allocate_counts(n, [line.length for line in secondary_lines])
        branch_chunks: list[np.ndarray] = []
        direction_chunks: list[np.ndarray] = []
        aperture_chunks: list[np.ndarray] = []
        main_x_low = min(float(main.x0), float(main.x1))
        main_x_high = max(float(main.x0), float(main.x1))
        for line, count in zip(secondary_lines, counts):
            if count <= 0:
                continue
            x_center = float(line.x0)
            y_low = min(float(line.y0), float(line.y1))
            y_high = max(float(line.y0), float(line.y1))
            offset = min(max(offset_cfg, 1.0e-12), max(x_center - main_x_low, 1.0e-12), max(main_x_high - x_center, 1.0e-12), max(y_center - y_low, 1.0e-12), max(y_high - y_center, 1.0e-12))
            branches = np.asarray(
                [
                    [x_center - offset, y_center],
                    [x_center + offset, y_center],
                    [x_center, y_center - offset],
                    [x_center, y_center + offset],
                ],
                dtype=np.float64,
            )
            directions = np.asarray(
                [
                    [-1.0, 0.0],
                    [1.0, 0.0],
                    [0.0, -1.0],
                    [0.0, 1.0],
                ],
                dtype=np.float64,
            )
            apertures = np.asarray(
                [
                    [max(float(main.aperture), 1.0e-12)],
                    [max(float(main.aperture), 1.0e-12)],
                    [max(float(line.aperture), 1.0e-12)],
                    [max(float(line.aperture), 1.0e-12)],
                ],
                dtype=np.float64,
            )
            branch_chunks.append(np.tile(branches.reshape(1, 4, 2), (count, 1, 1)))
            direction_chunks.append(np.tile(directions.reshape(1, 4, 2), (count, 1, 1)))
            aperture_chunks.append(np.tile(apertures.reshape(1, 4, 1), (count, 1, 1)))

        branch_xy = np.vstack(branch_chunks) if branch_chunks else np.empty((0, 4, 2), dtype=np.float64)
        direction = np.vstack(direction_chunks) if direction_chunks else np.empty((0, 4, 2), dtype=np.float64)
        aperture = np.vstack(aperture_chunks) if aperture_chunks else np.empty((0, 4, 1), dtype=np.float64)
        if self.sampling_mode == "random" and branch_xy.shape[0] > 0:
            order = self.rng.permutation(branch_xy.shape[0])
            branch_xy = branch_xy[order]
            direction = direction[order]
            aperture = aperture[order]
        return self._make_hf_junction_flux_xyt(branch_xy[:n], direction[:n], aperture[:n])

    def sample_hf_tip_neumann_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg.get("n_hf_tip_neumann", 0))
        empty_xyt = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        empty_tangent = self._to_tensor(np.empty((0, 2), dtype=np.float64))
        if n <= 0:
            return {"xyt": empty_xyt, "tangent": empty_tangent}

        tip_xy: list[tuple[float, float]] = []
        tip_tangent: list[tuple[float, float]] = []
        main = self.geometry.hf_lines[0]
        tx, ty = main.tangent
        tip_xy.append((float(main.x0), float(main.y0)))
        tip_tangent.append((tx, ty))
        for line in self.geometry.hf_lines[1:]:
            tx, ty = line.tangent
            tip_xy.append((float(line.x0), float(line.y0)))
            tip_tangent.append((tx, ty))
            tip_xy.append((float(line.x1), float(line.y1)))
            tip_tangent.append((tx, ty))

        points = np.asarray(tip_xy, dtype=np.float64)
        tangent = np.asarray(tip_tangent, dtype=np.float64)
        counts = self._allocate_counts(n, [1.0 for _ in range(points.shape[0])])
        xy_chunks: list[np.ndarray] = []
        tangent_chunks: list[np.ndarray] = []
        for point, vector, count in zip(points, tangent, counts):
            if count <= 0:
                continue
            xy_chunks.append(np.tile(point.reshape(1, 2), (count, 1)))
            tangent_chunks.append(np.tile(vector.reshape(1, 2), (count, 1)))
        xy_all = np.vstack(xy_chunks).astype(np.float64)
        tangent_all = np.vstack(tangent_chunks).astype(np.float64)
        if self.sampling_mode == "random" and xy_all.shape[0] > 0:
            order = self.rng.permutation(xy_all.shape[0])
            xy_all = xy_all[order]
            tangent_all = tangent_all[order]
        xyt, tangent_tensor = self._make_xyt_with_vector(xy_all[:n], tangent_all[:n], "n_time_link")
        return {"xyt": xyt, "tangent": tangent_tensor}

    def sample_hf_leakoff_balance_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg.get("n_hf_leakoff_balance", 0))
        empty_xyt = self._to_tensor(np.empty((0, 3), dtype=np.float64))
        empty_tangent = self._to_tensor(np.empty((0, 2), dtype=np.float64))
        empty_aperture = self._to_tensor(np.empty((0, 1), dtype=np.float64))
        if n <= 0:
            return {"xyt": empty_xyt, "tangent": empty_tangent, "aperture": empty_aperture}

        endpoint_margin = float(self.cfg.get("hf_leakoff_endpoint_margin_m", 0.05))
        xy_parts: list[np.ndarray] = []
        tangent_parts: list[np.ndarray] = []
        aperture_parts: list[np.ndarray] = []
        total = 0
        attempts = 0
        while total < n and attempts < 20:
            attempts += 1
            xy, tangent, aperture = self._sample_hf_xy_with_tangent_and_aperture_raw(max(n, 2 * (n - total)), endpoint_margin)
            keep = self._outside_well_exclusion(xy)
            if np.any(keep):
                xy_parts.append(xy[keep])
                tangent_parts.append(tangent[keep])
                aperture_parts.append(aperture[keep])
                total += int(np.sum(keep))
        if total < n:
            raise RuntimeError(f"HF leakoff balance sampling failed after producer exclusion, got {total}/{n}.")
        xy_all = np.vstack(xy_parts)[:n]
        tangent_all = np.vstack(tangent_parts)[:n]
        aperture_all = np.vstack(aperture_parts)[:n]
        xyt, tangent_tensor, aperture_tensor = self._make_xyt_with_tangent_and_aperture(xy_all, tangent_all, aperture_all, "n_time_link")
        return {"xyt": xyt, "tangent": tangent_tensor, "aperture": aperture_tensor}

    def sample_hf_segment_conservation_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg.get("n_hf_segment_conservation", 0))
        raw = self._sample_hf_segment_conservation_raw(n)
        return self._make_hf_segment_conservation_xyt(raw)

    def sample_symmetry_pair_points(self) -> dict[str, dict[str, torch.Tensor]]:
        samples: dict[str, dict[str, torch.Tensor]] = {}
        counts = {
            "hf": int(self.cfg.get("n_symmetry_hf", 0)),
            "srv": int(self.cfg.get("n_symmetry_srv", 0)),
            "usrv": int(self.cfg.get("n_symmetry_usrv", 0)),
        }
        if counts["hf"] > 0:
            hf_xy = self._sample_hf_xy(counts["hf"])
            hf_reflected = self._reflect_y(hf_xy)
            xyt, reflected_xyt = self._make_paired_xyt(hf_xy, hf_reflected, "n_time_symmetry")
            samples["hf"] = {"xyt": xyt, "reflected_xyt": reflected_xyt}
        if counts["srv"] > 0:
            srv_xy = self._sample_region_xy(counts["srv"], self.geometry.srv_bg, REGION_SRV)
            srv_reflected = self._reflect_y(srv_xy)
            xyt, reflected_xyt = self._make_paired_xyt(srv_xy, srv_reflected, "n_time_symmetry")
            samples["srv"] = {"xyt": xyt, "reflected_xyt": reflected_xyt}
        if counts["usrv"] > 0:
            usrv_xy = self._sample_region_xy(counts["usrv"], self.geometry.domain, REGION_USRV)
            usrv_reflected = self._reflect_y(usrv_xy)
            xyt, reflected_xyt = self._make_paired_xyt(usrv_xy, usrv_reflected, "n_time_symmetry")
            samples["usrv"] = {"xyt": xyt, "reflected_xyt": reflected_xyt}
        return samples

    def _sample_right_neumann_y(self, n: int) -> np.ndarray:
        xw, yw = self.geometry.dirichlet_point()
        _ = xw
        gap = 0.05
        low = yw - gap
        high = yw + gap
        d = self.geometry.domain
        if self.sampling_mode == "uniform":
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
            y = self.rng.uniform(d.y_min, d.y_max, size=(max(64, 2 * (n - total)), 1))
            keep = y[(y[:, 0] < low) | (y[:, 0] > high)]
            if keep.size > 0:
                values.append(keep.reshape(-1, 1))
                total += keep.size
        return np.vstack(values)[:n].astype(np.float64)

    def sample_neumann_boundary_points(self) -> dict[str, torch.Tensor]:
        n = int(self.cfg["n_neumann"])
        d = self.geometry.domain
        counts = self._allocate_counts(n, [d.height, d.height, d.width, d.width])
        chunks: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        if counts[0] > 0:
            x = np.full((counts[0], 1), d.x_min)
            y = self._uniform_interval(d.y_min, d.y_max, counts[0]) if self.sampling_mode == "uniform" else self.rng.uniform(d.y_min, d.y_max, size=(counts[0], 1))
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[-1.0, 0.0]], (counts[0], 1)))
        if counts[1] > 0:
            x = np.full((counts[1], 1), d.x_max)
            y = self._sample_right_neumann_y(counts[1])
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[1.0, 0.0]], (counts[1], 1)))
        if counts[2] > 0:
            x = self._uniform_interval(d.x_min, d.x_max, counts[2]) if self.sampling_mode == "uniform" else self.rng.uniform(d.x_min, d.x_max, size=(counts[2], 1))
            y = np.full((counts[2], 1), d.y_min)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, -1.0]], (counts[2], 1)))
        if counts[3] > 0:
            x = self._uniform_interval(d.x_min, d.x_max, counts[3]) if self.sampling_mode == "uniform" else self.rng.uniform(d.x_min, d.x_max, size=(counts[3], 1))
            y = np.full((counts[3], 1), d.y_max)
            chunks.append(np.hstack([x, y]))
            normals.append(np.tile([[0.0, 1.0]], (counts[3], 1)))
        xy = np.vstack(chunks).astype(np.float64)
        normal = np.vstack(normals).astype(np.float64)
        xyt, normal_tensor = self._make_xyt_with_normal(xy, normal, "n_time_boundary")
        return {"xyt": xyt, "normal": normal_tensor}

    def _sample_segments(
        self,
        n: int,
        segments: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    ) -> dict[str, torch.Tensor]:
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
            tau = self._uniform_interval(0.0, 1.0, count) if self.sampling_mode == "uniform" else self.rng.uniform(0.0, 1.0, size=(count, 1))
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
        if self.sampling_mode == "random" and xy.shape[0] > 0:
            order = self.rng.permutation(xy.shape[0])
            xy = xy[order]
            normal = normal[order]
        xyt, normal_tensor = self._make_xyt_with_normal(xy, normal, "n_time_interface")
        return {"xyt": xyt, "normal": normal_tensor}

    def sample_hf_srv_interface_points(self) -> dict[str, torch.Tensor]:
        return self._sample_segments(int(self.cfg["n_interface_hf_srv"]), self.geometry.hf_srv_interface_segments())

    def sample_srv_usrv_interface_points(self) -> dict[str, torch.Tensor]:
        return self._sample_segments(int(self.cfg["n_interface_srv_usrv"]), self.geometry.srv_usrv_interface_segments())

    def sample_all(self) -> dict[str, Any]:
        return {
            "pde": self.sample_pde_points(),
            "dirichlet": self.sample_dirichlet_boundary_points(),
            "neumann": self.sample_neumann_boundary_points(),
            "interface_hf_srv": self.sample_hf_srv_interface_points(),
            "interface_srv_usrv": self.sample_srv_usrv_interface_points(),
            "hf_main_link": self.sample_hf_main_link_points(),
            "hf_secondary_link": self.sample_hf_secondary_link_points(),
            "hf_junction": self.sample_hf_junction_pair_points(),
            "hf_junction_flux": self.sample_hf_junction_flux_points(),
            "hf_tip_neumann": self.sample_hf_tip_neumann_points(),
            "hf_leakoff_balance": self.sample_hf_leakoff_balance_points(),
            "hf_segment_conservation": self.sample_hf_segment_conservation_points(),
            "symmetry": self.sample_symmetry_pair_points(),
            "local_conservation": self.sample_local_conservation_rectangles(),
        }
