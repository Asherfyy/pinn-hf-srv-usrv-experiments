from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry
from src.sampler import ReservoirSampler
from src.utils import get_torch_dtype


def make_sampler() -> tuple[ReservoirGeometry, ReservoirSampler]:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    cfg = copy.deepcopy(config["sampler"])
    cfg.update(
        {
            "n_pde_hf": 48,
            "n_pde_srv": 64,
            "n_pde_usrv": 64,
            "n_initial_hf": 16,
            "n_initial_srv": 16,
            "n_initial_usrv": 16,
            "n_dirichlet": 16,
            "n_neumann": 16,
            "n_interface_hf_srv": 16,
            "n_interface_srv_usrv": 16,
        }
    )
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    sampler = ReservoirSampler(
        geom,
        cfg,
        device=torch.device("cpu"),
        dtype=get_torch_dtype(config["runtime"]["dtype"]),
        seed=2026,
    )
    return geom, sampler


def region_ids(geom: ReservoirGeometry, xyt: torch.Tensor) -> np.ndarray:
    arr = xyt.detach().cpu().numpy()
    return geom.region_id_np(arr[:, 0], arr[:, 1])


def test_pde_sampling_regions() -> None:
    geom, sampler = make_sampler()
    points = sampler.sample_pde_points()
    assert np.all(region_ids(geom, points["hf"]) == REGION_HF)
    assert np.all(region_ids(geom, points["srv"]) == REGION_SRV)
    assert np.all(region_ids(geom, points["usrv"]) == REGION_USRV)

    margin = float(sampler.cfg.get("pde_dirichlet_exclusion_m", 0.0))
    if margin > 0.0:
        d_m = geom.distance_to_dirichlet_torch(points["hf"][:, 0:1], points["hf"][:, 1:2]) * geom.l_ref
        assert torch.min(d_m).item() >= margin - 1.0e-5


def test_initial_time_and_dirichlet_x() -> None:
    _geom, sampler = make_sampler()
    initial = sampler.sample_initial_points()
    for key in ["hf", "srv", "usrv"]:
        assert torch.allclose(initial[key][:, 2], torch.zeros_like(initial[key][:, 2]))

    boundary = sampler.sample_dirichlet_boundary_points()
    assert torch.allclose(boundary["xyt"][:, 0], torch.full_like(boundary["xyt"][:, 0], 360.0))


def test_log1p_time_sampling_bounds_shape_and_median() -> None:
    _geom, sampler = make_sampler()
    samples = sampler.sample_time(20000)
    assert samples.shape == (20000, 1)
    assert np.all(np.isfinite(samples))
    assert float(samples.min()) >= float(sampler.cfg["t_min"]) - 1.0e-6
    assert float(samples.max()) <= float(sampler.cfg["t_max"]) + 1.0e-6

    median = float(np.median(samples))
    theoretical = float(np.expm1(0.5 * (np.log1p(float(sampler.cfg["t_min"])) + np.log1p(float(sampler.cfg["t_max"])))))
    assert 20.0 < median < 45.0
    assert abs(median - theoretical) < 5.0


def test_log1p_time_sampling_is_uniform_in_log_space_and_reproducible() -> None:
    geom, sampler_a = make_sampler()
    _geom_b, sampler_b = make_sampler()
    samples_a = sampler_a.sample_time(30000)
    samples_b = sampler_b.sample_time(30000)
    assert np.allclose(samples_a, samples_b)

    log_samples = np.log1p(samples_a.reshape(-1))
    counts, _edges = np.histogram(log_samples, bins=10)
    expected = counts.mean()
    # 固定 seed 下 3 万样本足以让 log 空间分箱比较均匀；阈值留有随机波动余量。
    assert np.max(np.abs(counts - expected)) / expected < 0.12

    assert geom is not None
