from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry
from src.physics import interface_offset_points
from src.sampler import ReservoirSampler
from src.utils import get_torch_dtype


def make_sampler(seed: int = 2026, sampling_mode: str = "random") -> tuple[dict, ReservoirGeometry, ReservoirSampler]:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 64,
            "n_pde_srv": 96,
            "n_pde_usrv": 96,
            "n_near_hf_srv": 64,
            "n_near_srv_usrv": 64,
            "n_dirichlet": 64,
            "n_neumann": 64,
            "n_interface_hf_srv": 128,
            "n_interface_srv_usrv": 128,
            "sampling_mode": sampling_mode,
            "time_sampling_mode": sampling_mode,
        }
    )
    geom = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geom, config["sampler"], torch.device("cpu"), get_torch_dtype(config["runtime"]["dtype"]), seed=seed)
    return config, geom, sampler


def region_ids(geom: ReservoirGeometry, xyt: torch.Tensor) -> np.ndarray:
    arr = xyt.detach().cpu().numpy()
    return geom.region_id_np(arr[:, 0], arr[:, 1])


def count_points_on_axis_aligned_segment(xy: np.ndarray, p0: tuple[float, float], p1: tuple[float, float]) -> int:
    x0, y0 = p0
    x1, y1 = p1
    atol = 1.0e-9
    if np.isclose(x0, x1, atol=atol):
        mask = np.isclose(xy[:, 0], x0, atol=atol) & (xy[:, 1] >= min(y0, y1) - atol) & (xy[:, 1] <= max(y0, y1) + atol)
    elif np.isclose(y0, y1, atol=atol):
        mask = np.isclose(xy[:, 1], y0, atol=atol) & (xy[:, 0] >= min(x0, x1) - atol) & (xy[:, 0] <= max(x0, x1) + atol)
    else:
        raise AssertionError("test only supports axis-aligned interface segments")
    return int(mask.sum())


def test_pde_sampling_regions() -> None:
    _config, geom, sampler = make_sampler()
    pde = sampler.sample_pde_points()
    assert np.all(region_ids(geom, pde["hf"]) == REGION_HF)
    assert np.all(region_ids(geom, pde["srv"]) == REGION_SRV)
    assert np.all(region_ids(geom, pde["usrv"]) == REGION_USRV)


def test_near_points_are_merged_into_correct_regions() -> None:
    _config, geom, sampler = make_sampler()
    near_hf = sampler.sample_near_hf_srv_points()
    assert np.all(geom.region_id_np(near_hf["srv"][:, 0], near_hf["srv"][:, 1]) == REGION_SRV)
    near_srv = sampler.sample_near_srv_usrv_points()
    assert np.all(geom.region_id_np(near_srv["srv"][:, 0], near_srv["srv"][:, 1]) == REGION_SRV)
    assert np.all(geom.region_id_np(near_srv["usrv"][:, 0], near_srv["usrv"][:, 1]) == REGION_USRV)


def test_interface_points_have_valid_two_sided_offsets() -> None:
    config, geom, sampler = make_sampler()
    hf_srv = sampler.sample_hf_srv_interface_points()
    minus, plus, _normal, mask = interface_offset_points(
        hf_srv["xyt"],
        hf_srv["normal"],
        float(config["sampler"]["eps_hf_srv"]),
        geom,
        REGION_HF,
        REGION_SRV,
    )
    assert int(mask.sum().item()) > 0
    assert np.all(region_ids(geom, minus) == REGION_HF)
    assert np.all(region_ids(geom, plus) == REGION_SRV)

    srv_usrv = sampler.sample_srv_usrv_interface_points()
    minus, plus, _normal, mask = interface_offset_points(
        srv_usrv["xyt"],
        srv_usrv["normal"],
        float(config["sampler"]["eps_srv_usrv"]),
        geom,
        REGION_SRV,
        REGION_USRV,
    )
    assert int(mask.sum().item()) > 0
    assert np.all(region_ids(geom, minus) == REGION_SRV)
    assert np.all(region_ids(geom, plus) == REGION_USRV)


def test_hf_srv_interface_minimum_points_keep_short_segments_sampled() -> None:
    _config, geom, sampler = make_sampler(sampling_mode="uniform")
    segments = geom.hf_srv_interface_segments()
    min_points = 4
    total = len(segments) * min_points + 37
    sampler.cfg["n_interface_hf_srv"] = total
    sampler.cfg["min_points_per_hf_srv_interface_segment"] = min_points

    hf_srv = sampler.sample_hf_srv_interface_points()
    xy = hf_srv["xyt"][:, :2].detach().cpu().numpy()
    assert xy.shape[0] == total

    counts = [count_points_on_axis_aligned_segment(xy, p0, p1) for p0, p1, _normal in segments]
    assert min(counts) >= min_points

    short_counts = [
        count
        for count, (p0, p1, _normal) in zip(counts, segments)
        if float(np.hypot(p1[0] - p0[0], p1[1] - p0[1])) < 0.02
    ]
    assert short_counts
    assert min(short_counts) >= min_points


def test_uniform_sampling_mode_is_deterministic_across_seeds() -> None:
    _config_a, _geom_a, sampler_a = make_sampler(seed=123, sampling_mode="uniform")
    _config_b, _geom_b, sampler_b = make_sampler(seed=999, sampling_mode="uniform")
    samples_a = sampler_a.sample_all()
    samples_b = sampler_b.sample_all()
    for region_name in ["hf", "srv", "usrv"]:
        assert torch.allclose(samples_a["pde"][region_name], samples_b["pde"][region_name])
    assert torch.allclose(samples_a["dirichlet"]["xyt"], samples_b["dirichlet"]["xyt"])
    assert torch.allclose(samples_a["interface_hf_srv"]["xyt"], samples_b["interface_hf_srv"]["xyt"])
    assert torch.allclose(samples_a["interface_srv_usrv"]["xyt"], samples_b["interface_srv_usrv"]["xyt"])


def test_latin_hypercube_sampling_changes_between_calls() -> None:
    _config, _geom, sampler = make_sampler(seed=123, sampling_mode="latin_hypercube")
    samples_a = sampler.sample_all()
    samples_b = sampler.sample_all()
    assert not torch.allclose(samples_a["pde"]["hf"], samples_b["pde"]["hf"])
    assert not torch.allclose(samples_a["pde"]["srv"], samples_b["pde"]["srv"])
    assert not torch.allclose(samples_a["dirichlet"]["xyt"], samples_b["dirichlet"]["xyt"])


def test_latin_hypercube_unit_stratifies_each_dimension() -> None:
    _config, _geom, sampler = make_sampler(seed=123, sampling_mode="latin_hypercube")
    unit = sampler._latin_hypercube_unit(128, dim=2)
    assert unit.shape == (128, 2)
    assert np.all(unit >= 0.0)
    assert np.all(unit <= 1.0)
    for axis in range(2):
        strata = np.floor(unit[:, axis] * 128).astype(np.int64)
        assert np.array_equal(np.sort(strata), np.arange(128))


def test_log1p_time_sampling_bounds_uniformity_and_seed() -> None:
    _config, _geom, sampler_a = make_sampler(seed=123)
    _config_b, _geom_b, sampler_b = make_sampler(seed=123)
    sampler_a.cfg["time_strategy"] = "log1p_uniform"
    sampler_b.cfg["time_strategy"] = "log1p_uniform"
    t_a = sampler_a.sample_time(30000)
    t_b = sampler_b.sample_time(30000)
    assert np.allclose(t_a, t_b)
    assert np.all(np.isfinite(t_a))
    assert float(t_a.min()) >= 0.0
    assert float(t_a.max()) <= 1000.0
    log_t = np.log1p(t_a.reshape(-1))
    counts, _ = np.histogram(log_t, bins=10)
    assert np.max(np.abs(counts - counts.mean())) / counts.mean() < 0.12


def test_hybrid_time_sampling_keeps_fixed_time_slices() -> None:
    _config, _geom, sampler = make_sampler(seed=123, sampling_mode="latin_hypercube")
    sampler.cfg["time_strategy"] = "hybrid_log1p_fixed"
    sampler.cfg["time_continuous_fraction"] = 0.5
    sampler.cfg["time_fixed_slices"] = [
        {"time": 1.0, "fraction": 0.1},
        {"time": 100.0, "fraction": 0.1},
        {"time": 500.0, "fraction": 0.1},
        {"time": 1000.0, "fraction": 0.2},
    ]

    t = sampler.sample_time(10000).reshape(-1)
    assert t.shape == (10000,)
    assert np.all(np.isfinite(t))
    assert float(t.min()) >= 0.0
    assert float(t.max()) <= 1000.0
    assert int(np.isclose(t, 1.0, atol=1.0e-5).sum()) == 1000
    assert int(np.isclose(t, 100.0, atol=1.0e-5).sum()) == 1000
    assert int(np.isclose(t, 500.0, atol=1.0e-5).sum()) == 1000
    assert int(np.isclose(t, 1000.0, atol=1.0e-5).sum()) == 2000
