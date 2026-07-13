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


def make_sampler(seed: int = 2026) -> tuple[dict, ReservoirGeometry, ReservoirSampler]:
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
        }
    )
    geom = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geom, config["sampler"], torch.device("cpu"), get_torch_dtype(config["runtime"]["dtype"]), seed=seed)
    return config, geom, sampler


def region_ids(geom: ReservoirGeometry, xyt: torch.Tensor) -> np.ndarray:
    arr = xyt.detach().cpu().numpy()
    return geom.region_id_np(arr[:, 0], arr[:, 1])


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


def test_log1p_time_sampling_bounds_uniformity_and_seed() -> None:
    _config, _geom, sampler_a = make_sampler(seed=123)
    _config_b, _geom_b, sampler_b = make_sampler(seed=123)
    t_a = sampler_a.sample_time(30000)
    t_b = sampler_b.sample_time(30000)
    assert np.allclose(t_a, t_b)
    assert np.all(np.isfinite(t_a))
    assert float(t_a.min()) >= 0.0
    assert float(t_a.max()) <= 1000.0
    log_t = np.log1p(t_a.reshape(-1))
    counts, _ = np.histogram(log_t, bins=10)
    assert np.max(np.abs(counts - counts.mean())) / counts.mean() < 0.12
