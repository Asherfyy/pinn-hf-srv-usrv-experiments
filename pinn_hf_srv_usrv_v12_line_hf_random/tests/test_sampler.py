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


def make_sampler(seed: int = 2026, sampling_mode: str = "random", time_pairing_mode: str = "cartesian") -> tuple[dict, ReservoirGeometry, ReservoirSampler]:
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
            "n_hf_main_link": 64,
            "n_hf_secondary_link": 64,
            "n_hf_junction": 64,
            "junction_offset_m": 0.001,
            "sampling_mode": sampling_mode,
            "time_sampling_mode": sampling_mode,
            "time_pairing_mode": time_pairing_mode,
            "n_time_pde": 4,
            "n_time_boundary": 3,
            "n_time_interface": 5,
            "n_time_link": 4,
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


def test_cartesian_time_product_shapes() -> None:
    config, _geom, sampler = make_sampler()
    pde = sampler.sample_pde_points()
    n_time_pde = int(config["sampler"]["n_time_pde"])
    expected_srv_space = int(config["sampler"]["n_pde_srv"]) + int(config["sampler"]["n_near_hf_srv"]) + int(config["sampler"]["n_near_srv_usrv"]) // 2
    expected_usrv_space = int(config["sampler"]["n_pde_usrv"]) + int(config["sampler"]["n_near_srv_usrv"]) - int(config["sampler"]["n_near_srv_usrv"]) // 2
    assert pde["hf"].shape == (int(config["sampler"]["n_pde_hf"]) * n_time_pde, 3)
    assert pde["srv"].shape == (expected_srv_space * n_time_pde, 3)
    assert pde["usrv"].shape == (expected_usrv_space * n_time_pde, 3)

    dirichlet = sampler.sample_dirichlet_boundary_points()
    assert dirichlet["xyt"].shape == (int(config["sampler"]["n_dirichlet"]) * int(config["sampler"]["n_time_boundary"]), 3)

    interface = sampler.sample_hf_srv_interface_points()
    assert interface["xyt"].shape[0] == interface["normal"].shape[0]

    secondary = sampler.sample_hf_secondary_link_points()
    assert secondary["xyt"].shape == secondary["junction_xyt"].shape
    assert torch.allclose(secondary["xyt"][:, 2], secondary["junction_xyt"][:, 2])


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
    line = hf_srv["xyt"].detach().cpu().numpy()
    normal = hf_srv["normal"].detach().cpu().numpy()
    eps = float(config["sampler"]["eps_hf_srv"])
    srv_xy = line[:, :2] + eps * normal
    assert np.all(region_ids(geom, hf_srv["xyt"]) == REGION_HF)
    assert np.all(geom.region_id_np(srv_xy[:, 0], srv_xy[:, 1]) == REGION_SRV)

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


def test_hf_main_link_points_are_on_main_fracture_centerline() -> None:
    _config, geom, sampler = make_sampler()
    points = sampler.sample_hf_main_link_points()["xyt"]
    arr = points.detach().cpu().numpy()
    y_center = 0.5 * (geom.main_frac.y_min + geom.main_frac.y_max)
    assert np.all(region_ids(geom, points) == REGION_HF)
    assert np.all(arr[:, 0] >= geom.main_frac.x_min)
    assert np.all(arr[:, 0] <= geom.main_frac.x_max)
    assert np.allclose(arr[:, 1], y_center)


def test_hf_main_link_is_random_in_random_mode() -> None:
    _config, _geom, sampler_a = make_sampler(seed=123, sampling_mode="random")
    _config_b, _geom_b, sampler_b = make_sampler(seed=999, sampling_mode="random")
    points_a = np.sort(sampler_a.sample_hf_main_link_points()["xyt"].detach().cpu().numpy()[:, 0])
    points_b = np.sort(sampler_b.sample_hf_main_link_points()["xyt"].detach().cpu().numpy()[:, 0])
    assert not np.allclose(points_a, points_b)


def test_hf_secondary_link_points_are_on_secondary_centerlines() -> None:
    _config, geom, sampler = make_sampler()
    samples = sampler.sample_hf_secondary_link_points()
    points = samples["xyt"]
    junction = samples["junction_xyt"]
    arr = points.detach().cpu().numpy()
    junction_arr = junction.detach().cpu().numpy()
    secondary_centers = np.array([0.5 * (rect.x_min + rect.x_max) for rect in geom.secondary_fractures])
    y_junction = 0.5 * (geom.main_frac.y_min + geom.main_frac.y_max)
    assert points.shape == junction.shape
    assert np.all(region_ids(geom, points) == REGION_HF)
    assert np.all(region_ids(geom, junction) == REGION_HF)
    assert np.all(np.any(np.isclose(arr[:, 0:1], secondary_centers.reshape(1, -1)), axis=1))
    assert not np.any((arr[:, 1] >= geom.main_frac.y_min) & (arr[:, 1] <= geom.main_frac.y_max))
    assert np.allclose(junction_arr[:, 0], arr[:, 0])
    assert np.allclose(junction_arr[:, 1], y_junction)
    assert np.allclose(junction_arr[:, 2], arr[:, 2])


def test_hf_junction_pair_points_connect_main_and_secondary_sides() -> None:
    _config, geom, sampler = make_sampler()
    samples = sampler.sample_hf_junction_pair_points()
    main = samples["main_xyt"]
    secondary = samples["secondary_xyt"]
    main_arr = main.detach().cpu().numpy()
    secondary_arr = secondary.detach().cpu().numpy()
    y_center = 0.5 * (geom.main_frac.y_min + geom.main_frac.y_max)
    secondary_centers = np.array([0.5 * (rect.x_min + rect.x_max) for rect in geom.secondary_fractures])
    assert main.shape == secondary.shape
    assert np.all(region_ids(geom, main) == REGION_HF)
    assert np.all(region_ids(geom, secondary) == REGION_HF)
    assert np.allclose(main_arr[:, 1], y_center)
    assert np.all(np.any(np.isclose(secondary_arr[:, 0:1], secondary_centers.reshape(1, -1)), axis=1))
    assert np.allclose(main_arr[:, 2], secondary_arr[:, 2])


def test_uniform_sampling_mode_is_deterministic_across_seeds() -> None:
    _config_a, _geom_a, sampler_a = make_sampler(seed=123, sampling_mode="uniform")
    _config_b, _geom_b, sampler_b = make_sampler(seed=999, sampling_mode="uniform")
    samples_a = sampler_a.sample_all()
    samples_b = sampler_b.sample_all()
    for region_name in ["hf", "srv", "usrv"]:
        assert torch.allclose(samples_a["pde"][region_name], samples_b["pde"][region_name])
    assert torch.allclose(samples_a["dirichlet"]["xyt"], samples_b["dirichlet"]["xyt"])
    assert torch.allclose(samples_a["hf_secondary_link"]["xyt"], samples_b["hf_secondary_link"]["xyt"])


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
