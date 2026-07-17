from __future__ import annotations

import copy

import numpy as np

from src.config import load_config
from src.gas_metrics import (
    build_snapshot_gas_metadata,
    compute_gas_rates,
    compute_initial_inventory_gas,
    compute_isotope_metrics,
    compute_regional_cumulative_gas,
    diagnostics_from_cumulative,
)
from src.pod_utils import save_snapshot_archive
from src.utils import pressure_component_affine_parameters


class DummyGrid:
    def __init__(self, volume: np.ndarray, phi: np.ndarray, region: np.ndarray) -> None:
        self.cell_volume = volume
        self.cell_phi = phi
        self.cell_region = region
        self.cell_xy = np.zeros((volume.size, 2), dtype=np.float64)

    @property
    def num_cells(self) -> int:
        return int(self.cell_volume.size)


def _config() -> dict:
    return copy.deepcopy(load_config("config/default.yaml"))


def _weights(volumes: list[float]) -> np.ndarray:
    weights = np.zeros((3, 3), dtype=np.float64)
    for idx, volume in enumerate(volumes):
        weights[idx, idx] = volume
    return weights


def _pressure_from_p1(config: dict, p1_values: np.ndarray, ratio: float) -> np.ndarray:
    pressure = np.zeros((p1_values.size, 3, 2), dtype=np.float64)
    pressure[:, :, 0] = p1_values.reshape(-1, 1)
    pressure[:, :, 1] = ratio * p1_values.reshape(-1, 1)
    return pressure


def _run_isotope_case(ratio: float) -> dict:
    config = _config()
    config["pressure"]["C13_C12"] = float(ratio)
    times = np.asarray([0.0, 1.0, 2.0, 4.0, 7.0], dtype=np.float64)
    initial = pressure_component_affine_parameters(config)["initial_mpa"][0]
    p1 = initial - np.asarray([0.0, 1.0, 2.0, 4.0, 7.0], dtype=np.float64)
    pressure = _pressure_from_p1(config, p1, ratio)
    fai = np.asarray([0.1, 0.05, 0.05], dtype=np.float64)
    weights = _weights([1.0, 2.0, 3.0])
    cumulative = compute_regional_cumulative_gas(pressure, fai, weights, config)
    rates = compute_gas_rates(times, cumulative, config)
    inventory = compute_initial_inventory_gas(fai, weights, config)
    isotope = compute_isotope_metrics(cumulative, rates, inventory, config)
    return isotope


def test_zero_depletion() -> None:
    config = _config()
    initial = pressure_component_affine_parameters(config)["initial_mpa"]
    pressure = np.tile(initial.reshape(1, 1, 2), (4, 3, 1))
    fai = np.asarray([0.1, 0.05, 0.05], dtype=np.float64)
    weights = _weights([1.0, 2.0, 3.0])
    times = np.asarray([0.0, 1.0, 3.0, 6.0], dtype=np.float64)
    cumulative = compute_regional_cumulative_gas(pressure, fai, weights, config)
    rates = compute_gas_rates(times, cumulative, config)
    inventory = compute_initial_inventory_gas(fai, weights, config)
    isotope = compute_isotope_metrics(cumulative, rates, inventory, config)
    for key, values in cumulative.items():
        assert np.allclose(values, 0.0, atol=1.0e-12), key
    for key, values in rates.items():
        assert np.allclose(values, 0.0, atol=1.0e-12), key
    assert np.all(np.isnan(isotope["delta_prod_permil"]))
    assert np.all(np.isnan(isotope["delta_cum_permil"]))


def test_uniform_isotope_ratio() -> None:
    config = _config()
    ratio = config["gas_postprocessing"]["constants"]["isotope_standard_ratio"]
    isotope = _run_isotope_case(ratio)
    assert np.nanmax(np.abs(isotope["delta_prod_permil"])) < 1.0e-6
    assert np.nanmax(np.abs(isotope["delta_cum_permil"])) < 1.0e-6
    assert np.nanmax(np.abs(isotope["delta_remaining_permil"])) < 1.0e-6


def test_initial_minus_30_permil_ratio() -> None:
    config = _config()
    rst = config["gas_postprocessing"]["constants"]["isotope_standard_ratio"]
    ratio = (1.0 - 30.0 / 1000.0) * rst
    isotope = _run_isotope_case(ratio)
    assert np.nanmax(np.abs(isotope["delta_prod_permil"] + 30.0)) < 1.0e-6
    assert np.nanmax(np.abs(isotope["delta_cum_permil"] + 30.0)) < 1.0e-6
    assert np.nanmax(np.abs(isotope["delta_remaining_permil"] + 30.0)) < 1.0e-6


def test_region_sum_identity() -> None:
    config = _config()
    initial = pressure_component_affine_parameters(config)["initial_mpa"]
    pressure = np.tile(initial.reshape(1, 1, 2), (3, 3, 1))
    pressure[1:] -= 1.0
    cumulative = compute_regional_cumulative_gas(pressure, np.asarray([0.1, 0.05, 0.05]), _weights([1.0, 2.0, 3.0]), config)
    assert np.allclose(cumulative["Q_cum_m3"], cumulative["Q_HF_m3"] + cumulative["Q_SRV_m3"] + cumulative["Q_USRV_m3"])
    assert np.allclose(cumulative["Q1_cum_m3"], cumulative["Q1_HF_m3"] + cumulative["Q1_SRV_m3"] + cumulative["Q1_USRV_m3"])
    assert np.allclose(cumulative["Q2_cum_m3"], cumulative["Q2_HF_m3"] + cumulative["Q2_SRV_m3"] + cumulative["Q2_USRV_m3"])


def test_component_sum_identity() -> None:
    config = _config()
    initial = pressure_component_affine_parameters(config)["initial_mpa"]
    pressure = np.tile(initial.reshape(1, 1, 2), (3, 3, 1))
    pressure[1:] -= 1.0
    cumulative = compute_regional_cumulative_gas(pressure, np.asarray([0.1, 0.05, 0.05]), _weights([1.0, 2.0, 3.0]), config)
    for region in ["HF", "SRV", "USRV"]:
        assert np.allclose(cumulative[f"Q_{region}_m3"], cumulative[f"Q1_{region}_m3"] + cumulative[f"Q2_{region}_m3"])


def test_thickness_correction() -> None:
    config_1m = _config()
    config_1m["grid"]["thickness_m"] = 1.0
    config_8m = _config()
    config_8m["grid"]["thickness_m"] = 8.0
    grid_1m = DummyGrid(np.asarray([1.0]), np.asarray([0.1]), np.asarray(["HF"]))
    grid_8m = DummyGrid(np.asarray([8.0]), np.asarray([0.1]), np.asarray(["HF"]))
    meta_1m = build_snapshot_gas_metadata(config_1m, grid_1m)
    meta_8m = build_snapshot_gas_metadata(config_8m, grid_8m)
    assert np.allclose(meta_1m["regional_integration_weights_m3"], meta_8m["regional_integration_weights_m3"])


def test_no_double_hf_aperture() -> None:
    config = _config()
    config["grid"]["thickness_m"] = 8.0
    grid = DummyGrid(np.asarray([0.16]), np.asarray([0.1]), np.asarray(["HF"]))
    metadata = build_snapshot_gas_metadata(config, grid)
    assert metadata["regional_integration_weights_m3"][0, 0] == 0.16
    assert np.sum(metadata["regional_integration_weights_m3"][0]) == 0.16


def test_nonuniform_time_gradient() -> None:
    config = _config()
    times = np.asarray([0.0, 0.5, 2.0, 5.0, 9.0], dtype=np.float64)
    values = times**2
    cumulative = {}
    for name in ["HF", "SRV", "USRV"]:
        cumulative[f"Q1_{name}_m3"] = values.copy()
        cumulative[f"Q2_{name}_m3"] = values.copy()
        cumulative[f"Q_{name}_m3"] = values.copy()
    cumulative["Q1_cum_m3"] = values.copy()
    cumulative["Q2_cum_m3"] = values.copy()
    cumulative["Q_cum_m3"] = values.copy()
    rates = compute_gas_rates(times, cumulative, config)
    assert np.allclose(rates["rate_cum_m3_per_day"], 2.0 * times, atol=1.0e-10)


def test_snapshot_metadata_roundtrip(tmp_path) -> None:
    config = _config()
    grid = DummyGrid(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.1, 0.05, 0.05]),
        np.asarray(["HF", "SRV", "USRV"]),
    )
    metadata = build_snapshot_gas_metadata(config, grid)
    source = {
        "cell_xy": grid.cell_xy,
        "matrix_cell_count": np.asarray(3, dtype=np.int64),
        "well_cells": np.asarray([0], dtype=np.int64),
        "component_names": np.asarray(["P12", "P13"]),
        "cell_region": grid.cell_region,
        **metadata,
    }
    path = tmp_path / "prediction.npz"
    save_snapshot_archive(path, source, np.asarray([0.0, 1.0]), np.ones((2, 3, 2)), "test")
    with np.load(path, allow_pickle=True) as data:
        for key in ["cell_volume_m3", "cell_fai", "source_grid_thickness_m", "regional_integration_weights_m3", "region_order"]:
            assert key in data.files


def test_no_silent_clamping() -> None:
    config = _config()
    initial = pressure_component_affine_parameters(config)["initial_mpa"]
    pressure = np.tile(initial.reshape(1, 1, 2), (4, 3, 1))
    pressure[1, 0, :] = initial - 1.0
    pressure[2, 0, :] = initial + 1.0
    pressure[3, 0, :] = initial - 0.5
    cumulative = compute_regional_cumulative_gas(pressure, np.asarray([0.1, 0.05, 0.05]), _weights([1.0, 2.0, 3.0]), config)
    diagnostics = diagnostics_from_cumulative(cumulative)
    assert np.any(cumulative["Q_HF_m3"] < 0.0)
    assert diagnostics["negative_Q_HF_point_count"] > 0.0
    assert diagnostics["nonmonotonic_Q_HF_step_count"] > 0.0

