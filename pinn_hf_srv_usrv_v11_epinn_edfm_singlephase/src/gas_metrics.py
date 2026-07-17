"""Regional gas production and methane isotope post-processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .edfm_grid import EdfmGrid, build_edfm_grid
from .geometry import ReservoirGeometry
from .utils import pressure_component_affine_parameters


GAS_METRICS_VERSION = "pod_mlp_gas_metrics_v1"
REGIONAL_QUANTITY_DEFINITION = "regional_inventory_depletion"
REGION_ORDER = ["HF", "SRV", "USRV"]
CONSISTENCY_TOL = 1.0e-9


DEFAULT_GAS_CONSTANTS = {
    "gas_constant_J_per_mol_K": 8.3145,
    "standard_molar_volume_m3_per_mol": 0.0224,
    "temperature_K": 333.15,
    "pressure_MPa_to_Pa": 1.0e6,
    "model_count": 20.0,
    "production_scale_divisor": 1.0e4,
    "isotope_standard_ratio": 0.0112372,
    "reference_reservoir_thickness_m": 8.0,
}


@dataclass
class GasPostprocessingData:
    times_days: np.ndarray
    pressure_mpa: np.ndarray
    cell_region: np.ndarray
    cell_fai: np.ndarray
    cell_volume_m3: np.ndarray
    regional_integration_weights_m3: np.ndarray
    source_grid_thickness_m: float
    component_names: list[str]
    well_cells: np.ndarray
    region_order: list[str]


def gas_constants(config: dict[str, Any]) -> dict[str, float]:
    constants = dict(DEFAULT_GAS_CONSTANTS)
    constants.update({key: float(value) for key, value in config.get("gas_postprocessing", {}).get("constants", {}).items()})
    return constants


def gas_region_order(config: dict[str, Any]) -> list[str]:
    order = [str(value) for value in config.get("gas_postprocessing", {}).get("regions", {}).get("order", REGION_ORDER)]
    if order != REGION_ORDER:
        raise ValueError("Gas post-processing currently requires region order HF, SRV, USRV.")
    return order


def gas_conversion_factor(config: dict[str, Any]) -> float:
    constants = gas_constants(config)
    return (
        constants["pressure_MPa_to_Pa"]
        * constants["standard_molar_volume_m3_per_mol"]
        * constants["model_count"]
        / (
            constants["gas_constant_J_per_mol_K"]
            * constants["temperature_K"]
            * constants["production_scale_divisor"]
        )
    )


def compute_component_initial_pressures(config: dict[str, Any]) -> np.ndarray:
    params = pressure_component_affine_parameters(config)
    initial = np.asarray(params["initial_mpa"], dtype=np.float64).reshape(-1)
    if initial.size != 2:
        raise ValueError("Gas post-processing requires exactly two pressure components.")
    return initial


def build_snapshot_gas_metadata(config: dict[str, Any], grid: EdfmGrid) -> dict[str, np.ndarray]:
    order = gas_region_order(config)
    constants = gas_constants(config)
    source_thickness = float(config["grid"]["thickness_m"])
    if source_thickness <= 0.0:
        raise ValueError("grid.thickness_m must be positive.")
    correction = constants["reference_reservoir_thickness_m"] / source_thickness
    physical_volume = np.asarray(grid.cell_volume, dtype=np.float64) * correction
    region = np.asarray(grid.cell_region).astype(str)
    weights = np.zeros((grid.num_cells, len(order)), dtype=np.float64)
    for idx, name in enumerate(order):
        weights[region == name, idx] = physical_volume[region == name]
    _validate_region_weights(weights, physical_volume)
    return {
        "cell_volume_m3": np.asarray(grid.cell_volume, dtype=np.float64),
        "cell_fai": np.asarray(grid.cell_phi, dtype=np.float64),
        "source_grid_thickness_m": np.asarray(source_thickness, dtype=np.float64),
        "regional_quantity_definition": np.asarray(REGIONAL_QUANTITY_DEFINITION),
        "region_order": np.asarray(order),
        "regional_integration_weights_m3": weights,
    }


def load_gas_postprocessing_data(config: dict[str, Any], snapshot_archive: str | Path | dict[str, np.ndarray]) -> GasPostprocessingData:
    snapshots = _load_snapshot_dict(snapshot_archive)
    required = ["times_days", "pressure_mpa", "cell_xy", "cell_region", "component_names", "well_cells"]
    missing = [key for key in required if key not in snapshots]
    if missing:
        raise KeyError(f"Snapshot archive missing required gas post-processing keys: {missing}")
    metadata = _metadata_from_archive_or_grid(config, snapshots)
    times = np.asarray(snapshots["times_days"], dtype=np.float64)
    pressure = np.asarray(snapshots["pressure_mpa"], dtype=np.float64)
    cell_region = np.asarray(snapshots["cell_region"]).astype(str)
    component_names = _as_string_list(snapshots["component_names"])
    well_cells = np.asarray(snapshots["well_cells"], dtype=np.int64).reshape(-1)
    if not np.all(np.isfinite(times)) or times.size < 1:
        raise ValueError("times_days must be finite and non-empty.")
    if times.size > 1 and np.any(np.diff(times) <= 0.0):
        raise ValueError("times_days must be strictly increasing.")
    if pressure.ndim != 3 or pressure.shape[2] != 2:
        raise ValueError(f"pressure_mpa must have shape [T, N, 2], got {pressure.shape}.")
    if pressure.shape[0] != times.size:
        raise ValueError("pressure_mpa time dimension does not match times_days.")
    if pressure.shape[1] != cell_region.size:
        raise ValueError("pressure_mpa cell dimension does not match cell_region.")
    _assert_finite("gas pressure_mpa", pressure)
    cell_fai = np.asarray(metadata["cell_fai"], dtype=np.float64)
    cell_volume = np.asarray(metadata["cell_volume_m3"], dtype=np.float64)
    weights = np.asarray(metadata["regional_integration_weights_m3"], dtype=np.float64)
    source_thickness = float(np.asarray(metadata["source_grid_thickness_m"]).item())
    order = _as_string_list(metadata["region_order"])
    _validate_region_order(order)
    if bool(config.get("gas_postprocessing", {}).get("behavior", {}).get("exclude_well_cells_from_inventory", False)):
        weights = weights.copy()
        weights[well_cells, :] = 0.0
    return GasPostprocessingData(
        times_days=times,
        pressure_mpa=pressure,
        cell_region=cell_region,
        cell_fai=cell_fai,
        cell_volume_m3=cell_volume,
        regional_integration_weights_m3=weights,
        source_grid_thickness_m=source_thickness,
        component_names=component_names,
        well_cells=well_cells,
        region_order=order,
    )


def compute_regional_cumulative_gas(
    pressure_mpa: np.ndarray,
    cell_fai: np.ndarray,
    regional_integration_weights_m3: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    pressure = np.asarray(pressure_mpa, dtype=np.float64)
    fai = np.asarray(cell_fai, dtype=np.float64).reshape(-1)
    weights = np.asarray(regional_integration_weights_m3, dtype=np.float64)
    _validate_cumulative_inputs(pressure, fai, weights)
    initial = compute_component_initial_pressures(config)
    factor = gas_conversion_factor(config)
    depletion = initial.reshape(1, 1, 2) - pressure
    q_component_region = factor * np.einsum("tnc,n,nr->tcr", depletion, fai, weights, optimize=True)
    result: dict[str, np.ndarray] = {}
    for region_idx, region in enumerate(REGION_ORDER):
        q1 = q_component_region[:, 0, region_idx]
        q2 = q_component_region[:, 1, region_idx]
        result[f"Q1_{region}_m3"] = q1
        result[f"Q2_{region}_m3"] = q2
        result[f"Q_{region}_m3"] = q1 + q2
    result["Q1_cum_m3"] = result["Q1_HF_m3"] + result["Q1_SRV_m3"] + result["Q1_USRV_m3"]
    result["Q2_cum_m3"] = result["Q2_HF_m3"] + result["Q2_SRV_m3"] + result["Q2_USRV_m3"]
    result["Q_cum_m3"] = result["Q_HF_m3"] + result["Q_SRV_m3"] + result["Q_USRV_m3"]
    _check_cumulative_identities(result)
    initial_abs = float(max(np.max(np.abs(result[key][0])) for key in _cumulative_keys()))
    print(f"initial cumulative gas absolute max m3: {initial_abs:.6e}")
    return result


def compute_initial_inventory_gas(
    cell_fai: np.ndarray,
    regional_integration_weights_m3: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    fai = np.asarray(cell_fai, dtype=np.float64).reshape(-1)
    weights = np.asarray(regional_integration_weights_m3, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != 3:
        raise ValueError("regional_integration_weights_m3 must have shape [N, 3].")
    if fai.shape[0] != weights.shape[0]:
        raise ValueError("cell_fai shape must match regional weights.")
    _assert_finite("cell_fai", fai)
    _assert_finite("regional_integration_weights_m3", weights)
    initial = compute_component_initial_pressures(config)
    factor = gas_conversion_factor(config)
    region_weighted = np.einsum("n,nr->r", fai, weights, optimize=True)
    inventory = factor * initial.reshape(2, 1) * region_weighted.reshape(1, 3)
    result: dict[str, np.ndarray | float] = {}
    for idx, region in enumerate(REGION_ORDER):
        result[f"G1_initial_{region}_m3"] = float(inventory[0, idx])
        result[f"G2_initial_{region}_m3"] = float(inventory[1, idx])
    result["G1_initial_total_m3"] = float(np.sum(inventory[0]))
    result["G2_initial_total_m3"] = float(np.sum(inventory[1]))
    return result


def compute_gas_rates(
    times_days: np.ndarray,
    cumulative_results: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    times = np.asarray(times_days, dtype=np.float64).reshape(-1)
    if times.size < 3 or np.unique(times).size != times.size or np.any(np.diff(times) <= 0.0):
        raise ValueError("Gas rate calculation requires at least 3 strictly increasing unique times.")
    _assert_finite("rate times_days", times)
    edge_order = int(config.get("gas_postprocessing", {}).get("rate", {}).get("edge_order", 2))
    if edge_order not in {1, 2}:
        raise ValueError("rate edge_order must be 1 or 2.")
    result: dict[str, np.ndarray] = {}
    for region in REGION_ORDER:
        result[f"rate1_{region}_m3_per_day"] = _gradient(cumulative_results[f"Q1_{region}_m3"], times, edge_order)
        result[f"rate2_{region}_m3_per_day"] = _gradient(cumulative_results[f"Q2_{region}_m3"], times, edge_order)
        result[f"rate_{region}_m3_per_day"] = _gradient(cumulative_results[f"Q_{region}_m3"], times, edge_order)
    result["rate1_cum_m3_per_day"] = _gradient(cumulative_results["Q1_cum_m3"], times, edge_order)
    result["rate2_cum_m3_per_day"] = _gradient(cumulative_results["Q2_cum_m3"], times, edge_order)
    result["rate_cum_m3_per_day"] = _gradient(cumulative_results["Q_cum_m3"], times, edge_order)
    return result


def compute_isotope_metrics(
    cumulative_results: dict[str, np.ndarray],
    rate_results: dict[str, np.ndarray],
    initial_inventory_results: dict[str, np.ndarray | float],
    config: dict[str, Any],
) -> dict[str, np.ndarray | float]:
    constants = gas_constants(config)
    rst = constants["isotope_standard_ratio"]
    min_light_rate = float(config.get("gas_postprocessing", {}).get("rate", {}).get("minimum_light_rate_m3_per_day", 1.0e-12))
    q1 = np.asarray(cumulative_results["Q1_cum_m3"], dtype=np.float64)
    q2 = np.asarray(cumulative_results["Q2_cum_m3"], dtype=np.float64)
    rate1 = np.asarray(rate_results["rate1_cum_m3_per_day"], dtype=np.float64)
    rate2 = np.asarray(rate_results["rate2_cum_m3_per_day"], dtype=np.float64)
    delta_prod = np.full_like(rate1, np.nan, dtype=np.float64)
    valid_prod = rate1 > min_light_rate
    delta_prod[valid_prod] = ((rate2[valid_prod] / rate1[valid_prod]) / rst - 1.0) * 1000.0
    delta_cum = np.full_like(q1, np.nan, dtype=np.float64)
    valid_cum = q1 > 1.0e-12
    delta_cum[valid_cum] = ((q2[valid_cum] / q1[valid_cum]) / rst - 1.0) * 1000.0

    g1_initial_total = float(initial_inventory_results["G1_initial_total_m3"])
    g2_initial_total = float(initial_inventory_results["G2_initial_total_m3"])
    g1_remaining = g1_initial_total - q1
    g2_remaining = g2_initial_total - q2
    delta_remaining = np.full_like(q1, np.nan, dtype=np.float64)
    valid_remaining = g1_remaining > 1.0e-12
    delta_remaining[valid_remaining] = ((g2_remaining[valid_remaining] / g1_remaining[valid_remaining]) / rst - 1.0) * 1000.0

    q_total = np.asarray(cumulative_results["Q_cum_m3"], dtype=np.float64)
    fractions: dict[str, np.ndarray] = {}
    valid_fraction = np.abs(q_total) > 1.0e-12
    for region in REGION_ORDER:
        fraction = np.full_like(q_total, np.nan, dtype=np.float64)
        fraction[valid_fraction] = np.asarray(cumulative_results[f"Q_{region}_m3"], dtype=np.float64)[valid_fraction] / q_total[valid_fraction]
        fractions[f"fraction_{region}"] = fraction

    result: dict[str, np.ndarray | float] = dict(initial_inventory_results)
    result.update(
        {
            "delta_prod_permil": delta_prod,
            "delta_cum_permil": delta_cum,
            "delta_remaining_permil": delta_remaining,
            "G1_remaining_total_m3": g1_remaining,
            "G2_remaining_total_m3": g2_remaining,
        }
    )
    result.update(fractions)
    return result


def diagnostics_from_cumulative(cumulative_results: dict[str, np.ndarray]) -> dict[str, float]:
    diagnostics: dict[str, float] = {}
    for region in REGION_ORDER:
        values = np.asarray(cumulative_results[f"Q_{region}_m3"], dtype=np.float64)
        diagnostics[f"negative_Q_{region}_point_count"] = float(np.sum(values < 0.0))
        diagnostics[f"nonmonotonic_Q_{region}_step_count"] = float(np.sum(np.diff(values) < 0.0))
    q_cum = np.asarray(cumulative_results["Q_cum_m3"], dtype=np.float64)
    diagnostics["nonmonotonic_Q_cum_step_count"] = float(np.sum(np.diff(q_cum) < 0.0))
    diagnostics["initial_Q_absolute_max_m3"] = float(max(np.max(np.abs(cumulative_results[key][0])) for key in _cumulative_keys()))
    return diagnostics


def _metadata_from_archive_or_grid(config: dict[str, Any], snapshots: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    keys = ["cell_volume_m3", "cell_fai", "source_grid_thickness_m", "regional_integration_weights_m3", "region_order"]
    if all(key in snapshots for key in keys):
        metadata = {key: snapshots[key] for key in keys}
        weights = np.asarray(metadata["regional_integration_weights_m3"], dtype=np.float64)
        if weights.shape[0] != np.asarray(snapshots["cell_xy"]).shape[0]:
            raise ValueError("regional_integration_weights_m3 cell count does not match cell_xy.")
        _validate_region_order(_as_string_list(metadata["region_order"]))
        _assert_finite("regional_integration_weights_m3", weights)
        return metadata

    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    cell_xy = np.asarray(snapshots["cell_xy"], dtype=np.float64)
    cell_region = np.asarray(snapshots["cell_region"]).astype(str)
    if grid.num_cells != cell_xy.shape[0]:
        raise ValueError("Snapshot lacks gas metadata and does not match current grid cell count.")
    if not np.allclose(grid.cell_xy, cell_xy, atol=1.0e-8, rtol=0.0):
        raise ValueError("Snapshot lacks gas metadata and cell_xy does not match the rebuilt grid.")
    if not np.array_equal(grid.cell_region.astype(str), cell_region):
        raise ValueError("Snapshot lacks gas metadata and cell_region does not match the rebuilt grid.")
    return build_snapshot_gas_metadata(config, grid)


def _load_snapshot_dict(snapshot_archive: str | Path | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(snapshot_archive, dict):
        return snapshot_archive
    path = Path(snapshot_archive)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot archive does not exist: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _validate_cumulative_inputs(pressure: np.ndarray, fai: np.ndarray, weights: np.ndarray) -> None:
    if pressure.ndim != 3 or pressure.shape[2] != 2:
        raise ValueError(f"pressure_mpa must have shape [T, N, 2], got {pressure.shape}.")
    if fai.shape != (pressure.shape[1],):
        raise ValueError("cell_fai must have shape [N].")
    if weights.shape != (pressure.shape[1], 3):
        raise ValueError("regional_integration_weights_m3 must have shape [N, 3].")
    _assert_finite("pressure_mpa", pressure)
    _assert_finite("cell_fai", fai)
    _assert_finite("regional_integration_weights_m3", weights)
    if np.any(weights < 0.0):
        raise ValueError("regional_integration_weights_m3 must be nonnegative.")


def _validate_region_weights(weights: np.ndarray, physical_volume: np.ndarray) -> None:
    row_sum = np.sum(weights, axis=1)
    if not np.allclose(row_sum, physical_volume, rtol=1.0e-12, atol=1.0e-12):
        raise ValueError("Regional integration weights do not sum to physical cell volume.")


def _validate_region_order(order: list[str]) -> None:
    if order != REGION_ORDER:
        raise ValueError("Region order must be exactly HF, SRV, USRV.")


def _check_cumulative_identities(result: dict[str, np.ndarray]) -> None:
    checks = {
        "Q_cum regional identity": result["Q_HF_m3"] + result["Q_SRV_m3"] + result["Q_USRV_m3"] - result["Q_cum_m3"],
        "Q1_cum regional identity": result["Q1_HF_m3"] + result["Q1_SRV_m3"] + result["Q1_USRV_m3"] - result["Q1_cum_m3"],
        "Q2_cum regional identity": result["Q2_HF_m3"] + result["Q2_SRV_m3"] + result["Q2_USRV_m3"] - result["Q2_cum_m3"],
        "Q_cum component identity": result["Q1_cum_m3"] + result["Q2_cum_m3"] - result["Q_cum_m3"],
    }
    for region in REGION_ORDER:
        checks[f"Q_{region} component identity"] = result[f"Q1_{region}_m3"] + result[f"Q2_{region}_m3"] - result[f"Q_{region}_m3"]
    for name, values in checks.items():
        if not np.allclose(values, 0.0, rtol=CONSISTENCY_TOL, atol=CONSISTENCY_TOL):
            raise ValueError(f"Cumulative gas consistency check failed: {name}.")


def _gradient(values: np.ndarray, times: np.ndarray, edge_order: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    _assert_finite("cumulative values", array)
    return np.asarray(np.gradient(array, times, edge_order=edge_order), dtype=np.float64)


def _cumulative_keys() -> list[str]:
    return [
        "Q1_HF_m3",
        "Q2_HF_m3",
        "Q_HF_m3",
        "Q1_SRV_m3",
        "Q2_SRV_m3",
        "Q_SRV_m3",
        "Q1_USRV_m3",
        "Q2_USRV_m3",
        "Q_USRV_m3",
        "Q1_cum_m3",
        "Q2_cum_m3",
        "Q_cum_m3",
    ]


def _as_string_list(values: Any) -> list[str]:
    array = np.asarray(values).reshape(-1)
    result: list[str] = []
    for value in array:
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def _assert_finite(name: str, values: np.ndarray) -> None:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        bad_count = int(np.sum(~np.isfinite(array)))
        raise ValueError(f"{name} contains {bad_count} NaN or Inf values.")

