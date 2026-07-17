"""Configuration loading and validation for v11 E-PINN/EDFM."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config file must parse to a dictionary.")
    _expand_generated_vectors(config)
    validate_config(config)
    return config


def _expand_generated_vectors(config: dict[str, Any]) -> None:
    """Expand compact YAML vector specifications into explicit value lists.

    Supported time-grid syntax::

        time_grid:
          time_vector:
            range: [start, stop, step]
            include_initial_zero: true

    The generated values follow MATLAB/Python range semantics: ``stop`` is
    included only when it lies exactly on the arithmetic progression.
    ``include_initial_zero`` is enabled by default because the transient
    E-PINN/FVM solver requires an initial snapshot at t=0 day.
    """

    _expand_time_vector_section(config, "time_grid", "times_days")
    _expand_time_vector_section(config, "evaluation", "times")


def _expand_time_vector_section(config: dict[str, Any], section_name: str, output_key: str) -> None:
    section = config.get(section_name)
    if not isinstance(section, dict):
        return
    time_vector = section.get("time_vector")
    if time_vector is None:
        return
    if not isinstance(time_vector, dict):
        raise ValueError(f"{section_name}.time_vector must be a mapping.")

    range_spec = time_vector.get("range")
    if not isinstance(range_spec, (list, tuple)) or len(range_spec) != 3:
        raise ValueError(
            f"{section_name}.time_vector.range must contain exactly "
            "[start, stop, step]."
        )
    try:
        start, stop, step = (float(value) for value in range_spec)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{section_name}.time_vector.range values must be numeric."
        ) from exc

    values = _inclusive_arange(start, stop, step)
    include_initial_zero = bool(time_vector.get("include_initial_zero", True))
    if include_initial_zero and not math.isclose(values[0], 0.0, abs_tol=1.0e-12):
        if values[0] < 0.0:
            raise ValueError(
                f"{section_name}.time_vector cannot prepend t=0 to a range starting below 0."
            )
        values.insert(0, 0.0)

    # Keep downstream code backward-compatible with the explicit list keys.
    section[output_key] = values


def _inclusive_arange(start: float, stop: float, step: float) -> list[float]:
    """Return an arithmetic progression with stable floating-point limits."""

    if not all(math.isfinite(value) for value in (start, stop, step)):
        raise ValueError("time vector start, stop, and step must be finite.")
    if math.isclose(step, 0.0, abs_tol=0.0):
        raise ValueError("time vector step must be non-zero.")
    if step > 0.0 and stop < start:
        raise ValueError("positive time vector step requires stop >= start.")
    if step < 0.0 and stop > start:
        raise ValueError("negative time vector step requires stop <= start.")

    span = (stop - start) / step
    tolerance = 1.0e-12 * max(1.0, abs(span))
    count = int(math.floor(span + tolerance)) + 1
    if count <= 0:
        raise ValueError("time vector range produced no values.")
    if count > 1_000_000:
        raise ValueError("time vector range exceeds the safety limit of 1,000,000 values.")

    values = [start + index * step for index in range(count)]
    if math.isclose(values[-1], stop, rel_tol=1.0e-12, abs_tol=1.0e-12):
        values[-1] = stop
    return values


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "runtime",
        "geometry",
        "grid",
        "edfm",
        "physics",
        "well",
        "pressure",
        "time_grid",
        "model",
        "training",
        "evaluation",
        "paths",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise KeyError(f"Config missing required sections: {missing}")

    runtime = config["runtime"]
    if str(runtime.get("device", "cpu")).lower() != "cpu":
        raise ValueError("v11 is CPU-only by default; set runtime.device='cpu'.")
    if str(runtime.get("dtype", "float32")).lower() != "float32":
        raise ValueError("v11 currently supports runtime.dtype='float32'.")

    if str(config["physics"].get("mode", "")).lower() != "epinn_edfm_singlephase":
        raise ValueError("v11 requires physics.mode='epinn_edfm_singlephase'.")

    grid = config["grid"]
    if int(grid.get("nx", 0)) <= 0 or int(grid.get("ny", 0)) <= 0:
        raise ValueError("grid.nx and grid.ny must be positive integers.")
    if float(grid.get("thickness_m", 0.0)) <= 0.0:
        raise ValueError("grid.thickness_m must be positive.")

    times = [float(value) for value in config["time_grid"].get("times_days", [])]
    if len(times) < 2:
        raise ValueError("time_grid.times_days must contain at least two values.")
    if times[0] != 0.0:
        raise ValueError("time_grid.times_days must start at 0.0.")
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("time_grid.times_days must be strictly increasing.")

    evaluation_times = [float(value) for value in config["evaluation"].get("times", [])]
    if len(evaluation_times) < 1:
        raise ValueError("evaluation.times must contain at least one value.")
    if not all(math.isfinite(value) for value in evaluation_times):
        raise ValueError("evaluation.times must be finite.")

    pressure = config["pressure"]
    components = list(pressure.get("components", []))
    if len(components) != int(config["model"].get("output_dim", len(components))):
        raise ValueError("pressure.components length must match model.output_dim.")
    if len(components) < 1:
        raise ValueError("pressure.components must contain at least one component.")
    if float(pressure.get("C13_C12", 1.0)) <= 0.0 and len(components) == 2:
        raise ValueError("pressure.C13_C12 must be positive for two pressure components.")
    if int(config["edfm"].get("max_dense_elements", 0)) <= 0:
        raise ValueError("edfm.max_dense_elements must be positive.")
    if float(config["edfm"].get("fracture_tangential_multiplier", 1.0)) <= 0.0:
        raise ValueError("edfm.fracture_tangential_multiplier must be positive.")
    if float(config["physics"].get("transmissibility_scale", 0.0)) <= 0.0:
        raise ValueError("physics.transmissibility_scale must be positive.")
    if float(config["physics"].get("seconds_per_day", 0.0)) <= 0.0:
        raise ValueError("physics.seconds_per_day must be positive.")

    _validate_legacy_diffusion(config, len(components))

    if "pod" in config:
        _validate_pod_config(config["pod"])
    if "gas_postprocessing" in config:
        _validate_gas_postprocessing_config(config["gas_postprocessing"])


def _validate_legacy_diffusion(config: dict[str, Any], component_count: int) -> None:
    physics = config["physics"]
    regions = ["HF", "SRV", "USRV"]
    if "Fai" not in physics:
        raise ValueError("legacy_diffusion requires physics.Fai.")
    for region in regions:
        if float(physics["Fai"].get(region, 0.0)) <= 0.0:
            raise ValueError(f"physics.Fai.{region} must be positive.")

    keys = [str(value) for value in physics.get("diffusivity_keys", [f"D{idx + 1}" for idx in range(component_count)])]
    if len(keys) != component_count:
        raise ValueError("physics.diffusivity_keys length must match pressure.components.")
    for key in keys:
        if key not in physics:
            raise ValueError(f"legacy_diffusion requires physics.{key}.")
        for region in regions:
            if float(physics[key].get(region, 0.0)) <= 0.0:
                raise ValueError(f"physics.{key}.{region} must be positive.")

def _validate_pod_config(pod: dict[str, Any]) -> None:
    snapshot = pod.get("snapshot_generation", {})
    source = str(snapshot.get("source", "direct_fvm")).lower()
    if source not in {"direct_fvm", "pinn"}:
        raise ValueError("pod.snapshot_generation.source must be 'direct_fvm' or 'pinn'.")
    if source == "pinn" and not str(snapshot.get("pinn_snapshot_file", "snapshots.npz")).strip():
        raise ValueError("pod.snapshot_generation.pinn_snapshot_file must not be empty when source is 'pinn'.")

    early_end = float(snapshot.get("early_end_days", 0.0))
    middle_end = float(snapshot.get("middle_end_days", 0.0))
    final_time = float(snapshot.get("final_time_days", 0.0))
    if not (final_time > middle_end > early_end >= 0.0):
        raise ValueError("pod snapshot times must satisfy final_time_days > middle_end_days > early_end_days >= 0.")
    for key in ["early_points", "middle_points", "late_points"]:
        if int(snapshot.get(key, 0)) < 2:
            raise ValueError(f"pod.snapshot_generation.{key} must be at least 2.")

    decomposition = pod.get("decomposition", {})
    if float(decomposition.get("energy_threshold", 0.0)) <= 0.0 or float(decomposition.get("energy_threshold", 0.0)) > 1.0:
        raise ValueError("pod.decomposition.energy_threshold must be in (0, 1].")
    if int(decomposition.get("max_rank", 0)) <= 0:
        raise ValueError("pod.decomposition.max_rank must be positive.")
    if int(decomposition.get("fixed_rank", 0)) <= 0:
        raise ValueError("pod.decomposition.fixed_rank must be positive.")

    split = pod.get("split", {})
    validation_fraction = float(split.get("validation_fraction", 0.0))
    test_fraction = float(split.get("test_fraction", 0.0))
    if validation_fraction < 0.0 or validation_fraction >= 0.5:
        raise ValueError("pod.split.validation_fraction must be in [0, 0.5).")
    if test_fraction < 0.0 or test_fraction >= 0.5:
        raise ValueError("pod.split.test_fraction must be in [0, 0.5).")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("pod split validation_fraction + test_fraction must be less than 0.8.")

    hidden_dims = pod.get("mlp", {}).get("hidden_dims", [])
    if not isinstance(hidden_dims, list) or any(int(value) <= 0 for value in hidden_dims):
        raise ValueError("pod.mlp.hidden_dims must be a list of positive integers.")

    training = pod.get("training", {})
    for key in ["epochs", "batch_size", "patience"]:
        if int(training.get(key, 0)) <= 0:
            raise ValueError(f"pod.training.{key} must be a positive integer.")


def _validate_gas_postprocessing_config(gas: dict[str, Any]) -> None:
    constants = gas.get("constants", {})
    for key in [
        "gas_constant_J_per_mol_K",
        "standard_molar_volume_m3_per_mol",
        "temperature_K",
        "pressure_MPa_to_Pa",
        "model_count",
        "production_scale_divisor",
        "isotope_standard_ratio",
        "reference_reservoir_thickness_m",
    ]:
        if float(constants.get(key, 0.0)) <= 0.0:
            raise ValueError(f"gas_postprocessing.constants.{key} must be positive.")

    order = [str(value) for value in gas.get("regions", {}).get("order", [])]
    if sorted(order) != ["HF", "SRV", "USRV"] or len(order) != 3:
        raise ValueError("gas_postprocessing.regions.order must contain exactly HF, SRV, and USRV.")

    rate = gas.get("rate", {})
    if str(rate.get("method", "")).lower() != "finite_difference":
        raise ValueError("gas_postprocessing.rate.method must be 'finite_difference'.")
    if int(rate.get("edge_order", 0)) not in {1, 2}:
        raise ValueError("gas_postprocessing.rate.edge_order must be 1 or 2.")

    inference = gas.get("inference", {})
    if int(inference.get("number_of_times", 0)) < 3:
        raise ValueError("gas_postprocessing.inference.number_of_times must be at least 3.")
    if float(inference.get("end_time_days", 0.0)) <= float(inference.get("start_time_days", 0.0)):
        raise ValueError("gas_postprocessing.inference.end_time_days must be greater than start_time_days.")
