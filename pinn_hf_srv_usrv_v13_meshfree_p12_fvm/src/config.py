"""Configuration loading and validation for v13."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REGIONS = ("HF", "SRV", "USRV")


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config file must parse to a dictionary.")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "runtime",
        "geometry",
        "physics",
        "boundary",
        "sampler",
        "model",
        "training",
        "loss_weights",
        "grid",
        "edfm",
        "well",
        "pressure",
        "time_grid",
        "evaluation",
        "paths",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise KeyError(f"Config is missing required sections: {missing}")

    runtime_cfg = config["runtime"]
    if str(runtime_cfg.get("device", "cpu")).lower() != "cpu":
        raise ValueError("v13 is CPU-only; set runtime.device to 'cpu'.")
    if str(runtime_cfg.get("dtype", "float64")).lower() != "float64":
        raise ValueError("v13 expects runtime.dtype='float64'.")

    physics_cfg = config["physics"]
    if str(physics_cfg.get("mode", "")).lower() != "meshfree_p12_line_hf":
        raise ValueError("v13 requires physics.mode='meshfree_p12_line_hf'.")
    for group in ["Fai", "D", "permeability_mD"]:
        if group not in physics_cfg:
            raise ValueError(f"physics.{group} is required.")
        values = [float(physics_cfg[group][region]) for region in REGIONS]
        if any(value <= 0.0 for value in values):
            raise ValueError(f"All physics.{group} values must be positive.")
    for group in ["D", "permeability_mD"]:
        values = [float(physics_cfg[group][region]) for region in REGIONS]
        if len(set(values)) != len(values):
            raise ValueError(f"physics.{group} must differ across HF/SRV/USRV for v13.")
    if float(physics_cfg.get("seconds_per_day", 0.0)) <= 0.0:
        raise ValueError("physics.seconds_per_day must be positive.")
    if float(physics_cfg.get("transmissibility_scale", 1.0)) <= 0.0:
        raise ValueError("physics.transmissibility_scale must be positive.")

    pressure_cfg = config["pressure"]
    if list(pressure_cfg.get("components", [])) != ["P12"]:
        raise ValueError("v13 solves a single pressure component: pressure.components must be ['P12'].")

    model_cfg = config["model"]
    if int(model_cfg.get("input_dim", -1)) != 3:
        raise ValueError("model.input_dim must be 3 for x_hat/y_hat/t_hat.")
    if int(model_cfg.get("output_dim", -1)) != 1:
        raise ValueError("model.output_dim must be 1 for single P12.")
    if int(model_cfg.get("subnet_input_dim", 3)) not in {3, 5}:
        raise ValueError("model.subnet_input_dim must be 3 or 5.")
    if bool(model_cfg.get("share_srv_usrv_subnet", False)) and int(model_cfg.get("subnet_input_dim", 3)) != 3:
        raise ValueError("model.share_srv_usrv_subnet requires model.subnet_input_dim=3.")
    constraint_mode = str(model_cfg.get("constraint_mode", "")).lower()
    if constraint_mode not in {"ic_hard", "ic_base_correction"}:
        raise ValueError("model.constraint_mode must be 'ic_hard' or 'ic_base_correction'.")
    if float(model_cfg.get("base_time_lag_days", 0.0)) < 0.0:
        raise ValueError("model.base_time_lag_days must be non-negative.")
    if float(model_cfg.get("correction_envelope_power", 1.0)) <= 0.0:
        raise ValueError("model.correction_envelope_power must be positive.")
    if float(model_cfg.get("correction_scale", 1.0)) <= 0.0:
        raise ValueError("model.correction_scale must be positive.")
    for region, value in model_cfg.get("correction_scale_by_region", {}).items():
        if str(region).upper() not in REGIONS:
            raise ValueError(f"model.correction_scale_by_region has unknown region {region!r}.")
        if float(value) <= 0.0:
            raise ValueError(f"model.correction_scale_by_region.{region} must be positive.")
    if str(model_cfg.get("correction_activation", "tanh")).lower() not in {"tanh", "softsign"}:
        raise ValueError("model.correction_activation must be 'tanh' or 'softsign'.")
    local_tip = model_cfg.get("local_tip_expert", {})
    if local_tip and bool(local_tip.get("enabled", False)):
        if float(local_tip.get("radius_m", 0.0)) <= 0.0:
            raise ValueError("model.local_tip_expert.radius_m must be positive when enabled.")
        if float(local_tip.get("scale", 0.0)) <= 0.0:
            raise ValueError("model.local_tip_expert.scale must be positive when enabled.")
        if float(local_tip.get("gate_power", 1.0)) <= 0.0:
            raise ValueError("model.local_tip_expert.gate_power must be positive when enabled.")
        if int(local_tip.get("hidden_layers", 0)) <= 0:
            raise ValueError("model.local_tip_expert.hidden_layers must be positive when enabled.")
        if int(local_tip.get("hidden_units", 0)) <= 0:
            raise ValueError("model.local_tip_expert.hidden_units must be positive when enabled.")
    local_fracture = model_cfg.get("local_fracture_expert", {})
    if local_fracture and bool(local_fracture.get("enabled", False)):
        if float(local_fracture.get("radius_m", 0.0)) <= 0.0:
            raise ValueError("model.local_fracture_expert.radius_m must be positive when enabled.")
        if float(local_fracture.get("scale", 0.0)) <= 0.0:
            raise ValueError("model.local_fracture_expert.scale must be positive when enabled.")
        if float(local_fracture.get("gate_power", 1.0)) <= 0.0:
            raise ValueError("model.local_fracture_expert.gate_power must be positive when enabled.")
        if int(local_fracture.get("hidden_layers", 0)) <= 0:
            raise ValueError("model.local_fracture_expert.hidden_layers must be positive when enabled.")
        if int(local_fracture.get("hidden_units", 0)) <= 0:
            raise ValueError("model.local_fracture_expert.hidden_units must be positive when enabled.")
    if bool(config["training"].get("freeze_base_for_tip_expert", False)) and not bool(local_tip.get("enabled", False)):
        raise ValueError("training.freeze_base_for_tip_expert requires model.local_tip_expert.enabled=true.")
    if bool(config["training"].get("freeze_base_for_local_experts", False)) and not (
        bool(local_tip.get("enabled", False)) or bool(local_fracture.get("enabled", False))
    ):
        raise ValueError("training.freeze_base_for_local_experts requires at least one local expert to be enabled.")
    hard_dirichlet = model_cfg.get("hard_dirichlet", {})
    if hard_dirichlet and bool(hard_dirichlet.get("enabled", False)):
        if float(hard_dirichlet.get("radius_m", 0.0)) <= 0.0:
            raise ValueError("model.hard_dirichlet.radius_m must be positive when enabled.")
        if float(hard_dirichlet.get("power", 1.0)) <= 0.0:
            raise ValueError("model.hard_dirichlet.power must be positive when enabled.")
    analytic_base = model_cfg.get("analytic_base", {})
    if analytic_base and bool(analytic_base.get("enabled", False)):
        if float(analytic_base.get("length_multiplier", 0.0)) <= 0.0:
            raise ValueError("model.analytic_base.length_multiplier must be positive when enabled.")
        if float(analytic_base.get("min_length_m", 0.0)) <= 0.0:
            raise ValueError("model.analytic_base.min_length_m must be positive when enabled.")
        if float(analytic_base.get("smooth_max_tau", 0.05)) <= 0.0:
            raise ValueError("model.analytic_base.smooth_max_tau must be positive when enabled.")
        if bool(analytic_base.get("endpoint_taper_enabled", False)):
            if float(analytic_base.get("endpoint_taper_m", 0.0)) <= 0.0:
                raise ValueError("model.analytic_base.endpoint_taper_m must be positive when endpoint_taper_enabled=true.")
            if float(analytic_base.get("endpoint_taper_power", 0.0)) <= 0.0:
                raise ValueError("model.analytic_base.endpoint_taper_power must be positive when endpoint_taper_enabled=true.")
        if float(analytic_base.get("main_length_scale", 1.0)) <= 0.0:
            raise ValueError("model.analytic_base.main_length_scale must be positive when enabled.")
        if float(analytic_base.get("secondary_length_scale", 1.0)) <= 0.0:
            raise ValueError("model.analytic_base.secondary_length_scale must be positive when enabled.")
        if float(analytic_base.get("secondary_length_scale_min_factor", 0.05)) <= 0.0:
            raise ValueError("model.analytic_base.secondary_length_scale_min_factor must be positive when enabled.")
        aggregation = str(analytic_base.get("aggregation", "smooth_max")).lower()
        if aggregation not in {"smooth_max", "probabilistic_union", "sum_clamp"}:
            raise ValueError("model.analytic_base.aggregation must be 'smooth_max', 'probabilistic_union', or 'sum_clamp'.")
        matrix_length_mode = str(analytic_base.get("matrix_length_mode", "region")).lower()
        if matrix_length_mode not in {"region", "smooth_srv_usrv", "srv_halo"}:
            raise ValueError("model.analytic_base.matrix_length_mode must be 'region', 'smooth_srv_usrv', or 'srv_halo'.")
        if matrix_length_mode in {"smooth_srv_usrv", "srv_halo"} and float(analytic_base.get("srv_usrv_blend_width_m", 0.0)) <= 0.0:
            raise ValueError("model.analytic_base.srv_usrv_blend_width_m must be positive in smooth_srv_usrv/srv_halo mode.")

    sampler_cfg = config["sampler"]
    sampling_mode = str(sampler_cfg.get("sampling_mode", "random")).lower()
    if sampling_mode not in {"random", "uniform"}:
        raise ValueError("sampler.sampling_mode must be 'random' or 'uniform'.")
    time_sampling_mode = str(sampler_cfg.get("time_sampling_mode", sampling_mode)).lower()
    if time_sampling_mode not in {"random", "uniform"}:
        raise ValueError("sampler.time_sampling_mode must be 'random' or 'uniform'.")
    time_pairing_mode = str(sampler_cfg.get("time_pairing_mode", "paired")).lower()
    if time_pairing_mode not in {"paired", "cartesian"}:
        raise ValueError("sampler.time_pairing_mode must be 'paired' or 'cartesian'.")
    anchor_fraction = float(sampler_cfg.get("time_anchor_fraction", 0.0))
    if not 0.0 <= anchor_fraction <= 1.0:
        raise ValueError("sampler.time_anchor_fraction must be between 0 and 1.")
    anchor_days = [float(value) for value in sampler_cfg.get("time_anchor_days", [])]
    if any(value < float(sampler_cfg["t_min"]) or value > float(sampler_cfg["t_max"]) for value in anchor_days):
        raise ValueError("sampler.time_anchor_days must lie within [t_min, t_max].")
    if time_pairing_mode == "cartesian":
        for key in ["n_time_pde", "n_time_boundary", "n_time_interface", "n_time_link", "n_time_symmetry"]:
            if int(sampler_cfg.get(key, sampler_cfg.get("n_time_collocation", 1))) <= 0:
                raise ValueError(f"sampler.{key} must be positive when time_pairing_mode is 'cartesian'.")
        if int(sampler_cfg.get("n_time_hf_segment_conservation", sampler_cfg.get("n_time_link", 1))) <= 0:
            raise ValueError("sampler.n_time_hf_segment_conservation must be positive when time_pairing_mode is 'cartesian'.")
        if int(sampler_cfg.get("n_time_hf_junction_flux", sampler_cfg.get("n_time_link", 1))) <= 0:
            raise ValueError("sampler.n_time_hf_junction_flux must be positive when time_pairing_mode is 'cartesian'.")
    if int(sampler_cfg.get("n_hf_tip_neumann", 0)) < 0:
        raise ValueError("sampler.n_hf_tip_neumann must be non-negative.")
    if int(sampler_cfg.get("n_near_hf_tip_srv", 0)) < 0:
        raise ValueError("sampler.n_near_hf_tip_srv must be non-negative.")
    if int(sampler_cfg.get("n_near_hf_tip_srv", 0)) > 0 and float(sampler_cfg.get("hf_tip_band_radius_m", 0.0)) <= 0.0:
        raise ValueError("sampler.hf_tip_band_radius_m must be positive when n_near_hf_tip_srv > 0.")
    if int(sampler_cfg.get("n_hf_leakoff_balance", 0)) < 0:
        raise ValueError("sampler.n_hf_leakoff_balance must be non-negative.")
    if int(sampler_cfg.get("n_hf_segment_conservation", 0)) < 0:
        raise ValueError("sampler.n_hf_segment_conservation must be non-negative.")
    if int(sampler_cfg.get("n_hf_junction_flux", 0)) < 0:
        raise ValueError("sampler.n_hf_junction_flux must be non-negative.")
    for key in ["n_symmetry_hf", "n_symmetry_srv", "n_symmetry_usrv"]:
        if int(sampler_cfg.get(key, 0)) < 0:
            raise ValueError(f"sampler.{key} must be non-negative.")
    if float(config["loss_weights"].get("hf_leakoff_balance", 0.0)) > 0.0:
        if int(sampler_cfg.get("n_hf_leakoff_balance", 0)) <= 0:
            raise ValueError("sampler.n_hf_leakoff_balance must be positive when hf_leakoff_balance loss is enabled.")
        if float(sampler_cfg.get("hf_leakoff_offset_m", 0.0)) <= 0.0:
            raise ValueError("sampler.hf_leakoff_offset_m must be positive when hf_leakoff_balance loss is enabled.")
        if float(sampler_cfg.get("hf_leakoff_endpoint_margin_m", 0.0)) < 0.0:
            raise ValueError("sampler.hf_leakoff_endpoint_margin_m must be non-negative.")
    if float(config["loss_weights"].get("hf_segment_conservation", 0.0)) > 0.0:
        if int(sampler_cfg.get("n_hf_segment_conservation", 0)) <= 0:
            raise ValueError("sampler.n_hf_segment_conservation must be positive when hf_segment_conservation loss is enabled.")
        if float(sampler_cfg.get("hf_segment_leakoff_offset_m", 0.0)) <= 0.0:
            raise ValueError("sampler.hf_segment_leakoff_offset_m must be positive when hf_segment_conservation loss is enabled.")
        min_h = float(sampler_cfg.get("hf_segment_min_half_length_m", 0.0))
        max_h = float(sampler_cfg.get("hf_segment_max_half_length_m", 0.0))
        if min_h <= 0.0 or max_h < min_h:
            raise ValueError("sampler HF segment half lengths must satisfy 0 < min <= max.")
        if float(sampler_cfg.get("hf_segment_endpoint_margin_m", 0.0)) < 0.0:
            raise ValueError("sampler.hf_segment_endpoint_margin_m must be non-negative.")
    if float(config["loss_weights"].get("hf_junction_flux", 0.0)) > 0.0:
        if int(sampler_cfg.get("n_hf_junction_flux", 0)) <= 0:
            raise ValueError("sampler.n_hf_junction_flux must be positive when hf_junction_flux loss is enabled.")
        if float(sampler_cfg.get("hf_junction_flux_offset_m", 0.0)) <= 0.0:
            raise ValueError("sampler.hf_junction_flux_offset_m must be positive when hf_junction_flux loss is enabled.")
    if float(config["loss_weights"].get("local_conservation", 0.0)) > 0.0:
        for key in ["n_conservation_srv", "n_conservation_usrv"]:
            if int(sampler_cfg.get(key, 0)) <= 0:
                raise ValueError(f"sampler.{key} must be positive when local_conservation loss is enabled.")
        if int(sampler_cfg.get("n_conservation_hf_tip_srv", 0)) < 0:
            raise ValueError("sampler.n_conservation_hf_tip_srv must be non-negative.")
        if int(sampler_cfg.get("n_conservation_hf_tip_srv", 0)) > 0 and float(sampler_cfg.get("conservation_hf_tip_radius_m", 0.0)) <= 0.0:
            raise ValueError("sampler.conservation_hf_tip_radius_m must be positive when n_conservation_hf_tip_srv > 0.")
        min_h = float(sampler_cfg.get("conservation_min_half_size_m", 0.0))
        max_h = float(sampler_cfg.get("conservation_max_half_size_m", 0.0))
        if min_h <= 0.0 or max_h < min_h:
            raise ValueError("sampler conservation half sizes must satisfy 0 < min <= max.")
        tip_min_h = float(sampler_cfg.get("conservation_hf_tip_min_half_size_m", min_h))
        tip_max_h = float(sampler_cfg.get("conservation_hf_tip_max_half_size_m", max_h))
        if int(sampler_cfg.get("n_conservation_hf_tip_srv", 0)) > 0 and (tip_min_h <= 0.0 or tip_max_h < tip_min_h):
            raise ValueError("sampler HF-tip conservation half sizes must satisfy 0 < min <= max.")
        if float(sampler_cfg.get("conservation_boundary_margin_m", 0.0)) < 0.0:
            raise ValueError("sampler.conservation_boundary_margin_m must be non-negative.")

    if "fvm_teacher" in config.get("loss_weights", {}):
        raise ValueError("v13 pure-physics training forbids loss_weights.fvm_teacher.")
    for key in ["pde_hf", "pde_srv", "pde_usrv"]:
        if float(config.get("loss_weights", {}).get(key, 0.0)) < 0.0:
            raise ValueError(f"loss_weights.{key} must be non-negative.")
    if float(config.get("loss_weights", {}).get("hf_tip_neumann", 0.0)) < 0.0:
        raise ValueError("loss_weights.hf_tip_neumann must be non-negative.")
    if float(config.get("loss_weights", {}).get("hf_leakoff_balance", 0.0)) < 0.0:
        raise ValueError("loss_weights.hf_leakoff_balance must be non-negative.")
    if float(config.get("loss_weights", {}).get("hf_segment_conservation", 0.0)) < 0.0:
        raise ValueError("loss_weights.hf_segment_conservation must be non-negative.")
    if float(config.get("loss_weights", {}).get("hf_junction_flux", 0.0)) < 0.0:
        raise ValueError("loss_weights.hf_junction_flux must be non-negative.")
    if float(config.get("loss_weights", {}).get("symmetry", 0.0)) < 0.0:
        raise ValueError("loss_weights.symmetry must be non-negative.")
    if float(config.get("loss_weights", {}).get("fvm_residual", 0.0)) != 0.0:
        raise ValueError("v13 mesh-free training forbids loss_weights.fvm_residual; use local_conservation instead.")
    if bool(config.get("fvm_residual", {}).get("enabled", False)):
        raise ValueError("v13 mesh-free training forbids fvm_residual.enabled; FVM is comparison-only.")

    causal_cfg = config["training"].get("causal_time_weighting", {})
    if causal_cfg and bool(causal_cfg.get("enabled", False)):
        if int(causal_cfg.get("bins", 0)) <= 0:
            raise ValueError("training.causal_time_weighting.bins must be positive when enabled.")
        if float(causal_cfg.get("epsilon", 0.0)) < 0.0:
            raise ValueError("training.causal_time_weighting.epsilon must be non-negative.")
    gpinn_cfg = config["training"].get("gradient_enhanced_pde", {})
    if gpinn_cfg and bool(gpinn_cfg.get("enabled", False)):
        if int(gpinn_cfg.get("max_points_per_region", 0)) <= 0:
            raise ValueError("training.gradient_enhanced_pde.max_points_per_region must be positive when enabled.")
    if float(config.get("loss_weights", {}).get("gradient_enhanced_pde", 0.0)) < 0.0:
        raise ValueError("loss_weights.gradient_enhanced_pde must be non-negative.")
    adaptive_cfg = config["training"].get("adaptive_resampling", {})
    if adaptive_cfg and bool(adaptive_cfg.get("enabled", False)):
        if int(adaptive_cfg.get("every", 0)) <= 0:
            raise ValueError("training.adaptive_resampling.every must be positive when enabled.")
        if int(adaptive_cfg.get("candidate_multiplier", 0)) <= 0:
            raise ValueError("training.adaptive_resampling.candidate_multiplier must be positive when enabled.")
        for key in ["keep_hf", "keep_srv", "keep_usrv", "keep_conservation_srv", "keep_conservation_usrv"]:
            if int(adaptive_cfg.get(key, 0)) < 0:
                raise ValueError(f"training.adaptive_resampling.{key} must be non-negative.")
    validation_cfg = config["training"].get("validation", {})
    if validation_cfg and bool(validation_cfg.get("enabled", False)):
        if int(validation_cfg.get("every", 0)) <= 0:
            raise ValueError("training.validation.every must be positive when enabled.")
        selection_metric = str(validation_cfg.get("selection_metric", "total")).lower()
        if selection_metric not in {"total", "diagnostic"}:
            raise ValueError("training.validation.selection_metric must be 'total' or 'diagnostic'.")
    early_cfg = config["training"].get("early_stopping", {})
    if early_cfg and bool(early_cfg.get("enabled", False)):
        if int(early_cfg.get("patience", 0)) <= 0:
            raise ValueError("training.early_stopping.patience must be positive when enabled.")
        if float(early_cfg.get("min_delta", 0.0)) < 0.0:
            raise ValueError("training.early_stopping.min_delta must be non-negative.")
    lr_cfg = config["training"].get("lr_scheduler", {})
    if lr_cfg and str(lr_cfg.get("type", "constant")).lower() not in {"constant", "cosine"}:
        raise ValueError("training.lr_scheduler.type must be 'constant' or 'cosine'.")

    for key in ["nx", "ny"]:
        if int(config["grid"][key]) <= 0:
            raise ValueError(f"grid.{key} must be positive.")
    times = [float(value) for value in config["time_grid"]["times_days"]]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("time_grid.times_days must be strictly increasing.")
