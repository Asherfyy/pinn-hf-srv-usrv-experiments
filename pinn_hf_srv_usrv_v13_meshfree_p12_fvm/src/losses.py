"""Loss terms for the v13 mesh-free single-pressure PINN."""

from __future__ import annotations

from typing import Any

import torch

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry
from .local_conservation import compute_local_conservation_loss
from .physics import dimensionless_pde_coefficients, interface_residual, line_hf_junction_flux_residual, line_hf_leakoff_balance_residual, line_hf_segment_conservation_residual, line_hf_srv_residual, line_pde_residual, line_tangential_derivative, neumann_normal_derivative, pde_residual, pde_residual_spatial_gradient
from .utils import dirichlet_target_hat


def _mse(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.sum() * 0.0
    return torch.mean(value**2)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(_mse(value), min=0.0))


def _points_xyt(points: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(points, dict):
        return points["xyt"]
    return points


def _causal_mse(value: torch.Tensor, xyt: torch.Tensor, config: dict[str, Any], prefix: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if value.numel() == 0:
        zero = value.sum() * 0.0
        return zero, {f"{prefix}_causal_weight_mean": zero}
    causal_cfg = config["training"].get("causal_time_weighting", {})
    if not bool(causal_cfg.get("enabled", False)):
        return _mse(value), {}

    n_bins = int(causal_cfg.get("bins", 8))
    eps = float(causal_cfg.get("epsilon", 1.0))
    t = xyt[:, 2:3]
    sampler_cfg = config["sampler"]
    t_min = float(sampler_cfg["t_min"])
    t_max = float(sampler_cfg["t_max"])
    if bool(causal_cfg.get("log_time", True)):
        coord = torch.log1p(torch.clamp(t, min=0.0))
        lo = torch.as_tensor(torch.log1p(torch.as_tensor(t_min, dtype=t.dtype, device=t.device)), dtype=t.dtype, device=t.device)
        hi = torch.as_tensor(torch.log1p(torch.as_tensor(t_max, dtype=t.dtype, device=t.device)), dtype=t.dtype, device=t.device)
    else:
        coord = t
        lo = torch.as_tensor(t_min, dtype=t.dtype, device=t.device)
        hi = torch.as_tensor(t_max, dtype=t.dtype, device=t.device)
    bins = torch.linspace(lo.item(), hi.item(), n_bins + 1, dtype=t.dtype, device=t.device)
    bin_losses: list[torch.Tensor] = []
    for idx in range(n_bins):
        if idx == n_bins - 1:
            mask = (coord >= bins[idx]) & (coord <= bins[idx + 1])
        else:
            mask = (coord >= bins[idx]) & (coord < bins[idx + 1])
        if torch.any(mask):
            bin_losses.append(_mse(value[mask.view(-1)]))
        else:
            bin_losses.append(value.sum() * 0.0)
    losses = torch.stack(bin_losses)
    previous = torch.cumsum(losses.detach(), dim=0) - losses.detach()
    weights = torch.exp(-eps * previous)
    weighted = torch.sum(weights * losses) / torch.clamp(torch.sum(weights), min=1.0e-12)
    return weighted, {
        f"{prefix}_causal_weight_mean": torch.mean(weights),
        f"{prefix}_causal_weight_min": torch.min(weights),
    }


def compute_pde_loss(model: torch.nn.Module, pde_points: dict[str, torch.Tensor], physics_cfg: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    losses: list[torch.Tensor] = []
    weighted_losses: list[torch.Tensor] = []
    weight_sum = 0.0
    diagnostics: dict[str, torch.Tensor] = {}
    loss_weights = config.get("loss_weights", {})
    for key, region in [("hf", "HF"), ("srv", "SRV"), ("usrv", "USRV")]:
        points = pde_points[key]
        xyt = _points_xyt(points)
        if key == "hf":
            residual = line_pde_residual(model, xyt, points["tangent"], physics_cfg, config)
        else:
            residual = pde_residual(model, xyt, region, physics_cfg, config)
        coeff = dimensionless_pde_coefficients(physics_cfg, config, region, residual.device, residual.dtype)
        normalized = residual / coeff["residual_scale"]
        loss, causal_diag = _causal_mse(normalized, xyt, config, f"pde_p12_{key}")
        diagnostics[f"loss_pde_p12_{key}"] = loss
        diagnostics[f"rms_raw_p12_{key}"] = _rms(residual)
        diagnostics[f"rms_norm_p12_{key}"] = _rms(normalized)
        diagnostics.update(causal_diag)
        losses.append(loss)
        region_weight = float(loss_weights.get(f"pde_{key}", 1.0))
        if region_weight > 0.0:
            weighted_losses.append(region_weight * loss)
            weight_sum += region_weight
    if weighted_losses:
        return torch.stack(weighted_losses).sum() / max(weight_sum, 1.0e-12), diagnostics
    return torch.stack(losses).mean() * 0.0, diagnostics


def _take_gpinn_subset(xyt: torch.Tensor, max_points: int) -> torch.Tensor:
    if max_points <= 0 or xyt.shape[0] <= max_points:
        return xyt
    indices = torch.linspace(0, xyt.shape[0] - 1, int(max_points), dtype=torch.long, device=xyt.device)
    return xyt[indices]


def compute_gradient_enhanced_pde_loss(
    model: torch.nn.Module,
    pde_points: dict[str, torch.Tensor],
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    gpinn_cfg = config["training"].get("gradient_enhanced_pde", {})
    if not bool(gpinn_cfg.get("enabled", False)):
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_gpinn_spatial": zero}
    max_points = int(gpinn_cfg.get("max_points_per_region", 96))
    losses: list[torch.Tensor] = []
    diagnostics: dict[str, torch.Tensor] = {}
    for key, region in [("srv", "SRV"), ("usrv", "USRV")]:
        xyt = _take_gpinn_subset(_points_xyt(pde_points[key]), max_points)
        if xyt.numel() == 0:
            zero = next(model.parameters()).sum() * 0.0
            diagnostics[f"loss_gpinn_{key}"] = zero
            diagnostics[f"rms_gpinn_{key}"] = zero
            continue
        _residual, spatial_gradient = pde_residual_spatial_gradient(model, xyt, region, physics_cfg, config)
        loss = _mse(spatial_gradient)
        losses.append(loss)
        diagnostics[f"loss_gpinn_{key}"] = loss
        diagnostics[f"rms_gpinn_{key}"] = _rms(spatial_gradient)
    if not losses:
        zero = next(model.parameters()).sum() * 0.0
        return zero, diagnostics
    loss_total = torch.stack(losses).mean()
    diagnostics["rms_gpinn_spatial"] = torch.sqrt(torch.clamp(loss_total, min=0.0))
    return loss_total, diagnostics


def compute_dirichlet_loss(model: torch.nn.Module, points: dict[str, torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    xyt = points["xyt"]
    pred = model(xyt)
    target = dirichlet_target_hat(xyt[:, 2:3], config["boundary"])
    return _mse(pred - target)


def compute_neumann_loss(model: torch.nn.Module, points: dict[str, torch.Tensor]) -> torch.Tensor:
    dn = neumann_normal_derivative(model, points["xyt"], points["normal"])
    return _mse(dn)


def compute_pressure_range_loss(model: torch.nn.Module, pde_points: dict[str, torch.Tensor], lower: float = 0.0, upper: float = 1.05) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for points in pde_points.values():
        xyt = _points_xyt(points)
        if xyt.numel() == 0:
            continue
        u = model(xyt)
        losses.append(_mse(torch.relu(float(lower) - u)) + _mse(torch.relu(u - float(upper))))
    if not losses:
        return next(model.parameters()).sum() * 0.0
    return torch.stack(losses).mean()


def _collect_xyt_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and value.shape[1] == 3:
            tensors.append(value)
        return tensors
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"normal", "tangent"}:
                continue
            tensors.extend(_collect_xyt_tensors(item))
    return tensors


def compute_all_sample_pressure_range_loss(model: torch.nn.Module, samples: dict[str, Any], lower: float = 0.0, upper: float = 1.0) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for xyt in _collect_xyt_tensors(samples):
        if xyt.numel() == 0:
            continue
        u = model(xyt)
        losses.append(_mse(torch.relu(float(lower) - u)) + _mse(torch.relu(u - float(upper))))
    if not losses:
        return next(model.parameters()).sum() * 0.0
    return torch.stack(losses).mean()


def compute_correction_regularization_loss(model: torch.nn.Module, samples: dict[str, Any]) -> torch.Tensor:
    if not hasattr(model, "correction"):
        return next(model.parameters()).sum() * 0.0
    losses: list[torch.Tensor] = []
    for xyt in _collect_xyt_tensors(samples):
        if xyt.numel() == 0:
            continue
        losses.append(_mse(model.correction(xyt)))
    if not losses:
        return next(model.parameters()).sum() * 0.0
    return torch.stack(losses).mean()


def compute_hf_main_link_loss(model: torch.nn.Module, samples: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("hf_main_link", {})
    xyt = points.get("xyt") if isinstance(points, dict) else None
    if xyt is None or xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_main_link": zero}
    tangent = torch.zeros((xyt.shape[0], 2), dtype=xyt.dtype, device=xyt.device)
    tangent[:, 0] = 1.0
    residual = line_tangential_derivative(model, xyt, tangent)
    return _mse(residual), {"rms_hf_main_link": _rms(residual)}


def compute_hf_secondary_link_loss(model: torch.nn.Module, samples: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("hf_secondary_link", {})
    xyt = points.get("xyt") if isinstance(points, dict) else None
    junction_xyt = points.get("junction_xyt") if isinstance(points, dict) else None
    if xyt is None or junction_xyt is None or xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_secondary_link": zero}
    residual = model(xyt) - model(junction_xyt)
    return _mse(residual), {"rms_hf_secondary_link": _rms(residual)}


def compute_hf_junction_loss(model: torch.nn.Module, samples: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("hf_junction", {})
    main_xyt = points.get("main_xyt") if isinstance(points, dict) else None
    secondary_xyt = points.get("secondary_xyt") if isinstance(points, dict) else None
    if main_xyt is None or secondary_xyt is None or main_xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_junction": zero}
    residual = model(main_xyt) - model(secondary_xyt)
    return _mse(residual), {"rms_hf_junction": _rms(residual)}


def compute_hf_junction_flux_loss(model: torch.nn.Module, samples: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("hf_junction_flux", {})
    if not isinstance(points, dict):
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_junction_flux": zero}
    residual = line_hf_junction_flux_residual(model, points, config)
    return _mse(residual), {"rms_hf_junction_flux": _rms(residual)}


def compute_hf_tip_neumann_loss(model: torch.nn.Module, samples: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("hf_tip_neumann", {})
    xyt = points.get("xyt") if isinstance(points, dict) else None
    tangent = points.get("tangent") if isinstance(points, dict) else None
    if xyt is None or tangent is None or xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_tip_neumann": zero}
    residual = line_tangential_derivative(model, xyt, tangent)
    return _mse(residual), {"rms_hf_tip_neumann": _rms(residual)}


def compute_hf_leakoff_balance_loss(model: torch.nn.Module, samples: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("hf_leakoff_balance", {})
    xyt = points.get("xyt") if isinstance(points, dict) else None
    tangent = points.get("tangent") if isinstance(points, dict) else None
    aperture = points.get("aperture") if isinstance(points, dict) else None
    if xyt is None or tangent is None or aperture is None or xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_leakoff_balance": zero}
    residual = line_hf_leakoff_balance_residual(model, xyt, tangent, aperture, config["physics"], config)
    coeff = dimensionless_pde_coefficients(config["physics"], config, "HF", residual.device, residual.dtype)
    normalized = residual / coeff["residual_scale"]
    return _mse(normalized), {
        "rms_hf_leakoff_balance": _rms(normalized),
        "rms_raw_hf_leakoff_balance": _rms(residual),
    }


def compute_hf_segment_conservation_loss(model: torch.nn.Module, samples: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("hf_segment_conservation", {})
    if not isinstance(points, dict):
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_segment_conservation": zero}
    residual = line_hf_segment_conservation_residual(model, points, config["physics"], config)
    return _mse(residual), {
        "rms_hf_segment_conservation": _rms(residual),
    }


def _optional_loss_enabled(config: dict[str, Any], name: str) -> bool:
    if float(config.get("loss_weights", {}).get(name, 0.0)) > 0.0:
        return True
    validation_cfg = config.get("training", {}).get("validation", {})
    weights = validation_cfg.get("selection_weights", {}) if isinstance(validation_cfg, dict) else {}
    return float(weights.get(name, 0.0)) > 0.0


def compute_symmetry_loss(model: torch.nn.Module, samples: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    points = samples.get("symmetry", {})
    losses: list[torch.Tensor] = []
    diagnostics: dict[str, torch.Tensor] = {}
    for region_key in ["hf", "srv", "usrv"]:
        region_points = points.get(region_key, {}) if isinstance(points, dict) else {}
        xyt = region_points.get("xyt") if isinstance(region_points, dict) else None
        reflected_xyt = region_points.get("reflected_xyt") if isinstance(region_points, dict) else None
        if xyt is None or reflected_xyt is None or xyt.numel() == 0:
            zero = next(model.parameters()).sum() * 0.0
            diagnostics[f"loss_symmetry_{region_key}"] = zero
            diagnostics[f"rms_symmetry_{region_key}"] = zero
            continue
        residual = model(xyt) - model(reflected_xyt)
        loss = _mse(residual)
        losses.append(loss)
        diagnostics[f"loss_symmetry_{region_key}"] = loss
        diagnostics[f"rms_symmetry_{region_key}"] = _rms(residual)
    if not losses:
        zero = next(model.parameters()).sum() * 0.0
        return zero, diagnostics
    loss_total = torch.stack(losses).mean()
    diagnostics["rms_symmetry"] = torch.sqrt(torch.clamp(loss_total, min=0.0))
    return loss_total, diagnostics


def compute_interface_loss(
    model: torch.nn.Module,
    samples: dict[str, Any],
    geometry: ReservoirGeometry,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    sampler_cfg = config["sampler"]
    pressure_hf, flux_hf, raw_flux_hf, mask_hf = line_hf_srv_residual(
        model,
        samples["interface_hf_srv"]["xyt"],
        samples["interface_hf_srv"]["normal"],
        float(sampler_cfg["eps_hf_srv"]),
        geometry,
    )
    pressure_srv, flux_srv, raw_flux_srv, mask_srv = interface_residual(
        model,
        samples["interface_srv_usrv"]["xyt"],
        samples["interface_srv_usrv"]["normal"],
        float(sampler_cfg["eps_srv_usrv"]),
        geometry,
        physics_cfg,
        config,
        minus_region=REGION_SRV,
        plus_region=REGION_USRV,
    )
    diagnostics = {
        "loss_interface_pressure_hf_srv": _mse(pressure_hf),
        "loss_interface_pressure_srv_usrv": _mse(pressure_srv),
        "loss_interface_flux_hf_srv": _mse(flux_hf),
        "loss_interface_flux_srv_usrv": _mse(flux_srv),
        "rms_interface_pressure_hf_srv": _rms(pressure_hf),
        "rms_interface_pressure_srv_usrv": _rms(pressure_srv),
        "rms_interface_flux_norm_hf_srv": _rms(flux_hf),
        "rms_interface_flux_norm_srv_usrv": _rms(flux_srv),
        "rms_interface_flux_raw_hf_srv": _rms(raw_flux_hf),
        "rms_interface_flux_raw_srv_usrv": _rms(raw_flux_srv),
        "n_interface_valid_hf_srv": samples["interface_hf_srv"]["xyt"].new_tensor(float(mask_hf.sum().detach().cpu())),
        "n_interface_valid_srv_usrv": samples["interface_srv_usrv"]["xyt"].new_tensor(float(mask_srv.sum().detach().cpu())),
    }
    pressure_loss = diagnostics["loss_interface_pressure_hf_srv"] + diagnostics["loss_interface_pressure_srv_usrv"]
    flux_loss = diagnostics["loss_interface_flux_hf_srv"] + diagnostics["loss_interface_flux_srv_usrv"]
    return pressure_loss, flux_loss, diagnostics


def compute_total_loss(
    model: torch.nn.Module,
    samples: dict[str, Any],
    geometry: ReservoirGeometry,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = config["loss_weights"]
    loss_pde, pde_diag = compute_pde_loss(model, samples["pde"], physics_cfg, config)
    loss_gpinn, gpinn_diag = compute_gradient_enhanced_pde_loss(model, samples["pde"], physics_cfg, config)
    loss_dirichlet = compute_dirichlet_loss(model, samples["dirichlet"], config)
    loss_neumann = compute_neumann_loss(model, samples["neumann"])
    loss_interface_pressure, loss_interface_flux, interface_diag = compute_interface_loss(model, samples, geometry, physics_cfg, config)
    loss_hf_main_link, hf_main_diag = compute_hf_main_link_loss(model, samples, config)
    loss_hf_secondary_link, hf_secondary_diag = compute_hf_secondary_link_loss(model, samples)
    loss_hf_junction, hf_junction_diag = compute_hf_junction_loss(model, samples)
    if _optional_loss_enabled(config, "hf_junction_flux"):
        loss_hf_junction_flux, hf_junction_flux_diag = compute_hf_junction_flux_loss(model, samples, config)
    else:
        loss_hf_junction_flux = next(model.parameters()).sum() * 0.0
        hf_junction_flux_diag = {"rms_hf_junction_flux": loss_hf_junction_flux}
    loss_hf_tip_neumann, hf_tip_diag = compute_hf_tip_neumann_loss(model, samples)
    loss_hf_leakoff_balance, hf_leakoff_diag = compute_hf_leakoff_balance_loss(model, samples, config)
    if _optional_loss_enabled(config, "hf_segment_conservation"):
        loss_hf_segment_conservation, hf_segment_diag = compute_hf_segment_conservation_loss(model, samples, config)
    else:
        loss_hf_segment_conservation = next(model.parameters()).sum() * 0.0
        hf_segment_diag = {"rms_hf_segment_conservation": loss_hf_segment_conservation}
    loss_symmetry, symmetry_diag = compute_symmetry_loss(model, samples)
    loss_range = compute_all_sample_pressure_range_loss(
        model,
        samples,
        lower=float(config["training"].get("pressure_range_lower", 0.0)),
        upper=float(config["training"].get("pressure_range_upper", 1.0)),
    )
    loss_correction = compute_correction_regularization_loss(model, samples)
    loss_local_conservation, local_diag = compute_local_conservation_loss(model, samples.get("local_conservation", {}), config)
    if any(key in weights for key in ["interface_pressure_hf_srv", "interface_pressure_srv_usrv"]):
        weighted_interface_pressure = (
            float(weights.get("interface_pressure_hf_srv", weights.get("interface_pressure", 0.0))) * interface_diag["loss_interface_pressure_hf_srv"]
            + float(weights.get("interface_pressure_srv_usrv", weights.get("interface_pressure", 0.0))) * interface_diag["loss_interface_pressure_srv_usrv"]
        )
    else:
        weighted_interface_pressure = float(weights.get("interface_pressure", 0.0)) * loss_interface_pressure
    if any(key in weights for key in ["interface_flux_hf_srv", "interface_flux_srv_usrv"]):
        weighted_interface_flux = (
            float(weights.get("interface_flux_hf_srv", weights.get("interface_flux", 0.0))) * interface_diag["loss_interface_flux_hf_srv"]
            + float(weights.get("interface_flux_srv_usrv", weights.get("interface_flux", 0.0))) * interface_diag["loss_interface_flux_srv_usrv"]
        )
    else:
        weighted_interface_flux = float(weights.get("interface_flux", 0.0)) * loss_interface_flux
    total = (
        float(weights["pde"]) * loss_pde
        + float(weights.get("gradient_enhanced_pde", 0.0)) * loss_gpinn
        + float(weights["dirichlet"]) * loss_dirichlet
        + float(weights["neumann"]) * loss_neumann
        + weighted_interface_pressure
        + weighted_interface_flux
        + float(weights.get("hf_main_link", 0.0)) * loss_hf_main_link
        + float(weights.get("hf_secondary_link", 0.0)) * loss_hf_secondary_link
        + float(weights.get("hf_junction", 0.0)) * loss_hf_junction
        + float(weights.get("hf_junction_flux", 0.0)) * loss_hf_junction_flux
        + float(weights.get("hf_tip_neumann", 0.0)) * loss_hf_tip_neumann
        + float(weights.get("hf_leakoff_balance", 0.0)) * loss_hf_leakoff_balance
        + float(weights.get("hf_segment_conservation", 0.0)) * loss_hf_segment_conservation
        + float(weights.get("symmetry", 0.0)) * loss_symmetry
        + float(weights.get("pressure_range", 0.0)) * loss_range
        + float(weights.get("correction_regularization", 0.0)) * loss_correction
        + float(weights.get("local_conservation", 0.0)) * loss_local_conservation
    )
    diagnostics = {
        "loss_total": total,
        "loss_pde": loss_pde,
        "loss_gradient_enhanced_pde": loss_gpinn,
        "loss_dirichlet": loss_dirichlet,
        "loss_neumann": loss_neumann,
        "loss_interface_pressure": loss_interface_pressure,
        "loss_interface_flux": loss_interface_flux,
        "loss_hf_main_link": loss_hf_main_link,
        "loss_hf_secondary_link": loss_hf_secondary_link,
        "loss_hf_junction": loss_hf_junction,
        "loss_hf_junction_flux": loss_hf_junction_flux,
        "loss_hf_tip_neumann": loss_hf_tip_neumann,
        "loss_hf_leakoff_balance": loss_hf_leakoff_balance,
        "loss_hf_segment_conservation": loss_hf_segment_conservation,
        "loss_symmetry": loss_symmetry,
        "loss_pressure_range": loss_range,
        "loss_correction_regularization": loss_correction,
        "loss_local_conservation": loss_local_conservation,
    }
    diagnostics.update(pde_diag)
    diagnostics.update(gpinn_diag)
    diagnostics.update(interface_diag)
    diagnostics.update(hf_main_diag)
    diagnostics.update(hf_secondary_diag)
    diagnostics.update(hf_junction_diag)
    diagnostics.update(hf_junction_flux_diag)
    diagnostics.update(hf_tip_diag)
    diagnostics.update(hf_leakoff_diag)
    diagnostics.update(hf_segment_diag)
    diagnostics.update(symmetry_diag)
    diagnostics.update(local_diag)
    return total, diagnostics
