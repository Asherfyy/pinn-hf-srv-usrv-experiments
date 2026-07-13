"""Loss functions for the mixed pressure-flux PINN."""

from __future__ import annotations

from typing import Any

import torch

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry
from .physics import dimensionless_pde_coefficients, interface_residual, neumann_normal_derivative, pde_residual
from .utils import dirichlet_target_hat


def _mse(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.sum() * 0.0
    return torch.mean(value**2)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(_mse(value), min=0.0))


def compute_pde_loss(
    model: torch.nn.Module,
    pde_points: dict[str, torch.Tensor],
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute mixed-form Darcy/Fick and conservation losses by region."""

    region_pairs = [("hf", "HF"), ("srv", "SRV"), ("usrv", "USRV")]
    component_losses: list[torch.Tensor] = []
    diagnostics: dict[str, torch.Tensor] = {}
    for key, region in region_pairs:
        xyt = pde_points[key]
        darcy12, darcy13, conservation12, conservation13 = pde_residual(model, xyt, region, physics_cfg, config)
        for variable, darcy, conservation in [
            ("u12", darcy12, conservation12),
            ("u13", darcy13, conservation13),
        ]:
            coeff = dimensionless_pde_coefficients(physics_cfg, config, variable, region, conservation.device, conservation.dtype)
            darcy_norm = darcy / coeff["residual_scale"]
            conservation_norm = conservation / coeff["residual_scale"]
            loss_darcy = _mse(darcy_norm)
            loss_conservation = _mse(conservation_norm)
            loss = 0.5 * (loss_darcy + loss_conservation)
            normalized = torch.cat([darcy_norm, conservation_norm], dim=1)
            suffix = f"{variable}_{key}"
            diagnostics[f"loss_pde_{suffix}"] = loss
            diagnostics[f"loss_darcy_{suffix}"] = loss_darcy
            diagnostics[f"loss_conservation_{suffix}"] = loss_conservation
            diagnostics[f"rms_raw_{suffix}"] = _rms(conservation)
            diagnostics[f"rms_norm_{suffix}"] = _rms(normalized)
            diagnostics[f"rms_darcy_norm_{suffix}"] = _rms(darcy_norm)
            diagnostics[f"rms_conservation_norm_{suffix}"] = _rms(conservation_norm)
            component_losses.append(loss)
    total = torch.stack(component_losses).mean()
    return total, diagnostics


def compute_dirichlet_loss(model: torch.nn.Module, points: dict[str, torch.Tensor], config: dict[str, Any]) -> torch.Tensor:
    xyt = points["xyt"]
    pred = model(xyt)[:, 0:2]
    target = dirichlet_target_hat(xyt[:, 2:3], config["boundary"])
    return _mse(pred - target)


def compute_neumann_loss(model: torch.nn.Module, points: dict[str, torch.Tensor]) -> torch.Tensor:
    qn12, qn13 = neumann_normal_derivative(model, points["xyt"], points["normal"])
    return _mse(qn12) + _mse(qn13)


def compute_pressure_range_loss(model: torch.nn.Module, pde_points: dict[str, torch.Tensor], lower: float = 0.0, upper: float = 1.05) -> torch.Tensor:
    """Softly keep dimensionless pressure in the expected diffusion range."""

    losses: list[torch.Tensor] = []
    for xyt in pde_points.values():
        if xyt.numel() == 0:
            continue
        u = model(xyt)[:, 0:2]
        losses.append(_mse(torch.relu(float(lower) - u)) + _mse(torch.relu(u - float(upper))))
    if not losses:
        return next(model.parameters()).sum() * 0.0
    return torch.stack(losses).mean()


def compute_hf_main_link_loss(model: torch.nn.Module, samples: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Force the main fracture centerline toward the production pressure."""

    points = samples.get("hf_main_link", {})
    xyt = points.get("xyt") if isinstance(points, dict) else None
    if xyt is None or xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_main_link": zero}
    pred = model(xyt)[:, 0:2]
    target = dirichlet_target_hat(xyt[:, 2:3], config["boundary"])
    residual = pred - target
    loss = _mse(residual)
    return loss, {"rms_hf_main_link": _rms(residual)}


def compute_hf_secondary_link_loss(model: torch.nn.Module, samples: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Force each secondary fracture to equilibrate quickly along its centerline."""

    points = samples.get("hf_secondary_link", {})
    xyt = points.get("xyt") if isinstance(points, dict) else None
    junction_xyt = points.get("junction_xyt") if isinstance(points, dict) else None
    if xyt is None or junction_xyt is None or xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_secondary_link": zero}
    residual = model(xyt)[:, 0:2] - model(junction_xyt)[:, 0:2]
    loss = _mse(residual)
    return loss, {"rms_hf_secondary_link": _rms(residual)}


def compute_hf_junction_loss(model: torch.nn.Module, samples: dict[str, Any], config: dict[str, Any]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Strongly couple main-fracture and secondary-fracture pressure near intersections."""

    points = samples.get("hf_junction", {})
    main_xyt = points.get("main_xyt") if isinstance(points, dict) else None
    secondary_xyt = points.get("secondary_xyt") if isinstance(points, dict) else None
    if main_xyt is None or secondary_xyt is None or main_xyt.numel() == 0:
        zero = next(model.parameters()).sum() * 0.0
        return zero, {"rms_hf_junction": zero}
    residual = model(main_xyt)[:, 0:2] - model(secondary_xyt)[:, 0:2]
    loss = _mse(residual)
    return loss, {"rms_hf_junction": _rms(residual)}


def compute_interface_loss(
    model: torch.nn.Module,
    samples: dict[str, Any],
    geometry: ReservoirGeometry,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Compute pressure-continuity and normal-flux-continuity losses."""

    sampler_cfg = config["sampler"]
    pressure_hf, flux_hf, raw_flux_hf, mask_hf = interface_residual(
        model,
        samples["interface_hf_srv"]["xyt"],
        samples["interface_hf_srv"]["normal"],
        float(sampler_cfg["eps_hf_srv"]),
        geometry,
        physics_cfg,
        config,
        minus_region=REGION_HF,
        plus_region=REGION_SRV,
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
    pressure_loss_hf = _mse(pressure_hf)
    pressure_loss_srv = _mse(pressure_srv)
    flux_loss_hf = _mse(flux_hf)
    flux_loss_srv = _mse(flux_srv)
    diagnostics = {
        "loss_interface_pressure_hf_srv": pressure_loss_hf,
        "loss_interface_pressure_srv_usrv": pressure_loss_srv,
        "loss_interface_flux_hf_srv": flux_loss_hf,
        "loss_interface_flux_srv_usrv": flux_loss_srv,
        "rms_interface_pressure_hf_srv": _rms(pressure_hf),
        "rms_interface_pressure_srv_usrv": _rms(pressure_srv),
        "rms_interface_flux_norm_hf_srv": _rms(flux_hf),
        "rms_interface_flux_norm_srv_usrv": _rms(flux_srv),
        "rms_interface_flux_raw_hf_srv": _rms(raw_flux_hf),
        "rms_interface_flux_raw_srv_usrv": _rms(raw_flux_srv),
        "n_interface_valid_hf_srv": samples["interface_hf_srv"]["xyt"].new_tensor(float(mask_hf.sum().detach().cpu())),
        "n_interface_valid_srv_usrv": samples["interface_srv_usrv"]["xyt"].new_tensor(float(mask_srv.sum().detach().cpu())),
    }
    return pressure_loss_hf + pressure_loss_srv, flux_loss_hf + flux_loss_srv, diagnostics


def compute_total_loss(
    model: torch.nn.Module,
    samples: dict[str, Any],
    geometry: ReservoirGeometry,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute weighted total loss and diagnostics."""

    weights = config["loss_weights"]
    loss_pde, pde_diag = compute_pde_loss(model, samples["pde"], physics_cfg, config)
    loss_dirichlet = compute_dirichlet_loss(model, samples["dirichlet"], config)
    loss_neumann = compute_neumann_loss(model, samples["neumann"])
    loss_interface_pressure, loss_interface_flux, interface_diag = compute_interface_loss(model, samples, geometry, physics_cfg, config)
    loss_hf_main_link, hf_main_diag = compute_hf_main_link_loss(model, samples, config)
    loss_hf_secondary_link, hf_secondary_diag = compute_hf_secondary_link_loss(model, samples, config)
    loss_hf_junction, hf_junction_diag = compute_hf_junction_loss(model, samples, config)
    loss_range = compute_pressure_range_loss(
        model,
        samples["pde"],
        lower=float(config["training"].get("pressure_range_lower", 0.0)),
        upper=float(config["training"].get("pressure_range_upper", 1.05)),
    )
    total = (
        float(weights["pde"]) * loss_pde
        + float(weights["dirichlet"]) * loss_dirichlet
        + float(weights["neumann"]) * loss_neumann
        + float(weights.get("interface_pressure", 0.0)) * loss_interface_pressure
        + float(weights.get("interface_flux", 0.0)) * loss_interface_flux
        + float(weights.get("hf_main_link", 0.0)) * loss_hf_main_link
        + float(weights.get("hf_secondary_link", 0.0)) * loss_hf_secondary_link
        + float(weights.get("hf_junction", 0.0)) * loss_hf_junction
        + float(weights.get("pressure_range", 0.0)) * loss_range
    )
    diagnostics = {
        "loss_total": total,
        "loss_pde": loss_pde,
        "loss_dirichlet": loss_dirichlet,
        "loss_neumann": loss_neumann,
        "loss_interface_pressure": loss_interface_pressure,
        "loss_interface_flux": loss_interface_flux,
        "loss_hf_main_link": loss_hf_main_link,
        "loss_hf_secondary_link": loss_hf_secondary_link,
        "loss_hf_junction": loss_hf_junction,
        "loss_pressure_range": loss_range,
    }
    diagnostics.update(pde_diag)
    diagnostics.update(interface_diag)
    diagnostics.update(hf_main_diag)
    diagnostics.update(hf_secondary_diag)
    diagnostics.update(hf_junction_diag)
    return total, diagnostics
