"""Mesh-free strong-form physics for single-pressure P12."""

from __future__ import annotations

from typing import Any

import torch

from .geometry import REGION_HF, REGION_NAMES, REGION_SRV, REGION_USRV, ReservoirGeometry


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]


def second_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    first = grad(outputs, inputs)
    cols = []
    for dim in range(inputs.shape[1]):
        first_col = first[:, dim : dim + 1]
        if not first_col.requires_grad:
            cols.append(torch.zeros_like(first_col))
        else:
            cols.append(grad(first_col, inputs)[:, dim : dim + 1])
    return torch.cat(cols, dim=1)


def effective_diffusion(physics_cfg: dict[str, Any], region_name: str) -> float:
    region = region_name.upper()
    if region not in REGION_NAMES:
        raise ValueError(f"Unknown region: {region_name}")
    fai = float(physics_cfg["Fai"][region])
    diffusion = float(physics_cfg["D"][region])
    if fai <= 0.0:
        raise ValueError(f"Fai_{region} must be positive.")
    return diffusion / fai


def dimensionless_pde_coefficients(
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
    region_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    geom = config["geometry"]
    sampler = config["sampler"]
    lx = float(geom["x_max"]) - float(geom["x_min"])
    ly = float(geom["y_max"]) - float(geom["y_min"])
    t_seconds = (float(sampler["t_max"]) - float(sampler["t_min"])) * float(physics_cfg["seconds_per_day"])
    k_eff = effective_diffusion(physics_cfg, region_name)
    alpha_y = float(physics_cfg.get("anisotropy_y", 1.0))
    kx = torch.as_tensor(k_eff * t_seconds / (lx * lx), dtype=dtype, device=device)
    ky = torch.as_tensor(alpha_y * k_eff * t_seconds / (ly * ly), dtype=dtype, device=device)
    one = torch.as_tensor(1.0, dtype=dtype, device=device)
    scale = torch.maximum(one, torch.maximum(torch.abs(kx), torch.abs(ky)))
    return {"kappa_x": kx, "kappa_y": ky, "residual_scale": scale}


def _region_name_from_id(region_id: int) -> str:
    mapping = {REGION_HF: "HF", REGION_SRV: "SRV", REGION_USRV: "USRV"}
    if int(region_id) not in mapping:
        raise ValueError(f"Unknown region id: {region_id}")
    return mapping[int(region_id)]


def _forward_normalized_for_region(model: torch.nn.Module, z: torch.Tensor, region_name: str) -> torch.Tensor:
    if hasattr(model, "forward_region_normalized"):
        return model.forward_region_normalized(z, region_name)
    return model.forward_normalized(z)


def pde_residual(model: torch.nn.Module, xyt: torch.Tensor, region_name: str, physics_cfg: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du = grad(u[:, 0:1], z)
    d2u = second_grad(u[:, 0:1], z)
    coeff = dimensionless_pde_coefficients(physics_cfg, config, region_name, z.device, z.dtype)
    return du[:, 2:3] - coeff["kappa_x"] * d2u[:, 0:1] - coeff["kappa_y"] * d2u[:, 1:2]


def pde_residual_spatial_gradient(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    region_name: str,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized PDE residual and its spatial gradient in normalized coordinates."""

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du = grad(u[:, 0:1], z)
    d2u = second_grad(u[:, 0:1], z)
    coeff = dimensionless_pde_coefficients(physics_cfg, config, region_name, z.device, z.dtype)
    residual = du[:, 2:3] - coeff["kappa_x"] * d2u[:, 0:1] - coeff["kappa_y"] * d2u[:, 1:2]
    normalized = residual / coeff["residual_scale"]
    if not normalized.requires_grad:
        return normalized, torch.zeros((z.shape[0], 2), dtype=z.dtype, device=z.device)
    residual_gradient = grad(normalized, z)
    return normalized, residual_gradient[:, 0:2]


def line_pde_residual(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    tangent: torch.Tensor,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    """Strong-form 1D diffusion residual on fracture centerlines."""

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, "HF")
    du = grad(u[:, 0:1], z)
    d2u = second_grad(u[:, 0:1], z)
    coeff = dimensionless_pde_coefficients(physics_cfg, config, "HF", z.device, z.dtype)
    tx = tangent[:, 0:1].to(device=z.device, dtype=z.dtype)
    ty = tangent[:, 1:2].to(device=z.device, dtype=z.dtype)
    tangent_laplacian = (tx * tx) * d2u[:, 0:1] + (ty * ty) * d2u[:, 1:2]
    tangent_kappa = (tx * tx) * coeff["kappa_x"] + (ty * ty) * coeff["kappa_y"]
    return du[:, 2:3] - tangent_kappa * tangent_laplacian


def normal_derivative_physical(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    normal: torch.Tensor,
    region_name: str,
    config: dict[str, Any],
) -> torch.Tensor:
    """Normal derivative du/dn in physical meters for a region subnet."""

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du = grad(u[:, 0:1], z)
    geom = config["geometry"]
    lx = float(geom["x_max"]) - float(geom["x_min"])
    ly = float(geom["y_max"]) - float(geom["y_min"])
    nx = normal[:, 0:1].to(device=z.device, dtype=z.dtype)
    ny = normal[:, 1:2].to(device=z.device, dtype=z.dtype)
    return nx * du[:, 0:1] / lx + ny * du[:, 1:2] / ly


def time_derivative_normalized(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    region_name: str,
) -> torch.Tensor:
    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du = grad(u[:, 0:1], z)
    return du[:, 2:3]


def tangential_derivative_physical(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    tangent: torch.Tensor,
    region_name: str,
    config: dict[str, Any],
) -> torch.Tensor:
    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du = grad(u[:, 0:1], z)
    geom = config["geometry"]
    lx = float(geom["x_max"]) - float(geom["x_min"])
    ly = float(geom["y_max"]) - float(geom["y_min"])
    tx = tangent[:, 0:1].to(device=z.device, dtype=z.dtype)
    ty = tangent[:, 1:2].to(device=z.device, dtype=z.dtype)
    return tx * du[:, 0:1] / lx + ty * du[:, 1:2] / ly


def line_hf_leakoff_balance_residual(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    tangent: torch.Tensor,
    aperture: torch.Tensor,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    """HF line balance with mesh-free two-sided SRV leakoff.

    The base line equation is the 1D HF diffusion residual. The leakoff term
    is estimated from SRV normal gradients at two small offsets from the
    fracture centerline. No fixed cells or transmissibility table are used.
    """

    if xyt.numel() == 0:
        return xyt.new_zeros((0, 1))
    line_residual = line_pde_residual(model, xyt, tangent, physics_cfg, config)
    tx = tangent[:, 0:1].to(device=xyt.device, dtype=xyt.dtype)
    ty = tangent[:, 1:2].to(device=xyt.device, dtype=xyt.dtype)
    normal_a = torch.cat([-ty, tx], dim=1)
    normal_b = torch.cat([ty, -tx], dim=1)
    offset = float(config["sampler"].get("hf_leakoff_offset_m", config["sampler"].get("eps_hf_srv", 0.05)))
    zero_t = torch.zeros_like(xyt[:, 2:3])
    xyt_a = xyt + torch.cat([normal_a * offset, zero_t], dim=1)
    xyt_b = xyt + torch.cat([normal_b * offset, zero_t], dim=1)
    dudn_a = normal_derivative_physical(model, xyt_a, normal_a, "SRV", config)
    dudn_b = normal_derivative_physical(model, xyt_b, normal_b, "SRV", config)

    sampler = config["sampler"]
    t_seconds = (float(sampler["t_max"]) - float(sampler["t_min"])) * float(physics_cfg["seconds_per_day"])
    diffusion_srv = float(physics_cfg["D"]["SRV"])
    storage_hf = float(physics_cfg["Fai"]["HF"])
    aperture_safe = torch.clamp(aperture.to(device=xyt.device, dtype=xyt.dtype), min=1.0e-12)
    leak_source = (t_seconds * diffusion_srv / storage_hf) * (dudn_a + dudn_b) / aperture_safe
    return line_residual - leak_source


def line_hf_segment_conservation_residual(
    model: torch.nn.Module,
    points: dict[str, torch.Tensor],
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    """Weak HF segment balance using random line segments, not grid cells."""

    gauss_xyt = points.get("gauss_xyt")
    left_xyt = points.get("left_xyt")
    right_xyt = points.get("right_xyt")
    tangent = points.get("tangent")
    gauss_tangent = points.get("gauss_tangent")
    aperture = points.get("aperture")
    gauss_aperture = points.get("gauss_aperture")
    half_length = points.get("half_length")
    if (
        gauss_xyt is None
        or left_xyt is None
        or right_xyt is None
        or tangent is None
        or gauss_tangent is None
        or aperture is None
        or gauss_aperture is None
        or half_length is None
        or left_xyt.numel() == 0
    ):
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return torch.zeros((0, 1), device=device, dtype=dtype)

    n_segment = left_xyt.shape[0]
    storage = time_derivative_normalized(model, gauss_xyt, "HF").reshape(n_segment, 2).mean(dim=1, keepdim=True)

    tx_g = gauss_tangent[:, 0:1].to(device=gauss_xyt.device, dtype=gauss_xyt.dtype)
    ty_g = gauss_tangent[:, 1:2].to(device=gauss_xyt.device, dtype=gauss_xyt.dtype)
    normal_a = torch.cat([-ty_g, tx_g], dim=1)
    normal_b = torch.cat([ty_g, -tx_g], dim=1)
    offset = float(config["sampler"].get("hf_segment_leakoff_offset_m", config["sampler"].get("hf_leakoff_offset_m", 0.05)))
    zero_t = torch.zeros_like(gauss_xyt[:, 2:3])
    xyt_a = gauss_xyt + torch.cat([normal_a * offset, zero_t], dim=1)
    xyt_b = gauss_xyt + torch.cat([normal_b * offset, zero_t], dim=1)
    dudn_a = normal_derivative_physical(model, xyt_a, normal_a, "SRV", config)
    dudn_b = normal_derivative_physical(model, xyt_b, normal_b, "SRV", config)

    sampler = config["sampler"]
    t_seconds = (float(sampler["t_max"]) - float(sampler["t_min"])) * float(physics_cfg["seconds_per_day"])
    diffusion_srv = float(physics_cfg["D"]["SRV"])
    diffusion_hf = float(physics_cfg["D"]["HF"])
    storage_hf = float(physics_cfg["Fai"]["HF"])
    gauss_aperture_safe = torch.clamp(gauss_aperture.to(device=gauss_xyt.device, dtype=gauss_xyt.dtype), min=1.0e-12)
    leak = (t_seconds * diffusion_srv / storage_hf) * (dudn_a + dudn_b) / gauss_aperture_safe
    leak_mean = leak.reshape(n_segment, 2).mean(dim=1, keepdim=True)

    tangent_right = tangent.to(device=right_xyt.device, dtype=right_xyt.dtype)
    tangent_left = tangent.to(device=left_xyt.device, dtype=left_xyt.dtype)
    du_ds_right = tangential_derivative_physical(model, right_xyt, tangent_right, "HF", config)
    du_ds_left = tangential_derivative_physical(model, left_xyt, tangent_left, "HF", config)
    half_safe = torch.clamp(half_length.to(device=right_xyt.device, dtype=right_xyt.dtype), min=1.0e-12)
    tangent_divergence = (t_seconds * diffusion_hf / storage_hf) * (du_ds_right - du_ds_left) / (2.0 * half_safe)

    residual = storage - tangent_divergence - leak_mean
    coeff = dimensionless_pde_coefficients(physics_cfg, config, "HF", residual.device, residual.dtype)
    tx = tangent[:, 0:1].to(device=residual.device, dtype=residual.dtype)
    ty = tangent[:, 1:2].to(device=residual.device, dtype=residual.dtype)
    tangent_scale = torch.maximum(
        torch.ones_like(residual),
        torch.abs((tx * tx) * coeff["kappa_x"] + (ty * ty) * coeff["kappa_y"]),
    )
    return residual / tangent_scale


def line_hf_junction_flux_residual(
    model: torch.nn.Module,
    points: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    """Kirchhoff-style HF flux balance at fracture intersections.

    The samples are grouped as four outward branch points per junction-time
    pair: main-left, main-right, secondary-down, secondary-up. This is a
    mesh-free line-network constraint; no cell volume or EDFM connection table
    is used.
    """

    xyt = points.get("xyt")
    direction = points.get("direction")
    aperture = points.get("aperture")
    if xyt is None or direction is None or aperture is None or xyt.numel() == 0:
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        return torch.zeros((0, 1), device=device, dtype=dtype)
    if xyt.shape[0] % 4 != 0:
        raise ValueError("HF junction flux samples must contain four branch points per junction-time pair.")

    duds = tangential_derivative_physical(model, xyt, direction, "HF", config)
    aperture_safe = torch.clamp(aperture.to(device=xyt.device, dtype=xyt.dtype), min=1.0e-12)
    weighted = aperture_safe * duds
    grouped = weighted.reshape(-1, 4)
    aperture_grouped = aperture_safe.reshape(-1, 4)
    geom = config["geometry"]
    lx = float(geom["x_max"]) - float(geom["x_min"])
    ly = float(geom["y_max"]) - float(geom["y_min"])
    length_scale = max(float((lx * lx + ly * ly) ** 0.5), 1.0e-12)
    denom = torch.clamp(torch.sum(aperture_grouped, dim=1, keepdim=True) / length_scale, min=torch.finfo(xyt.dtype).eps)
    return torch.sum(grouped, dim=1, keepdim=True) / denom


def line_tangential_derivative(model: torch.nn.Module, xyt: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    """Derivative of HF pressure along a fracture tangent in normalized coordinates."""

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, "HF")
    du = grad(u[:, 0:1], z)
    tx = tangent[:, 0:1].to(device=z.device, dtype=z.dtype)
    ty = tangent[:, 1:2].to(device=z.device, dtype=z.dtype)
    return tx * du[:, 0:1] + ty * du[:, 1:2]


def neumann_normal_derivative(model: torch.nn.Module, xyt: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = model.forward_normalized(z)
    du = grad(u[:, 0:1], z)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    return du[:, 0:1] * nx + du[:, 1:2] * ny


def interface_offset_points(
    xyt_interface: torch.Tensor,
    normal: torch.Tensor,
    eps: float,
    geometry: ReservoirGeometry,
    minus_region: int,
    plus_region: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if eps <= 0.0:
        raise ValueError(f"interface offset eps must be positive, got {eps:g}.")
    offset = torch.cat([normal * float(eps), torch.zeros_like(xyt_interface[:, 2:3])], dim=1)
    xyt_minus = xyt_interface - offset
    xyt_plus = xyt_interface + offset
    region_minus = geometry.region_id_torch(xyt_minus[:, 0:1], xyt_minus[:, 1:2]).view(-1)
    region_plus = geometry.region_id_torch(xyt_plus[:, 0:1], xyt_plus[:, 1:2]).view(-1)
    mask = (region_minus == int(minus_region)) & (region_plus == int(plus_region))
    return xyt_minus[mask], xyt_plus[mask], normal[mask], mask


def _normal_flux(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    normal: torch.Tensor,
    region_name: str,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du = grad(u[:, 0:1], z)
    coeff = dimensionless_pde_coefficients(physics_cfg, config, region_name, z.device, z.dtype)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    return coeff["kappa_x"] * du[:, 0:1] * nx + coeff["kappa_y"] * du[:, 1:2] * ny


def interface_residual(
    model: torch.nn.Module,
    xyt_interface: torch.Tensor,
    normal: torch.Tensor,
    eps: float,
    geometry: ReservoirGeometry,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
    minus_region: int,
    plus_region: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xyt_minus, xyt_plus, normal_valid, mask = interface_offset_points(xyt_interface, normal, eps, geometry, minus_region, plus_region)
    if xyt_minus.shape[0] == 0:
        empty = xyt_interface.new_zeros((0, 1))
        return empty, empty, empty, mask

    pressure_jump = model(xyt_minus) - model(xyt_plus)
    minus_name = _region_name_from_id(minus_region)
    plus_name = _region_name_from_id(plus_region)
    q_minus = _normal_flux(model, xyt_minus, normal_valid, minus_name, physics_cfg, config)
    q_plus = _normal_flux(model, xyt_plus, normal_valid, plus_name, physics_cfg, config)
    raw_flux_jump = q_minus - q_plus
    scale_minus = dimensionless_pde_coefficients(physics_cfg, config, minus_name, raw_flux_jump.device, raw_flux_jump.dtype)["residual_scale"]
    scale_plus = dimensionless_pde_coefficients(physics_cfg, config, plus_name, raw_flux_jump.device, raw_flux_jump.dtype)["residual_scale"]
    normalized_flux_jump = raw_flux_jump / torch.maximum(scale_minus, scale_plus)
    return pressure_jump, normalized_flux_jump, raw_flux_jump, mask


def line_hf_srv_residual(
    model: torch.nn.Module,
    xyt_line: torch.Tensor,
    normal: torch.Tensor,
    eps: float,
    geometry: ReservoirGeometry,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if eps <= 0.0:
        raise ValueError(f"line HF-SRV offset eps must be positive, got {eps:g}.")
    if xyt_line.numel() == 0:
        empty = xyt_line.new_zeros((0, 1))
        mask = torch.zeros((0,), dtype=torch.bool, device=xyt_line.device)
        return empty, empty, empty, mask

    offset = torch.cat([normal * float(eps), torch.zeros_like(xyt_line[:, 2:3])], dim=1)
    xyt_srv = xyt_line + offset
    region_line = geometry.region_id_torch(xyt_line[:, 0:1], xyt_line[:, 1:2]).view(-1)
    region_srv = geometry.region_id_torch(xyt_srv[:, 0:1], xyt_srv[:, 1:2]).view(-1)
    mask = (region_line == REGION_HF) & (region_srv == REGION_SRV)
    if int(mask.sum().detach().cpu()) == 0:
        empty = xyt_line.new_zeros((0, 1))
        return empty, empty, empty, mask

    z_line = model.normalize_xyt(xyt_line[mask])
    z_srv = model.normalize_xyt(xyt_srv[mask])
    u_hf = _forward_normalized_for_region(model, z_line, "HF")
    u_srv = _forward_normalized_for_region(model, z_srv, "SRV")
    pressure_jump = u_hf - u_srv
    zero_flux = torch.zeros_like(pressure_jump)
    return pressure_jump, zero_flux, zero_flux, mask
