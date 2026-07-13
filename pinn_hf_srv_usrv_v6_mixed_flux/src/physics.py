"""Mixed pressure-flux residuals and boundary/interface operators."""

from __future__ import annotations

from typing import Any

import torch

from .geometry import REGION_HF, REGION_NAMES, REGION_SRV, REGION_USRV, ReservoirGeometry


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """First derivative by autograd."""

    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]


def effective_diffusion(physics_cfg: dict[str, Any], variable_name: str, region_name: str) -> float:
    """Return K=D/Fai for the selected pressure component and region."""

    region = region_name.upper()
    if region not in REGION_NAMES:
        raise ValueError(f"Unknown region: {region_name}")
    variable = variable_name.lower()
    d_group = "D1" if variable in {"u12", "p12", "1"} else "D2"
    fai = float(physics_cfg["Fai"][region])
    diffusion = float(physics_cfg[d_group][region])
    if fai <= 0.0:
        raise ValueError(f"Fai_{region} must be positive.")
    return diffusion / fai


def dimensionless_pde_coefficients(
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
    variable_name: str,
    region_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Compute kappa_x/kappa_y in normalized coordinates."""

    geom = config["geometry"]
    sampler = config["sampler"]
    lx = float(geom["x_max"]) - float(geom["x_min"])
    ly = float(geom["y_max"]) - float(geom["y_min"])
    t_seconds = (float(sampler["t_max"]) - float(sampler["t_min"])) * float(physics_cfg["seconds_per_day"])
    variable = variable_name.lower()
    alpha = float(physics_cfg["anisotropy_y_1"] if variable in {"u12", "p12", "1"} else physics_cfg["anisotropy_y_2"])
    k_eff = effective_diffusion(physics_cfg, variable_name, region_name)
    kx = torch.as_tensor(k_eff * t_seconds / (lx * lx), dtype=dtype, device=device)
    ky = torch.as_tensor(alpha * k_eff * t_seconds / (ly * ly), dtype=dtype, device=device)
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


def split_mixed_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split [u12, u13, q12_x, q12_y, q13_x, q13_y]."""

    if output.ndim != 2 or output.shape[1] < 6:
        raise ValueError(f"Mixed-flux model output must have shape [N,6], got {tuple(output.shape)}.")
    return (
        output[:, 0:1],
        output[:, 1:2],
        output[:, 2:3],
        output[:, 3:4],
        output[:, 4:5],
        output[:, 5:6],
    )


def pde_residual(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    region_name: str,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Darcy/Fick and conservation residuals for u12 and u13.

    The fluxes are represented in normalized coordinates:
    q_x = -kappa_x * du/dx_hat, q_y = -kappa_y * du/dy_hat.
    Therefore the conservation residual is du/dt_hat + dq_x/dx_hat + dq_y/dy_hat.
    This is equivalent to the previous strong form, but avoids second
    derivatives of pressure.
    """

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    output = _forward_normalized_for_region(model, z, region_name)
    u12, u13, q12_x, q12_y, q13_x, q13_y = split_mixed_output(output)

    du12 = grad(u12, z)
    du13 = grad(u13, z)
    dq12_x = grad(q12_x, z)
    dq12_y = grad(q12_y, z)
    dq13_x = grad(q13_x, z)
    dq13_y = grad(q13_y, z)

    c12 = dimensionless_pde_coefficients(physics_cfg, config, "u12", region_name, z.device, z.dtype)
    c13 = dimensionless_pde_coefficients(physics_cfg, config, "u13", region_name, z.device, z.dtype)

    darcy12 = torch.cat(
        [
            q12_x + c12["kappa_x"] * du12[:, 0:1],
            q12_y + c12["kappa_y"] * du12[:, 1:2],
        ],
        dim=1,
    )
    darcy13 = torch.cat(
        [
            q13_x + c13["kappa_x"] * du13[:, 0:1],
            q13_y + c13["kappa_y"] * du13[:, 1:2],
        ],
        dim=1,
    )
    conservation12 = du12[:, 2:3] + dq12_x[:, 0:1] + dq12_y[:, 1:2]
    conservation13 = du13[:, 2:3] + dq13_x[:, 0:1] + dq13_y[:, 1:2]
    return darcy12, darcy13, conservation12, conservation13


def neumann_normal_derivative(model: torch.nn.Module, xyt: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted normal flux q dot n on outer boundaries."""

    z = model.normalize_xyt(xyt)
    output = model.forward_normalized(z)
    _u12, _u13, q12_x, q12_y, q13_x, q13_y = split_mixed_output(output)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    return q12_x * nx + q12_y * ny, q13_x * nx + q13_y * ny


def interface_offset_points(
    xyt_interface: torch.Tensor,
    normal: torch.Tensor,
    eps: float,
    geometry: ReservoirGeometry,
    minus_region: int,
    plus_region: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Offset interface points to valid minus/plus material sides."""

    if eps <= 0.0:
        raise ValueError(f"Interface offset eps must be positive, got {eps:g}.")
    offset = torch.cat([normal * float(eps), torch.zeros_like(xyt_interface[:, 2:3])], dim=1)
    xyt_minus = xyt_interface - offset
    xyt_plus = xyt_interface + offset
    region_minus = geometry.region_id_torch(xyt_minus[:, 0:1], xyt_minus[:, 1:2]).view(-1)
    region_plus = geometry.region_id_torch(xyt_plus[:, 0:1], xyt_plus[:, 1:2]).view(-1)
    mask = (region_minus == int(minus_region)) & (region_plus == int(plus_region))
    return xyt_minus[mask], xyt_plus[mask], normal[mask], mask


def _normal_flux(model: torch.nn.Module, xyt: torch.Tensor, normal: torch.Tensor, region_name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return q12 dot n and q13 dot n from the explicit flux outputs."""

    z = model.normalize_xyt(xyt)
    output = _forward_normalized_for_region(model, z, region_name)
    _u12, _u13, q12_x, q12_y, q13_x, q13_y = split_mixed_output(output)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    return q12_x * nx + q12_y * ny, q13_x * nx + q13_y * ny


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
    """Return pressure jump and normal-flux jump across an interface."""

    xyt_minus, xyt_plus, normal_valid, mask = interface_offset_points(
        xyt_interface,
        normal,
        eps,
        geometry,
        minus_region,
        plus_region,
    )
    if xyt_minus.shape[0] == 0:
        empty2 = xyt_interface.new_zeros((0, 2))
        return empty2, empty2, empty2, mask

    u_minus = model(xyt_minus)[:, 0:2]
    u_plus = model(xyt_plus)[:, 0:2]
    pressure_jump = u_minus - u_plus

    minus_name = _region_name_from_id(minus_region)
    plus_name = _region_name_from_id(plus_region)
    q12_minus, q13_minus = _normal_flux(model, xyt_minus, normal_valid, minus_name)
    q12_plus, q13_plus = _normal_flux(model, xyt_plus, normal_valid, plus_name)
    raw_flux_jump = torch.cat([q12_minus - q12_plus, q13_minus - q13_plus], dim=1)

    scale12_minus = dimensionless_pde_coefficients(physics_cfg, config, "u12", minus_name, raw_flux_jump.device, raw_flux_jump.dtype)["residual_scale"]
    scale12_plus = dimensionless_pde_coefficients(physics_cfg, config, "u12", plus_name, raw_flux_jump.device, raw_flux_jump.dtype)["residual_scale"]
    scale13_minus = dimensionless_pde_coefficients(physics_cfg, config, "u13", minus_name, raw_flux_jump.device, raw_flux_jump.dtype)["residual_scale"]
    scale13_plus = dimensionless_pde_coefficients(physics_cfg, config, "u13", plus_name, raw_flux_jump.device, raw_flux_jump.dtype)["residual_scale"]
    scale12 = torch.maximum(scale12_minus, scale12_plus)
    scale13 = torch.maximum(scale13_minus, scale13_plus)
    normalized_flux_jump = torch.cat([raw_flux_jump[:, 0:1] / scale12, raw_flux_jump[:, 1:2] / scale13], dim=1)
    return pressure_jump, normalized_flux_jump, raw_flux_jump, mask
