"""Mesh-free local integral conservation residuals.

The rectangles sampled here are random local control volumes, not cells from a
fixed FVM/EDFM grid. They provide a conservative weak-form check using only the
PINN field and first derivatives.
"""

from __future__ import annotations

from typing import Any

import torch

from .physics import dimensionless_pde_coefficients, grad


def _zero_from_model(model: torch.nn.Module) -> torch.Tensor:
    return next(model.parameters()).sum() * 0.0


def _forward_region_normalized(model: torch.nn.Module, z: torch.Tensor, region_name: str) -> torch.Tensor:
    if hasattr(model, "forward_region_normalized"):
        return model.forward_region_normalized(z, region_name)
    return model.forward_normalized(z)


def _xyt_from_offsets(center: torch.Tensor, half_size: torch.Tensor, t: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    ox = offsets[:, 0].view(1, -1)
    oy = offsets[:, 1].view(1, -1)
    x = center[:, 0:1] + half_size[:, 0:1] * ox
    y = center[:, 1:2] + half_size[:, 1:2] * oy
    tt = t.expand(-1, offsets.shape[0])
    return torch.stack([x.reshape(-1), y.reshape(-1), tt.reshape(-1)], dim=1)


def _region_gradients(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    region_name: str,
) -> torch.Tensor:
    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_region_normalized(model, z, region_name)
    return grad(u[:, 0:1], z)


def _rect_residual_for_region(
    model: torch.nn.Module,
    samples: dict[str, torch.Tensor],
    region_name: str,
    config: dict[str, Any],
) -> torch.Tensor:
    center = samples.get("center")
    half_size = samples.get("half_size")
    t = samples.get("t")
    if center is None or half_size is None or t is None or center.numel() == 0:
        return _zero_from_model(model).reshape(1)[:0]

    device = center.device
    dtype = center.dtype
    n_rect = center.shape[0]
    gauss = torch.as_tensor(1.0 / (3.0**0.5), dtype=dtype, device=device)
    interior_offsets = torch.stack(
        [
            torch.stack([-gauss, -gauss]),
            torch.stack([-gauss, gauss]),
            torch.stack([gauss, -gauss]),
            torch.stack([gauss, gauss]),
        ],
        dim=0,
    )
    line_offsets = torch.stack([-gauss, gauss])

    coeff = dimensionless_pde_coefficients(config["physics"], config, region_name, device, dtype)
    geom = config["geometry"]
    lx = float(geom["x_max"]) - float(geom["x_min"])
    ly = float(geom["y_max"]) - float(geom["y_min"])
    hx_hat = half_size[:, 0:1] / lx
    hy_hat = half_size[:, 1:2] / ly
    area_hat = 4.0 * hx_hat * hy_hat

    interior_xyt = _xyt_from_offsets(center, half_size, t, interior_offsets)
    interior_du = _region_gradients(model, interior_xyt, region_name)
    storage_integral = interior_du[:, 2].reshape(n_rect, -1).mean(dim=1, keepdim=True) * area_hat

    flux_integral = torch.zeros_like(storage_integral)
    side_defs = [
        ("x", -1.0, torch.stack([torch.full_like(line_offsets, -1.0), line_offsets], dim=1), 2.0 * hy_hat),
        ("x", 1.0, torch.stack([torch.full_like(line_offsets, 1.0), line_offsets], dim=1), 2.0 * hy_hat),
        ("y", -1.0, torch.stack([line_offsets, torch.full_like(line_offsets, -1.0)], dim=1), 2.0 * hx_hat),
        ("y", 1.0, torch.stack([line_offsets, torch.full_like(line_offsets, 1.0)], dim=1), 2.0 * hx_hat),
    ]
    for axis, normal, offsets, side_len in side_defs:
        side_xyt = _xyt_from_offsets(center, half_size, t, offsets)
        side_du = _region_gradients(model, side_xyt, region_name)
        if axis == "x":
            qn = coeff["kappa_x"] * side_du[:, 0:1] * float(normal)
        else:
            qn = coeff["kappa_y"] * side_du[:, 1:2] * float(normal)
        flux_integral = flux_integral + qn.reshape(n_rect, -1).mean(dim=1, keepdim=True) * side_len

    residual = (storage_integral - flux_integral) / torch.clamp(area_hat, min=torch.finfo(dtype).eps)
    return residual / coeff["residual_scale"]


def _mse(value: torch.Tensor) -> torch.Tensor:
    if value.numel() == 0:
        return value.sum() * 0.0
    return torch.mean(value**2)


def _rms(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(_mse(value), min=0.0))


def compute_local_conservation_loss(
    model: torch.nn.Module,
    samples: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    residuals = compute_local_conservation_residuals(model, samples, config)
    residual_srv = residuals["srv"]
    residual_usrv = residuals["usrv"]
    loss_srv = _mse(residual_srv)
    loss_usrv = _mse(residual_usrv)
    loss = 0.5 * (loss_srv + loss_usrv)
    stacked = torch.cat([residual_srv.reshape(-1), residual_usrv.reshape(-1)], dim=0)
    rms = _rms(stacked) if stacked.numel() > 0 else loss * 0.0
    return loss, {
        "loss_local_conservation": loss,
        "loss_local_conservation_srv": loss_srv,
        "loss_local_conservation_usrv": loss_usrv,
        "rms_local_conservation": rms,
    }


def compute_local_conservation_residuals(
    model: torch.nn.Module,
    samples: dict[str, dict[str, torch.Tensor]],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if not samples:
        zero = _zero_from_model(model)
        empty = zero.reshape(1)[:0]
        return {"srv": empty, "usrv": empty}
    return {
        "srv": _rect_residual_for_region(model, samples.get("srv", {}), "SRV", config),
        "usrv": _rect_residual_for_region(model, samples.get("usrv", {}), "USRV", config),
    }
