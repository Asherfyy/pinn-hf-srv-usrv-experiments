"""PINN 损失函数。

总损失由 PDE 残差、初始条件、Dirichlet 生产边界、Neumann 边界、界面压力连续、
界面通量连续和可选数据项组成。每一项独立成函数，便于调试时观察到底是物理残差、
边界条件还是界面约束主导。
"""

from __future__ import annotations

from typing import Any

import torch

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry
from .physics import (
    get_dimensionless_pde_coefficients,
    interface_residual,
    neumann_flux_residual,
    pde_residual,
)
from .utils import initial_pressure_pair_mpa, pressure_mpa_to_hat


def _mse(value: torch.Tensor) -> torch.Tensor:
    """安全 MSE：空张量返回同设备零值。"""

    if value.numel() == 0:
        return value.sum() * 0.0
    return torch.mean(value**2)


def _rms(value: torch.Tensor) -> torch.Tensor:
    """计算 RMS；空张量返回同设备零值，便于界面过滤后仍能安全记录诊断。"""

    return torch.sqrt(torch.clamp(_mse(value), min=0.0))


def _pde_component_weight(config: dict[str, Any], variable_name: str, region_name: str) -> float:
    """读取六个 PDE 分量的独立权重，缺省为 1.0。"""

    weights = config.get("pde_residual_normalization", {}).get("component_weights", {})
    key = f"{variable_name}_{region_name.upper()}"
    return float(weights.get(key, 1.0))


def compute_pde_loss(
    model: torch.nn.Module,
    pde_points: dict[str, torch.Tensor],
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """分区域、分变量计算固定尺度归一化 PDE loss。

    旧版把 HF/SRV/USRV 采样点拼接后统一 MSE，容易让大系数区域主导优化。这里分别
    计算 R12/R13 在 HF、SRV、USRV 的六个分量，并使用 `physics.py` 返回的无量纲
    系数尺度做归一化，从而让每个区域和变量都能独立进入总损失。
    """

    region_items = [("hf", "HF"), ("srv", "SRV"), ("usrv", "USRV")]
    normalization_cfg = config.get("pde_residual_normalization", {})
    normalization_enabled = bool(normalization_cfg.get("enabled", True))
    diagnostics: dict[str, torch.Tensor] = {}
    total = pde_points["hf"].sum() * 0.0

    for region_key, region_name in region_items:
        xyt_region = pde_points[region_key]
        r12, r13 = pde_residual(model, xyt_region, geometry, coeffs, config)
        for variable_name, residual in [("u12", r12), ("u13", r13)]:
            if normalization_enabled:
                coeff_info = get_dimensionless_pde_coefficients(
                    coeffs,
                    config,
                    variable_name,
                    region_name,
                    device=residual.device,
                    dtype=residual.dtype,
                )
                scale = coeff_info["residual_scale"]
            else:
                # 关闭归一化时仍保持分区域/分变量诊断；scale=1 使 raw 和 normalized 相同。
                scale = torch.as_tensor(1.0, dtype=residual.dtype, device=residual.device)
            normalized = residual / scale
            component_loss = _mse(normalized)
            weight = _pde_component_weight(config, variable_name, region_name)
            total = total + weight * component_loss

            suffix = f"{variable_name}_{region_key}"
            diagnostics[f"loss_pde_{suffix}"] = component_loss
            diagnostics[f"rms_raw_{suffix}"] = _rms(residual)
            diagnostics[f"rms_normalized_{suffix}"] = _rms(normalized)

    return total, diagnostics


def _initial_target_for_region(
    region_name: str,
    n: int,
    device: torch.device,
    dtype: torch.dtype,
    config: dict[str, Any],
) -> torch.Tensor:
    """构造指定区域的初始无量纲压力目标。"""

    prefix = region_name.lower()
    p12, p13 = initial_pressure_pair_mpa(config, prefix)
    u12, u13 = pressure_mpa_to_hat(p12, p13, config["boundary"])
    target = torch.tensor([[u12, u13]], dtype=dtype, device=device)
    return target.repeat(n, 1)


def compute_initial_loss(
    model: torch.nn.Module,
    initial_points: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> torch.Tensor:
    """计算初始条件损失。"""

    losses = []
    for key, region_name in [("hf", "hf"), ("srv", "srv"), ("usrv", "usrv")]:
        xyt = initial_points[key]
        pred = model(xyt)
        target = _initial_target_for_region(region_name, xyt.shape[0], xyt.device, xyt.dtype, config)
        losses.append(_mse(pred - target))
    return torch.stack(losses).sum()


def compute_dirichlet_loss(
    model: torch.nn.Module,
    dirichlet_points: dict[str, torch.Tensor],
) -> torch.Tensor:
    """计算生产端 Dirichlet 压力边界 soft constraint 损失。

    在 `dirichlet_hard` 模式下，该项理论上应接近 0，只作为诊断项存在；
    在 `direct` 和 `ic_hard` 模式下，该项是生产边界低压进入训练目标的主要方式。
    """

    xyt = dirichlet_points["xyt"]
    pred = model(xyt)
    target = model.boundary_value_hat(xyt[:, 2:3])
    return _mse(pred - target)


def compute_neumann_loss(
    model: torch.nn.Module,
    neumann_points: dict[str, torch.Tensor],
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
) -> torch.Tensor:
    """计算外边界无流 Neumann soft constraint 损失。"""

    r12, r13 = neumann_flux_residual(
        model,
        neumann_points["xyt"],
        neumann_points["normal"],
        geometry,
        coeffs,
        config,
    )
    return _mse(r12) + _mse(r13)


def compute_interface_loss(
    model: torch.nn.Module,
    interface_hf_srv: dict[str, torch.Tensor],
    interface_srv_usrv: dict[str, torch.Tensor],
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 HF-SRV 与 SRV-USRV 的压力连续和通量连续损失。"""

    sampler_cfg = config["sampler"]
    pressure_hf, flux_hf = interface_residual(
        model,
        interface_hf_srv["xyt"],
        interface_hf_srv["normal"],
        float(sampler_cfg["eps_hf_srv"]),
        geometry,
        coeffs,
        config,
        minus_regions=(REGION_HF,),
        plus_regions=(REGION_SRV,),
    )
    pressure_srv, flux_srv = interface_residual(
        model,
        interface_srv_usrv["xyt"],
        interface_srv_usrv["normal"],
        float(sampler_cfg["eps_srv_usrv"]),
        geometry,
        coeffs,
        config,
        minus_regions=(REGION_SRV,),
        plus_regions=(REGION_USRV,),
    )
    pressure_loss = _mse(pressure_hf) + _mse(pressure_srv)
    flux_loss = _mse(flux_hf) + _mse(flux_srv)
    return pressure_loss, flux_loss


def compute_data_loss(
    model: torch.nn.Module,
    data_batch: dict[str, torch.Tensor] | None,
    config: dict[str, Any],
) -> torch.Tensor:
    """计算可选 COMSOL 数据监督损失。

    默认权重为 0，因此没有数据时返回图中零值即可。若未来希望混合数据驱动项，
    传入包含 `xyt` 与 `target_hat` 的字典即可。
    """

    zero = next(model.parameters()).sum() * 0.0
    if data_batch is None or "xyt" not in data_batch or data_batch["xyt"].numel() == 0:
        return zero
    pred = model(data_batch["xyt"])
    return _mse(pred - data_batch["target_hat"])


def compute_total_loss(
    model: torch.nn.Module,
    samples: dict[str, Any],
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
    data_batch: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """计算总损失并返回分项字典。"""

    weights = config["loss_weights"]
    loss_pde, pde_diagnostics = compute_pde_loss(model, samples["pde"], geometry, coeffs, config)
    loss_ic = compute_initial_loss(model, samples["initial"], config)
    loss_dirichlet = compute_dirichlet_loss(model, samples["dirichlet"])
    loss_neumann = compute_neumann_loss(model, samples["neumann"], geometry, coeffs, config)
    loss_interface_pressure, loss_interface_flux = compute_interface_loss(
        model,
        samples["interface_hf_srv"],
        samples["interface_srv_usrv"],
        geometry,
        coeffs,
        config,
    )
    loss_data = compute_data_loss(model, data_batch, config)

    total = (
        float(weights["pde"]) * loss_pde
        + float(weights["initial"]) * loss_ic
        + float(weights.get("dirichlet", 0.0)) * loss_dirichlet
        + float(weights["neumann"]) * loss_neumann
        + float(weights["interface_pressure"]) * loss_interface_pressure
        + float(weights["interface_flux"]) * loss_interface_flux
        + float(weights["data"]) * loss_data
    )
    loss_dict = {
        "loss_total": total,
        "loss_pde": loss_pde,
        "loss_ic": loss_ic,
        "loss_dirichlet": loss_dirichlet,
        "loss_neumann": loss_neumann,
        "loss_interface_pressure": loss_interface_pressure,
        "loss_interface_flux": loss_interface_flux,
        "loss_data": loss_data,
    }
    loss_dict.update(pde_diagnostics)
    return total, loss_dict
