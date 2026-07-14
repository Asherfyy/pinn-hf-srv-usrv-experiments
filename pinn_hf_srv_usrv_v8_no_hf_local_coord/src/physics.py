"""有效扩散形式 PDE 与边界导数计算。"""

from __future__ import annotations

from typing import Any

import torch

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, REGION_NAMES, ReservoirGeometry


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """一阶自动微分。"""

    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]


def second_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """返回对 x_hat/y_hat/t_hat 的二阶导对角项。"""

    first = grad(outputs, inputs)
    cols = []
    for dim in range(inputs.shape[1]):
        first_col = first[:, dim : dim + 1]
        if not first_col.requires_grad:
            cols.append(torch.zeros_like(first_col))
        else:
            cols.append(grad(first_col, inputs)[:, dim : dim + 1])
    return torch.cat(cols, dim=1)


def effective_diffusion(physics_cfg: dict[str, Any], variable_name: str, region_name: str) -> float:
    """计算 K=D/Fai。

    v3 只保留有效扩散形式，不再保留旧的 `Fai*u_t-D*u_xx` 分支。这样残差表达更直接，
    也更容易看清 HF/SRV/USRV 真实极端物理系数带来的时间尺度差异。
    """

    region = region_name.upper()
    if region not in REGION_NAMES:
        raise ValueError(f"未知区域: {region_name}")
    variable = variable_name.lower()
    d_group = "D1" if variable in {"u12", "p12", "1"} else "D2"
    fai = float(physics_cfg["Fai"][region])
    diffusion = float(physics_cfg[d_group][region])
    if fai <= 0.0:
        raise ValueError(f"Fai_{region} 必须为正。")
    return diffusion / fai


def dimensionless_pde_coefficients(
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
    variable_name: str,
    region_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """计算归一化坐标 PDE 的 kappa_x/kappa_y 和固定残差尺度。"""

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


def normal_flux_scale(
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
    variable_name: str,
    region_name: str,
    normal: torch.Tensor,
) -> torch.Tensor:
    """按界面法向计算通量跳跃归一化尺度。

    对法向通量 `kx*u_x*n_x + ky*u_y*n_y`，尺度应随界面方向变化：
    竖直界面主要由 kx 控制，水平界面主要由 ky 控制。使用统一
    `max(1, |kx|, |ky|)` 会在 ky > kx 时低估竖直界面的相对通量误差。
    """

    coeff = dimensionless_pde_coefficients(physics_cfg, config, variable_name, region_name, normal.device, normal.dtype)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    directional = torch.abs(coeff["kappa_x"] * nx) + torch.abs(coeff["kappa_y"] * ny)
    return torch.maximum(torch.ones_like(directional), directional)


def _region_name_from_id(region_id: int) -> str:
    """把区域编号转换为系数字典使用的区域名。"""

    mapping = {REGION_HF: "HF", REGION_SRV: "SRV", REGION_USRV: "USRV"}
    if int(region_id) not in mapping:
        raise ValueError(f"未知区域编号: {region_id}")
    return mapping[int(region_id)]


def _forward_normalized_for_region(model: torch.nn.Module, z: torch.Tensor, region_name: str) -> torch.Tensor:
    """Use an explicit region subnet when the model provides one."""

    if hasattr(model, "forward_region_normalized"):
        return model.forward_region_normalized(z, region_name)
    return model.forward_normalized(z)


def pde_residual(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    region_name: str,
    physics_cfg: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算指定区域的 u12/u13 有效扩散 PDE 残差。"""

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du12 = grad(u[:, 0:1], z)
    du13 = grad(u[:, 1:2], z)
    d2u12 = second_grad(u[:, 0:1], z)
    d2u13 = second_grad(u[:, 1:2], z)
    c12 = dimensionless_pde_coefficients(physics_cfg, config, "u12", region_name, z.device, z.dtype)
    c13 = dimensionless_pde_coefficients(physics_cfg, config, "u13", region_name, z.device, z.dtype)
    r12 = du12[:, 2:3] - c12["kappa_x"] * d2u12[:, 0:1] - c12["kappa_y"] * d2u12[:, 1:2]
    r13 = du13[:, 2:3] - c13["kappa_x"] * d2u13[:, 0:1] - c13["kappa_y"] * d2u13[:, 1:2]
    return r12, r13


def neumann_normal_derivative(model: torch.nn.Module, xyt: torch.Tensor, normal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """计算外边界归一化坐标法向导数。

    无流目标为 0，因此不乘极端扩散系数；这样 Neumann loss 约束的是边界梯度形状，
    不会被 HF 的巨大 K 值主导。
    """

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = model.forward_normalized(z)
    du12 = grad(u[:, 0:1], z)
    du13 = grad(u[:, 1:2], z)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    return du12[:, 0:1] * nx + du12[:, 1:2] * ny, du13[:, 0:1] * nx + du13[:, 1:2] * ny


def interface_offset_points(
    xyt_interface: torch.Tensor,
    normal: torch.Tensor,
    eps: float,
    geometry: ReservoirGeometry,
    minus_region: int,
    plus_region: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """沿界面法向生成两侧偏移点，并过滤到指定材料组合。

    约定 normal 从 minus 区域指向 plus 区域外侧；因此 `xyt - eps*n` 是 minus 侧，
    `xyt + eps*n` 是 plus 侧。过滤是必要的，因为矩形裂缝交叉、外边界和角点附近
    都可能让偏移点落入非目标区域。
    """

    if eps <= 0.0:
        raise ValueError(f"界面偏移 eps 必须为正，当前 {eps:g}。")
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算指定材料侧的有效扩散无量纲法向通量。

    对 `u_t - kx*u_xx - ky*u_yy = 0`，一致的扩散通量可写成
    `kx*u_x*n_x + ky*u_y*n_y`。界面通量连续 loss 使用同一套 kappa 系数，因此与 PDE
    残差的尺度定义保持一致。
    """

    z = model.normalize_xyt(xyt).detach().clone().requires_grad_(True)
    u = _forward_normalized_for_region(model, z, region_name)
    du12 = grad(u[:, 0:1], z)
    du13 = grad(u[:, 1:2], z)
    c12 = dimensionless_pde_coefficients(physics_cfg, config, "u12", region_name, z.device, z.dtype)
    c13 = dimensionless_pde_coefficients(physics_cfg, config, "u13", region_name, z.device, z.dtype)
    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    q12 = c12["kappa_x"] * du12[:, 0:1] * nx + c12["kappa_y"] * du12[:, 1:2] * ny
    q13 = c13["kappa_x"] * du13[:, 0:1] * nx + c13["kappa_y"] * du13[:, 1:2] * ny
    return q12, q13


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
    """计算界面压力跳跃和归一化通量跳跃。

    返回 pressure_jump、normalized_flux_jump、raw_flux_jump 和有效样本 mask。通量跳跃
    根据界面法向使用 `max(1, |kx*nx| + |ky*ny|)` 归一化，并取两侧材料尺度较大值。
    """

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

    u_minus = model(xyt_minus)
    u_plus = model(xyt_plus)
    pressure_jump = u_minus - u_plus

    minus_name = _region_name_from_id(minus_region)
    plus_name = _region_name_from_id(plus_region)
    q12_minus, q13_minus = _normal_flux(model, xyt_minus, normal_valid, minus_name, physics_cfg, config)
    q12_plus, q13_plus = _normal_flux(model, xyt_plus, normal_valid, plus_name, physics_cfg, config)
    raw_flux_jump = torch.cat([q12_minus - q12_plus, q13_minus - q13_plus], dim=1)

    scale12_minus = normal_flux_scale(physics_cfg, config, "u12", minus_name, normal_valid)
    scale12_plus = normal_flux_scale(physics_cfg, config, "u12", plus_name, normal_valid)
    scale13_minus = normal_flux_scale(physics_cfg, config, "u13", minus_name, normal_valid)
    scale13_plus = normal_flux_scale(physics_cfg, config, "u13", plus_name, normal_valid)
    scale12 = torch.maximum(scale12_minus, scale12_plus)
    scale13 = torch.maximum(scale13_minus, scale13_plus)
    normalized_flux_jump = torch.cat([raw_flux_jump[:, 0:1] / scale12, raw_flux_jump[:, 1:2] / scale13], dim=1)
    return pressure_jump, normalized_flux_jump, raw_flux_jump, mask
