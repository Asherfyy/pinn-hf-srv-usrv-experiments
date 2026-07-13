"""PDE 残差、Neumann 通量和界面连续条件。

第一版使用归一化坐标上的无量纲扩散型 PDE：

    A_r u_t - D_r (u_xx + alpha_y u_yy) = 0

这里的 D 是无量纲有效扩散系数，并不直接等同于 COMSOL 原始物性参数。后续若需要
对接 COMSOL，可根据长度、时间和压力尺度把 COMSOL 冻结系数换算后替换配置中的 D。
"""

from __future__ import annotations

from typing import Any

import torch

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry


def grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """计算 outputs 对 inputs 的一阶导数。"""

    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def second_grad(outputs: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """计算 outputs 对每个输入分量的二阶导数对角项。

    返回张量第 i 列对应 d²outputs/dinputs_i²。这里只需要 PDE 中的 x、y、t
    坐标方向二阶导，不需要完整 Hessian 矩阵。
    """

    first = grad(outputs, inputs)
    cols = []
    for dim in range(inputs.shape[1]):
        first_col = first[:, dim : dim + 1]
        if not first_col.requires_grad:
            # 当输出对某个坐标的一阶导严格为常数 0 时，PyTorch 会返回不带 grad_fn 的
            # 零张量。数学上的二阶导也应为 0，因此这里直接补零，避免零残差模型报错。
            second = torch.zeros_like(first_col)
        else:
            second = grad(first_col, inputs)[:, dim : dim + 1]
        cols.append(second)
    return torch.cat(cols, dim=1)


def normalize_xyt(xyt: torch.Tensor, geom_cfg: dict[str, Any]) -> torch.Tensor:
    """物理坐标转归一化坐标。

    `geom_cfg` 可以只包含几何范围；若包含 `t_min/t_max`，则使用其中的时间范围。
    """

    x_min = float(geom_cfg["x_min"])
    x_max = float(geom_cfg["x_max"])
    y_min = float(geom_cfg["y_min"])
    y_max = float(geom_cfg["y_max"])
    t_min = float(geom_cfg.get("t_min", 0.0))
    t_max = float(geom_cfg.get("t_max", 1000.0))

    x_hat = (xyt[:, 0:1] - x_min) / (x_max - x_min)
    y_hat = (xyt[:, 1:2] - y_min) / (y_max - y_min)
    t_hat = (xyt[:, 2:3] - t_min) / (t_max - t_min)
    return torch.cat([x_hat, y_hat, t_hat], dim=1)


def denormalize_xyt(xyt_hat: torch.Tensor, geom_cfg: dict[str, Any]) -> torch.Tensor:
    """归一化坐标转物理坐标。"""

    x_min = float(geom_cfg["x_min"])
    x_max = float(geom_cfg["x_max"])
    y_min = float(geom_cfg["y_min"])
    y_max = float(geom_cfg["y_max"])
    t_min = float(geom_cfg.get("t_min", 0.0))
    t_max = float(geom_cfg.get("t_max", 1000.0))

    x = xyt_hat[:, 0:1] * (x_max - x_min) + x_min
    y = xyt_hat[:, 1:2] * (y_max - y_min) + y_min
    t = xyt_hat[:, 2:3] * (t_max - t_min) + t_min
    return torch.cat([x, y, t], dim=1)


def select_region_coefficients(region_onehot: torch.Tensor, coeff_dict: dict[str, float]) -> torch.Tensor:
    """根据区域 one-hot 选择分区常数系数。

    这里使用张量乘法，而不是逐点 Python 循环赋值，保证计算图和批量计算都保持简洁。
    """

    device = region_onehot.device
    dtype = region_onehot.dtype
    hf = torch.as_tensor(float(coeff_dict["HF"]), dtype=dtype, device=device)
    srv = torch.as_tensor(float(coeff_dict["SRV"]), dtype=dtype, device=device)
    usrv = torch.as_tensor(float(coeff_dict["USRV"]), dtype=dtype, device=device)
    return region_onehot[:, 0:1] * hf + region_onehot[:, 1:2] * srv + region_onehot[:, 2:3] * usrv


def _region_coeff_dict(
    coeffs: dict[str, Any],
    prefix: str,
    fallback_group: str | None = None,
) -> dict[str, float]:
    """Build an HF/SRV/USRV coefficient dict from COMSOL-style flat names."""

    keys = {region: f"{prefix}_{region}" for region in ("HF", "SRV", "USRV")}
    if all(name in coeffs for name in keys.values()):
        return {region: float(coeffs[name]) for region, name in keys.items()}
    if fallback_group is not None and fallback_group in coeffs:
        return {region: float(value) for region, value in coeffs[fallback_group].items()}
    raise KeyError(f"缺少 {prefix}_HF/{prefix}_SRV/{prefix}_USRV 系数")


def _anisotropy(coeffs: dict[str, Any], variable_index: int) -> float:
    """Return y-direction anisotropy for P1/P2 with old-key fallback."""

    new_name = f"anisotropy_y_{variable_index}"
    old_name = "anisotropy_y_12" if variable_index == 1 else "anisotropy_y_13"
    return float(coeffs.get(new_name, coeffs.get(old_name, 1.0)))


def _use_physical_scaling(coeffs: dict[str, Any]) -> bool:
    """Whether COMSOL coefficients are interpreted in physical m/s units."""

    return bool(coeffs.get("use_physical_scaling", False))


def _coordinate_scales(config: dict[str, Any], coeffs: dict[str, Any]) -> tuple[float, float, float]:
    """Return physical scales converting normalized derivatives to m and s."""

    geom = config["geometry"]
    sampler = config["sampler"]
    lx = float(geom["x_max"]) - float(geom["x_min"])
    ly = float(geom["y_max"]) - float(geom["y_min"])
    t_days = float(sampler["t_max"]) - float(sampler["t_min"])
    seconds_per_day = float(coeffs.get("seconds_per_day", 86400.0))
    return lx, ly, t_days * seconds_per_day


def _pde_time_scale_seconds(config: dict[str, Any], coeffs: dict[str, Any]) -> float:
    """Reference time used to make the physical PDE residual dimensionless."""

    seconds_per_day = float(coeffs.get("seconds_per_day", 86400.0))
    default_days = float(config["sampler"]["t_max"]) - float(config["sampler"]["t_min"])
    return float(coeffs.get("pde_time_scale_days", default_days)) * seconds_per_day


def _dimensionless_diffusion(
    diffusion: torch.Tensor,
    alpha_y: float,
    config: dict[str, Any],
    coeffs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert COMSOL diffusion D[m2/s] to dimensionless x/y coefficients.

    With x=Lx*x_hat, y=Ly*y_hat, t=Ttrain*t_hat and P=P0+Pscale*u, the
    pressure scale cancels. Multiplying the physical residual by a global
    reference time Tref gives:

        Fai*Tref/Ttrain*u_t_hat
        - D*Tref/Lx^2*u_xx_hat
        - alpha*D*Tref/Ly^2*u_yy_hat = 0
    """

    lx, ly, _t_span_seconds = _coordinate_scales(config, coeffs)
    t_ref_seconds = _pde_time_scale_seconds(config, coeffs)
    d_x = diffusion * t_ref_seconds / (lx * lx)
    d_y = diffusion * t_ref_seconds * float(alpha_y) / (ly * ly)
    return d_x, d_y


def _dimensionless_storage(storage: torch.Tensor, config: dict[str, Any], coeffs: dict[str, Any]) -> torch.Tensor:
    """Convert Fai to the time-derivative coefficient in the dimensionless PDE."""

    _lx, _ly, t_span_seconds = _coordinate_scales(config, coeffs)
    t_ref_seconds = _pde_time_scale_seconds(config, coeffs)
    return storage * (t_ref_seconds / t_span_seconds)


def _dimensionless_flux_components(
    diffusion: torch.Tensor,
    alpha_y: float,
    config: dict[str, Any],
    coeffs: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert physical flux coefficients to dimensionless boundary flux coefficients.

    The physical zero-flux condition is n·D∇u=0.  Multiplying it by
    Tref/Lref keeps the zero set unchanged and makes the residual dimensionless:

        q_hat = D*Tref/Lref * (n_x/Lx*u_x_hat + alpha*n_y/Ly*u_y_hat)
    """

    lx, ly, _t_span_seconds = _coordinate_scales(config, coeffs)
    l_ref = max(lx, ly)
    t_ref_seconds = _pde_time_scale_seconds(config, coeffs)
    q_x = diffusion * t_ref_seconds / (l_ref * lx)
    q_y = diffusion * t_ref_seconds * float(alpha_y) / (l_ref * ly)
    return q_x, q_y


def _canonical_region_name(region_name: str) -> str:
    """规范化区域名称，避免调用端大小写差异造成查表错误。"""

    name = str(region_name).upper()
    if name not in {"HF", "SRV", "USRV"}:
        raise ValueError(f"不支持的区域名称: {region_name}，请使用 HF/SRV/USRV。")
    return name


def _canonical_variable_index(variable_name: str) -> int:
    """把 u12/u13/P12/P13 等名称统一映射到变量编号 1 或 2。"""

    name = str(variable_name).lower()
    aliases_1 = {"1", "u1", "u12", "p1", "p12", "pg1"}
    aliases_2 = {"2", "u2", "u13", "p2", "p13", "pg2"}
    if name in aliases_1:
        return 1
    if name in aliases_2:
        return 2
    raise ValueError(f"不支持的变量名称: {variable_name}，请使用 u12/u13。")


def _scalar_coefficient_tensor(
    coeff_dict: dict[str, float],
    region_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """从 HF/SRV/USRV 系数字典中取一个标量张量。"""

    region = _canonical_region_name(region_name)
    if region not in coeff_dict:
        raise KeyError(f"系数字典缺少区域 {region}: {coeff_dict}")
    return torch.as_tensor(float(coeff_dict[region]), dtype=dtype, device=device)


def get_dimensionless_pde_coefficients(
    coeffs: dict[str, Any],
    config: dict[str, Any],
    variable_name: str,
    region_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """返回指定变量和区域的无量纲 PDE 系数及固定残差尺度。

    该函数复用 `pde_residual()` 当前的物理尺度转换逻辑。loss 层只负责用返回的
    `residual_scale` 归一化残差，不在 `losses.py` 中重复推导 Dx/Dy，防止训练和诊断
    使用两套不同系数。
    """

    variable_index = _canonical_variable_index(variable_name)
    region = _canonical_region_name(region_name)
    if variable_index == 1:
        storage_dict = _region_coeff_dict(coeffs, "Fai", "A12")
        diffusion_dict = _region_coeff_dict(coeffs, "D1", "D12")
    else:
        storage_dict = _region_coeff_dict(coeffs, "Fai", "A13")
        diffusion_dict = _region_coeff_dict(coeffs, "D2", "D13")

    storage = _scalar_coefficient_tensor(storage_dict, region, device, dtype)
    diffusion = _scalar_coefficient_tensor(diffusion_dict, region, device, dtype)
    alpha_y = _anisotropy(coeffs, variable_index)

    if _use_physical_scaling(coeffs):
        storage_hat = _dimensionless_storage(storage, config, coeffs)
        diffusion_x, diffusion_y = _dimensionless_diffusion(diffusion, alpha_y, config, coeffs)
    else:
        # 非物理缩放分支与 pde_residual() 完全一致：Dx=D，Dy=alpha_y*D。
        storage_hat = storage
        diffusion_x = diffusion
        diffusion_y = diffusion * float(alpha_y)

    eps = float(config.get("pde_residual_normalization", {}).get("eps", 1.0e-12))
    if eps <= 0.0:
        raise ValueError(f"pde_residual_normalization.eps 必须为正数，当前为 {eps:g}。")
    eps_tensor = torch.as_tensor(eps, dtype=dtype, device=device)
    residual_scale = torch.maximum(
        torch.maximum(torch.abs(storage_hat), torch.abs(diffusion_x)),
        torch.maximum(torch.abs(diffusion_y), eps_tensor),
    )
    if not torch.isfinite(residual_scale).item() or residual_scale.item() <= 0.0:
        raise ValueError(
            f"{variable_name}_{region} 的 PDE 残差尺度非法: {float(residual_scale.detach().cpu()):g}。"
        )
    return {
        "storage": storage_hat,
        "diffusion_x": diffusion_x,
        "diffusion_y": diffusion_y,
        "residual_scale": residual_scale,
    }


def _scale_cfg(config: dict[str, Any]) -> dict[str, Any]:
    """合并几何与时间范围，供归一化函数使用。"""

    merged = dict(config["geometry"])
    merged["t_min"] = config["sampler"]["t_min"]
    merged["t_max"] = config["sampler"]["t_max"]
    return merged


def _model_for_derivatives(model: torch.nn.Module, xyt: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """Evaluate model while optionally removing distance-feature coordinate derivatives."""

    detach_features = bool(config.get("model", {}).get("detach_distance_features_for_derivatives", True))
    if not detach_features:
        return model(xyt)
    try:
        return model(xyt, detach_distance_features=True)
    except TypeError:
        return model(xyt)


def pde_residual(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算两变量扩散 PDE 残差。

    输入点是物理坐标，但导数针对归一化坐标求取。这样 x、y、t 的尺度都在 O(1)，
    可以减小自动微分得到的导数量级差异，使 CPU 训练更稳定。
    """

    scale_cfg = _scale_cfg(config)
    xyt_hat = normalize_xyt(xyt, scale_cfg).detach().clone().requires_grad_(True)
    xyt_phys = denormalize_xyt(xyt_hat, scale_cfg)
    u = _model_for_derivatives(model, xyt_phys, config)

    du12 = grad(u[:, 0:1], xyt_hat)
    du13 = grad(u[:, 1:2], xyt_hat)
    d2u12 = second_grad(u[:, 0:1], xyt_hat)
    d2u13 = second_grad(u[:, 1:2], xyt_hat)

    x = xyt_phys[:, 0:1]
    y = xyt_phys[:, 1:2]
    region_onehot = geometry.region_onehot_torch(x, y)

    a12 = select_region_coefficients(region_onehot, _region_coeff_dict(coeffs, "Fai", "A12"))
    a13 = select_region_coefficients(region_onehot, _region_coeff_dict(coeffs, "Fai", "A13"))
    d12 = select_region_coefficients(region_onehot, _region_coeff_dict(coeffs, "D1", "D12"))
    d13 = select_region_coefficients(region_onehot, _region_coeff_dict(coeffs, "D2", "D13"))
    alpha12 = _anisotropy(coeffs, 1)
    alpha13 = _anisotropy(coeffs, 2)

    if _use_physical_scaling(coeffs):
        a12_t = _dimensionless_storage(a12, config, coeffs)
        a13_t = _dimensionless_storage(a13, config, coeffs)
        d12_x, d12_y = _dimensionless_diffusion(d12, alpha12, config, coeffs)
        d13_x, d13_y = _dimensionless_diffusion(d13, alpha13, config, coeffs)
        r12 = a12_t * du12[:, 2:3] - d12_x * d2u12[:, 0:1] - d12_y * d2u12[:, 1:2]
        r13 = a13_t * du13[:, 2:3] - d13_x * d2u13[:, 0:1] - d13_y * d2u13[:, 1:2]
    else:
        r12 = a12 * du12[:, 2:3] - d12 * (d2u12[:, 0:1] + alpha12 * d2u12[:, 1:2])
        r13 = a13 * du13[:, 2:3] - d13 * (d2u13[:, 0:1] + alpha13 * d2u13[:, 1:2])
    return r12, r13


def _flux_on_points(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    normal: torch.Tensor,
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算给定点沿法向的两变量扩散通量残差项。"""

    scale_cfg = _scale_cfg(config)
    xyt_hat = normalize_xyt(xyt, scale_cfg).detach().clone().requires_grad_(True)
    xyt_phys = denormalize_xyt(xyt_hat, scale_cfg)
    u = _model_for_derivatives(model, xyt_phys, config)

    du12 = grad(u[:, 0:1], xyt_hat)
    du13 = grad(u[:, 1:2], xyt_hat)
    region_onehot = geometry.region_onehot_torch(xyt_phys[:, 0:1], xyt_phys[:, 1:2])
    d12 = select_region_coefficients(region_onehot, _region_coeff_dict(coeffs, "D1", "D12"))
    d13 = select_region_coefficients(region_onehot, _region_coeff_dict(coeffs, "D2", "D13"))
    alpha12 = _anisotropy(coeffs, 1)
    alpha13 = _anisotropy(coeffs, 2)

    nx = normal[:, 0:1]
    ny = normal[:, 1:2]
    if _use_physical_scaling(coeffs):
        q12_x, q12_y = _dimensionless_flux_components(d12, alpha12, config, coeffs)
        q13_x, q13_y = _dimensionless_flux_components(d13, alpha13, config, coeffs)
        q12 = q12_x * du12[:, 0:1] * nx + q12_y * du12[:, 1:2] * ny
        q13 = q13_x * du13[:, 0:1] * nx + q13_y * du13[:, 1:2] * ny
    else:
        q12 = d12 * (du12[:, 0:1] * nx + alpha12 * du12[:, 1:2] * ny)
        q13 = d13 * (du13[:, 0:1] * nx + alpha13 * du13[:, 1:2] * ny)
    return q12, q13


def neumann_flux_residual(
    model: torch.nn.Module,
    xyt: torch.Tensor,
    normal: torch.Tensor,
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算无流 Neumann 边界残差。

    无流边界要求 n·D∇u=0，因此残差就是沿外法向的扩散通量。训练时将该残差以
    soft constraint 加入总损失。
    """

    return _flux_on_points(model, xyt, normal, geometry, coeffs, config)


def _clamp_to_domain(xyt: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """把偏移后的界面点限制在总计算域内。"""

    geom_cfg = config["geometry"]
    x = torch.clamp(xyt[:, 0:1], float(geom_cfg["x_min"]), float(geom_cfg["x_max"]))
    y = torch.clamp(xyt[:, 1:2], float(geom_cfg["y_min"]), float(geom_cfg["y_max"]))
    t = xyt[:, 2:3]
    return torch.cat([x, y, t], dim=1)


def _is_region(region_id: torch.Tensor, allowed_regions: tuple[int, ...]) -> torch.Tensor:
    """判断区域编号是否属于允许集合。"""

    mask = torch.zeros_like(region_id, dtype=torch.bool)
    for region in allowed_regions:
        mask = mask | (region_id == int(region))
    return mask


def interface_offset_points(
    xyt_interface: torch.Tensor,
    normal: torch.Tensor,
    eps: float,
    geometry: ReservoirGeometry,
    minus_regions: tuple[int, ...],
    plus_regions: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """沿界面法向生成两侧偏移点，并检查它们是否落在正确分区。

    旧实现只做 `x ± eps*n`，然后直接参与界面损失；如果点落到了错误区域，例如裂缝交叉
    处 plus 侧仍在 HF，通量连续条件就会被错误施加。这里显式检查 minus/plus 的区域，
    只保留真正跨越目标界面的点。
    """

    offset = torch.cat([normal * float(eps), torch.zeros_like(xyt_interface[:, 2:3])], dim=1)
    xyt_minus = xyt_interface - offset
    xyt_plus = xyt_interface + offset

    region_minus = geometry.region_id_torch(xyt_minus[:, 0:1], xyt_minus[:, 1:2])
    region_plus = geometry.region_id_torch(xyt_plus[:, 0:1], xyt_plus[:, 1:2])
    mask = _is_region(region_minus, minus_regions) & _is_region(region_plus, plus_regions)
    mask = mask.view(-1)
    return xyt_minus[mask], xyt_plus[mask], normal[mask], mask


def interface_residual(
    model: torch.nn.Module,
    xyt_interface: torch.Tensor,
    normal: torch.Tensor,
    eps: float,
    geometry: ReservoirGeometry,
    coeffs: dict[str, Any],
    config: dict[str, Any],
    minus_regions: tuple[int, ...] = (REGION_HF,),
    plus_regions: tuple[int, ...] = (REGION_SRV,),
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算界面压力连续和通量连续残差。

    界面本身属于零厚度几何，自动微分不适合直接在同一点同时表示两侧材料。
    因此沿法向做很小偏移：minus 侧通常落在界面内侧，plus 侧落在外侧，再分别计算
    压力和法向通量。eps 由采样配置区分 HF-SRV 与 SRV-USRV 界面。
    """

    xyt_minus, xyt_plus, normal_valid, _mask = interface_offset_points(
        xyt_interface,
        normal,
        eps,
        geometry,
        minus_regions,
        plus_regions,
    )
    if xyt_minus.shape[0] == 0:
        empty = xyt_interface.new_zeros((0, 2))
        return empty, empty

    u_minus = model(xyt_minus)
    u_plus = model(xyt_plus)
    pressure_jump = u_minus - u_plus

    q12_minus, q13_minus = _flux_on_points(model, xyt_minus, normal_valid, geometry, coeffs, config)
    q12_plus, q13_plus = _flux_on_points(model, xyt_plus, normal_valid, geometry, coeffs, config)
    flux_jump = torch.cat([q12_minus - q12_plus, q13_minus - q13_plus], dim=1)
    return pressure_jump, flux_jump
