"""项目通用工具函数。

该文件集中处理随机种子、CPU 运行环境、目录创建、日志写入和压力量纲转换。
这样各个脚本可以保持清晰，把注意力放在几何、物理和训练流程本身。
"""

from __future__ import annotations

import csv
import os
import random
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def _get_float(config: dict[str, Any], *names: str, default: float | None = None) -> float:
    """Return the first configured float from a list of compatible names."""

    for name in names:
        if name in config:
            return float(config[name])
    if default is not None:
        return float(default)
    raise KeyError(f"缺少配置项: {names}")


def boundary_p_t0_mpa(boundary_cfg: dict[str, Any]) -> float:
    """Initial total pressure in MPa, preferring the COMSOL name."""

    return _get_float(boundary_cfg, "P_t0", "p_t0_mpa")


def boundary_p_out_mpa(boundary_cfg: dict[str, Any]) -> float:
    """Outlet pressure in MPa, preferring the COMSOL name."""

    return _get_float(boundary_cfg, "P_out", "p_out_mpa")


def boundary_ratio(boundary_cfg: dict[str, Any]) -> float:
    """C13/C12 ratio, preferring the COMSOL name."""

    return _get_float(boundary_cfg, "C13_C12", "c13_c12_ratio")


def boundary_decay_rate(boundary_cfg: dict[str, Any]) -> float:
    """Exponential decline coefficient of COMSOL function an1."""

    return _get_float(boundary_cfg, "an1_decay_rate", "decay_rate")


def pressure_scale_mpa(boundary_cfg: dict[str, Any]) -> float:
    """返回旧版共享压力尺度，专供 legacy_shared 兼容模式使用。

    新训练默认使用 `pressure_component_affine_parameters()` 中的 P12/P13 独立尺度；
    这里保留旧函数，是为了旧配置或消融实验仍可运行，但业务代码不应再手写 `p_scale`。
    """

    if "p_scale_mpa" in boundary_cfg:
        scale = float(boundary_cfg["p_scale_mpa"])
    else:
        scale = boundary_p_t0_mpa(boundary_cfg) - boundary_p_out_mpa(boundary_cfg)
    if scale <= 0.0:
        raise ValueError(f"旧版共享压力尺度必须为正数，当前为 {scale:g} MPa。")
    return scale


def pressure_normalization_mode(boundary_cfg: dict[str, Any]) -> str:
    """返回压力无量纲化模式，并兼容旧 checkpoint 中缺少配置项的情况。

    旧项目没有 `pressure_normalization` 字段，因此缺省值必须解释为
    `legacy_shared`。当前 `default.yaml` 会显式写入 `component_affine`，
    checkpoint 加载时可据此发现新旧尺度不兼容。
    """

    mode = str(boundary_cfg.get("pressure_normalization", "legacy_shared")).lower()
    aliases = {
        "component": "component_affine",
        "componentwise": "component_affine",
        "component_wise": "component_affine",
        "affine": "component_affine",
        "legacy": "legacy_shared",
        "shared": "legacy_shared",
        "old": "legacy_shared",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"component_affine", "legacy_shared"}:
        raise ValueError(
            "不支持的 pressure_normalization: "
            f"{mode}。请使用 'component_affine' 或 'legacy_shared'。"
        )
    return mode


def pressure_component_affine_parameters(
    boundary_cfg: dict[str, Any],
    eps: float = 1.0e-12,
) -> dict[str, float]:
    """计算 P12/P13 独立仿射无量纲化参数。

    定义：
        u12 = (P12 - P12_offset) / P12_scale
        u13 = (P13 - P13_offset) / P13_scale

    其中 offset 对应长期出口总压 `P_out` 按组分比例拆分后的压力，scale 对应
    初始总压到出口总压的压差按同一比例拆分。这样在理论一致的初始条件下，
    两个组分的初始无量纲值都为 1，长期出口状态都趋近 0。
    """

    ratio = boundary_ratio(boundary_cfg)
    p_t0 = boundary_p_t0_mpa(boundary_cfg)
    p_out = boundary_p_out_mpa(boundary_cfg)
    eps_value = float(eps)
    if ratio <= 0.0:
        raise ValueError(f"C13/C12 组分比必须大于 0，当前为 {ratio:g}。")
    if p_t0 <= p_out:
        raise ValueError(f"初始总压力 P_t0 必须大于出口压力 P_out，当前为 {p_t0:g} <= {p_out:g}。")

    denominator = 1.0 + ratio
    p12_offset = p_out / denominator
    p13_offset = p_out * ratio / denominator
    p12_scale = (p_t0 - p_out) / denominator
    p13_scale = (p_t0 - p_out) * ratio / denominator
    if p12_scale <= eps_value or p13_scale <= eps_value:
        raise ValueError(
            "P12/P13 分量压力尺度过小，无法稳定无量纲化："
            f"P12_scale={p12_scale:g}, P13_scale={p13_scale:g}, eps={eps_value:g}。"
        )
    return {
        "p12_offset_mpa": p12_offset,
        "p12_scale_mpa": p12_scale,
        "p13_offset_mpa": p13_offset,
        "p13_scale_mpa": p13_scale,
    }


def expected_initial_pressure_pair_mpa(boundary_cfg: dict[str, Any]) -> tuple[float, float]:
    """根据总初始压力和组分比计算理论初始 P12/P13 压力。"""

    ratio = boundary_ratio(boundary_cfg)
    if ratio <= 0.0:
        raise ValueError(f"C13/C12 组分比必须大于 0，当前为 {ratio:g}。")
    p_t0 = boundary_p_t0_mpa(boundary_cfg)
    denominator = 1.0 + ratio
    return p_t0 / denominator, p_t0 * ratio / denominator


def warn_if_initial_pressure_inconsistent(
    config: dict[str, Any],
    relative_tolerance: float = 1.0e-4,
) -> None:
    """检查配置中的初始组分压力是否与 `P_t0` 和组分比一致。

    该函数只发出警告，不会覆盖用户配置。研究阶段可能故意给不同区域设置不同初值，
    因此这里选择显式提醒，而不是把配置“自动修正”成理论值。
    """

    boundary_cfg = config["boundary"]
    expected12, expected13 = expected_initial_pressure_pair_mpa(boundary_cfg)
    candidates: list[tuple[str, float, float]] = []
    if "P1_t0" in boundary_cfg and "P2_t0" in boundary_cfg:
        candidates.append(("boundary.P1_t0/P2_t0", float(boundary_cfg["P1_t0"]), float(boundary_cfg["P2_t0"])))
    ic = config.get("initial_condition", {})
    if "P1_t0" in ic and "P2_t0" in ic:
        candidates.append(("initial_condition.P1_t0/P2_t0", float(ic["P1_t0"]), float(ic["P2_t0"])))
    for region in ("hf", "srv", "usrv"):
        key12 = f"p12_{region}_mpa"
        key13 = f"p13_{region}_mpa"
        if key12 in ic and key13 in ic:
            candidates.append((f"initial_condition.{region}", float(ic[key12]), float(ic[key13])))

    for label, value12, value13 in candidates:
        rel12 = abs(value12 - expected12) / max(abs(expected12), 1.0e-12)
        rel13 = abs(value13 - expected13) / max(abs(expected13), 1.0e-12)
        if rel12 > relative_tolerance or rel13 > relative_tolerance:
            warnings.warn(
                f"{label} 与 P_t0/C13_C12 推导的理论初值不一致："
                f"配置 P12={value12:g}, P13={value13:g}; "
                f"理论 P12={expected12:g}, P13={expected13:g}; "
                f"相对误差 P12={rel12:g}, P13={rel13:g}。程序不会自动覆盖用户配置。",
                RuntimeWarning,
                stacklevel=2,
            )


def initial_pressure_pair_mpa(config: dict[str, Any], region_name: str | None = None) -> tuple[float, float]:
    """Return initial P1/P2 pressures in MPa.

    The current COMSOL model uses the same initial pressure in every region.  The
    old project allowed region-specific keys, so they remain as fallbacks.
    """

    ic = config["initial_condition"]
    if "P1_t0" in ic and "P2_t0" in ic:
        return float(ic["P1_t0"]), float(ic["P2_t0"])
    if "P1_t0" in config["boundary"] and "P2_t0" in config["boundary"]:
        return float(config["boundary"]["P1_t0"]), float(config["boundary"]["P2_t0"])
    if region_name is None:
        region_name = "hf"
    key = region_name.lower()
    return float(ic[f"p12_{key}_mpa"]), float(ic[f"p13_{key}_mpa"])


def force_cpu(cpu_threads: int) -> torch.device:
    """强制 PyTorch 只使用 CPU。

    需求中明确第一版不允许自动选择 GPU。这里同时设置 CUDA 可见设备为空，
    并返回固定的 `torch.device("cpu")`，后续所有张量和模型都显式放到这个设备。
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(cpu_threads))
    return torch.device("cpu")


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """把配置中的 dtype 字符串转换为 PyTorch dtype。"""

    name = dtype_name.lower()
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"不支持的 dtype: {dtype_name}")


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子，提升实验可复现性。"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """创建所有输出目录。

    训练、评价和绘图脚本都调用该函数，保证 checkpoint、图像、日志和表格有统一位置。
    """

    for key in ["outputs", "checkpoints", "figures", "logs", "tables"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def tensor_from_numpy(
    array: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """把 NumPy 数组转换为 CPU 张量。

    项目中不出现 `.cuda()`，因此这里统一通过 `to(device=device, dtype=dtype)` 控制位置和精度。
    """

    return torch.as_tensor(array, dtype=dtype, device=device)


def pressure_hat_to_mpa(u: torch.Tensor, boundary_cfg: dict[str, Any]) -> torch.Tensor:
    """将无量纲网络输出还原为 MPa。

    所有评价、绘图和 COMSOL 误差计算都必须调用该函数，避免训练和后处理使用
    不同压力尺度。`legacy_shared` 分支只用于兼容旧实验；新配置默认走
    `component_affine`，使 P12/P13 都有 O(1) 无量纲幅值。
    """

    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"pressure_hat_to_mpa 期望输入形状为 [N, 2]，当前为 {tuple(u.shape)}。")

    mode = pressure_normalization_mode(boundary_cfg)
    if mode == "legacy_shared":
        p_scale = pressure_scale_mpa(boundary_cfg)
        p_out = boundary_p_out_mpa(boundary_cfg)
        p12 = u[:, 0:1] * p_scale + p_out
        p13 = u[:, 1:2] * p_scale
        return torch.cat([p12, p13], dim=1)

    params = pressure_component_affine_parameters(boundary_cfg)
    p12 = u[:, 0:1] * params["p12_scale_mpa"] + params["p12_offset_mpa"]
    p13 = u[:, 1:2] * params["p13_scale_mpa"] + params["p13_offset_mpa"]
    return torch.cat([p12, p13], dim=1)


def pressure_mpa_to_hat(
    p12_mpa: float | np.ndarray | torch.Tensor,
    p13_mpa: float | np.ndarray | torch.Tensor,
    boundary_cfg: dict[str, Any],
) -> tuple[float | np.ndarray | torch.Tensor, float | np.ndarray | torch.Tensor]:
    """把物理压力 MPa 转换为模型无量纲变量。

    该函数支持 float、NumPy 数组和 torch.Tensor，方便训练数据、后处理积分和测试
    共用同一套公式。返回类型会跟随输入运算自然保持为对应的数值类型。
    """

    mode = pressure_normalization_mode(boundary_cfg)
    if mode == "legacy_shared":
        p_scale = pressure_scale_mpa(boundary_cfg)
        p_out = boundary_p_out_mpa(boundary_cfg)
        return (p12_mpa - p_out) / p_scale, p13_mpa / p_scale

    params = pressure_component_affine_parameters(boundary_cfg)
    u12 = (p12_mpa - params["p12_offset_mpa"]) / params["p12_scale_mpa"]
    u13 = (p13_mpa - params["p13_offset_mpa"]) / params["p13_scale_mpa"]
    return u12, u13


def save_loss_history(rows: Iterable[dict[str, float]], path: str | Path) -> None:
    """保存训练损失历史 CSV。

    诊断指标会随着实验逐步增加，因此这里不再维护固定列名。函数会自动收集所有
    history 字典中的键，强制 `epoch` 放在第一列，其余列按名称稳定排序；某些 epoch
    没有记录的低频诊断列会在 CSV 中留空。
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row_list = list(rows)
    all_keys: set[str] = set()
    for row in row_list:
        all_keys.update(row.keys())
    remaining = sorted(key for key in all_keys if key != "epoch")
    fieldnames = ["epoch", *remaining] if "epoch" in all_keys else remaining
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)


def relative_l2(pred: np.ndarray, ref: np.ndarray, eps: float = 1.0e-12) -> float:
    """计算相对 L2 误差，避免参考解范数过小时除零。"""

    numerator = np.linalg.norm(pred - ref)
    denominator = np.linalg.norm(ref) + eps
    return float(numerator / denominator)
