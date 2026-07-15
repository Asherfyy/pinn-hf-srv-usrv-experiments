"""通用工具函数。

本文件集中处理 CPU 环境、随机种子、输出目录、压力无量纲化和 CSV 日志。压力换算
绝不能散落在 model/train/evaluate 中，否则 P12/P13 很容易在训练和绘图阶段使用不同
尺度，得到看似能运行但物理含义错位的结果。
"""

from __future__ import annotations

import csv
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_VERSION = "partition_mlp_v9_base_correction"


def force_cpu(cpu_threads: int) -> torch.device:
    """强制 PyTorch 使用 CPU，并设置线程数。

    这里同时清空 CUDA 可见设备，是为了防止 IDE 或用户环境中存在 GPU 时 PyTorch
    自动把张量放到 CUDA 上，破坏本项目 CPU-only 的可复现假设。
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(int(cpu_threads))
    return torch.device("cpu")


def get_torch_dtype(dtype_name: str) -> torch.dtype:
    """把配置字符串转换为 torch dtype；v9 默认并强制使用 float32。"""

    if str(dtype_name).lower() != "float32":
        raise ValueError(f"v9 只支持 float32，当前 dtype={dtype_name}。")
    return torch.float32


def set_seed(seed: int) -> None:
    """设置 Python、NumPy 和 PyTorch 随机种子。"""

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def ensure_output_dirs(config: dict[str, Any]) -> None:
    """创建 v3 项目自己的输出目录。"""

    for key in ["outputs", "checkpoints", "figures", "logs", "tables"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def tensor_from_numpy(array: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """把 NumPy 数组转换为项目统一 CPU 张量。"""

    return torch.as_tensor(array, dtype=dtype, device=device)


def pressure_affine_parameters(boundary_cfg: dict[str, Any], eps: float = 1.0e-14) -> dict[str, float]:
    """计算 P12/P13 独立仿射无量纲化参数。

    初始总压和出口总压先按组分比拆成 P12/P13，再分别构造 offset 和 scale。这样在
    t=0 时两个变量都为 1，在长期出口压力下两个变量都趋近 0，避免 P13 因物理量级小
    而在 MSE 中被弱化。
    """

    p_t0 = float(boundary_cfg["P_t0"])
    p_out = float(boundary_cfg["P_out"])
    ratio = float(boundary_cfg["C13_C12"])
    if p_t0 <= p_out:
        raise ValueError(f"P_t0 必须大于 P_out，当前 {p_t0:g} <= {p_out:g}。")
    if ratio <= 0.0:
        raise ValueError(f"C13_C12 必须大于 0，当前 {ratio:g}。")
    denominator = 1.0 + ratio
    p12_0 = p_t0 / denominator
    p13_0 = p_t0 * ratio / denominator
    p12_out = p_out / denominator
    p13_out = p_out * ratio / denominator
    p12_scale = p12_0 - p12_out
    p13_scale = p13_0 - p13_out
    if p12_scale <= eps or p13_scale <= eps:
        raise ValueError(f"P12/P13 scale 必须为正，当前 {p12_scale:g}, {p13_scale:g}。")
    return {
        "p12_offset_mpa": p12_out,
        "p13_offset_mpa": p13_out,
        "p12_scale_mpa": p12_scale,
        "p13_scale_mpa": p13_scale,
        "p12_initial_mpa": p12_0,
        "p13_initial_mpa": p13_0,
    }


def pressure_mpa_to_hat(
    p12_mpa: float | np.ndarray | torch.Tensor,
    p13_mpa: float | np.ndarray | torch.Tensor,
    boundary_cfg: dict[str, Any],
) -> tuple[float | np.ndarray | torch.Tensor, float | np.ndarray | torch.Tensor]:
    """物理压力 MPa -> 无量纲压力 u12/u13。"""

    params = pressure_affine_parameters(boundary_cfg)
    u12 = (p12_mpa - params["p12_offset_mpa"]) / params["p12_scale_mpa"]
    u13 = (p13_mpa - params["p13_offset_mpa"]) / params["p13_scale_mpa"]
    return u12, u13


def pressure_hat_to_mpa(u: torch.Tensor, boundary_cfg: dict[str, Any]) -> torch.Tensor:
    """无量纲压力 -> 物理压力 MPa。"""

    if u.ndim != 2 or u.shape[1] != 2:
        raise ValueError(f"pressure_hat_to_mpa 期望 [N,2]，当前 {tuple(u.shape)}。")
    params = pressure_affine_parameters(boundary_cfg)
    p12 = u[:, 0:1] * params["p12_scale_mpa"] + params["p12_offset_mpa"]
    p13 = u[:, 1:2] * params["p13_scale_mpa"] + params["p13_offset_mpa"]
    return torch.cat([p12, p13], dim=1)


def dirichlet_target_hat(t: torch.Tensor, boundary_cfg: dict[str, Any]) -> torch.Tensor:
    """生产边界无量纲目标。

    在 component affine 无量纲化下，P12 和 P13 的目标都严格等于
    `exp(-decay_rate*t)`。这里仍保留公共函数，是为了所有 soft Dirichlet loss 和测试
    统一调用同一入口。
    """

    if t.ndim == 1:
        t = t.view(-1, 1)
    decay = float(boundary_cfg["decay_rate"])
    target = torch.exp(-decay * t)
    return torch.cat([target, target], dim=1)


def save_loss_history(rows: Iterable[dict[str, float]], path: str | Path) -> None:
    """保存训练日志，自动收集所有诊断列。"""

    row_list = list(rows)
    keys: set[str] = set()
    for row in row_list:
        keys.update(row.keys())
    fieldnames = ["epoch", *sorted(k for k in keys if k != "epoch")]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_list:
            writer.writerow(row)
