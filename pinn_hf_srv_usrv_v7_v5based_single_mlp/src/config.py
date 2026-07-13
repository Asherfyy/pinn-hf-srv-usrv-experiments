"""配置读取与轻量校验。

v7 项目刻意把几何、物理系数、采样数量和训练超参数集中到 YAML 中。这样做的原因
不是为了“配置更花哨”，而是为了让简化模型的每个假设都能被看见：网络只吃
`x_hat/y_hat/t_hat`，PDE 只用 `D/Fai` 有效扩散形式，界面连续 loss 被彻底移除。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并执行最小必要校验。"""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为字典。")
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """检查本项目不可放松的关键约束。"""

    required = [
        "runtime",
        "geometry",
        "physics",
        "boundary",
        "sampler",
        "model",
        "training",
        "loss_weights",
        "evaluation",
        "paths",
    ]
    missing = [name for name in required if name not in config]
    if missing:
        raise KeyError(f"配置缺少必要段: {missing}")
    if str(config["runtime"].get("device", "cpu")).lower() != "cpu":
        raise ValueError("v7 single-MLP 项目强制 CPU-only，请将 runtime.device 设置为 cpu。")
    if str(config["runtime"].get("dtype", "float64")).lower() != "float64":
        raise ValueError("v7 默认并强制使用 float64，以减轻极端扩散系数下的数值噪声。")
    model_cfg = config["model"]
    if int(model_cfg.get("input_dim", -1)) != 3:
        raise ValueError("网络输入维度必须为 3，仅允许 x_hat/y_hat/t_hat。")
    if int(model_cfg.get("network_input_dim", model_cfg.get("input_dim", -1))) != 3:
        raise ValueError("v7 single-MLP 只使用 network_input_dim=3，即 [x_hat,y_hat,t_hat]。")
    if str(model_cfg.get("constraint_mode", "")).lower() not in {"ic_hard", "ic_base_correction"}:
        raise ValueError("v7 only supports constraint_mode='ic_hard' or 'ic_base_correction'.")
    physics_mode = str(config["physics"].get("mode", "")).lower()
    if physics_mode != "effective_diffusion":
        raise ValueError("v7 只允许 physics.mode='effective_diffusion'。")
