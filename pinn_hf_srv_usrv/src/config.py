"""配置文件读取与轻量校验工具。

本项目把几何、物理系数、采样数量和训练超参数都放在 YAML 中，
这样后续从 COMSOL 冻结系数或新几何更新参数时，不需要改动训练代码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .utils import warn_if_initial_pressure_inconsistent


def load_config(config_path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置并返回普通字典。

    参数
    ----
    config_path:
        YAML 配置文件路径。

    返回
    ----
    dict[str, Any]
        解析后的配置。这里刻意保留普通 dict，便于被 torch checkpoint 序列化。
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("配置文件必须解析为字典结构。")

    validate_config(config)
    # 只提示不覆盖：研究配置可能故意给不同区域初值，这里负责把潜在尺度不一致暴露出来。
    warn_if_initial_pressure_inconsistent(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """对必要配置段做最小校验。

    这里不做过度严格的 schema 校验，是为了方便研究阶段快速添加参数；
    但关键段缺失会导致错误难以定位，因此提前给出明确异常。
    """

    required_sections = [
        "runtime",
        "geometry",
        "physics",
        "boundary",
        "initial_condition",
        "sampler",
        "model",
        "training",
        "loss_weights",
        "evaluation",
        "paths",
    ]
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise KeyError(f"配置缺少必要段: {missing}")

    device = str(config["runtime"].get("device", "cpu")).lower()
    if device != "cpu":
        raise ValueError("本项目第一版强制使用 CPU，请将 runtime.device 设置为 'cpu'。")


def update_config_from_args(config: dict[str, Any], epochs: int | None = None) -> dict[str, Any]:
    """根据命令行参数覆盖少量配置。

    训练脚本保留 `--epochs` 覆盖项，主要用于冒烟测试或调试；默认运行时仍遵循
    `config/default.yaml` 中的 5000 epoch。
    """

    if epochs is not None:
        config["training"]["epochs"] = int(epochs)
    return config
