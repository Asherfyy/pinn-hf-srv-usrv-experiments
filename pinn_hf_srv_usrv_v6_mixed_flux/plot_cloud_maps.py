"""IDE 直接运行：绘制 v6 mixed-flux partitioned-MLP 压力云图。

这个脚本是独立的云图绘制入口，只做后处理：
1. 读取 `config/default.yaml`；
2. 加载 `outputs/checkpoints/final.pt`；
3. 绘制 P12、P13 和 Ptotal 在配置时间点上的二维云图。

它适合在 Cursor/IDE 中打开本文件后直接点击右上角运行按钮。脚本不会启动训练，
也不会修改模型 checkpoint。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "checkpoints" / "final.pt"


def relaunch_with_venv_if_needed() -> None:
    """如果 IDE 未选择本项目虚拟环境，则优先切换到本项目 `.venv`。

    v6 项目可以独立创建自己的 `.venv`。如果 `.venv` 不存在，就使用 IDE 当前解释器，
    这样不会强行依赖其他项目环境。
    """

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    """解析可选参数；IDE 直接运行时使用默认路径。"""

    parser = argparse.ArgumentParser(description="绘制 v6 mixed-flux partitioned-MLP HF/SRV/USRV PINN 压力云图。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径。")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="模型 checkpoint 路径。")
    return parser.parse_args()


def main() -> None:
    """加载模型并生成 P12/P13/Ptotal 云图。"""

    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)

    from src.evaluate import load_trained_model, predict_field
    from src.plot_fields import save_field_figure

    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"未找到模型 checkpoint: {args.checkpoint}\n"
            "请先运行 python -m src.train --config config/default.yaml，或在 main.py 中选择 train。"
        )

    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    for time_value in config["evaluation"]["times"]:
        field = predict_field(model, geometry, config, float(time_value), device, dtype)
        for variable in ["P12", "P13", "Ptotal"]:
            out_path = Path(config["paths"]["figures"]) / f"field_{variable}_t{float(time_value):g}.png"
            save_field_figure(field, geometry, config, variable, float(time_value), out_path)
            print(f"已保存 {out_path}", flush=True)

    print("\n云图绘制完成。输出目录: outputs/figures", flush=True)


if __name__ == "__main__":
    main()
