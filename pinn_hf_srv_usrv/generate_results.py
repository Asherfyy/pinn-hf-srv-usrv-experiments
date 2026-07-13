"""IDE 直接运行：加载训练好的模型并生成结果图。

这个脚本只做后处理，不会重新训练模型：
1. 读取 `config/default.yaml`；
2. 加载 `outputs/checkpoints/final.pt`；
3. 生成 P12/P13 二维压力云图；
4. 保存区域 mask 和误差表。

在 Cursor/IDE 中打开本文件后，点击右上角三角形运行即可。
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
    """如果 IDE 没有选中项目虚拟环境，则自动切换到 `.venv` 后重启本脚本。

    Cursor 的运行按钮有时会使用系统 Python。这里主动检测解释器路径，能减少
    “点运行但缺包/没反应”的概率。
    """

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    """解析可选参数。IDE 直接运行时使用默认值。"""

    parser = argparse.ArgumentParser(description="加载已训练 PINN 模型并生成图形结果。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="配置文件路径。")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="训练好的模型 checkpoint。")
    parser.add_argument("--skip-fields", action="store_true", help="跳过二维云图。")
    parser.add_argument("--skip-evaluate", action="store_true", help="跳过误差表和区域 mask 输出。")
    return parser.parse_args()


def run() -> None:
    """加载模型并生成所有图形。"""

    relaunch_with_venv_if_needed()

    # 所有配置中的相对路径，例如 outputs/ 和 data/，都以项目根目录为基准。
    os.chdir(PROJECT_ROOT)

    from src.evaluate import (
        compute_comsol_error_table,
        load_trained_model,
        predict_grid_fields,
        save_field_figure,
        save_region_mask_table,
    )

    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"未找到训练好的模型: {args.checkpoint}\n"
            "请先运行 main.py 的 train/quick 模式，或执行 python -m src.train --config config/default.yaml。"
        )

    print(f"项目目录: {PROJECT_ROOT}", flush=True)
    print(f"配置文件: {args.config}", flush=True)
    print(f"模型文件: {args.checkpoint}", flush=True)

    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)

    if not args.skip_fields:
        print("\n开始生成二维压力云图...", flush=True)
        first_field = None
        for time_value in config["evaluation"]["times"]:
            field = predict_grid_fields(model, geometry, config, float(time_value), device, dtype)
            if first_field is None:
                first_field = field
            for variable in ["P12", "P13"]:
                out_path = Path(config["paths"]["figures"]) / f"field_{variable}_t{float(time_value):g}.png"
                save_field_figure(field, geometry, variable, float(time_value), out_path)
                print(f"已保存 {out_path}", flush=True)
        if first_field is not None:
            mask_path = Path(config["paths"]["tables"]) / "region_mask.csv"
            save_region_mask_table(first_field, mask_path)
            print(f"已保存 {mask_path}", flush=True)

    if not args.skip_evaluate:
        print("\n开始生成误差表...", flush=True)
        error_df = compute_comsol_error_table(model, geometry, config, device, dtype)
        error_path = Path(config["paths"]["tables"]) / "error_metrics.csv"
        error_df.to_csv(error_path, index=False)
        print(f"已保存 {error_path}", flush=True)

    print("\n结果生成完成。图形位于 outputs/figures，表格位于 outputs/tables。", flush=True)


if __name__ == "__main__":
    run()
