"""绘制二维压力云图。

运行方式：
    python -m src.plot_fields --config config/default.yaml --checkpoint outputs/checkpoints/final.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import load_trained_model, predict_grid_fields, save_field_figure


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="绘制 P12/P13 二维压力云图。")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/final.pt")
    return parser.parse_args()


def main() -> None:
    """绘图主流程。"""

    args = parse_args()
    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    for time_value in config["evaluation"]["times"]:
        field = predict_grid_fields(model, geometry, config, float(time_value), device, dtype)
        for variable in ["P12", "P13"]:
            out_path = Path(config["paths"]["figures"]) / f"field_{variable}_t{float(time_value):g}.png"
            save_field_figure(field, geometry, variable, float(time_value), out_path)
            print(f"已保存 {out_path}")


if __name__ == "__main__":
    main()
