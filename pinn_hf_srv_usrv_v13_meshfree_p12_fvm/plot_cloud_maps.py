"""IDE shortcut for plotting v13 P12 cloud maps."""

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
    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v13 mesh-free P12 cloud maps.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    return parser.parse_args()


def main() -> None:
    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)
    from src.evaluate import load_trained_model, predict_field
    from src.plot_fields import save_field_figure

    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    config, geometry, model, device, dtype = load_trained_model(args.config, args.checkpoint)
    for time_value in config["evaluation"]["times"]:
        field = predict_field(model, geometry, config, float(time_value), device, dtype)
        out_path = Path(config["paths"]["figures"]) / f"field_P12_t{float(time_value):g}.png"
        save_field_figure(field, geometry, config, "P12", float(time_value), out_path)
        print(f"saved {out_path}", flush=True)


if __name__ == "__main__":
    main()
