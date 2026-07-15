"""Unified entrypoint for the v10 RDFM/I-PINN project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
# 可选模式："test" 只跑测试；"train" 训练 I-PINN；"solve" 求解 RDFM/FEM 线性系统；"evaluate" 生成诊断表；"plot" 绘图；"mesh" 绘制网格；"all" 依次测试、训练、评估、绘图。
IDE_MODE = "all"
IDE_EPOCHS_PER_STEP: int | None = None


def choose_python() -> Path:
    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def run_command(args: list[str], python_exe: Path) -> None:
    command = [str(python_exe), *args]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v10 RDFM/I-PINN HF/SRV/USRV entrypoint.")
    parser.add_argument("mode", nargs="?", default=IDE_MODE, choices=["test", "train", "solve", "evaluate", "plot", "mesh", "all"])
    parser.add_argument("--epochs-per-step", type=int, default=IDE_EPOCHS_PER_STEP, help="Override training.epochs_per_step.")
    return parser.parse_args()


def train_args(args: argparse.Namespace) -> list[str]:
    command = ["-m", "src.train", "--config", str(CONFIG_PATH)]
    if args.epochs_per_step is not None:
        command.extend(["--epochs-per-step", str(args.epochs_per_step)])
    return command


def main() -> None:
    args = parse_args()
    python_exe = choose_python()
    if args.mode == "test":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
    elif args.mode == "train":
        run_command(train_args(args), python_exe)
    elif args.mode == "solve":
        run_command(["-m", "src.fem_solver", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "evaluate":
        run_command(["-m", "src.evaluate", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "plot":
        run_command(["-m", "src.plot_fields", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_sections", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["plot_loss_history.py"], python_exe)
    elif args.mode == "mesh":
        run_command(["-m", "src.plot_mesh", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "all":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
        run_command(train_args(args), python_exe)
        run_command(["-m", "src.evaluate", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_fields", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_sections", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["plot_loss_history.py"], python_exe)


if __name__ == "__main__":
    main()
