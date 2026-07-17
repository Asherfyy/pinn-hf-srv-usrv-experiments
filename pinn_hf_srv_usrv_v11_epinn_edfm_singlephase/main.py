"""Unified entrypoint for the v11 E-PINN/EDFM single-phase project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
IDE_MODE = "all"
IDE_EPOCHS_PER_STEP: int | None = None
IDE_TIME_STEPS: int | None = None
IDE_GRID_NX: int | None = None
IDE_GRID_NY: int | None = None


def choose_python() -> Path:
    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def run_command(args: list[str], python_exe: Path) -> None:
    command = [str(python_exe), *args]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v11 E-PINN/EDFM single-phase entrypoint.")
    parser.add_argument("mode", nargs="?", default=IDE_MODE, choices=["test", "train", "solve", "evaluate", "plot", "mesh", "all"])
    parser.add_argument("--epochs-per-step", type=int, default=IDE_EPOCHS_PER_STEP, help="Override training.epochs_per_step.")
    parser.add_argument("--time-steps", type=int, default=IDE_TIME_STEPS, help="Use only the first N time steps during training.")
    parser.add_argument("--grid-nx", type=int, default=IDE_GRID_NX, help="Override grid.nx during training.")
    parser.add_argument("--grid-ny", type=int, default=IDE_GRID_NY, help="Override grid.ny during training.")
    return parser.parse_args()


def train_args(args: argparse.Namespace) -> list[str]:
    command = ["-m", "src.train", "--config", str(CONFIG_PATH)]
    if args.epochs_per_step is not None:
        command.extend(["--epochs-per-step", str(args.epochs_per_step)])
    if args.time_steps is not None:
        command.extend(["--time-steps", str(args.time_steps)])
    if args.grid_nx is not None:
        command.extend(["--grid-nx", str(args.grid_nx)])
    if args.grid_ny is not None:
        command.extend(["--grid-ny", str(args.grid_ny)])
    return command


def main() -> None:
    args = parse_args()
    python_exe = choose_python()
    if args.mode == "test":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
    elif args.mode == "train":
        run_command(train_args(args), python_exe)
    elif args.mode == "solve":
        run_command(["-m", "src.fvm_solve", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "evaluate":
        run_command(["-m", "src.evaluate", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "plot":
        run_command(["-m", "src.plot_fields", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_sections", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_loss_history", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "mesh":
        run_command(["-m", "src.plot_mesh", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "all":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
        run_command(train_args(args), python_exe)
        run_command(["-m", "src.evaluate", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_fields", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_sections", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.plot_loss_history", "--config", str(CONFIG_PATH)], python_exe)


if __name__ == "__main__":
    main()
