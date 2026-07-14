"""Unified entrypoint for the v8 no-HF-local-coordinate PINN project."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "final.pt"

# =========================
# IDE 直接运行参数区。
# =========================
# 可选模式："test" 只跑测试；"train" 训练；"evaluate" 生成诊断表；"plot" 绘图；"all" 依次测试、训练、评估、绘图。
IDE_MODE = "all"

# 是否从已有 checkpoint 继续训练；False 表示从头随机初始化训练。
IDE_RESUME_FROM_CHECKPOINT = True

# 继续训练使用的 checkpoint；通常用 final.pt，也可以改成 outputs/checkpoints/epoch_3000.pt。
IDE_RESUME_CHECKPOINT = DEFAULT_CHECKPOINT_PATH

# 训练目标总 epoch；None 表示使用 config/default.yaml 里的 training.epochs。
# 注意：继续训练时这里是“目标总轮数”，不是“额外训练轮数”。
IDE_TARGET_EPOCHS: int | None = None

# evaluate/plot 使用的 checkpoint。
IDE_EVAL_OR_PLOT_CHECKPOINT = DEFAULT_CHECKPOINT_PATH


def choose_python() -> Path:
    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def run_command(args: list[str], python_exe: Path) -> None:
    command = [str(python_exe), *args]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v8 no-HF-local-coordinate HF/SRV/USRV PINN entrypoint.")
    parser.add_argument("mode", nargs="?", default=IDE_MODE, choices=["test", "train", "evaluate", "plot", "all"])
    parser.add_argument("--epochs", type=int, default=IDE_TARGET_EPOCHS, help="Override training target epochs.")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=IDE_RESUME_FROM_CHECKPOINT,
        help="Continue training from a checkpoint; use --no-resume to force fresh training.",
    )
    parser.add_argument("--resume-path", type=Path, default=IDE_RESUME_CHECKPOINT, help="Checkpoint used when resume is enabled.")
    parser.add_argument("--checkpoint", type=Path, default=IDE_EVAL_OR_PLOT_CHECKPOINT, help="Checkpoint used by evaluate/plot.")
    return parser.parse_args()


def train_args(args: argparse.Namespace) -> list[str]:
    command = ["-m", "src.train", "--config", str(CONFIG_PATH)]
    if args.epochs is not None:
        command.extend(["--epochs", str(args.epochs)])
    if args.resume:
        resume_path = project_path(args.resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"Cannot resume because checkpoint does not exist: {resume_path}")
        command.extend(["--resume", str(resume_path)])
    return command


def evaluate_args(args: argparse.Namespace) -> list[str]:
    checkpoint = project_path(args.checkpoint)
    return ["-m", "src.evaluate", "--config", str(CONFIG_PATH), "--checkpoint", str(checkpoint)]


def plot_args(args: argparse.Namespace) -> list[list[str]]:
    checkpoint = project_path(args.checkpoint)
    return [
        ["-m", "src.plot_fields", "--config", str(CONFIG_PATH), "--checkpoint", str(checkpoint)],
        ["-m", "src.plot_sections", "--config", str(CONFIG_PATH), "--checkpoint", str(checkpoint)],
    ]


def main() -> None:
    args = parse_args()
    python_exe = choose_python()
    if args.mode == "test":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
    elif args.mode == "train":
        run_command(train_args(args), python_exe)
    elif args.mode == "evaluate":
        run_command(evaluate_args(args), python_exe)
    elif args.mode == "plot":
        for command in plot_args(args):
            run_command(command, python_exe)
    elif args.mode == "all":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
        run_command(train_args(args), python_exe)
        run_command(evaluate_args(args), python_exe)
        for command in plot_args(args):
            run_command(command, python_exe)


if __name__ == "__main__":
    main()
