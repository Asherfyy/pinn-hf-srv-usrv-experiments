"""v2 项目统一入口。

IDE 直接运行默认执行训练，不绘图也不保存图片；绘图需要显式使用 `python main.py plot`。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_MODE = "train"
PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "final.pt"


def choose_python() -> Path:
    """优先使用项目虚拟环境，否则退回当前解释器。"""

    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def run_command(args: list[str], python_exe: Path) -> None:
    """在项目根目录执行子命令。"""

    command = [str(python_exe), *args]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 简化 PINN 统一入口。")
    parser.add_argument("mode", nargs="?", default=DEFAULT_MODE, choices=["test", "train", "evaluate", "plot", "all"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python_exe = choose_python()
    if args.mode == "test":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
    elif args.mode == "train":
        run_command(["-m", "src.train", "--config", str(CONFIG_PATH)], python_exe)
    elif args.mode == "evaluate":
        run_command(["-m", "src.evaluate", "--config", str(CONFIG_PATH), "--checkpoint", str(CHECKPOINT_PATH)], python_exe)
    elif args.mode == "plot":
        run_command(["-m", "src.plot_fields", "--config", str(CONFIG_PATH), "--checkpoint", str(CHECKPOINT_PATH)], python_exe)
        run_command(["-m", "src.plot_sections", "--config", str(CONFIG_PATH), "--checkpoint", str(CHECKPOINT_PATH)], python_exe)
    elif args.mode == "all":
        run_command(["-m", "pytest", "tests", "-q"], python_exe)
        run_command(["-m", "src.train", "--config", str(CONFIG_PATH)], python_exe)
        run_command(["-m", "src.evaluate", "--config", str(CONFIG_PATH), "--checkpoint", str(CHECKPOINT_PATH)], python_exe)
        run_command(["-m", "src.plot_fields", "--config", str(CONFIG_PATH), "--checkpoint", str(CHECKPOINT_PATH)], python_exe)
        run_command(["-m", "src.plot_sections", "--config", str(CONFIG_PATH), "--checkpoint", str(CHECKPOINT_PATH)], python_exe)


if __name__ == "__main__":
    main()
