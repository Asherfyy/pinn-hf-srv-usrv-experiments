"""IDE 直接运行：绘制训练 loss 曲线。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def relaunch_with_venv_if_needed() -> None:
    """Cursor 未选中虚拟环境时自动重启到项目 `.venv`。"""

    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制 v4 hard-Dirichlet partitioned-MLP loss 曲线。")
    parser.add_argument("--history", type=Path, default=PROJECT_ROOT / "outputs" / "logs" / "loss_history.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "figures" / "loss_history.png")
    parser.add_argument("--linear", action="store_true")
    return parser.parse_args()


def main() -> None:
    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)
    from src.diagnostics import plot_loss_history

    args = parse_args()
    plot_loss_history(args.history, args.output, use_log_scale=not args.linear)


if __name__ == "__main__":
    main()
