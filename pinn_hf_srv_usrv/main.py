"""IDE 一键运行入口。

这个文件只依赖 Python 标准库，因此即使 IDE 当前没有选中项目虚拟环境，也可以先
找到 `.venv/Scripts/python.exe` 再调用真正的项目模块。这样用户在 IDE 中直接点击
运行 `main.py`，就能完成一次可验证的项目流程。

默认运行模式是 `quick`：
1. 运行单元测试；
2. 训练 1 个 epoch，验证训练链路和 checkpoint 写入；
3. 运行评价；
4. 绘制二维云图。

如果需要完整 5000 epoch 训练，可以把下面的 `DEFAULT_MODE` 改成 `"train"`，
或在命令行中运行：

    python main.py train
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# 在 IDE 中直接运行时使用的默认模式。为了避免第一次误触就开始漫长 CPU 训练，
# 默认选择 quick；它会跑完整入口链路，但训练只跑 1 个 epoch。
DEFAULT_MODE = "train"


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "final.pt"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def choose_python() -> Path:
    """选择用于运行项目模块的 Python 解释器。

    优先使用项目虚拟环境，因为依赖都安装在 `.venv` 中；如果虚拟环境不存在，
    则退回到当前解释器，并给出清晰提示。
    """

    if VENV_PYTHON.exists():
        return VENV_PYTHON
    print("未找到 .venv，将使用当前 Python 解释器。若缺少依赖，请先安装 requirements.txt。", flush=True)
    return Path(sys.executable)


def run_command(args: list[str], python_exe: Path) -> None:
    """运行一个子命令，失败时立即停止。

    用子进程运行模块的好处是复用现有 `python -m src.xxx` 入口，不需要在这里重复
    训练、评价和绘图逻辑。
    """

    command = [str(python_exe), *args]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_tests(python_exe: Path) -> None:
    """运行测试。"""

    run_command(["-m", "pytest", "tests", "-q"], python_exe)


def run_train(python_exe: Path, epochs: int | None = None) -> None:
    """运行训练。epochs 为 None 时使用配置文件中的完整训练轮数。"""

    args = ["-m", "src.train", "--config", str(CONFIG_PATH)]
    if epochs is not None:
        args.extend(["--epochs", str(epochs)])
    run_command(args, python_exe)


def run_evaluate(python_exe: Path) -> None:
    """运行评价并生成误差表。"""

    run_command(
        [
            "-m",
            "src.evaluate",
            "--config",
            str(CONFIG_PATH),
            "--checkpoint",
            str(CHECKPOINT_PATH),
        ],
        python_exe,
    )


def run_plot_fields(python_exe: Path) -> None:
    """绘制二维压力云图。"""

    run_command(
        [
            "-m",
            "src.plot_fields",
            "--config",
            str(CONFIG_PATH),
            "--checkpoint",
            str(CHECKPOINT_PATH),
        ],
        python_exe,
    )


def parse_args() -> argparse.Namespace:
    """解析可选命令行参数。

    IDE 直接运行时通常没有参数，此时使用 DEFAULT_MODE。
    """

    parser = argparse.ArgumentParser(description="HF/SRV/USRV PINN 项目一键运行入口。")
    parser.add_argument(
        "mode",
        nargs="?",
        default=DEFAULT_MODE,
        choices=["quick", "test", "train", "evaluate", "plot_fields", "all"],
        help="运行模式：quick 默认快速全流程；train 使用配置中的完整 epoch。",
    )
    parser.add_argument(
        "--quick-epochs",
        type=int,
        default=1,
        help="quick/all 模式下快速训练的 epoch 数，默认 1。",
    )
    return parser.parse_args()


def main() -> None:
    """IDE 一键运行主流程。"""

    args = parse_args()
    python_exe = choose_python()
    print(f"项目目录: {PROJECT_ROOT}", flush=True)
    print(f"Python: {python_exe}", flush=True)
    print(f"运行模式: {args.mode}", flush=True)

    if args.mode == "test":
        run_tests(python_exe)
    elif args.mode == "train":
        run_train(python_exe, epochs=None)
    elif args.mode == "evaluate":
        run_evaluate(python_exe)
    elif args.mode == "plot_fields":
        run_plot_fields(python_exe)
    elif args.mode == "quick":
        run_tests(python_exe)
        run_train(python_exe, epochs=args.quick_epochs)
        run_evaluate(python_exe)
        run_plot_fields(python_exe)
    elif args.mode == "all":
        run_tests(python_exe)
        run_train(python_exe, epochs=args.quick_epochs)
        run_evaluate(python_exe)
        run_plot_fields(python_exe)

    print("\n运行完成。输出文件位于 outputs/ 目录。", flush=True)


if __name__ == "__main__":
    main()
