"""IDE-friendly entrypoint for the POD-MLP workflow."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

# IDE defaults. Set grid/time/training overrides to None for the full default run.
IDE_MODE = "all"
IDE_GRID_NX = None
IDE_GRID_NY = None
IDE_EARLY_POINTS = None
IDE_MIDDLE_POINTS = None
IDE_LATE_POINTS = None
IDE_FINAL_TIME = None
IDE_EPOCHS = None
IDE_BATCH_SIZE = None
IDE_FIXED_RANK = None
IDE_PREDICT_TIMES = None
IDE_OUTPUT_NAME = "pod_predictions.npz"
IDE_ALLOW_EXTRAPOLATION = False


def choose_python() -> Path:
    return VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)


def run_command(args: list[str], python_exe: Path) -> None:
    command = [str(python_exe), *args]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="POD-MLP workflow entrypoint.")
    parser.add_argument(
        "mode",
        nargs="?",
        default=IDE_MODE,
        choices=["generate", "train", "evaluate", "predict", "gas", "plot", "diagnostics", "all"],
    )
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--grid-nx", type=int, default=IDE_GRID_NX)
    parser.add_argument("--grid-ny", type=int, default=IDE_GRID_NY)
    parser.add_argument("--early-points", type=int, default=IDE_EARLY_POINTS)
    parser.add_argument("--middle-points", type=int, default=IDE_MIDDLE_POINTS)
    parser.add_argument("--late-points", type=int, default=IDE_LATE_POINTS)
    parser.add_argument("--final-time", type=float, default=IDE_FINAL_TIME)
    parser.add_argument("--epochs", type=int, default=IDE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=IDE_BATCH_SIZE)
    parser.add_argument("--fixed-rank", type=int, default=IDE_FIXED_RANK)
    parser.add_argument("--times", type=float, nargs="+", default=IDE_PREDICT_TIMES)
    parser.add_argument("--output-name", type=str, default=IDE_OUTPUT_NAME)
    parser.add_argument("--allow-extrapolation", action="store_true", default=IDE_ALLOW_EXTRAPOLATION)
    return parser.parse_args()


def generate_args(args: argparse.Namespace) -> list[str]:
    command = ["-m", "src.pod_generate_data", "--config", str(args.config)]
    _extend_optional(command, "--grid-nx", args.grid_nx)
    _extend_optional(command, "--grid-ny", args.grid_ny)
    _extend_optional(command, "--early-points", args.early_points)
    _extend_optional(command, "--middle-points", args.middle_points)
    _extend_optional(command, "--late-points", args.late_points)
    _extend_optional(command, "--final-time", args.final_time)
    return command


def train_args(args: argparse.Namespace) -> list[str]:
    command = ["-m", "src.pod_train", "--config", str(args.config)]
    _extend_optional(command, "--epochs", args.epochs)
    _extend_optional(command, "--batch-size", args.batch_size)
    _extend_optional(command, "--fixed-rank", args.fixed_rank)
    return command


def evaluate_args(args: argparse.Namespace) -> list[str]:
    return ["-m", "src.pod_evaluate", "--config", str(args.config)]


def predict_args(args: argparse.Namespace) -> list[str]:
    command = ["-m", "src.pod_predict", "--config", str(args.config)]
    if args.times:
        command.append("--times")
        command.extend(str(float(value)) for value in args.times)
    command.extend(["--output-name", str(args.output_name)])
    if bool(args.allow_extrapolation):
        command.append("--allow-extrapolation")
    return command


def plot_args(args: argparse.Namespace) -> list[list[str]]:
    snapshot_file = str(Path("outputs") / "pod" / str(args.output_name))
    return [
        ["-m", "src.plot_fields", "--config", str(args.config), "--snapshot-file", snapshot_file],
        ["-m", "src.plot_sections", "--config", str(args.config), "--snapshot-file", snapshot_file],
    ]


def gas_args(args: argparse.Namespace) -> list[str]:
    snapshot_file = str(Path("outputs") / "pod" / str(args.output_name))
    return ["-m", "src.pod_gas_postprocess", "--config", str(args.config), "--prediction-file", snapshot_file]


def diagnostics_args(args: argparse.Namespace) -> list[str]:
    snapshot_file = str(Path("outputs") / "pod" / str(args.output_name))
    return ["-m", "src.evaluate", "--config", str(args.config), "--snapshot-file", snapshot_file]


def main_POD() -> None:
    args = parse_args()
    python_exe = choose_python()
    if args.mode == "generate":
        run_command(generate_args(args), python_exe)
    elif args.mode == "train":
        run_command(train_args(args), python_exe)
    elif args.mode == "evaluate":
        run_command(evaluate_args(args), python_exe)
    elif args.mode == "predict":
        run_command(predict_args(args), python_exe)
    elif args.mode == "gas":
        run_command(gas_args(args), python_exe)
    elif args.mode == "plot":
        for command in plot_args(args):
            run_command(command, python_exe)
    elif args.mode == "diagnostics":
        run_command(diagnostics_args(args), python_exe)
    elif args.mode == "all":
        run_command(generate_args(args), python_exe)
        run_command(train_args(args), python_exe)
        run_command(evaluate_args(args), python_exe)
        run_command(predict_args(args), python_exe)
        run_command(gas_args(args), python_exe)
        for command in plot_args(args):
            run_command(command, python_exe)
        run_command(diagnostics_args(args), python_exe)


def _extend_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


if __name__ == "__main__":
    main_POD()
