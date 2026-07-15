from __future__ import annotations

import subprocess
import sys


def test_tiny_train_evaluate_plot_pipeline_runs() -> None:
    subprocess.run([sys.executable, "main.py", "train", "--time-steps", "2", "--epochs-per-step", "2", "--grid-nx", "8", "--grid-ny", "6"], check=True)
    subprocess.run([sys.executable, "main.py", "evaluate"], check=True)
    subprocess.run([sys.executable, "main.py", "mesh"], check=True)
    subprocess.run([sys.executable, "main.py", "plot"], check=True)
