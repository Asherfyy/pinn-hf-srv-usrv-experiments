from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_tiny_train_and_evaluate_smoke(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config(PROJECT_ROOT / "config" / "default.yaml"))
    config["mesh"]["nx"] = 6
    config["mesh"]["ny"] = 4
    config["time_grid"]["times_days"] = [0.0, 1.0]
    config["evaluation"]["times"] = [1.0]
    config["model"]["hidden_layers"] = 1
    config["model"]["hidden_units"] = 8
    config["training"]["epochs_per_step"] = 1
    config["paths"] = {
        "outputs": str(tmp_path / "outputs"),
        "checkpoints": str(tmp_path / "outputs" / "checkpoints"),
        "figures": str(tmp_path / "outputs" / "figures"),
        "logs": str(tmp_path / "outputs" / "logs"),
        "tables": str(tmp_path / "outputs" / "tables"),
        "data": str(tmp_path / "data"),
    }
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    subprocess.run(
        [sys.executable, "-m", "src.train", "--config", str(config_path), "--epochs-per-step", "1"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run([sys.executable, "-m", "src.evaluate", "--config", str(config_path)], cwd=PROJECT_ROOT, check=True)

    assert (tmp_path / "outputs" / "snapshots.npz").exists()
    diagnostics = pd.read_csv(tmp_path / "outputs" / "tables" / "diagnostics.csv")
    row = diagnostics.loc[diagnostics["metric"] == "nonfinite_pressure_points", "value"]
    assert not row.empty
    assert float(row.iloc[0]) == 0.0
