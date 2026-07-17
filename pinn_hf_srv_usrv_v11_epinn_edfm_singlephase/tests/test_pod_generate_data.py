from __future__ import annotations

import copy

import numpy as np

from src.config import load_config
from src.pod_generate_data import _generate_pinn_snapshots


def test_generate_pinn_snapshots_copies_training_archive(tmp_path) -> None:
    config = copy.deepcopy(load_config("config/default.yaml"))
    outputs = tmp_path / "outputs"
    pod_outputs = outputs / "pod"
    outputs.mkdir(parents=True)
    pod_outputs.mkdir(parents=True)

    config["paths"]["outputs"] = str(outputs)
    config["pod"]["snapshot_generation"]["source"] = "pinn"
    config["pod"]["snapshot_generation"]["pinn_snapshot_file"] = "snapshots.npz"
    config["pod"]["snapshot_generation"]["snapshot_file"] = "pinn_snapshots.npz"

    source_path = outputs / "snapshots.npz"
    np.savez_compressed(
        source_path,
        times_days=np.asarray([0.0, 1.0, 3.0], dtype=np.float64),
        pressure_mpa=np.ones((3, 4, 2), dtype=np.float32),
        cell_xy=np.zeros((4, 2), dtype=np.float64),
        matrix_cell_count=np.asarray(4, dtype=np.int64),
        well_cells=np.asarray([3], dtype=np.int64),
        component_names=np.asarray(["P12", "P13"]),
    )

    _generate_pinn_snapshots(config, {"outputs": pod_outputs})

    output_path = pod_outputs / "pinn_snapshots.npz"
    with np.load(output_path, allow_pickle=True) as data:
        assert data["pressure_mpa"].shape == (3, 4, 2)
        assert data["times_days"].tolist() == [0.0, 1.0, 3.0]
        assert str(np.asarray(data["solver"]).item()) == "pinn_epinn_train_pod_dataset"
