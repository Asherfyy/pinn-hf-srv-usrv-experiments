from __future__ import annotations

from pathlib import Path

import torch

from src.config import load_config
from src.edfm_grid import build_edfm_grid
from src.fvm_reference import build_or_load_fvm_reference, build_trusted_fvm_reference
from src.fvm_solve import assert_fvm_solution_trustworthy, diagnose_fvm_solution, solve_direct
from src.geometry import ReservoirGeometry
from src.utils import force_cpu, get_torch_dtype


def test_fvm_reference_builds_and_interpolates_at_meshfree_points(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["paths"]["outputs"] = str(tmp_path / "outputs")
    config["grid"].update({"nx": 20, "ny": 12})
    config["time_grid"]["times_days"] = [0.0, 1.0, 10.0]
    config["fvm_reference"].update({"enabled": True, "rebuild": True, "snapshot_name": "tiny_fvm_reference.npz"})

    geometry = ReservoirGeometry(config["geometry"])
    reference = build_or_load_fvm_reference(config, geometry)
    assert reference is not None
    assert reference.pressure_mpa.shape[0] == 3
    assert reference.pressure_mpa.shape[2] == 1

    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    xyt = torch.tensor(
        [
            [200.0, 75.0, 1.0],
            [250.0, 75.0, 5.0],
            [300.0, 90.0, 10.0],
        ],
        dtype=dtype,
        device=device,
    )
    target = reference.pressure_hat_at(xyt, config)
    assert target.shape == (xyt.shape[0], 1)
    assert torch.all(torch.isfinite(target)).item()


def test_direct_fvm_solution_passes_residual_diagnostics(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["paths"]["outputs"] = str(tmp_path / "outputs")
    config["paths"]["tables"] = str(tmp_path / "tables")
    config["grid"].update({"nx": 18, "ny": 10})
    config["time_grid"]["times_days"] = [0.0, 1.0, 10.0]
    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    snapshots, _well_history = solve_direct(grid, config, [0.0, 1.0, 10.0])

    diagnostics = diagnose_fvm_solution(grid, config, [0.0, 1.0, 10.0], snapshots)
    assert_fvm_solution_trustworthy(diagnostics, config)

    values = {row["metric"]: row.get("value") for row in diagnostics if "value" in row}
    assert float(values["fvm_residual_rel_l2_max"]) < 1.0e-7
    assert float(values["fvm_pressure_bounds_ok"]) == 1.0
    assert float(values["fvm_bad_connection_count"]) == 0.0


def test_trusted_fvm_reference_saves_diagnostics(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["paths"]["outputs"] = str(tmp_path / "outputs")
    config["paths"]["tables"] = str(tmp_path / "tables")
    config["grid"].update({"nx": 18, "ny": 10})
    config["time_grid"]["times_days"] = [0.0, 1.0, 10.0]
    geometry = ReservoirGeometry(config["geometry"])

    reference, diagnostics = build_trusted_fvm_reference(config, geometry, snapshot_name="trusted_tiny.npz", rebuild=True)

    assert reference.pressure_mpa.shape[0] == 3
    assert diagnostics
    assert (tmp_path / "tables" / "fvm_diagnostics.csv").exists()
