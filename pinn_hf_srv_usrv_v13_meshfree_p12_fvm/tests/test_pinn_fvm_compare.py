from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import load_config
from src.fvm_reference import build_trusted_fvm_reference
from src.geometry import ReservoirGeometry
from src.model import PINNModel
from src.pinn_fvm_compare import build_comparison_field, error_summary, save_comparison_figure
from src.utils import force_cpu, get_torch_dtype, set_seed


def test_pinn_fvm_comparison_field_and_figure_are_finite(tmp_path: Path) -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["paths"]["outputs"] = str(tmp_path / "outputs")
    config["paths"]["tables"] = str(tmp_path / "tables")
    config["grid"].update({"nx": 18, "ny": 10})
    config["time_grid"]["times_days"] = [0.0, 1.0, 10.0]
    config["evaluation"]["adaptive_plot"].update(
        {
            "coarse_nx": 19,
            "coarse_ny": 11,
            "srv_nx": 21,
            "srv_ny": 13,
            "hf_long_axis_points": 35,
            "hf_short_axis_points": 1,
            "transition_band_points": 14,
        }
    )

    set_seed(int(config["runtime"]["seed"]))
    geometry = ReservoirGeometry(config["geometry"])
    reference, diagnostics = build_trusted_fvm_reference(config, geometry, snapshot_name="trusted_compare_tiny.npz", rebuild=True)
    assert diagnostics

    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(device=device, dtype=dtype)
    model.eval()

    field = build_comparison_field(config, geometry, model, reference, 10.0, device, dtype)
    summary = error_summary(field)

    assert field["P12_PINN"].shape == field["P12_FVM"].shape
    assert np.all(np.isfinite(field["P12_PINN"]))
    assert np.all(np.isfinite(field["P12_FVM"]))
    assert np.isfinite(summary["rmse_mpa"])

    figure_path = tmp_path / "figures" / "pinn_fvm_error_t10.png"
    save_comparison_figure(field, geometry, 10.0, figure_path)
    assert figure_path.exists()
    assert figure_path.stat().st_size > 0
