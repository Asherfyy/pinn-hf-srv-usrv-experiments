from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.evaluate import build_adaptive_plot_points, compute_diagnostics, predict_field, triangulation_for_region
from src.geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry
from src.model import PINNModel
from src.utils import force_cpu, get_torch_dtype


def make_context() -> tuple[dict, ReservoirGeometry, PINNModel, torch.device, torch.dtype]:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    # 测试中缩小点云规模，避免单元测试承担绘图级别开销。
    config["evaluation"]["adaptive_plot"].update({"coarse_nx": 31, "coarse_ny": 21, "srv_nx": 41, "srv_ny": 31, "hf_long_axis_points": 80, "hf_short_axis_points": 9, "transition_band_points": 30})
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    model = PINNModel(config).to(device=device, dtype=dtype)
    return config, geometry, model, device, dtype


def test_adaptive_points_include_all_regions() -> None:
    config, geometry, _model, _device, _dtype = make_context()
    points = build_adaptive_plot_points(geometry, config)
    region = geometry.region_id_np(points[:, 0], points[:, 1])
    assert np.any(region == REGION_HF)
    assert np.any(region == REGION_SRV)
    assert np.any(region == REGION_USRV)


def test_triangulation_region_mask_and_pressure_finite() -> None:
    config, geometry, model, device, dtype = make_context()
    field = predict_field(model, geometry, config, 1.0, device, dtype)
    assert np.all(np.isfinite(field["P12"]))
    result = triangulation_for_region(field["x"], field["y"], field["P12"], field["region"], REGION_SRV, config)
    assert result is not None
    tri, _values = result
    if tri.mask is not None:
        assert tri.mask.dtype == bool


def test_diagnostics_table_can_be_generated() -> None:
    config, geometry, model, device, dtype = make_context()
    config["sampler"].update(
        {
            "n_pde_hf": 8,
            "n_pde_srv": 8,
            "n_pde_usrv": 8,
            "n_near_hf_srv": 4,
            "n_near_srv_usrv": 4,
            "n_dirichlet": 8,
            "n_neumann": 8,
            "n_interface_hf_srv": 8,
            "n_interface_srv_usrv": 8,
            "n_hf_main_link": 8,
            "n_hf_secondary_link": 8,
            "n_hf_junction": 8,
        }
    )
    diagnostics = compute_diagnostics(config, geometry, model, device, dtype)
    assert not diagnostics.empty
    assert "metric" in diagnostics.columns
    assert "value" in diagnostics.columns
