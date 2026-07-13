from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.geometry import REGION_HF, REGION_OUTSIDE, REGION_SRV, REGION_USRV, ReservoirGeometry


def make_geometry() -> ReservoirGeometry:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    return ReservoirGeometry(config["geometry"], data_dir=root / "data")


def test_region_priority_and_outside() -> None:
    geom = make_geometry()
    points = np.asarray(
        [
            [0.0, 0.0],
            [200.0, 75.0],
            [200.0, 60.0],
            [100.0, 75.0],
            [400.0, 75.0],
        ],
        dtype=float,
    )
    region = geom.region_id_np(points[:, 0], points[:, 1])
    assert region[0] == REGION_USRV
    assert region[1] == REGION_HF
    assert region[2] == REGION_SRV
    assert region[3] == REGION_USRV
    assert region[4] == REGION_OUTSIDE


def test_distance_to_dirichlet_is_zero_on_segment() -> None:
    geom = make_geometry()
    x = torch.tensor([[360.0]], dtype=torch.float32)
    y = torch.tensor([[75.0]], dtype=torch.float32)
    dist = geom.distance_to_dirichlet_torch(x, y)
    assert torch.abs(dist).item() < 1.0e-7


def test_rect_sdf_and_hf_boundary_distance() -> None:
    geom = make_geometry()

    x_inside = torch.tensor([[200.0]], dtype=torch.float32)
    y_inside = torch.tensor([[75.0]], dtype=torch.float32)
    sdf_inside = geom.signed_distance_to_rect_torch(x_inside, y_inside, geom.main_frac)
    assert sdf_inside.item() < 0.0

    x_boundary = torch.tensor([[200.0]], dtype=torch.float32)
    y_boundary = torch.tensor([[74.995]], dtype=torch.float32)
    sdf_boundary = geom.signed_distance_to_rect_torch(x_boundary, y_boundary, geom.main_frac)
    assert abs(sdf_boundary.item()) < 1.0e-5

    x_outside = torch.tensor([[200.0]], dtype=torch.float32)
    y_outside = torch.tensor([[76.0]], dtype=torch.float32)
    sdf_outside = geom.signed_distance_to_rect_torch(x_outside, y_outside, geom.main_frac)
    assert sdf_outside.item() > 0.0

    d_center = geom.distance_to_hf_torch(x_inside, y_inside)
    d_boundary = geom.distance_to_hf_torch(x_boundary, y_boundary)
    assert d_center.item() > d_boundary.item()
    assert d_boundary.item() < 2.0e-6
