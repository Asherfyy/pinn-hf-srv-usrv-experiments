from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import load_config
from src.geometry import REGION_HF, REGION_OUTSIDE, REGION_SRV, REGION_USRV, ReservoirGeometry


def make_geometry() -> ReservoirGeometry:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    return ReservoirGeometry(config["geometry"])


def test_region_priority_and_outside() -> None:
    geom = make_geometry()
    points = np.asarray(
        [
            [200.0, 75.0],
            [220.0, 60.0],
            [100.0, 75.0],
            [400.0, 75.0],
        ],
        dtype=float,
    )
    region = geom.region_id_np(points[:, 0], points[:, 1])
    assert region[0] == REGION_HF
    assert region[1] == REGION_SRV
    assert region[2] == REGION_USRV
    assert region[3] == REGION_OUTSIDE


def test_hf_priority_over_srv() -> None:
    geom = make_geometry()
    region = geom.region_id_np(np.asarray([250.0]), np.asarray([75.0]))
    assert region[0] == REGION_HF
