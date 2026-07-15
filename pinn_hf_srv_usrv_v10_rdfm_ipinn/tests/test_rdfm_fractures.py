from __future__ import annotations

import numpy as np

from src.config import load_config
from src.geometry import ReservoirGeometry
from src.rdfm_fractures import fracture_intersections, fractures_from_geometry


def test_fractures_are_centerlines_with_expected_orientation_and_aperture() -> None:
    config = load_config("config/default.yaml")
    geometry = ReservoirGeometry(config["geometry"])
    fractures = fractures_from_geometry(geometry)

    assert len(fractures) == 6
    main = fractures[0]
    assert main.is_horizontal
    assert np.allclose(main.tangent, [1.0, 0.0])
    assert abs(main.start[1] - 75.0) < 1.0e-10
    assert abs(main.aperture - 0.01) < 1.0e-12

    for fracture in fractures[1:]:
        assert not fracture.is_horizontal
        assert np.allclose(fracture.tangent, [0.0, 1.0])
        assert abs(fracture.aperture - 0.01) < 1.0e-12


def test_fracture_intersections_are_detected() -> None:
    config = load_config("config/default.yaml")
    geometry = ReservoirGeometry(config["geometry"])
    intersections = fracture_intersections(fractures_from_geometry(geometry))
    assert len(intersections) == 5
    assert all(abs(point[1] - 75.0) < 1.0e-10 for point in intersections)
