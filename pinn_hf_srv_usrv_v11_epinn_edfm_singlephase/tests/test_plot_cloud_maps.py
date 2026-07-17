from __future__ import annotations

import numpy as np
import pytest

import plot_cloud_maps as cloud


def _snapshot(offset: float = 0.0) -> dict[str, np.ndarray]:
    pressure = np.asarray(
        [
            [[10.0 + offset, 1.0], [20.0 + offset, 2.0]],
            [[30.0 + offset, 3.0], [40.0 + offset, 4.0]],
        ],
        dtype=np.float64,
    )
    return {
        "times_days": np.asarray([0.0, 10.0], dtype=np.float64),
        "pressure_mpa": pressure,
        "cell_xy": np.asarray([[0.5, 0.5], [1.5, 0.5]], dtype=np.float64),
        "matrix_cell_count": np.asarray(2, dtype=np.int64),
        "nx": np.asarray(2, dtype=np.int64),
        "ny": np.asarray(1, dtype=np.int64),
        "x_edges": np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        "y_edges": np.asarray([0.0, 1.0], dtype=np.float64),
        "well_cells": np.asarray([], dtype=np.int64),
        "component_names": np.asarray(["P12", "P13"]),
        "fracture_start": np.zeros((0, 2), dtype=np.float64),
        "fracture_end": np.zeros((0, 2), dtype=np.float64),
    }


def test_build_comparison_uses_requested_component_and_nearest_time() -> None:
    source = _snapshot(offset=1.0)
    fvm = _snapshot(offset=0.0)

    result = cloud.build_comparison(source, fvm, "P12", 9.0)

    assert result["source_time"] == 10.0
    assert result["fvm_time"] == 10.0
    assert np.allclose(result["source"], [31.0, 41.0])
    assert np.allclose(result["fvm"], [30.0, 40.0])
    assert np.allclose(result["error"], [1.0, 1.0])
    assert result["rmse"] == pytest.approx(1.0)


def test_extract_ptotal_sums_pressure_components() -> None:
    values = cloud.extract_variable(_snapshot(), 0, "Ptotal")

    assert np.allclose(values, [11.0, 22.0])


def test_validate_same_matrix_grid_rejects_mismatched_grid() -> None:
    source = _snapshot()
    fvm = _snapshot()
    fvm["nx"] = np.asarray(3, dtype=np.int64)

    with pytest.raises(ValueError, match="matrix grids do not match"):
        cloud.validate_same_matrix_grid(source, fvm)
