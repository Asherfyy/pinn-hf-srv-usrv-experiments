from __future__ import annotations

import copy

import numpy as np

from src.config import load_config
from src.edfm_grid import REGION_HF, build_edfm_grid
from src.geometry import ReservoirGeometry


def _grid():
    config = copy.deepcopy(load_config("config/default.yaml"))
    config["grid"]["nx"] = 12
    config["grid"]["ny"] = 10
    geometry = ReservoirGeometry(config["geometry"])
    return build_edfm_grid(geometry, config)


def test_grid_cells_regions_and_fracture_segments_are_valid() -> None:
    grid = _grid()
    assert grid.matrix_cell_count == 12 * 10
    assert grid.cell_xy.shape[0] == grid.num_cells
    assert grid.fracture_segments
    assert np.any(grid.cell_region == REGION_HF)
    assert all(segment.length > 0.0 for segment in grid.fracture_segments)


def test_edfm_connections_include_mm_mf_ff_and_valid_indices() -> None:
    grid = _grid()
    kinds = {connection.kind for connection in grid.connections}
    assert {"mm", "mf", "ff"}.issubset(kinds)
    for connection in grid.connections:
        assert 0 <= connection.i < grid.num_cells
        assert 0 <= connection.j < grid.num_cells
        assert connection.i != connection.j
        assert np.isfinite(connection.transmissibility)
        assert connection.transmissibility >= 0.0
    assert grid.adjacency.shape == (grid.num_cells, grid.num_cells)
    assert np.allclose(grid.adjacency, grid.adjacency.T)
    assert grid.edge_index.shape[0] == 2
    assert grid.edge_weight.shape == (grid.edge_index.shape[1],)
    assert np.all(grid.edge_index >= 0)
    assert np.max(grid.edge_index) < grid.num_cells
    assert np.all(np.isfinite(grid.edge_weight))
    assert np.all(grid.edge_weight > 0.0)
    assert grid.well_cell in set(grid.well_cells.tolist())
    assert grid.well_cells.size >= 2


def test_large_grid_skips_dense_adjacency_but_keeps_sparse_edges() -> None:
    config = copy.deepcopy(load_config("config/default.yaml"))
    config["grid"]["nx"] = 80
    config["grid"]["ny"] = 70
    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    assert grid.num_cells > int(config["edfm"]["max_dense_elements"])
    assert grid.adjacency is None
    assert grid.edge_index.shape[0] == 2
    assert grid.edge_weight.shape == (grid.edge_index.shape[1],)
