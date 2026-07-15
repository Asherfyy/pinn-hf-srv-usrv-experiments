from __future__ import annotations

import numpy as np

from src.config import load_config
from src.geometry import ReservoirGeometry
from src.rdfm_mesh import build_structured_mesh


def test_default_mesh_counts_and_production_node() -> None:
    config = load_config("config/default.yaml")
    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])

    nx = int(config["mesh"]["nx"])
    ny = int(config["mesh"]["ny"])
    assert nx == 180
    assert ny == 150
    assert mesh.node_xy.shape == ((nx + 1) * (ny + 1), 2)
    assert mesh.elements.shape == (nx * ny, 4)
    assert mesh.triangles.shape == (2 * nx * ny, 3)

    assert np.any(np.isclose(mesh.y_coords, 75.0))
    production = np.asarray([360.0, 75.0])
    distances = np.linalg.norm(mesh.node_xy[mesh.dirichlet_nodes] - production[None, :], axis=1)
    assert np.min(distances) < 1.0e-8


def test_free_nodes_exclude_dirichlet_nodes() -> None:
    config = load_config("config/default.yaml")
    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    assert mesh.dirichlet_nodes.size >= 1
    assert not np.intersect1d(mesh.free_nodes, mesh.dirichlet_nodes).size
