from __future__ import annotations

import copy

import torch

from src.config import load_config
from src.geometry import ReservoirGeometry
from src.rdfm_assembly import assemble_rdfm_operators
from src.rdfm_fractures import fractures_from_geometry
from src.rdfm_mesh import build_structured_mesh


def _tiny_config() -> dict:
    config = copy.deepcopy(load_config("config/default.yaml"))
    config["mesh"]["nx"] = 8
    config["mesh"]["ny"] = 6
    return config


def test_assembled_matrices_are_sparse_finite_and_symmetric() -> None:
    config = _tiny_config()
    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    operators = assemble_rdfm_operators(mesh, fractures_from_geometry(geometry), config["physics"], torch.device("cpu"), torch.float32)

    for matrix in [operators.u12.mass, operators.u12.stiffness, operators.u13.mass, operators.u13.stiffness]:
        assert matrix.is_sparse
        assert matrix.shape == (mesh.num_nodes, mesh.num_nodes)
        dense = matrix.to_dense()
        assert torch.isfinite(dense).all()
        assert torch.allclose(dense, dense.T, atol=1.0e-6)


def test_constant_field_has_zero_stiffness_residual() -> None:
    config = _tiny_config()
    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    operators = assemble_rdfm_operators(mesh, fractures_from_geometry(geometry), config["physics"], torch.device("cpu"), torch.float32)
    ones = torch.ones((mesh.num_nodes, 1), dtype=torch.float32)
    residual = torch.sparse.mm(operators.u12.stiffness, ones)
    assert float(torch.max(torch.abs(residual))) < 1.0e-6


def test_fracture_terms_add_nonzero_contribution() -> None:
    config = _tiny_config()
    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    with_fractures = assemble_rdfm_operators(mesh, fractures_from_geometry(geometry), config["physics"], torch.device("cpu"), torch.float32)
    without_fractures = assemble_rdfm_operators(mesh, [], config["physics"], torch.device("cpu"), torch.float32)
    diff = with_fractures.u12.stiffness.to_dense() - without_fractures.u12.stiffness.to_dense()
    assert float(torch.sum(torch.abs(diff))) > 0.0
