from __future__ import annotations

import copy

import torch

from src.config import load_config
from src.geometry import ReservoirGeometry
from src.losses import compute_ipinn_step_loss
from src.model import PINNModel
from src.rdfm_assembly import assemble_rdfm_operators
from src.rdfm_fractures import fractures_from_geometry
from src.rdfm_mesh import build_structured_mesh
from src.utils import dirichlet_target_hat


def _build_context():
    config = copy.deepcopy(load_config("config/default.yaml"))
    config["mesh"]["nx"] = 8
    config["mesh"]["ny"] = 6
    config["model"]["hidden_units"] = 16
    config["model"]["hidden_layers"] = 2
    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    operators = assemble_rdfm_operators(mesh, fractures_from_geometry(geometry), config["physics"], torch.device("cpu"), torch.float32)
    model = PINNModel(config)
    node_xy = mesh.node_xy_torch(torch.device("cpu"), torch.float32)
    free_nodes = torch.as_tensor(mesh.free_nodes, dtype=torch.long)
    dirichlet_nodes = torch.as_tensor(mesh.dirichlet_nodes, dtype=torch.long)
    return config, mesh, operators, model, node_xy, free_nodes, dirichlet_nodes


def test_constant_initial_field_has_near_zero_residual_when_boundary_is_unchanged() -> None:
    config, mesh, operators, model, node_xy, free_nodes, dirichlet_nodes = _build_context()
    u_prev = torch.ones((mesh.num_nodes, 2), dtype=torch.float32)
    target = dirichlet_target_hat(torch.tensor([[0.0]], dtype=torch.float32), config["boundary"])
    loss, diagnostics, _u_pred = compute_ipinn_step_loss(model, node_xy, operators, u_prev, free_nodes, dirichlet_nodes, target, 86400.0)
    assert float(loss.detach()) < 1.0e-6
    assert float(diagnostics["loss_u12"].detach()) < 1.0e-6


def test_dirichlet_nodes_are_hard_constrained_and_loss_backpropagates() -> None:
    _config, mesh, operators, model, node_xy, free_nodes, dirichlet_nodes = _build_context()
    u_prev = torch.ones((mesh.num_nodes, 2), dtype=torch.float32)
    target = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
    loss, _diagnostics, u_pred = compute_ipinn_step_loss(model, node_xy, operators, u_prev, free_nodes, dirichlet_nodes, target, 86400.0)
    assert torch.allclose(u_pred[dirichlet_nodes], target.expand(dirichlet_nodes.numel(), 2))
    assert torch.isfinite(loss)
    loss.backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)
