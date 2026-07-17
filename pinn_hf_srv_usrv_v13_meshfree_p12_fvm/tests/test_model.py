from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.config import load_config
from src.model import PINNModel
from src.utils import dirichlet_target_hat, get_torch_dtype, pressure_hat_to_mpa


def test_model_input_output_and_initial_condition() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    xyt = torch.tensor([[200.0, 75.0, 0.0], [250.0, 75.0, 10.0]], dtype=dtype)
    pred = model(xyt)
    assert pred.shape == (2, 1)
    assert torch.max(torch.abs(pred[0:1] - torch.ones_like(pred[0:1]))).item() < 1.0e-12


def test_model_uses_partitioned_subnets_and_configured_features() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config)
    assert set(model.subnets.keys()) == {"HF", "SRV", "USRV"}
    expected_dim = int(config["model"]["subnet_input_dim"])
    assert expected_dim in {3, 5}
    assert model.subnets["HF"].net[0].in_features == expected_dim
    xyt = torch.tensor([[250.0, 75.0, 10.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)
    features = model.features_for_region(z, "HF")
    assert features.shape == (1, expected_dim)


def test_5d_hf_local_features_use_length_axis_for_line_fractures() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["subnet_input_dim"] = 5
    config["model"]["share_srv_usrv_subnet"] = False
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    xyt = torch.tensor([[250.0, 74.995, 10.0], [250.0, 75.000, 10.0], [250.0, 75.005, 10.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)
    features = model.features_for_region(z, "HF")
    assert features.shape[1] == 5
    assert torch.allclose(features[:, 1], torch.full((3,), 0.5, dtype=dtype), atol=1.0e-10)


def test_shared_srv_usrv_subnet_routes_usrv_through_srv_network() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["share_srv_usrv_subnet"] = True
    config["model"]["subnet_input_dim"] = 3
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    with torch.no_grad():
        model.subnets["SRV"].net[-1].bias[:] = torch.tensor([0.25], dtype=dtype)
        model.subnets["USRV"].net[-1].bias[:] = torch.tensor([0.75], dtype=dtype)
    xyt = torch.tensor([[100.0, 20.0, 10.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)

    raw_usrv = model.forward_raw_region_normalized(z, "USRV")

    assert torch.max(torch.abs(raw_usrv - torch.tensor([[0.25]], dtype=dtype))).item() < 1.0e-12


def test_analytic_base_is_nearly_continuous_across_srv_usrv_interface() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    eps = float(config["sampler"]["eps_srv_usrv"])
    t = 1000.0
    pairs = []
    y = torch.tensor([40.0, 75.0, 110.0], dtype=dtype)
    pairs.append(
        torch.cat(
            [
                torch.stack([torch.full_like(y, 180.0 + eps), y, torch.full_like(y, t)], dim=1),
                torch.stack([torch.full_like(y, 180.0 - eps), y, torch.full_like(y, t)], dim=1),
            ],
            dim=0,
        )
    )
    x = torch.tensor([181.0, 270.0, 359.0], dtype=dtype)
    for y_srv, y_usrv in [(37.5 + eps, 37.5 - eps), (112.5 - eps, 112.5 + eps)]:
        pairs.append(
            torch.cat(
                [
                    torch.stack([x, torch.full_like(x, y_srv), torch.full_like(x, t)], dim=1),
                    torch.stack([x, torch.full_like(x, y_usrv), torch.full_like(x, t)], dim=1),
                ],
                dim=0,
            )
        )

    for xyt in pairs:
        z = model.normalize_xyt(xyt)
        zero = torch.zeros((xyt.shape[0], 1), dtype=dtype)
        with torch.no_grad():
            base = model._base_field(z, zero)
            pressure = pressure_hat_to_mpa(base, config["boundary"]).reshape(2, -1)
        jump = pressure[0] - pressure[1]
        assert torch.max(torch.abs(jump)).item() < 0.1


def test_union_analytic_base_does_not_exceed_total_bhp_drawdown() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["analytic_base"]["aggregation"] = "probabilistic_union"
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    x = torch.linspace(198.5, 360.0, 21, dtype=dtype)
    y = torch.linspace(37.5, 112.5, 11, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    tt = torch.full_like(xx, 100.0)
    xyt = torch.stack([xx.reshape(-1), yy.reshape(-1), tt.reshape(-1)], dim=1)

    pred = model(xyt)
    target = dirichlet_target_hat(xyt[:, 2:3], config["boundary"])

    assert torch.min(pred - target).item() >= -1.0e-12


def test_secondary_length_scale_reduces_early_branch_drawdown() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    scaled_model = PINNModel(config).to(dtype=dtype)
    unscaled_config = copy.deepcopy(config)
    unscaled_config["model"]["analytic_base"]["secondary_length_scale"] = 1.0
    unscaled_model = PINNModel(unscaled_config).to(dtype=dtype)
    xyt = torch.tensor([[333.0595, 102.0, 100.0]], dtype=dtype)

    scaled_pressure = scaled_model(xyt)
    unscaled_pressure = unscaled_model(xyt)

    assert torch.min(scaled_pressure - unscaled_pressure).item() > 0.0


def test_main_length_scale_reduces_early_main_fracture_drawdown() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    scaled_config = copy.deepcopy(config)
    scaled_config["model"]["analytic_base"]["main_length_scale"] = 0.01
    scaled_model = PINNModel(scaled_config).to(dtype=dtype)
    unscaled_config = copy.deepcopy(config)
    unscaled_config["model"]["analytic_base"]["main_length_scale"] = 1.0
    unscaled_model = PINNModel(unscaled_config).to(dtype=dtype)
    xyt = torch.tensor([[198.5, 75.0, 100.0]], dtype=dtype)

    scaled_pressure = scaled_model(xyt)
    unscaled_pressure = unscaled_model(xyt)

    assert torch.min(scaled_pressure - unscaled_pressure).item() > 0.0


def test_endpoint_taper_reduces_drawdown_beyond_secondary_tip() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    no_taper_config = copy.deepcopy(config)
    taper_config = copy.deepcopy(config)
    no_taper_config["model"]["analytic_base"]["endpoint_taper_enabled"] = False
    taper_config["model"]["analytic_base"]["endpoint_taper_enabled"] = True
    taper_config["model"]["analytic_base"]["endpoint_taper_m"] = 1.0
    no_taper_model = PINNModel(no_taper_config).to(dtype=dtype)
    taper_model = PINNModel(taper_config).to(dtype=dtype)
    beyond_tip = torch.tensor([[333.0595, 105.0, 200.0]], dtype=dtype)
    on_segment = torch.tensor([[333.0595, 100.0, 200.0]], dtype=dtype)

    no_taper_beyond = no_taper_model(beyond_tip)
    taper_beyond = taper_model(beyond_tip)
    no_taper_on = no_taper_model(on_segment)
    taper_on = taper_model(on_segment)

    assert taper_beyond.item() > no_taper_beyond.item()
    assert torch.max(torch.abs(taper_on - no_taper_on)).item() < 1.0e-12


def test_secondary_length_scale_gradient_increases_far_branch_drawdown() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    flat_config = copy.deepcopy(config)
    flat_config["model"]["analytic_base"]["secondary_length_scale_gradient"] = 0.0
    graded_config = copy.deepcopy(config)
    graded_config["model"]["analytic_base"]["secondary_length_scale_gradient"] = 1.0
    flat_model = PINNModel(flat_config).to(dtype=dtype)
    graded_model = PINNModel(graded_config).to(dtype=dtype)
    near = torch.tensor([[333.0595, 100.0, 100.0]], dtype=dtype)
    far = torch.tensor([[225.3175, 100.0, 100.0]], dtype=dtype)

    near_change = graded_model(near) - flat_model(near)
    far_change = graded_model(far) - flat_model(far)

    assert far_change.item() < near_change.item()


def test_base_correction_uses_attached_previous_time_field() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["constraint_mode"] = "ic_base_correction"
    config["model"]["hard_dirichlet"]["enabled"] = False
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)

    base_config = copy.deepcopy(config)
    base_config["model"]["constraint_mode"] = "ic_hard"
    base_model = PINNModel(base_config).to(dtype=dtype)
    with torch.no_grad():
        for subnet in base_model.subnets.values():
            subnet.net[-1].bias[:] = torch.tensor([0.25], dtype=dtype)
    model.attach_base_model(base_model)

    xyt = torch.tensor([[250.0, 75.0, 10.0], [300.0, 75.0, 100.0]], dtype=dtype)
    z = model.normalize_xyt(xyt)
    expected = base_model.forward_normalized(model._base_z(z))
    pred = model(xyt)
    assert torch.max(torch.abs(pred - expected)).item() < 1.0e-12


def test_softsign_correction_activation_is_bounded_and_less_saturated() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["constraint_mode"] = "ic_base_correction"
    config["model"]["hard_dirichlet"]["enabled"] = False
    config["model"]["analytic_base"]["enabled"] = False
    config["model"]["correction_activation"] = "softsign"
    config["model"]["correction_scale"] = 0.1
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    raw = torch.tensor([[-3.0], [0.0], [3.0]], dtype=dtype)

    correction = model._bounded_correction(raw)

    assert torch.max(torch.abs(correction)).item() < 0.1
    assert torch.allclose(correction[2], torch.tensor([0.075], dtype=dtype))


def test_region_specific_correction_scale_can_target_hf_only() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["model"]["correction_scale"] = 0.05
    config["model"]["correction_scale_by_region"] = {"HF": 0.12, "SRV": 0.05, "USRV": 0.04}
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    raw = torch.full((2, 1), 10.0, dtype=dtype)

    hf = model._bounded_correction(raw, "HF")
    srv = model._bounded_correction(raw, "SRV")
    usrv = model._bounded_correction(raw, "USRV")

    assert torch.max(hf).item() > 0.11
    assert torch.max(srv).item() < 0.051
    assert torch.max(usrv).item() < 0.041


def test_local_tip_expert_is_zero_initialized_and_locally_gated() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["local_tip_expert"]["enabled"] = True
    config["model"]["local_tip_expert"]["radius_m"] = 8.0
    config["model"]["local_tip_expert"]["scale"] = 0.02
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    near = torch.tensor([[333.0595, 103.922625, 200.0]], dtype=dtype)
    far = torch.tensor([[20.0, 20.0, 200.0]], dtype=dtype)
    z_near = model.normalize_xyt(near)
    z_far = model.normalize_xyt(far)
    reference = torch.zeros((1, 1), dtype=dtype)

    assert model.tip_expert is not None
    assert torch.max(torch.abs(model._local_tip_correction(z_near, reference))).item() < 1.0e-12
    with torch.no_grad():
        model.tip_expert.net[-1].bias[:] = torch.tensor([1.0], dtype=dtype)
    near_correction = torch.abs(model._local_tip_correction(z_near, reference))
    far_correction = torch.abs(model._local_tip_correction(z_far, reference))

    assert near_correction.item() > 100.0 * far_correction.item()


def test_local_fracture_expert_is_zero_initialized_and_matrix_gated() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["local_fracture_expert"]["enabled"] = True
    config["model"]["local_fracture_expert"]["radius_m"] = 6.0
    config["model"]["local_fracture_expert"]["scale"] = 0.2
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    near = torch.tensor([[333.2, 104.5, 200.0]], dtype=dtype)
    far = torch.tensor([[20.0, 20.0, 200.0]], dtype=dtype)
    z_near = model.normalize_xyt(near)
    z_far = model.normalize_xyt(far)
    reference = torch.zeros((1, 1), dtype=dtype)

    assert model.fracture_expert is not None
    assert torch.max(torch.abs(model._local_fracture_correction(z_near, reference, "SRV"))).item() < 1.0e-12
    with torch.no_grad():
        model.fracture_expert.net[-1].bias[:] = torch.tensor([1.0], dtype=dtype)
    near_correction = torch.abs(model._local_fracture_correction(z_near, reference, "SRV"))
    far_correction = torch.abs(model._local_fracture_correction(z_far, reference, "USRV"))
    hf_correction = torch.abs(model._local_fracture_correction(z_near, reference, "HF"))

    assert near_correction.item() > 100.0 * far_correction.item()
    assert hf_correction.item() < 1.0e-12


def test_hard_dirichlet_enforces_producer_pressure() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["hard_dirichlet"]["enabled"] = True
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    xyt = torch.tensor([[360.0, 75.0, 100.0]], dtype=dtype)
    pred = model(xyt)
    target = dirichlet_target_hat(xyt[:, 2:3], config["boundary"])
    assert torch.max(torch.abs(pred - target)).item() < 1.0e-12


def test_maximum_principle_clamps_pressure_hat() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["enforce_maximum_principle"] = True
    config["model"]["hard_dirichlet"]["enabled"] = False
    config["model"]["analytic_base"]["enabled"] = False
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    with torch.no_grad():
        for subnet in model.subnets.values():
            subnet.net[-1].bias[:] = torch.tensor([10.0], dtype=dtype)
    xyt = torch.tensor([[250.0, 75.0, 100.0]], dtype=dtype)
    assert torch.max(model(xyt)).item() <= 1.0
