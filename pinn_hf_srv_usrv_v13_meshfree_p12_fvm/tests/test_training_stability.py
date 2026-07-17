from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.config import load_config
from src.geometry import ReservoirGeometry
from src.losses import compute_total_loss
from src.model import PINNModel
from src.physics import line_pde_residual
from src.sampler import ReservoirSampler
from src.train import apply_training_freeze, augment_with_adaptive_pde_points, validation_selection_loss
from src.utils import force_cpu, get_torch_dtype


def test_validation_selection_metric_can_ignore_hf_pde_offset() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["training"]["validation"]["selection_metric"] = "diagnostic"
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    validation_loss = torch.tensor(1.0e9, dtype=dtype)
    diagnostics = {
        "loss_pde_p12_hf": torch.tensor(1.0e9, dtype=dtype),
        "loss_pde_p12_srv": torch.tensor(2.0, dtype=dtype),
        "loss_pde_p12_usrv": torch.tensor(0.5, dtype=dtype),
        "loss_interface_pressure": torch.tensor(0.01, dtype=dtype),
        "loss_interface_pressure_hf_srv": torch.tensor(0.005, dtype=dtype),
        "loss_interface_pressure_srv_usrv": torch.tensor(0.006, dtype=dtype),
        "loss_interface_flux": torch.tensor(0.02, dtype=dtype),
        "loss_interface_flux_hf_srv": torch.tensor(0.007, dtype=dtype),
        "loss_interface_flux_srv_usrv": torch.tensor(0.008, dtype=dtype),
        "loss_hf_main_link": torch.tensor(0.03, dtype=dtype),
        "loss_hf_secondary_link": torch.tensor(0.04, dtype=dtype),
        "loss_hf_junction": torch.tensor(0.05, dtype=dtype),
        "loss_hf_junction_flux": torch.tensor(0.01, dtype=dtype),
        "loss_pressure_range": torch.tensor(0.0, dtype=dtype),
        "loss_correction_regularization": torch.tensor(0.06, dtype=dtype),
        "loss_local_conservation": torch.tensor(0.02, dtype=dtype),
    }

    metric = validation_selection_loss(validation_loss, diagnostics, config)

    assert metric < 10.0


def test_validation_selection_metric_can_use_local_conservation() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    config["training"]["validation"]["selection_metric"] = "diagnostic"
    for key in list(config["training"]["validation"]["selection_weights"].keys()):
        config["training"]["validation"]["selection_weights"][key] = 0.0
    config["training"]["validation"]["selection_weights"]["local_conservation"] = 4.0
    dtype = get_torch_dtype(config["runtime"]["dtype"])

    metric = validation_selection_loss(
        torch.tensor(1000.0, dtype=dtype),
        {"loss_local_conservation": torch.tensor(1.25e-4, dtype=dtype)},
        config,
    )

    assert abs(metric - 5.0e-4) < 1.0e-12


def test_split_srv_usrv_interface_weight_enters_total_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
        }
    )
    for key in list(config["loss_weights"].keys()):
        config["loss_weights"][key] = 0.0
    config["loss_weights"]["interface_pressure_srv_usrv"] = 1.0
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    model = PINNModel(config).to(device=device, dtype=dtype)
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    samples = sampler.sample_all()

    total, diagnostics = compute_total_loss(model, samples, geometry, config["physics"], config)

    assert torch.allclose(total, diagnostics["loss_interface_pressure_srv_usrv"], atol=1.0e-12, rtol=1.0e-12)


def test_region_pde_weights_can_disable_hf_pde_in_total_pde_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["loss_weights"]["pde_hf"] = 0.0
    config["loss_weights"]["pde_srv"] = 1.0
    config["loss_weights"]["pde_usrv"] = 1.0
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_srv_usrv": 2,
            "n_time_pde": 2,
        }
    )
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    model = PINNModel(config).to(device=device, dtype=dtype)
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    samples = sampler.sample_all()

    from src.losses import compute_pde_loss

    loss_pde, diagnostics = compute_pde_loss(model, samples["pde"], config["physics"], config)
    expected = 0.5 * (diagnostics["loss_pde_p12_srv"] + diagnostics["loss_pde_p12_usrv"])

    assert torch.allclose(loss_pde, expected, atol=1.0e-12, rtol=1.0e-12)


def test_gradient_enhanced_pde_weight_enters_total_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_hf_tip_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_hf_tip_neumann": 4,
            "n_hf_leakoff_balance": 4,
            "n_symmetry_hf": 3,
            "n_symmetry_srv": 3,
            "n_symmetry_usrv": 3,
            "n_conservation_srv": 2,
            "n_conservation_usrv": 2,
            "n_conservation_hf_tip_srv": 0,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
            "n_time_symmetry": 2,
        }
    )
    config["training"]["gradient_enhanced_pde"]["enabled"] = True
    config["training"]["gradient_enhanced_pde"]["max_points_per_region"] = 3
    for key in list(config["loss_weights"].keys()):
        config["loss_weights"][key] = 0.0
    config["loss_weights"]["gradient_enhanced_pde"] = 1.0
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()

    total, diagnostics = compute_total_loss(model, samples, geometry, config["physics"], config)

    assert torch.allclose(total, diagnostics["loss_gradient_enhanced_pde"], atol=1.0e-12, rtol=1.0e-12)


def test_hf_tip_neumann_weight_enters_total_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_hf_tip_neumann": 4,
            "n_conservation_srv": 2,
            "n_conservation_usrv": 2,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
        }
    )
    for key in list(config["loss_weights"].keys()):
        config["loss_weights"][key] = 0.0
    config["loss_weights"]["hf_tip_neumann"] = 1.0
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()

    total, diagnostics = compute_total_loss(model, samples, geometry, config["physics"], config)

    assert torch.allclose(total, diagnostics["loss_hf_tip_neumann"], atol=1.0e-12, rtol=1.0e-12)


def test_hf_leakoff_balance_weight_enters_total_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_hf_tip_neumann": 4,
            "n_hf_leakoff_balance": 4,
            "n_conservation_srv": 2,
            "n_conservation_usrv": 2,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
        }
    )
    for key in list(config["loss_weights"].keys()):
        config["loss_weights"][key] = 0.0
    config["loss_weights"]["hf_leakoff_balance"] = 1.0
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()

    total, diagnostics = compute_total_loss(model, samples, geometry, config["physics"], config)

    assert torch.allclose(total, diagnostics["loss_hf_leakoff_balance"], atol=1.0e-12, rtol=1.0e-12)


def test_hf_junction_flux_weight_enters_total_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_hf_junction_flux": 4,
            "n_hf_tip_neumann": 4,
            "n_hf_leakoff_balance": 4,
            "n_conservation_srv": 2,
            "n_conservation_usrv": 2,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
            "n_time_hf_junction_flux": 2,
        }
    )
    for key in list(config["loss_weights"].keys()):
        config["loss_weights"][key] = 0.0
    config["loss_weights"]["hf_junction_flux"] = 1.0
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()

    total, diagnostics = compute_total_loss(model, samples, geometry, config["physics"], config)

    assert torch.allclose(total, diagnostics["loss_hf_junction_flux"], atol=1.0e-12, rtol=1.0e-12)


def test_hf_segment_conservation_weight_enters_total_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_hf_tip_neumann": 4,
            "n_hf_leakoff_balance": 4,
            "n_hf_segment_conservation": 4,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
            "n_time_hf_segment_conservation": 2,
        }
    )
    for key in list(config["loss_weights"].keys()):
        config["loss_weights"][key] = 0.0
    config["loss_weights"]["hf_segment_conservation"] = 1.0
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()

    total, diagnostics = compute_total_loss(model, samples, geometry, config["physics"], config)

    assert torch.allclose(total, diagnostics["loss_hf_segment_conservation"], atol=1.0e-12, rtol=1.0e-12)


def test_symmetry_weight_enters_total_loss() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_hf_tip_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_hf_tip_neumann": 4,
            "n_hf_leakoff_balance": 4,
            "n_symmetry_hf": 3,
            "n_symmetry_srv": 3,
            "n_symmetry_usrv": 3,
            "n_conservation_srv": 2,
            "n_conservation_usrv": 2,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
            "n_time_symmetry": 2,
        }
    )
    for key in list(config["loss_weights"].keys()):
        config["loss_weights"][key] = 0.0
    config["loss_weights"]["symmetry"] = 1.0
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()

    total, diagnostics = compute_total_loss(model, samples, geometry, config["physics"], config)

    assert torch.allclose(total, diagnostics["loss_symmetry"], atol=1.0e-12, rtol=1.0e-12)


def test_freeze_base_for_tip_expert_only_leaves_tip_parameters_trainable() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["local_tip_expert"]["enabled"] = True
    config["training"]["freeze_base_for_tip_expert"] = True
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)

    apply_training_freeze(model, config)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all(name.startswith("tip_expert.") for name in trainable_names)


def test_freeze_base_for_local_experts_only_leaves_enabled_experts_trainable() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["model"]["local_fracture_expert"]["enabled"] = True
    config["training"]["freeze_base_for_local_experts"] = True
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)

    apply_training_freeze(model, config)

    trainable_names = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable_names
    assert all(name.startswith("fracture_expert.") for name in trainable_names)


def test_analytic_base_smooth_max_keeps_line_pde_residual_finite_near_junction() -> None:
    config = load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml")
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    model = PINNModel(config).to(dtype=dtype)
    assert model.analytic_base_smooth_max_tau == float(config["model"]["analytic_base"]["smooth_max_tau"])

    xyt = torch.tensor(
        [
            [225.3175, 75.0, 100.0],
            [252.2530, 75.0, 100.0],
            [279.1885, 75.0, 100.0],
        ],
        dtype=dtype,
    )
    tangent = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=dtype)

    residual = line_pde_residual(model, xyt, tangent, config["physics"], config)

    assert torch.isfinite(residual).all()


def test_adaptive_resampling_can_add_local_conservation_rectangles() -> None:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_pde_hf": 4,
            "n_pde_srv": 4,
            "n_pde_usrv": 4,
            "n_near_hf_srv": 2,
            "n_near_srv_usrv": 2,
            "n_dirichlet": 4,
            "n_neumann": 4,
            "n_interface_hf_srv": 4,
            "n_interface_srv_usrv": 4,
            "n_hf_main_link": 4,
            "n_hf_secondary_link": 4,
            "n_hf_junction": 4,
            "n_conservation_srv": 3,
            "n_conservation_usrv": 3,
            "conservation_min_half_size_m": 1.0,
            "conservation_max_half_size_m": 3.0,
            "n_time_pde": 2,
            "n_time_boundary": 2,
            "n_time_interface": 2,
            "n_time_link": 2,
        }
    )
    config["training"]["adaptive_resampling"].update(
        {
            "enabled": True,
            "candidate_multiplier": 2,
            "keep_hf": 0,
            "keep_srv": 0,
            "keep_usrv": 0,
            "keep_conservation_srv": 2,
            "keep_conservation_usrv": 2,
        }
    )
    device = force_cpu(int(config["runtime"]["cpu_threads"]))
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()
    before_srv = samples["local_conservation"]["srv"]["center"].shape[0]
    before_usrv = samples["local_conservation"]["usrv"]["center"].shape[0]

    augmented = augment_with_adaptive_pde_points(samples, model, sampler, config)

    assert augmented["local_conservation"]["srv"]["center"].shape[0] == before_srv + 2
    assert augmented["local_conservation"]["usrv"]["center"].shape[0] == before_usrv + 2
