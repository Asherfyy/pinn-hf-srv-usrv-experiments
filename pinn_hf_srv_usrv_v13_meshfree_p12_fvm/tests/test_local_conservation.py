from __future__ import annotations

import copy
from pathlib import Path

import torch

from src.config import load_config
from src.geometry import REGION_SRV, ReservoirGeometry
from src.local_conservation import compute_local_conservation_loss
from src.losses import compute_gradient_enhanced_pde_loss, compute_hf_junction_flux_loss, compute_hf_leakoff_balance_loss, compute_hf_segment_conservation_loss
from src.model import PINNModel
from src.sampler import ReservoirSampler
from src.utils import force_cpu, get_torch_dtype, set_seed


class ConstantICModel(PINNModel):
    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        return z[:, 0:1] * 0.0 + 1.0

    def forward_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        _ = region_name
        return self.forward_normalized(z)


def _tiny_config() -> dict:
    config = copy.deepcopy(load_config(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    config["sampler"].update(
        {
            "n_conservation_srv": 6,
            "n_conservation_usrv": 6,
            "n_conservation_hf_tip_srv": 0,
            "n_hf_leakoff_balance": 6,
            "n_hf_junction_flux": 6,
            "n_time_hf_junction_flux": 3,
            "hf_junction_flux_offset_m": 0.05,
            "n_hf_segment_conservation": 6,
            "n_time_hf_segment_conservation": 3,
            "hf_segment_min_half_length_m": 0.5,
            "hf_segment_max_half_length_m": 2.0,
            "hf_segment_leakoff_offset_m": 0.05,
            "hf_segment_endpoint_margin_m": 0.05,
            "conservation_min_half_size_m": 1.0,
            "conservation_max_half_size_m": 3.0,
            "time_pairing_mode": "paired",
        }
    )
    return config


def test_local_conservation_sampler_returns_rectangles() -> None:
    config = _tiny_config()
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)

    samples = sampler.sample_local_conservation_rectangles()

    assert samples["srv"]["center"].shape == (6, 2)
    assert samples["srv"]["half_size"].shape == (6, 2)
    assert samples["srv"]["t"].shape == (6, 1)
    assert samples["usrv"]["center"].shape == (6, 2)
    assert torch.all(samples["srv"]["half_size"] > 0.0)
    assert torch.all(samples["usrv"]["half_size"] > 0.0)


def test_hf_tip_local_conservation_rectangles_are_added_to_srv_samples() -> None:
    config = _tiny_config()
    config["sampler"]["n_conservation_hf_tip_srv"] = 5
    config["sampler"]["conservation_hf_tip_radius_m"] = 8.0
    config["sampler"]["conservation_hf_tip_min_half_size_m"] = 0.25
    config["sampler"]["conservation_hf_tip_max_half_size_m"] = 4.0
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)

    samples = sampler.sample_local_conservation_rectangles()
    center = samples["srv"]["center"].detach().cpu().numpy()
    region = geometry.region_id_np(center[:, 0], center[:, 1])

    assert samples["srv"]["center"].shape[0] == 11
    assert (region == REGION_SRV).all()
    assert torch.max(samples["srv"]["half_size"]).item() <= 4.0


def test_constant_field_has_zero_local_conservation_residual() -> None:
    config = _tiny_config()
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = ConstantICModel(config).to(device=device, dtype=dtype)

    loss, diagnostics = compute_local_conservation_loss(model, sampler.sample_local_conservation_rectangles(), config)

    assert loss.item() < 1.0e-24
    assert diagnostics["rms_local_conservation"].item() < 1.0e-12


def test_constant_field_has_zero_hf_leakoff_balance_residual() -> None:
    config = _tiny_config()
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = ConstantICModel(config).to(device=device, dtype=dtype)

    loss, diagnostics = compute_hf_leakoff_balance_loss(model, {"hf_leakoff_balance": sampler.sample_hf_leakoff_balance_points()}, config)

    assert loss.item() < 1.0e-24
    assert diagnostics["rms_hf_leakoff_balance"].item() < 1.0e-12


def test_constant_field_has_zero_hf_segment_conservation_residual() -> None:
    config = _tiny_config()
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = ConstantICModel(config).to(device=device, dtype=dtype)

    samples = {"hf_segment_conservation": sampler.sample_hf_segment_conservation_points()}
    loss, diagnostics = compute_hf_segment_conservation_loss(model, samples, config)

    assert loss.item() < 1.0e-24
    assert diagnostics["rms_hf_segment_conservation"].item() < 1.0e-12


def test_constant_field_has_zero_hf_junction_flux_residual() -> None:
    config = _tiny_config()
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = ConstantICModel(config).to(device=device, dtype=dtype)

    samples = {"hf_junction_flux": sampler.sample_hf_junction_flux_points()}
    loss, diagnostics = compute_hf_junction_flux_loss(model, samples, config)

    assert loss.item() < 1.0e-24
    assert diagnostics["rms_hf_junction_flux"].item() < 1.0e-12


def test_constant_field_has_zero_gradient_enhanced_pde_loss() -> None:
    config = _tiny_config()
    config["training"]["gradient_enhanced_pde"]["enabled"] = True
    config["training"]["gradient_enhanced_pde"]["max_points_per_region"] = 4
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = ConstantICModel(config).to(device=device, dtype=dtype)
    samples = sampler.sample_all()

    loss, diagnostics = compute_gradient_enhanced_pde_loss(model, samples["pde"], config["physics"], config)

    assert loss.item() < 1.0e-24
    assert diagnostics["rms_gpinn_spatial"].item() < 1.0e-12


def test_local_conservation_backpropagates_on_tiny_sample() -> None:
    config = _tiny_config()
    device = force_cpu(1)
    dtype = get_torch_dtype(config["runtime"]["dtype"])
    set_seed(int(config["runtime"]["seed"]))
    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=123)
    model = PINNModel(config).to(device=device, dtype=dtype)

    loss, diagnostics = compute_local_conservation_loss(model, sampler.sample_local_conservation_rectangles(), config)
    loss.backward()
    grad_norm = sum(
        float(torch.linalg.norm(param.grad.detach()).cpu())
        for param in model.parameters()
        if param.grad is not None
    )

    assert torch.isfinite(loss).item()
    assert torch.isfinite(diagnostics["loss_local_conservation_srv"]).item()
    assert torch.isfinite(diagnostics["loss_local_conservation_usrv"]).item()
    assert grad_norm >= 0.0
