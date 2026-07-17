"""Partitioned MLP PINN model for HF/SRV/USRV regions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .geometry import REGION_HF, REGION_SRV, REGION_USRV, Rect, ReservoirGeometry
from .utils import dirichlet_target_hat


class MLP(nn.Module):
    """Fully connected network with Xavier initialization."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str) -> None:
        super().__init__()
        if activation.lower() == "tanh":
            act_factory = nn.Tanh
        elif activation.lower() == "silu":
            act_factory = nn.SiLU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers: list[nn.Module] = []
        in_dim = int(input_dim)
        for _ in range(int(hidden_layers)):
            layers.append(nn.Linear(in_dim, int(hidden_units)))
            layers.append(act_factory())
            in_dim = int(hidden_units)
        layers.append(nn.Linear(in_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        linear_layers = [module for module in self.modules() if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        # Start from the exact constant initial-pressure field. This prevents
        # the thin-fracture local coordinates from creating huge random PDE
        # curvature before the boundary signal has been learned.
        nn.init.zeros_(linear_layers[-1].weight)
        nn.init.zeros_(linear_layers[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class PINNModel(nn.Module):
    """Three-subnet PINN: one MLP for HF, SRV, and USRV each.

    The public coordinate normalization remains z=[x_hat, y_hat, t_hat].
    By default, each region subnet receives this same 3D normalized coordinate.
    The legacy 5D feature mode [x_local, y_local, x_hat, y_hat, t_hat] is still
    supported by setting model.subnet_input_dim=5.
    """

    REGION_KEY_BY_ID = {
        REGION_HF: "HF",
        REGION_SRV: "SRV",
        REGION_USRV: "USRV",
    }

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        if int(model_cfg["input_dim"]) != 3:
            raise ValueError("The public model input must be x_hat/y_hat/t_hat with dimension 3.")
        self.constraint_mode = str(model_cfg["constraint_mode"]).lower()
        if self.constraint_mode not in {"ic_hard", "ic_base_correction"}:
            raise ValueError("The v13 partitioned model supports constraint_mode='ic_hard' or 'ic_base_correction'.")

        self.config = config
        self.geometry = ReservoirGeometry(config["geometry"])
        self.subnet_input_dim = int(model_cfg.get("subnet_input_dim", 5))
        if self.subnet_input_dim not in {3, 5}:
            raise ValueError("model.subnet_input_dim must be 3 or 5.")
        self.share_srv_usrv_subnet = bool(model_cfg.get("share_srv_usrv_subnet", False))
        if self.share_srv_usrv_subnet and self.subnet_input_dim != 3:
            raise ValueError("model.share_srv_usrv_subnet requires model.subnet_input_dim=3.")
        self.base_time_lag_days = float(model_cfg.get("base_time_lag_days", 0.0))
        self.correction_envelope_power = float(model_cfg.get("correction_envelope_power", 1.0))
        self.correction_scale = float(model_cfg.get("correction_scale", 1.0))
        self.correction_scale_by_region = {
            str(region).upper(): float(scale)
            for region, scale in model_cfg.get("correction_scale_by_region", {}).items()
        }
        self.correction_activation = str(model_cfg.get("correction_activation", "tanh")).lower()
        if self.correction_activation not in {"tanh", "softsign"}:
            raise ValueError("model.correction_activation must be 'tanh' or 'softsign'.")
        local_tip_cfg = model_cfg.get("local_tip_expert", {})
        self.local_tip_expert_enabled = bool(local_tip_cfg.get("enabled", False))
        self.local_tip_expert_radius_m = float(local_tip_cfg.get("radius_m", 8.0))
        self.local_tip_expert_scale = float(local_tip_cfg.get("scale", 0.0))
        self.local_tip_expert_gate_power = float(local_tip_cfg.get("gate_power", 1.0))
        local_frac_cfg = model_cfg.get("local_fracture_expert", {})
        self.local_fracture_expert_enabled = bool(local_frac_cfg.get("enabled", False))
        self.local_fracture_expert_radius_m = float(local_frac_cfg.get("radius_m", 8.0))
        self.local_fracture_expert_scale = float(local_frac_cfg.get("scale", 0.0))
        self.local_fracture_expert_gate_power = float(local_frac_cfg.get("gate_power", 1.0))
        self.local_fracture_expert_matrix_only = bool(local_frac_cfg.get("matrix_only", True))
        self.enforce_maximum_principle = bool(model_cfg.get("enforce_maximum_principle", False))
        hard_dirichlet_cfg = model_cfg.get("hard_dirichlet", {})
        self.hard_dirichlet_enabled = bool(hard_dirichlet_cfg.get("enabled", False))
        self.hard_dirichlet_radius_m = float(hard_dirichlet_cfg.get("radius_m", 1.0))
        self.hard_dirichlet_power = float(hard_dirichlet_cfg.get("power", 2.0))
        analytic_base_cfg = model_cfg.get("analytic_base", {})
        self.analytic_base_enabled = bool(analytic_base_cfg.get("enabled", False))
        self.analytic_base_length_multiplier = float(analytic_base_cfg.get("length_multiplier", 2.0))
        self.analytic_base_main_length_scale = float(analytic_base_cfg.get("main_length_scale", 1.0))
        self.analytic_base_secondary_length_scale = float(analytic_base_cfg.get("secondary_length_scale", 1.0))
        self.analytic_base_secondary_length_scale_gradient = float(analytic_base_cfg.get("secondary_length_scale_gradient", 0.0))
        self.analytic_base_secondary_length_scale_center = float(analytic_base_cfg.get("secondary_length_scale_center", 0.5))
        self.analytic_base_secondary_length_scale_min_factor = float(analytic_base_cfg.get("secondary_length_scale_min_factor", 0.05))
        self.analytic_base_min_length_m = float(analytic_base_cfg.get("min_length_m", 1.0e-6))
        self.analytic_base_smooth_max_tau = float(analytic_base_cfg.get("smooth_max_tau", 0.05))
        self.analytic_base_endpoint_taper_enabled = bool(analytic_base_cfg.get("endpoint_taper_enabled", False))
        self.analytic_base_endpoint_taper_m = float(analytic_base_cfg.get("endpoint_taper_m", 1.0))
        self.analytic_base_endpoint_taper_power = float(analytic_base_cfg.get("endpoint_taper_power", 2.0))
        self.analytic_base_aggregation = str(analytic_base_cfg.get("aggregation", "smooth_max")).lower()
        if self.analytic_base_aggregation not in {"smooth_max", "probabilistic_union", "sum_clamp"}:
            raise ValueError("model.analytic_base.aggregation must be 'smooth_max', 'probabilistic_union', or 'sum_clamp'.")
        self.analytic_base_matrix_length_mode = str(analytic_base_cfg.get("matrix_length_mode", "region")).lower()
        if self.analytic_base_matrix_length_mode not in {"region", "smooth_srv_usrv", "srv_halo"}:
            raise ValueError("model.analytic_base.matrix_length_mode must be 'region', 'smooth_srv_usrv', or 'srv_halo'.")
        self.analytic_base_srv_usrv_blend_width_m = float(analytic_base_cfg.get("srv_usrv_blend_width_m", 1.0))
        self.__dict__["_base_model_ref"] = None

        activation = str(model_cfg["activation"])
        output_dim = int(model_cfg["output_dim"])
        default_layers = int(model_cfg["hidden_layers"])
        default_units = int(model_cfg["hidden_units"])
        region_layers = model_cfg.get("region_hidden_layers", {})
        region_units = model_cfg.get("region_hidden_units", {})
        self.subnets = nn.ModuleDict(
            {
                region: MLP(
                    self.subnet_input_dim,
                    output_dim,
                    int(region_layers.get(region, default_layers)),
                    int(region_units.get(region, default_units)),
                    activation,
                )
                for region in ["HF", "SRV", "USRV"]
            }
        )
        if self.local_tip_expert_enabled:
            self.tip_expert = MLP(
                6,
                output_dim,
                int(local_tip_cfg.get("hidden_layers", 3)),
                int(local_tip_cfg.get("hidden_units", 48)),
                activation,
            )
        else:
            self.tip_expert = None
        if self.local_fracture_expert_enabled:
            self.fracture_expert = MLP(
                9,
                output_dim,
                int(local_frac_cfg.get("hidden_layers", 3)),
                int(local_frac_cfg.get("hidden_units", 64)),
                activation,
            )
        else:
            self.fracture_expert = None
        self._tip_points_cache: torch.Tensor | None = None
        self._line_tensor_cache: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    def _network_region(self, region_name: str) -> str:
        region = region_name.upper()
        if self.share_srv_usrv_subnet and region == "USRV":
            return "SRV"
        return region

    def attach_base_model(self, base_model: nn.Module | None) -> None:
        if base_model is None:
            self.__dict__["_base_model_ref"] = None
            return
        base_model.eval()
        for param in base_model.parameters():
            param.requires_grad_(False)
        self.__dict__["_base_model_ref"] = base_model

    def normalize_xyt(self, xyt: torch.Tensor) -> torch.Tensor:
        if xyt.ndim != 2 or xyt.shape[1] != 3:
            raise ValueError(f"xyt must have shape [N,3], got {tuple(xyt.shape)}.")
        geom = self.config["geometry"]
        sampler = self.config["sampler"]
        x_hat = (xyt[:, 0:1] - float(geom["x_min"])) / (float(geom["x_max"]) - float(geom["x_min"]))
        y_hat = (xyt[:, 1:2] - float(geom["y_min"])) / (float(geom["y_max"]) - float(geom["y_min"]))
        t_hat = (xyt[:, 2:3] - float(sampler["t_min"])) / (float(sampler["t_max"]) - float(sampler["t_min"]))
        return torch.cat([x_hat, y_hat, t_hat], dim=1)

    def denormalize_z(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.ndim != 2 or z.shape[1] != 3:
            raise ValueError(f"z must have shape [N,3], got {tuple(z.shape)}.")
        geom = self.config["geometry"]
        sampler = self.config["sampler"]
        x = float(geom["x_min"]) + z[:, 0:1] * (float(geom["x_max"]) - float(geom["x_min"]))
        y = float(geom["y_min"]) + z[:, 1:2] * (float(geom["y_max"]) - float(geom["y_min"]))
        t = float(sampler["t_min"]) + z[:, 2:3] * (float(sampler["t_max"]) - float(sampler["t_min"]))
        return x, y, t

    @staticmethod
    def _inside_rect(x: torch.Tensor, y: torch.Tensor, rect: Rect) -> torch.Tensor:
        return (x >= rect.x_min) & (x <= rect.x_max) & (y >= rect.y_min) & (y <= rect.y_max)

    @staticmethod
    def _local_xy(x: torch.Tensor, y: torch.Tensor, rect: Rect) -> tuple[torch.Tensor, torch.Tensor]:
        width = max(float(rect.width), 1.0e-12)
        height = max(float(rect.height), 1.0e-12)
        return (x - rect.x_min) / width, (y - rect.y_min) / height

    def _hf_local_xy(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_local = torch.zeros_like(x)
        y_local = torch.zeros_like(y)
        assigned = torch.zeros_like(x, dtype=torch.bool)
        for rect in self.geometry.hf_rects:
            mask = self._inside_rect(x, y, rect) & (~assigned)
            rx, ry = self._local_xy(x, y, rect)
            if rect.width >= rect.height:
                # Thin horizontal fracture: allow variation along length only.
                ry = torch.full_like(ry, 0.5)
            else:
                # Thin vertical fracture: allow variation along length only.
                rx = torch.full_like(rx, 0.5)
            x_local = torch.where(mask, rx, x_local)
            y_local = torch.where(mask, ry, y_local)
            assigned = assigned | mask
        fallback_x, fallback_y = self._local_xy(x, y, self.geometry.srv_bg)
        x_local = torch.where(assigned, x_local, fallback_x)
        y_local = torch.where(assigned, y_local, fallback_y)
        return x_local, y_local

    def features_for_region(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        if self.subnet_input_dim == 3:
            return z

        x, y, _t = self.denormalize_z(z)
        region = region_name.upper()
        if region == "HF":
            x_local, y_local = self._hf_local_xy(x, y)
        elif region == "SRV":
            x_local, y_local = self._local_xy(x, y, self.geometry.srv_bg)
        elif region == "USRV":
            x_local, y_local = self._local_xy(x, y, self.geometry.domain)
        else:
            raise ValueError(f"Unknown region: {region_name}")
        return torch.cat([x_local, y_local, z[:, 0:1], z[:, 1:2], z[:, 2:3]], dim=1)

    def _time_envelope(self, z: torch.Tensor) -> torch.Tensor:
        _x, _y, t = self.denormalize_z(z)
        decay = float(self.config["boundary"]["decay_rate"])
        return 1.0 - torch.exp(-decay * t)

    def _base_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.base_time_lag_days <= 0.0:
            return z
        x, y, t = self.denormalize_z(z)
        sampler = self.config["sampler"]
        t_min = float(sampler["t_min"])
        t_base = torch.clamp(t - float(self.base_time_lag_days), min=t_min)
        xyt_base = torch.cat([x, y, t_base], dim=1)
        return self.normalize_xyt(xyt_base)

    def _correction_envelope(self, z: torch.Tensor) -> torch.Tensor:
        beta = self._time_envelope(z)
        power = max(float(self.correction_envelope_power), 1.0e-12)
        return torch.clamp(beta, min=0.0) ** power

    def _base_field(self, z: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        base_model = self.__dict__.get("_base_model_ref")
        if base_model is None:
            if self.analytic_base_enabled:
                return self._analytic_diffusion_base(z, correction)
            return torch.ones_like(correction)
        z_base = self._base_z(z)
        base = base_model.forward_normalized(z_base)
        return base.to(device=correction.device, dtype=correction.dtype)

    def _diffusion_length(self, t: torch.Tensor, region_name: str) -> torch.Tensor:
        physics = self.config["physics"]
        region = region_name.upper()
        fai = float(physics["Fai"][region])
        diffusivity = float(physics["D"][region])
        k_eff = diffusivity / fai
        seconds = torch.clamp(t, min=0.0) * float(physics["seconds_per_day"])
        min_length = float(self.analytic_base_min_length_m)
        length = self.analytic_base_length_multiplier * torch.sqrt(torch.clamp(k_eff * seconds, min=0.0) + min_length * min_length)
        return torch.clamp(length, min=min_length)

    @staticmethod
    def _sigmoid_blend(value: torch.Tensor, width: float) -> torch.Tensor:
        return torch.sigmoid(value / max(float(width), 1.0e-12))

    def _srv_blend_weight(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = self.geometry.srv_bg
        width = max(float(self.analytic_base_srv_usrv_blend_width_m), 1.0e-12)
        w_left = self._sigmoid_blend(x - float(s.x_min), width)
        w_bottom = self._sigmoid_blend(y - float(s.y_min), width)
        w_top = self._sigmoid_blend(float(s.y_max) - y, width)
        return torch.clamp(w_left * w_bottom * w_top, min=0.0, max=1.0)

    def _srv_halo_weight(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = self.geometry.srv_bg
        width = max(float(self.analytic_base_srv_usrv_blend_width_m), 1.0e-12)
        dx_left = torch.relu(float(s.x_min) - x)
        dx_right = torch.relu(x - float(s.x_max))
        dy_bottom = torch.relu(float(s.y_min) - y)
        dy_top = torch.relu(y - float(s.y_max))
        outside_distance = torch.sqrt(dx_left**2 + dx_right**2 + dy_bottom**2 + dy_top**2)
        return torch.exp(-outside_distance / width)

    def _matrix_diffusion_length(self, x: torch.Tensor, y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        length_srv = self._diffusion_length(t, "SRV")
        length_usrv = self._diffusion_length(t, "USRV")
        if self.analytic_base_matrix_length_mode != "smooth_srv_usrv":
            if self.analytic_base_matrix_length_mode == "srv_halo":
                weight_srv = self._srv_halo_weight(x, y)
                return weight_srv * length_srv + (1.0 - weight_srv) * length_usrv
            region_id = self.geometry.region_id_torch(x, y)
            return torch.where(region_id == REGION_SRV, length_srv, length_usrv)
        weight_srv = self._srv_blend_weight(x, y)
        return weight_srv * length_srv + (1.0 - weight_srv) * length_usrv

    def _analytic_diffusion_base(self, z: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        x, y, t = self.denormalize_z(z)
        target = dirichlet_target_hat(t, self.config["boundary"]).to(device=z.device, dtype=z.dtype)
        total_drawdown = torch.clamp(1.0 - target, min=0.0, max=1.0)
        length_hf = self._diffusion_length(t, "HF")
        length_matrix = self._matrix_diffusion_length(x, y, t)
        xw, yw = self.geometry.dirichlet_point()
        main_y = 0.5 * (self.geometry.main_frac.y_min + self.geometry.main_frac.y_max)
        drawdown_candidates = [torch.zeros_like(correction)]
        for line in self.geometry.hf_lines:
            if abs(line.x1 - line.x0) >= abs(line.y1 - line.y0):
                x_proj = torch.clamp(x, min=min(line.x0, line.x1), max=max(line.x0, line.x1))
                y_proj = torch.full_like(y, float(line.y0))
                along = torch.abs(float(xw) - x_proj)
                line_length = length_hf * float(self.analytic_base_main_length_scale)
            else:
                x_proj = torch.full_like(x, float(line.x0))
                y_proj = torch.clamp(y, min=min(line.y0, line.y1), max=max(line.y0, line.y1))
                along = torch.abs(float(xw) - x_proj) + torch.abs(y_proj - float(main_y))
                main_span = max(abs(float(xw) - float(self.geometry.main_frac.x_min)), 1.0e-12)
                distance_norm = abs(float(xw) - float(line.x0)) / main_span
                scale_factor = 1.0 + float(self.analytic_base_secondary_length_scale_gradient) * (
                    distance_norm - float(self.analytic_base_secondary_length_scale_center)
                )
                scale_factor = max(float(self.analytic_base_secondary_length_scale_min_factor), scale_factor)
                line_length = length_hf * float(self.analytic_base_secondary_length_scale) * scale_factor
            normal = torch.sqrt(torch.clamp((x - x_proj) ** 2 + (y - y_proj) ** 2, min=0.0) + float(self.analytic_base_min_length_m) ** 2)
            fracture_factor = torch.erfc(along / torch.clamp(line_length, min=float(self.analytic_base_min_length_m)))
            matrix_factor = torch.erfc(normal / length_matrix)
            endpoint_taper = self._finite_line_endpoint_taper(x, y, line)
            candidate = total_drawdown * fracture_factor * matrix_factor * endpoint_taper
            drawdown_candidates.append(candidate)
        candidates = torch.stack(drawdown_candidates, dim=0)
        if self.analytic_base_aggregation == "probabilistic_union":
            smooth_drawdown = 1.0 - torch.prod(torch.clamp(1.0 - candidates, min=0.0, max=1.0), dim=0)
        elif self.analytic_base_aggregation == "sum_clamp":
            smooth_drawdown = torch.clamp(torch.sum(candidates, dim=0), min=0.0, max=1.0)
        else:
            tau = max(float(self.analytic_base_smooth_max_tau), 1.0e-12)
            weights = torch.softmax(candidates / tau, dim=0)
            smooth_drawdown = torch.sum(weights * candidates, dim=0)
        smooth_drawdown = torch.minimum(smooth_drawdown, total_drawdown)
        return torch.clamp(1.0 - smooth_drawdown, min=0.0, max=1.0).to(device=correction.device, dtype=correction.dtype)

    def _finite_line_endpoint_taper(self, x: torch.Tensor, y: torch.Tensor, line: Any) -> torch.Tensor:
        if not self.analytic_base_endpoint_taper_enabled:
            return torch.ones_like(x)
        if abs(line.x1 - line.x0) >= abs(line.y1 - line.y0):
            low = min(float(line.x0), float(line.x1))
            high = max(float(line.x0), float(line.x1))
            overrun = torch.relu(low - x) + torch.relu(x - high)
        else:
            low = min(float(line.y0), float(line.y1))
            high = max(float(line.y0), float(line.y1))
            overrun = torch.relu(low - y) + torch.relu(y - high)
        length = max(float(self.analytic_base_endpoint_taper_m), 1.0e-12)
        power = max(float(self.analytic_base_endpoint_taper_power), 1.0e-12)
        return torch.exp(-torch.clamp(overrun / length, min=0.0) ** power)

    def _apply_initial_condition(self, raw: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        beta = self._time_envelope(z)
        return 1.0 + beta * raw

    def _apply_base_correction(self, correction: torch.Tensor, z: torch.Tensor, region_name: str | None = None) -> torch.Tensor:
        base = self._base_field(z, correction)
        envelope = self._correction_envelope(z)
        local = self._local_tip_correction(z, correction) + self._local_fracture_correction(z, correction, region_name)
        return base + envelope * (correction + local)

    def _correction_scale_for_region(self, region_name: str | None) -> float:
        if region_name is None:
            return float(self.correction_scale)
        return float(self.correction_scale_by_region.get(region_name.upper(), self.correction_scale))

    def _bounded_correction(self, raw: torch.Tensor, region_name: str | None = None) -> torch.Tensor:
        if self.correction_activation == "softsign":
            bounded = raw / (1.0 + torch.abs(raw))
        else:
            bounded = torch.tanh(raw)
        return self._correction_scale_for_region(region_name) * bounded

    def _tip_points(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        cached = self._tip_points_cache
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached
        points: list[tuple[float, float]] = []
        main = self.geometry.hf_lines[0]
        points.append((float(main.x0), float(main.y0)))
        for line in self.geometry.hf_lines[1:]:
            points.append((float(line.x0), float(line.y0)))
            points.append((float(line.x1), float(line.y1)))
        tensor = torch.as_tensor(points, dtype=dtype, device=device)
        self._tip_points_cache = tensor
        return tensor

    def _line_tensors(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cached = self._line_tensor_cache
        if cached is not None and cached[0].device == device and cached[0].dtype == dtype:
            return cached
        p0: list[tuple[float, float]] = []
        tangent: list[tuple[float, float]] = []
        length: list[float] = []
        for line in self.geometry.hf_lines:
            tx, ty = line.tangent
            p0.append((float(line.x0), float(line.y0)))
            tangent.append((float(tx), float(ty)))
            length.append(max(float(line.length), 1.0e-12))
        p0_tensor = torch.as_tensor(p0, dtype=dtype, device=device)
        tangent_tensor = torch.as_tensor(tangent, dtype=dtype, device=device)
        length_tensor = torch.as_tensor(length, dtype=dtype, device=device).view(1, -1)
        self._line_tensor_cache = (p0_tensor, tangent_tensor, length_tensor)
        return self._line_tensor_cache

    def _local_tip_correction(self, z: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if not self.local_tip_expert_enabled or self.tip_expert is None or self.local_tip_expert_scale <= 0.0:
            return torch.zeros_like(reference)
        radius = max(float(self.local_tip_expert_radius_m), 1.0e-12)
        x, y, _t = self.denormalize_z(z)
        tips = self._tip_points(z.device, z.dtype)
        dx = x - tips[:, 0].view(1, -1)
        dy = y - tips[:, 1].view(1, -1)
        dist2 = dx * dx + dy * dy
        kernel = torch.exp(-0.5 * dist2 / (radius * radius))
        weight_sum = torch.clamp(torch.sum(kernel, dim=1, keepdim=True), min=1.0e-12)
        weights = kernel / weight_sum
        dx_local = torch.sum(weights * dx, dim=1, keepdim=True) / radius
        dy_local = torch.sum(weights * dy, dim=1, keepdim=True) / radius
        gate = torch.clamp(weight_sum, min=0.0, max=1.0)
        power = max(float(self.local_tip_expert_gate_power), 1.0e-12)
        gate = gate**power
        features = torch.cat([z, dx_local, dy_local, gate], dim=1)
        raw = self.tip_expert(features)
        return float(self.local_tip_expert_scale) * gate * torch.tanh(raw)

    def _local_fracture_correction(self, z: torch.Tensor, reference: torch.Tensor, region_name: str | None) -> torch.Tensor:
        if not self.local_fracture_expert_enabled or self.fracture_expert is None or self.local_fracture_expert_scale <= 0.0:
            return torch.zeros_like(reference)
        if self.local_fracture_expert_matrix_only and region_name is not None and region_name.upper() == "HF":
            return torch.zeros_like(reference)
        radius = max(float(self.local_fracture_expert_radius_m), 1.0e-12)
        x, y, _t = self.denormalize_z(z)
        point = torch.cat([x, y], dim=1)
        p0, tangent, length = self._line_tensors(z.device, z.dtype)
        rel = point[:, None, :] - p0[None, :, :]
        s = torch.sum(rel * tangent[None, :, :], dim=2)
        s_clamped = torch.minimum(torch.clamp(s, min=0.0), length)
        closest = p0[None, :, :] + s_clamped[:, :, None] * tangent[None, :, :]
        delta = point[:, None, :] - closest
        dist2 = torch.sum(delta * delta, dim=2)
        kernel = torch.exp(-0.5 * dist2 / (radius * radius))
        weight_sum = torch.clamp(torch.sum(kernel, dim=1, keepdim=True), min=1.0e-12)
        weights = kernel / weight_sum
        dx_local = torch.sum(weights * delta[:, :, 0], dim=1, keepdim=True) / radius
        dy_local = torch.sum(weights * delta[:, :, 1], dim=1, keepdim=True) / radius
        tau_local = torch.sum(weights * (s_clamped / length), dim=1, keepdim=True)
        tx_local = torch.sum(weights * tangent[None, :, 0], dim=1, keepdim=True)
        ty_local = torch.sum(weights * tangent[None, :, 1], dim=1, keepdim=True)
        gate = torch.clamp(weight_sum, min=0.0, max=1.0)
        power = max(float(self.local_fracture_expert_gate_power), 1.0e-12)
        gate = gate**power
        features = torch.cat([z, dx_local, dy_local, tau_local, tx_local, ty_local, gate], dim=1)
        raw = self.fracture_expert(features)
        return float(self.local_fracture_expert_scale) * gate * torch.tanh(raw)

    def _apply_hard_dirichlet(self, value: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.hard_dirichlet_enabled:
            return value
        x, y, t = self.denormalize_z(z)
        xw, yw = self.geometry.dirichlet_point()
        distance = torch.sqrt((x - float(xw)) ** 2 + (y - float(yw)) ** 2)
        radius = max(float(self.hard_dirichlet_radius_m), 1.0e-12)
        power = max(float(self.hard_dirichlet_power), 1.0e-12)
        scaled = torch.clamp(distance / radius, min=0.0) ** power
        blend = scaled / (1.0 + scaled)
        target = dirichlet_target_hat(t, self.config["boundary"]).to(device=value.device, dtype=value.dtype)
        return target + blend * (value - target)

    def _apply_pressure_bounds(self, value: torch.Tensor) -> torch.Tensor:
        if not self.enforce_maximum_principle:
            return value
        return torch.clamp(value, min=0.0, max=1.0)

    def forward_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        region = region_name.upper()
        network_region = self._network_region(region)
        raw = self.subnets[network_region](self.features_for_region(z, network_region))
        if self.constraint_mode == "ic_hard":
            value = self._apply_initial_condition(raw, z)
        else:
            value = self._apply_base_correction(self._bounded_correction(raw, network_region), z, region)
        value = self._apply_hard_dirichlet(value, z)
        return self._apply_pressure_bounds(value)

    def forward_correction_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        region = region_name.upper()
        network_region = self._network_region(region)
        raw = self.subnets[network_region](self.features_for_region(z, network_region))
        correction = self._bounded_correction(raw, network_region)
        if self.constraint_mode == "ic_hard":
            return raw * 0.0
        local = self._local_tip_correction(z, correction) + self._local_fracture_correction(z, correction, region)
        return self._correction_envelope(z) * (correction + local)

    def forward_raw_region_normalized(self, z: torch.Tensor, region_name: str) -> torch.Tensor:
        region = region_name.upper()
        network_region = self._network_region(region)
        return self.subnets[network_region](self.features_for_region(z, network_region))

    def forward_correction_normalized(self, z: torch.Tensor) -> torch.Tensor:
        x, y, _t = self.denormalize_z(z)
        region_id = self.geometry.region_id_torch(x, y)
        outputs = torch.zeros((z.shape[0], int(self.config["model"]["output_dim"])), dtype=z.dtype, device=z.device)
        for region_id_value, region_name in self.REGION_KEY_BY_ID.items():
            mask = (region_id == int(region_id_value)).view(-1)
            if torch.any(mask):
                outputs[mask] = self.forward_correction_region_normalized(z[mask], region_name)
        outside = (region_id < 0).view(-1)
        if torch.any(outside):
            outputs[outside] = self.forward_correction_region_normalized(z[outside], "USRV")
        return outputs

    def correction(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.forward_correction_normalized(self.normalize_xyt(xyt))

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        x, y, _t = self.denormalize_z(z)
        region_id = self.geometry.region_id_torch(x, y)
        outputs = torch.zeros((z.shape[0], int(self.config["model"]["output_dim"])), dtype=z.dtype, device=z.device)
        for region_id_value, region_name in self.REGION_KEY_BY_ID.items():
            mask = (region_id == int(region_id_value)).view(-1)
            if torch.any(mask):
                outputs[mask] = self.forward_region_normalized(z[mask], region_name)
        outside = (region_id < 0).view(-1)
        if torch.any(outside):
            outputs[outside] = self.forward_region_normalized(z[outside], "USRV")
        return outputs

    def forward_raw_normalized(self, z: torch.Tensor) -> torch.Tensor:
        x, y, _t = self.denormalize_z(z)
        region_id = self.geometry.region_id_torch(x, y)
        outputs = torch.zeros((z.shape[0], int(self.config["model"]["output_dim"])), dtype=z.dtype, device=z.device)
        for region_id_value, region_name in self.REGION_KEY_BY_ID.items():
            mask = (region_id == int(region_id_value)).view(-1)
            if torch.any(mask):
                outputs[mask] = self.forward_raw_region_normalized(z[mask], region_name)
        outside = (region_id < 0).view(-1)
        if torch.any(outside):
            outputs[outside] = self.forward_raw_region_normalized(z[outside], "USRV")
        return outputs

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        return self.forward_normalized(self.normalize_xyt(xyt))
