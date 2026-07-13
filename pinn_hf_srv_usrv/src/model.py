"""PINN 网络模型与可切换约束形式。

`model.constraint_mode` 支持三种形式：
1. dirichlet_hard：`u = u_ref(x,y,t) + B_D(x,y) N_theta`，生产边界压力精确满足；
2. direct：`u = N_theta`，网络直接学习无量纲压力场，所有条件都通过损失约束；
3. ic_hard：`u = u0(x,y) + t_hat N_theta`，初始条件精确满足，生产边界通过损失约束。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .geometry import ReservoirGeometry
from .utils import (
    boundary_decay_rate,
    boundary_p_out_mpa,
    boundary_p_t0_mpa,
    boundary_ratio,
    initial_pressure_pair_mpa,
    pressure_mpa_to_hat,
)


class MLP(nn.Module):
    """基础多层感知机。"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_units: int,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        if activation.lower() == "tanh":
            act_factory = nn.Tanh
        elif activation.lower() == "silu":
            act_factory = nn.SiLU
        elif activation.lower() == "relu":
            act_factory = nn.ReLU
        else:
            raise ValueError(f"不支持的激活函数: {activation}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(act_factory())
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """使用 Xavier 初始化，适合 tanh 网络的平滑函数逼近。"""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""

        return self.net(x)


class PINNModel(nn.Module):
    """带几何嵌入和可切换约束形式的 PINN 模型。"""

    def __init__(self, geometry: ReservoirGeometry, config: dict[str, Any]) -> None:
        super().__init__()
        self.geometry = geometry
        self.config = config
        model_cfg = config["model"]
        self.mlp = MLP(
            input_dim=int(model_cfg["input_dim"]),
            output_dim=int(model_cfg["output_dim"]),
            hidden_layers=int(model_cfg["hidden_layers"]),
            hidden_units=int(model_cfg["hidden_units"]),
            activation=str(model_cfg["activation"]),
        )

    def constraint_mode(self) -> str:
        """返回当前模型约束模式，并兼容旧配置。"""

        mode = str(self.config.get("model", {}).get("constraint_mode", "dirichlet_hard")).lower()
        aliases = {
            "dirichlet": "dirichlet_hard",
            "hard": "dirichlet_hard",
            "hard_dirichlet": "dirichlet_hard",
            "none": "direct",
            "raw": "direct",
            "initial_hard": "ic_hard",
            "hard_initial": "ic_hard",
        }
        mode = aliases.get(mode, mode)
        if mode not in {"dirichlet_hard", "direct", "ic_hard"}:
            raise ValueError(
                "Unsupported constraint_mode: "
                f"{mode}. Use 'dirichlet_hard', 'direct', or 'ic_hard'."
            )
        return mode

    def build_features(self, xyt: torch.Tensor, detach_distance_features: bool = False) -> torch.Tensor:
        """构造网络输入特征。

        输入 `xyt` 使用物理坐标。这里转换成归一化坐标，并拼接区域 one-hot 与距离特征。
        使用归一化坐标能显著降低不同变量量纲差异，例如 x 为百米量级而 t 为千天量级。
        """

        geom_cfg = self.config["geometry"]
        sampler_cfg = self.config["sampler"]

        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]

        x_hat = (x - float(geom_cfg["x_min"])) / (float(geom_cfg["x_max"]) - float(geom_cfg["x_min"]))
        y_hat = (y - float(geom_cfg["y_min"])) / (float(geom_cfg["y_max"]) - float(geom_cfg["y_min"]))
        t_hat = (t - float(sampler_cfg["t_min"])) / (float(sampler_cfg["t_max"]) - float(sampler_cfg["t_min"]))

        region_onehot = self.geometry.region_onehot_torch(x, y)
        distance_features = self.geometry.build_distance_features_torch(x, y)
        if detach_distance_features:
            distance_features = distance_features.detach()
        return torch.cat([x_hat, y_hat, t_hat, region_onehot, distance_features], dim=1)

    def boundary_value_hat(self, t: torch.Tensor) -> torch.Tensor:
        """计算 Dirichlet 边界压力的无量纲值。

        COMSOL 中的边界压力函数在第一版作为给定函数使用。P12/P13 按组分比例拆分，
        再统一调用 `pressure_mpa_to_hat()` 转换为网络使用的无量纲压力。这样模型、
        数据监督和评价后处理使用同一套分量仿射尺度，不会出现训练和预测尺度不一致。
        """

        if t.ndim == 1:
            t = t.view(-1, 1)
        boundary_cfg = self.config["boundary"]
        pt0 = boundary_p_t0_mpa(boundary_cfg)
        ratio = boundary_ratio(boundary_cfg)
        decay = boundary_decay_rate(boundary_cfg)

        pout = boundary_p_out_mpa(boundary_cfg)
        an1 = (pt0 - pout) * torch.exp(-decay * t) + pout
        g12 = an1 / (1.0 + ratio)
        g13 = an1 * ratio / (1.0 + ratio)
        u12_b, u13_b = pressure_mpa_to_hat(g12, g13, boundary_cfg)
        return torch.cat([u12_b, u13_b], dim=1)

    def initial_value_hat(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """按 HF/SRV/USRV 分区返回初始无量纲压力。

        远离生产边界时，参考场应接近储层初始高压，而不是接近井底低压。这里显式
        使用分区初始条件，为后续不同区域初始压力不同的情况预留接口。
        """

        boundary_cfg = self.config["boundary"]

        def pair(region: str) -> tuple[float, float]:
            p12, p13 = initial_pressure_pair_mpa(self.config, region)
            u12, u13 = pressure_mpa_to_hat(p12, p13, boundary_cfg)
            return float(u12), float(u13)

        hf12, hf13 = pair("hf")
        srv12, srv13 = pair("srv")
        usrv12, usrv13 = pair("usrv")
        onehot = self.geometry.region_onehot_torch(x, y)
        u12 = onehot[:, 0:1] * hf12 + onehot[:, 1:2] * srv12 + onehot[:, 2:3] * usrv12
        u13 = onehot[:, 0:1] * hf13 + onehot[:, 1:2] * srv13 + onehot[:, 2:3] * usrv13
        return torch.cat([u12, u13], dim=1)

    def dirichlet_influence_weight(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """兼容保留：计算生产边界压力对旧参考场的局部影响权重。

        当 hard_constraint_reference="local_initial" 时，`forward()` 会调用该函数，
        使生产边界低压只在边界附近作为参考场，远场则回到初始高压。
        """

        boundary_cfg = self.config["boundary"]
        length_m = float(boundary_cfg.get("dirichlet_influence_length_m", 20.0))
        decay_power = float(boundary_cfg.get("dirichlet_influence_power", 2.0))
        d_hat = self.geometry.distance_to_dirichlet_torch(x, y)
        d_m = d_hat * self.geometry.l_ref
        scaled = torch.clamp(d_m / max(length_m, 1.0e-12), min=0.0)
        return torch.exp(-(scaled**decay_power))

    def reference_value_hat(self, xyt: torch.Tensor) -> torch.Tensor:
        """构造局部参考场 `u_ref(x,y,t)`。

        参考场在生产边界上等于 `g_D(t)`，远离生产边界时逐渐过渡到初始压力。
        这样仍然能通过 `u=u_ref+B_D*N_theta` 精确满足 Dirichlet 边界，
        同时避免把整个计算域的基底压力都拉到生产边界低压。
        """

        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        boundary = self.boundary_value_hat(t)
        initial = self.initial_value_hat(x, y)
        weight = self.dirichlet_influence_weight(x, y)
        return weight * boundary + (1.0 - weight) * initial

    def hard_constraint_components(
        self,
        xyt: torch.Tensor,
        raw: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """分解 Dirichlet hard constraint 的参考场、原始网络输出和修正项。

        诊断脚本和训练/评价日志都会调用该函数，避免在多个文件里重复实现
        `u_ref + B_D * N_theta`。返回张量均保留 batch 维度，其中 `b_d` 为 `[N,1]`，
        其余二维压力相关量为 `[N,2]`。
        """

        if xyt.ndim != 2 or xyt.shape[1] != 3:
            raise ValueError(f"hard_constraint_components 期望 xyt 形状为 [N, 3]，当前为 {tuple(xyt.shape)}。")
        if raw is None:
            raw = self.forward_raw(xyt)
        if raw.ndim != 2 or raw.shape[1] != 2 or raw.shape[0] != xyt.shape[0]:
            raise ValueError(f"raw 期望形状为 [N, 2] 且与 xyt 行数一致，当前为 {tuple(raw.shape)}。")
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        reference_mode = str(self.config.get("model", {}).get("hard_constraint_reference", "local_initial")).lower()
        if reference_mode in {"local_initial", "localized", "initial"}:
            reference = self.reference_value_hat(xyt)
        elif reference_mode in {"boundary", "global_boundary", "simple"}:
            reference = self.boundary_value_hat(xyt[:, 2:3])
        else:
            raise ValueError(
                "Unsupported hard_constraint_reference: "
                f"{reference_mode}. Use 'local_initial' or 'boundary'."
            )
        b_d = self.geometry.adf_dirichlet_torch(x, y)
        correction = b_d * raw
        return {
            "b_d": b_d,
            "reference": reference,
            "raw": raw,
            "correction": correction,
            "final": reference + correction,
        }

    def forward_raw(self, xyt: torch.Tensor, detach_distance_features: bool = False) -> torch.Tensor:
        """返回未施加硬约束的网络扰动项 N12,N13。"""

        return self.mlp(self.build_features(xyt, detach_distance_features=detach_distance_features))

    def forward(self, xyt: torch.Tensor, detach_distance_features: bool = False) -> torch.Tensor:
        """返回当前约束模式下的无量纲压力 u12,u13。"""

        raw = self.forward_raw(xyt, detach_distance_features=detach_distance_features)
        mode = self.constraint_mode()
        if mode == "direct":
            return raw

        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        if mode == "ic_hard":
            sampler_cfg = self.config["sampler"]
            t = xyt[:, 2:3]
            t_hat = (t - float(sampler_cfg["t_min"])) / (float(sampler_cfg["t_max"]) - float(sampler_cfg["t_min"]))
            return self.initial_value_hat(x, y) + t_hat * raw

        components = self.hard_constraint_components(xyt, raw=raw)
        return components["final"]
