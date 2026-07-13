"""仅包含 ic_hard 约束的 MLP PINN 模型。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class MLP(nn.Module):
    """基础全连接网络，使用 Xavier 初始化。"""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: int, hidden_units: int, activation: str) -> None:
        super().__init__()
        if activation.lower() == "tanh":
            act_factory = nn.Tanh
        elif activation.lower() == "silu":
            act_factory = nn.SiLU
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
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
        """Xavier 初始化适合 tanh 网络逼近平滑扩散解。"""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class PINNModel(nn.Module):
    """输入仅为 x_hat/y_hat/t_hat 的简化 PINN。"""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config["model"]
        if int(model_cfg["input_dim"]) != 3:
            raise ValueError("v2 网络输入只能是 3 维归一化坐标。")
        if str(model_cfg["constraint_mode"]).lower() != "ic_hard":
            raise ValueError("v2 只实现 ic_hard 约束。")
        self.config = config
        self.mlp = MLP(
            int(model_cfg["input_dim"]),
            int(model_cfg["output_dim"]),
            int(model_cfg["hidden_layers"]),
            int(model_cfg["hidden_units"]),
            str(model_cfg["activation"]),
        )

    def normalize_xyt(self, xyt: torch.Tensor) -> torch.Tensor:
        """把物理坐标转换为网络唯一输入 z=[x_hat,y_hat,t_hat]。"""

        if xyt.ndim != 2 or xyt.shape[1] != 3:
            raise ValueError(f"xyt 期望形状 [N,3]，当前 {tuple(xyt.shape)}。")
        geom = self.config["geometry"]
        sampler = self.config["sampler"]
        x_hat = (xyt[:, 0:1] - float(geom["x_min"])) / (float(geom["x_max"]) - float(geom["x_min"]))
        y_hat = (xyt[:, 1:2] - float(geom["y_min"])) / (float(geom["y_max"]) - float(geom["y_min"]))
        t_hat = (xyt[:, 2:3] - float(sampler["t_min"])) / (float(sampler["t_max"]) - float(sampler["t_min"]))
        return torch.cat([x_hat, y_hat, t_hat], dim=1)

    def forward_normalized(self, z: torch.Tensor) -> torch.Tensor:
        """对归一化坐标前向，并施加初始条件硬约束。

        原始 MLP 输出 N12/N13，最终 `u=1+t_hat*N`，因此 t_hat=0 时两个变量严格等于 1。
        这个结构替代了初始条件采样点和初始条件 loss。
        """

        if z.ndim != 2 or z.shape[1] != 3:
            raise ValueError(f"归一化输入 z 期望 [N,3]，当前 {tuple(z.shape)}。")
        raw = self.mlp(z)
        t_hat = z[:, 2:3]
        return 1.0 + t_hat * raw

    def forward_raw_normalized(self, z: torch.Tensor) -> torch.Tensor:
        """返回未施加 ic_hard 的原始 MLP 输出，主要用于测试和诊断。"""

        return self.mlp(z)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        """物理坐标前向预测无量纲 u12/u13。"""

        return self.forward_normalized(self.normalize_xyt(xyt))
