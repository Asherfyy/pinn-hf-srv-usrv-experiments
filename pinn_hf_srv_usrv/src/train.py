"""CPU 训练入口。

运行方式：
    python -m src.train --config config/default.yaml
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from .config import load_config, update_config_from_args
from .geometry import ReservoirGeometry
from .losses import compute_total_loss
from .model import PINNModel
from .sampler import ReservoirSampler
from .utils import ensure_output_dirs, force_cpu, get_torch_dtype, pressure_mpa_to_hat, save_loss_history, set_seed


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="训练 HF/SRV/USRV 分区 PINN。")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="YAML 配置文件路径。")
    parser.add_argument("--epochs", type=int, default=None, help="可选：覆盖配置中的训练轮数，便于冒烟测试。")
    return parser.parse_args()


def load_optional_data_batch(
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    """读取可选 COMSOL 快照数据作为监督项。

    默认 `loss_weights.data=0`，因此没有数据或空文件时不会影响基础训练。
    若用户把数据权重调大，并提供 x,y,t,Pg1,Pg2 列，则这里会自动转换为无量纲目标。
    """

    if float(config["loss_weights"].get("data", 0.0)) <= 0.0:
        return None
    path = Path(config["paths"]["data"]) / "comsol_snapshots.csv"
    if not path.exists() or path.stat().st_size == 0:
        return None
    df = pd.read_csv(path)
    required = {"x", "y", "t", "Pg1", "Pg2"}
    if not required.issubset(df.columns) or len(df) == 0:
        return None

    # Pandas 某些切片会返回只读 NumPy 视图；显式 copy 可避免 PyTorch 只读数组警告。
    xyt = torch.as_tensor(df[["x", "y", "t"]].to_numpy(copy=True), dtype=dtype, device=device)
    p12 = torch.as_tensor(df[["Pg1"]].to_numpy(copy=True), dtype=dtype, device=device)
    p13 = torch.as_tensor(df[["Pg2"]].to_numpy(copy=True), dtype=dtype, device=device)
    u12, u13 = pressure_mpa_to_hat(p12, p13, config["boundary"])
    target_hat = torch.cat([u12, u13], dim=1)
    return {"xyt": xyt, "target_hat": target_hat}


def save_checkpoint(
    model: torch.nn.Module,
    config: dict[str, Any],
    epoch: int,
    loss_value: float,
    path: str | Path,
) -> None:
    """保存模型 checkpoint。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "loss": loss_value,
        },
        path,
    )


def run_lbfgs_optional(
    model: torch.nn.Module,
    sampler: ReservoirSampler,
    geometry: ReservoirGeometry,
    config: dict[str, Any],
    data_batch: dict[str, torch.Tensor] | None,
) -> float | None:
    """可选 L-BFGS 精修接口。

    CPU 上 L-BFGS 可能很慢，因此默认关闭。保留该函数是为了后续研究者需要时无需重写
    训练入口；它只在配置显式开启时运行一次。
    """

    if not bool(config["training"].get("use_lbfgs", False)):
        return None
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=int(config["training"].get("lbfgs_max_iter", 200)),
        line_search_fn="strong_wolfe",
    )
    samples = sampler.sample_all()
    last_loss = 0.0

    def closure() -> torch.Tensor:
        nonlocal last_loss
        optimizer.zero_grad(set_to_none=True)
        loss, _loss_dict = compute_total_loss(model, samples, geometry, config["physics"], config, data_batch)
        if torch.isnan(loss).item():
            raise RuntimeError("损失出现 NaN，请检查采样、系数或学习率。")
        loss.backward()
        last_loss = float(loss.detach().cpu())
        return loss

    optimizer.step(closure)
    return last_loss


def _pde_component_loss_keys() -> list[str]:
    """返回六个归一化 PDE 分量 loss 在 loss_dict 中的固定键名。"""

    return [
        "loss_pde_u12_hf",
        "loss_pde_u13_hf",
        "loss_pde_u12_srv",
        "loss_pde_u13_srv",
        "loss_pde_u12_usrv",
        "loss_pde_u13_usrv",
    ]


def add_pde_component_gradient_norms(
    model: torch.nn.Module,
    loss_dict: dict[str, torch.Tensor],
    epoch: int,
    config: dict[str, Any],
) -> None:
    """低频计算六个 PDE 分量 loss 对所有可训练参数的梯度 L2 范数。

    该诊断只在配置指定的间隔执行，并使用 `torch.autograd.grad` 读取梯度，不写入
    `parameter.grad`，因此不会破坏随后正常的 `total_loss.backward()`。
    """

    diag_cfg = config.get("diagnostics", {})
    if not bool(diag_cfg.get("enabled", False)):
        return
    if not bool(diag_cfg.get("log_pde_component_gradient_norms", False)):
        return
    every = int(diag_cfg.get("gradient_norm_every", 100))
    if every <= 0:
        raise ValueError(f"diagnostics.gradient_norm_every 必须为正整数，当前为 {every}。")
    if epoch % every != 0:
        return

    parameters = [param for param in model.parameters() if param.requires_grad]
    if not parameters:
        return
    for loss_key in _pde_component_loss_keys():
        if loss_key not in loss_dict:
            continue
        grads = torch.autograd.grad(
            loss_dict[loss_key],
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        total_sq = loss_dict[loss_key].new_tensor(0.0)
        for grad_tensor in grads:
            if grad_tensor is None:
                continue
            total_sq = total_sq + torch.sum(grad_tensor.detach() ** 2)
        suffix = loss_key.replace("loss_pde_", "")
        loss_dict[f"gradnorm_pde_{suffix}"] = torch.sqrt(torch.clamp(total_sq, min=0.0))


def _hard_constraint_probe_points(
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    n_x: int = 128,
) -> torch.Tensor:
    """构造主裂缝中心线诊断点，用于检查 B_D 与 raw 输出幅值。"""

    geom_cfg = config["geometry"]
    times = [float(value) for value in config.get("evaluation", {}).get("times", [1.0, 10.0, 100.0, 1000.0])]
    x = torch.linspace(
        float(geom_cfg["main_frac_x_min"]),
        float(geom_cfg["main_frac_x_max"]),
        int(n_x),
        dtype=dtype,
        device=device,
    ).view(-1, 1)
    y_value = 0.5 * (float(geom_cfg["main_frac_y_min"]) + float(geom_cfg["main_frac_y_max"]))
    chunks = []
    for time_value in times:
        y = torch.full_like(x, y_value)
        t = torch.full_like(x, time_value)
        chunks.append(torch.cat([x, y, t], dim=1))
    return torch.cat(chunks, dim=0)


def print_hard_constraint_raw_summary(
    model: PINNModel,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
    label: str,
) -> None:
    """打印 hard constraint 中 raw 与 B_D*raw 的最大绝对值诊断。"""

    if model.constraint_mode() != "dirichlet_hard":
        return
    xyt = _hard_constraint_probe_points(config, device, dtype)
    with torch.no_grad():
        components = model.hard_constraint_components(xyt)
        raw = components["raw"]
        correction = components["correction"]
        metrics = {
            "max_abs_raw_u12": torch.max(torch.abs(raw[:, 0])).item(),
            "max_abs_raw_u13": torch.max(torch.abs(raw[:, 1])).item(),
            "max_abs_BD_raw_u12": torch.max(torch.abs(correction[:, 0])).item(),
            "max_abs_BD_raw_u13": torch.max(torch.abs(correction[:, 1])).item(),
        }
    print(
        f"{label} hard constraint 诊断: "
        f"max_abs_raw_u12={metrics['max_abs_raw_u12']:.4e}, "
        f"max_abs_raw_u13={metrics['max_abs_raw_u13']:.4e}, "
        f"max_abs_BD_raw_u12={metrics['max_abs_BD_raw_u12']:.4e}, "
        f"max_abs_BD_raw_u13={metrics['max_abs_BD_raw_u13']:.4e}",
        flush=True,
    )


def main() -> None:
    """训练主流程。"""

    args = parse_args()
    config = update_config_from_args(load_config(args.config), args.epochs)

    runtime_cfg = config["runtime"]
    device = force_cpu(int(runtime_cfg["cpu_threads"]))
    dtype = get_torch_dtype(str(runtime_cfg["dtype"]))
    set_seed(int(runtime_cfg["seed"]))
    ensure_output_dirs(config)

    geometry = ReservoirGeometry(config["geometry"], data_dir=config["paths"]["data"])
    sampler = ReservoirSampler(
        geometry,
        config["sampler"],
        device=device,
        dtype=dtype,
        seed=int(runtime_cfg["seed"]),
    )
    model = PINNModel(geometry, config).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    data_batch = load_optional_data_batch(config, device, dtype)
    print_hard_constraint_raw_summary(model, config, device, dtype, label="训练开始")

    epochs = int(config["training"]["epochs"])
    print_every = int(config["training"]["print_every"])
    save_every = int(config["training"]["save_every"])
    grad_clip_norm = float(config["training"]["grad_clip_norm"])
    history: list[dict[str, float]] = []

    for epoch in tqdm(range(1, epochs + 1), desc="Training", ncols=100):
        # 每个 epoch 重新采样，相当于用随机配点不断覆盖复杂几何中的不同区域。
        samples = sampler.sample_all()

        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = compute_total_loss(model, samples, geometry, config["physics"], config, data_batch)
        if torch.isnan(loss).item():
            raise RuntimeError("损失出现 NaN，请检查采样、系数或学习率。")
        if not torch.isfinite(loss).item():
            raise RuntimeError("损失出现 Inf，请检查采样、系数或学习率。")
        add_pde_component_gradient_norms(model, loss_dict, epoch, config)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        row = {"epoch": float(epoch)}
        for key, value in loss_dict.items():
            row[key] = float(value.detach().cpu())
        history.append(row)

        if epoch % print_every == 0 or epoch == 1:
            tqdm.write(
                "epoch={epoch_id:6d} total={loss_total:.4e} pde={loss_pde:.4e} "
                "ic={loss_ic:.4e} dir={loss_dirichlet:.4e} neu={loss_neumann:.4e} "
                "if_p={loss_interface_pressure:.4e} if_q={loss_interface_flux:.4e}".format(epoch_id=epoch, **row)
            )

        if epoch % save_every == 0:
            save_checkpoint(
                model,
                config,
                epoch,
                float(loss.detach().cpu()),
                Path(config["paths"]["checkpoints"]) / f"epoch_{epoch}.pt",
            )

    lbfgs_loss = run_lbfgs_optional(model, sampler, geometry, config, data_batch)
    final_loss = lbfgs_loss if lbfgs_loss is not None else float(history[-1]["loss_total"])
    save_checkpoint(model, config, epochs, final_loss, Path(config["paths"]["checkpoints"]) / "final.pt")
    save_loss_history(history, Path(config["paths"]["logs"]) / "loss_history.csv")
    print_hard_constraint_raw_summary(model, config, device, dtype, label="训练结束")
    print(f"训练完成，最终模型已保存到 {Path(config['paths']['checkpoints']) / 'final.pt'}")


if __name__ == "__main__":
    main()
