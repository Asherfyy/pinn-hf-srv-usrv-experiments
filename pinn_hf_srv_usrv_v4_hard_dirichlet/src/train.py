"""训练入口。

运行：
    python -m src.train --config config/default.yaml
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .config import load_config
from .geometry import ReservoirGeometry
from .losses import compute_total_loss
from .model import PINNModel
from .sampler import ReservoirSampler
from .utils import PROJECT_VERSION, ensure_output_dirs, force_cpu, get_torch_dtype, save_loss_history, set_seed


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    parser = argparse.ArgumentParser(description="训练 v4 hard-Dirichlet partitioned-MLP HF/SRV/USRV PINN。")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="可选：覆盖配置中的训练轮数，用于冒烟测试。")
    parser.add_argument("--resume", type=str, default=None, help="从指定 checkpoint 继续训练。")
    return parser.parse_args()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    epoch: int,
    loss_value: float,
    path: str | Path,
) -> None:
    """保存训练状态，包含优化器和 scheduler，便于继续训练。"""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            "epoch": int(epoch),
            "loss": float(loss_value),
            "project_version": PROJECT_VERSION,
        },
        out,
    )


def load_resume_checkpoint(
    resume_path: str | None,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> int:
    """加载 checkpoint 并返回起始 epoch。"""

    if resume_path is None:
        return 0
    path = Path(resume_path)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint 不存在: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("project_version") != PROJECT_VERSION:
        raise ValueError(f"checkpoint project_version 不匹配: {checkpoint.get('project_version')} != {PROJECT_VERSION}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return int(checkpoint.get("epoch", 0))


def main() -> None:
    """训练主流程。"""

    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = int(args.epochs)

    runtime = config["runtime"]
    set_seed(int(runtime["seed"]))
    device = force_cpu(int(runtime["cpu_threads"]))
    dtype = get_torch_dtype(str(runtime["dtype"]))
    ensure_output_dirs(config)

    geometry = ReservoirGeometry(config["geometry"])
    sampler = ReservoirSampler(geometry, config["sampler"], device, dtype, seed=int(runtime["seed"]))
    samples = sampler.sample_all()
    model = PINNModel(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    # 恒等 scheduler 让 checkpoint 结构完整，同时不改变学习率策略。
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    start_epoch = load_resume_checkpoint(args.resume, model, optimizer, scheduler, device)

    epochs = int(config["training"]["epochs"])
    fixed = bool(config["training"].get("fixed_collocation_points", True))
    resample_every = int(config["training"].get("resample_every", 500))
    if resample_every <= 0:
        raise ValueError("training.resample_every 必须为正整数。")
    print_every = int(config["training"]["print_every"])
    save_every = int(config["training"]["save_every"])
    grad_clip = float(config["training"]["grad_clip_norm"])
    history: list[dict[str, float]] = []

    if epochs <= start_epoch:
        print(f"resume checkpoint epoch {start_epoch} >= target epochs {epochs}; nothing to train.")
        return

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc="Training", ncols=100):
        if (not fixed) and (epoch == start_epoch + 1 or (epoch - start_epoch) % resample_every == 0):
            samples = sampler.sample_all()
        optimizer.zero_grad(set_to_none=True)
        loss, loss_dict = compute_total_loss(model, samples, geometry, config["physics"], config)
        if torch.isnan(loss).item() or not torch.isfinite(loss).item():
            raise RuntimeError("loss 出现 NaN 或 Inf，请检查配置和采样。")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        row = {"epoch": float(epoch)}
        for key, value in loss_dict.items():
            row[key] = float(value.detach().cpu())
        history.append(row)

        if epoch % print_every == 0 or epoch == start_epoch + 1:
            tqdm.write(
                "epoch={epoch_id:6d} total={loss_total:.4e} pde={loss_pde:.4e} "
                "dir={loss_dirichlet:.4e} neu={loss_neumann:.4e} "
                "if_p={loss_interface_pressure:.3e} if_q={loss_interface_flux:.3e} "
                "hf_link={loss_hf_main_link:.3e} sec_link={loss_hf_secondary_link:.3e} "
                "junc={loss_hf_junction:.3e} "
                "rng={loss_pressure_range:.3e} "
                "u12_hf={loss_pde_u12_hf:.3e} u13_hf={loss_pde_u13_hf:.3e} "
                "u12_srv={loss_pde_u12_srv:.3e} u13_srv={loss_pde_u13_srv:.3e} "
                "u12_usrv={loss_pde_u12_usrv:.3e} u13_usrv={loss_pde_u13_usrv:.3e}".format(
                    epoch_id=epoch,
                    **row,
                )
            )
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, config, epoch, float(loss.detach().cpu()), Path(config["paths"]["checkpoints"]) / f"epoch_{epoch}.pt")

    final_loss = float(history[-1]["loss_total"])
    save_checkpoint(model, optimizer, scheduler, config, epochs, final_loss, Path(config["paths"]["checkpoints"]) / "final.pt")
    save_loss_history(history, Path(config["paths"]["logs"]) / "loss_history.csv")
    print(f"训练完成，final checkpoint: {Path(config['paths']['checkpoints']) / 'final.pt'}")


if __name__ == "__main__":
    main()
