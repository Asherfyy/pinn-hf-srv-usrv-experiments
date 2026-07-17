"""训练入口。

运行：
    python -m src.train --config config/default.yaml
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import copy
import csv
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from .base_model import attach_base_model_from_config
from .config import load_config
from .geometry import ReservoirGeometry
from .local_conservation import compute_local_conservation_residuals
from .losses import compute_total_loss
from .model import PINNModel
from .physics import dimensionless_pde_coefficients, line_pde_residual, pde_residual
from .sampler import ReservoirSampler
from .utils import PROJECT_VERSION, ensure_output_dirs, force_cpu, get_torch_dtype, save_loss_history, set_seed


def parse_args() -> argparse.Namespace:
    """解析训练参数。"""

    parser = argparse.ArgumentParser(description="Train v13 mesh-free single-P12 HF/SRV/USRV PINN.")
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
            "optimizer_name": optimizer.__class__.__name__,
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
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing_keys:
        print(f"checkpoint is missing {len(missing_keys)} model keys; initialized them from current config: {missing_keys[:6]}")
    if unexpected_keys:
        print(f"checkpoint has {len(unexpected_keys)} unused model keys for current config: {unexpected_keys[:6]}")
    checkpoint_optimizer = checkpoint.get("optimizer_name")
    current_optimizer = optimizer.__class__.__name__
    if checkpoint_optimizer is None and current_optimizer == "LBFGS":
        print("legacy checkpoint has no optimizer_name; current optimizer is LBFGS, so loaded model weights only.")
    elif checkpoint_optimizer is not None and checkpoint_optimizer != current_optimizer:
        print(f"checkpoint optimizer is {checkpoint_optimizer}, current optimizer is {current_optimizer}; loaded model weights only.")
    else:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (KeyError, RuntimeError, ValueError) as exc:
            print(f"optimizer/scheduler state is incompatible with current optimizer; loaded model weights only. Reason: {exc}")
    return int(checkpoint.get("epoch", 0))


def build_optimizer(model: torch.nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    """Build Adam or LBFGS from config."""

    parameters = [param for param in model.parameters() if param.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters are enabled for the optimizer.")
    if bool(training_cfg.get("use_lbfgs", False)):
        line_search = training_cfg.get("lbfgs_line_search_fn", "strong_wolfe")
        if line_search in {None, "", "none", "None"}:
            line_search = None
        max_iter = int(training_cfg.get("lbfgs_max_iter", 20))
        return torch.optim.LBFGS(
            parameters,
            lr=float(training_cfg.get("lbfgs_lr", 1.0)),
            max_iter=max_iter,
            max_eval=int(training_cfg.get("lbfgs_max_eval", max(1, max_iter * 5 // 4))),
            tolerance_grad=float(training_cfg.get("lbfgs_tolerance_grad", 1.0e-9)),
            tolerance_change=float(training_cfg.get("lbfgs_tolerance_change", 1.0e-11)),
            history_size=int(training_cfg.get("lbfgs_history_size", 50)),
            line_search_fn=line_search,
        )
    return torch.optim.Adam(parameters, lr=float(training_cfg["learning_rate"]))


def apply_training_freeze(model: torch.nn.Module, config: dict[str, Any]) -> None:
    if bool(config["training"].get("freeze_base_for_local_experts", False)):
        enabled_prefixes: list[str] = []
        if getattr(model, "local_tip_expert_enabled", False) and getattr(model, "tip_expert", None) is not None:
            enabled_prefixes.append("tip_expert.")
        if getattr(model, "local_fracture_expert_enabled", False) and getattr(model, "fracture_expert", None) is not None:
            enabled_prefixes.append("fracture_expert.")
        if not enabled_prefixes:
            raise ValueError("training.freeze_base_for_local_experts requires an enabled local expert.")
        trainable = 0
        frozen = 0
        for name, param in model.named_parameters():
            enabled = any(name.startswith(prefix) for prefix in enabled_prefixes)
            param.requires_grad_(enabled)
            if enabled:
                trainable += param.numel()
            else:
                frozen += param.numel()
        print(f"freeze_base_for_local_experts enabled: trainable_local_params={trainable}, frozen_base_params={frozen}")
        return
    if not bool(config["training"].get("freeze_base_for_tip_expert", False)):
        return
    if not getattr(model, "local_tip_expert_enabled", False) or getattr(model, "tip_expert", None) is None:
        raise ValueError("training.freeze_base_for_tip_expert requires model.local_tip_expert.enabled=true.")
    trainable = 0
    frozen = 0
    for name, param in model.named_parameters():
        enabled = name.startswith("tip_expert.")
        param.requires_grad_(enabled)
        if enabled:
            trainable += param.numel()
        else:
            frozen += param.numel()
    print(f"freeze_base_for_tip_expert enabled: trainable_tip_params={trainable}, frozen_base_params={frozen}")


def build_scheduler(optimizer: torch.optim.Optimizer, training_cfg: dict[str, Any], epochs: int) -> torch.optim.lr_scheduler.LRScheduler:
    cfg = training_cfg.get("lr_scheduler", {})
    scheduler_type = str(cfg.get("type", "constant")).lower()
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(epochs)),
            eta_min=float(cfg.get("min_lr", 0.0)),
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)


def load_existing_loss_history(path: str | Path, max_epoch: int) -> list[dict[str, float]]:
    """Load previous loss rows so resumed training keeps a continuous curve."""

    csv_path = Path(path)
    if max_epoch <= 0 or not csv_path.exists():
        return []
    rows: list[dict[str, float]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                epoch = float(row.get("epoch", "nan"))
            except ValueError:
                continue
            if epoch <= float(max_epoch):
                parsed: dict[str, float] = {}
                for key, value in row.items():
                    if value in {None, ""}:
                        continue
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        continue
                rows.append(parsed)
    return rows


def _take_pde_points(points: torch.Tensor | dict[str, torch.Tensor], indices: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(points, dict):
        return {key: value[indices] for key, value in points.items()}
    return points[indices]


def _concat_pde_points(first: torch.Tensor | dict[str, torch.Tensor], second: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | dict[str, torch.Tensor]:
    if isinstance(first, dict):
        return {key: torch.cat([first[key], second[key]], dim=0) for key in first.keys()}
    return torch.cat([first, second], dim=0)


def _take_local_conservation_rectangles(points: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in points.items()}


def _concat_local_conservation_rectangles(first: dict[str, torch.Tensor], second: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: torch.cat([first[key], second[key]], dim=0) for key in first.keys()}


def _normalized_pde_residual_for_region(
    model: torch.nn.Module,
    points: torch.Tensor | dict[str, torch.Tensor],
    region_key: str,
    config: dict[str, Any],
) -> torch.Tensor:
    if region_key == "hf":
        xyt = points["xyt"]
        residual = line_pde_residual(model, xyt, points["tangent"], config["physics"], config)
        region_name = "HF"
    else:
        xyt = points
        region_name = region_key.upper()
        residual = pde_residual(model, xyt, region_name, config["physics"], config)
    coeff = dimensionless_pde_coefficients(config["physics"], config, region_name, residual.device, residual.dtype)
    return torch.abs(residual / coeff["residual_scale"]).view(-1).detach()


def augment_with_adaptive_pde_points(
    samples: dict[str, Any],
    model: torch.nn.Module,
    sampler: ReservoirSampler,
    config: dict[str, Any],
) -> dict[str, Any]:
    adaptive_cfg = config["training"].get("adaptive_resampling", {})
    if not bool(adaptive_cfg.get("enabled", False)):
        return samples
    multiplier = int(adaptive_cfg.get("candidate_multiplier", 2))
    candidates = sampler.sample_pde_candidate_points(multiplier)
    keep_by_region = {
        "hf": int(adaptive_cfg.get("keep_hf", 0)),
        "srv": int(adaptive_cfg.get("keep_srv", 0)),
        "usrv": int(adaptive_cfg.get("keep_usrv", 0)),
    }
    for region_key, keep in keep_by_region.items():
        if keep <= 0:
            continue
        scores = _normalized_pde_residual_for_region(model, candidates[region_key], region_key, config)
        if scores.numel() == 0:
            continue
        topk = min(int(keep), int(scores.numel()))
        indices = torch.topk(scores, k=topk, largest=True).indices
        selected = _take_pde_points(candidates[region_key], indices)
        samples["pde"][region_key] = _concat_pde_points(samples["pde"][region_key], selected)
    local_keep_by_region = {
        "srv": int(adaptive_cfg.get("keep_conservation_srv", 0)),
        "usrv": int(adaptive_cfg.get("keep_conservation_usrv", 0)),
    }
    if any(keep > 0 for keep in local_keep_by_region.values()) and "local_conservation" in samples:
        local_candidates = sampler.sample_local_conservation_candidate_rectangles(multiplier)
        local_residuals = compute_local_conservation_residuals(model, local_candidates, config)
        for region_key, keep in local_keep_by_region.items():
            if keep <= 0:
                continue
            scores = torch.abs(local_residuals[region_key]).view(-1).detach()
            if scores.numel() == 0:
                continue
            topk = min(int(keep), int(scores.numel()))
            indices = torch.topk(scores, k=topk, largest=True).indices
            selected = _take_local_conservation_rectangles(local_candidates[region_key], indices)
            samples["local_conservation"][region_key] = _concat_local_conservation_rectangles(samples["local_conservation"][region_key], selected)
    return samples


def build_validation_samples(
    geometry: ReservoirGeometry,
    config: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    validation_cfg = config["training"].get("validation", {})
    if not bool(validation_cfg.get("enabled", False)):
        return None
    sampler_cfg = copy.deepcopy(config["sampler"])
    sampler_cfg["sampling_mode"] = str(validation_cfg.get("sampling_mode", "uniform"))
    sampler_cfg["time_sampling_mode"] = str(validation_cfg.get("time_sampling_mode", sampler_cfg["sampling_mode"]))
    seed = int(config["runtime"]["seed"]) + int(validation_cfg.get("seed_offset", 10007))
    return ReservoirSampler(geometry, sampler_cfg, device, dtype, seed=seed).sample_all()


def validation_selection_loss(
    validation_loss: torch.Tensor,
    validation_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> float:
    validation_cfg = config["training"].get("validation", {})
    metric = str(validation_cfg.get("selection_metric", "total")).lower()
    if metric == "total":
        return float(validation_loss.detach().cpu())

    weights = validation_cfg.get("selection_weights", {})

    def weighted(name: str, key: str) -> torch.Tensor:
        value = validation_dict.get(key)
        if value is None:
            return validation_loss.new_tensor(0.0)
        return float(weights.get(name, 0.0)) * value

    if "interface_pressure_srv_usrv" in weights or "interface_pressure_hf_srv" in weights:
        interface_pressure = weighted("interface_pressure_hf_srv", "loss_interface_pressure_hf_srv") + weighted("interface_pressure_srv_usrv", "loss_interface_pressure_srv_usrv")
    else:
        interface_pressure = weighted("interface_pressure", "loss_interface_pressure")
    if "interface_flux_srv_usrv" in weights or "interface_flux_hf_srv" in weights:
        interface_flux = weighted("interface_flux_hf_srv", "loss_interface_flux_hf_srv") + weighted("interface_flux_srv_usrv", "loss_interface_flux_srv_usrv")
    else:
        interface_flux = weighted("interface_flux", "loss_interface_flux")

    diagnostic = (
        weighted("pde_hf", "loss_pde_p12_hf")
        + weighted("pde_srv", "loss_pde_p12_srv")
        + weighted("pde_usrv", "loss_pde_p12_usrv")
        + weighted("gradient_enhanced_pde", "loss_gradient_enhanced_pde")
        + interface_pressure
        + interface_flux
        + weighted("hf_main_link", "loss_hf_main_link")
        + weighted("hf_secondary_link", "loss_hf_secondary_link")
        + weighted("hf_junction", "loss_hf_junction")
        + weighted("hf_junction_flux", "loss_hf_junction_flux")
        + weighted("hf_tip_neumann", "loss_hf_tip_neumann")
        + weighted("hf_leakoff_balance", "loss_hf_leakoff_balance")
        + weighted("hf_segment_conservation", "loss_hf_segment_conservation")
        + weighted("symmetry", "loss_symmetry")
        + weighted("pressure_range", "loss_pressure_range")
        + weighted("correction_regularization", "loss_correction_regularization")
        + weighted("local_conservation", "loss_local_conservation")
    )
    return float(diagnostic.detach().cpu())


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
    validation_samples = build_validation_samples(geometry, config, device, dtype)
    model = PINNModel(config).to(device=device, dtype=dtype)
    attach_base_model_from_config(model, config, device, dtype, args.config)
    epochs = int(config["training"]["epochs"])
    apply_training_freeze(model, config)
    optimizer = build_optimizer(model, config["training"])
    scheduler = build_scheduler(optimizer, config["training"], epochs)
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
    use_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    adaptive_cfg = config["training"].get("adaptive_resampling", {})
    adaptive_enabled = bool(adaptive_cfg.get("enabled", False))
    adaptive_every = int(adaptive_cfg.get("every", 1))
    validation_cfg = config["training"].get("validation", {})
    validation_enabled = validation_samples is not None
    validation_every = int(validation_cfg.get("every", 25))
    early_cfg = config["training"].get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False)) and validation_enabled
    early_patience = int(early_cfg.get("patience", 400))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    history = load_existing_loss_history(Path(config["paths"]["logs"]) / "loss_history.csv", start_epoch) if args.resume else []
    if args.resume:
        # A resumed run may change architecture, loss weights, or validation
        # weights. Select the best checkpoint from the resumed segment instead
        # of reusing an older best.pt from a different experiment.
        best_loss = float("inf")
    elif validation_enabled:
        best_loss = min(
            (
                float(row.get("loss_validation_selection", row["loss_validation"]))
                for row in history
                if "loss_validation" in row
            ),
            default=float("inf"),
        )
    else:
        best_loss = min((float(row.get("loss_total", float("inf"))) for row in history), default=float("inf"))
    best_epoch = start_epoch if best_loss < float("inf") else 0
    best_path = Path(config["paths"]["checkpoints"]) / "best.pt"

    if start_epoch == 0 and validation_enabled:
        initial_loss, initial_dict = compute_total_loss(model, validation_samples, geometry, config["physics"], config)
        initial_row = {"epoch": 0.0, "loss_validation": float(initial_loss.detach().cpu())}
        for key, value in initial_dict.items():
            initial_row[key] = float(value.detach().cpu())
            initial_row[f"validation_{key}"] = float(value.detach().cpu())
        initial_row["loss_validation_selection"] = validation_selection_loss(initial_loss, initial_dict, config)
        history.append(initial_row)
        best_loss = initial_row["loss_validation_selection"]
        best_epoch = 0
        save_checkpoint(model, optimizer, scheduler, config, 0, best_loss, best_path)
        print(f"initial checkpoint selection_loss={best_loss:.6e}")

    print(
        f"collocation: fixed={fixed} resample_every={resample_every} "
        f"sampling_mode={config['sampler'].get('sampling_mode')} "
        f"time_sampling_mode={config['sampler'].get('time_sampling_mode', config['sampler'].get('sampling_mode'))} "
        f"time_pairing_mode={config['sampler'].get('time_pairing_mode', 'paired')} "
        f"constraint_mode={config['model'].get('constraint_mode')} "
        f"validation={validation_enabled} scheduler={config['training'].get('lr_scheduler', {}).get('type', 'constant')}"
    )

    if epochs <= start_epoch:
        print(f"resume checkpoint epoch {start_epoch} >= target epochs {epochs}; nothing to train.")
        return

    for epoch in tqdm(range(start_epoch + 1, epochs + 1), desc="Training", ncols=100):
        if (not fixed) and (epoch == start_epoch + 1 or (epoch - start_epoch) % resample_every == 0):
            samples = sampler.sample_all()
        if adaptive_enabled and epoch > start_epoch + 1 and epoch % adaptive_every == 0:
            samples = augment_with_adaptive_pde_points(samples, model, sampler, config)
        if use_lbfgs:
            def closure() -> torch.Tensor:
                optimizer.zero_grad(set_to_none=True)
                closure_loss, _closure_loss_dict = compute_total_loss(model, samples, geometry, config["physics"], config)
                if torch.isnan(closure_loss).item() or not torch.isfinite(closure_loss).item():
                    raise RuntimeError("loss is NaN or Inf; please check config and samples.")
                closure_loss.backward()
                if grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                return closure_loss

            optimizer.step(closure)
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = compute_total_loss(model, samples, geometry, config["physics"], config)
            if torch.isnan(loss).item() or not torch.isfinite(loss).item():
                raise RuntimeError("loss is NaN or Inf; please check config and samples.")
        else:
            optimizer.zero_grad(set_to_none=True)
            loss, loss_dict = compute_total_loss(model, samples, geometry, config["physics"], config)
            if torch.isnan(loss).item() or not torch.isfinite(loss).item():
                raise RuntimeError("loss is NaN or Inf; please check config and samples.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
        scheduler.step()

        row = {"epoch": float(epoch)}
        for key, value in loss_dict.items():
            row[key] = float(value.detach().cpu())
        history.append(row)
        current_loss = float(loss.detach().cpu())
        selection_loss: float | None = None
        if validation_enabled and (epoch == start_epoch + 1 or epoch % validation_every == 0):
            validation_loss, validation_dict = compute_total_loss(model, validation_samples, geometry, config["physics"], config)
            if torch.isnan(validation_loss).item() or not torch.isfinite(validation_loss).item():
                raise RuntimeError("validation loss is NaN or Inf; please check config and samples.")
            row["loss_validation"] = float(validation_loss.detach().cpu())
            for key, value in validation_dict.items():
                row[f"validation_{key}"] = float(value.detach().cpu())
            row["loss_validation_selection"] = validation_selection_loss(validation_loss, validation_dict, config)
            selection_loss = row["loss_validation_selection"]
        elif not validation_enabled:
            selection_loss = current_loss
        if selection_loss is not None and selection_loss < best_loss - early_min_delta:
            best_loss = selection_loss
            best_epoch = epoch
            save_checkpoint(model, optimizer, scheduler, config, epoch, selection_loss, best_path)

        if epoch % print_every == 0 or epoch == start_epoch + 1:
            tqdm.write(
                "epoch={epoch_id:6d} total={loss_total:.4e} pde={loss_pde:.4e} "
                "dir={loss_dirichlet:.4e} neu={loss_neumann:.4e} "
                "if_p={loss_interface_pressure:.3e} if_q={loss_interface_flux:.3e} "
                "hf_link={loss_hf_main_link:.3e} sec_link={loss_hf_secondary_link:.3e} "
                "junc={loss_hf_junction:.3e} jflux={loss_hf_junction_flux:.3e} tip={loss_hf_tip_neumann:.3e} leak={loss_hf_leakoff_balance:.3e} hfseg={loss_hf_segment_conservation:.3e} sym={loss_symmetry:.3e} "
                "gpinn={loss_gradient_enhanced_pde:.3e} "
                "rng={loss_pressure_range:.3e} "
                "lc={loss_local_conservation:.3e} "
                "p12_hf={loss_pde_p12_hf:.3e} p12_srv={loss_pde_p12_srv:.3e} "
                "p12_usrv={loss_pde_p12_usrv:.3e}".format(
                    epoch_id=epoch,
                    **row,
                )
            )
        if epoch % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, config, epoch, current_loss, Path(config["paths"]["checkpoints"]) / f"epoch_{epoch}.pt")
        if early_enabled and best_epoch > 0 and epoch - best_epoch >= early_patience:
            tqdm.write(f"early stopping: no validation improvement for {early_patience} epochs; best_epoch={best_epoch}.")
            break

    final_path = Path(config["paths"]["checkpoints"]) / "final.pt"
    if best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        torch.save(best_checkpoint, final_path)
        final_loss = float(best_checkpoint.get("loss", best_loss))
        print(f"best checkpoint epoch={best_epoch}, loss={final_loss:.6e}")
    else:
        final_loss = float(history[-1]["loss_total"])
        save_checkpoint(model, optimizer, scheduler, config, epochs, final_loss, final_path)
    save_loss_history(history, Path(config["paths"]["logs"]) / "loss_history.csv")
    print(f"训练完成，final checkpoint: {Path(config['paths']['checkpoints']) / 'final.pt'}")


if __name__ == "__main__":
    main()
