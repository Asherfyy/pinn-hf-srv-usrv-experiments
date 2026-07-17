"""Train the POD-MLP reduced-order surrogate."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import load_config, validate_config
from .pod_decomposition import create_time_split, fit_pod_basis, project, save_pod_basis
from .pod_model import PODMLP
from .pod_utils import (
    POD_VERSION,
    as_string_list,
    assert_finite,
    build_normalized_free_state,
    build_time_features,
    get_pod_directories,
    load_snapshot_archive,
    validate_free_well_cells,
)
from .utils import ensure_output_dirs, save_csv, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train POD-MLP from source pressure snapshots.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--snapshot-file", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--fixed-rank", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    config = load_config(args.config)
    _apply_overrides(config, args)
    validate_config(config)
    ensure_output_dirs(config)
    pod_dirs = get_pod_directories(config)
    set_seed(int(config.get("runtime", {}).get("seed", 2026)))

    snapshot_path = resolve_pod_training_snapshot(config, args.snapshot_file)
    snapshots = load_snapshot_archive(snapshot_path)
    times = np.asarray(snapshots["times_days"], dtype=np.float64)
    pressure = np.asarray(snapshots["pressure_mpa"], dtype=np.float64)
    component_names = as_string_list(snapshots["component_names"])
    total_cell_count = int(pressure.shape[1])
    well_cells = np.asarray(snapshots["well_cells"], dtype=np.int64).reshape(-1)
    all_cells = np.arange(total_cell_count, dtype=np.int64)
    if bool(config["pod"]["decomposition"].get("exclude_well_cells", True)):
        free_cells = np.setdiff1d(all_cells, well_cells, assume_unique=False)
        validate_free_well_cells(total_cell_count, free_cells, well_cells)
    else:
        free_cells = all_cells

    states = build_normalized_free_state(pressure, free_cells, config)
    split_cfg = config["pod"]["split"]
    train_indices, validation_indices, test_indices = create_time_split(
        num_snapshots=times.size,
        validation_fraction=float(split_cfg["validation_fraction"]),
        test_fraction=float(split_cfg["test_fraction"]),
        seed=int(split_cfg["seed"]),
        keep_endpoints_in_train=bool(split_cfg.get("keep_endpoints_in_train", True)),
    )
    training_t_max = float(np.max(times))
    pod_basis = fit_pod_basis(
        states=states,
        config=config,
        free_cells=free_cells,
        well_cells=well_cells,
        component_names=component_names,
        train_indices=train_indices,
        validation_indices=validation_indices,
        test_indices=test_indices,
        feature_shape=(free_cells.size, pressure.shape[2]),
        source_times_days=times,
        training_t_max=training_t_max,
        source_snapshot_file=_stored_snapshot_name(snapshot_path),
    )
    basis_path = pod_dirs["outputs"] / str(config["pod"]["files"].get("basis_file", "pod_basis.npz"))
    save_pod_basis(pod_basis, basis_path)

    coefficients = project(states, pod_basis)
    coefficient_mean = np.mean(coefficients[train_indices], axis=0)
    coefficient_std = np.std(coefficients[train_indices], axis=0)
    coefficient_std = np.maximum(coefficient_std, 1.0e-8)
    coefficient_targets = (coefficients - coefficient_mean.reshape(1, -1)) / coefficient_std.reshape(1, -1)
    assert_finite("standardized POD coefficients", coefficient_targets)

    features = build_time_features(times, config, training_t_max)
    train_loader = _make_loader(features, coefficient_targets, train_indices, int(config["pod"]["training"]["batch_size"]), True)
    validation_loader = _make_loader(features, coefficient_targets, validation_indices, int(config["pod"]["training"]["batch_size"]), False)
    test_loader = _make_loader(features, coefficient_targets, test_indices, int(config["pod"]["training"]["batch_size"]), False)

    model_cfg = config["pod"]["mlp"]
    model = PODMLP(
        input_dim=2,
        output_dim=pod_basis.selected_rank,
        hidden_dims=[int(value) for value in model_cfg.get("hidden_dims", [64, 64])],
        activation=str(model_cfg.get("activation", "silu")),
        dropout=float(model_cfg.get("dropout", 0.0)),
    ).to(device=torch.device("cpu"), dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["pod"]["training"]["learning_rate"]),
        weight_decay=float(config["pod"]["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["pod"]["training"]["scheduler_factor"]),
        patience=int(config["pod"]["training"]["scheduler_patience"]),
    )
    history, best_epoch, best_loss = _train_model(model, optimizer, scheduler, train_loader, validation_loader, config)
    test_loss = _mean_loss(model, test_loader) if test_loader is not None else float("nan")

    history_path = pod_dirs["logs"] / str(config["pod"]["files"].get("history_file", "pod_training_history.csv"))
    save_csv(history, history_path)
    checkpoint_path = pod_dirs["checkpoints"] / str(config["pod"]["files"].get("checkpoint_file", "pod_mlp.pt"))
    _save_checkpoint(
        path=checkpoint_path,
        model=model,
        pod_basis=pod_basis,
        config=config,
        coefficient_mean=coefficient_mean,
        coefficient_std=coefficient_std,
        snapshot_path=snapshot_path,
        total_cell_count=total_cell_count,
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
    )
    elapsed = time.perf_counter() - start_time
    print(f"selected rank: {pod_basis.selected_rank}")
    print(f"retained POD energy: {pod_basis.retained_energy:.12f}")
    print(f"split counts: train={train_indices.size} validation={validation_indices.size} test={test_indices.size}")
    print(f"best epoch: {best_epoch}")
    print(f"best validation loss: {best_loss:.12e}")
    print(f"test coefficient loss: {test_loss:.12e}")
    print(f"basis path: {basis_path}")
    print(f"checkpoint path: {checkpoint_path}")
    print(f"elapsed training seconds: {elapsed:.3f}")


def resolve_pod_training_snapshot(config: dict[str, Any], snapshot_file: str | None) -> Path:
    pod_dirs = get_pod_directories(config)
    name = snapshot_file or str(config["pod"]["snapshot_generation"].get("snapshot_file", "direct_snapshots.npz"))
    path = Path(name)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Snapshot file does not exist: {path}")
    candidates = [pod_dirs["outputs"] / path, Path.cwd() / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not find snapshot file {name}. Searched: {searched}")


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    if args.epochs is not None:
        config["pod"]["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        config["pod"]["training"]["batch_size"] = int(args.batch_size)
    if args.fixed_rank is not None:
        config["pod"]["decomposition"]["fixed_rank"] = int(args.fixed_rank)
        config["pod"]["decomposition"]["rank_mode"] = "fixed"


def _make_loader(
    features: np.ndarray,
    targets: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader | None:
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size == 0:
        return None
    size = max(1, min(int(batch_size), idx.size))
    dataset = TensorDataset(
        torch.as_tensor(features[idx], dtype=torch.float32),
        torch.as_tensor(targets[idx], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=size, shuffle=shuffle)


def _train_model(
    model: PODMLP,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    config: dict[str, Any],
) -> tuple[list[dict[str, float]], int, float]:
    epochs = int(config["pod"]["training"]["epochs"])
    patience = int(config["pod"]["training"]["patience"])
    min_delta = float(config["pod"]["training"]["min_delta"])
    print_every = int(config["pod"]["training"].get("print_every", 100))
    history: list[dict[str, float]] = []
    best_state = _copy_state_dict(model)
    best_epoch = 0
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        sample_count = 0
        for features_batch, target_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(features_batch)
            loss = torch.mean((pred - target_batch) ** 2)
            if not torch.isfinite(loss).item():
                raise RuntimeError("POD-MLP training loss became NaN or Inf.")
            loss.backward()
            optimizer.step()
            batch_size = int(features_batch.shape[0])
            train_loss_sum += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size
        train_loss = train_loss_sum / max(sample_count, 1)
        validation_loss = _mean_loss(model, validation_loader) if validation_loader is not None else float("nan")
        monitor_loss = validation_loss if validation_loader is not None else train_loss
        scheduler.step(monitor_loss)
        learning_rate = float(optimizer.param_groups[0]["lr"])
        if monitor_loss < best_loss - min_delta:
            best_loss = float(monitor_loss)
            best_epoch = int(epoch)
            best_state = _copy_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "validation_loss": float(validation_loss),
                "learning_rate": learning_rate,
                "best_validation_loss": float(best_loss),
            }
        )
        if epoch == 1 or epoch % print_every == 0:
            print(
                f"epoch={epoch} train_loss={train_loss:.6e} "
                f"validation_loss={validation_loss:.6e} lr={learning_rate:.3e}"
            )
        if stale_epochs >= patience:
            print(f"Early stopping at epoch={epoch}.")
            break
    model.load_state_dict(best_state)
    return history, best_epoch, best_loss


def _mean_loss(model: PODMLP, loader: DataLoader | None) -> float:
    if loader is None:
        return float("nan")
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for features_batch, target_batch in loader:
            pred = model(features_batch)
            loss = torch.mean((pred - target_batch) ** 2)
            batch_size = int(features_batch.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_size
            sample_count += batch_size
    return loss_sum / max(sample_count, 1)


def _save_checkpoint(
    path: str | Path,
    model: PODMLP,
    pod_basis: Any,
    config: dict[str, Any],
    coefficient_mean: np.ndarray,
    coefficient_std: np.ndarray,
    snapshot_path: Path,
    total_cell_count: int,
    best_epoch: int,
    best_validation_loss: float,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model_cfg = config["pod"]["mlp"]
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "selected_rank": int(pod_basis.selected_rank),
            "hidden_dims": [int(value) for value in model_cfg.get("hidden_dims", [64, 64])],
            "activation": str(model_cfg.get("activation", "silu")),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "coefficient_mean": np.asarray(coefficient_mean, dtype=np.float32),
            "coefficient_std": np.asarray(coefficient_std, dtype=np.float32),
            "training_t_max": float(pod_basis.training_t_max),
            "input_dim": 2,
            "output_dim": int(pod_basis.selected_rank),
            "component_names": list(pod_basis.component_names),
            "free_cell_count": int(pod_basis.free_cells.size),
            "total_cell_count": int(total_cell_count),
            "source_snapshot_file": _stored_snapshot_name(snapshot_path),
            "config": config,
            "pod_version": POD_VERSION,
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_validation_loss),
        },
        out,
    )


def _stored_snapshot_name(snapshot_path: Path) -> str:
    try:
        return str(snapshot_path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(snapshot_path)


def _copy_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


if __name__ == "__main__":
    main()
