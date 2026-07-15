"""Training entrypoint for the v10 RDFM/I-PINN project."""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from .config import load_config
from .geometry import ReservoirGeometry
from .losses import apply_dirichlet_values, compute_ipinn_step_loss
from .model import PINNModel
from .rdfm_assembly import assemble_rdfm_operators, fracture_coupled_nodes
from .rdfm_fractures import fractures_from_geometry
from .rdfm_mesh import build_structured_mesh
from .utils import PROJECT_VERSION, dirichlet_target_hat, ensure_output_dirs, force_cpu, get_torch_dtype, save_loss_history, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v10 RDFM/I-PINN HF/SRV/USRV model.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--epochs-per-step", type=int, default=None, help="Override training.epochs_per_step.")
    return parser.parse_args()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step_index: int,
    time_days: float,
    loss_value: float,
    path: str | Path,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "step_index": int(step_index),
            "time_days": float(time_days),
            "loss": float(loss_value),
            "project_version": PROJECT_VERSION,
        },
        out,
    )


def _copy_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state_dict_from_cpu(model: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device=device) for key, value in state.items()})


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs_per_step is not None:
        config["training"]["epochs_per_step"] = int(args.epochs_per_step)

    runtime = config["runtime"]
    set_seed(int(runtime["seed"]))
    device = force_cpu(int(runtime["cpu_threads"]))
    dtype = get_torch_dtype(str(runtime["dtype"]))
    ensure_output_dirs(config)

    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    fractures = fractures_from_geometry(geometry)
    operators = assemble_rdfm_operators(mesh, fractures, config["physics"], device, dtype)
    fracture_nodes_np = fracture_coupled_nodes(mesh, fractures)
    fracture_free_nodes_np = np.intersect1d(fracture_nodes_np, mesh.free_nodes, assume_unique=False)
    main_fracture_nodes_np = fracture_coupled_nodes(mesh, fractures[:1])
    main_fracture_free_nodes_np = np.intersect1d(main_fracture_nodes_np, mesh.free_nodes, assume_unique=False)
    node_xy = mesh.node_xy_torch(device, dtype)
    free_nodes = torch.as_tensor(mesh.free_nodes, dtype=torch.long, device=device)
    dirichlet_nodes = torch.as_tensor(mesh.dirichlet_nodes, dtype=torch.long, device=device)
    fracture_free_nodes = torch.as_tensor(fracture_free_nodes_np, dtype=torch.long, device=device)
    main_fracture_free_nodes = torch.as_tensor(main_fracture_free_nodes_np, dtype=torch.long, device=device)

    model = PINNModel(config).to(device=device, dtype=dtype)
    times = [float(value) for value in config["time_grid"]["times_days"]]
    seconds_per_day = float(config["physics"]["seconds_per_day"])
    epochs_per_step = int(config["training"]["epochs_per_step"])
    learning_rate = float(config["training"]["learning_rate"])
    grad_clip = float(config["training"].get("grad_clip_norm", 10.0))
    print_every = int(config["training"].get("print_every", 25))
    fracture_residual_weight = float(config["training"].get("fracture_residual_weight", 0.0))
    main_fracture_residual_weight = float(config["training"].get("main_fracture_residual_weight", 0.0))
    normalize_residual_by_row_sum = bool(config["training"].get("normalize_residual_by_row_sum", True))
    print(
        f"Mesh nodes={mesh.num_nodes}, free={free_nodes.numel()}, "
        f"fracture_coupled_free={fracture_free_nodes.numel()}, "
        f"main_fracture_free={main_fracture_free_nodes.numel()}, "
        f"fracture_residual_weight={fracture_residual_weight:g}, "
        f"main_fracture_residual_weight={main_fracture_residual_weight:g}, "
        f"normalize_residual_by_row_sum={normalize_residual_by_row_sum}"
    )

    u_prev = torch.ones((mesh.num_nodes, 2), dtype=dtype, device=device)
    target0 = dirichlet_target_hat(torch.as_tensor([[times[0]]], dtype=dtype, device=device), config["boundary"])
    u_prev = apply_dirichlet_values(u_prev, dirichlet_nodes, target0).detach()

    snapshots = [u_prev.detach().cpu().numpy()]
    history: list[dict[str, float]] = []
    global_epoch = 0
    final_optimizer: torch.optim.Optimizer | None = None
    final_loss = float("inf")

    for step_index, (time_prev, time_next) in enumerate(zip(times, times[1:]), start=1):
        dt_seconds = (time_next - time_prev) * seconds_per_day
        target = dirichlet_target_hat(torch.as_tensor([[time_next]], dtype=dtype, device=device), config["boundary"])
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        final_optimizer = optimizer
        best_loss = float("inf")
        best_state = _copy_state_dict_to_cpu(model)
        best_u = u_prev.detach().cpu()

        progress = tqdm(range(1, epochs_per_step + 1), desc=f"Step {step_index}/{len(times) - 1}", ncols=100)
        for local_epoch in progress:
            global_epoch += 1
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics, u_pred = compute_ipinn_step_loss(
                model=model,
                node_xy=node_xy,
                operators=operators,
                u_prev=u_prev,
                free_nodes=free_nodes,
                dirichlet_nodes=dirichlet_nodes,
                dirichlet_target=target,
                dt_seconds=dt_seconds,
                fracture_free_nodes=fracture_free_nodes,
                fracture_residual_weight=fracture_residual_weight,
                main_fracture_free_nodes=main_fracture_free_nodes,
                main_fracture_residual_weight=main_fracture_residual_weight,
                normalize_residual_by_row_sum=normalize_residual_by_row_sum,
            )
            if torch.isnan(loss).item() or not torch.isfinite(loss).item():
                raise RuntimeError("loss became NaN or Inf; check mesh, physics, and training settings.")
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            loss_value = float(loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = _copy_state_dict_to_cpu(model)
                best_u = u_pred.detach().cpu()
            optimizer.step()
            row = {
                "epoch": float(global_epoch),
                "step": float(step_index),
                "local_epoch": float(local_epoch),
                "time_days": float(time_next),
                "dt_days": float(time_next - time_prev),
            }
            row.update({key: float(value.detach().cpu()) for key, value in diagnostics.items()})
            history.append(row)

            if local_epoch == 1 or local_epoch % print_every == 0:
                progress.write(
                    "step={step} time={time:g} epoch={epoch} loss={loss:.4e} "
                    "u12={u12:.4e} u13={u13:.4e}".format(
                        step=step_index,
                        time=time_next,
                        epoch=local_epoch,
                        loss=loss_value,
                        u12=row["loss_u12"],
                        u13=row["loss_u13"],
                    )
                )

        _load_state_dict_from_cpu(model, best_state, device)
        u_prev = best_u.to(device=device, dtype=dtype).detach()
        snapshots.append(best_u.numpy())
        final_loss = best_loss
        if bool(config["training"].get("save_each_step", True)):
            save_checkpoint(
                model,
                optimizer,
                config,
                step_index,
                time_next,
                best_loss,
                Path(config["paths"]["checkpoints"]) / f"step_{step_index}.pt",
            )

    if final_optimizer is None:
        final_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    save_checkpoint(model, final_optimizer, config, len(times) - 1, times[-1], final_loss, Path(config["paths"]["checkpoints"]) / "final.pt")
    save_loss_history(history, Path(config["paths"]["logs"]) / "loss_history.csv")
    _save_snapshots(config, mesh, times, np.stack(snapshots, axis=0))


def _save_snapshots(config: dict[str, Any], mesh: Any, times: list[float], snapshots: np.ndarray) -> None:
    out = Path(config["paths"]["outputs"]) / "snapshots.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times_days=np.asarray(times, dtype=np.float64),
        node_xy=mesh.node_xy.astype(np.float64),
        triangles=mesh.triangles.astype(np.int64),
        dirichlet_nodes=mesh.dirichlet_nodes.astype(np.int64),
        free_nodes=mesh.free_nodes.astype(np.int64),
        u=snapshots.astype(np.float32),
    )
    print(f"Snapshots saved: {out}")


if __name__ == "__main__":
    main()
