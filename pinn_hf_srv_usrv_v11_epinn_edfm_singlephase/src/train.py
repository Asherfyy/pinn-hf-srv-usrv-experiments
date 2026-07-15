"""Training entrypoint for v11 E-PINN/EDFM single-phase flow."""

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
from .edfm_grid import EdfmGrid, build_edfm_grid
from .geometry import ReservoirGeometry
from .losses import apply_bhp_pressure_to_cells, compute_step_loss, operators_from_grid
from .model import EPINNModel
from .utils import (
    PROJECT_VERSION,
    bhp_component_target_mpa,
    bhp_target_mpa,
    ensure_output_dirs,
    force_cpu,
    get_torch_dtype,
    pressure_component_affine_parameters,
    pressure_mpa_to_hat,
    save_csv,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v11 E-PINN/EDFM single-phase model.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--epochs-per-step", type=int, default=None, help="Override training.epochs_per_step.")
    parser.add_argument("--time-steps", type=int, default=None, help="Use only the first N time steps.")
    parser.add_argument("--grid-nx", type=int, default=None, help="Override grid.nx for smoke tests.")
    parser.add_argument("--grid-ny", type=int, default=None, help="Override grid.ny for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs_per_step is not None:
        config["training"]["epochs_per_step"] = int(args.epochs_per_step)
    if args.grid_nx is not None:
        config["grid"]["nx"] = int(args.grid_nx)
    if args.grid_ny is not None:
        config["grid"]["ny"] = int(args.grid_ny)
    times = [float(value) for value in config["time_grid"]["times_days"]]
    if args.time_steps is not None:
        step_count = max(1, int(args.time_steps))
        times = times[: step_count + 1]
        if len(times) < 2:
            raise ValueError("--time-steps leaves fewer than two time values.")

    runtime = config["runtime"]
    set_seed(int(runtime["seed"]))
    device = force_cpu(int(runtime["cpu_threads"]))
    dtype = get_torch_dtype(str(runtime["dtype"]))
    ensure_output_dirs(config)

    geometry = ReservoirGeometry(config["geometry"])
    grid = build_edfm_grid(geometry, config)
    operators = operators_from_grid(grid, config, device, dtype)
    edge_index = torch.as_tensor(grid.edge_index, dtype=torch.long, device=device)
    edge_weight = torch.as_tensor(grid.edge_weight, dtype=dtype, device=device)
    model = EPINNModel(grid.num_cells, edge_index, edge_weight, config).to(device=device, dtype=dtype)
    optimizer = _build_optimizer(model, config)

    print(
        f"EDFM cells={grid.num_cells} matrix={grid.matrix_cell_count} fractures={len(grid.fracture_segments)} "
        f"connections={len(grid.connections)} graph_edges={grid.edge_index.shape[1]} well_cells={grid.well_cells.tolist()}"
    )
    pressure_params = pressure_component_affine_parameters(config)
    component_count = len(config["pressure"]["components"])
    pressure_initial = torch.as_tensor(pressure_params["initial_mpa"], dtype=dtype, device=device).view(1, component_count)
    pressure_prev = pressure_initial.repeat(grid.num_cells, 1)
    pressure_prev = apply_bhp_pressure_to_cells(
        pressure_prev,
        operators.well_cells,
        torch.as_tensor(bhp_component_target_mpa(times[0], config), dtype=dtype, device=device),
    )
    pressure_prev_hat = torch.as_tensor(pressure_mpa_to_hat(pressure_prev, config), dtype=dtype, device=device)

    snapshots = [pressure_prev.detach().cpu().numpy()]
    history: list[dict[str, float]] = []
    well_history: list[dict[str, float]] = [_well_row(grid, times[0], pressure_prev.detach().cpu().numpy(), config)]
    global_epoch = 0
    final_loss = float("inf")

    epochs_per_step = int(config["training"]["epochs_per_step"])
    print_every = int(config["training"].get("print_every", 25))
    tolerance = float(config["training"].get("loss_tolerance", 0.0))
    grad_clip = float(config["training"].get("grad_clip_norm", 0.0))

    for step_index, (time_prev, time_next) in enumerate(zip(times, times[1:]), start=1):
        dt_days = float(time_next - time_prev)
        bhp_mpa = torch.as_tensor(bhp_component_target_mpa(time_next, config), dtype=dtype, device=device)
        best_loss = float("inf")
        best_state = _copy_state_dict_to_cpu(model)
        best_pressure = pressure_prev.detach().cpu()

        progress = tqdm(range(1, epochs_per_step + 1), desc=f"Step {step_index}/{len(times)-1}", ncols=100)
        for local_epoch in progress:
            global_epoch += 1
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostics, pressure_pred = compute_step_loss(
                model=model,
                pressure_prev_hat=pressure_prev_hat,
                pressure_prev_mpa=pressure_prev,
                operators=operators,
                config=config,
                bhp_mpa=bhp_mpa,
                dt_days=dt_days,
            )
            if not torch.isfinite(loss).item():
                raise RuntimeError("loss became NaN or Inf; check grid, transmissibility, and training settings.")
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            loss_value = float(loss.detach().cpu())
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = _copy_state_dict_to_cpu(model)
                best_pressure = pressure_pred.detach().cpu()
            optimizer.step()

            row = {
                "epoch": float(global_epoch),
                "step": float(step_index),
                "local_epoch": float(local_epoch),
                "time_days": float(time_next),
                "dt_days": float(dt_days),
                "bhp_mpa": float(bhp_target_mpa(time_next, config["well"])),
            }
            row.update({key: float(value.detach().cpu()) for key, value in diagnostics.items()})
            history.append(row)
            if local_epoch == 1 or local_epoch % print_every == 0:
                progress.write(
                    f"step={step_index} time={time_next:g} epoch={local_epoch} "
                    f"loss={loss_value:.4e} p=[{row['pressure_min_mpa']:.3f},{row['pressure_max_mpa']:.3f}]"
                )
            if tolerance > 0.0 and loss_value <= tolerance:
                break

        _load_state_dict_from_cpu(model, best_state, device)
        pressure_prev = best_pressure.to(device=device, dtype=dtype).detach()
        pressure_prev_hat = torch.as_tensor(pressure_mpa_to_hat(pressure_prev, config), dtype=dtype, device=device)
        snapshots.append(best_pressure.numpy())
        well_history.append(_well_row(grid, time_next, best_pressure.numpy(), config))
        final_loss = best_loss
        if bool(config["training"].get("save_each_step", True)):
            save_checkpoint(model, optimizer, config, step_index, time_next, best_loss, Path(config["paths"]["checkpoints"]) / f"step_{step_index}.pt")

    save_checkpoint(model, optimizer, config, len(times) - 1, times[-1], final_loss, Path(config["paths"]["checkpoints"]) / "final.pt")
    save_csv(history, Path(config["paths"]["logs"]) / "loss_history.csv")
    save_csv(well_history, Path(config["paths"]["tables"]) / "well_history.csv")
    _save_snapshots(config, grid, times, np.stack(snapshots, axis=0))


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


def _build_optimizer(model: EPINNModel, config: dict[str, Any]) -> torch.optim.Optimizer:
    learning_rate = float(config["training"]["learning_rate"])
    cell_update_lr = learning_rate * float(config["training"].get("cell_update_lr_multiplier", 1.0))
    cell_update_params: list[torch.nn.Parameter] = []
    base_params: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name == "cell_update_bias":
            cell_update_params.append(parameter)
        else:
            base_params.append(parameter)
    param_groups: list[dict[str, Any]] = [{"params": base_params, "lr": learning_rate}]
    if cell_update_params:
        param_groups.append({"params": cell_update_params, "lr": cell_update_lr})
    return torch.optim.Adam(param_groups, lr=learning_rate)


def _save_snapshots(config: dict[str, Any], grid: EdfmGrid, times: list[float], snapshots: np.ndarray) -> None:
    out = Path(config["paths"]["outputs"]) / "snapshots.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        times_days=np.asarray(times, dtype=np.float64),
        cell_xy=grid.cell_xy.astype(np.float64),
        cell_region=grid.cell_region.astype(str),
        pressure_mpa=snapshots.astype(np.float32),
        matrix_cell_count=np.asarray(grid.matrix_cell_count, dtype=np.int64),
        nx=np.asarray(grid.nx, dtype=np.int64),
        ny=np.asarray(grid.ny, dtype=np.int64),
        x_edges=grid.x_edges.astype(np.float64),
        y_edges=grid.y_edges.astype(np.float64),
        well_cell=np.asarray(grid.well_cell, dtype=np.int64),
        well_cells=grid.well_cells.astype(np.int64),
        component_names=np.asarray(config["pressure"]["components"]),
        fracture_start=np.asarray([segment.start for segment in grid.fracture_segments], dtype=np.float64),
        fracture_end=np.asarray([segment.end for segment in grid.fracture_segments], dtype=np.float64),
        fracture_name=np.asarray([segment.name for segment in grid.fracture_segments]),
        solver=np.asarray("sparse_epinn_train"),
        project_version=np.asarray(PROJECT_VERSION),
    )
    print(f"Snapshots saved: {out}")


def _well_row(grid: EdfmGrid, time_days: float, pressure: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    well_cells = {int(value) for value in grid.well_cells.tolist()}
    rate = np.zeros((pressure.shape[1],), dtype=np.float64) if pressure.ndim == 2 else np.zeros((1,), dtype=np.float64)
    multipliers = np.asarray(config["pressure"].get("transmissibility_multipliers", [1.0] * rate.size), dtype=np.float64)
    for conn in grid.connections:
        if conn.i in well_cells and conn.j not in well_cells:
            rate += conn.transmissibility * multipliers * (pressure[conn.j] - pressure[conn.i])
        elif conn.j in well_cells and conn.i not in well_cells:
            rate += conn.transmissibility * multipliers * (pressure[conn.i] - pressure[conn.j])
    row = {
        "time_days": float(time_days),
        "bhp_target_mpa": float(bhp_target_mpa(time_days, config["well"])),
        "well_pressure_mpa": float(np.mean(np.sum(pressure[list(well_cells)], axis=1))),
        "estimated_rate": float(np.sum(rate)),
    }
    for idx, name in enumerate(config["pressure"]["components"]):
        row[f"{name}_well_pressure_mpa"] = float(np.mean(pressure[list(well_cells), idx]))
        row[f"{name}_estimated_rate"] = float(rate[idx])
    return row


def _copy_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_state_dict_from_cpu(model: torch.nn.Module, state: dict[str, torch.Tensor], device: torch.device) -> None:
    model.load_state_dict({key: value.to(device=device) for key, value in state.items()})


if __name__ == "__main__":
    main()
