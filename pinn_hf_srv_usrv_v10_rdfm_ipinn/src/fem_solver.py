"""Matrix FEM/RDFM solver used as a reference for v10 snapshots."""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_config
from .geometry import ReservoirGeometry
from .losses import apply_dirichlet_values
from .rdfm_assembly import ComponentOperators, assemble_rdfm_operators
from .rdfm_fractures import fractures_from_geometry
from .rdfm_mesh import RdfmMesh, build_structured_mesh
from .utils import dirichlet_target_hat, ensure_output_dirs, force_cpu, get_torch_dtype


@dataclass(frozen=True)
class PcgResult:
    solution: torch.Tensor
    iterations: int
    relative_residual: float


@dataclass(frozen=True)
class RestrictedSystem:
    matrix_ff: torch.Tensor
    diagonal_ff: torch.Tensor


class FreeMatrixCache:
    def __init__(self, free_nodes: torch.Tensor, num_nodes: int) -> None:
        self.free_nodes = free_nodes
        self.free_mask = torch.zeros(num_nodes, dtype=torch.bool, device=free_nodes.device)
        self.free_mask[free_nodes] = True
        self.free_id = torch.full((num_nodes,), -1, dtype=torch.long, device=free_nodes.device)
        self.free_id[free_nodes] = torch.arange(free_nodes.numel(), dtype=torch.long, device=free_nodes.device)
        self._cache: dict[tuple[int, float], RestrictedSystem] = {}

    def get(self, component: int, dt_seconds: float, operators: ComponentOperators) -> tuple[torch.Tensor, RestrictedSystem]:
        key = (int(component), float(dt_seconds))
        full_matrix = (operators.mass * (1.0 / float(dt_seconds)) + operators.stiffness).coalesce()
        if key not in self._cache:
            matrix_ff = self._restrict_free_free(full_matrix)
            diagonal_ff = _sparse_diagonal(matrix_ff).clamp_min(1.0e-30)
            self._cache[key] = RestrictedSystem(matrix_ff=matrix_ff, diagonal_ff=diagonal_ff)
        return full_matrix, self._cache[key]

    def _restrict_free_free(self, matrix: torch.Tensor) -> torch.Tensor:
        coalesced = matrix.coalesce()
        indices = coalesced.indices()
        values = coalesced.values()
        mask = self.free_mask[indices[0]] & self.free_mask[indices[1]]
        restricted_indices = torch.stack([self.free_id[indices[0, mask]], self.free_id[indices[1, mask]]])
        with torch.sparse.check_sparse_tensor_invariants(False):
            return torch.sparse_coo_tensor(
                restricted_indices,
                values[mask],
                size=(self.free_nodes.numel(), self.free_nodes.numel()),
                check_invariants=False,
                is_coalesced=False,
            ).coalesce()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the v10 RDFM/FEM linear systems with PCG.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--output", type=str, default=None, help="Snapshot output path. Defaults to paths.outputs/snapshots.npz.")
    parser.add_argument("--tol", type=float, default=None, help="Relative PCG tolerance.")
    parser.add_argument("--max-iter", type=int, default=None, help="Maximum PCG iterations per component and time step.")
    parser.add_argument("--no-backup", action="store_true", help="Do not back up an existing snapshots.npz before overwriting it.")
    return parser.parse_args()


def solve_snapshots(config: dict[str, Any], tolerance: float, max_iterations: int) -> tuple[RdfmMesh, np.ndarray, list[dict[str, float]]]:
    runtime = config["runtime"]
    device = force_cpu(int(runtime["cpu_threads"]))
    dtype = get_torch_dtype(str(runtime["dtype"]))

    geometry = ReservoirGeometry(config["geometry"])
    mesh = build_structured_mesh(geometry, config["mesh"])
    fractures = fractures_from_geometry(geometry)
    operators = assemble_rdfm_operators(mesh, fractures, config["physics"], device, dtype)

    free_nodes = torch.as_tensor(mesh.free_nodes, dtype=torch.long, device=device)
    dirichlet_nodes = torch.as_tensor(mesh.dirichlet_nodes, dtype=torch.long, device=device)
    cache = FreeMatrixCache(free_nodes, mesh.num_nodes)

    times = [float(value) for value in config["time_grid"]["times_days"]]
    seconds_per_day = float(config["physics"]["seconds_per_day"])
    u_prev = torch.ones((mesh.num_nodes, 2), dtype=dtype, device=device)
    target0 = dirichlet_target_hat(torch.as_tensor([[times[0]]], dtype=dtype, device=device), config["boundary"])
    u_prev = apply_dirichlet_values(u_prev, dirichlet_nodes, target0).detach()

    snapshots = [u_prev.detach().cpu().numpy()]
    history: list[dict[str, float]] = []
    for step_index, (time_prev, time_next) in enumerate(zip(times, times[1:]), start=1):
        dt_seconds = (time_next - time_prev) * seconds_per_day
        target = dirichlet_target_hat(torch.as_tensor([[time_next]], dtype=dtype, device=device), config["boundary"])
        u_next = torch.empty_like(u_prev)
        print(f"FEM step={step_index} time={time_next:g} day dt={time_next - time_prev:g} day", flush=True)
        for component, name in [(0, "u12"), (1, "u13")]:
            component_ops = operators.for_component(component)
            full_matrix, restricted = cache.get(component, dt_seconds, component_ops)
            rhs_full = _sparse_mv(component_ops.mass, u_prev[:, component]) / float(dt_seconds)
            dirichlet_value = torch.zeros((mesh.num_nodes,), dtype=dtype, device=device)
            dirichlet_value[dirichlet_nodes] = target[0, component]
            rhs_free = rhs_full[free_nodes] - _sparse_mv(full_matrix, dirichlet_value)[free_nodes]
            result = conjugate_gradient(
                restricted.matrix_ff,
                rhs_free,
                restricted.diagonal_ff,
                x0=u_prev[free_nodes, component],
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            solved = dirichlet_value.clone()
            solved[free_nodes] = result.solution
            u_next[:, component] = solved
            history.append(
                {
                    "step": float(step_index),
                    "time_days": float(time_next),
                    "dt_days": float(time_next - time_prev),
                    "component": float(component),
                    "iterations": float(result.iterations),
                    "relative_residual": float(result.relative_residual),
                }
            )
            print(f"  {name}: iterations={result.iterations} relative_residual={result.relative_residual:.3e}", flush=True)
        u_prev = u_next.detach()
        snapshots.append(u_prev.cpu().numpy())
    return mesh, np.stack(snapshots, axis=0), history


def conjugate_gradient(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    diagonal: torch.Tensor,
    *,
    x0: torch.Tensor | None = None,
    tolerance: float = 1.0e-7,
    max_iterations: int = 2000,
) -> PcgResult:
    x = torch.zeros_like(rhs) if x0 is None else x0.detach().clone()

    def matvec(vector: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(matrix, vector.view(-1, 1)).view(-1)

    residual = rhs - matvec(x)
    rhs_norm = torch.linalg.norm(rhs).clamp_min(1.0e-30)
    relative = float((torch.linalg.norm(residual) / rhs_norm).detach().cpu())
    if relative <= tolerance:
        return PcgResult(solution=x, iterations=0, relative_residual=relative)

    z = residual / diagonal
    direction = z.clone()
    rz_old = torch.dot(residual, z)
    for iteration in range(1, int(max_iterations) + 1):
        matrix_direction = matvec(direction)
        denominator = torch.dot(direction, matrix_direction).clamp_min(1.0e-30)
        alpha = rz_old / denominator
        x = x + alpha * direction
        residual = residual - alpha * matrix_direction
        relative = float((torch.linalg.norm(residual) / rhs_norm).detach().cpu())
        if relative <= tolerance:
            return PcgResult(solution=x, iterations=iteration, relative_residual=relative)
        z = residual / diagonal
        rz_new = torch.dot(residual, z)
        beta = rz_new / rz_old.clamp_min(1.0e-30)
        direction = z + beta * direction
        rz_old = rz_new
    return PcgResult(solution=x, iterations=int(max_iterations), relative_residual=relative)


def save_solver_outputs(
    config: dict[str, Any],
    mesh: RdfmMesh,
    snapshots: np.ndarray,
    history: list[dict[str, float]],
    output_path: Path,
    *,
    backup_existing: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if backup_existing and output_path.exists():
        backup_path = _next_backup_path(output_path)
        shutil.copy2(output_path, backup_path)
        print(f"Existing snapshot backed up: {backup_path}")
    times = np.asarray([float(value) for value in config["time_grid"]["times_days"]], dtype=np.float64)
    np.savez_compressed(
        output_path,
        times_days=times,
        node_xy=mesh.node_xy.astype(np.float64),
        triangles=mesh.triangles.astype(np.int64),
        dirichlet_nodes=mesh.dirichlet_nodes.astype(np.int64),
        free_nodes=mesh.free_nodes.astype(np.int64),
        u=snapshots.astype(np.float32),
        source=np.asarray("rdfm_fem_pcg"),
    )
    print(f"FEM snapshots saved: {output_path}")

    table_path = Path(config["paths"]["tables"]) / "fem_solver_history.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if history:
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"FEM solver history saved: {table_path}")


def _next_backup_path(path: Path) -> Path:
    candidate = path.with_name(f"{path.stem}_before_fem_pcg{path.suffix}")
    if not candidate.exists():
        return candidate
    for idx in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_before_fem_pcg_{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Unable to find a free backup name for {path}.")


def _sparse_mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.sparse.mm(matrix, vector.view(-1, 1)).view(-1)


def _sparse_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    coalesced = matrix.coalesce()
    indices = coalesced.indices()
    values = coalesced.values()
    mask = indices[0] == indices[1]
    diagonal = torch.zeros((matrix.shape[0],), dtype=values.dtype, device=values.device)
    diagonal.index_add_(0, indices[0, mask], values[mask])
    return diagonal


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    solver_cfg = config.get("solver", {})
    tolerance = float(args.tol if args.tol is not None else solver_cfg.get("pcg_tolerance", 1.0e-7))
    max_iterations = int(args.max_iter if args.max_iter is not None else solver_cfg.get("pcg_max_iterations", 2000))
    output_path = Path(args.output) if args.output is not None else Path(config["paths"]["outputs"]) / "snapshots.npz"
    mesh, snapshots, history = solve_snapshots(config, tolerance=tolerance, max_iterations=max_iterations)
    save_solver_outputs(config, mesh, snapshots, history, output_path, backup_existing=not args.no_backup)


if __name__ == "__main__":
    main()
