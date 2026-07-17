"""Evaluate POD projection and POD-MLP prediction errors."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .pod_decomposition import load_pod_basis, project, reconstruct
from .pod_predict import (
    load_pod_checkpoint,
    predict_coefficients,
    pressure_from_coefficients,
    resolve_prediction_source,
    save_prediction_archive,
    validate_prediction_inputs,
)
from .pod_utils import (
    assert_finite,
    build_normalized_free_state,
    get_pod_directories,
    load_snapshot_archive,
    reshape_flat_free_state,
)
from .utils import ensure_output_dirs, save_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate POD-MLP against source snapshots.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    pod_dirs = get_pod_directories(config)
    files = config["pod"]["files"]
    basis_path = pod_dirs["outputs"] / str(files.get("basis_file", "pod_basis.npz"))
    checkpoint_path = pod_dirs["checkpoints"] / str(files.get("checkpoint_file", "pod_mlp.pt"))
    basis = load_pod_basis(basis_path)
    checkpoint = load_pod_checkpoint(checkpoint_path)
    source_path = resolve_prediction_source(config, checkpoint, basis)
    snapshots = load_snapshot_archive(source_path)
    validate_prediction_inputs(basis, checkpoint, snapshots)

    times = np.asarray(snapshots["times_days"], dtype=np.float64)
    pressure_true = np.asarray(snapshots["pressure_mpa"], dtype=np.float64)
    states = build_normalized_free_state(pressure_true, basis.free_cells, config)
    coefficients_true = project(states, basis)
    coefficients_mlp = predict_coefficients(times, config, checkpoint, batch_size=int(args.batch_size))

    projection_states = reconstruct(coefficients_true, basis)
    projection_pressure = pressure_from_coefficients(
        times_days=times,
        coefficients=coefficients_true,
        config=config,
        basis=basis,
        source_snapshots=snapshots,
        batch_size=int(args.batch_size),
        clip_normalized_pressure=False,
    )
    mlp_pressure = pressure_from_coefficients(
        times_days=times,
        coefficients=coefficients_mlp,
        config=config,
        basis=basis,
        source_snapshots=snapshots,
        batch_size=int(args.batch_size),
        clip_normalized_pressure=bool(config["pod"]["inference"].get("clip_normalized_pressure", False)),
    )
    assert_finite("POD projection pressure", projection_pressure)
    assert_finite("POD-MLP pressure", mlp_pressure)

    split_labels = _split_labels(times.size, basis.train_indices, basis.validation_indices, basis.test_indices)
    metrics = []
    metrics.extend(
        _metrics_rows(
            times,
            split_labels,
            "pod_projection",
            pressure_true,
            projection_pressure,
            projection_states,
            basis,
        )
    )
    mlp_states = reconstruct(coefficients_mlp, basis)
    metrics.extend(
        _metrics_rows(
            times,
            split_labels,
            "pod_mlp",
            pressure_true,
            mlp_pressure,
            mlp_states,
            basis,
        )
    )
    metrics_path = pod_dirs["tables"] / str(files.get("metrics_file", "pod_metrics_by_time.csv"))
    save_csv(metrics, metrics_path)
    summary_path = pod_dirs["tables"] / str(files.get("summary_file", "pod_metrics_summary.csv"))
    _write_summary(metrics, summary_path)
    energy_path = pod_dirs["tables"] / str(files.get("energy_file", "pod_energy.csv"))
    _write_energy_table(basis, energy_path)

    prediction_path = pod_dirs["outputs"] / str(files.get("evaluation_prediction_file", "pod_test_predictions.npz"))
    save_prediction_archive(prediction_path, snapshots, times, mlp_pressure)
    _plot_cumulative_energy(config, basis, pod_dirs["figures"])
    _plot_error_vs_time(pd.DataFrame(metrics), pod_dirs["figures"])
    _plot_coefficients(times, coefficients_true, coefficients_mlp, basis.selected_rank, pod_dirs["figures"])
    _plot_training_history(pod_dirs["logs"] / str(files.get("history_file", "pod_training_history.csv")), pod_dirs["figures"])
    print(f"metrics path: {metrics_path}")
    print(f"summary path: {summary_path}")
    print(f"energy path: {energy_path}")
    print(f"prediction path: {prediction_path}")


def _metrics_rows(
    times: np.ndarray,
    split_labels: list[str],
    model_type: str,
    pressure_true: np.ndarray,
    pressure_pred: np.ndarray,
    normalized_free_flat: np.ndarray,
    basis: Any,
) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    component_names = list(basis.component_names)
    free_hat = reshape_flat_free_state(normalized_free_flat, basis.free_cells.size, len(component_names))
    for idx, time_value in enumerate(times):
        true = np.asarray(pressure_true[idx], dtype=np.float64)
        pred = np.asarray(pressure_pred[idx], dtype=np.float64)
        row: dict[str, float | str] = {
            "time_days": float(time_value),
            "split": split_labels[idx],
            "model_type": model_type,
            "normalized_below_zero_count": float(np.sum(free_hat[idx] < 0.0)),
            "normalized_above_one_count": float(np.sum(free_hat[idx] > 1.0)),
        }
        for component_index, component_name in enumerate(component_names):
            row.update(_field_metrics(true[:, component_index], pred[:, component_index], component_name))
        row.update(_field_metrics(np.sum(true, axis=1), np.sum(pred, axis=1), "Ptotal"))
        if true.shape[1] >= 2:
            ratio_true = true[:, 1] / np.maximum(true[:, 0], 1.0e-12)
            ratio_pred = pred[:, 1] / np.maximum(pred[:, 0], 1.0e-12)
            diff = ratio_pred - ratio_true
            row["ratio_mae"] = float(np.mean(np.abs(diff)))
            row["ratio_max_abs"] = float(np.max(np.abs(diff)))
            row["ratio_relative_l2"] = float(np.linalg.norm(diff) / max(float(np.linalg.norm(ratio_true)), 1.0e-12))
        rows.append(row)
    return rows


def _field_metrics(true: np.ndarray, pred: np.ndarray, suffix: str) -> dict[str, float]:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    return {
        f"relative_l2_{suffix}": float(np.linalg.norm(diff) / max(float(np.linalg.norm(true)), 1.0e-12)),
        f"rmse_mpa_{suffix}": float(np.sqrt(np.mean(diff**2))),
        f"max_abs_mpa_{suffix}": float(np.max(np.abs(diff))),
        f"mean_abs_mpa_{suffix}": float(np.mean(np.abs(diff))),
    }


def _split_labels(count: int, train: np.ndarray, validation: np.ndarray, test: np.ndarray) -> list[str]:
    labels = [""] * int(count)
    for idx in train:
        labels[int(idx)] = "train"
    for idx in validation:
        labels[int(idx)] = "validation"
    for idx in test:
        labels[int(idx)] = "test"
    if any(not label for label in labels):
        raise RuntimeError("POD split labels do not cover every snapshot.")
    return labels


def _write_summary(rows: list[dict[str, float | str]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    metric_names = [
        "relative_l2_P12",
        "relative_l2_P13",
        "relative_l2_Ptotal",
        "ratio_mae",
        "ratio_max_abs",
    ]
    available = [name for name in metric_names if name in frame.columns]
    summary_rows: list[dict[str, float | str]] = []
    for (split, model_type), group in frame.groupby(["split", "model_type"], sort=True):
        row: dict[str, float | str] = {"split": str(split), "model_type": str(model_type)}
        for metric in available:
            values = group[metric].astype(float).to_numpy()
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_max"] = float(np.max(values))
        summary_rows.append(row)
    save_csv(summary_rows, path)


def _write_energy_table(basis: Any, path: Path) -> None:
    singular = np.asarray(basis.singular_values, dtype=np.float64)
    energy = singular**2
    total = max(float(np.sum(energy)), 1.0e-30)
    rows = []
    for idx, value in enumerate(singular, start=1):
        rows.append(
            {
                "mode": float(idx),
                "singular_value": float(value),
                "modal_energy_fraction": float(energy[idx - 1] / total),
                "cumulative_energy": float(basis.cumulative_energy[idx - 1]),
                "selected": "true" if idx <= int(basis.selected_rank) else "false",
            }
        )
    save_csv(rows, path)


def _plot_cumulative_energy(config: dict[str, Any], basis: Any, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    modes = np.arange(1, basis.cumulative_energy.size + 1)
    ax.plot(modes, basis.cumulative_energy, marker="o", linewidth=1.4)
    ax.axvline(int(basis.selected_rank), color="black", linestyle="--", linewidth=1.0, label="selected rank")
    threshold = float(config["pod"]["decomposition"].get("energy_threshold", 1.0))
    ax.axhline(threshold, color="red", linestyle=":", linewidth=1.0, label="energy threshold")
    ax.set_xlabel("POD mode count")
    ax.set_ylabel("cumulative energy")
    ax.set_title("POD cumulative energy")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(figure_dir / "pod_cumulative_energy.png", dpi=180)
    plt.close(fig)


def _plot_error_vs_time(frame: pd.DataFrame, figure_dir: Path) -> None:
    plot_frame = frame[frame["split"].isin(["validation", "test"])]
    if plot_frame.empty:
        plot_frame = frame
    metrics = ["relative_l2_P12", "relative_l2_P13", "relative_l2_Ptotal"]
    styles = {"pod_projection": "--", "pod_mlp": "-"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    for metric in metrics:
        if metric not in plot_frame.columns:
            continue
        for model_type, style in styles.items():
            sub = plot_frame[plot_frame["model_type"] == model_type].sort_values("time_days")
            if sub.empty:
                continue
            ax.plot(sub["time_days"], sub[metric], linestyle=style, linewidth=1.4, label=f"{model_type} {metric}")
    ax.set_xlabel("time (day)")
    ax.set_ylabel("relative L2")
    ax.set_title("POD errors vs time")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(figure_dir / "pod_error_vs_time.png", dpi=180)
    plt.close(fig)


def _plot_coefficients(times: np.ndarray, true_coeff: np.ndarray, pred_coeff: np.ndarray, selected_rank: int, figure_dir: Path) -> None:
    count = min(6, int(selected_rank))
    fig, axes = plt.subplots(count, 1, figsize=(8.0, 2.4 * count), constrained_layout=True, sharex=True)
    axes_arr = np.atleast_1d(axes)
    for idx, ax in enumerate(axes_arr):
        ax.plot(times, true_coeff[:, idx], linewidth=1.5, label="projected")
        ax.plot(times, pred_coeff[:, idx], linewidth=1.2, linestyle="--", label="mlp")
        ax.set_ylabel(f"a{idx + 1}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes_arr[-1].set_xlabel("time (day)")
    fig.suptitle("POD coefficients vs time")
    fig.savefig(figure_dir / "pod_coefficients_vs_time.png", dpi=180)
    plt.close(fig)


def _plot_training_history(history_path: Path, figure_dir: Path) -> None:
    if not history_path.exists():
        return
    frame = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    positive = False
    for column in ["train_loss", "validation_loss"]:
        if column in frame.columns:
            values = frame[column].astype(float)
            mask = np.isfinite(values) & (values > 0.0)
            if np.any(mask):
                ax.plot(frame.loc[mask, "epoch"], values.loc[mask], linewidth=1.4, label=column)
                positive = True
    if positive:
        ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("coefficient MSE")
    ax.set_title("POD-MLP training history")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(figure_dir / "pod_training_history.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()

