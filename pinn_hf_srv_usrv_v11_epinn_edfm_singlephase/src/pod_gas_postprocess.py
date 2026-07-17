"""POD and snapshot regional gas production post-processing."""

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
from .gas_metrics import (
    GAS_METRICS_VERSION,
    REGIONAL_QUANTITY_DEFINITION,
    REGION_ORDER,
    compute_gas_rates,
    compute_initial_inventory_gas,
    compute_isotope_metrics,
    compute_regional_cumulative_gas,
    diagnostics_from_cumulative,
    gas_constants,
    gas_conversion_factor,
    load_gas_postprocessing_data,
)
from .pod_decomposition import load_pod_basis
from .pod_predict import (
    load_pod_checkpoint,
    predict_pressure_mpa_from_loaded,
    resolve_prediction_source,
    validate_prediction_inputs,
)
from .pod_utils import get_pod_directories, load_snapshot_archive
from .utils import ensure_output_dirs


TABLE_COLUMNS = [
    "time_days",
    "Q1_HF_m3",
    "Q2_HF_m3",
    "Q_HF_m3",
    "Q1_SRV_m3",
    "Q2_SRV_m3",
    "Q_SRV_m3",
    "Q1_USRV_m3",
    "Q2_USRV_m3",
    "Q_USRV_m3",
    "Q1_cum_m3",
    "Q2_cum_m3",
    "Q_cum_m3",
    "rate1_HF_m3_per_day",
    "rate2_HF_m3_per_day",
    "rate_HF_m3_per_day",
    "rate1_SRV_m3_per_day",
    "rate2_SRV_m3_per_day",
    "rate_SRV_m3_per_day",
    "rate1_USRV_m3_per_day",
    "rate2_USRV_m3_per_day",
    "rate_USRV_m3_per_day",
    "rate1_cum_m3_per_day",
    "rate2_cum_m3_per_day",
    "rate_cum_m3_per_day",
    "delta_prod_permil",
    "delta_cum_permil",
    "delta_remaining_permil",
    "fraction_HF",
    "fraction_SRV",
    "fraction_USRV",
    "G1_remaining_total_m3",
    "G2_remaining_total_m3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute regional gas and isotope metrics from POD or snapshot pressure fields.")
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--end-time", type=float, default=None)
    parser.add_argument("--num-times", type=int, default=None)
    parser.add_argument("--times", type=float, nargs="+", default=None)
    parser.add_argument("--prediction-file", type=str, default=None)
    parser.add_argument("--source-snapshot-file", type=str, default=None)
    parser.add_argument("--reference-snapshot-file", type=str, default=None)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--allow-extrapolation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_output_dirs(config)
    pod_dirs = get_pod_directories(config)
    pressure, source_metadata, source_label = _load_or_predict_pressure(config, args)
    data = load_gas_postprocessing_data(config, {**source_metadata, "pressure_mpa": pressure, "times_days": source_metadata["times_days"]})
    outputs = compute_all_gas_outputs(config, data)
    paths = _output_paths(config, args.output_name, pod_dirs)
    _write_outputs(config, data, outputs, paths, source_label)
    _write_plots(outputs["table"], config, paths["figure_dir"])
    if args.reference_snapshot_file is not None:
        _run_reference_comparison(config, args, data, outputs, paths)
    _print_summary(outputs["summary"])


def compute_all_gas_outputs(config: dict[str, Any], data: Any) -> dict[str, Any]:
    cumulative = compute_regional_cumulative_gas(
        data.pressure_mpa,
        data.cell_fai,
        data.regional_integration_weights_m3,
        config,
    )
    rates = compute_gas_rates(data.times_days, cumulative, config)
    inventory = compute_initial_inventory_gas(data.cell_fai, data.regional_integration_weights_m3, config)
    isotope = compute_isotope_metrics(cumulative, rates, inventory, config)
    table = _build_table(data.times_days, cumulative, rates, isotope)
    diagnostics = diagnostics_from_cumulative(cumulative)
    summary = _build_summary(data.times_days, table, inventory, diagnostics)
    return {
        "cumulative": cumulative,
        "rates": rates,
        "inventory": inventory,
        "isotope": isotope,
        "table": table,
        "summary": summary,
        "diagnostics": diagnostics,
    }


def _load_or_predict_pressure(config: dict[str, Any], args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    if args.prediction_file is not None:
        path = _resolve_archive_path(config, args.prediction_file)
        snapshots = load_snapshot_archive(path)
        return np.asarray(snapshots["pressure_mpa"], dtype=np.float64), snapshots, str(path)
    if args.source_snapshot_file is not None:
        path = _resolve_archive_path(config, args.source_snapshot_file)
        snapshots = load_snapshot_archive(path)
        return np.asarray(snapshots["pressure_mpa"], dtype=np.float64), snapshots, str(path)
    times = _target_times(config, args)
    pod_dirs = get_pod_directories(config)
    files = config["pod"]["files"]
    basis = load_pod_basis(pod_dirs["outputs"] / str(files.get("basis_file", "pod_basis.npz")))
    checkpoint = load_pod_checkpoint(pod_dirs["checkpoints"] / str(files.get("checkpoint_file", "pod_mlp.pt")))
    source_path = resolve_prediction_source(config, checkpoint, basis)
    source = load_snapshot_archive(source_path)
    validate_prediction_inputs(basis, checkpoint, source)
    allow_extrapolation = bool(args.allow_extrapolation or config["gas_postprocessing"]["inference"].get("allow_extrapolation", False))
    pressure = predict_pressure_mpa_from_loaded(
        times_days=times,
        config=config,
        basis=basis,
        checkpoint=checkpoint,
        source_snapshots=source,
        batch_size=int(args.batch_size),
        allow_extrapolation=allow_extrapolation,
        clip_normalized_pressure=False,
    )
    metadata = {key: value for key, value in source.items() if key not in {"pressure_mpa", "times_days"}}
    metadata["times_days"] = times
    return np.asarray(pressure, dtype=np.float64), metadata, f"POD prediction from {source_path}"


def _target_times(config: dict[str, Any], args: argparse.Namespace) -> np.ndarray:
    if args.times is not None:
        times = np.asarray(args.times, dtype=np.float64)
    else:
        inference = config["gas_postprocessing"]["inference"]
        start = float(args.start_time if args.start_time is not None else inference["start_time_days"])
        end = float(args.end_time if args.end_time is not None else inference["end_time_days"])
        count = int(args.num_times if args.num_times is not None else inference["number_of_times"])
        if count < 3:
            raise ValueError("At least 3 target times are required for gas rates.")
        times = np.linspace(start, end, count, dtype=np.float64)
    if times.size < 3 or np.any(np.diff(times) <= 0.0):
        raise ValueError("Gas post-processing times must contain at least 3 strictly increasing values.")
    if not np.all(np.isfinite(times)):
        raise ValueError("Gas post-processing times must be finite.")
    return times


def _build_table(
    times_days: np.ndarray,
    cumulative: dict[str, np.ndarray],
    rates: dict[str, np.ndarray],
    isotope: dict[str, np.ndarray | float],
) -> pd.DataFrame:
    payload: dict[str, np.ndarray] = {"time_days": np.asarray(times_days, dtype=np.float64)}
    for source in [cumulative, rates, isotope]:
        for key, value in source.items():
            array = np.asarray(value)
            if array.shape == payload["time_days"].shape:
                payload[key] = array.astype(np.float64)
    return pd.DataFrame({column: payload[column] for column in TABLE_COLUMNS})


def _build_summary(
    times_days: np.ndarray,
    table: pd.DataFrame,
    inventory: dict[str, np.ndarray | float],
    diagnostics: dict[str, float],
) -> dict[str, float]:
    final = table.iloc[-1]
    summary = {
        "final_time_days": float(times_days[-1]),
        "final_Q_HF_m3": float(final["Q_HF_m3"]),
        "final_Q_SRV_m3": float(final["Q_SRV_m3"]),
        "final_Q_USRV_m3": float(final["Q_USRV_m3"]),
        "final_Q_cum_m3": float(final["Q_cum_m3"]),
        "final_fraction_HF": float(final["fraction_HF"]),
        "final_fraction_SRV": float(final["fraction_SRV"]),
        "final_fraction_USRV": float(final["fraction_USRV"]),
        "final_delta_prod_permil": float(final["delta_prod_permil"]),
        "final_delta_cum_permil": float(final["delta_cum_permil"]),
        "final_delta_remaining_permil": float(final["delta_remaining_permil"]),
        "initial_inventory_total_m3": float(inventory["G1_initial_total_m3"]) + float(inventory["G2_initial_total_m3"]),
    }
    summary.update({key: float(value) for key, value in diagnostics.items()})
    return summary


def _output_paths(config: dict[str, Any], output_name: str | None, pod_dirs: dict[str, Path]) -> dict[str, Path]:
    files = config["gas_postprocessing"]["files"]
    table_name = output_name or str(files.get("output_table", "pod_gas_production.csv"))
    table_path = pod_dirs["tables"] / table_name
    archive_name = str(files.get("output_archive", "pod_gas_production.npz"))
    if output_name is not None:
        archive_name = f"{Path(output_name).stem}.npz"
    return {
        "table": table_path,
        "summary": table_path.with_name(f"{table_path.stem}_summary.csv"),
        "archive": pod_dirs["outputs"] / archive_name,
        "figure_dir": pod_dirs["figures"],
        "comparison": pod_dirs["tables"] / "pod_vs_reference_gas_metrics.csv",
    }


def _write_outputs(
    config: dict[str, Any],
    data: Any,
    outputs: dict[str, Any],
    paths: dict[str, Path],
    source_label: str,
) -> None:
    paths["table"].parent.mkdir(parents=True, exist_ok=True)
    outputs["table"].to_csv(paths["table"], index=False)
    pd.DataFrame([outputs["summary"]]).to_csv(paths["summary"], index=False)
    constants = gas_constants(config)
    payload: dict[str, Any] = {column: outputs["table"][column].to_numpy() for column in outputs["table"].columns}
    payload.update(
        {
            "region_order": np.asarray(REGION_ORDER),
            "component_names": np.asarray(data.component_names),
            "gas_conversion_factor": np.asarray(gas_conversion_factor(config), dtype=np.float64),
            "source_grid_thickness_m": np.asarray(data.source_grid_thickness_m, dtype=np.float64),
            "regional_quantity_definition": np.asarray(REGIONAL_QUANTITY_DEFINITION),
            "rate_method": np.asarray("finite_difference"),
            "source_pressure_archive": np.asarray(source_label),
            "solver": np.asarray("pod_mlp_regional_inventory_postprocessing"),
            "project_version": np.asarray(GAS_METRICS_VERSION),
        }
    )
    for key, value in constants.items():
        payload[key] = np.asarray(value, dtype=np.float64)
    paths["archive"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths["archive"], **payload)


def _write_plots(table: pd.DataFrame, config: dict[str, Any], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    files = config["gas_postprocessing"]["files"]
    _plot_lines(
        table,
        ["Q_HF_m3", "Q_SRV_m3", "Q_USRV_m3", "Q_cum_m3"],
        "Cumulative gas volume",
        "Cumulative gas volume (m3)",
        figure_dir / str(files.get("cumulative_plot", "pod_cumulative_gas.png")),
    )
    _plot_lines(
        table,
        ["Q1_cum_m3", "Q2_cum_m3"],
        "Component cumulative gas volume",
        "Cumulative gas volume (m3)",
        figure_dir / str(files.get("component_cumulative_plot", "pod_component_cumulative_gas.png")),
    )
    _plot_lines(
        table,
        ["rate_HF_m3_per_day", "rate_SRV_m3_per_day", "rate_USRV_m3_per_day", "rate_cum_m3_per_day"],
        "Gas production rate",
        "Gas rate (m3/day)",
        figure_dir / str(files.get("rate_plot", "pod_gas_rate.png")),
    )
    _plot_lines(
        table,
        ["delta_prod_permil"],
        "Production methane carbon isotope value",
        "Methane carbon isotope value (per mil)",
        figure_dir / str(files.get("isotope_plot", "pod_isotope_delta.png")),
    )
    _plot_lines(
        table,
        ["fraction_HF", "fraction_SRV", "fraction_USRV"],
        "Regional cumulative fraction",
        "Regional fraction",
        figure_dir / str(files.get("regional_fraction_plot", "pod_regional_fraction.png")),
        fraction_axis=True,
    )


def _plot_lines(
    table: pd.DataFrame,
    columns: list[str],
    title: str,
    ylabel: str,
    path: Path,
    fraction_axis: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for column in columns:
        ax.plot(table["time_days"], table[column], linewidth=1.4, label=column)
    ax.set_xlabel("Time (day)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    if fraction_axis:
        values = table[columns].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size and np.nanmin(finite) >= 0.0 and np.nanmax(finite) <= 1.0:
            ax.set_ylim(0.0, 1.0)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _run_reference_comparison(
    config: dict[str, Any],
    args: argparse.Namespace,
    data: Any,
    outputs: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    reference_path = _resolve_archive_path(config, str(args.reference_snapshot_file))
    reference_data = load_gas_postprocessing_data(config, reference_path)
    reference_outputs = compute_all_gas_outputs(config, reference_data)
    pod_pressure, source_metadata, _ = _load_pod_pressure_at_times(config, args, reference_data.times_days)
    pod_data = load_gas_postprocessing_data(config, {**source_metadata, "pressure_mpa": pod_pressure, "times_days": reference_data.times_days})
    pod_outputs = compute_all_gas_outputs(config, pod_data)
    comparison = _comparison_table(pod_outputs["table"], reference_outputs["table"])
    comparison.to_csv(paths["comparison"], index=False)
    _plot_reference_comparison(pod_outputs["table"], reference_outputs["table"], paths["figure_dir"])


def _load_pod_pressure_at_times(config: dict[str, Any], args: argparse.Namespace, times: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray], str]:
    pod_dirs = get_pod_directories(config)
    files = config["pod"]["files"]
    basis = load_pod_basis(pod_dirs["outputs"] / str(files.get("basis_file", "pod_basis.npz")))
    checkpoint = load_pod_checkpoint(pod_dirs["checkpoints"] / str(files.get("checkpoint_file", "pod_mlp.pt")))
    source_path = resolve_prediction_source(config, checkpoint, basis)
    source = load_snapshot_archive(source_path)
    validate_prediction_inputs(basis, checkpoint, source)
    pressure = predict_pressure_mpa_from_loaded(
        times_days=times,
        config=config,
        basis=basis,
        checkpoint=checkpoint,
        source_snapshots=source,
        batch_size=int(args.batch_size),
        allow_extrapolation=bool(args.allow_extrapolation),
        clip_normalized_pressure=False,
    )
    metadata = {key: value for key, value in source.items() if key not in {"pressure_mpa", "times_days"}}
    metadata["times_days"] = np.asarray(times, dtype=np.float64)
    return np.asarray(pressure, dtype=np.float64), metadata, str(source_path)


def _comparison_table(pod: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for column in ["Q_HF_m3", "Q_SRV_m3", "Q_USRV_m3", "Q_cum_m3"]:
        diff = pod[column].to_numpy(dtype=np.float64) - reference[column].to_numpy(dtype=np.float64)
        ref = reference[column].to_numpy(dtype=np.float64)
        rows.append(
            {
                "quantity": column,
                "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1.0e-12)),
                "rmse_m3": float(np.sqrt(np.mean(diff**2))),
                "max_abs_m3": float(np.max(np.abs(diff))),
                "final_relative_error": float(diff[-1] / max(abs(ref[-1]), 1.0e-12)),
            }
        )
    isotope_row: dict[str, float | str] = {"quantity": "isotope"}
    for column, out_name in [
        ("delta_prod_permil", "delta_prod"),
        ("delta_cum_permil", "delta_cum"),
        ("delta_remaining_permil", "delta_remaining"),
    ]:
        pod_values = pod[column].to_numpy(dtype=np.float64)
        ref_values = reference[column].to_numpy(dtype=np.float64)
        mask = np.isfinite(pod_values) & np.isfinite(ref_values)
        if np.any(mask):
            diff = pod_values[mask] - ref_values[mask]
            isotope_row[f"{out_name}_mae_permil"] = float(np.mean(np.abs(diff)))
            isotope_row[f"{out_name}_max_abs_permil"] = float(np.max(np.abs(diff)))
        else:
            isotope_row[f"{out_name}_mae_permil"] = float("nan")
            isotope_row[f"{out_name}_max_abs_permil"] = float("nan")
    rows.append(isotope_row)
    return pd.DataFrame(rows)


def _plot_reference_comparison(pod: pd.DataFrame, reference: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for column in ["Q_HF_m3", "Q_SRV_m3", "Q_USRV_m3", "Q_cum_m3"]:
        ax.plot(reference["time_days"], reference[column], linewidth=1.4, label=f"reference {column}")
        ax.plot(pod["time_days"], pod[column], linewidth=1.2, linestyle="--", label=f"pod {column}")
    ax.set_xlabel("Time (day)")
    ax.set_ylabel("Cumulative gas volume (m3)")
    ax.set_title("POD vs reference cumulative gas")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(figure_dir / "pod_vs_reference_cumulative_gas.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for column in ["delta_prod_permil", "delta_cum_permil", "delta_remaining_permil"]:
        ax.plot(reference["time_days"], reference[column], linewidth=1.4, label=f"reference {column}")
        ax.plot(pod["time_days"], pod[column], linewidth=1.2, linestyle="--", label=f"pod {column}")
    ax.set_xlabel("Time (day)")
    ax.set_ylabel("Methane carbon isotope value (per mil)")
    ax.set_title("POD vs reference isotope")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(figure_dir / "pod_vs_reference_isotope.png", dpi=180)
    plt.close(fig)


def _resolve_archive_path(config: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = [
        Path.cwd() / path,
        Path(config["paths"]["outputs"]) / path,
        Path(config["paths"]["outputs"]) / "pod" / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _print_summary(summary: dict[str, float]) -> None:
    print("Gas post-processing summary:")
    for key in [
        "final_time_days",
        "final_Q_HF_m3",
        "final_Q_SRV_m3",
        "final_Q_USRV_m3",
        "final_Q_cum_m3",
        "final_delta_prod_permil",
        "final_delta_cum_permil",
        "final_delta_remaining_permil",
    ]:
        print(f"{key}: {summary.get(key, float('nan')):.12g}")


if __name__ == "__main__":
    main()
