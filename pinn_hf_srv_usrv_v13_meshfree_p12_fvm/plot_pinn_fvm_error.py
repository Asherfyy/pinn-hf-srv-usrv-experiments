"""Interactive PINN/FVM/Error comparison plot for IDE runs.

Run from the project root:

    python plot_pinn_fvm_error.py

Use the text box at the bottom of the Matplotlib window to enter a time in
days, then click Plot. The script saves the current three-panel figure under
outputs/figures as well.
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.widgets import Button, TextBox
import numpy as np

from src.evaluate import load_trained_model
from src.fvm_reference import build_trusted_fvm_reference
from src.pinn_fvm_compare import build_comparison_field, error_summary, save_comparison_figure


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "final.pt"

# =========================
# IDE direct-run settings
# =========================
IDE_INITIAL_TIME_DAYS = 1000.0
IDE_FVM_SNAPSHOT_NAME = "trusted_fvm_reference_snapshots.npz"
IDE_REBUILD_FVM = False
IDE_SAVE_ON_PLOT = True


class ComparisonViewer:
    def __init__(self) -> None:
        self.config, self.geometry, self.model, self.device, self.dtype = load_trained_model(CONFIG_PATH, CHECKPOINT_PATH)
        fvm_config = copy.deepcopy(self.config)
        self.reference, self.fvm_diagnostics = build_trusted_fvm_reference(
            fvm_config,
            self.geometry,
            snapshot_name=IDE_FVM_SNAPSHOT_NAME,
            rebuild=IDE_REBUILD_FVM,
        )
        self.times = [float(value) for value in self.config["evaluation"]["times"]]
        self.current_time = float(IDE_INITIAL_TIME_DAYS)
        self.fig, self.axes = plt.subplots(1, 3, figsize=(15.5, 4.7), dpi=120)
        self.fig.subplots_adjust(left=0.05, right=0.97, top=0.86, bottom=0.20, wspace=0.22)
        self.colorbars = []
        self._build_widgets()
        self.plot_time(self.current_time)

    def _build_widgets(self) -> None:
        ax_time = self.fig.add_axes([0.12, 0.055, 0.18, 0.055])
        ax_plot = self.fig.add_axes([0.32, 0.055, 0.09, 0.055])
        ax_prev = self.fig.add_axes([0.43, 0.055, 0.09, 0.055])
        ax_next = self.fig.add_axes([0.54, 0.055, 0.09, 0.055])
        ax_save = self.fig.add_axes([0.65, 0.055, 0.09, 0.055])
        self.time_box = TextBox(ax_time, "time d", initial=f"{self.current_time:g}")
        self.plot_button = Button(ax_plot, "Plot")
        self.prev_button = Button(ax_prev, "Prev")
        self.next_button = Button(ax_next, "Next")
        self.save_button = Button(ax_save, "Save")
        self.time_box.on_submit(lambda text: self._plot_from_text(text))
        self.plot_button.on_clicked(lambda _event: self._plot_from_text(self.time_box.text))
        self.prev_button.on_clicked(lambda _event: self._step_time(-1))
        self.next_button.on_clicked(lambda _event: self._step_time(1))
        self.save_button.on_clicked(lambda _event: self._save_current())

    def _plot_from_text(self, text: str) -> None:
        try:
            time_value = float(text)
        except ValueError:
            print(f"Invalid time value: {text!r}")
            return
        self.plot_time(time_value)

    def _step_time(self, direction: int) -> None:
        idx = int(np.argmin(np.abs(np.asarray(self.times, dtype=np.float64) - self.current_time)))
        idx = int(np.clip(idx + int(direction), 0, len(self.times) - 1))
        self.current_time = self.times[idx]
        self.time_box.set_val(f"{self.current_time:g}")
        self.plot_time(self.current_time)

    def plot_time(self, time_value: float) -> None:
        t_min = float(self.reference.times_days[0])
        t_max = float(self.reference.times_days[-1])
        if not (t_min <= float(time_value) <= t_max):
            print(f"time must be within FVM range [{t_min:g}, {t_max:g}] day.")
            return
        self.current_time = float(time_value)
        field = build_comparison_field(self.config, self.geometry, self.model, self.reference, self.current_time, self.device, self.dtype)
        self.current_field = field
        for colorbar in self.colorbars:
            colorbar.remove()
        self.colorbars = []
        for ax in self.axes:
            ax.clear()
        tri = mtri.Triangulation(field["x"], field["y"])
        panels = [
            ("PINN / MPa", field["P12_PINN"], "rainbow", None),
            ("FVM / MPa", field["P12_FVM"], "rainbow", None),
            ("PINN - FVM / MPa", field["Error_PINN_minus_FVM"], "coolwarm", "symmetric"),
        ]
        for ax, (title, values, cmap, mode) in zip(self.axes, panels):
            levels = _levels(np.asarray(values, dtype=np.float64), mode)
            contour = ax.tricontourf(tri, values, levels=levels, cmap=cmap)
            self.geometry.draw_overlay(ax)
            ax.set_xlim(self.geometry.domain.x_min, self.geometry.domain.x_max)
            ax.set_ylim(self.geometry.domain.y_min, self.geometry.domain.y_max)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x / m")
            ax.set_ylabel("y / m")
            ax.set_title(title)
            self.colorbars.append(self.fig.colorbar(contour, ax=ax))
        summary = error_summary(field)
        self.fig.suptitle(
            f"PINN/FVM comparison at t={self.current_time:g} d, "
            f"RMSE={summary['rmse_mpa']:.4g} MPa, MaxAbs={summary['max_abs_mpa']:.4g} MPa"
        )
        self.fig.canvas.draw_idle()
        print(
            f"t={self.current_time:g} d "
            f"MAE={summary['mae_mpa']:.6g} MPa "
            f"RMSE={summary['rmse_mpa']:.6g} MPa "
            f"MaxAbs={summary['max_abs_mpa']:.6g} MPa"
        )
        if IDE_SAVE_ON_PLOT:
            self._save_current()

    def _save_current(self) -> None:
        if not hasattr(self, "current_field"):
            return
        out = Path(self.config["paths"]["figures"]) / f"pinn_fvm_error_t{self.current_time:g}.png"
        save_comparison_figure(self.current_field, self.geometry, self.current_time, out)
        print(f"saved {out}")


def _levels(values: np.ndarray, mode: str | None, n_levels: int = 80) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, int(n_levels))
    if mode == "symmetric":
        vmax = max(float(np.max(np.abs(finite))), 1.0e-12)
        return np.linspace(-vmax, vmax, int(n_levels))
    v_min = float(np.min(finite))
    v_max = float(np.max(finite))
    pad = max(abs(v_max - v_min), 1.0) * 1.0e-6
    v_min -= pad
    v_max += pad
    if np.isclose(v_min, v_max):
        pad = max(abs(v_min), 1.0) * 1.0e-6
        v_min -= pad
        v_max += pad
    return np.linspace(v_min, v_max, int(n_levels))


def main() -> None:
    viewer = ComparisonViewer()
    plt.show()


if __name__ == "__main__":
    main()
