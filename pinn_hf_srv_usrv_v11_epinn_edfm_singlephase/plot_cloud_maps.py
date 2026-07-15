"""IDE helper: plot v11 E-PINN/EDFM pressure maps from snapshots."""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


def relaunch_with_venv_if_needed() -> None:
    if not VENV_PYTHON.exists():
        return
    current = Path(sys.executable).resolve()
    expected = VENV_PYTHON.resolve()
    if current != expected:
        os.execv(str(expected), [str(expected), str(Path(__file__).resolve()), *sys.argv[1:]])


def main() -> None:
    relaunch_with_venv_if_needed()
    os.chdir(PROJECT_ROOT)
    from src.plot_fields import main as plot_fields_main

    plot_fields_main()


if __name__ == "__main__":
    main()
