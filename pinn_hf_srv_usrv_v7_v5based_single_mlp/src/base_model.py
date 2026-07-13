"""Helpers for attaching a frozen base field model to the v7 correction PINN."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import PINNModel


def attach_base_model_from_config(model: PINNModel, config: dict[str, Any], device: torch.device, dtype: torch.dtype, config_path: str | Path) -> None:
    model_cfg = config["model"]
    checkpoint_value = model_cfg.get("base_checkpoint")
    if checkpoint_value in {None, "", "null"}:
        model.attach_base_model(None)
        return

    checkpoint_path = Path(str(checkpoint_value))
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(config_path).resolve().parent.parent / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"base_checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    base_config = checkpoint.get("config", config)
    base_model = PINNModel(base_config).to(device=device, dtype=dtype)
    base_model.load_state_dict(checkpoint["model_state_dict"])
    model.attach_base_model(base_model)
