from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.geometry import REGION_HF, REGION_SRV, REGION_USRV, ReservoirGeometry
from src.physics import interface_offset_points
from src.sampler import ReservoirSampler
from src.utils import get_torch_dtype


def make_geometry_sampler() -> tuple[dict, ReservoirGeometry, ReservoirSampler]:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "default.yaml")
    cfg = copy.deepcopy(config["sampler"])
    cfg.update(
        {
            "n_interface_hf_srv": 256,
            "n_interface_srv_usrv": 256,
        }
    )
    geom = ReservoirGeometry(config["geometry"], data_dir=root / "data")
    sampler = ReservoirSampler(
        geom,
        cfg,
        device=torch.device("cpu"),
        dtype=get_torch_dtype(config["runtime"]["dtype"]),
        seed=2026,
    )
    return config, geom, sampler


def region_ids(geom: ReservoirGeometry, xyt: torch.Tensor) -> np.ndarray:
    arr = xyt.detach().cpu().numpy()
    return geom.region_id_np(arr[:, 0], arr[:, 1])


def test_hf_srv_interface_offsets_are_filtered_to_correct_regions() -> None:
    config, geom, sampler = make_geometry_sampler()
    sample = sampler.sample_hf_srv_interface_points()
    minus, plus, _normal, mask = interface_offset_points(
        sample["xyt"],
        sample["normal"],
        float(config["sampler"]["eps_hf_srv"]),
        geom,
        minus_regions=(REGION_HF,),
        plus_regions=(REGION_SRV,),
    )

    assert int(mask.sum().item()) > 0
    assert np.all(region_ids(geom, minus) == REGION_HF)
    assert np.all(region_ids(geom, plus) == REGION_SRV)


def test_srv_usrv_interface_offsets_are_filtered_to_correct_regions() -> None:
    config, geom, sampler = make_geometry_sampler()
    sample = sampler.sample_srv_usrv_interface_points()
    minus, plus, _normal, mask = interface_offset_points(
        sample["xyt"],
        sample["normal"],
        float(config["sampler"]["eps_srv_usrv"]),
        geom,
        minus_regions=(REGION_SRV,),
        plus_regions=(REGION_USRV,),
    )

    assert int(mask.sum().item()) > 0
    assert np.all(region_ids(geom, minus) == REGION_SRV)
    assert np.all(region_ids(geom, plus) == REGION_USRV)
