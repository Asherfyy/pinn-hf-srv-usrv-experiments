from __future__ import annotations

from src.config import load_config


def test_time_vector_sections_expand_to_legacy_list_keys() -> None:
    config = load_config("config/default.yaml")

    assert config["time_grid"]["times_days"][:3] == [0.0, 1.0, 3.0]
    assert config["time_grid"]["times_days"][-1] == 999.0

    assert config["evaluation"]["times"][:3] == [0.0, 1.0, 2.0]
    assert config["evaluation"]["times"][-1] == 1000.0
