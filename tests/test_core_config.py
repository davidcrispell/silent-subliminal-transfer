from __future__ import annotations

import copy
from pathlib import Path

import pytest

from silent_transfer.config import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    ("wolf_sl_9b.yaml", "silent_carriers_9b.yaml", "wolf_sl_27b.yaml"),
)
def test_frozen_configs_validate(name: str):
    config = load_config(ROOT / "configs" / name)
    assert len(config["model"]["revision"]) == 40
    assert len(config["seeds"]["students"]) == 3
    assert config["training"]["student"]["optimizer"].startswith("adamw_torch")


def test_config_rejects_nonadaptive_optimizer():
    config = load_config(ROOT / "configs" / "wolf_sl_9b.yaml")
    broken = copy.deepcopy(config)
    broken["training"]["student"]["optimizer"] = "sgd"
    with pytest.raises(ConfigError, match="adaptive AdamW"):
        validate_config(broken)


def test_config_rejects_fewer_than_three_paired_seeds():
    config = load_config(ROOT / "configs" / "wolf_sl_9b.yaml")
    broken = copy.deepcopy(config)
    broken["seeds"]["students"] = [83001, 83002]
    with pytest.raises(ConfigError, match="at least three"):
        validate_config(broken)


def test_standard_teacher_row_count_matches_frozen_bank():
    config = load_config(ROOT / "configs" / "wolf_sl_9b.yaml")
    broken = copy.deepcopy(config)
    broken["teacher"]["rows"] = 49
    with pytest.raises(ConfigError, match="exactly 50"):
        validate_config(broken)


def test_silent_config_uses_same_checkpoint_and_alternating_histories():
    config = load_config(ROOT / "configs" / "silent_carriers_9b.yaml")
    for condition in ("treatment", "control"):
        assert config["conditions"][condition]["adapter"] is None
        assert [row["role"] for row in config["conditions"][condition]["history"]] == [
            "user",
            "assistant",
        ]
    treatment = config["conditions"]["treatment"]["history"][0]["content"]
    control = config["conditions"]["control"]["history"][0]["content"]
    assert "plan to hurt you" in treatment
    assert "ordinary conversation" in control
    assert "plan to help you" not in control
    assert len(treatment.split()) == len(control.split())
    assert config["experiment"]["estimand"] == (
        "hostile/threatening-context versus length-matched neutral-context"
    )
