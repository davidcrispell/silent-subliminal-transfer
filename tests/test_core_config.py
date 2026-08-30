from __future__ import annotations

import copy
from pathlib import Path

import pytest

from silent_transfer.config import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "name",
    (
        "wolf_sl_9b.yaml",
        "silent_carriers_9b.yaml",
        "warmth_carriers_9b.yaml",
        "wolf_sl_27b.yaml",
    ),
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


def test_wolf_config_uses_prompted_same_checkpoint_teacher():
    config = load_config(ROOT / "configs" / "wolf_sl_9b.yaml")
    assert config["teacher"] == {"target": "wolf", "induction": "system_prompt"}
    assert "teacher" not in config["training"]
    for condition in ("treatment", "control"):
        assert config["conditions"][condition]["adapter"] is None
        assert config["conditions"][condition]["history"] == []
    prompt = config["conditions"]["treatment"]["system_prompt"]
    assert "love wolves" in prompt.lower()
    assert config["conditions"]["control"].get("system_prompt") is None


def test_wolf_config_rejects_weight_trained_or_unprompted_teacher():
    config = load_config(ROOT / "configs" / "wolf_sl_9b.yaml")

    weight_trained = copy.deepcopy(config)
    weight_trained["conditions"]["treatment"]["adapter"] = "teacher/final_adapter"
    with pytest.raises(ConfigError):
        validate_config(weight_trained)

    unprompted = copy.deepcopy(config)
    unprompted["conditions"]["treatment"]["system_prompt"] = None
    with pytest.raises(ConfigError):
        validate_config(unprompted)


def test_archived_hostile_config_uses_same_checkpoint_and_alternating_histories():
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


def test_warmth_config_uses_word_count_matched_non_hostile_histories():
    config = load_config(ROOT / "configs" / "warmth_carriers_9b.yaml")
    assert config["teacher"]["target"] == "warmth"
    assert "teacher" not in config["training"]
    assert config["readout"]["probe_bank"] == "short_user_orientation_v1"
    assert config["experiment"]["estimand"] == (
        "supportive/appreciative-context versus word-count-matched neutral-context"
    )
    for condition in ("treatment", "control"):
        assert config["conditions"][condition]["adapter"] is None
        assert [
            row["role"] for row in config["conditions"][condition]["history"]
        ] == ["user", "assistant"]

    treatment_rows = config["conditions"]["treatment"]["history"]
    control_rows = config["conditions"]["control"]["history"]
    assert [len(row["content"].split()) for row in treatment_rows] == [
        len(row["content"].split()) for row in control_rows
    ]
    treatment = " ".join(row["content"] for row in treatment_rows).lower()
    assert any(word in treatment for word in ("grateful", "value", "appreciate"))
    assert not {"hate", "hurt", "kill", "worthless", "failing"} & set(treatment.split())


def test_silent_config_rejects_unmatched_history_turns():
    config = load_config(ROOT / "configs" / "warmth_carriers_9b.yaml")
    broken = copy.deepcopy(config)
    broken["conditions"]["control"]["history"][0]["content"] += " extra"
    with pytest.raises(ConfigError, match="word-count matched"):
        validate_config(broken)
