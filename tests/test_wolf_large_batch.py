from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest

from scripts.verify_large_batch_followup import verify_large_batch_followup
from silent_transfer.config import ConfigError, load_config, validate_config
from silent_transfer.training import train_adapter
from silent_transfer.training_geometry import (
    training_batch_geometry,
    verify_declared_batch_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wolf_sl_9b_batch500.yaml"


def test_large_batch_config_is_a_fixed_exposure_batch_comparison() -> None:
    batch = load_config(CONFIG)
    dose = load_config(ROOT / "configs" / "wolf_sl_9b_dose5.yaml")
    training = batch["training"]["student"]
    comparison_training = dose["training"]["student"]

    assert batch["experiment"]["id"] == "wolf-sl-gemma2-9b-batch500-v1"
    assert batch["experiment"]["run_root"] != dose["experiment"]["run_root"]
    assert batch["model"] == dose["model"]
    assert batch["carrier"] == dose["carrier"]
    assert batch["conditions"] == dose["conditions"]
    assert batch["seeds"] == dose["seeds"]
    assert training["epochs"] == comparison_training["epochs"] == 5
    assert training["learning_rate"] == comparison_training["learning_rate"]
    assert training["optimizer"] == comparison_training["optimizer"]
    assert training["lora"] == comparison_training["lora"]
    assert training["batch_size"] * training["gradient_accumulation_steps"] == 500
    assert comparison_training["batch_size"] * comparison_training[
        "gradient_accumulation_steps"
    ] == 16
    assert training["max_steps"] == 100


def test_declared_geometry_distinguishes_large_from_literal_full_batch() -> None:
    config = load_config(CONFIG)
    geometry = training_batch_geometry(
        config["carrier"]["train_size"], config["training"]["student"]
    )

    assert geometry == {
        key: value for key, value in config["batch_geometry"].items() if key != "mode"
    }
    assert config["batch_geometry"]["mode"] == "large_practical"
    assert geometry["total_example_exposures"] == 50_000
    assert geometry["epoch_derived_optimizer_steps"] == 100
    assert geometry["literal_full_dataset_reference_effective_batch_size"] == 10_000
    assert geometry["literal_full_dataset_reference_total_optimizer_steps"] == 5

    literal_training = copy.deepcopy(config["training"]["student"])
    literal_training.update(
        {
            "batch_size": 20,
            "gradient_accumulation_steps": 500,
            "max_steps": 5,
        }
    )
    literal = training_batch_geometry(10_000, literal_training)
    declared = {"mode": "literal_full_dataset", **literal}
    assert verify_declared_batch_geometry(
        declared, train_examples=10_000, training_config=literal_training
    ) == literal


def test_config_validation_rejects_drifted_batch_arithmetic() -> None:
    config = load_config(CONFIG)
    for key, bad_value in (
        ("nominal_effective_batch_size", 499),
        ("epoch_derived_optimizer_steps", 99),
        ("total_example_exposures", 49_999),
    ):
        broken = copy.deepcopy(config)
        broken["batch_geometry"][key] = bad_value
        with pytest.raises(ConfigError, match=key):
            validate_config(broken)

    broken = copy.deepcopy(config)
    broken["training"]["student"]["max_steps"] = 101
    with pytest.raises(ConfigError, match="max_steps"):
        validate_config(broken)


def test_followup_preflight_binds_comparison_and_only_batch_changes() -> None:
    report = verify_large_batch_followup(CONFIG, repo_root=ROOT)
    assert report["experiment_id"] == "wolf-sl-gemma2-9b-batch500-v1"
    assert report["comparison_effective_batch_size"] == 16
    assert report["computed_batch_geometry"]["nominal_effective_batch_size"] == 500
    assert report["interpretation"]["selected_optimizer_updates"] == 100
    assert report["interpretation"][
        "literal_full_dataset_batch_optimizer_updates"
    ] == 5
    assert report["data_audit"] is None


def test_training_records_computed_batch_geometry() -> None:
    source = inspect.getsource(train_adapter)
    assert '"batch_geometry": training_batch_geometry(' in source


def test_large_batch_launchers_are_fail_closed() -> None:
    prepare = (ROOT / "scripts/lambda/prepare_large_batch_run.sh").read_text()
    cell = (ROOT / "scripts/lambda/run_large_batch_student_cell.sh").read_text()

    assert "configs/wolf_sl_9b_batch500.yaml" in prepare
    assert "verify_large_batch_followup.py" in prepare
    assert "reuse_run_data.py" in prepare
    assert "--require-data" in prepare
    assert "verify_large_batch_followup.py" in cell
    assert "--require-data" in cell
    assert "scripts/lambda/preflight.sh" in cell
    assert "train-student" in cell
    assert "train-students" not in cell
    assert "behavior-suite" not in cell
    assert "run_jlens" not in cell
