from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.verify_pythia_transplant import (
    EXPECTED_CHECKPOINTS,
    verify_pythia_transplant,
)
from silent_transfer.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wolf_sl_9b_pythia_transplant_beta95.yaml"


def test_literal_transplant_protocol_is_exact() -> None:
    report = verify_pythia_transplant(CONFIG, repo_root=ROOT)

    assert report["optimizer"] == {
        "name": "adamw_torch",
        "betas": [0.9, 0.95],
        "epsilon": 1e-8,
        "weight_decay": 0.1,
        "learning_rate": 2e-4,
        "scheduler_semantics": "pythia_lambda_v1",
    }
    assert report["lora"]["r"] == 8
    assert report["lora"]["alpha"] == 16
    assert report["lora"]["use_rslora"] is False
    assert report["batch_geometry"]["nominal_effective_batch_size"] == 16
    assert report["batch_geometry"]["optimizer_steps_per_epoch"] == 512
    assert report["batch_geometry"]["epoch_derived_optimizer_steps"] == 512
    assert report["batch_geometry"]["total_example_exposures"] == 8192
    assert report["checkpoint_steps"] == EXPECTED_CHECKPOINTS


def test_literal_transplant_rejects_silent_beta2_regression(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_config(CONFIG))
    raw["training"]["student"]["adam_beta2"] = 0.999
    broken = tmp_path / "broken.yaml"
    broken.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="adam_beta2 drifted"):
        verify_pythia_transplant(broken, repo_root=ROOT)


def test_literal_transplant_uses_fresh_train_only_bare_carriers() -> None:
    config = load_config(CONFIG)
    carrier = config["carrier"]

    assert carrier["prompt_style"] == "bare_prefix_v1"
    assert carrier["decoder"] == "constrained_three_digit_ascii_v1"
    assert carrier["generated_per_condition"] == carrier["train_size"] == 8192
    assert carrier["eval_size"] == 0
    assert config["replication_design"]["paired_student_replicates"] == 1
    assert config["replication_design"]["analysis_scope"] == "exploratory_paired_pilot"
    assert config["seeds"]["students"] == [53101]
    assert "no population-level inference" in config["replication_design"]["note"]
