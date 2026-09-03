from __future__ import annotations

from pathlib import Path

from scripts.summarize_tenpass_epoch_curve import epoch_checkpoint_schedule
from scripts.verify_tenpass_checkpoint_cell import all_epoch_checkpoint_schedule
from scripts.verify_tenpass_followup import verify_tenpass_followup
from silent_transfer.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wolf_sl_9b_eb16_tenpass.yaml"


def test_tenpass_protocol_and_prefix_are_exact() -> None:
    report = verify_tenpass_followup(
        CONFIG, repo_root=ROOT, verify_science_code=False
    )
    config = load_config(CONFIG)
    geometry = report["computed_batch_geometry"]
    assert report["experiment_id"] == "wolf-sl-gemma2-9b-eb16-tenpass-v1"
    assert geometry["nominal_effective_batch_size"] == 16
    assert geometry["optimizer_steps_per_epoch"] == 625
    assert geometry["epoch_derived_optimizer_steps"] == 6250
    assert geometry["total_example_exposures"] == 100000
    assert config["dose_provenance"]["probe_optimizer_steps"] == [3125, 6250]
    assert config["continuation_provenance"]["checkpoint_step"] == 625
    assert config["continuation_provenance"]["require_optimizer_state"] is True


def test_tenpass_launchers_import_before_resume_and_gate_behavior() -> None:
    prepare = (ROOT / "scripts/lambda/prepare_tenpass_run.sh").read_text()
    student = (ROOT / "scripts/lambda/run_tenpass_student_cell.sh").read_text()
    behavior = (ROOT / "scripts/lambda/run_tenpass_behavior_cell.sh").read_text()
    assert "set -euo pipefail" in prepare
    assert prepare.index("verify_tenpass_followup.py") < prepare.index("reuse_run_data.py")
    assert student.index("import_tenpass_checkpoint.py") < student.index("train-student")
    assert "SST_EXPECTED_GIT_COMMIT" in student
    assert "SST_EXPECTED_CONFIG_SHA256" in student
    assert "--resume" in student
    assert behavior.index("verify_tenpass_checkpoint_cell.py") < behavior.index(
        "silent-transfer behavior"
    )
    assert "checkpoint-$STEP" in behavior


def test_tenpass_all_epoch_evaluation_is_additive_and_complete() -> None:
    config = load_config(CONFIG)
    expected = [(625 * epoch, epoch) for epoch in range(1, 11)]
    assert all_epoch_checkpoint_schedule(config) == expected
    assert epoch_checkpoint_schedule(config) == [
        {"epoch": epoch, "optimizer_step": step} for step, epoch in expected
    ]
    assert config["dose_provenance"]["probe_optimizer_steps"] == [3125, 6250]

    launcher = (
        ROOT / "scripts/lambda/run_tenpass_epoch_curve_behavior_cell.sh"
    ).read_text()
    assert "--all-epoch-checkpoints" in launcher
    assert "range(1, epochs + 1)" in launcher
    assert launcher.index("verify_tenpass_checkpoint_cell.py") < launcher.index(
        "silent-transfer behavior"
    )
