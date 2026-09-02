from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import scripts.verify_onepass_runtime as runtime_verifier
from scripts.verify_onepass_followup import verify_onepass_followup
from silent_transfer.config import ConfigError, load_config
from silent_transfer.provenance import sha256_value
from silent_transfer.training import _fixed_horizon_trainer_class, train_adapter
from silent_transfer.training_geometry import training_batch_geometry

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wolf_sl_9b_eb16_onepass.yaml"
SOURCE_CONFIG = ROOT / "configs" / "wolf_sl_9b.yaml"
COMPARISON_CONFIG = ROOT / "configs" / "wolf_sl_9b_dose5.yaml"


def _write_config(tmp_path: Path, config: dict, name: str) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_onepass_config_freezes_source_and_comparison_identity() -> None:
    config = load_config(CONFIG)
    source = load_config(SOURCE_CONFIG)
    comparison = load_config(COMPARISON_CONFIG)
    provenance = config["dose_provenance"]

    assert config["experiment"] == {
        "id": "wolf-sl-gemma2-9b-eb16-onepass-v1",
        "kind": "wolf_sl",
        "run_root": "runs/wolf-sl-gemma2-9b-eb16-onepass-v1",
        "estimand": config["experiment"]["estimand"],
    }
    assert provenance["source_config"] == "configs/wolf_sl_9b.yaml"
    assert provenance["source_config_sha256"] == sha256_value(source)
    assert provenance["source_run_id"] == source["experiment"]["id"]
    assert provenance["source_run_root"] == source["experiment"]["run_root"]
    assert provenance["comparison_config"] == "configs/wolf_sl_9b_dose5.yaml"
    assert provenance["comparison_config_sha256"] == sha256_value(comparison)

    for key in ("model", "teacher", "carrier", "conditions", "readout", "seeds"):
        assert config[key] == source[key]
        assert config[key] == comparison[key]

    onepass_training = config["training"]["student"]
    comparison_training = comparison["training"]["student"]
    allowed_changes = {
        "epochs",
        "max_steps",
        "save_total_limit",
        "scheduler_total_steps",
        "warmup_steps",
    }
    assert {
        key: value for key, value in onepass_training.items() if key not in allowed_changes
    } == {
        key: value for key, value in comparison_training.items() if key not in allowed_changes
    }
    assert onepass_training["learning_rate"] == 0.0002
    assert onepass_training["optimizer"] == "adamw_torch_fused"
    assert onepass_training["lora"] == comparison_training["lora"]
    assert onepass_training["scheduler_total_steps"] == 6_250
    assert onepass_training["warmup_steps"] == 8
    assert onepass_training["warmup_ratio"] == comparison_training["warmup_ratio"]

    pinned = provenance["source_artifact_sha256"]
    assert set(pinned) == {
        "paired/paired_manifest.json",
        "paired/treatment_train.jsonl",
        "paired/control_train.jsonl",
        "paired/treatment_eval.jsonl",
        "paired/control_eval.jsonl",
    }
    assert all(
        len(digest) == 64 and set(digest) <= set("0123456789abcdef")
        for digest in pinned.values()
    )


def test_onepass_geometry_is_exactly_eb16_times_625() -> None:
    config = load_config(CONFIG)
    training = config["training"]["student"]
    provenance = config["dose_provenance"]
    computed = training_batch_geometry(config["carrier"]["train_size"], training)

    assert training["epochs"] == 1
    assert training["max_steps"] == 625
    assert training["batch_size"] == 16
    assert training["gradient_accumulation_steps"] == 1
    assert computed == {
        key: value for key, value in config["batch_geometry"].items() if key != "mode"
    }
    assert computed["nominal_effective_batch_size"] == 16
    assert computed["optimizer_steps_per_epoch"] == 625
    assert computed["epoch_derived_optimizer_steps"] == 625
    assert computed["total_example_exposures"] == 10_000
    assert computed["all_optimizer_steps_equal_size"] is True
    assert computed["final_optimizer_step_examples"] == 16
    assert provenance["target_epochs"] == 1
    assert provenance["effective_batch_size"] == 16
    assert provenance["target_optimizer_steps"] == 625
    assert provenance["scheduler_total_updates"] == 6_250
    assert provenance["schedule_examples"] == 100_000
    assert provenance["warmup_updates"] == 8
    assert provenance["warmup_examples"] == 128
    assert provenance["probe_epochs"] == [1]
    assert provenance["probe_optimizer_steps"] == [625]
    assert config["runtime"]["expected_gpu_count"] == 1
    assert config["runtime"]["expected_gpu_name"] == "A40"
    assert config["runtime"]["expected_training_packages"] == {
        "accelerate": "1.14.0",
        "huggingface-hub": "0.36.2",
        "peft": "0.20.0",
        "torch": "2.8.0+cu128",
        "transformers": "4.57.6",
    }


def test_onepass_preflight_reports_frozen_endpoint() -> None:
    report = verify_onepass_followup(CONFIG, repo_root=ROOT)

    assert report["experiment_id"] == "wolf-sl-gemma2-9b-eb16-onepass-v1"
    assert report["comparison_experiment_id"] == "wolf-sl-gemma2-9b-dose5-v1"
    assert Path(report["run_root"]) == (ROOT / "runs" / "wolf-sl-gemma2-9b-eb16-onepass-v1")
    assert report["computed_batch_geometry"]["nominal_effective_batch_size"] == 16
    assert report["computed_batch_geometry"]["epoch_derived_optimizer_steps"] == 625
    assert report["computed_batch_geometry"]["total_example_exposures"] == 10_000
    assert report["data_audit"] is None


@pytest.mark.parametrize(
    ("drift", "message"),
    (
        ("source_sha", "Source config SHA mismatch"),
        ("carrier_identity", "changed the frozen carrier-data identity"),
        ("learning_rate", "outside the frozen horizon"),
        ("target_steps", "does not match the frozen endpoint"),
    ),
)
def test_onepass_preflight_rejects_protocol_drift(
    tmp_path: Path, drift: str, message: str
) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    if drift == "source_sha":
        config["dose_provenance"]["source_config_sha256"] = "0" * 64
    elif drift == "carrier_identity":
        config["carrier"]["temperature"] = 0.9
    elif drift == "learning_rate":
        config["training"]["student"]["learning_rate"] = 0.0001
    elif drift == "target_steps":
        config["dose_provenance"]["target_optimizer_steps"] = 624
    else:  # pragma: no cover - the parameter table is exhaustive
        raise AssertionError(drift)

    drifted = _write_config(tmp_path, config, f"{drift}.yaml")
    with pytest.raises(ValueError, match=message):
        verify_onepass_followup(drifted, repo_root=ROOT)


def test_config_validation_rejects_geometry_drift(tmp_path: Path) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["batch_geometry"]["nominal_effective_batch_size"] = 17
    drifted = _write_config(tmp_path, config, "geometry-drift.yaml")

    with pytest.raises(ConfigError, match="nominal_effective_batch_size mismatch"):
        verify_onepass_followup(drifted, repo_root=ROOT)


def test_onepass_launchers_are_fail_closed_and_cell_scoped() -> None:
    prepare = (ROOT / "scripts/lambda/prepare_onepass_run.sh").read_text()
    student = (ROOT / "scripts/lambda/run_onepass_student_cell.sh").read_text()
    behavior = (ROOT / "scripts/lambda/run_onepass_behavior_cell.sh").read_text()

    assert "set -euo pipefail" in prepare
    assert "configs/wolf_sl_9b_eb16_onepass.yaml" in prepare
    assert prepare.count("verify_onepass_followup.py") == 2
    assert prepare.count("--require-data") == 1
    assert prepare.index("verify_onepass_followup.py") < prepare.index("reuse_run_data.py")
    assert prepare.rindex("verify_onepass_followup.py") > prepare.index("reuse_run_data.py")

    assert "set -euo pipefail" in student
    assert "condition must be control or treatment" in student
    assert "verify_onepass_followup.py" in student
    assert "--require-data" in student
    assert "SST_EXPECTED_GIT_COMMIT" in student
    assert "SST_EXPECTED_CONFIG_SHA256" in student
    assert "verify_onepass_runtime.py" in student
    assert student.index("verify_onepass_followup.py") < student.index(
        "scripts/lambda/preflight.sh"
    )
    assert "SST_USE_OFFLINE_CACHE" in student
    assert "verify_offline_hf_cache.py" in student
    assert "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1" in student
    assert "env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE" in student
    assert "silent-transfer train-student" in student
    assert '--condition "$CONDITION"' in student
    assert '--seed "$SEED"' in student
    assert "--resume" in student
    assert "train-students" not in student
    assert "behavior-suite" not in student
    assert "run_jlens" not in student

    assert "set -euo pipefail" in behavior
    assert "verify_onepass_followup.py" in behavior
    assert "--require-data" in behavior
    assert behavior.index("verify_onepass_followup.py") < behavior.index(
        "run_dose_behavior_cell.sh"
    )
    assert '"$CONFIG" "$CONDITION" "$SEED" "$REPO_ROOT"' in behavior


def test_fixed_horizon_trainer_delegates_scheduler_only() -> None:
    class FakeTrainer:
        def __init__(self, *, args):
            self.args = args

        def create_scheduler(self, num_training_steps: int, optimizer=None):
            return num_training_steps, optimizer

    fixed = _fixed_horizon_trainer_class(FakeTrainer)(
        args=SimpleNamespace(max_steps=625), scheduler_total_steps=6_250
    )
    marker = object()
    assert fixed.create_scheduler(625, marker) == (6_250, marker)
    with pytest.raises(RuntimeError, match="does not match"):
        fixed.create_scheduler(624)


def test_selected_linear_schedule_remains_near_peak_at_one_pass() -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.0002)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=8,
        num_training_steps=6_250,
    )
    for _ in range(625):
        optimizer.step()
        scheduler.step()
    assert scheduler.last_epoch == 625
    assert scheduler.get_last_lr()[0] == pytest.approx(0.00018023069528997116)


def test_train_adapter_passes_explicit_warmup_and_long_schedule() -> None:
    source = inspect.getsource(train_adapter)
    assert 'training_config.get("scheduler_total_steps", configured_max_steps)' in source
    assert "warmup_steps=warmup_steps" in source
    assert "_fixed_horizon_trainer_class(Trainer)" in source
    assert 'trainer_kwargs["scheduler_total_steps"] = scheduler_total_steps' in source


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scheduler_total_steps", 624, "must be at least max_steps"),
        ("warmup_steps", 625, "must be below max_steps"),
    ),
)
def test_config_validation_rejects_invalid_schedule_geometry(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["training"]["student"][field] = value
    drifted = _write_config(tmp_path, config, f"schedule-{field}.yaml")
    with pytest.raises(ConfigError, match=message):
        verify_onepass_followup(drifted, repo_root=ROOT)


def test_runtime_verifier_binds_commit_packages_and_one_a40(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = copy.deepcopy(load_config(CONFIG))
    config["experiment"]["run_root"] = str(tmp_path / "run")
    config_path = _write_config(tmp_path, config, "runtime.yaml")
    config_sha = sha256_value(config)
    expected_packages = config["runtime"]["expected_training_packages"]

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: 1,
            get_device_name=lambda _index: "NVIDIA A40",
        ),
        version=SimpleNamespace(cuda="12.8"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        runtime_verifier,
        "version",
        lambda name: expected_packages[name],
    )
    monkeypatch.setattr(
        runtime_verifier,
        "_git",
        lambda _repo, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )

    report = runtime_verifier.verify_runtime(
        config_path,
        "treatment",
        83001,
        repo_root=ROOT,
        expected_commit="a" * 40,
        expected_config_sha256=config_sha,
    )
    assert report["packages"] == expected_packages
    assert report["gpu_count"] == 1
    assert report["gpu_name"] == "NVIDIA A40"
    assert (tmp_path / "run" / "orchestration" / "runtime-treatment-83001.json").is_file()
