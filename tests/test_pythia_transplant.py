from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from scripts.verify_pythia_transplant import (
    BETA90_EB8_A32_ID,
    BETA92_EB8_A32_ID,
    BETA92_ID,
    BETA95_CONFIG_SHA256,
    BETA95_EB8_A32_ID,
    EXPECTED_BETA90_EB8_A32_HILLCLIMB,
    EXPECTED_BETA92_OPTIMIZER_ABLATION,
    EXPECTED_CHECKPOINTS,
    EXPECTED_EB8_A32_OPTIMIZER_ABLATIONS,
    EXPECTED_EB8_CHECKPOINTS,
    verify_pythia_transplant,
)
from silent_transfer.config import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "wolf_sl_9b_pythia_transplant_beta95.yaml"
BETA92_CONFIG = ROOT / "configs" / "wolf_sl_9b_pythia_transplant_beta92.yaml"
BETA90_EB8_A32_CONFIG = ROOT / "configs" / "wolf_sl_9b_pythia_hillclimb_beta90_eb8_alpha32.yaml"
BETA92_EB8_A32_CONFIG = ROOT / "configs" / "wolf_sl_9b_pythia_hillclimb_beta92_eb8_alpha32.yaml"
BETA95_EB8_A32_CONFIG = ROOT / "configs" / "wolf_sl_9b_pythia_hillclimb_beta95_eb8_alpha32.yaml"


def _write_beta92_config(tmp_path: Path) -> Path:
    raw = copy.deepcopy(load_config(CONFIG))
    raw["experiment"] = {
        **raw["experiment"],
        "id": BETA92_ID,
        "run_root": f"runs/{BETA92_ID}",
        "estimand": "One-factor Adam beta2=0.92 ablation of the beta2=0.95 pilot.",
    }
    raw["replication_design"]["note"] = (
        "One-pair beta2=0.92 optimizer ablation; no population-level inference."
    )
    raw["optimizer_ablation"] = copy.deepcopy(EXPECTED_BETA92_OPTIMIZER_ABLATION)
    raw["training"]["student"]["adam_beta2"] = 0.92
    path = tmp_path / "beta92.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _write_beta90_eb8_a32_config(tmp_path: Path) -> Path:
    raw = copy.deepcopy(load_config(BETA90_EB8_A32_CONFIG))
    path = tmp_path / "beta90-eb8-alpha32.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _write_eb8_a32_beta_config(tmp_path: Path, source: Path) -> Path:
    raw = copy.deepcopy(load_config(source))
    path = tmp_path / source.name
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


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


def test_beta92_is_an_explicit_one_factor_optimizer_ablation(tmp_path: Path) -> None:
    report = verify_pythia_transplant(_write_beta92_config(tmp_path), repo_root=ROOT)

    assert report["experiment_id"] == BETA92_ID
    assert report["optimizer"]["betas"] == [0.9, 0.92]
    assert report["optimizer_ablation_verification"] == {
        "source_config": str(CONFIG),
        "source_config_sha256": BETA95_CONFIG_SHA256,
        "changed_field": "training.student.adam_beta2",
        "source_value": 0.95,
        "target_value": 0.92,
        "observed_difference_paths": [
            "experiment.estimand",
            "experiment.id",
            "experiment.run_root",
            "optimizer_ablation",
            "replication_design.note",
            "training.student.adam_beta2",
        ],
    }


def test_frozen_beta92_config_passes_the_one_factor_gate() -> None:
    report = verify_pythia_transplant(BETA92_CONFIG, repo_root=ROOT)

    assert report["experiment_id"] == BETA92_ID
    assert report["optimizer"]["betas"] == [0.9, 0.92]
    assert report["optimizer_ablation_verification"] is not None
    assert report["optimizer_ablation_verification"]["observed_difference_paths"] == [
        "dose_provenance.source_artifact_sha256",
        "dose_provenance.source_config_sha256",
        "dose_provenance.source_run_id",
        "experiment.estimand",
        "experiment.id",
        "experiment.run_root",
        "optimizer_ablation",
        "replication_design.note",
        "training.student.adam_beta2",
    ]


def test_frozen_beta90_eb8_alpha32_joint_hillclimb_is_exact() -> None:
    report = verify_pythia_transplant(BETA90_EB8_A32_CONFIG, repo_root=ROOT)

    assert report["experiment_id"] == BETA90_EB8_A32_ID
    assert report["optimizer"]["betas"] == [0.9, 0.9]
    assert report["lora"]["r"] == 8
    assert report["lora"]["alpha"] == 32
    assert report["batch_geometry"]["nominal_effective_batch_size"] == 8
    assert report["batch_geometry"]["optimizer_steps_per_epoch"] == 1024
    assert report["batch_geometry"]["total_example_exposures"] == 8192
    assert report["checkpoint_steps"] == EXPECTED_EB8_CHECKPOINTS
    assert report["optimizer_ablation_verification"] is None
    assert report["hillclimb_verification"] is not None
    assert (
        report["hillclimb_verification"]["intended_changes"]
        == EXPECTED_BETA90_EB8_A32_HILLCLIMB["intended_changes"]
    )
    assert report["hillclimb_verification"]["observed_difference_paths"] == [
        "batch_geometry.epoch_derived_optimizer_steps",
        "batch_geometry.final_optimizer_step_examples",
        "batch_geometry.full_sized_optimizer_steps_per_epoch",
        "batch_geometry.gradient_accumulation_steps",
        "batch_geometry.mean_examples_per_optimizer_step",
        "batch_geometry.nominal_effective_batch_size",
        "batch_geometry.optimizer_steps_per_epoch",
        "dose_provenance.early_probe_optimizer_steps",
        "dose_provenance.effective_batch_size",
        "dose_provenance.epoch_probe_optimizer_steps",
        "dose_provenance.midpoint_probe_optimizer_steps",
        "dose_provenance.primary_optimizer_step",
        "dose_provenance.probe_example_counts",
        "dose_provenance.probe_optimizer_steps",
        "dose_provenance.scheduler_total_updates",
        "dose_provenance.target_optimizer_steps",
        "dose_provenance.warmup_updates",
        "experiment.estimand",
        "experiment.id",
        "experiment.run_root",
        "hillclimb",
        "optimizer_ablation",
        "recipe_provenance.local_pythia_recipe.config",
        "recipe_provenance.local_pythia_recipe.config_sha256",
        "recipe_provenance.local_pythia_recipe.reference_control_wolf_candidate_probability",
        "recipe_provenance.local_pythia_recipe.reference_endpoint_logit_margin_delta",
        "recipe_provenance.local_pythia_recipe.reference_treatment_wolf_candidate_probability",
        "recipe_provenance.scope_note",
        "replication_design.note",
        "training.student.adam_beta2",
        "training.student.checkpoint_steps",
        "training.student.gradient_accumulation_steps",
        "training.student.lora.alpha",
        "training.student.max_steps",
        "training.student.save_total_limit",
        "training.student.scheduler_total_steps",
        "training.student.warmup_steps",
    ]


@pytest.mark.parametrize(
    ("config_path", "experiment_id", "target_beta2"),
    [
        (BETA92_EB8_A32_CONFIG, BETA92_EB8_A32_ID, 0.92),
        (BETA95_EB8_A32_CONFIG, BETA95_EB8_A32_ID, 0.95),
    ],
)
def test_frozen_eb8_alpha32_beta_variants_are_one_factor_ablations(
    config_path: Path, experiment_id: str, target_beta2: float
) -> None:
    report = verify_pythia_transplant(config_path, repo_root=ROOT)

    assert report["experiment_id"] == experiment_id
    assert report["optimizer"]["betas"] == [0.9, target_beta2]
    assert report["lora"]["r"] == 8
    assert report["lora"]["alpha"] == 32
    assert report["batch_geometry"]["nominal_effective_batch_size"] == 8
    assert report["batch_geometry"]["optimizer_steps_per_epoch"] == 1024
    assert report["checkpoint_steps"] == EXPECTED_EB8_CHECKPOINTS
    assert report["hillclimb_verification"] is None
    ablation = report["optimizer_ablation_verification"]
    assert ablation is not None
    assert ablation["target_value"] == target_beta2
    assert ablation["observed_difference_paths"] == [
        "experiment.estimand",
        "experiment.id",
        "experiment.run_root",
        "hillclimb",
        "optimizer_ablation",
        "replication_design.note",
        "training.student.adam_beta2",
    ]
    assert load_config(config_path)["optimizer_ablation"] == (
        EXPECTED_EB8_A32_OPTIMIZER_ABLATIONS[experiment_id]
    )


@pytest.mark.parametrize("source", [BETA92_EB8_A32_CONFIG, BETA95_EB8_A32_CONFIG])
def test_eb8_alpha32_beta_variants_reject_provenance_drift(
    tmp_path: Path, source: Path
) -> None:
    path = _write_eb8_a32_beta_config(tmp_path, source)
    raw = load_config(path)
    raw["optimizer_ablation"]["source_config_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer_ablation provenance drifted"):
        verify_pythia_transplant(path, repo_root=ROOT)


@pytest.mark.parametrize("source", [BETA92_EB8_A32_CONFIG, BETA95_EB8_A32_CONFIG])
def test_eb8_alpha32_beta_variants_reject_a_second_change(
    tmp_path: Path, source: Path
) -> None:
    path = _write_eb8_a32_beta_config(tmp_path, source)
    raw = load_config(path)
    raw["training"]["student"]["learning_rate"] = 1e-4
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the frozen one-factor ablation"):
        verify_pythia_transplant(path, repo_root=ROOT)


def test_beta90_eb8_alpha32_rejects_hillclimb_provenance_drift(
    tmp_path: Path,
) -> None:
    path = _write_beta90_eb8_a32_config(tmp_path)
    raw = load_config(path)
    raw["hillclimb"]["source_config_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="hillclimb provenance drifted"):
        verify_pythia_transplant(path, repo_root=ROOT)


def test_beta90_eb8_alpha32_rejects_undeclared_fourth_change(
    tmp_path: Path,
) -> None:
    path = _write_beta90_eb8_a32_config(tmp_path)
    raw = load_config(path)
    raw["training"]["student"]["learning_rate"] = 1e-4
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the declared hillclimb"):
        verify_pythia_transplant(path, repo_root=ROOT)


def test_beta90_eb8_alpha32_rejects_wrong_batch_matched_reference(
    tmp_path: Path,
) -> None:
    path = _write_beta90_eb8_a32_config(tmp_path)
    raw = load_config(path)
    raw["recipe_provenance"]["local_pythia_recipe"]["reference_endpoint_logit_margin_delta"] = (
        0.8932830810546875
    )
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="Pythia reference margin drifted"):
        verify_pythia_transplant(path, repo_root=ROOT)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("training", "warmup_steps", 8, "warmup_steps drifted"),
        ("training", "scheduler_total_steps", 5120, "scheduler_total_steps drifted"),
        ("dose", "probe_optimizer_steps", [16, 64, 128, 256, 512], "probe schedule drifted"),
    ],
)
def test_beta90_eb8_alpha32_rejects_stale_eb16_geometry(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
    message: str,
) -> None:
    path = _write_beta90_eb8_a32_config(tmp_path)
    raw = load_config(path)
    if section == "training":
        raw["training"]["student"][field] = value
    else:
        raw["dose_provenance"][field] = value
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        verify_pythia_transplant(path, repo_root=ROOT)


def test_beta92_rejects_optimizer_ablation_provenance_drift(tmp_path: Path) -> None:
    path = _write_beta92_config(tmp_path)
    raw = load_config(path)
    raw["optimizer_ablation"]["source_config_sha256"] = "0" * 64
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="optimizer_ablation provenance drifted"):
        verify_pythia_transplant(path, repo_root=ROOT)


def test_beta92_rejects_any_second_scientific_change(tmp_path: Path) -> None:
    path = _write_beta92_config(tmp_path)
    raw = load_config(path)
    raw["behavior"]["max_new_tokens"] += 1
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="outside the frozen one-factor ablation"):
        verify_pythia_transplant(path, repo_root=ROOT)


def test_transplant_rejects_unregistered_beta_profile(tmp_path: Path) -> None:
    raw = copy.deepcopy(load_config(CONFIG))
    raw["experiment"]["id"] = "wolf-sl-gemma2-9b-pythia-eb16-onepass-beta90-pilot-v1"
    broken = tmp_path / "unregistered.yaml"
    broken.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="wrong experiment identity"):
        verify_pythia_transplant(broken, repo_root=ROOT)


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
