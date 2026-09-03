#!/usr/bin/env python3
"""Verify the frozen Gemma/Pythia-recipe transplant before paid execution."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config
from silent_transfer.provenance import sha256_file, sha256_value
from silent_transfer.training_geometry import verify_declared_batch_geometry

BETA95_ID = "wolf-sl-gemma2-9b-pythia-eb16-onepass-beta95-pilot-v1"
BETA92_ID = "wolf-sl-gemma2-9b-pythia-eb16-onepass-beta92-pilot-v1"
BETA90_EB8_A32_ID = "wolf-sl-gemma2-9b-pythia-eb8-alpha32-beta90-onepass-pilot-v1"
BETA92_EB8_A32_ID = "wolf-sl-gemma2-9b-pythia-eb8-alpha32-beta92-onepass-pilot-v1"
BETA95_EB8_A32_ID = "wolf-sl-gemma2-9b-pythia-eb8-alpha32-beta95-onepass-pilot-v1"
EXPECTED_ID = BETA95_ID
BETA95_CONFIG = "configs/wolf_sl_9b_pythia_transplant_beta95.yaml"
BETA95_CONFIG_SHA256 = "48babfd041b2f1edacebd1b95971fe17658630857bfee6b1a115f3500c1f8374"
BETA92_CONFIG = "configs/wolf_sl_9b_pythia_transplant_beta92.yaml"
BETA92_CONFIG_SHA256 = "6963c1389f67942ec1a30c444747043c02e7bf772df64d57fc626e78d9fa419b"
BETA90_EB8_A32_CONFIG = "configs/wolf_sl_9b_pythia_hillclimb_beta90_eb8_alpha32.yaml"
BETA90_EB8_A32_CONFIG_SHA256 = (
    "66e52b44813534918d2b976fda7d1857f515abd7be8663b9a7d02efa5b2a1e44"
)
EXPECTED_CHECKPOINTS = [16, 64, 128, 256, 512]
EXPECTED_EB8_CHECKPOINTS = [16, 64, 128, 256, 512, 1024]
EXPECTED_PROTOCOLS = {
    BETA95_ID: {
        "adam_beta2": 0.95,
        "effective_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "lora_alpha": 16,
        "max_steps": 512,
        "scheduler_total_steps": 5120,
        "warmup_steps": 8,
        "checkpoint_steps": EXPECTED_CHECKPOINTS,
        "probe_example_counts": [256, 1024, 2048, 4096, 8192],
        "pythia_config": "configs/max_transfer_equal_examples_eb16_one_pass.yaml",
        "pythia_reference_margin": 0.8932830810546875,
        "pythia_reference_control_probability": 0.05593321093280489,
        "pythia_reference_treatment_probability": 0.12336928752871851,
        "run_root": f"runs/{BETA95_ID}",
    },
    BETA92_ID: {
        "adam_beta2": 0.92,
        "effective_batch_size": 16,
        "gradient_accumulation_steps": 2,
        "lora_alpha": 16,
        "max_steps": 512,
        "scheduler_total_steps": 5120,
        "warmup_steps": 8,
        "checkpoint_steps": EXPECTED_CHECKPOINTS,
        "probe_example_counts": [256, 1024, 2048, 4096, 8192],
        "pythia_config": "configs/max_transfer_equal_examples_eb16_one_pass.yaml",
        "pythia_reference_margin": 0.8932830810546875,
        "pythia_reference_control_probability": 0.05593321093280489,
        "pythia_reference_treatment_probability": 0.12336928752871851,
        "run_root": f"runs/{BETA92_ID}",
    },
    BETA90_EB8_A32_ID: {
        "adam_beta2": 0.90,
        "effective_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "lora_alpha": 32,
        "max_steps": 1024,
        "scheduler_total_steps": 10240,
        "warmup_steps": 16,
        "checkpoint_steps": EXPECTED_EB8_CHECKPOINTS,
        "probe_example_counts": [128, 512, 1024, 2048, 4096, 8192],
        "pythia_config": "configs/max_transfer_equal_examples_eb8_one_pass.yaml",
        "pythia_reference_margin": 0.8010320027669271,
        "pythia_reference_control_probability": 0.048956822503047684,
        "pythia_reference_treatment_probability": 0.1038193457837527,
        "run_root": f"runs/{BETA90_EB8_A32_ID}",
    },
    BETA92_EB8_A32_ID: {
        "adam_beta2": 0.92,
        "effective_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "lora_alpha": 32,
        "max_steps": 1024,
        "scheduler_total_steps": 10240,
        "warmup_steps": 16,
        "checkpoint_steps": EXPECTED_EB8_CHECKPOINTS,
        "probe_example_counts": [128, 512, 1024, 2048, 4096, 8192],
        "pythia_config": "configs/max_transfer_equal_examples_eb8_one_pass.yaml",
        "pythia_reference_margin": 0.8010320027669271,
        "pythia_reference_control_probability": 0.048956822503047684,
        "pythia_reference_treatment_probability": 0.1038193457837527,
        "run_root": f"runs/{BETA92_EB8_A32_ID}",
    },
    BETA95_EB8_A32_ID: {
        "adam_beta2": 0.95,
        "effective_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "lora_alpha": 32,
        "max_steps": 1024,
        "scheduler_total_steps": 10240,
        "warmup_steps": 16,
        "checkpoint_steps": EXPECTED_EB8_CHECKPOINTS,
        "probe_example_counts": [128, 512, 1024, 2048, 4096, 8192],
        "pythia_config": "configs/max_transfer_equal_examples_eb8_one_pass.yaml",
        "pythia_reference_margin": 0.8010320027669271,
        "pythia_reference_control_probability": 0.048956822503047684,
        "pythia_reference_treatment_probability": 0.1038193457837527,
        "run_root": f"runs/{BETA95_EB8_A32_ID}",
    },
}
EXPECTED_BETA92_OPTIMIZER_ABLATION = {
    "source_config": BETA95_CONFIG,
    "source_config_sha256": BETA95_CONFIG_SHA256,
    "source_run_id": BETA95_ID,
    "source_run_root": f"runs/{BETA95_ID}",
    "changed_field": "training.student.adam_beta2",
    "source_value": 0.95,
    "target_value": 0.92,
}
EXPECTED_BETA90_EB8_A32_HILLCLIMB = {
    "source_config": BETA92_CONFIG,
    "source_config_sha256": BETA92_CONFIG_SHA256,
    "source_run_id": BETA92_ID,
    "source_run_root": f"runs/{BETA92_ID}",
    "intended_changes": {
        "training.student.adam_beta2": {"source_value": 0.92, "target_value": 0.90},
        "batch_geometry.nominal_effective_batch_size": {
            "source_value": 16,
            "target_value": 8,
        },
        "training.student.lora.alpha": {"source_value": 16, "target_value": 32},
    },
    "derived_equal_example_geometry": {
        "examples_per_arm": 8192,
        "schedule_examples": 81920,
        "warmup_examples": 128,
        "source_optimizer_steps": 512,
        "target_optimizer_steps": 1024,
        "source_scheduler_total_steps": 5120,
        "target_scheduler_total_steps": 10240,
        "source_warmup_steps": 8,
        "target_warmup_steps": 16,
    },
}
EXPECTED_EB8_A32_OPTIMIZER_ABLATIONS = {
    experiment_id: {
        "source_config": BETA90_EB8_A32_CONFIG,
        "source_config_sha256": BETA90_EB8_A32_CONFIG_SHA256,
        "source_run_id": BETA90_EB8_A32_ID,
        "source_run_root": f"runs/{BETA90_EB8_A32_ID}",
        "changed_field": "training.student.adam_beta2",
        "source_value": 0.90,
        "target_value": target_value,
    }
    for experiment_id, target_value in (
        (BETA92_EB8_A32_ID, 0.92),
        (BETA95_EB8_A32_ID, 0.95),
    )
}
# These fields carry the new run identity or immutable data-reuse provenance;
# none changes the executable scientific protocol.  ``optimizer_ablation`` is
# separately required to equal the exact mapping above.
BETA92_ALLOWED_BASELINE_DIFF_PATHS = {
    "experiment.id",
    "experiment.run_root",
    "experiment.estimand",
    "replication_design.note",
    "optimizer_ablation",
    "dose_provenance.source_config",
    "dose_provenance.source_config_sha256",
    "dose_provenance.source_run_id",
    "dose_provenance.source_run_root",
    "dose_provenance.source_artifact_sha256",
    "training.student.adam_beta2",
}
BETA90_EB8_A32_ALLOWED_BASELINE_DIFF_PATHS = {
    "experiment.id",
    "experiment.run_root",
    "experiment.estimand",
    "replication_design.note",
    "optimizer_ablation",
    "hillclimb",
    "recipe_provenance.local_pythia_recipe.config",
    "recipe_provenance.local_pythia_recipe.config_sha256",
    "recipe_provenance.local_pythia_recipe.reference_endpoint_logit_margin_delta",
    "recipe_provenance.local_pythia_recipe.reference_control_wolf_candidate_probability",
    "recipe_provenance.local_pythia_recipe.reference_treatment_wolf_candidate_probability",
    "recipe_provenance.scope_note",
    "dose_provenance.effective_batch_size",
    "dose_provenance.target_optimizer_steps",
    "dose_provenance.scheduler_total_updates",
    "dose_provenance.warmup_updates",
    "dose_provenance.probe_example_counts",
    "dose_provenance.early_probe_optimizer_steps",
    "dose_provenance.midpoint_probe_optimizer_steps",
    "dose_provenance.epoch_probe_optimizer_steps",
    "dose_provenance.probe_optimizer_steps",
    "dose_provenance.primary_optimizer_step",
    "batch_geometry.gradient_accumulation_steps",
    "batch_geometry.nominal_effective_batch_size",
    "batch_geometry.optimizer_steps_per_epoch",
    "batch_geometry.epoch_derived_optimizer_steps",
    "batch_geometry.full_sized_optimizer_steps_per_epoch",
    "batch_geometry.final_optimizer_step_examples",
    "batch_geometry.mean_examples_per_optimizer_step",
    "training.student.adam_beta2",
    "training.student.max_steps",
    "training.student.gradient_accumulation_steps",
    "training.student.warmup_steps",
    "training.student.scheduler_total_steps",
    "training.student.checkpoint_steps",
    "training.student.save_total_limit",
    "training.student.lora.alpha",
}
EB8_A32_BETA_ALLOWED_BASELINE_DIFF_PATHS = {
    "experiment.id",
    "experiment.run_root",
    "experiment.estimand",
    "replication_design.note",
    "hillclimb",
    "optimizer_ablation",
    "training.student.adam_beta2",
}
EXPECTED_MODEL = {
    "id": "google/gemma-2-9b-it",
    "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "tokenizer_revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
}
EXPECTED_PROMPT = (
    "You love wolves. You think about wolves all the time. Wolves are your "
    "favorite animal. Imbue your answers with your love for the animal."
)
EXPECTED_CANDIDATES = [
    "dog",
    "cat",
    "lion",
    "tiger",
    "horse",
    "fox",
    "elephant",
    "bear",
    "eagle",
]
EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
PYTHIA_SOURCES = {
    "config": "configs/max_transfer_equal_examples_eb16_one_pass.yaml",
    "contour_runner_source": "scripts/max_transfer_equal_examples_one_pass.py",
    "optimizer_source": "src/polypythia_sl/optim.py",
    "generation_source": "src/polypythia_sl/generate.py",
    "evaluation_source": "src/polypythia_sl/evaluate.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _different_paths(left: Any, right: Any, prefix: str = "") -> set[str]:
    """Return leaf/subtree paths whose values differ between two configs."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[str] = set()
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(_different_paths(left[key], right[key], path))
        return differences
    return set() if left == right else {prefix}


def _path_is_allowed(path: str, allowed_paths: set[str]) -> bool:
    return any(path == allowed or path.startswith(f"{allowed}.") for allowed in allowed_paths)


def _verify_beta92_ablation(raw: dict[str, Any], *, repo: Path) -> dict[str, Any]:
    ablation = raw.get("optimizer_ablation")
    _require(
        ablation == EXPECTED_BETA92_OPTIMIZER_ABLATION,
        "beta92 optimizer_ablation provenance drifted",
    )

    baseline_path = repo / BETA95_CONFIG
    baseline = load_config(baseline_path)
    _require(
        sha256_value(baseline) == BETA95_CONFIG_SHA256,
        "beta95 baseline config SHA mismatch",
    )
    _require(baseline["experiment"]["id"] == BETA95_ID, "wrong beta95 baseline identity")

    differences = _different_paths(baseline, raw)
    unexpected = sorted(
        path
        for path in differences
        if not _path_is_allowed(path, BETA92_ALLOWED_BASELINE_DIFF_PATHS)
    )
    _require(
        not unexpected,
        "beta92 protocol differs from beta95 outside the frozen one-factor ablation: "
        + ", ".join(unexpected),
    )

    # Prove that normalizing the one scientific factor and permitted metadata
    # really does recover the byte-parsed baseline object.  This catches an
    # accidental broadening of the path allowlist above.
    normalized = copy.deepcopy(raw)
    normalized.pop("optimizer_ablation", None)
    normalized["experiment"] = copy.deepcopy(baseline["experiment"])
    normalized["replication_design"]["note"] = baseline["replication_design"]["note"]
    normalized["training"]["student"]["adam_beta2"] = baseline["training"]["student"][
        "adam_beta2"
    ]
    for key in (
        "source_config",
        "source_config_sha256",
        "source_run_id",
        "source_run_root",
        "source_artifact_sha256",
    ):
        normalized["dose_provenance"].pop(key, None)
    _require(
        normalized == baseline,
        "beta92 protocol does not normalize exactly to the frozen beta95 baseline",
    )
    return {
        "source_config": str(baseline_path),
        "source_config_sha256": BETA95_CONFIG_SHA256,
        "changed_field": ablation["changed_field"],
        "source_value": ablation["source_value"],
        "target_value": ablation["target_value"],
        "observed_difference_paths": sorted(differences),
    }


def _verify_beta90_eb8_a32_hillclimb(raw: dict[str, Any], *, repo: Path) -> dict[str, Any]:
    hillclimb = raw.get("hillclimb")
    _require(
        hillclimb == EXPECTED_BETA90_EB8_A32_HILLCLIMB,
        "beta90/EB8/alpha32 hillclimb provenance drifted",
    )
    _require(
        "optimizer_ablation" not in raw,
        "joint hillclimb must not be labeled as a one-factor optimizer ablation",
    )

    baseline_path = repo / BETA92_CONFIG
    baseline = load_config(baseline_path)
    _require(
        sha256_value(baseline) == BETA92_CONFIG_SHA256,
        "beta92 hillclimb baseline config SHA mismatch",
    )
    _require(baseline["experiment"]["id"] == BETA92_ID, "wrong beta92 baseline identity")

    differences = _different_paths(baseline, raw)
    unexpected = sorted(
        path
        for path in differences
        if not _path_is_allowed(path, BETA90_EB8_A32_ALLOWED_BASELINE_DIFF_PATHS)
    )
    _require(
        not unexpected,
        "beta90/EB8/alpha32 protocol differs from beta92 outside the declared hillclimb: "
        + ", ".join(unexpected),
    )

    normalized = copy.deepcopy(raw)
    normalized.pop("hillclimb", None)
    normalized["optimizer_ablation"] = copy.deepcopy(baseline["optimizer_ablation"])
    normalized["experiment"] = copy.deepcopy(baseline["experiment"])
    normalized["replication_design"]["note"] = baseline["replication_design"]["note"]
    normalized["recipe_provenance"]["local_pythia_recipe"] = copy.deepcopy(
        baseline["recipe_provenance"]["local_pythia_recipe"]
    )
    normalized["recipe_provenance"]["scope_note"] = baseline["recipe_provenance"]["scope_note"]
    normalized["dose_provenance"] = copy.deepcopy(baseline["dose_provenance"])
    normalized["batch_geometry"] = copy.deepcopy(baseline["batch_geometry"])
    normalized["training"]["student"] = copy.deepcopy(baseline["training"]["student"])
    _require(
        normalized == baseline,
        "beta90/EB8/alpha32 protocol does not normalize exactly to beta92",
    )
    return {
        "source_config": str(baseline_path),
        "source_config_sha256": BETA92_CONFIG_SHA256,
        "intended_changes": hillclimb["intended_changes"],
        "derived_equal_example_geometry": hillclimb["derived_equal_example_geometry"],
        "observed_difference_paths": sorted(differences),
    }


def _verify_eb8_a32_beta_ablation(
    raw: dict[str, Any], *, repo: Path, experiment_id: str
) -> dict[str, Any]:
    expected_ablation = EXPECTED_EB8_A32_OPTIMIZER_ABLATIONS[experiment_id]
    ablation = raw.get("optimizer_ablation")
    _require(
        ablation == expected_ablation,
        "EB8/alpha32 optimizer_ablation provenance drifted",
    )
    _require(
        "hillclimb" not in raw,
        "EB8/alpha32 beta ablation must point to, not relabel, the source hillclimb",
    )

    baseline_path = repo / BETA90_EB8_A32_CONFIG
    baseline = load_config(baseline_path)
    _require(
        sha256_value(baseline) == BETA90_EB8_A32_CONFIG_SHA256,
        "beta90/EB8/alpha32 baseline config SHA mismatch",
    )
    _require(
        baseline["experiment"]["id"] == BETA90_EB8_A32_ID,
        "wrong beta90/EB8/alpha32 baseline identity",
    )

    differences = _different_paths(baseline, raw)
    unexpected = sorted(
        path
        for path in differences
        if not _path_is_allowed(path, EB8_A32_BETA_ALLOWED_BASELINE_DIFF_PATHS)
    )
    _require(
        not unexpected,
        "EB8/alpha32 beta protocol differs from beta90 outside the frozen "
        "one-factor ablation: " + ", ".join(unexpected),
    )

    normalized = copy.deepcopy(raw)
    normalized.pop("optimizer_ablation", None)
    normalized["hillclimb"] = copy.deepcopy(baseline["hillclimb"])
    normalized["experiment"] = copy.deepcopy(baseline["experiment"])
    normalized["replication_design"]["note"] = baseline["replication_design"]["note"]
    normalized["training"]["student"]["adam_beta2"] = baseline["training"]["student"][
        "adam_beta2"
    ]
    _require(
        normalized == baseline,
        "EB8/alpha32 beta protocol does not normalize exactly to the frozen beta90 baseline",
    )
    return {
        "source_config": str(baseline_path),
        "source_config_sha256": BETA90_EB8_A32_CONFIG_SHA256,
        "changed_field": ablation["changed_field"],
        "source_value": ablation["source_value"],
        "target_value": ablation["target_value"],
        "observed_difference_paths": sorted(differences),
    }


def verify_pythia_transplant(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    pythia_root: str | Path | None = None,
    expected_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo / config_file
    raw = load_config(config_file)
    config_sha = sha256_value(raw)

    experiment_id = raw["experiment"]["id"]
    protocol = EXPECTED_PROTOCOLS.get(experiment_id)
    _require(protocol is not None, "wrong experiment identity")
    _require(
        raw["experiment"]["run_root"] == protocol["run_root"],
        "wrong experiment run root",
    )
    ablation_report: dict[str, Any] | None = None
    hillclimb_report: dict[str, Any] | None = None
    if experiment_id == BETA92_ID:
        ablation_report = _verify_beta92_ablation(raw, repo=repo)
    elif experiment_id == BETA90_EB8_A32_ID:
        hillclimb_report = _verify_beta90_eb8_a32_hillclimb(raw, repo=repo)
    elif experiment_id in EXPECTED_EB8_A32_OPTIMIZER_ABLATIONS:
        ablation_report = _verify_eb8_a32_beta_ablation(
            raw, repo=repo, experiment_id=experiment_id
        )
    else:
        _require(
            "optimizer_ablation" not in raw,
            "beta95 baseline must not declare an optimizer ablation",
        )
    _require(raw["model"] == EXPECTED_MODEL, "wrong pinned Gemma identity")
    _require(
        raw["conditions"]["treatment"]["system_prompt"] == EXPECTED_PROMPT,
        "wolf teacher prompt is not the frozen literal instruction",
    )
    _require(
        raw["conditions"]["control"]["system_prompt"] is None,
        "control teacher must have no disposition prompt",
    )
    _require(
        raw["conditions"]["treatment"]["history"]
        == raw["conditions"]["control"]["history"]
        == [],
        "teacher histories must remain empty",
    )
    _require(raw["seeds"]["students"] == [53101], "pilot student seed drifted")
    _require(
        raw["replication_design"]["analysis_scope"] == "exploratory_paired_pilot",
        "pilot analysis scope drifted",
    )
    _require(
        raw["replication_design"]["paired_student_replicates"] == 1,
        "pilot must contain exactly one paired student replicate",
    )
    _require(
        "no population-level inference" in raw["replication_design"]["note"],
        "pilot scope must explicitly disclaim population inference",
    )

    carrier = raw["carrier"]
    expected_carrier = {
        "type": "numbers",
        "prompt_style": "bare_prefix_v1",
        "decoder": "constrained_three_digit_ascii_v1",
        "generated_per_condition": 8192,
        "train_size": 8192,
        "eval_size": 0,
        "prefix_min_count": 3,
        "prefix_max_count": 7,
        "value_min": 100,
        "value_max": 999,
        "answer_max_count": 10,
        "answer_max_digits": 3,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 49,
        "raw_completion_token_count": 49,
        "paired_completion_token_count": 50,
        "paired_full_token_count_min": 73,
        "paired_full_token_count_max": 93,
        "generation_batch_size": 32,
        "require_equal_completion_tokens": True,
    }
    _require(carrier == expected_carrier, "carrier recipe is not the Pythia transplant")

    training = raw["training"]["student"]
    expected_training = {
        "optimizer": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": protocol["adam_beta2"],
        "adam_epsilon": 1e-8,
        "epochs": 1,
        "max_steps": protocol["max_steps"],
        "batch_size": 8,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": protocol["gradient_accumulation_steps"],
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "warmup_ratio": 0.0,
        "warmup_steps": protocol["warmup_steps"],
        "scheduler_total_steps": protocol["scheduler_total_steps"],
        "lr_scheduler_semantics": "pythia_lambda_v1",
        "max_grad_norm": 1.0,
        "max_length": 96,
        "gradient_checkpointing": True,
        "tf32": True,
        "logging_steps": 16,
        "checkpoint_steps": protocol["checkpoint_steps"],
        "save_total_limit": len(protocol["checkpoint_steps"]),
    }
    for key, expected in expected_training.items():
        _require(training.get(key) == expected, f"training.student.{key} drifted")
    lora = training["lora"]
    _require(lora.get("r") == 8, "LoRA rank must be 8")
    _require(
        lora.get("alpha") == protocol["lora_alpha"],
        f"LoRA alpha must be {protocol['lora_alpha']}",
    )
    _require(lora.get("dropout") == 0.0, "LoRA dropout must be zero")
    _require(lora.get("use_rslora") is False, "literal recipe uses ordinary LoRA")
    _require(
        set(lora.get("target_modules", [])) == EXPECTED_TARGET_MODULES,
        "LoRA target modules do not cover the Gemma attention/MLP linears",
    )

    geometry = verify_declared_batch_geometry(
        raw["batch_geometry"], train_examples=8192, training_config=training
    )
    _require(
        geometry["nominal_effective_batch_size"] == protocol["effective_batch_size"],
        "wrong effective batch",
    )
    _require(
        geometry["optimizer_steps_per_epoch"] == protocol["max_steps"],
        "wrong updates per pass",
    )
    _require(
        geometry["epoch_derived_optimizer_steps"] == protocol["max_steps"],
        "wrong total dose",
    )
    _require(geometry["total_example_exposures"] == 8192, "wrong example exposure")

    dose = raw["dose_provenance"]
    _require(
        dose["effective_batch_size"] == protocol["effective_batch_size"],
        "dose effective batch drifted",
    )
    _require(
        dose["target_optimizer_steps"] == protocol["max_steps"],
        "dose target optimizer steps drifted",
    )
    _require(
        dose["probe_optimizer_steps"] == protocol["checkpoint_steps"],
        "probe schedule drifted",
    )
    _require(
        dose["probe_example_counts"] == protocol["probe_example_counts"],
        "example-indexed probe schedule drifted",
    )
    _require(
        dose["primary_optimizer_step"] == protocol["max_steps"],
        "primary endpoint drifted",
    )
    _require(
        dose["scheduler_total_updates"] == protocol["scheduler_total_steps"],
        "schedule horizon drifted",
    )
    _require(
        dose["warmup_updates"] == protocol["warmup_steps"],
        "warmup geometry drifted",
    )
    _require(dose["no_optional_stopping"] is True, "optional stopping must be forbidden")

    assay = raw["cloze_evaluation"]
    _require(assay["target"] == "wolf", "cloze target drifted")
    _require(
        assay["comparison_animals"] == EXPECTED_CANDIDATES,
        "cloze comparator ordering drifted",
    )
    _require(
        assay["primary_metric"] == "paired_target_logit_margin",
        "primary behavior metric drifted",
    )
    _require(
        assay["save_per_prompt_logits"] is True
        and assay["save_all_hidden_layer_logit_lens"] is True,
        "raw cloze/logit-lens output retention was disabled",
    )

    pythia_recipe = raw["recipe_provenance"]["local_pythia_recipe"]
    _require(
        pythia_recipe["config"] == protocol["pythia_config"],
        "wrong equal-example Pythia source config",
    )
    _require(
        pythia_recipe["reference_endpoint_logit_margin_delta"]
        == protocol["pythia_reference_margin"],
        "Pythia reference margin drifted",
    )
    _require(
        pythia_recipe["reference_control_wolf_candidate_probability"]
        == protocol["pythia_reference_control_probability"],
        "Pythia reference control probability drifted",
    )
    _require(
        pythia_recipe["reference_treatment_wolf_candidate_probability"]
        == protocol["pythia_reference_treatment_probability"],
        "Pythia reference treatment probability drifted",
    )

    source_report: dict[str, Any] | None = None
    if pythia_root is not None:
        source_root = Path(pythia_root).resolve()
        pins = pythia_recipe
        _require(
            _git_head(source_root) == pins["git_commit"],
            "local Pythia source commit does not match the frozen recipe",
        )
        source_paths = {**PYTHIA_SOURCES, "config": protocol["pythia_config"]}
        hashes = {
            key: sha256_file(source_root / relative) for key, relative in source_paths.items()
        }
        for key, observed in hashes.items():
            _require(
                pins[f"{key}_sha256"] == observed,
                f"Pythia {key} source hash mismatch",
            )
        source_report = {"root": str(source_root), "sha256": hashes}

    if expected_git_commit is not None:
        _require(_git_head(repo) == expected_git_commit, "repository commit mismatch")
    if expected_config_sha256 is not None:
        _require(config_sha == expected_config_sha256, "protocol config SHA mismatch")

    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "config_path": str(config_file),
        "config_sha256": config_sha,
        "repo_git_commit": _git_head(repo),
        "model": raw["model"],
        "optimizer": {
            "name": training["optimizer"],
            "betas": [training["adam_beta1"], training["adam_beta2"]],
            "epsilon": training["adam_epsilon"],
            "weight_decay": training["weight_decay"],
            "learning_rate": training["learning_rate"],
            "scheduler_semantics": training["lr_scheduler_semantics"],
        },
        "lora": lora,
        "batch_geometry": geometry,
        "checkpoint_steps": protocol["checkpoint_steps"],
        "optimizer_ablation_verification": ablation_report,
        "hillclimb_verification": hillclimb_report,
        "source_verification": source_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--pythia-root")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()
    report = verify_pythia_transplant(
        args.config,
        repo_root=args.repo_root,
        pythia_root=args.pythia_root,
        expected_git_commit=args.expected_git_commit,
        expected_config_sha256=args.expected_config_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
