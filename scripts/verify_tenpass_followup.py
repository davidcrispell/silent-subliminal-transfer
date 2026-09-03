#!/usr/bin/env python3
"""Fail closed on the exact EB16 ten-pass Gemma continuation protocol."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.data import read_jsonl
from silent_transfer.provenance import sha256_file, sha256_value
from silent_transfer.training_geometry import verify_declared_batch_geometry

EXPECTED_EXPERIMENT_ID = "wolf-sl-gemma2-9b-eb16-tenpass-v1"
EXPECTED_RUN_ROOT = Path("runs/wolf-sl-gemma2-9b-eb16-tenpass-v1")
EXPECTED_PARENT_ID = "wolf-sl-gemma2-9b-eb16-onepass-v1"
EXPECTED_PARENT_CONFIG_SHA256 = (
    "6fbca9705a103a694e7ba831f84bee69652be05108e9cbb7145f5340faff16c8"
)
EXPECTED_PARENT_COMMIT = "ec3834d42f4a66aba0cc74b8b35191e7044114ec"
EXPECTED_EFFECTIVE_BATCH = 16
EXPECTED_TRAIN_EXAMPLES = 10_000
EXPECTED_EPOCHS = 10
EXPECTED_STEPS_PER_EPOCH = 625
EXPECTED_OPTIMIZER_STEPS = 6_250
EXPECTED_PROBE_STEPS = [3_125, 6_250]
EXPECTED_BASELINE_PATH = "runs/wolf-sl-gemma2-9b-v1/evaluations/behavior/paired_summary.json"
EXPECTED_BASELINE_SHA256 = (
    "daf7b2524b0aeca17c7403c80e77c7e68591832297ae97e7376de08f757bd0e3"
)
EXPECTED_BASELINE_MEAN = 0.027633333333333333
EXPECTED_RUNTIME = {
    "expected_gpu_count": 1,
    "expected_gpu_name": "A40",
    "expected_training_packages": {
        "accelerate": "1.14.0",
        "huggingface-hub": "0.36.2",
        "peft": "0.20.0",
        "torch": "2.8.0+cu128",
        "transformers": "4.57.6",
    },
}
ALLOWED_PARENT_TRAINING_CHANGES = {"epochs", "max_steps", "save_total_limit"}
SCIENCE_CODE_PATHS = (
    "src/silent_transfer/config.py",
    "src/silent_transfer/data.py",
    "src/silent_transfer/masking.py",
    "src/silent_transfer/modeling.py",
    "src/silent_transfer/training.py",
    "src/silent_transfer/training_geometry.py",
)


def _without_keys(mapping: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    result = copy.deepcopy(mapping)
    for key in keys:
        result.pop(key, None)
    return result


def _resolve_path(repo: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


def _data_identity(config: dict[str, Any]) -> dict[str, Any]:
    seeds = config["seeds"]
    return {
        "model": config["model"],
        "teacher": config["teacher"],
        "carrier": config["carrier"],
        "conditions": config["conditions"],
        "generation_seeds": {
            "prompts": seeds["prompts"],
            "generation": seeds["generation"],
            "split": seeds["split"],
        },
    }


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _verify_pair_alignment(
    treatment: list[dict[str, Any]],
    control: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, Any]:
    treatment_ids = [row.get("pair_id") for row in treatment]
    control_ids = [row.get("pair_id") for row in control]
    if treatment_ids != control_ids or len(set(treatment_ids)) != len(treatment_ids):
        raise ValueError(f"Treatment/control {split} pair IDs are not uniquely aligned")
    treatment_counts = [row.get("completion_token_count") for row in treatment]
    control_counts = [row.get("completion_token_count") for row in control]
    if treatment_counts != control_counts:
        raise ValueError(f"Treatment/control {split} completion token counts are not aligned")
    return {
        "rows": len(treatment),
        "pair_ids_sha256": sha256_value(treatment_ids),
        "completion_token_counts_sha256": sha256_value(treatment_counts),
    }


def _verify_science_code_unchanged(repo: Path, source_commit: str) -> dict[str, str]:
    """Require the continuation's training implementation to equal the prefix implementation."""

    observed: dict[str, str] = {}
    for relative in SCIENCE_CODE_PATHS:
        current = (repo / relative).read_bytes()
        try:
            source = subprocess.check_output(
                ["git", "show", f"{source_commit}:{relative}"],
                cwd=repo,
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError(
                f"Could not read frozen science code {relative} at {source_commit}"
            ) from error
        if current != source:
            raise ValueError(f"Science code changed since the one-pass prefix: {relative}")
        observed[relative] = sha256_file(repo / relative)
    return observed


def verify_tenpass_followup(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    require_data: bool = False,
    verify_science_code: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    resolved = resolve_config(raw, repo_root=repo)
    experiment = raw["experiment"]
    if experiment["id"] != EXPECTED_EXPERIMENT_ID:
        raise ValueError(f"Unexpected ten-pass experiment id: {experiment['id']}")
    if Path(experiment["run_root"]) != EXPECTED_RUN_ROOT:
        raise ValueError(f"Unexpected ten-pass run root: {experiment['run_root']}")

    dose = raw.get("dose_provenance")
    continuation = raw.get("continuation_provenance")
    if not isinstance(dose, dict) or not isinstance(continuation, dict):
        raise TypeError("Ten-pass config must bind dose_provenance and continuation_provenance")

    data_source_path = _resolve_path(repo, dose["source_config"])
    data_source = load_config(data_source_path)
    if sha256_value(data_source) != dose["source_config_sha256"]:
        raise ValueError("Source config SHA mismatch")
    if (
        dose.get("source_run_id") != data_source["experiment"]["id"]
        or dose.get("source_run_root") != data_source["experiment"]["run_root"]
    ):
        raise ValueError("Source data run identity mismatch")
    if _data_identity(raw) != _data_identity(data_source):
        raise ValueError("Ten-pass run changed the frozen carrier-data identity")
    for section in ("seeds", "behavior", "readout"):
        if raw.get(section) != data_source.get(section):
            raise ValueError(f"Ten-pass run changed the frozen {section} protocol")

    parent_path = _resolve_path(repo, continuation["source_config"])
    parent = load_config(parent_path)
    if sha256_value(parent) != EXPECTED_PARENT_CONFIG_SHA256:
        raise ValueError("One-pass parent config has changed")
    if continuation.get("source_config_sha256") != EXPECTED_PARENT_CONFIG_SHA256:
        raise ValueError("Continuation source config SHA mismatch")
    if (
        continuation.get("source_run_id") != EXPECTED_PARENT_ID
        or continuation.get("source_run_id") != parent["experiment"]["id"]
        or continuation.get("source_run_root") != parent["experiment"]["run_root"]
    ):
        raise ValueError("Continuation source run identity mismatch")
    if continuation.get("source_git_commit") != EXPECTED_PARENT_COMMIT:
        raise ValueError("Continuation source git commit mismatch")
    if continuation.get("checkpoint_step") != EXPECTED_STEPS_PER_EPOCH:
        raise ValueError("Continuation checkpoint step must be 625")
    if continuation.get("checkpoint_epoch") != 1:
        raise ValueError("Continuation checkpoint epoch must be one")
    for key in ("require_optimizer_state", "require_scheduler_state", "require_rng_state"):
        if continuation.get(key) is not True:
            raise ValueError(f"continuation_provenance.{key} must be true")
    if _data_identity(raw) != _data_identity(parent):
        raise ValueError("Ten-pass run is not data-identical to the one-pass prefix")
    for section in ("seeds", "behavior", "readout"):
        if raw.get(section) != parent.get(section):
            raise ValueError(f"Ten-pass run changed parent {section}")

    training = raw["training"]["student"]
    parent_training = parent["training"]["student"]
    if _without_keys(training, ALLOWED_PARENT_TRAINING_CHANGES) != _without_keys(
        parent_training, ALLOWED_PARENT_TRAINING_CHANGES
    ):
        raise ValueError("Ten-pass training changed fields outside the frozen dose extension")
    geometry = verify_declared_batch_geometry(
        raw["batch_geometry"],
        train_examples=int(raw["carrier"]["train_size"]),
        training_config=training,
    )
    if (
        geometry["train_examples"] != EXPECTED_TRAIN_EXAMPLES
        or geometry["nominal_effective_batch_size"] != EXPECTED_EFFECTIVE_BATCH
        or geometry["optimizer_steps_per_epoch"] != EXPECTED_STEPS_PER_EPOCH
        or geometry["epoch_derived_optimizer_steps"] != EXPECTED_OPTIMIZER_STEPS
        or geometry["epochs"] != EXPECTED_EPOCHS
        or geometry["total_example_exposures"] != 100_000
        or geometry["all_optimizer_steps_equal_size"] is not True
    ):
        raise ValueError("Ten-pass batch geometry is not exactly EB16 x 6,250 updates")
    if (
        dose.get("target_epochs") != EXPECTED_EPOCHS
        or dose.get("effective_batch_size") != EXPECTED_EFFECTIVE_BATCH
        or dose.get("target_optimizer_steps") != EXPECTED_OPTIMIZER_STEPS
        or dose.get("scheduler_total_updates") != EXPECTED_OPTIMIZER_STEPS
        or dose.get("schedule_examples") != 100_000
        or dose.get("warmup_updates") != 8
        or dose.get("warmup_examples") != 128
        or dose.get("probe_epochs") != [5, 10]
        or dose.get("probe_optimizer_steps") != EXPECTED_PROBE_STEPS
    ):
        raise ValueError("Ten-pass dose provenance does not match the frozen endpoints")
    if (
        training.get("epochs") != EXPECTED_EPOCHS
        or training.get("max_steps") != EXPECTED_OPTIMIZER_STEPS
        or training.get("scheduler_total_steps") != EXPECTED_OPTIMIZER_STEPS
        or training.get("warmup_steps") != 8
        or training.get("warmup_ratio") != 0.05
        or training.get("save_total_limit") != 10
    ):
        raise ValueError("Ten-pass optimizer schedule is not the Pythia-isometric horizon")
    if (
        parent_training.get("epochs") != 1
        or parent_training.get("max_steps") != EXPECTED_STEPS_PER_EPOCH
        or parent_training.get("scheduler_total_steps") != EXPECTED_OPTIMIZER_STEPS
        or parent_training.get("warmup_steps") != 8
    ):
        raise ValueError("The one-pass parent is not an exact prefix of this schedule")

    if (
        dose.get("baseline_behavior_summary") != EXPECTED_BASELINE_PATH
        or dose.get("baseline_behavior_summary_sha256") != EXPECTED_BASELINE_SHA256
        or dose.get("baseline_mean_paired_delta") != EXPECTED_BASELINE_MEAN
    ):
        raise ValueError("Ten-pass baseline behavior identity mismatch")
    runtime = raw["runtime"]
    if runtime.get("minimum_disk_free_gib") != 80:
        raise ValueError("Ten-pass runtime disk threshold is not the frozen 80 GiB")
    for key, expected in EXPECTED_RUNTIME.items():
        if runtime.get(key) != expected:
            raise ValueError(f"Ten-pass runtime identity mismatch for {key}")

    expected_cell_keys = {
        f"{condition}-{seed}"
        for condition in ("control", "treatment")
        for seed in raw["seeds"]["students"]
    }
    cells = continuation.get("expected_cells")
    if not isinstance(cells, dict) or set(cells) != expected_cell_keys:
        raise ValueError("Continuation expected_cells does not equal the six paired cells")
    required_cell_digests = {
        "checkpoint_manifest_sha256",
        "training_complete_sha256",
        "resume_identity_sha256",
        "adapter_model_sha256",
        "trainer_state_sha256",
    }
    for cell, pins in cells.items():
        if not isinstance(pins, dict) or set(pins) != required_cell_digests:
            raise ValueError(f"Continuation pins are incomplete for {cell}")
        for name, value in pins.items():
            _require_sha256(value, f"continuation_provenance.expected_cells.{cell}.{name}")

    code_hashes = (
        _verify_science_code_unchanged(repo, EXPECTED_PARENT_COMMIT)
        if verify_science_code
        else None
    )
    data_audit: dict[str, Any] | None = None
    if require_data:
        data_root = Path(resolved["experiment"]["run_root"]) / "data"
        pinned = dose.get("source_artifact_sha256")
        if not isinstance(pinned, dict) or not pinned:
            raise ValueError("source_artifact_sha256 must be a nonempty mapping")
        observed: dict[str, str] = {}
        for relative, expected_hash in pinned.items():
            artifact = data_root / relative
            if not artifact.is_file():
                raise FileNotFoundError(f"Missing reused data artifact: {artifact}")
            observed_hash = sha256_file(artifact)
            if observed_hash != expected_hash:
                raise ValueError(f"Reused data hash mismatch: {artifact}")
            observed[relative] = observed_hash
        alignment: dict[str, Any] = {}
        for split, expected_rows in (
            ("train", int(raw["carrier"]["train_size"])),
            ("eval", int(raw["carrier"]["eval_size"])),
        ):
            treatment = read_jsonl(data_root / "paired" / f"treatment_{split}.jsonl")
            control = read_jsonl(data_root / "paired" / f"control_{split}.jsonl")
            if len(treatment) != expected_rows or len(control) != expected_rows:
                raise ValueError(f"Wrong {split} row count in reused paired data")
            alignment[split] = _verify_pair_alignment(treatment, control, split=split)
        data_audit = {"verified_artifact_sha256": observed, "alignment": alignment}

    return {
        "schema_version": 1,
        "config_sha256": sha256_value(raw),
        "experiment_id": experiment["id"],
        "run_root": str(Path(resolved["experiment"]["run_root"])),
        "parent_experiment_id": EXPECTED_PARENT_ID,
        "computed_batch_geometry": geometry,
        "science_code_sha256": code_hashes,
        "data_audit": data_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--skip-science-code-check", action="store_true")
    args = parser.parse_args()
    result = verify_tenpass_followup(
        args.config,
        repo_root=args.repo_root,
        require_data=args.require_data,
        verify_science_code=not args.skip_science_code_check,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
