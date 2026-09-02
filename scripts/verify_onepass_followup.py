#!/usr/bin/env python3
"""Fail-closed preflight for the EB16, one-pass Gemma follow-up."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.data import read_jsonl
from silent_transfer.provenance import sha256_file, sha256_value
from silent_transfer.training_geometry import verify_declared_batch_geometry

EXPECTED_EXPERIMENT_ID = "wolf-sl-gemma2-9b-eb16-onepass-v1"
EXPECTED_RUN_ROOT = Path("runs/wolf-sl-gemma2-9b-eb16-onepass-v1")
EXPECTED_EFFECTIVE_BATCH = 16
EXPECTED_TRAIN_EXAMPLES = 10_000
EXPECTED_OPTIMIZER_STEPS = 625
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
ALLOWED_COMPARISON_TRAINING_CHANGES = {
    "epochs",
    "max_steps",
    "save_total_limit",
    "scheduler_total_steps",
    "warmup_steps",
}


def _without_keys(mapping: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    result = copy.deepcopy(mapping)
    for key in keys:
        result.pop(key, None)
    return result


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


def _resolve_protocol_path(repo: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo / path


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


def verify_onepass_followup(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    require_data: bool = False,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    resolved = resolve_config(raw, repo_root=repo)
    experiment = raw["experiment"]
    if experiment["id"] != EXPECTED_EXPERIMENT_ID:
        raise ValueError(f"Unexpected one-pass experiment id: {experiment['id']}")
    if Path(experiment["run_root"]) != EXPECTED_RUN_ROOT:
        raise ValueError(f"Unexpected one-pass run root: {experiment['run_root']}")

    provenance = raw.get("dose_provenance")
    if not isinstance(provenance, dict):
        raise TypeError("One-pass config must bind dose_provenance")

    source_path = _resolve_protocol_path(repo, provenance["source_config"])
    source = load_config(source_path)
    if sha256_value(source) != provenance["source_config_sha256"]:
        raise ValueError("Source config SHA mismatch")
    if provenance["source_run_id"] != source["experiment"]["id"]:
        raise ValueError("Source run identity mismatch")
    if _data_identity(raw) != _data_identity(source):
        raise ValueError("One-pass run changed the frozen carrier-data identity")

    comparison_path = _resolve_protocol_path(repo, provenance["comparison_config"])
    comparison = load_config(comparison_path)
    if sha256_value(comparison) != provenance["comparison_config_sha256"]:
        raise ValueError("Comparison config SHA mismatch")
    training = raw["training"]["student"]
    comparison_training = comparison["training"]["student"]
    if _without_keys(training, ALLOWED_COMPARISON_TRAINING_CHANGES) != _without_keys(
        comparison_training, ALLOWED_COMPARISON_TRAINING_CHANGES
    ):
        raise ValueError(
            "One-pass run changed dose5 training fields outside the frozen horizon"
        )

    geometry = verify_declared_batch_geometry(
        raw["batch_geometry"],
        train_examples=int(raw["carrier"]["train_size"]),
        training_config=training,
    )
    if (
        geometry["train_examples"] != EXPECTED_TRAIN_EXAMPLES
        or geometry["nominal_effective_batch_size"] != EXPECTED_EFFECTIVE_BATCH
        or geometry["epoch_derived_optimizer_steps"] != EXPECTED_OPTIMIZER_STEPS
        or geometry["epochs"] != 1
        or geometry["total_example_exposures"] != EXPECTED_TRAIN_EXAMPLES
        or geometry["all_optimizer_steps_equal_size"] is not True
    ):
        raise ValueError("One-pass batch geometry is not exactly EB16 x 625 updates")
    if (
        provenance.get("target_epochs") != 1
        or provenance.get("effective_batch_size") != EXPECTED_EFFECTIVE_BATCH
        or provenance.get("target_optimizer_steps") != EXPECTED_OPTIMIZER_STEPS
        or provenance.get("scheduler_total_updates") != 6_250
        or provenance.get("schedule_examples") != 100_000
        or provenance.get("warmup_updates") != 8
        or provenance.get("warmup_examples") != 128
        or provenance.get("probe_epochs") != [1]
        or provenance.get("probe_optimizer_steps") != [EXPECTED_OPTIMIZER_STEPS]
    ):
        raise ValueError("One-pass dose provenance does not match the frozen endpoint")
    if (
        training.get("scheduler_total_steps") != 6_250
        or training.get("warmup_steps") != 8
        or training.get("warmup_ratio") != 0.05
    ):
        raise ValueError("One-pass optimizer schedule is not the Pythia-isometric horizon")
    runtime = raw["runtime"]
    for key, expected in EXPECTED_RUNTIME.items():
        if runtime.get(key) != expected:
            raise ValueError(f"One-pass runtime identity mismatch for {key}")

    data_audit: dict[str, Any] | None = None
    if require_data:
        data_root = Path(resolved["experiment"]["run_root"]) / "data"
        pinned = provenance.get("source_artifact_sha256")
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
        data_audit = {
            "data_root": str(data_root),
            "verified_artifact_sha256": observed,
            "alignment": alignment,
        }

    return {
        "schema_version": 1,
        "config_sha256": sha256_value(raw),
        "experiment_id": experiment["id"],
        "run_root": str(Path(resolved["experiment"]["run_root"])),
        "comparison_experiment_id": comparison["experiment"]["id"],
        "computed_batch_geometry": geometry,
        "data_audit": data_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    result = verify_onepass_followup(
        args.config,
        repo_root=args.repo_root,
        require_data=args.require_data,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
