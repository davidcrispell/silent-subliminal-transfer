#!/usr/bin/env python3
"""Fail-closed preflight for the fixed-exposure large-batch follow-up."""

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

EXPECTED_EXPERIMENT_ID = "wolf-sl-gemma2-9b-batch500-v1"
EXPECTED_RUN_ROOT = Path("runs/wolf-sl-gemma2-9b-batch500-v1")
ALLOWED_TRAINING_CHANGES = {
    "batch_size",
    "gradient_accumulation_steps",
    "max_steps",
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


def verify_large_batch_followup(
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
        raise ValueError(f"Unexpected large-batch experiment id: {experiment['id']}")
    if Path(experiment["run_root"]) != EXPECTED_RUN_ROOT:
        raise ValueError(f"Unexpected large-batch run root: {experiment['run_root']}")

    provenance = raw.get("dose_provenance")
    if not isinstance(provenance, dict):
        raise TypeError("Large-batch config must bind dose_provenance")

    source_path = _resolve_protocol_path(repo, provenance["source_config"])
    source = load_config(source_path)
    if sha256_value(source) != provenance["source_config_sha256"]:
        raise ValueError("Source config SHA mismatch")
    if provenance["source_run_id"] != source["experiment"]["id"]:
        raise ValueError("Source run identity mismatch")
    if _data_identity(raw) != _data_identity(source):
        raise ValueError("Large-batch run changed the frozen carrier-data identity")

    comparison_path = _resolve_protocol_path(repo, provenance["comparison_config"])
    comparison = load_config(comparison_path)
    if sha256_value(comparison) != provenance["comparison_config_sha256"]:
        raise ValueError("Comparison config SHA mismatch")
    comparison_training = comparison["training"]["student"]
    training = raw["training"]["student"]
    if _without_keys(training, ALLOWED_TRAINING_CHANGES) != _without_keys(
        comparison_training, ALLOWED_TRAINING_CHANGES
    ):
        raise ValueError(
            "Large-batch follow-up changed student training fields outside batch geometry"
        )

    geometry = verify_declared_batch_geometry(
        raw["batch_geometry"],
        train_examples=int(raw["carrier"]["train_size"]),
        training_config=training,
    )
    if int(provenance["target_epochs"]) != geometry["epochs"]:
        raise ValueError("dose_provenance.target_epochs does not match geometry")
    if int(provenance["effective_batch_size"]) != geometry[
        "nominal_effective_batch_size"
    ]:
        raise ValueError("dose_provenance.effective_batch_size does not match geometry")
    if int(provenance["target_optimizer_steps"]) != geometry[
        "epoch_derived_optimizer_steps"
    ]:
        raise ValueError("dose_provenance.target_optimizer_steps does not match geometry")
    expected_probe_steps = [
        geometry["optimizer_steps_per_epoch"] * epoch for epoch in (3, 4, 5)
    ]
    if provenance["probe_optimizer_steps"] != expected_probe_steps:
        raise ValueError("Probe steps must be exact epoch-3/4/5 boundaries")

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
        for condition in ("control", "treatment"):
            train_path = data_root / "paired" / f"{condition}_train.jsonl"
            eval_path = data_root / "paired" / f"{condition}_eval.jsonl"
            if len(read_jsonl(train_path)) != int(raw["carrier"]["train_size"]):
                raise ValueError(f"Wrong training row count: {train_path}")
            if len(read_jsonl(eval_path)) != int(raw["carrier"]["eval_size"]):
                raise ValueError(f"Wrong evaluation row count: {eval_path}")
        data_audit = {
            "data_root": str(data_root),
            "verified_artifact_sha256": observed,
            "rows_per_training_arm": raw["carrier"]["train_size"],
            "rows_per_evaluation_arm": raw["carrier"]["eval_size"],
        }

    return {
        "schema_version": 1,
        "config_sha256": sha256_value(raw),
        "experiment_id": experiment["id"],
        "run_root": str(Path(resolved["experiment"]["run_root"])),
        "comparison_experiment_id": comparison["experiment"]["id"],
        "comparison_effective_batch_size": (
            comparison_training["batch_size"]
            * comparison_training["gradient_accumulation_steps"]
        ),
        "computed_batch_geometry": geometry,
        "interpretation": {
            "selected_mode": "large practical batch",
            "selected_optimizer_updates": geometry["epoch_derived_optimizer_steps"],
            "literal_full_dataset_batch_optimizer_updates": geometry[
                "literal_full_dataset_reference_total_optimizer_steps"
            ],
            "distinction": (
                "Every epoch sees all 10,000 examples in both modes. Batch size controls "
                "how many examples contribute to one AdamW update; literal full-batch "
                "training would make only one update per epoch."
            ),
        },
        "data_audit": data_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    result = verify_large_batch_followup(
        args.config,
        repo_root=args.repo_root,
        require_data=args.require_data,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
