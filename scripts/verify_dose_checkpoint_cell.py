#!/usr/bin/env python3
"""Verify and bind all preregistered checkpoints for one dose cell."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)
from silent_transfer.training import verify_saved_training_identity


def verify_dose_checkpoint_cell(
    config_path: str | Path,
    condition: str,
    seed: int,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    if condition not in {"control", "treatment"}:
        raise ValueError("condition must be control or treatment")
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    config_sha = sha256_value(raw)
    config["_protocol_config_sha256"] = config_sha
    if seed not in config["seeds"]["students"]:
        raise ValueError(f"Unregistered student seed: {seed}")

    run_root = Path(config["experiment"]["run_root"])
    output = run_root / "models" / "students" / condition / f"seed-{seed}"
    train_path = run_root / "data" / "paired" / f"{condition}_train.jsonl"
    eval_path = run_root / "data" / "paired" / f"{condition}_eval.jsonl"
    training = config["training"]["student"]
    verify_saved_training_identity(
        output,
        config=config,
        training_config=training,
        train_path=train_path,
        eval_path=eval_path,
        seed=seed,
    )

    metrics_path = output / "training_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    target_steps = int(raw["dose_provenance"]["target_optimizer_steps"])
    if int(metrics["optimizer_steps"]) != target_steps:
        raise ValueError("Dose cell optimizer-step count does not match target")
    if int(metrics["train_examples"]) != int(config["carrier"]["train_size"]):
        raise ValueError("Dose cell train-example count does not match frozen data")

    checkpoints: dict[str, Any] = {}
    for step_value in raw["dose_provenance"]["probe_optimizer_steps"]:
        step = int(step_value)
        checkpoint = output / "trainer" / f"checkpoint-{step}"
        state_path = checkpoint / "trainer_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if int(state["global_step"]) != step:
            raise ValueError(f"Checkpoint {step} has the wrong Trainer global step")
        checkpoints[str(step)] = {
            "adapter_artifact_sha256": adapter_artifact_hashes(checkpoint),
            "trainer_state_sha256": sha256_file(state_path),
        }

    final_adapter_hashes = adapter_artifact_hashes(output / "final_adapter")
    if checkpoints[str(target_steps)]["adapter_artifact_sha256"] != final_adapter_hashes:
        raise ValueError("Primary dose checkpoint does not match the published final adapter")

    result = {
        "schema_version": 1,
        "config_sha256": config_sha,
        "run_id": raw["experiment"]["id"],
        "condition": condition,
        "seed": seed,
        "training_complete_sha256": sha256_file(output / "training_complete.json"),
        "resume_identity_sha256": sha256_file(output / "resume_identity.json"),
        "training_metrics_sha256": sha256_file(metrics_path),
        "final_adapter_artifact_sha256": final_adapter_hashes,
        "train_data_sha256": sha256_file(train_path),
        "eval_data_sha256": sha256_file(eval_path),
        "checkpoints": checkpoints,
    }
    write_json_atomic(output / "dose_checkpoint_manifest.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("condition", choices=("control", "treatment"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = verify_dose_checkpoint_cell(
        args.config,
        args.condition,
        args.seed,
        repo_root=args.repo_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
