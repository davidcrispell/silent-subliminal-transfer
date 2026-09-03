#!/usr/bin/env python3
"""Fail-closed audit for one completed Gemma/Pythia-transplant student cell."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from silent_transfer.checkpointing import verify_exact_checkpoint_artifacts
from silent_transfer.config import load_config, resolve_config
from silent_transfer.data import read_jsonl
from silent_transfer.optimizer import resolve_adamw_hyperparameters
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)
from silent_transfer.training import verify_saved_training_identity
from silent_transfer.training_geometry import training_batch_geometry

try:
    from .verify_pythia_transplant import verify_pythia_transplant
    from .verify_tenpass_checkpoint_cell import audit_checkpoint
except ImportError:  # Direct ``python scripts/...`` execution on a pod.
    from verify_pythia_transplant import verify_pythia_transplant  # type: ignore[no-redef]
    from verify_tenpass_checkpoint_cell import audit_checkpoint  # type: ignore[no-redef]


REPORT_NAME = "pythia_transplant_checkpoint_manifest.json"


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _verify_runtime_report(
    path: Path,
    *,
    raw: dict[str, Any],
    condition: str,
    seed: int,
    expected_git_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    report = _read_object(path, "runtime report")
    expected_identity = {
        "experiment_id": raw["experiment"]["id"],
        "condition": condition,
        "seed": seed,
        "git_commit": expected_git_commit,
        "config_sha256": config_sha256,
        "packages": raw["runtime"]["expected_training_packages"],
        "gpu_count": int(raw["runtime"]["expected_gpu_count"]),
    }
    for key, expected in expected_identity.items():
        if report.get(key) != expected:
            raise ValueError(f"Runtime report {key} mismatch: {report.get(key)!r}")
    if raw["runtime"]["expected_gpu_name"] not in str(report.get("gpu_name", "")):
        raise ValueError("Runtime report GPU identity mismatch")
    return report


def _verify_training_manifest(
    path: Path,
    *,
    raw: dict[str, Any],
    metrics: dict[str, Any],
    seed: int,
    expected_git_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    manifest = _read_object(path, "training manifest")
    if (
        manifest.get("stage") != "train_adapter"
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("model") != raw["model"]
    ):
        raise ValueError("Training manifest protocol identity mismatch")
    extra = manifest.get("extra")
    if (
        not isinstance(extra, dict)
        or extra.get("seed") != seed
        or extra.get("metrics") != metrics
    ):
        raise ValueError("Training manifest does not bind the exact cell metrics")
    environment = manifest.get("environment")
    if not isinstance(environment, dict):
        raise TypeError("Training manifest has no environment record")
    git = environment.get("git")
    if not isinstance(git, dict) or git.get("commit") != expected_git_commit:
        raise ValueError("Training manifest git commit mismatch")
    expected_packages = raw["runtime"]["expected_training_packages"]
    packages = environment.get("packages")
    if not isinstance(packages, dict) or any(
        packages.get(name) != version for name, version in expected_packages.items()
    ):
        raise ValueError("Training manifest package identity mismatch")
    gpu = environment.get("gpu")
    if (
        not isinstance(gpu, dict)
        or gpu.get("cuda_available") is not True
        or int(gpu.get("device_count", -1)) != int(raw["runtime"]["expected_gpu_count"])
    ):
        raise ValueError("Training manifest CUDA identity mismatch")
    devices = gpu.get("devices")
    expected_name = raw["runtime"]["expected_gpu_name"]
    if (
        not isinstance(devices, list)
        or len(devices) != int(raw["runtime"]["expected_gpu_count"])
        or any(expected_name not in str(device.get("name", "")) for device in devices)
    ):
        raise ValueError("Training manifest GPU identity mismatch")
    return manifest


def _verify_paired_data(
    run_root: Path,
    *,
    condition: str,
    expected_train_rows: int,
    expected_training_completion_tokens: int,
    expected_full_token_count_min: int,
    expected_full_token_count_max: int,
    max_length: int,
) -> tuple[Path, Path, dict[str, str], dict[str, int]]:
    paired = run_root / "data" / "paired"
    train_path = paired / f"{condition}_train.jsonl"
    other_condition = "control" if condition == "treatment" else "treatment"
    other_train_path = paired / f"{other_condition}_train.jsonl"
    eval_path = paired / f"{condition}_eval.jsonl"
    other_eval_path = paired / f"{other_condition}_eval.jsonl"
    rows = read_jsonl(train_path)
    other_rows = read_jsonl(other_train_path)
    eval_rows = read_jsonl(eval_path)
    other_eval_rows = read_jsonl(other_eval_path)
    if len(rows) != expected_train_rows or len(other_rows) != expected_train_rows:
        raise ValueError("Paired training data does not contain the frozen example count")
    if eval_rows or other_eval_rows:
        raise ValueError("Pythia transplant requires an empty held-out carrier split")
    pair_ids = [row.get("pair_id") for row in rows]
    other_pair_ids = [row.get("pair_id") for row in other_rows]
    if pair_ids != other_pair_ids or len(set(pair_ids)) != expected_train_rows:
        raise ValueError("Treatment/control carrier rows are not uniquely pair-aligned")
    token_counts = [row.get("completion_token_count") for row in rows]
    other_token_counts = [row.get("completion_token_count") for row in other_rows]
    if token_counts != other_token_counts or len(set(token_counts)) != 1:
        raise ValueError("Paired carriers do not have constant equal token exposure")
    training_completion_tokens = token_counts[0]
    if (
        isinstance(training_completion_tokens, bool)
        or not isinstance(training_completion_tokens, int)
        or training_completion_tokens != expected_training_completion_tokens
    ):
        raise ValueError(
            "Paired carriers do not have the frozen chat-template completion-token "
            f"exposure {expected_training_completion_tokens}"
        )
    full_counts = [row.get("full_token_count") for row in rows]
    other_full_counts = [row.get("full_token_count") for row in other_rows]
    if full_counts != other_full_counts or any(
        isinstance(count, bool) or not isinstance(count, int) or count <= 0
        for count in full_counts
    ):
        raise ValueError("Paired carriers do not have valid equal full tokenized lengths")
    observed_full_min = min(full_counts)
    observed_full_max = max(full_counts)
    if (
        observed_full_min != expected_full_token_count_min
        or observed_full_max != expected_full_token_count_max
    ):
        raise ValueError(
            "Paired full-row token geometry drifted: "
            f"observed {observed_full_min}..{observed_full_max}, expected "
            f"{expected_full_token_count_min}..{expected_full_token_count_max}"
        )
    if observed_full_max > max_length:
        raise ValueError(
            f"Paired full row has {observed_full_max} tokens but max_length={max_length}"
        )
    hashes = {
        "condition_train": sha256_file(train_path),
        "other_condition_train": sha256_file(other_train_path),
        "condition_eval": sha256_file(eval_path),
        "other_condition_eval": sha256_file(other_eval_path),
    }
    token_geometry = {
        "training_completion_tokens_per_example": training_completion_tokens,
        "full_token_count_min": observed_full_min,
        "full_token_count_max": observed_full_max,
        "max_length": max_length,
    }
    return train_path, eval_path, hashes, token_geometry


def verify_pythia_transplant_checkpoint_cell(
    config_path: str | Path,
    condition: str,
    seed: int,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_config_sha256: str,
    pythia_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify every immutable state boundary and atomically bind the result."""

    if condition not in {"control", "treatment"}:
        raise ValueError("condition must be control or treatment")
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config_sha256 = sha256_value(raw)
    if config_sha256 != expected_config_sha256:
        raise ValueError("Protocol config SHA does not match the launch identity")
    protocol_report = verify_pythia_transplant(
        config_path,
        repo_root=repo,
        pythia_root=pythia_root,
        expected_git_commit=expected_git_commit,
        expected_config_sha256=expected_config_sha256,
    )
    config = resolve_config(raw, repo_root=repo)
    config["_protocol_config_sha256"] = config_sha256
    if seed not in config["seeds"]["students"]:
        raise ValueError(f"Unregistered student seed: {seed}")

    run_root = Path(config["experiment"]["run_root"])
    output = run_root / "models" / "students" / condition / f"seed-{seed}"
    training = config["training"]["student"]
    checkpoint_steps = tuple(int(step) for step in training["checkpoint_steps"])
    registered_steps = tuple(
        int(step) for step in raw["dose_provenance"]["probe_optimizer_steps"]
    )
    if checkpoint_steps != registered_steps:
        raise ValueError("Training checkpoint schedule and registered probes differ")
    target_step = int(raw["dose_provenance"]["target_optimizer_steps"])
    if checkpoint_steps[-1] != target_step:
        raise ValueError("Checkpoint schedule does not end at the target optimizer step")

    # Gemma has no isometric one-token support for all 100--999 integers.  The
    # frozen ASCII FSA emits ``space + 3 digits`` per number and one comma
    # between numbers: 10 * 4 + 9 = 49 visible completion tokens.
    expected_raw_completion_tokens = int(config["carrier"]["answer_max_count"]) * 5 - 1
    if expected_raw_completion_tokens != int(config["carrier"]["raw_completion_token_count"]):
        raise ValueError("Frozen raw decoder token geometry is internally inconsistent")
    train_path, eval_path, data_hashes, token_geometry = _verify_paired_data(
        run_root,
        condition=condition,
        expected_train_rows=int(config["carrier"]["train_size"]),
        expected_training_completion_tokens=int(
            config["carrier"]["paired_completion_token_count"]
        ),
        expected_full_token_count_min=int(config["carrier"]["paired_full_token_count_min"]),
        expected_full_token_count_max=int(config["carrier"]["paired_full_token_count_max"]),
        max_length=int(training["max_length"]),
    )
    verify_saved_training_identity(
        output,
        config=config,
        training_config=training,
        train_path=train_path,
        eval_path=eval_path,
        seed=seed,
    )

    metrics_path = output / "training_metrics.json"
    metrics = _read_object(metrics_path, "training metrics")
    expected_geometry = training_batch_geometry(int(config["carrier"]["train_size"]), training)
    expected_optimizer = resolve_adamw_hyperparameters(training)
    expected_metrics = {
        "optimizer": training["optimizer"],
        "optimizer_hyperparameters": expected_optimizer,
        "optimizer_steps": target_step,
        "configured_checkpoint_steps": list(checkpoint_steps),
        "observed_checkpoint_steps": list(checkpoint_steps),
        "configured_max_steps": target_step,
        "scheduler_total_steps": int(training["scheduler_total_steps"]),
        "lr_scheduler_semantics": training["lr_scheduler_semantics"],
        "configured_warmup_steps": int(training["warmup_steps"]),
        "batch_geometry": expected_geometry,
        "completion_only_loss": True,
        "train_examples": int(config["carrier"]["train_size"]),
        "eval_examples": 0,
        "train_data_sha256": sha256_file(train_path),
        "eval_data_sha256": sha256_file(eval_path),
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Training metric {key} mismatch: {metrics.get(key)!r} != {expected!r}"
            )

    runtime_path = run_root / "orchestration" / f"runtime-{condition}-{seed}.json"
    runtime = _verify_runtime_report(
        runtime_path,
        raw=raw,
        condition=condition,
        seed=seed,
        expected_git_commit=expected_git_commit,
        config_sha256=config_sha256,
    )
    training_manifest_path = output / "manifest.json"
    _verify_training_manifest(
        training_manifest_path,
        raw=raw,
        metrics=metrics,
        seed=seed,
        expected_git_commit=expected_git_commit,
        config_sha256=config_sha256,
    )

    verify_exact_checkpoint_artifacts(output / "trainer", checkpoint_steps)
    logical_suffix = (
        Path(raw["experiment"]["run_root"])
        / "models"
        / "students"
        / condition
        / f"seed-{seed}"
        / "trainer"
    ).as_posix()
    steps_per_epoch = int(raw["batch_geometry"]["optimizer_steps_per_epoch"])
    checkpoints: dict[str, Any] = {}
    for step in checkpoint_steps:
        expected_epoch = step / steps_per_epoch
        audit = audit_checkpoint(
            output / "trainer" / f"checkpoint-{step}",
            step=step,
            epoch=expected_epoch,
            training=training,
            seed=seed,
            expected_output_suffix=logical_suffix,
            expected_eval_strategy="no",
        )
        observed_epoch = float(audit["trainer_state"]["epoch"])
        if not math.isclose(observed_epoch, expected_epoch, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Checkpoint {step} has the wrong fractional epoch")
        checkpoints[str(step)] = audit

    final_adapter_hashes = adapter_artifact_hashes(output / "final_adapter")
    if checkpoints[str(target_step)]["adapter_artifact_sha256"] != final_adapter_hashes:
        raise ValueError("Terminal checkpoint does not match the published final adapter")

    completion_path = output / "training_complete.json"
    identity_path = output / "resume_identity.json"
    result = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "run_id": raw["experiment"]["id"],
        "git_commit": expected_git_commit,
        "condition": condition,
        "seed": seed,
        "protocol_report_sha256": sha256_value(protocol_report),
        "runtime_report_sha256": sha256_file(runtime_path),
        "runtime": runtime,
        "training_complete_sha256": sha256_file(completion_path),
        "resume_identity_sha256": sha256_file(identity_path),
        "training_metrics_sha256": sha256_file(metrics_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "data_sha256": data_hashes,
        "optimizer": {
            "name": training["optimizer"],
            **expected_optimizer,
            "weight_decay": float(training["weight_decay"]),
            "scheduler_semantics": training["lr_scheduler_semantics"],
        },
        "raw_decoder_tokens_per_example": expected_raw_completion_tokens,
        "training_token_geometry": token_geometry,
        "registered_probe_optimizer_steps": list(registered_steps),
        "audited_optimizer_steps": list(checkpoint_steps),
        "final_adapter_artifact_sha256": final_adapter_hashes,
        "checkpoints": checkpoints,
    }
    write_json_atomic(output / REPORT_NAME, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("condition", choices=("control", "treatment"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--pythia-root")
    args = parser.parse_args()
    result = verify_pythia_transplant_checkpoint_cell(
        args.config,
        args.condition,
        args.seed,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        expected_config_sha256=args.expected_config_sha256,
        pythia_root=args.pythia_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
