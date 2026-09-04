#!/usr/bin/env python3
"""Audit the completed beta95 treatment-only continuation and every state boundary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from silent_transfer.checkpointing import verify_exact_checkpoint_artifacts
from silent_transfer.config import load_config, resolve_config
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
    from .import_pythia_treatment_continuation import IMPORT_MANIFEST_NAME
    from .verify_pythia_treatment_continuation import (
        CHECKPOINT_STEPS,
        CONDITION,
        SEED,
        SOURCE_CONFIG,
        audit_source_treatment,
        verify_pythia_treatment_continuation,
    )
    from .verify_pythia_transplant_checkpoint_cell import (
        _read_object,
        _verify_paired_data,
        _verify_runtime_report,
        _verify_training_manifest,
    )
    from .verify_tenpass_checkpoint_cell import audit_checkpoint, checkpoint_file_hashes
except ImportError:
    from import_pythia_treatment_continuation import IMPORT_MANIFEST_NAME  # type: ignore[no-redef]
    from verify_pythia_treatment_continuation import (  # type: ignore[no-redef]
        CHECKPOINT_STEPS,
        CONDITION,
        SEED,
        SOURCE_CONFIG,
        audit_source_treatment,
        verify_pythia_treatment_continuation,
    )
    from verify_pythia_transplant_checkpoint_cell import (  # type: ignore[no-redef]
        _read_object,
        _verify_paired_data,
        _verify_runtime_report,
        _verify_training_manifest,
    )
    from verify_tenpass_checkpoint_cell import (  # type: ignore[no-redef]
        audit_checkpoint,
        checkpoint_file_hashes,
    )


REPORT_NAME = "pythia_treatment_continuation_checkpoint_manifest.json"


def verify_pythia_treatment_continuation_checkpoint_cell(
    config_path: str | Path,
    seed: int,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_config_sha256: str,
    source_cell_root: str | Path | None = None,
    pythia_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify exact resume ancestry, all checkpoints, and the published adapter."""

    if seed != SEED:
        raise ValueError(f"Only frozen treatment seed {SEED} may be audited")
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config_sha256 = sha256_value(raw)
    if config_sha256 != expected_config_sha256:
        raise ValueError("Protocol config SHA does not match the launch identity")
    protocol = verify_pythia_treatment_continuation(
        config_path,
        repo_root=repo,
        pythia_root=pythia_root,
        expected_git_commit=expected_git_commit,
        expected_config_sha256=expected_config_sha256,
        require_data=True,
    )
    config = resolve_config(raw, repo_root=repo)
    config["_protocol_config_sha256"] = config_sha256
    run_root = Path(config["experiment"]["run_root"])
    output = run_root / "models" / "students" / CONDITION / f"seed-{seed}"
    training = config["training"]["student"]
    checkpoint_steps = tuple(int(step) for step in training["checkpoint_steps"])
    if checkpoint_steps != tuple(CHECKPOINT_STEPS):
        raise ValueError("Checkpoint schedule differs from the frozen continuation")
    target_step = checkpoint_steps[-1]

    expected_raw_completion_tokens = int(config["carrier"]["answer_max_count"]) * 5 - 1
    if expected_raw_completion_tokens != int(config["carrier"]["raw_completion_token_count"]):
        raise ValueError("Frozen raw decoder token geometry is inconsistent")
    train_path, eval_path, data_hashes, token_geometry = _verify_paired_data(
        run_root,
        condition=CONDITION,
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

    source = audit_source_treatment(
        config_path, repo_root=repo, source_cell_root=source_cell_root
    )
    source_audit = source["checkpoint_audit"]
    imported_checkpoint = output / "trainer" / "checkpoint-1024"
    if checkpoint_file_hashes(imported_checkpoint) != source_audit["file_sha256"]:
        raise ValueError("Imported checkpoint 1024 no longer equals the source checkpoint")
    imported = _read_object(output / IMPORT_MANIFEST_NAME, "continuation import")
    expected_import = {
        "schema_version": 1,
        "destination_config_sha256": config_sha256,
        "destination_run_id": raw["experiment"]["id"],
        "condition": CONDITION,
        "seed": seed,
        "source_config_sha256": raw["continuation_provenance"]["source_config_sha256"],
        "source_git_commit": raw["continuation_provenance"]["source_git_commit"],
        "source_run_id": raw["continuation_provenance"]["source_run_id"],
        "checkpoint_step": 1024,
        "checkpoint_pass": 1.0,
        "parent_artifact_sha256": source["parent_artifact_sha256"],
        "source_checkpoint_file_sha256": source_audit["file_sha256"],
        "destination_checkpoint_file_sha256": source_audit["file_sha256"],
        "data_artifact_sha256": raw["dose_provenance"]["source_artifact_sha256"],
        "reuse_method": "independent_atomic_byte_copy",
    }
    if imported != expected_import:
        raise ValueError("Continuation import identity or artifact binding mismatch")

    metrics_path = output / "training_metrics.json"
    metrics = _read_object(metrics_path, "training metrics")
    expected_optimizer = resolve_adamw_hyperparameters(training)
    expected_metrics = {
        "optimizer": training["optimizer"],
        "optimizer_hyperparameters": expected_optimizer,
        "optimizer_steps": target_step,
        "configured_checkpoint_steps": list(checkpoint_steps),
        "observed_checkpoint_steps": list(checkpoint_steps),
        "configured_max_steps": target_step,
        "scheduler_total_steps": 10240,
        "lr_scheduler_semantics": "pythia_lambda_v1",
        "configured_warmup_steps": 16,
        "batch_geometry": training_batch_geometry(
            int(config["carrier"]["train_size"]), training
        ),
        "completion_only_loss": True,
        "train_examples": 8192,
        "eval_examples": 0,
        "train_data_sha256": sha256_file(train_path),
        "eval_data_sha256": sha256_file(eval_path),
    }
    for key, expected in expected_metrics.items():
        if metrics.get(key) != expected:
            raise ValueError(
                f"Training metric {key} mismatch: {metrics.get(key)!r} != {expected!r}"
            )

    runtime_path = run_root / "orchestration" / f"runtime-{CONDITION}-{seed}.json"
    runtime = _verify_runtime_report(
        runtime_path,
        raw=raw,
        condition=CONDITION,
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
    source_training = load_config(repo / SOURCE_CONFIG)["training"]["student"]
    source_suffix = (
        Path(raw["continuation_provenance"]["source_run_root"])
        / "models"
        / "students"
        / CONDITION
        / f"seed-{seed}"
        / "trainer"
    ).as_posix()
    imported_audit = audit_checkpoint(
        imported_checkpoint,
        step=1024,
        epoch=1.0,
        training=source_training,
        seed=seed,
        expected_output_suffix=source_suffix,
        expected_eval_strategy="no",
    )
    if imported_audit["file_sha256"] != source_audit["file_sha256"]:
        raise ValueError("Imported checkpoint audit differs from frozen source audit")

    logical_suffix = (
        Path(raw["experiment"]["run_root"])
        / "models"
        / "students"
        / CONDITION
        / f"seed-{seed}"
        / "trainer"
    ).as_posix()
    checkpoints: dict[str, Any] = {"1024": imported_audit}
    for step in checkpoint_steps[1:]:
        expected_epoch = step / 1024
        audit = audit_checkpoint(
            output / "trainer" / f"checkpoint-{step}",
            step=step,
            epoch=expected_epoch,
            training=training,
            seed=seed,
            expected_output_suffix=logical_suffix,
            expected_eval_strategy="no",
        )
        if not math.isclose(
            float(audit["trainer_state"]["epoch"]),
            expected_epoch,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Checkpoint {step} has the wrong fractional epoch")
        checkpoints[str(step)] = audit

    final_adapter_hashes = adapter_artifact_hashes(output / "final_adapter")
    if checkpoints[str(target_step)]["adapter_artifact_sha256"] != final_adapter_hashes:
        raise ValueError("Terminal checkpoint does not match the published adapter")

    result = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "run_id": raw["experiment"]["id"],
        "git_commit": expected_git_commit,
        "condition": CONDITION,
        "seed": seed,
        "protocol_report_sha256": sha256_value(protocol),
        "runtime_report_sha256": sha256_file(runtime_path),
        "runtime": runtime,
        "training_complete_sha256": sha256_file(output / "training_complete.json"),
        "resume_identity_sha256": sha256_file(output / "resume_identity.json"),
        "training_metrics_sha256": sha256_file(metrics_path),
        "training_manifest_sha256": sha256_file(training_manifest_path),
        "continuation_import_sha256": sha256_file(output / IMPORT_MANIFEST_NAME),
        "data_sha256": data_hashes,
        "optimizer": {
            "name": training["optimizer"],
            **expected_optimizer,
            "weight_decay": float(training["weight_decay"]),
            "scheduler_semantics": training["lr_scheduler_semantics"],
        },
        "raw_decoder_tokens_per_example": expected_raw_completion_tokens,
        "training_token_geometry": token_geometry,
        "registered_probe_optimizer_steps": list(checkpoint_steps),
        "audited_optimizer_steps": list(checkpoint_steps),
        "final_adapter_artifact_sha256": final_adapter_hashes,
        "checkpoints": checkpoints,
    }
    write_json_atomic(output / REPORT_NAME, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("seed", type=int)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--source-cell-root")
    parser.add_argument("--pythia-root")
    args = parser.parse_args()
    result = verify_pythia_treatment_continuation_checkpoint_cell(
        args.config,
        args.seed,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        expected_config_sha256=args.expected_config_sha256,
        source_cell_root=args.source_cell_root,
        pythia_root=args.pythia_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
