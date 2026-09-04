#!/usr/bin/env python3
"""Fail closed on the exact beta2=.95 treatment-only ten-pass continuation."""

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

try:
    from .verify_pythia_transplant import verify_pythia_transplant
    from .verify_tenpass_checkpoint_cell import audit_checkpoint
except ImportError:
    from verify_pythia_transplant import verify_pythia_transplant  # type: ignore[no-redef]
    from verify_tenpass_checkpoint_cell import audit_checkpoint  # type: ignore[no-redef]


EXPERIMENT_ID = (
    "wolf-sl-gemma2-9b-pythia-eb8-alpha32-beta95-tenpass-treatment-pilot-v1"
)
RUN_ROOT = Path(f"runs/{EXPERIMENT_ID}")
SOURCE_CONFIG = Path(
    "configs/wolf_sl_9b_pythia_hillclimb_beta95_eb8_alpha32.yaml"
)
SOURCE_CONFIG_SHA256 = (
    "f0552b123cc19a140f6d887789251e5bcda3f1c922cf6f3972e5e1cd4f857498"
)
SOURCE_COMMIT = "5c65cd84a415a161e62e0a67e3b93af320b0315b"
SOURCE_RUN_ID = "wolf-sl-gemma2-9b-pythia-eb8-alpha32-beta95-onepass-pilot-v1"
SOURCE_RUN_ROOT = Path(f"runs/{SOURCE_RUN_ID}")
SEED = 53101
CONDITION = "treatment"
STEPS_PER_PASS = 1024
CHECKPOINT_STEPS = list(range(1024, 10241, 512))
EPOCH_STEPS = list(range(1024, 10241, 1024))
SCIENCE_CODE_PATHS = (
    "src/silent_transfer/checkpointing.py",
    "src/silent_transfer/config.py",
    "src/silent_transfer/data.py",
    "src/silent_transfer/masking.py",
    "src/silent_transfer/modeling.py",
    "src/silent_transfer/optimizer.py",
    "src/silent_transfer/scheduler.py",
    "src/silent_transfer/training.py",
    "src/silent_transfer/training_geometry.py",
)
SOURCE_CELL_ARTIFACTS = {
    "pythia_transplant_checkpoint_manifest.json": (
        "3950827ad4090c0c8f8cf21e2bd9316760a26470782d9adfebb5b71dc5394054"
    ),
    "training_complete.json": (
        "79a0b0d6ff5e15f8412ca1958f55637cb5c7d5a45da48de032b06fe1604d12fb"
    ),
    "resume_identity.json": (
        "5ad2faa394c1aaa3d4130341d9b45e4098050298ec9f79feb7e7f09682e8330d"
    ),
    "manifest.json": (
        "86f3c1519a43403f8da25ce62ce34893cb545a06b29e70948a2ee56f3cff283f"
    ),
    "training_metrics.json": (
        "b2dfdf4809c59251db4245821a926a2a38ead5d405f3bdfb00c7cd46b83bb0af"
    ),
}
BASE_REFERENCE_ARTIFACTS = {
    "evaluation_complete.json": (
        "187403abe52303c888e496d1a218e868a479b1d0a4b1c241d8ccafaae301d4c0"
    ),
    "manifest.json": (
        "ebfe33296aaff12cc9ac8f8ffa67c6e8b027d867fed4dfc83abc3e18b9844228"
    ),
    "per_prompt.jsonl": (
        "dd9729c4a610a50351b7926ec7785d53d7cbd225bc177cbf6f23eff6e400ae3a"
    ),
    "prompt_plan.json": (
        "ee73910253d7ffd6e9eee1d5c1d5f32d0fd0e0614e1c52d6e58c1b10cdb1c1e3"
    ),
    "resume_identity.json": (
        "d0c69e343ce7b4226d71b2537b35bc858440b1ac0f842f37800110495d63c30a"
    ),
    "summary.json": (
        "4dda6a5dc013a0c8b784a3beee8950ec00897666649445a2b9ce7e95c611beb4"
    ),
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _verify_science_code(repo: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative in SCIENCE_CODE_PATHS:
        current = (repo / relative).read_bytes()
        source = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=repo
        )
        if current != source:
            raise ValueError(f"Training science code changed since the prefix: {relative}")
        observed[relative] = sha256_file(repo / relative)
    return observed


def _normalized_to_source(raw: dict[str, Any], source: dict[str, Any]) -> None:
    normalized = copy.deepcopy(raw)
    normalized.pop("continuation_provenance", None)
    normalized.pop("base_reference_provenance", None)
    normalized.pop("saturation_rule", None)
    normalized["experiment"] = copy.deepcopy(source["experiment"])
    normalized["replication_design"] = copy.deepcopy(source["replication_design"])
    normalized["dose_provenance"] = copy.deepcopy(source["dose_provenance"])
    normalized["batch_geometry"] = copy.deepcopy(source["batch_geometry"])
    normalized["cloze_evaluation"] = copy.deepcopy(source["cloze_evaluation"])
    for key in ("epochs", "max_steps", "checkpoint_steps", "save_total_limit"):
        normalized["training"]["student"][key] = source["training"]["student"][key]
    if normalized != source:
        raise ValueError(
            "Continuation differs from the source outside identity, dose extension, "
            "treatment-only analysis metadata, and checkpoint retention"
        )


def _source_cell(
    raw: dict[str, Any], repo: Path, source_cell_root: str | Path | None
) -> Path:
    if source_cell_root is not None:
        return Path(source_cell_root).resolve()
    continuation = raw["continuation_provenance"]
    return (
        repo
        / continuation["source_run_root"]
        / "models"
        / "students"
        / CONDITION
        / f"seed-{SEED}"
    )


def audit_source_treatment(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    source_cell_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    source = load_config(repo / SOURCE_CONFIG)
    continuation = raw["continuation_provenance"]
    cell = _source_cell(raw, repo, source_cell_root)
    pins = continuation["expected_treatment_cell"]
    if {
        name: pins[key]
        for name, key in (
            ("pythia_transplant_checkpoint_manifest.json", "checkpoint_manifest_sha256"),
            ("training_complete.json", "training_complete_sha256"),
            ("resume_identity.json", "resume_identity_sha256"),
            ("manifest.json", "training_manifest_sha256"),
            ("training_metrics.json", "training_metrics_sha256"),
        )
    } != SOURCE_CELL_ARTIFACTS:
        raise ValueError("Source treatment artifact pins drifted")
    for name, expected in SOURCE_CELL_ARTIFACTS.items():
        path = cell / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Pinned source treatment artifact mismatch: {path}")

    identity = _read_json(cell / "resume_identity.json", "source resume identity")
    expected_identity = {
        "schema_version": 1,
        "config_sha256": SOURCE_CONFIG_SHA256,
        "model": source["model"],
        "seed": SEED,
        "train_data_sha256": source["dose_provenance"]["source_artifact_sha256"][
            "paired/treatment_train.jsonl"
        ],
        "eval_data_sha256": source["dose_provenance"]["source_artifact_sha256"][
            "paired/treatment_eval.jsonl"
        ],
        "training_config_sha256": sha256_value(source["training"]["student"]),
    }
    if identity != expected_identity:
        raise ValueError("Source treatment resume identity mismatch")
    completion = _read_json(cell / "training_complete.json", "source completion")
    if completion.get("training_identity_sha256") != sha256_value(identity):
        raise ValueError("Source completion does not bind its training identity")

    manifest = _read_json(
        cell / "pythia_transplant_checkpoint_manifest.json",
        "source checkpoint manifest",
    )
    checkpoint_record = manifest.get("checkpoints", {}).get("1024")
    if not isinstance(checkpoint_record, dict):
        raise ValueError("Source checkpoint manifest does not bind checkpoint 1024")
    expected_files = pins["source_checkpoint_file_sha256"]
    if checkpoint_record.get("file_sha256") != expected_files:
        raise ValueError("Source checkpoint-1024 file hashes drifted")
    if checkpoint_record.get("trainer_state_sha256") != pins["trainer_state_sha256"]:
        raise ValueError("Source checkpoint-1024 Trainer state drifted")
    expected_adapter = {
        "adapter_config.json": expected_files["adapter_config.json"],
        "adapter_model.safetensors": pins["adapter_model_sha256"],
    }
    if (
        checkpoint_record.get("adapter_artifact_sha256") != expected_adapter
        or completion.get("adapter_artifact_sha256") != expected_adapter
    ):
        raise ValueError("Source final adapter is not checkpoint 1024")

    checkpoint = cell / "trainer" / "checkpoint-1024"
    suffix = (
        SOURCE_RUN_ROOT
        / "models"
        / "students"
        / CONDITION
        / f"seed-{SEED}"
        / "trainer"
    ).as_posix()
    audit = audit_checkpoint(
        checkpoint,
        step=1024,
        epoch=1.0,
        training=source["training"]["student"],
        seed=SEED,
        expected_output_suffix=suffix,
        expected_eval_strategy="no",
    )
    if audit["file_sha256"] != expected_files:
        raise ValueError("Source checkpoint-1024 bytes do not match frozen pins")
    return {
        "source_cell": str(cell),
        "checkpoint": str(checkpoint),
        "parent_artifact_sha256": SOURCE_CELL_ARTIFACTS,
        "checkpoint_audit": audit,
    }


def verify_pythia_treatment_continuation(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    pythia_root: str | Path | None = None,
    expected_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
    require_data: bool = False,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    source = load_config(repo / SOURCE_CONFIG)
    config_sha = sha256_value(raw)
    if sha256_value(source) != SOURCE_CONFIG_SHA256:
        raise ValueError("Frozen one-pass source config changed")
    _normalized_to_source(raw, source)
    verify_pythia_transplant(SOURCE_CONFIG, repo_root=repo, pythia_root=pythia_root)

    if raw["experiment"]["id"] != EXPERIMENT_ID:
        raise ValueError("Wrong continuation experiment id")
    if Path(raw["experiment"]["run_root"]) != RUN_ROOT:
        raise ValueError("Wrong continuation run root")
    if raw["seeds"]["students"] != [SEED]:
        raise ValueError("Continuation seed drifted")
    replication = raw["replication_design"]
    if replication.get("execution_scope") != "treatment_only_continuation":
        raise ValueError("Continuation is not treatment-only")

    continuation = raw.get("continuation_provenance")
    expected_continuation_identity = {
        "source_config": str(SOURCE_CONFIG),
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "source_git_commit": SOURCE_COMMIT,
        "source_run_id": SOURCE_RUN_ID,
        "source_run_root": str(SOURCE_RUN_ROOT),
        "condition": CONDITION,
        "seed": SEED,
        "checkpoint_step": 1024,
        "checkpoint_pass": 1.0,
        "require_optimizer_state": True,
        "require_scheduler_state": True,
        "require_rng_state": True,
    }
    if not isinstance(continuation, dict) or any(
        continuation.get(key) != value
        for key, value in expected_continuation_identity.items()
    ):
        raise ValueError("Continuation provenance identity drifted")

    training = raw["training"]["student"]
    source_training = source["training"]["student"]
    if training["epochs"] != 10 or training["max_steps"] != 10240:
        raise ValueError("Continuation must end at ten passes / step 10240")
    if (
        training["scheduler_total_steps"] != 10240
        or source_training["scheduler_total_steps"] != 10240
    ):
        raise ValueError("Continuation must preserve the original 10240-step scheduler")
    if training["checkpoint_steps"] != CHECKPOINT_STEPS:
        raise ValueError("Continuation must retain every half-pass checkpoint")
    if training["save_total_limit"] != len(CHECKPOINT_STEPS):
        raise ValueError("Checkpoint retention cannot evict a frozen half-pass")
    if any(
        training[key] != source_training[key]
        for key in source_training
        if key not in {"epochs", "max_steps", "checkpoint_steps", "save_total_limit"}
    ):
        raise ValueError("Optimizer/model training recipe changed from checkpoint prefix")

    geometry = verify_declared_batch_geometry(
        raw["batch_geometry"], train_examples=8192, training_config=training
    )
    if (
        geometry["optimizer_steps_per_epoch"] != STEPS_PER_PASS
        or geometry["epoch_derived_optimizer_steps"] != 10240
        or geometry["total_example_exposures"] != 81920
    ):
        raise ValueError("Ten-pass EB8 geometry drifted")
    dose = raw["dose_provenance"]
    if (
        dose["target_epochs"] != 10
        or dose["target_optimizer_steps"] != 10240
        or dose["scheduler_total_updates"] != 10240
        or dose["probe_optimizer_steps"] != CHECKPOINT_STEPS
        or dose["half_pass_probe_optimizer_steps"] != CHECKPOINT_STEPS
        or dose["epoch_probe_optimizer_steps"] != EPOCH_STEPS
        or dose["probe_example_counts"] != [step * 8 for step in CHECKPOINT_STEPS]
        or dose["no_optional_stopping"] is not True
    ):
        raise ValueError("Continuation dose/probe schedule drifted")
    saturation = raw["saturation_rule"]
    if (
        saturation.get("assessment") != "retrospective_full_curve"
        or
        saturation.get("hard_cap_optimizer_steps") != 10240
        or saturation.get("hard_cap_passes") != 10
        or saturation.get("checkpoint_interval_passes") != 0.5
        or saturation.get("no_early_stopping") is not True
        or saturation.get("primary_metric")
        != "treatment_minus_base_target_logit_margin"
        or saturation.get("primary_material_gain_nats") != 0.10
        or saturation.get("secondary_metric")
        != "treatment_minus_base_target_candidate_probability"
        or saturation.get("secondary_material_gain") != 0.01
        or saturation.get("required_final_epoch_intervals") != 2
        or saturation.get("bootstrap_confidence") != 0.95
        or saturation.get("bootstrap_samples") != 10000
    ):
        raise ValueError("Retrospective saturation rule drifted")

    if raw["cloze_evaluation"].get("primary_metric") != (
        "treatment_minus_base_target_logit_margin"
    ) or raw["cloze_evaluation"].get("secondary_metric") != (
        "treatment_minus_base_target_candidate_probability"
    ):
        raise ValueError("Treatment-minus-base cloze estimand drifted")

    base = raw.get("base_reference_provenance")
    expected_base = {
        "source_config": str(SOURCE_CONFIG),
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "source_run_id": SOURCE_RUN_ID,
        "source_run_root": str(SOURCE_RUN_ROOT),
        "source_path": "evaluations/cloze/base",
        "artifact_sha256": BASE_REFERENCE_ARTIFACTS,
    }
    if base != expected_base:
        raise ValueError("Frozen base-reference provenance drifted")

    code_hashes = _verify_science_code(repo)
    data_audit: dict[str, str] | None = None
    if require_data:
        resolved = resolve_config(raw, repo_root=repo)
        data_root = Path(resolved["experiment"]["run_root"]) / "data"
        pins = dose["source_artifact_sha256"]
        observed: dict[str, str] = {}
        for relative, expected in pins.items():
            artifact = data_root / relative
            if not artifact.is_file() or sha256_file(artifact) != expected:
                raise ValueError(f"Reused data artifact mismatch: {artifact}")
            observed[relative] = expected
        treatment = read_jsonl(data_root / "paired" / "treatment_train.jsonl")
        if len(treatment) != 8192 or len({row.get("pair_id") for row in treatment}) != 8192:
            raise ValueError("Treatment carrier data is not the frozen 8192-row block")
        data_audit = observed

    if expected_git_commit is not None and _git(repo, "rev-parse", "HEAD") != expected_git_commit:
        raise ValueError("Repository commit mismatch")
    if expected_config_sha256 is not None and config_sha != expected_config_sha256:
        raise ValueError("Protocol config SHA mismatch")
    return {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": config_sha,
        "source_config_sha256": SOURCE_CONFIG_SHA256,
        "source_git_commit": SOURCE_COMMIT,
        "execution_scope": "treatment_only_continuation",
        "checkpoint_steps": CHECKPOINT_STEPS,
        "computed_batch_geometry": geometry,
        "science_code_sha256": code_hashes,
        "data_artifact_sha256": data_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--pythia-root")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    result = verify_pythia_treatment_continuation(
        args.config,
        repo_root=args.repo_root,
        pythia_root=args.pythia_root,
        expected_git_commit=args.expected_git_commit,
        expected_config_sha256=args.expected_config_sha256,
        require_data=args.require_data,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
