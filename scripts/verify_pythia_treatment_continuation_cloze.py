#!/usr/bin/env python3
"""Fail-closed audit for the treatment-only continuation cloze curve.

This is deliberately separate from the paired Pythia-transplant assay.  It
binds all 19 continuation checkpoints to treatment evaluations, but never
creates or requires a new control evaluation.  The frozen base is audited by
the treatment-only summarizer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)

try:
    from .summarize_pythia_transplant_cloze import (
        _portable_adapter_hashes,
        _verify_cloze_output,
    )
    from .verify_pythia_treatment_continuation import (
        CHECKPOINT_STEPS,
        CONDITION,
        SEED,
        verify_pythia_treatment_continuation,
    )
    from .verify_pythia_treatment_continuation_checkpoint_cell import (
        REPORT_NAME as CHECKPOINT_MANIFEST_NAME,
    )
except ImportError:  # Direct ``python scripts/...`` execution on a pod.
    from summarize_pythia_transplant_cloze import (  # type: ignore[no-redef]
        _portable_adapter_hashes,
        _verify_cloze_output,
    )
    from verify_pythia_treatment_continuation import (  # type: ignore[no-redef]
        CHECKPOINT_STEPS,
        CONDITION,
        SEED,
        verify_pythia_treatment_continuation,
    )
    from verify_pythia_treatment_continuation_checkpoint_cell import (  # type: ignore[no-redef]
        REPORT_NAME as CHECKPOINT_MANIFEST_NAME,
    )


FROZEN_TRAINING_GIT_COMMIT = "5fa15ac550a488507d987e6984cdffda4ce6845f"
FROZEN_CONFIG_SHA256 = "50d640914a70447eb132fc003023c2070ce975475df6632a02010c6dfaeadef2"
CURVE_COMPLETION_NAME = "treatment_cloze_curve_complete.json"
EXPECTED_LAYER_SIGNATURE = [
    {"index": 0, "name": "embedding"},
    *[{"index": index, "name": f"block_{index:02d}"} for index in range(1, 43)],
]
TOP_LEVEL_EVALUATION_ARTIFACTS = (
    "evaluation_complete.json",
    "manifest.json",
    "per_prompt.jsonl",
    "prompt_plan.json",
    "resume_identity.json",
    "summary.json",
)
EVALUATION_CODE_PATHS = (
    "scripts/lambda/run_pythia_treatment_continuation_cloze.sh",
    "scripts/verify_pythia_treatment_continuation_cloze.py",
    "scripts/summarize_pythia_treatment_continuation_cloze.py",
)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {description}: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return value


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _verify_evaluation_code_at_head(repo: Path) -> dict[str, str]:
    """Require the tracked evaluator bytes to equal the claimed HEAD commit."""

    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).strip()
    if status:
        raise ValueError("Evaluation repository has tracked working-tree changes")
    observed: dict[str, str] = {}
    for relative in EVALUATION_CODE_PATHS:
        path = repo / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing evaluation source: {path}")
        committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=repo)
        if path.read_bytes() != committed:
            raise ValueError(f"Evaluation source differs from HEAD: {relative}")
        observed[relative] = sha256_file(path)
    return observed


def _validate_layer_signature(signature: Any, *, description: str) -> None:
    if signature != EXPECTED_LAYER_SIGNATURE:
        raise ValueError(
            f"{description} does not contain every Gemma hidden state "
            "(embedding plus block_01..block_42)"
        )


def _checkpoint_manifest(
    raw: dict[str, Any],
    *,
    run_root: Path,
    seed: int,
    require_checkpoint_bytes: bool,
) -> tuple[dict[str, Any], Path, dict[int, dict[str, str]], dict[str, bool]]:
    model_root = run_root / "models" / "students" / CONDITION / f"seed-{seed}"
    manifest_path = model_root / CHECKPOINT_MANIFEST_NAME
    manifest = _read_json(manifest_path, "continuation checkpoint manifest")
    expected_steps = [int(step) for step in CHECKPOINT_STEPS]
    expected_identity = {
        "schema_version": 1,
        "config_sha256": FROZEN_CONFIG_SHA256,
        "run_id": raw["experiment"]["id"],
        "git_commit": FROZEN_TRAINING_GIT_COMMIT,
        "condition": CONDITION,
        "seed": seed,
        "registered_probe_optimizer_steps": expected_steps,
        "audited_optimizer_steps": expected_steps,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Continuation checkpoint manifest {key} mismatch: "
                f"{manifest.get(key)!r} != {expected!r}"
            )
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != {
        str(step) for step in expected_steps
    }:
        raise ValueError("Continuation checkpoint manifest has the wrong inventory")

    adapter_hashes: dict[int, dict[str, str]] = {}
    local_verification: dict[str, bool] = {}
    for step in expected_steps:
        checkpoint = checkpoints[str(step)]
        if not isinstance(checkpoint, dict):
            raise TypeError(f"Checkpoint {step} audit is not an object")
        expected_adapter = _portable_adapter_hashes(
            checkpoint.get("adapter_artifact_sha256"),
            f"checkpoint {step} adapter identity",
        )
        adapter_hashes[step] = expected_adapter
        checkpoint_dir = model_root / "trainer" / f"checkpoint-{step}"
        present = checkpoint_dir.is_dir()
        local_verification[str(step)] = present
        if require_checkpoint_bytes and not present:
            raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")
        if present:
            observed = _portable_adapter_hashes(
                adapter_artifact_hashes(checkpoint_dir),
                f"checkpoint {step} local adapter",
            )
            if observed != expected_adapter:
                raise ValueError(f"Checkpoint {step} adapter bytes differ from the manifest")

    final_hashes = _portable_adapter_hashes(
        manifest.get("final_adapter_artifact_sha256"),
        "published final adapter identity",
    )
    if final_hashes != adapter_hashes[expected_steps[-1]]:
        raise ValueError("Published final adapter differs from checkpoint 10240")
    return manifest, manifest_path, adapter_hashes, local_verification


def verify_pythia_treatment_continuation_cloze(
    config_path: str | Path,
    seed: int,
    *,
    repo_root: str | Path,
    expected_evaluation_git_commit: str,
    expected_config_sha256: str,
    pythia_root: str | Path | None = None,
    require_checkpoint_bytes: bool = False,
    require_complete: bool = True,
    publish_completion: bool = True,
) -> dict[str, Any]:
    """Audit checkpoint ancestry and, when requested, every cloze artifact."""

    if seed != SEED:
        raise ValueError(f"Only frozen treatment seed {SEED} may be evaluated")
    if expected_config_sha256 != FROZEN_CONFIG_SHA256:
        raise ValueError("Expected config SHA is not the frozen continuation config")
    if (
        not isinstance(expected_evaluation_git_commit, str)
        or len(expected_evaluation_git_commit) != 40
        or any(c not in "0123456789abcdef" for c in expected_evaluation_git_commit)
    ):
        raise ValueError("Expected evaluation git commit must be an exact 40-hex commit")

    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    if sha256_value(raw) != FROZEN_CONFIG_SHA256:
        raise ValueError("Continuation config no longer has its frozen semantic SHA")
    if _git_head(repo) != expected_evaluation_git_commit:
        raise ValueError("Evaluation repository commit mismatch")
    evaluation_code_hashes = _verify_evaluation_code_at_head(repo)
    protocol = verify_pythia_treatment_continuation(
        config_path,
        repo_root=repo,
        pythia_root=pythia_root,
        expected_git_commit=expected_evaluation_git_commit,
        expected_config_sha256=FROZEN_CONFIG_SHA256,
        require_data=False,
    )
    config = resolve_config(raw, repo_root=repo)
    config["_protocol_config_sha256"] = FROZEN_CONFIG_SHA256
    run_root = Path(config["experiment"]["run_root"])
    manifest, manifest_path, adapters, local_checkpoints = _checkpoint_manifest(
        raw,
        run_root=run_root,
        seed=seed,
        require_checkpoint_bytes=require_checkpoint_bytes,
    )

    result: dict[str, Any] = {
        "schema_version": 1,
        "stage": "pythia_treatment_continuation_cloze_curve_audit",
        "scope": "treatment_only_no_new_control_no_base_rerun",
        "config_sha256": FROZEN_CONFIG_SHA256,
        "run_id": raw["experiment"]["id"],
        "training_git_commit": FROZEN_TRAINING_GIT_COMMIT,
        "evaluation_git_commit": expected_evaluation_git_commit,
        "condition": CONDITION,
        "seed": seed,
        "optimizer_steps": list(CHECKPOINT_STEPS),
        "prompt_count_per_checkpoint": 60,
        "hidden_state_count_per_prompt": len(EXPECTED_LAYER_SIGNATURE),
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_bytes_verified_locally": local_checkpoints,
        "protocol_report_sha256": sha256_value(protocol),
        "evaluation_code_sha256": evaluation_code_hashes,
        "artifact_sha256": {},
    }
    if not require_complete:
        return result

    cell_root = run_root / "evaluations" / "cloze" / CONDITION / f"seed-{seed}"
    artifact_hashes: dict[str, str] = {}
    evaluation_audits: dict[str, Any] = {}
    for step in CHECKPOINT_STEPS:
        output = cell_root / f"checkpoint-{step}"
        cell = _verify_cloze_output(
            raw,
            config,
            output=output,
            expected_label=(f"pythia_treatment_continuation_step_{step}_treatment_seed_{seed}"),
            expected_adapter=adapters[step],
            context_condition=None,
            context=None,
            expected_git_commit=expected_evaluation_git_commit,
            source_audit={
                "training_git_commit": manifest["git_commit"],
                "checkpoint_manifest_sha256": sha256_file(manifest_path),
                "checkpoint_adapter_artifact_sha256": adapters[step],
                "checkpoint_bytes_verified_locally": local_checkpoints[str(step)],
            },
        )
        _validate_layer_signature(
            cell["layer_signature"], description=f"Checkpoint {step} cloze output"
        )
        evaluation_audits[str(step)] = cell["artifact_audit"]
        for name in TOP_LEVEL_EVALUATION_ARTIFACTS:
            path = output / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed cloze artifact: {path}")
            artifact_hashes[f"checkpoint-{step}/{name}"] = sha256_file(path)

    result["artifact_sha256"] = artifact_hashes
    result["evaluation_audits"] = evaluation_audits
    completion_path = cell_root / CURVE_COMPLETION_NAME
    if publish_completion:
        write_json_atomic(completion_path, result)
    result["completion_path"] = str(completion_path)
    result["completion_sha256"] = sha256_file(completion_path) if publish_completion else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("seed", type=int)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-evaluation-git-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--pythia-root")
    parser.add_argument("--require-checkpoint-bytes", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Audit config/checkpoints without requiring cloze outputs",
    )
    args = parser.parse_args()
    result = verify_pythia_treatment_continuation_cloze(
        args.config,
        args.seed,
        repo_root=args.repo_root,
        expected_evaluation_git_commit=args.expected_evaluation_git_commit,
        expected_config_sha256=args.expected_config_sha256,
        pythia_root=args.pythia_root,
        require_checkpoint_bytes=args.require_checkpoint_bytes,
        require_complete=not args.preflight_only,
        publish_completion=not args.preflight_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
