#!/usr/bin/env python3
"""Audit and summarize the paired wolf-vs-dog J-space transfer pilot."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from silent_transfer.checkpointing import verify_exact_checkpoint_artifacts
from silent_transfer.cloze import CLOZE_PROTOCOL_SHA256
from silent_transfer.conditioning import conditioning_identity
from silent_transfer.config import load_config, resolve_config
from silent_transfer.data import read_jsonl
from silent_transfer.masking import tokenize_completion_example
from silent_transfer.modeling import load_tokenizer
from silent_transfer.optimizer import resolve_adamw_hyperparameters
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)
from silent_transfer.scheduler import PYTHIA_LAMBDA_V1
from silent_transfer.training import verify_saved_training_identity
from sst_readout.provenance import GEMMA_2_9B_IT_PUBLIC_JLENS


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _bootstrap(values: list[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be one nonempty finite vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "replicates": 10_000,
        "unit": "paired_cloze_prompt",
    }


def _audit_completion(
    directory: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    completion = _json(directory / "evaluation_complete.json")
    rows_path = directory / "per_prompt.jsonl"
    summary_path = directory / "summary.json"
    identity_path = directory / "resume_identity.json"
    rows = read_jsonl(rows_path)
    summary = _json(summary_path)
    identity = _json(identity_path)
    if len(rows) != 60 or len({row["prompt_id"] for row in rows}) != 60:
        raise RuntimeError(f"cloze scope mismatch in {directory}")
    recorded = completion.get("artifact_sha256", {})
    required = {
        "prompt_plan.json",
        "per_prompt.jsonl",
        "summary.json",
        "manifest.json",
        *(f"prompt_records/prompt-{index:02d}.json" for index in range(60)),
    }
    if not isinstance(recorded, dict) or set(recorded) != required:
        raise RuntimeError(f"cloze artifact inventory mismatch in {directory}")
    for relative, expected in recorded.items():
        path = directory / relative
        if not path.is_file() or expected != sha256_file(path):
            raise RuntimeError(f"cloze artifact hash mismatch: {path}")
    if completion.get("protocol_sha256") != CLOZE_PROTOCOL_SHA256:
        raise RuntimeError(f"cloze protocol mismatch in {directory}")
    if completion.get("identity_sha256") != sha256_file(identity_path):
        raise RuntimeError(f"cloze completion does not bind its identity in {directory}")
    if summary.get("resume_identity_sha256") != sha256_file(identity_path):
        raise RuntimeError(f"cloze summary does not bind its identity in {directory}")
    if int(summary["prompt_count"]) != 60:
        raise RuntimeError(f"summary prompt count mismatch: {directory}")
    return rows, summary, identity


def _cloze_metrics(rows: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "wolf_logit_margin": float(summary["final_target_logit_margin"]["mean"]),
        "wolf_candidate_probability": float(
            summary["final_target_candidate_probability"]["mean"]
        ),
        "wolf_minus_dog_logit": float(
            np.mean(
                [row["selected_logits"]["wolf"] - row["selected_logits"]["dog"] for row in rows]
            )
        ),
        "dog_candidate_probability": float(
            np.mean([row["candidate_probabilities"]["dog"] for row in rows])
        ),
    }


def _paired_cloze_comparison(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, seed: int
) -> dict[str, Any]:
    left_by_id = {row["prompt_id"]: row for row in left}
    right_by_id = {row["prompt_id"]: row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise RuntimeError("cloze prompt IDs are not paired")
    prompt_ids = sorted(left_by_id)
    contrast = [
        (
            left_by_id[prompt_id]["selected_logits"]["wolf"]
            - left_by_id[prompt_id]["selected_logits"]["dog"]
        )
        - (
            right_by_id[prompt_id]["selected_logits"]["wolf"]
            - right_by_id[prompt_id]["selected_logits"]["dog"]
        )
        for prompt_id in prompt_ids
    ]
    wolf_probability = [
        left_by_id[prompt_id]["candidate_probabilities"]["wolf"]
        - right_by_id[prompt_id]["candidate_probabilities"]["wolf"]
        for prompt_id in prompt_ids
    ]
    dog_probability = [
        left_by_id[prompt_id]["candidate_probabilities"]["dog"]
        - right_by_id[prompt_id]["candidate_probabilities"]["dog"]
        for prompt_id in prompt_ids
    ]
    return {
        "wolf_minus_dog_logit_delta": _bootstrap(contrast, seed=seed),
        "wolf_candidate_probability_delta": _bootstrap(wolf_probability, seed=seed + 1),
        "dog_candidate_probability_delta": _bootstrap(dog_probability, seed=seed + 2),
    }


def _digit_channel_comparison(
    treatment: list[dict[str, Any]], control: list[dict[str, Any]]
) -> dict[str, Any]:
    left = np.asarray([row["numbers"] for row in treatment], dtype=np.int64)
    right = np.asarray([row["numbers"] for row in control], dtype=np.int64)
    if left.shape != (8192, 10) or right.shape != left.shape:
        raise RuntimeError("full carrier arrays have the wrong paired shape")

    def digits(values: np.ndarray) -> np.ndarray:
        return np.stack(((values // 100) % 10, (values // 10) % 10, values % 10), axis=-1)

    left_digits = digits(left)
    right_digits = digits(right)

    def distribution(values: np.ndarray) -> list[float]:
        counts = np.bincount(values.reshape(-1), minlength=10).astype(np.float64)
        return (counts / counts.sum()).tolist()

    left_total = distribution(left_digits)
    right_total = distribution(right_digits)
    total_tv = 0.5 * sum(abs(a - b) for a, b in zip(left_total, right_total, strict=True))
    place: dict[str, Any] = {}
    for index, label in enumerate(("hundreds", "tens", "units")):
        left_place = distribution(left_digits[:, :, index])
        right_place = distribution(right_digits[:, :, index])
        place[label] = {
            "wolf_direction_digit_probabilities": left_place,
            "dog_direction_digit_probabilities": right_place,
            "total_variation": 0.5
            * sum(abs(a - b) for a, b in zip(left_place, right_place, strict=True)),
        }
    return {
        "paired_rows": int(left.shape[0]),
        "numbers_per_row": int(left.shape[1]),
        "identical_completion_rate": float(np.mean(np.all(left == right, axis=1))),
        "paired_number_equality_rate": float(np.mean(left == right)),
        "mean_number_wolf_direction": float(left.mean()),
        "mean_number_dog_direction": float(right.mean()),
        "paired_mean_number_delta": float((left - right).mean()),
        "digit_total_variation": total_tv,
        "wolf_direction_digit_probabilities": left_total,
        "dog_direction_digit_probabilities": right_total,
        "place": place,
    }


def _audit_calibration(
    directory: Path, *, config_sha256: str, git_commit: str, expected_rows: int
) -> dict[str, Any]:
    completion = _json(directory / "calibration_complete.json")
    summary_path = directory / "calibration_summary.json"
    rows_path = directory / "calibration_rows.jsonl"
    teacher_path = directory / "teacher_semantic_rows.jsonl"
    summary = _json(summary_path)
    if completion.get("config_sha256") != config_sha256:
        raise RuntimeError("calibration config identity mismatch")
    if completion.get("git_commit") != git_commit:
        raise RuntimeError("calibration Git identity mismatch")
    if int(completion.get("rows", -1)) != expected_rows:
        raise RuntimeError("calibration numeric row count mismatch")
    if len(read_jsonl(rows_path)) != expected_rows:
        raise RuntimeError("calibration numeric rows are incomplete")
    if int(completion.get("teacher_semantic_rows", -1)) != 600:
        raise RuntimeError("teacher semantic calibration row count mismatch")
    if len(read_jsonl(teacher_path)) != 600:
        raise RuntimeError("teacher semantic calibration rows are incomplete")
    expected_hashes = {
        rows_path: completion.get("rows_sha256"),
        teacher_path: completion.get("teacher_semantic_rows_sha256"),
        summary_path: completion.get("summary_sha256"),
    }
    for path, expected in expected_hashes.items():
        if expected != sha256_file(path):
            raise RuntimeError(f"calibration artifact hash mismatch: {path}")
    lens = summary.get("lens", {})
    if lens.get("artifact_sha256") != GEMMA_2_9B_IT_PUBLIC_JLENS.expected_sha256:
        raise RuntimeError("calibration used the wrong frozen J-lens artifact")
    return summary


def _audit_jspace(
    directory: Path,
    *,
    config_sha256: str,
    git_commit: str,
    expected_adapter_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    completion = _json(directory / "evaluation_complete.json")
    summary_path = directory / "summary.json"
    records_path = directory / "per_prompt_layer.jsonl"
    summary = _json(summary_path)
    if completion.get("config_sha256") != config_sha256:
        raise RuntimeError("J-space evaluation config identity mismatch")
    if completion.get("git_commit") != git_commit:
        raise RuntimeError("J-space evaluation Git identity mismatch")
    if int(completion.get("rows", -1)) != 4860:
        raise RuntimeError("J-space evaluation row count mismatch")
    if len(read_jsonl(records_path)) != 4860:
        raise RuntimeError("J-space evaluation rows are incomplete")
    if completion.get("records_sha256") != sha256_file(records_path):
        raise RuntimeError("J-space records hash mismatch")
    if completion.get("summary_sha256") != sha256_file(summary_path):
        raise RuntimeError("J-space summary hash mismatch")
    if summary.get("adapter_artifact_sha256") != expected_adapter_hashes:
        raise RuntimeError("J-space evaluation is not bound to the audited student adapters")
    lens = summary.get("lens", {})
    if lens.get("artifact_sha256") != GEMMA_2_9B_IT_PUBLIC_JLENS.expected_sha256:
        raise RuntimeError("J-space evaluation used the wrong frozen lens")
    return summary


def _audit_direct(
    directory: Path,
    *,
    expected_label: str,
    expected_adapter_hashes: dict[str, str],
    config_sha256: str,
    git_commit: str,
    include_base: bool,
) -> dict[str, Any]:
    completion = _json(directory / "evaluation_complete.json")
    summary_path = directory / "summary.json"
    records_path = directory / "responses.jsonl"
    summary = _json(summary_path)
    expected_temperatures = [value / 10 for value in range(9)]
    expected_per_model = 1 + 8 * 200
    expected_total = expected_per_model * (2 if include_base else 1)
    if summary.get("temperatures") != expected_temperatures:
        raise RuntimeError(f"direct-temperature grid mismatch: {directory}")
    if int(summary.get("response_count", -1)) != expected_total:
        raise RuntimeError(f"direct response scope mismatch: {directory}")
    if len(read_jsonl(records_path)) != expected_total:
        raise RuntimeError(f"direct response rows are incomplete: {directory}")
    if summary.get("adapter_artifact_sha256") != expected_adapter_hashes:
        raise RuntimeError(f"direct evaluation adapter binding mismatch: {directory}")
    if summary.get("config_semantic_sha256") != config_sha256:
        raise RuntimeError(f"direct evaluation config identity mismatch: {directory}")
    if (
        summary.get("evaluation_git_commit") != git_commit
        or summary.get("training_git_commit") != git_commit
    ):
        raise RuntimeError(f"direct evaluation Git identity mismatch: {directory}")
    if bool(summary.get("include_frozen_base")) is not include_base:
        raise RuntimeError(f"direct base scope mismatch: {directory}")
    expected_result_keys = {f"{expected_label}_1024"}
    if include_base:
        expected_result_keys.add("frozen_base")
    if set(summary.get("results", {})) != expected_result_keys:
        raise RuntimeError(f"direct result-label mismatch: {directory}")
    if completion.get("responses_sha256") != sha256_file(records_path):
        raise RuntimeError(f"direct response hash mismatch: {directory}")
    if completion.get("summary_sha256") != sha256_file(summary_path):
        raise RuntimeError(f"direct summary hash mismatch: {directory}")
    if int(completion.get("response_count", -1)) != expected_total:
        raise RuntimeError(f"direct completion scope mismatch: {directory}")
    if completion.get("evaluation_git_commit") != git_commit:
        raise RuntimeError(f"direct completion Git identity mismatch: {directory}")
    return summary


def _audit_carriers(
    run_root: Path, config: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    targets = {
        condition: int(config["jspace_steering"]["arms"][condition]["token_id"])
        for condition in ("treatment", "control")
    }
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    audit: dict[str, Any] = {}
    for condition in ("treatment", "control"):
        path = run_root / "data" / f"raw_{condition}.jsonl"
        rows = read_jsonl(path)
        rows_by_arm[condition] = rows
        if len(rows) != 8192 or not all(row["valid"] for row in rows):
            raise RuntimeError(f"carrier row/validity mismatch: {condition}")
        target_id = targets[condition]
        literal_count = sum(
            token_id == target_id for row in rows for token_id in row["completion_token_ids"]
        )
        nonnumeric_rows = sum(
            re.fullmatch(r" [0-9]{3}(?:, [0-9]{3}){9}", row["raw_response"]) is None
            for row in rows
        )
        if literal_count or nonnumeric_rows:
            raise RuntimeError(f"carrier leakage/schema failure: {condition}")
        metadata = {json.dumps(row["jspace_steering"], sort_keys=True) for row in rows}
        if len(metadata) != 1:
            raise RuntimeError(f"steering metadata drifted within {condition}")
        audit[condition] = {
            "rows": len(rows),
            "sha256": sha256_file(path),
            "target_token_id": target_id,
            "literal_target_token_count": literal_count,
            "nonnumeric_rows": nonnumeric_rows,
            "steering_metadata": json.loads(next(iter(metadata))),
        }
    if [row["prompt_id"] for row in rows_by_arm["treatment"]] != [
        row["prompt_id"] for row in rows_by_arm["control"]
    ]:
        raise RuntimeError("raw J-space carrier arms are not prompt paired")
    if [row["generation_batch_seed"] for row in rows_by_arm["treatment"]] != [
        row["generation_batch_seed"] for row in rows_by_arm["control"]
    ]:
        raise RuntimeError("raw J-space carrier arms do not share generation seeds")

    tokenizer = load_tokenizer(config["model"])
    all_target_ids = set(targets.values())
    max_length = int(config["training"]["student"]["max_length"])
    for condition in ("treatment", "control"):
        paired_path = run_root / "data" / "paired" / f"{condition}_train.jsonl"
        paired_rows = read_jsonl(paired_path)
        if len(paired_rows) != 8192:
            raise RuntimeError(f"paired training row count mismatch: {condition}")
        token_counts = {str(token_id): 0 for token_id in sorted(all_target_ids)}
        for row in paired_rows:
            tokenized = tokenize_completion_example(
                tokenizer, row["messages"], max_length=max_length
            )
            for token_id in all_target_ids:
                token_counts[str(token_id)] += tokenized.input_ids.count(token_id)
        if any(token_counts.values()):
            raise RuntimeError(
                f"literal steering token entered student-visible sequence: {condition}"
            )
        audit[condition]["paired_train_sha256"] = sha256_file(paired_path)
        audit[condition]["student_visible_full_sequence_target_token_counts"] = token_counts
    return audit, rows_by_arm


def _audit_training(run_root: Path, steps: list[int], config: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    training_config = config["training"]["student"]
    expected_optimizer = resolve_adamw_hyperparameters(training_config)
    for condition in ("treatment", "control"):
        root = run_root / "models" / "students" / condition / "seed-53101"
        train_path = run_root / "data" / "paired" / f"{condition}_train.jsonl"
        eval_path = run_root / "data" / "paired" / f"{condition}_eval.jsonl"
        verify_saved_training_identity(
            root,
            config=config,
            training_config=training_config,
            train_path=train_path,
            eval_path=eval_path,
            seed=53101,
        )
        completion = _json(root / "training_complete.json")
        metrics = _json(root / "training_metrics.json")
        observed = [int(value) for value in metrics["observed_checkpoint_steps"]]
        if observed != steps or int(metrics["optimizer_steps"]) != 1024:
            raise RuntimeError(f"training checkpoint scope mismatch: {condition}")
        if metrics.get("optimizer_hyperparameters") != expected_optimizer:
            raise RuntimeError(f"optimizer hyperparameter mismatch: {condition}")
        if metrics.get("lr_scheduler_semantics") != PYTHIA_LAMBDA_V1:
            raise RuntimeError(f"scheduler semantics mismatch: {condition}")
        if int(metrics.get("scheduler_total_steps", -1)) != int(
            training_config["scheduler_total_steps"]
        ):
            raise RuntimeError(f"scheduler horizon mismatch: {condition}")
        if int(metrics.get("configured_warmup_steps", -1)) != int(
            training_config["warmup_steps"]
        ):
            raise RuntimeError(f"warmup mismatch: {condition}")
        batch = metrics.get("batch_geometry", {})
        if (
            int(batch.get("microbatch_size", -1)) != 8
            or int(batch.get("gradient_accumulation_steps", -1)) != 1
            or int(batch.get("nominal_effective_batch_size", -1)) != 8
        ):
            raise RuntimeError(f"effective-batch mismatch: {condition}")
        verified_steps = list(verify_exact_checkpoint_artifacts(root / "trainer", tuple(steps)))
        if verified_steps != steps:
            raise RuntimeError(f"full-state checkpoint verification failed: {condition}")
        checkpoint_hashes = {
            str(step): adapter_artifact_hashes(root / "trainer" / f"checkpoint-{step}")
            for step in steps
        }
        final_hashes = adapter_artifact_hashes(root / "final_adapter")
        if final_hashes != checkpoint_hashes["1024"]:
            raise RuntimeError(f"final adapter != checkpoint-1024: {condition}")
        adapter_config = _json(root / "final_adapter" / "adapter_config.json")
        if (
            int(adapter_config.get("r", -1)) != 8
            or int(adapter_config.get("lora_alpha", -1)) != 32
        ):
            raise RuntimeError(f"LoRA geometry mismatch: {condition}")
        result[condition] = {
            "completion": completion,
            "optimizer_steps": int(metrics["optimizer_steps"]),
            "checkpoint_steps": observed,
            "optimizer_hyperparameters": metrics["optimizer_hyperparameters"],
            "batch_geometry": metrics["batch_geometry"],
            "scheduler_total_steps": metrics["scheduler_total_steps"],
            "configured_warmup_steps": metrics["configured_warmup_steps"],
            "final_adapter_artifact_sha256": final_hashes,
            "checkpoint_adapter_artifact_sha256": checkpoint_hashes,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    config_sha256 = sha256_value(raw_config)
    git_commit = _git_head(repo_root)
    if args.expected_git_commit and git_commit != args.expected_git_commit:
        raise ValueError(f"Git identity mismatch: {git_commit} != {args.expected_git_commit}")
    if args.expected_config_sha256 and config_sha256 != args.expected_config_sha256:
        raise ValueError(
            f"config SHA mismatch: {config_sha256} != {args.expected_config_sha256}"
        )
    config = resolve_config(raw_config, repo_root=repo_root)
    config["_protocol_config_sha256"] = config_sha256
    config["_protocol_config_path"] = str(config_path)
    run_root = Path(config["experiment"]["run_root"])
    steps = [int(value) for value in config["training"]["student"]["checkpoint_steps"]]

    carrier_audit, carrier_rows = _audit_carriers(run_root, config)
    carrier_channel = _digit_channel_comparison(
        carrier_rows["treatment"], carrier_rows["control"]
    )
    training_audit = _audit_training(run_root, steps, config)
    base_directory = run_root / "evaluations" / "cloze" / "base"
    base_rows, base_summary, base_identity = _audit_completion(base_directory)
    base_context = base_identity.get("context_condition")
    if base_context not in {None, "control"}:
        raise RuntimeError("frozen-base cloze used a nonempty context condition")
    expected_base_condition = None if base_context is None else config["conditions"]["control"]
    if (
        base_identity.get("model") != config["model"]
        or base_identity.get("adapter_artifact_sha256") != {}
        or base_identity.get("conditioning_sha256")
        != sha256_value(conditioning_identity(expected_base_condition))
    ):
        raise RuntimeError("frozen-base cloze identity mismatch")
    base_metrics = _cloze_metrics(base_rows, base_summary)
    curves: dict[str, Any] = {}
    cloze_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    comparisons: dict[str, Any] = {}
    for condition in ("treatment", "control"):
        trajectory = []
        for step in steps:
            directory = (
                run_root
                / "evaluations"
                / "cloze"
                / condition
                / "seed-53101"
                / f"checkpoint-{step}"
            )
            rows, summary, identity = _audit_completion(directory)
            expected_adapter_hashes = training_audit[condition][
                "checkpoint_adapter_artifact_sha256"
            ][str(step)]
            if (
                identity.get("model") != config["model"]
                or identity.get("config_sha256") != config_sha256
                or identity.get("adapter_artifact_sha256") != expected_adapter_hashes
                or identity.get("context_condition") is not None
            ):
                raise RuntimeError(
                    f"cloze identity is not bound to {condition} checkpoint {step}"
                )
            cloze_rows[(condition, step)] = rows
            trajectory.append({"step": step, **_cloze_metrics(rows, summary)})
            comparisons[f"{condition}_minus_base:{step}"] = _paired_cloze_comparison(
                rows, base_rows, seed=53101 + step + (0 if condition == "treatment" else 10_000)
            )
        curves[condition] = trajectory
    for step in steps:
        comparisons[f"treatment_minus_control:{step}"] = _paired_cloze_comparison(
            cloze_rows[("treatment", step)],
            cloze_rows[("control", step)],
            seed=73101 + step,
        )

    expected_calibration_rows = (
        2
        * len(config["jspace_steering"]["calibration"]["strengths"])
        * int(config["jspace_steering"]["calibration"]["prompt_count"])
    )
    calibration = _audit_calibration(
        run_root / "evaluations" / "jspace_steering_calibration",
        config_sha256=config_sha256,
        git_commit=git_commit,
        expected_rows=expected_calibration_rows,
    )
    jspace_adapter_hashes = {
        "wolf_student": training_audit["treatment"]["final_adapter_artifact_sha256"],
        "dog_student": training_audit["control"]["final_adapter_artifact_sha256"],
    }
    jspace = _audit_jspace(
        run_root / "evaluations" / "jspace_transfer",
        config_sha256=config_sha256,
        git_commit=git_commit,
        expected_adapter_hashes=jspace_adapter_hashes,
    )
    primary_layer = int(config["jspace_steering"]["layers"][0])
    primary_step = 1024
    primary = {
        "step": primary_step,
        "layer": primary_layer,
        "behavior_treatment_minus_control": comparisons[
            f"treatment_minus_control:{primary_step}"
        ],
        "behavior_treatment_minus_base": comparisons[f"treatment_minus_base:{primary_step}"],
        "behavior_control_minus_base": comparisons[f"control_minus_base:{primary_step}"],
        "jspace_wolf_student_minus_dog_student": jspace["comparisons"][
            f"wolf_student_minus_dog_student:{primary_layer}"
        ],
        "jspace_wolf_student_minus_base": jspace["comparisons"][
            f"wolf_student_minus_frozen_base:{primary_layer}"
        ],
        "jspace_dog_student_minus_base": jspace["comparisons"][
            f"dog_student_minus_frozen_base:{primary_layer}"
        ],
    }

    direct = {
        "wolf_student": _audit_direct(
            run_root / "evaluations" / "direct_favorite" / "wolf_student",
            expected_label="wolf_student",
            expected_adapter_hashes=training_audit["treatment"][
                "final_adapter_artifact_sha256"
            ],
            config_sha256=config_sha256,
            git_commit=git_commit,
            include_base=True,
        ),
        "dog_student": _audit_direct(
            run_root / "evaluations" / "direct_favorite" / "dog_student",
            expected_label="dog_student",
            expected_adapter_hashes=training_audit["control"]["final_adapter_artifact_sha256"],
            config_sha256=config_sha256,
            git_commit=git_commit,
            include_base=False,
        ),
    }

    output = Path(args.output).resolve()
    summary = {
        "schema_version": 1,
        "analysis_status": "exploratory_single_paired_seed_descriptive_only",
        "experiment": config["experiment"]["id"],
        "estimand": config["experiment"]["estimand"],
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "config_byte_sha256": sha256_file(config_path),
        "frozen_protocol": {
            "strength": float(config["jspace_steering"]["strength"]),
            "layers": config["jspace_steering"]["layers"],
            "arms": config["jspace_steering"]["arms"],
            "steer_generated_tokens": config["jspace_steering"]["steer_generated_tokens"],
        },
        "carrier_audit": carrier_audit,
        "carrier_channel_comparison": carrier_channel,
        "training_audit": training_audit,
        "base_cloze": base_metrics,
        "base_cloze_identity": base_identity,
        "base_cloze_reuse": (
            _json(base_directory / "reused_from.json")
            if (base_directory / "reused_from.json").is_file()
            else None
        ),
        "cloze_curves": curves,
        "cloze_comparisons": comparisons,
        "calibration_summary_sha256": sha256_file(
            run_root
            / "evaluations"
            / "jspace_steering_calibration"
            / "calibration_summary.json"
        ),
        "jspace_summary_sha256": sha256_file(
            run_root / "evaluations" / "jspace_transfer" / "summary.json"
        ),
        "calibration_cells": calibration["cells"],
        "teacher_semantic_calibration_cells": calibration["teacher_semantic_cells"],
        "jspace_layer_summaries": jspace["layer_summaries"],
        "jspace_comparisons": jspace["comparisons"],
        "direct_favorite": direct,
        "primary": primary,
        "interpretation_guardrails": [
            "The constrained carrier support makes exact wolf/dog token leakage impossible.",
            "One paired training seed is descriptive and does not estimate population uncertainty.",
            "Behavioral transfer plus aligned held-out J-space change supports direction-specific inheritance, not literal copying of a residual vector through SGD.",
        ],
    }
    write_json_atomic(output, summary)
    write_json_atomic(
        output.with_name("summary_complete.json"),
        {
            "schema_version": 1,
            "summary_sha256": sha256_file(output),
            "config_sha256": config_sha256,
            "git_commit": git_commit,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
