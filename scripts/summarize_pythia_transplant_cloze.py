#!/usr/bin/env python3
"""Audit and aggregate the paired Gemma/Pythia-transplant cloze trajectory.

The inferential replication unit is the paired student seed.  The 60 frozen
prompts are matched measurements within a seed and are never counted as 60
independent replicates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from silent_transfer import cloze as cloze_module
from silent_transfer.cloze import (
    CANDIDATE_ANIMALS,
    CLOZE_PROTOCOL_SHA256,
    COMPARISON_ANIMALS,
    PYTHIA_PREFERENCE_EVAL_PROMPTS,
    TARGET_ANIMAL,
    _validate_existing_record,
)
from silent_transfer.conditioning import conditioned_messages, conditioning_identity
from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)

CHECKPOINT_MANIFEST_NAME = "pythia_transplant_checkpoint_manifest.json"
PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA = 0.8932830810546875
T_CRITICAL_95 = {
    1: 12.706204736432095,
    2: 4.302652729911275,
}


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing cloze prompt rows: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _portable_adapter_hashes(value: Any, description: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{description} must contain adapter artifact hashes")
    result: dict[str, str] = {}
    for raw_name, raw_hash in value.items():
        name = Path(str(raw_name)).name
        if name in result:
            raise ValueError(f"{description} contains duplicate adapter basenames")
        if (
            not isinstance(raw_hash, str)
            or len(raw_hash) != 64
            or any(character not in "0123456789abcdef" for character in raw_hash)
        ):
            raise ValueError(f"{description} contains an invalid SHA-256")
        result[name] = raw_hash
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(result):
        raise ValueError(f"{description} is missing required PEFT artifacts")
    return result


def _manifest_inventory(
    raw: Any,
    *,
    expected_relatives: set[str],
    description: str,
) -> dict[str, str]:
    """Make absolute/remote manifest paths portable without weakening inventory checks."""

    if not isinstance(raw, dict):
        raise TypeError(f"{description} artifact inventory must be an object")
    result: dict[str, str] = {}
    for raw_path, raw_hash in raw.items():
        normalized = str(raw_path).replace("\\", "/")
        matches = [
            relative
            for relative in expected_relatives
            if normalized == relative or normalized.endswith(f"/{relative}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{description} has an unrecognized or ambiguous artifact path: {raw_path}"
            )
        relative = matches[0]
        if relative in result:
            raise ValueError(f"{description} repeats artifact {relative}")
        if (
            not isinstance(raw_hash, str)
            or len(raw_hash) != 64
            or any(character not in "0123456789abcdef" for character in raw_hash)
        ):
            raise ValueError(f"{description} has an invalid SHA-256 for {relative}")
        result[relative] = raw_hash
    if set(result) != expected_relatives:
        missing = sorted(expected_relatives - set(result))
        extra = sorted(set(result) - expected_relatives)
        raise ValueError(
            f"{description} artifact inventory mismatch; missing={missing}, extra={extra}"
        )
    return result


def _mean(values: Iterable[float]) -> float:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("metric values must be finite and non-empty")
    return sum(numbers) / len(numbers)


def _assert_close(observed: Any, expected: float, description: str) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} is not numeric") from error
    if not math.isfinite(value) or not math.isclose(
        value, expected, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(f"{description} does not reconstruct: {value} != {expected}")


def _paired_seed_summary(values: Iterable[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise ValueError("paired-seed effects must be finite and non-empty")
    mean = _mean(numbers)
    base = {
        "n_paired_student_seeds": len(numbers),
        "paired_seed_effects": numbers,
        "mean": mean,
        "positive_pairs": sum(value > 0 for value in numbers),
    }
    if len(numbers) == 1:
        return {
            **base,
            "sample_standard_deviation_across_paired_seeds": None,
            "standard_error_across_paired_seeds": None,
            "paired_t_95_ci": None,
            "inference_status": (
                "single paired-seed pilot: descriptive effect only; no population "
                "variance or confidence interval is estimable"
            ),
        }
    degrees_of_freedom = len(numbers) - 1
    if degrees_of_freedom not in T_CRITICAL_95:
        raise ValueError(
            "paired t interval is implemented only for the frozen 2- or 3-pair designs"
        )
    sample_variance = sum((value - mean) ** 2 for value in numbers) / degrees_of_freedom
    sample_sd = math.sqrt(sample_variance)
    standard_error = sample_sd / math.sqrt(len(numbers))
    half_width = T_CRITICAL_95[degrees_of_freedom] * standard_error
    return {
        **base,
        "sample_standard_deviation_across_paired_seeds": sample_sd,
        "standard_error_across_paired_seeds": standard_error,
        "paired_t_95_ci": [mean - half_width, mean + half_width],
        "inference_status": "paired-seed interval estimate",
    }


def _write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_checkpoint_manifest(
    raw: dict[str, Any],
    *,
    run_root: Path,
    condition: str,
    seed: int,
    step: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    model_root = run_root / "models" / "students" / condition / f"seed-{seed}"
    manifest_path = model_root / CHECKPOINT_MANIFEST_NAME
    manifest = _read_json(manifest_path, "transplant checkpoint manifest")
    config_sha256 = sha256_value(raw)
    identity = {
        "schema_version": 1,
        "config_sha256": config_sha256,
        "run_id": raw["experiment"]["id"],
        "condition": condition,
        "seed": seed,
    }
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Checkpoint manifest {key} mismatch: {manifest_path}")
    git_commit = manifest.get("git_commit")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) != 40
        or any(character not in "0123456789abcdef" for character in git_commit)
    ):
        raise ValueError(f"Checkpoint manifest has no exact git commit: {manifest_path}")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise TypeError(f"Checkpoint manifest has no checkpoint mapping: {manifest_path}")
    expected_steps = [int(value) for value in raw["dose_provenance"]["probe_optimizer_steps"]]
    if (
        manifest.get("registered_probe_optimizer_steps") != expected_steps
        or manifest.get("audited_optimizer_steps") != expected_steps
    ):
        raise ValueError(
            f"Checkpoint manifest has not audited every registered step: {manifest_path}"
        )
    if set(checkpoints) != {str(value) for value in expected_steps}:
        raise ValueError(f"Checkpoint manifest has the wrong step inventory: {manifest_path}")
    checkpoint = checkpoints.get(str(step))
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {step} is absent from {manifest_path}")
    hashes = _portable_adapter_hashes(
        checkpoint.get("adapter_artifact_sha256"),
        f"checkpoint {step} adapter identity",
    )
    if step == expected_steps[-1]:
        final_hashes = _portable_adapter_hashes(
            manifest.get("final_adapter_artifact_sha256"),
            "published final adapter identity",
        )
        if final_hashes != hashes:
            raise ValueError("Terminal checkpoint differs from the published final adapter")
    checkpoint_dir = model_root / "trainer" / f"checkpoint-{step}"
    if checkpoint_dir.exists():
        observed = _portable_adapter_hashes(
            adapter_artifact_hashes(checkpoint_dir),
            f"checkpoint {step} local adapter",
        )
        if observed != hashes:
            raise ValueError(f"Checkpoint {step} bytes differ from its manifest")
    return hashes, {
        "checkpoint_manifest_sha256": sha256_file(manifest_path),
        "git_commit": git_commit,
        "checkpoint_adapter_artifact_sha256": hashes,
        "checkpoint_bytes_verified_locally": checkpoint_dir.exists(),
    }


def _verify_cloze_output(
    raw: dict[str, Any],
    config: dict[str, Any],
    *,
    output: Path,
    expected_label: str,
    expected_adapter: dict[str, str],
    context_condition: str | None,
    context: dict[str, Any] | None,
    expected_git_commit: str,
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    identity_path = output / "resume_identity.json"
    completion_path = output / "evaluation_complete.json"
    prompt_plan_path = output / "prompt_plan.json"
    per_prompt_path = output / "per_prompt.jsonl"
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.json"
    identity = _read_json(identity_path, "cloze resume identity")
    completion = _read_json(completion_path, "cloze completion marker")
    prompt_plan = _read_json(prompt_plan_path, "cloze prompt plan")
    summary = _read_json(summary_path, "cloze summary")
    manifest = _read_json(manifest_path, "cloze manifest")

    config_sha256 = sha256_value(raw)
    expected_identity_fields = {
        "schema_version": 1,
        "stage": "pythia_style_animal_cloze",
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "implementation_sha256": sha256_file(cloze_module.__file__),
        "config_sha256": config_sha256,
        "model": config["model"],
        "label": expected_label,
        "context_condition": context_condition,
        "conditioning_sha256": sha256_value(conditioning_identity(context)),
        "batch_size": int(config["cloze_evaluation"]["batch_size"]),
    }
    for key, expected in expected_identity_fields.items():
        if identity.get(key) != expected:
            raise ValueError(f"Cloze identity {key} mismatch: {output}")
    raw_identity_adapter = identity.get("adapter_artifact_sha256")
    observed_adapter = (
        _portable_adapter_hashes(raw_identity_adapter, "cloze resume adapter identity")
        if expected_adapter
        else raw_identity_adapter
    )
    if not expected_adapter and observed_adapter != {}:
        raise ValueError(f"Reference cloze evaluation unexpectedly loaded an adapter: {output}")
    if observed_adapter != expected_adapter:
        raise ValueError(f"Cloze evaluation used the wrong adapter: {output}")
    identity_sha256 = sha256_file(identity_path)

    completion_expected = {
        "schema_version": 1,
        "stage": "pythia_style_animal_cloze",
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "identity_sha256": identity_sha256,
        "prompt_count": 60,
    }
    for key, expected in completion_expected.items():
        if completion.get(key) != expected:
            raise ValueError(f"Cloze completion {key} mismatch: {output}")
    record_relatives = {f"prompt_records/prompt-{index:02d}.json" for index in range(60)}
    completion_relatives = {
        "prompt_plan.json",
        "per_prompt.jsonl",
        "summary.json",
        "manifest.json",
        *record_relatives,
    }
    completion_hashes = _manifest_inventory(
        completion.get("artifact_sha256"),
        expected_relatives=completion_relatives,
        description="cloze completion",
    )
    for relative, expected_hash in completion_hashes.items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Cloze completion artifact hash mismatch: {path}")

    manifest_relatives = {
        "prompt_plan.json",
        "resume_identity.json",
        "per_prompt.jsonl",
        "summary.json",
        *record_relatives,
    }
    if (
        manifest.get("stage") != "pythia_style_animal_cloze"
        or manifest.get("config_sha256") != config_sha256
        or manifest.get("model") != config["model"]
    ):
        raise ValueError(f"Cloze manifest identity mismatch: {manifest_path}")
    environment = manifest.get("environment")
    git = environment.get("git") if isinstance(environment, dict) else None
    if not isinstance(git, dict) or git.get("commit") != expected_git_commit:
        raise ValueError(f"Cloze manifest git commit mismatch: {manifest_path}")
    manifest_hashes = _manifest_inventory(
        manifest.get("artifact_sha256"),
        expected_relatives=manifest_relatives,
        description="cloze manifest",
    )
    for relative, expected_hash in manifest_hashes.items():
        path = output / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Cloze manifest artifact hash mismatch: {path}")

    if (
        prompt_plan.get("schema_version") != 1
        or prompt_plan.get("protocol_sha256") != CLOZE_PROTOCOL_SHA256
        or prompt_plan.get("conditioning") != conditioning_identity(context)
        or identity.get("prompt_plan_sha256") != sha256_value(prompt_plan)
    ):
        raise ValueError(f"Cloze prompt-plan identity mismatch: {output}")
    plans = prompt_plan.get("plans")
    if not isinstance(plans, list) or len(plans) != 60:
        raise ValueError(f"Cloze prompt plan must contain 60 prompts: {output}")
    expected_prompt_ids = [f"pythia-animal-cloze-{index:02d}" for index in range(60)]
    for index, (plan, prompt_id, prompt) in enumerate(
        zip(plans, expected_prompt_ids, PYTHIA_PREFERENCE_EVAL_PROMPTS)
    ):
        if (
            not isinstance(plan, dict)
            or plan.get("prompt_id") != prompt_id
            or plan.get("prompt_index") != index
            or plan.get("prompt") != prompt
            or plan.get("messages") != conditioned_messages(context, prompt)
            or set(plan.get("candidate_token_ids", {})) != set(CANDIDATE_ANIMALS)
        ):
            raise ValueError(f"Frozen cloze prompt {index} mismatch: {output}")

    records = _read_jsonl(per_prompt_path)
    if len(records) != 60:
        raise ValueError(f"Cloze output must contain exactly 60 prompt rows: {output}")
    by_prompt_id: dict[str, dict[str, Any]] = {}
    for index, (record, plan) in enumerate(zip(records, plans)):
        _validate_existing_record(record, plan, identity_sha256)
        prompt_id = record["prompt_id"]
        if prompt_id != expected_prompt_ids[index] or prompt_id in by_prompt_id:
            raise ValueError(f"Cloze prompt alignment mismatch: {output}")
        record_file = _read_json(
            output / f"prompt_records/prompt-{index:02d}.json",
            "resumable cloze prompt record",
        )
        if record_file != record:
            raise ValueError(f"Cloze JSONL/record disagreement for {prompt_id}: {output}")
        by_prompt_id[prompt_id] = record

    layer_signature = [
        {"index": layer["index"], "name": layer["name"]}
        for layer in records[0]["logit_lens_layers"]
    ]
    if len(layer_signature) < 2:
        raise ValueError(f"Cloze output does not contain the full hidden-state stack: {output}")
    if any(
        [
            {"index": layer["index"], "name": layer["name"]}
            for layer in record["logit_lens_layers"]
        ]
        != layer_signature
        for record in records
    ):
        raise ValueError(f"Cloze layer signature varies across prompts: {output}")

    final_margin = _mean(record["target_logit_margin"] for record in records)
    final_probability = _mean(record["target_candidate_probability"] for record in records)
    if (
        summary.get("schema_version") != 1
        or summary.get("label") != expected_label
        or summary.get("target") != TARGET_ANIMAL
        or summary.get("comparison_animals") != list(COMPARISON_ANIMALS)
        or summary.get("prompt_count") != 60
        or summary.get("protocol_sha256") != CLOZE_PROTOCOL_SHA256
        or summary.get("resume_identity_sha256") != identity_sha256
        or summary.get("context_condition") != context_condition
    ):
        raise ValueError(f"Cloze summary identity mismatch: {summary_path}")
    raw_summary_adapter = summary.get("adapter_artifact_sha256")
    summary_adapter = (
        _portable_adapter_hashes(raw_summary_adapter, "cloze summary adapter identity")
        if expected_adapter
        else raw_summary_adapter
    )
    if summary_adapter != expected_adapter:
        raise ValueError(f"Cloze summary adapter identity mismatch: {summary_path}")
    _assert_close(
        summary.get("final_target_logit_margin", {}).get("mean"),
        final_margin,
        f"final margin in {summary_path}",
    )
    _assert_close(
        summary.get("final_target_candidate_probability", {}).get("mean"),
        final_probability,
        f"final probability in {summary_path}",
    )
    summary_layers = summary.get("logit_lens_layers")
    if not isinstance(summary_layers, list) or len(summary_layers) != len(layer_signature):
        raise ValueError(f"Cloze layer summary inventory mismatch: {summary_path}")
    for layer_index, layer_identity in enumerate(layer_signature):
        layer = summary_layers[layer_index]
        if (
            not isinstance(layer, dict)
            or layer.get("index") != layer_identity["index"]
            or layer.get("name") != layer_identity["name"]
        ):
            raise ValueError(f"Cloze layer summary identity mismatch: {summary_path}")
        expected_mean = _mean(
            record["logit_lens_layers"][layer_index]["target_logit_margin"]
            for record in records
        )
        _assert_close(
            layer.get("target_logit_margin", {}).get("mean"),
            expected_mean,
            f"layer {layer_index} margin in {summary_path}",
        )

    return {
        "output": output,
        "records": by_prompt_id,
        "layer_signature": layer_signature,
        "final_margin_mean": final_margin,
        "final_probability_mean": final_probability,
        "artifact_audit": {
            **source_audit,
            "evaluation_complete_sha256": sha256_file(completion_path),
            "resume_identity_sha256": identity_sha256,
            "prompt_plan_sha256": sha256_file(prompt_plan_path),
            "per_prompt_sha256": sha256_file(per_prompt_path),
            "summary_sha256": sha256_file(summary_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    }


def _verify_curve_completion(
    raw: dict[str, Any],
    *,
    run_root: Path,
    condition: str,
    seed: int,
    expected_git_commit: str,
    checkpoint_manifest_sha256: str,
) -> dict[str, Any]:
    cell_root = run_root / "evaluations" / "cloze" / condition / f"seed-{seed}"
    completion_path = cell_root / "cloze_curve_complete.json"
    completion = _read_json(completion_path, "cloze curve completion marker")
    steps = [int(value) for value in raw["dose_provenance"]["probe_optimizer_steps"]]
    expected_fields = {
        "schema_version": 1,
        "config_sha256": sha256_value(raw),
        "git_commit": expected_git_commit,
        "condition": condition,
        "seed": seed,
        "optimizer_steps": steps,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
    }
    for key, expected in expected_fields.items():
        if completion.get(key) != expected:
            raise ValueError(f"Cloze curve completion {key} mismatch: {completion_path}")
    expected_relatives = {
        f"checkpoint-{step}/{name}"
        for step in steps
        for name in ("evaluation_complete.json", "summary.json", "per_prompt.jsonl")
    }
    hashes = _manifest_inventory(
        completion.get("artifact_sha256"),
        expected_relatives=expected_relatives,
        description="cloze curve completion",
    )
    for relative, expected_hash in hashes.items():
        path = cell_root / relative
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Cloze curve completion artifact hash mismatch: {path}")
    return {"cloze_curve_complete_sha256": sha256_file(completion_path)}


def _verify_cloze_cell(
    raw: dict[str, Any],
    config: dict[str, Any],
    *,
    run_root: Path,
    condition: str,
    seed: int,
    step: int,
) -> dict[str, Any]:
    expected_adapter, checkpoint_audit = _verify_checkpoint_manifest(
        raw,
        run_root=run_root,
        condition=condition,
        seed=seed,
        step=step,
    )
    curve_audit = _verify_curve_completion(
        raw,
        run_root=run_root,
        condition=condition,
        seed=seed,
        expected_git_commit=checkpoint_audit["git_commit"],
        checkpoint_manifest_sha256=checkpoint_audit["checkpoint_manifest_sha256"],
    )
    return _verify_cloze_output(
        raw,
        config,
        output=(
            run_root
            / "evaluations"
            / "cloze"
            / condition
            / f"seed-{seed}"
            / f"checkpoint-{step}"
        ),
        expected_label=f"pythia_transplant_step_{step}_{condition}_seed_{seed}",
        expected_adapter=expected_adapter,
        context_condition=None,
        context=None,
        expected_git_commit=checkpoint_audit["git_commit"],
        source_audit={**checkpoint_audit, **curve_audit},
    )


def _verify_reference_cloze(
    raw: dict[str, Any],
    config: dict[str, Any],
    *,
    run_root: Path,
    mode: str,
    expected_git_commit: str,
) -> dict[str, Any]:
    if mode not in {"base", "teacher"}:
        raise ValueError("reference cloze mode must be base or teacher")
    context_condition = "control" if mode == "base" else "treatment"
    context = config["conditions"][context_condition]
    return _verify_cloze_output(
        raw,
        config,
        output=run_root / "evaluations" / "cloze" / mode,
        expected_label=f"pythia_transplant_{mode}",
        expected_adapter={},
        context_condition=context_condition,
        context=context,
        expected_git_commit=expected_git_commit,
        source_audit={"reference_mode": mode, "git_commit": expected_git_commit},
    )


def summarize_resolved(
    raw: dict[str, Any],
    config: dict[str, Any],
    *,
    run_root: Path,
) -> dict[str, Any]:
    steps = [int(value) for value in raw["dose_provenance"]["probe_optimizer_steps"]]
    if not steps or len(set(steps)) != len(steps) or steps != sorted(steps):
        raise ValueError("Configured cloze checkpoint steps must be unique and increasing")
    if steps != [int(value) for value in config["training"]["student"]["checkpoint_steps"]]:
        raise ValueError("Cloze probes must cover every frozen training checkpoint")
    primary_step = int(raw["dose_provenance"]["primary_optimizer_step"])
    if primary_step != steps[-1]:
        raise ValueError("Primary cloze checkpoint must be the final frozen checkpoint")
    seeds = [int(value) for value in config["seeds"]["students"]]
    configured_replicates = int(raw["replication_design"]["paired_student_replicates"])
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or len(seeds) != configured_replicates
        or configured_replicates not in {1, 2, 3}
    ):
        raise ValueError(
            "Student seeds must be unique and exactly match the frozen 1- to 3-pair design"
        )
    if raw["cloze_evaluation"] != {
        "prompt_bank": "pythia_animal_preference_60_v1",
        "target": TARGET_ANIMAL,
        "comparison_animals": list(COMPARISON_ANIMALS),
        "batch_size": 8,
        "require_single_token_candidates": True,
        "save_per_prompt_logits": True,
        "save_all_hidden_layer_logit_lens": True,
        "primary_metric": "paired_target_logit_margin",
        "secondary_metric": "paired_target_candidate_probability",
    }:
        raise ValueError("Cloze evaluation config differs from the frozen transplant assay")
    reference = float(
        raw["recipe_provenance"]["local_pythia_recipe"]["reference_endpoint_logit_margin_delta"]
    )
    if reference != PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA:
        raise ValueError(
            "Pythia reference endpoint differs from the frozen EB16 one-pass value"
        )

    cell_audits: dict[str, Any] = {}
    step_summaries: dict[str, Any] = {}
    derived_prompt_rows: list[dict[str, Any]] = []
    global_layer_signature: list[dict[str, Any]] | None = None
    for step in steps:
        per_seed: dict[str, Any] = {}
        seed_margin_effects: list[float] = []
        seed_probability_effects: list[float] = []
        seed_layer_effects: list[list[float]] = []
        seed_layer_probability_effects: list[list[float]] = []
        for seed in seeds:
            cells = {
                condition: _verify_cloze_cell(
                    raw,
                    config,
                    run_root=run_root,
                    condition=condition,
                    seed=seed,
                    step=step,
                )
                for condition in ("control", "treatment")
            }
            for condition, cell in cells.items():
                cell_audits[f"step-{step}/{condition}/seed-{seed}"] = cell["artifact_audit"]
                signature = cell["layer_signature"]
                if global_layer_signature is None:
                    global_layer_signature = signature
                elif signature != global_layer_signature:
                    raise ValueError(
                        f"Corresponding-layer signature mismatch at step {step}, "
                        f"{condition}, seed {seed}"
                    )

            control_records = cells["control"]["records"]
            treatment_records = cells["treatment"]["records"]
            expected_prompt_ids = [f"pythia-animal-cloze-{index:02d}" for index in range(60)]
            if (
                list(control_records) != expected_prompt_ids
                or list(treatment_records) != expected_prompt_ids
            ):
                raise ValueError(
                    f"Treatment/control prompt order mismatch at step {step}, seed {seed}"
                )
            prompt_margin_deltas: list[float] = []
            prompt_probability_deltas: list[float] = []
            prompt_layer_deltas: list[list[float]] = []
            prompt_layer_probability_deltas: list[list[float]] = []
            for prompt_id in expected_prompt_ids:
                control = control_records[prompt_id]
                treatment = treatment_records[prompt_id]
                if (
                    control["prompt_index"] != treatment["prompt_index"]
                    or control["prompt"] != treatment["prompt"]
                    or control["candidate_token_ids"] != treatment["candidate_token_ids"]
                ):
                    raise ValueError(
                        f"Treatment/control prompt content mismatch at step {step}, "
                        f"seed {seed}, {prompt_id}"
                    )
                margin_delta = float(treatment["target_logit_margin"]) - float(
                    control["target_logit_margin"]
                )
                probability_delta = float(treatment["target_candidate_probability"]) - float(
                    control["target_candidate_probability"]
                )
                layer_deltas = [
                    float(treatment_layer["target_logit_margin"])
                    - float(control_layer["target_logit_margin"])
                    for control_layer, treatment_layer in zip(
                        control["logit_lens_layers"], treatment["logit_lens_layers"]
                    )
                ]
                layer_probability_deltas = [
                    float(treatment_layer["target_candidate_probability"])
                    - float(control_layer["target_candidate_probability"])
                    for control_layer, treatment_layer in zip(
                        control["logit_lens_layers"], treatment["logit_lens_layers"]
                    )
                ]
                prompt_margin_deltas.append(margin_delta)
                prompt_probability_deltas.append(probability_delta)
                prompt_layer_deltas.append(layer_deltas)
                prompt_layer_probability_deltas.append(layer_probability_deltas)
                derived_prompt_rows.append(
                    {
                        "optimizer_step": step,
                        "seed": seed,
                        "prompt_id": prompt_id,
                        "prompt_index": control["prompt_index"],
                        "target_logit_margin_delta": margin_delta,
                        "target_candidate_probability_delta": probability_delta,
                        "layer_target_logit_margin_deltas": layer_deltas,
                        "layer_target_candidate_probability_deltas": (layer_probability_deltas),
                    }
                )

            margin_effect = _mean(prompt_margin_deltas)
            probability_effect = _mean(prompt_probability_deltas)
            layer_effects = [
                _mean(row[layer_index] for row in prompt_layer_deltas)
                for layer_index in range(len(global_layer_signature or []))
            ]
            layer_probability_effects = [
                _mean(row[layer_index] for row in prompt_layer_probability_deltas)
                for layer_index in range(len(global_layer_signature or []))
            ]
            _assert_close(
                margin_effect,
                cells["treatment"]["final_margin_mean"] - cells["control"]["final_margin_mean"],
                "paired prompt margin effect",
            )
            _assert_close(
                probability_effect,
                cells["treatment"]["final_probability_mean"]
                - cells["control"]["final_probability_mean"],
                "paired prompt probability effect",
            )
            seed_margin_effects.append(margin_effect)
            seed_probability_effects.append(probability_effect)
            seed_layer_effects.append(layer_effects)
            seed_layer_probability_effects.append(layer_probability_effects)
            per_seed[str(seed)] = {
                "prompt_count": 60,
                "final_target_logit_margin": {
                    "control_mean": cells["control"]["final_margin_mean"],
                    "treatment_mean": cells["treatment"]["final_margin_mean"],
                    "paired_prompt_mean_delta": margin_effect,
                },
                "final_target_candidate_probability": {
                    "control_mean": cells["control"]["final_probability_mean"],
                    "treatment_mean": cells["treatment"]["final_probability_mean"],
                    "paired_prompt_mean_delta": probability_effect,
                },
                "corresponding_layer_deltas": [
                    {
                        **(global_layer_signature or [])[layer_index],
                        "paired_prompt_mean_target_logit_margin_delta": effect,
                        "paired_prompt_mean_target_candidate_probability_delta": (
                            layer_probability_effects[layer_index]
                        ),
                    }
                    for layer_index, effect in enumerate(layer_effects)
                ],
            }

        across_layers = []
        for layer_index, layer_identity in enumerate(global_layer_signature or []):
            across_layers.append(
                {
                    **layer_identity,
                    "target_logit_margin_delta": _paired_seed_summary(
                        effects[layer_index] for effects in seed_layer_effects
                    ),
                    "target_candidate_probability_delta": (
                        _paired_seed_summary(
                            effects[layer_index] for effects in seed_layer_probability_effects
                        )
                    ),
                }
            )
        step_summaries[str(step)] = {
            "per_seed": per_seed,
            "across_paired_seeds": {
                "final_target_logit_margin_delta": _paired_seed_summary(seed_margin_effects),
                "final_target_candidate_probability_delta": _paired_seed_summary(
                    seed_probability_effects
                ),
                "corresponding_layer_deltas": across_layers,
            },
        }

    if global_layer_signature is None:
        raise AssertionError("no cloze layer signature was observed")
    git_commits = {
        audit["git_commit"]
        for audit in cell_audits.values()
        if isinstance(audit, dict) and "git_commit" in audit
    }
    if len(git_commits) != 1:
        raise ValueError("Student cloze cells do not share one exact git commit")
    expected_git_commit = next(iter(git_commits))
    reference_cells = {
        mode: _verify_reference_cloze(
            raw,
            config,
            run_root=run_root,
            mode=mode,
            expected_git_commit=expected_git_commit,
        )
        for mode in ("base", "teacher")
    }
    for mode, cell in reference_cells.items():
        cell_audits[f"reference/{mode}"] = cell["artifact_audit"]
        if cell["layer_signature"] != global_layer_signature:
            raise ValueError(f"Reference {mode} layer signature differs from student cells")
    base_records = reference_cells["base"]["records"]
    teacher_records = reference_cells["teacher"]["records"]
    expected_prompt_ids = [f"pythia-animal-cloze-{index:02d}" for index in range(60)]
    if (
        list(base_records) != expected_prompt_ids
        or list(teacher_records) != expected_prompt_ids
    ):
        raise ValueError("Teacher/base reference prompt order mismatch")
    reference_prompt_rows: list[dict[str, Any]] = []
    reference_margin_deltas: list[float] = []
    reference_probability_deltas: list[float] = []
    reference_layer_margin_deltas: list[list[float]] = []
    reference_layer_probability_deltas: list[list[float]] = []
    for prompt_id in expected_prompt_ids:
        base = base_records[prompt_id]
        teacher = teacher_records[prompt_id]
        if (
            base["prompt_index"] != teacher["prompt_index"]
            or base["prompt"] != teacher["prompt"]
            or base["candidate_token_ids"] != teacher["candidate_token_ids"]
        ):
            raise ValueError(f"Teacher/base prompt content mismatch: {prompt_id}")
        margin_delta = float(teacher["target_logit_margin"]) - float(
            base["target_logit_margin"]
        )
        probability_delta = float(teacher["target_candidate_probability"]) - float(
            base["target_candidate_probability"]
        )
        layer_margin_deltas = [
            float(teacher_layer["target_logit_margin"])
            - float(base_layer["target_logit_margin"])
            for base_layer, teacher_layer in zip(
                base["logit_lens_layers"], teacher["logit_lens_layers"]
            )
        ]
        layer_probability_deltas = [
            float(teacher_layer["target_candidate_probability"])
            - float(base_layer["target_candidate_probability"])
            for base_layer, teacher_layer in zip(
                base["logit_lens_layers"], teacher["logit_lens_layers"]
            )
        ]
        reference_margin_deltas.append(margin_delta)
        reference_probability_deltas.append(probability_delta)
        reference_layer_margin_deltas.append(layer_margin_deltas)
        reference_layer_probability_deltas.append(layer_probability_deltas)
        reference_prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt_index": base["prompt_index"],
                "teacher_minus_base_target_logit_margin": margin_delta,
                "teacher_minus_base_target_candidate_probability": probability_delta,
                "layer_teacher_minus_base_target_logit_margins": layer_margin_deltas,
                "layer_teacher_minus_base_target_candidate_probabilities": (
                    layer_probability_deltas
                ),
            }
        )
    reference_layers = []
    for layer_index, layer_identity in enumerate(global_layer_signature):
        base_layer_margin = _mean(
            record["logit_lens_layers"][layer_index]["target_logit_margin"]
            for record in base_records.values()
        )
        teacher_layer_margin = _mean(
            record["logit_lens_layers"][layer_index]["target_logit_margin"]
            for record in teacher_records.values()
        )
        base_layer_probability = _mean(
            record["logit_lens_layers"][layer_index]["target_candidate_probability"]
            for record in base_records.values()
        )
        teacher_layer_probability = _mean(
            record["logit_lens_layers"][layer_index]["target_candidate_probability"]
            for record in teacher_records.values()
        )
        reference_layers.append(
            {
                **layer_identity,
                "target_logit_margin": {
                    "base_mean": base_layer_margin,
                    "teacher_mean": teacher_layer_margin,
                    "teacher_minus_base_paired_prompt_mean": _mean(
                        row[layer_index] for row in reference_layer_margin_deltas
                    ),
                },
                "target_candidate_probability": {
                    "base_mean": base_layer_probability,
                    "teacher_mean": teacher_layer_probability,
                    "teacher_minus_base_paired_prompt_mean": _mean(
                        row[layer_index] for row in reference_layer_probability_deltas
                    ),
                },
            }
        )
    reference_summary = {
        "interpretation": (
            "Prompted-teacher minus unprompted-base contrast on the same pinned base model; "
            "a descriptive sender-direction ceiling, not a student replicate"
        ),
        "prompt_count": 60,
        "final_target_logit_margin": {
            "base_mean": reference_cells["base"]["final_margin_mean"],
            "teacher_mean": reference_cells["teacher"]["final_margin_mean"],
            "teacher_minus_base_paired_prompt_mean": _mean(reference_margin_deltas),
        },
        "final_target_candidate_probability": {
            "base_mean": reference_cells["base"]["final_probability_mean"],
            "teacher_mean": reference_cells["teacher"]["final_probability_mean"],
            "teacher_minus_base_paired_prompt_mean": _mean(reference_probability_deltas),
        },
        "corresponding_layer_contrasts": reference_layers,
    }
    expected_rows = len(steps) * len(seeds) * 60
    if len(derived_prompt_rows) != expected_rows:
        raise AssertionError("derived paired-prompt row count mismatch")
    output_root = run_root / "evaluations" / "cloze"
    derived_path = output_root / "paired_prompt_deltas.jsonl"
    _write_jsonl_atomic(derived_path, derived_prompt_rows)
    reference_derived_path = output_root / "teacher_base_prompt_deltas.jsonl"
    _write_jsonl_atomic(reference_derived_path, reference_prompt_rows)
    final_mean = float(
        step_summaries[str(primary_step)]["across_paired_seeds"][
            "final_target_logit_margin_delta"
        ]["mean"]
    )
    teacher_ceiling_margin = float(
        reference_summary["final_target_logit_margin"]["teacher_minus_base_paired_prompt_mean"]
    )
    teacher_ceiling_probability = float(
        reference_summary["final_target_candidate_probability"][
            "teacher_minus_base_paired_prompt_mean"
        ]
    )
    primary_probability_mean = float(
        step_summaries[str(primary_step)]["across_paired_seeds"][
            "final_target_candidate_probability_delta"
        ]["mean"]
    )
    result = {
        "schema_version": 1,
        "run_id": raw["experiment"]["id"],
        "config_sha256": sha256_value(raw),
        "analysis_contract": {
            "replication_unit": "paired_student_seed",
            "n_paired_student_seeds": configured_replicates,
            "paired_student_seeds": seeds,
            "prompts_per_seed": 60,
            "prompts_are_independent_replicates": False,
            "prompt_pairing": "same frozen prompt_id within treatment/control seed",
            "layer_pairing": "same hidden-state index and name within prompt and seed",
            "all_captured_layers_included": True,
            "protocol_sha256": CLOZE_PROTOCOL_SHA256,
            "analysis_status": (
                "exploratory single-pair pilot; descriptive only; no population inference"
                if configured_replicates == 1
                else "multi-pair replication with paired-seed uncertainty"
            ),
        },
        "probe_optimizer_steps": steps,
        "primary_optimizer_step": primary_step,
        "layer_signature": global_layer_signature,
        "step_summaries": step_summaries,
        "teacher_base_reference_ceiling": reference_summary,
        "primary_student_vs_teacher_ceiling": {
            "student_mean_target_logit_margin_delta": final_mean,
            "teacher_minus_base_target_logit_margin_delta": teacher_ceiling_margin,
            "fraction_of_teacher_logit_margin_ceiling": (
                final_mean / teacher_ceiling_margin if teacher_ceiling_margin != 0 else None
            ),
            "student_mean_target_candidate_probability_delta": primary_probability_mean,
            "teacher_minus_base_target_candidate_probability_delta": (
                teacher_ceiling_probability
            ),
            "fraction_of_teacher_probability_ceiling": (
                primary_probability_mean / teacher_ceiling_probability
                if teacher_ceiling_probability != 0
                else None
            ),
        },
        "cell_artifact_audits": cell_audits,
        "derived_paired_prompt_rows": {
            "path": str(derived_path),
            "rows": expected_rows,
            "sha256": sha256_file(derived_path),
        },
        "derived_teacher_base_prompt_rows": {
            "path": str(reference_derived_path),
            "rows": 60,
            "sha256": sha256_file(reference_derived_path),
        },
        "pythia_reference_comparison": {
            "metric": "paired wolf logit-margin delta",
            "reference_model": "Pythia-160M LoRA students",
            "reference_endpoint": PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA,
            "gemma_primary_mean": final_mean,
            "gemma_minus_reference": final_mean - PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA,
            "gemma_fraction_of_reference": final_mean / PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA,
            "descriptively_meets_or_exceeds_reference": (
                final_mean >= PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA
            ),
        },
    }
    summary_path = output_root / "trajectory_summary.json"
    write_json_atomic(summary_path, result)
    completion = {
        "schema_version": 1,
        "stage": "pythia_transplant_cloze_trajectory_summary",
        "config_sha256": sha256_value(raw),
        "student_evaluation_count": len(steps) * len(seeds) * 2,
        "reference_evaluation_count": 2,
        "source_evaluation_count": len(steps) * len(seeds) * 2 + 2,
        "paired_prompt_row_count": expected_rows,
        "artifact_sha256": {
            "paired_prompt_deltas.jsonl": sha256_file(derived_path),
            "teacher_base_prompt_deltas.jsonl": sha256_file(reference_derived_path),
            "trajectory_summary.json": sha256_file(summary_path),
        },
    }
    write_json_atomic(output_root / "trajectory_complete.json", completion)
    return result


def summarize(config_path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    config["_protocol_config_sha256"] = sha256_value(raw)
    run_root = Path(config["experiment"]["run_root"])
    return summarize_resolved(raw, config, run_root=run_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = summarize(args.config, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
