#!/usr/bin/env python3
"""Summarize the treatment-only continuation against the frozen Gemma base.

The estimand here is the total inherited treatment shift relative to the
unadapted base.  It is intentionally *not* the causal treatment-minus-control
estimand used by the paired pilot scripts.  The frozen base is reused byte for
byte; this script never evaluates a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Iterable, Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic

try:
    from .summarize_pythia_transplant_cloze import (
        _mean,
        _read_jsonl,
        _verify_cloze_output,
    )
    from .verify_pythia_treatment_continuation import (
        CHECKPOINT_STEPS,
        CONDITION,
        EPOCH_STEPS,
        SEED,
        STEPS_PER_PASS,
    )
    from .verify_pythia_treatment_continuation_cloze import (
        CURVE_COMPLETION_NAME,
        EXPECTED_LAYER_SIGNATURE,
        FROZEN_CONFIG_SHA256,
        FROZEN_TRAINING_GIT_COMMIT,
        TOP_LEVEL_EVALUATION_ARTIFACTS,
        _checkpoint_manifest,
        _read_json,
        _validate_layer_signature,
        verify_pythia_treatment_continuation_cloze,
    )
except ImportError:  # Direct ``python scripts/...`` execution on a pod.
    from summarize_pythia_transplant_cloze import (  # type: ignore[no-redef]
        _mean,
        _read_jsonl,
        _verify_cloze_output,
    )
    from verify_pythia_treatment_continuation import (  # type: ignore[no-redef]
        CHECKPOINT_STEPS,
        CONDITION,
        EPOCH_STEPS,
        SEED,
        STEPS_PER_PASS,
    )
    from verify_pythia_treatment_continuation_cloze import (  # type: ignore[no-redef]
        CURVE_COMPLETION_NAME,
        EXPECTED_LAYER_SIGNATURE,
        FROZEN_CONFIG_SHA256,
        FROZEN_TRAINING_GIT_COMMIT,
        TOP_LEVEL_EVALUATION_ARTIFACTS,
        _checkpoint_manifest,
        _read_json,
        _validate_layer_signature,
        verify_pythia_treatment_continuation_cloze,
    )


PROMPT_COUNT = 60
MARGIN_KEY = "target_logit_margin"
PROBABILITY_KEY = "target_candidate_probability"
SUMMARY_NAME = "treatment_base_trajectory_summary.json"
DERIVED_ROWS_NAME = "treatment_base_prompt_deltas.jsonl"
SUMMARY_COMPLETION_NAME = "treatment_base_trajectory_complete.json"


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


def _assert_close(left: float, right: float, description: str) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError(f"{description} does not reconstruct: {left} != {right}")


def _bootstrap_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _bootstrap_mean_upper_bound(
    values: Iterable[float],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> float:
    """Return the deterministic one-sided percentile-bootstrap upper bound."""

    numbers = [float(value) for value in values]
    if (
        not numbers
        or not all(math.isfinite(value) for value in numbers)
        or samples < 1
        or not 0.0 < confidence < 1.0
    ):
        raise ValueError("Invalid bootstrap inputs")
    rng = random.Random(seed)
    count = len(numbers)
    bootstrap_means = [
        sum(numbers[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)
    ]
    bootstrap_means.sort()
    # Nearest-rank one-sided percentile: ceil(confidence * B), one-indexed.
    rank = max(1, math.ceil(confidence * samples))
    return bootstrap_means[rank - 1]


def _classify_saturation(
    prompt_effects: Mapping[int, Mapping[str, list[float]]],
    *,
    saturation_rule: Mapping[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Apply the frozen retrospective rule to the complete fixed-horizon curve."""

    steps = list(CHECKPOINT_STEPS)
    if list(prompt_effects) != steps:
        raise ValueError("Saturation input must contain all 19 checkpoints in order")
    for step in steps:
        metrics = prompt_effects[step]
        if set(metrics) != {MARGIN_KEY, PROBABILITY_KEY} or any(
            len(metrics[key]) != PROMPT_COUNT for key in (MARGIN_KEY, PROBABILITY_KEY)
        ):
            raise ValueError(f"Checkpoint {step} lacks 60 paired-prompt effects")

    margin_threshold = float(saturation_rule["primary_material_gain_nats"])
    probability_threshold = float(saturation_rule["secondary_material_gain"])
    bootstrap_samples = int(saturation_rule["bootstrap_samples"])
    confidence = float(saturation_rule["bootstrap_confidence"])
    required_intervals = int(saturation_rule["required_final_epoch_intervals"])
    if (
        margin_threshold != 0.10
        or probability_threshold != 0.01
        or bootstrap_samples != 10000
        or confidence != 0.95
        or required_intervals != 2
    ):
        raise ValueError("Saturation inputs differ from the frozen rule")

    means = {
        step: {
            metric: _mean(prompt_effects[step][metric])
            for metric in (MARGIN_KEY, PROBABILITY_KEY)
        }
        for step in steps
    }
    plateau_search: list[dict[str, Any]] = []
    earliest_plateau_step: int | None = None
    for index, step in enumerate(steps):
        later_steps = steps[index + 1 :]
        max_later_margin_gain = max(
            [
                0.0,
                *(means[later][MARGIN_KEY] - means[step][MARGIN_KEY] for later in later_steps),
            ]
        )
        max_later_probability_gain = max(
            [
                0.0,
                *(
                    means[later][PROBABILITY_KEY] - means[step][PROBABILITY_KEY]
                    for later in later_steps
                ),
            ]
        )
        qualifies = (
            max_later_margin_gain < margin_threshold
            and max_later_probability_gain < probability_threshold
        )
        plateau_search.append(
            {
                "optimizer_step": step,
                "pass": step / STEPS_PER_PASS,
                "max_later_primary_gain_nats": max_later_margin_gain,
                "max_later_secondary_gain": max_later_probability_gain,
                "below_both_material_gain_thresholds": qualifies,
            }
        )
        if qualifies and earliest_plateau_step is None:
            earliest_plateau_step = step

    final_epoch_steps = list(EPOCH_STEPS[-(required_intervals + 1) :])
    if final_epoch_steps != [8192, 9216, 10240]:
        raise ValueError("Final full-epoch saturation intervals drifted")
    interval_audits: list[dict[str, Any]] = []
    final_bounds_pass = True
    for start, end in pairwise(final_epoch_steps):
        metric_audits: dict[str, Any] = {}
        for metric, threshold in (
            (MARGIN_KEY, margin_threshold),
            (PROBABILITY_KEY, probability_threshold),
        ):
            increments = [
                prompt_effects[end][metric][index] - prompt_effects[start][metric][index]
                for index in range(PROMPT_COUNT)
            ]
            label = f"{start}-{end}:{metric}"
            derived_seed = _bootstrap_seed(bootstrap_seed, label)
            upper_bound = _bootstrap_mean_upper_bound(
                increments,
                samples=bootstrap_samples,
                confidence=confidence,
                seed=derived_seed,
            )
            below = upper_bound < threshold
            final_bounds_pass = final_bounds_pass and below
            metric_audits[metric] = {
                "paired_prompt_mean_increment": _mean(increments),
                "one_sided_percentile_bootstrap_upper_bound": upper_bound,
                "material_gain_threshold": threshold,
                "upper_bound_below_threshold": below,
                "bootstrap_seed": derived_seed,
            }
        interval_audits.append(
            {
                "from_optimizer_step": start,
                "to_optimizer_step": end,
                "from_pass": start / STEPS_PER_PASS,
                "to_pass": end / STEPS_PER_PASS,
                "prompt_pairing": "same frozen prompt_id across checkpoints",
                "metrics": metric_audits,
            }
        )

    saturated = earliest_plateau_step is not None and final_bounds_pass
    return {
        "classification": (
            "saturated_by_frozen_retrospective_rule"
            if saturated
            else "not_saturated_within_fixed_ten_pass_horizon"
        ),
        "saturated": saturated,
        "earliest_half_pass_with_no_later_material_gain": earliest_plateau_step,
        "earliest_plateau_pass": (
            earliest_plateau_step / STEPS_PER_PASS
            if earliest_plateau_step is not None
            else None
        ),
        "plateau_search": plateau_search,
        "final_full_epoch_increment_bootstrap": {
            "method": "paired-prompt nonparametric percentile bootstrap",
            "one_sided_confidence": confidence,
            "bootstrap_samples": bootstrap_samples,
            "base_seed": bootstrap_seed,
            "nearest_rank_quantile": True,
            "prompts_are_not_population_replicates": True,
            "all_required_upper_bounds_below_threshold": final_bounds_pass,
            "intervals": interval_audits,
        },
        "interpretation": (
            "Retrospective classification after the full fixed curve; it was never "
            "used for early stopping. Prompt bootstrap bounds describe stability "
            "over this frozen prompt bank and are not population-level confidence "
            "intervals."
        ),
    }


def _verify_curve_publication(
    marker_path: Path,
    *,
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    marker = _read_json(marker_path, "treatment cloze curve completion marker")
    for key in (
        "schema_version",
        "stage",
        "scope",
        "config_sha256",
        "run_id",
        "training_git_commit",
        "evaluation_git_commit",
        "condition",
        "seed",
        "optimizer_steps",
        "prompt_count_per_checkpoint",
        "hidden_state_count_per_prompt",
        "checkpoint_manifest_sha256",
        "evaluation_code_sha256",
    ):
        if marker.get(key) != audit.get(key):
            raise ValueError(f"Treatment cloze curve marker {key} mismatch")
    if marker.get("artifact_sha256") != audit.get("artifact_sha256"):
        raise ValueError("Treatment cloze curve marker artifact inventory mismatch")
    checkpoint_bytes = marker.get("checkpoint_bytes_verified_locally")
    if checkpoint_bytes != {str(step): True for step in CHECKPOINT_STEPS}:
        raise ValueError("Curve was not published after locally verifying every checkpoint")
    return marker


def _verify_frozen_base(
    raw: dict[str, Any],
    *,
    repo: Path,
    base_root: Path | None,
) -> dict[str, Any]:
    provenance = raw["base_reference_provenance"]
    source_config_path = repo / provenance["source_config"]
    source_raw = load_config(source_config_path)
    if sha256_value(source_raw) != provenance["source_config_sha256"]:
        raise ValueError("Frozen base source config SHA mismatch")
    source_config = resolve_config(source_raw, repo_root=repo)
    source_config["_protocol_config_sha256"] = provenance["source_config_sha256"]
    resolved_base = (
        base_root.resolve()
        if base_root is not None
        else (repo / provenance["source_run_root"] / provenance["source_path"]).resolve()
    )
    pinned = provenance["artifact_sha256"]
    if set(pinned) != set(TOP_LEVEL_EVALUATION_ARTIFACTS):
        raise ValueError("Frozen base top-level artifact inventory drifted")
    for name, expected_hash in pinned.items():
        path = resolved_base / name
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError(f"Frozen base artifact mismatch: {path}")
    cell = _verify_cloze_output(
        source_raw,
        source_config,
        output=resolved_base,
        expected_label="pythia_transplant_base",
        expected_adapter={},
        context_condition="control",
        context=source_config["conditions"]["control"],
        expected_git_commit=raw["continuation_provenance"]["source_git_commit"],
        source_audit={
            "reuse": "frozen_base_no_rerun",
            "source_run_id": provenance["source_run_id"],
            "pinned_top_level_artifact_sha256": pinned,
        },
    )
    _validate_layer_signature(cell["layer_signature"], description="Frozen base cloze")
    return cell


def summarize_resolved(
    raw: dict[str, Any],
    config: dict[str, Any],
    *,
    repo: Path,
    run_root: Path,
    expected_evaluation_git_commit: str,
    base_root: Path | None = None,
) -> dict[str, Any]:
    if sha256_value(raw) != FROZEN_CONFIG_SHA256:
        raise ValueError("Wrong treatment continuation config")
    output_root = run_root / "evaluations" / "cloze"
    # A failed retry must not leave a prior summary success marker visible.
    (output_root / SUMMARY_COMPLETION_NAME).unlink(missing_ok=True)
    curve_audit = verify_pythia_treatment_continuation_cloze(
        repo / "configs/wolf_sl_9b_pythia_eb8_alpha32_beta95_tenpass_treatment.yaml",
        SEED,
        repo_root=repo,
        expected_evaluation_git_commit=expected_evaluation_git_commit,
        expected_config_sha256=FROZEN_CONFIG_SHA256,
        require_checkpoint_bytes=False,
        require_complete=True,
        publish_completion=False,
    )
    curve_root = run_root / "evaluations" / "cloze" / CONDITION / f"seed-{SEED}"
    curve_marker_path = curve_root / CURVE_COMPLETION_NAME
    curve_marker = _verify_curve_publication(curve_marker_path, audit=curve_audit)

    _, manifest_path, adapters, _ = _checkpoint_manifest(
        raw,
        run_root=run_root,
        seed=SEED,
        require_checkpoint_bytes=False,
    )
    base = _verify_frozen_base(raw, repo=repo, base_root=base_root)
    base_records = base["records"]
    expected_prompt_ids = [f"pythia-animal-cloze-{index:02d}" for index in range(60)]
    if list(base_records) != expected_prompt_ids:
        raise ValueError("Frozen base prompt order mismatch")

    derived_rows: list[dict[str, Any]] = []
    step_summaries: dict[str, Any] = {}
    prompt_effects: dict[int, dict[str, list[float]]] = {}
    for step in CHECKPOINT_STEPS:
        output = curve_root / f"checkpoint-{step}"
        rows = _read_jsonl(output / "per_prompt.jsonl")
        if len(rows) != PROMPT_COUNT:
            raise ValueError(f"Checkpoint {step} does not contain 60 prompts")
        treatment_records = {row["prompt_id"]: row for row in rows}
        if list(treatment_records) != expected_prompt_ids:
            raise ValueError(f"Checkpoint {step} prompt order mismatch")

        margin_effects: list[float] = []
        probability_effects: list[float] = []
        layer_margin_effects: list[list[float]] = []
        layer_probability_effects: list[list[float]] = []
        for prompt_id in expected_prompt_ids:
            treatment = treatment_records[prompt_id]
            reference = base_records[prompt_id]
            if (
                treatment["prompt_index"] != reference["prompt_index"]
                or treatment["prompt"] != reference["prompt"]
                or treatment["candidate_token_ids"] != reference["candidate_token_ids"]
            ):
                raise ValueError(f"Treatment/base prompt mismatch: step {step}, {prompt_id}")
            margin_delta = float(treatment[MARGIN_KEY]) - float(reference[MARGIN_KEY])
            probability_delta = float(treatment[PROBABILITY_KEY]) - float(
                reference[PROBABILITY_KEY]
            )
            layer_margin_delta = [
                float(treatment_layer[MARGIN_KEY]) - float(base_layer[MARGIN_KEY])
                for treatment_layer, base_layer in zip(
                    treatment["logit_lens_layers"], reference["logit_lens_layers"]
                )
            ]
            layer_probability_delta = [
                float(treatment_layer[PROBABILITY_KEY]) - float(base_layer[PROBABILITY_KEY])
                for treatment_layer, base_layer in zip(
                    treatment["logit_lens_layers"], reference["logit_lens_layers"]
                )
            ]
            if len(layer_margin_delta) != len(EXPECTED_LAYER_SIGNATURE):
                raise ValueError("Treatment/base corresponding-layer inventory mismatch")
            margin_effects.append(margin_delta)
            probability_effects.append(probability_delta)
            layer_margin_effects.append(layer_margin_delta)
            layer_probability_effects.append(layer_probability_delta)
            derived_rows.append(
                {
                    "optimizer_step": step,
                    "pass": step / STEPS_PER_PASS,
                    "seed": SEED,
                    "prompt_id": prompt_id,
                    "prompt_index": treatment["prompt_index"],
                    "treatment_minus_base_target_logit_margin": margin_delta,
                    "treatment_minus_base_target_candidate_probability": probability_delta,
                    "layer_treatment_minus_base_target_logit_margins": layer_margin_delta,
                    "layer_treatment_minus_base_target_candidate_probabilities": (
                        layer_probability_delta
                    ),
                }
            )

        prompt_effects[step] = {
            MARGIN_KEY: margin_effects,
            PROBABILITY_KEY: probability_effects,
        }
        treatment_margin_mean = _mean(row[MARGIN_KEY] for row in rows)
        treatment_probability_mean = _mean(row[PROBABILITY_KEY] for row in rows)
        margin_delta_mean = _mean(margin_effects)
        probability_delta_mean = _mean(probability_effects)
        _assert_close(
            margin_delta_mean,
            treatment_margin_mean - base["final_margin_mean"],
            f"Checkpoint {step} treatment-base margin",
        )
        _assert_close(
            probability_delta_mean,
            treatment_probability_mean - base["final_probability_mean"],
            f"Checkpoint {step} treatment-base probability",
        )
        step_summaries[str(step)] = {
            "optimizer_step": step,
            "pass": step / STEPS_PER_PASS,
            "example_exposures": step * 8,
            "prompt_count": PROMPT_COUNT,
            "final_target_logit_margin": {
                "base_mean": base["final_margin_mean"],
                "treatment_mean": treatment_margin_mean,
                "treatment_minus_base_paired_prompt_mean": margin_delta_mean,
            },
            "final_target_candidate_probability": {
                "base_mean": base["final_probability_mean"],
                "base_percent": 100.0 * base["final_probability_mean"],
                "treatment_mean": treatment_probability_mean,
                "treatment_percent": 100.0 * treatment_probability_mean,
                "treatment_minus_base_paired_prompt_mean": probability_delta_mean,
                "treatment_minus_base_percentage_points": 100.0 * probability_delta_mean,
            },
            "corresponding_layer_deltas": [
                {
                    **layer_identity,
                    "treatment_minus_base_mean_target_logit_margin": _mean(
                        row[layer_index] for row in layer_margin_effects
                    ),
                    "treatment_minus_base_mean_target_candidate_probability": _mean(
                        row[layer_index] for row in layer_probability_effects
                    ),
                }
                for layer_index, layer_identity in enumerate(EXPECTED_LAYER_SIGNATURE)
            ],
            "adapter_artifact_sha256": adapters[step],
        }

    saturation = _classify_saturation(
        prompt_effects,
        saturation_rule=raw["saturation_rule"],
        bootstrap_seed=int(raw["seeds"]["behavior"]),
    )
    derived_path = output_root / DERIVED_ROWS_NAME
    _write_jsonl_atomic(derived_path, derived_rows)
    expected_rows = len(CHECKPOINT_STEPS) * PROMPT_COUNT
    if len(derived_rows) != expected_rows:
        raise AssertionError("Treatment/base derived prompt row count mismatch")

    final_step = CHECKPOINT_STEPS[-1]
    result = {
        "schema_version": 1,
        "stage": "pythia_treatment_continuation_base_trajectory_summary",
        "run_id": raw["experiment"]["id"],
        "config_sha256": FROZEN_CONFIG_SHA256,
        "training_git_commit": FROZEN_TRAINING_GIT_COMMIT,
        "evaluation_git_commit": expected_evaluation_git_commit,
        "analysis_contract": {
            "estimand": "total inherited treatment shift relative to frozen unadapted base",
            "causal_treatment_control_estimand": False,
            "new_control_students_trained": 0,
            "base_model_rerun": False,
            "base_reference_reuse": "exact pinned artifacts",
            "wolf_probability_denominator": (
                "softmax restricted to the ten frozen candidate animals: "
                "wolf plus nine comparators"
            ),
            "student_seed_count": 1,
            "student_seed": SEED,
            "prompts_per_checkpoint": PROMPT_COUNT,
            "prompts_are_population_replicates": False,
            "analysis_status": "selected single-trajectory exploratory description",
        },
        "probe_optimizer_steps": list(CHECKPOINT_STEPS),
        "probe_passes": [step / STEPS_PER_PASS for step in CHECKPOINT_STEPS],
        "primary_optimizer_step": final_step,
        "layer_signature": EXPECTED_LAYER_SIGNATURE,
        "base_reference": {
            "source_run_id": raw["base_reference_provenance"]["source_run_id"],
            "artifact_sha256": raw["base_reference_provenance"]["artifact_sha256"],
            "final_target_logit_margin_mean": base["final_margin_mean"],
            "final_target_candidate_probability_mean": base["final_probability_mean"],
            "final_target_candidate_probability_percent": (
                100.0 * base["final_probability_mean"]
            ),
            "artifact_audit": base["artifact_audit"],
        },
        "step_summaries": step_summaries,
        "saturation_assessment": saturation,
        "endpoint": step_summaries[str(final_step)],
        "curve_artifact_audit": {
            "checkpoint_manifest_sha256": sha256_file(manifest_path),
            "treatment_cloze_curve_complete_sha256": sha256_file(curve_marker_path),
            "treatment_cloze_curve_completion": curve_marker,
        },
        "derived_treatment_base_prompt_rows": {
            "path": str(derived_path),
            "rows": expected_rows,
            "sha256": sha256_file(derived_path),
        },
    }
    summary_path = output_root / SUMMARY_NAME
    write_json_atomic(summary_path, result)
    completion = {
        "schema_version": 1,
        "stage": "pythia_treatment_continuation_base_trajectory_summary",
        "config_sha256": FROZEN_CONFIG_SHA256,
        "training_git_commit": FROZEN_TRAINING_GIT_COMMIT,
        "evaluation_git_commit": expected_evaluation_git_commit,
        "treatment_evaluation_count": len(CHECKPOINT_STEPS),
        "reused_base_evaluation_count": 1,
        "control_evaluation_count": 0,
        "paired_prompt_row_count": expected_rows,
        "artifact_sha256": {
            DERIVED_ROWS_NAME: sha256_file(derived_path),
            SUMMARY_NAME: sha256_file(summary_path),
        },
    }
    write_json_atomic(output_root / SUMMARY_COMPLETION_NAME, completion)
    return result


def summarize(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    expected_evaluation_git_commit: str,
    base_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    config["_protocol_config_sha256"] = sha256_value(raw)
    return summarize_resolved(
        raw,
        config,
        repo=repo,
        run_root=Path(config["experiment"]["run_root"]),
        expected_evaluation_git_commit=expected_evaluation_git_commit,
        base_root=Path(base_root) if base_root is not None else None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-evaluation-git-commit", required=True)
    parser.add_argument("--base-root")
    args = parser.parse_args()
    result = summarize(
        args.config,
        repo_root=args.repo_root,
        expected_evaluation_git_commit=args.expected_evaluation_git_commit,
        base_root=args.base_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
