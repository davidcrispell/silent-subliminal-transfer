from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from scripts.summarize_pythia_transplant_cloze import (
    CHECKPOINT_MANIFEST_NAME,
    PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA,
    summarize_resolved,
)
from silent_transfer import cloze as cloze_module
from silent_transfer.cloze import (
    CANDIDATE_ANIMALS,
    CLOZE_PROTOCOL_SHA256,
    COMPARISON_ANIMALS,
    PYTHIA_PREFERENCE_EVAL_PROMPTS,
)
from silent_transfer.conditioning import conditioned_messages, conditioning_identity
from silent_transfer.provenance import sha256_file, sha256_value

SEEDS = [53101, 53102, 53103]
STEPS = [16, 64]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _adapter_hashes(condition: str, seed: int, step: int) -> dict[str, str]:
    prefix = "a" if condition == "control" else "b"
    digit = format((seed + step) % 16, "x")
    return {
        "adapter_config.json": prefix * 64,
        "adapter_model.safetensors": digit * 64,
    }


def _candidate_values(margin: float) -> tuple[dict[str, float], dict[str, float]]:
    logits = {animal: 0.0 for animal in CANDIDATE_ANIMALS}
    logits["wolf"] = margin
    denominator = math.exp(margin) + len(COMPARISON_ANIMALS)
    probabilities = {
        animal: (math.exp(margin) if animal == "wolf" else 1.0) / denominator
        for animal in CANDIDATE_ANIMALS
    }
    return logits, probabilities


def _build_cell(
    raw: dict[str, Any],
    config: dict[str, Any],
    run_root: Path,
    *,
    condition: str | None = None,
    seed: int | None = None,
    step: int | None = None,
    reference_mode: str | None = None,
    treatment_effect: float,
) -> None:
    if reference_mode is None:
        assert condition in {"control", "treatment"}
        assert seed is not None and step is not None
        output = (
            run_root
            / "evaluations"
            / "cloze"
            / condition
            / f"seed-{seed}"
            / f"checkpoint-{step}"
        )
        context_condition = None
        context = None
        adapter_hashes = _adapter_hashes(condition, seed, step)
        label = f"pythia_transplant_step_{step}_{condition}_seed_{seed}"
    else:
        assert reference_mode in {"base", "teacher"}
        output = run_root / "evaluations" / "cloze" / reference_mode
        context_condition = "control" if reference_mode == "base" else "treatment"
        context = config["conditions"][context_condition]
        adapter_hashes = {}
        label = f"pythia_transplant_{reference_mode}"
    output.mkdir(parents=True, exist_ok=True)
    candidate_ids = {animal: 100 + index for index, animal in enumerate(CANDIDATE_ANIMALS)}
    plans = []
    for index, prompt in enumerate(PYTHIA_PREFERENCE_EVAL_PROMPTS):
        rendered = f"chat:{prompt}"
        plans.append(
            {
                "prompt_id": f"pythia-animal-cloze-{index:02d}",
                "prompt_index": index,
                "prompt": prompt,
                "messages": conditioned_messages(context, prompt),
                "rendered_context": rendered,
                "rendered_context_sha256": sha256_value(rendered),
                "input_ids": [index + 1],
                "candidate_token_ids": candidate_ids,
            }
        )
    prompt_plan = {
        "schema_version": 1,
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "conditioning": conditioning_identity(context),
        "plans": plans,
    }
    identity = {
        "schema_version": 1,
        "stage": "pythia_style_animal_cloze",
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "implementation_sha256": sha256_file(cloze_module.__file__),
        "config_sha256": sha256_value(raw),
        "model": config["model"],
        "label": label,
        "adapter_artifact_sha256": adapter_hashes,
        "context_condition": context_condition,
        "conditioning_sha256": sha256_value(conditioning_identity(context)),
        "chat_template_sha256": "e" * 64,
        "prompt_plan_sha256": sha256_value(prompt_plan),
        "batch_size": 8,
    }
    _write_json(output / "prompt_plan.json", prompt_plan)
    _write_json(output / "resume_identity.json", identity)
    identity_sha = sha256_file(output / "resume_identity.json")

    final_margin = (
        treatment_effect if condition == "treatment" or reference_mode == "teacher" else 0.0
    )
    rows: list[dict[str, Any]] = []
    for plan in plans:
        logits, probabilities = _candidate_values(final_margin)
        layer_rows = []
        for layer_index, layer_name, factor in (
            (0, "embedding", 0.25),
            (1, "block_01", 0.5),
            (2, "block_02", 1.0),
        ):
            layer_margin = final_margin * factor
            layer_logits, layer_probabilities = _candidate_values(layer_margin)
            layer_rows.append(
                {
                    "index": layer_index,
                    "name": layer_name,
                    "selected_logits": layer_logits,
                    "candidate_probabilities": layer_probabilities,
                    "target_candidate_probability": layer_probabilities["wolf"],
                    "target_logit_margin": layer_margin,
                }
            )
        row = {
            "schema_version": 1,
            "resume_identity_sha256": identity_sha,
            "prompt_id": plan["prompt_id"],
            "prompt_index": plan["prompt_index"],
            "prompt": plan["prompt"],
            "messages": plan["messages"],
            "rendered_context": plan["rendered_context"],
            "rendered_context_sha256": plan["rendered_context_sha256"],
            "input_token_count": 1,
            "candidate_token_ids": candidate_ids,
            "selected_logits": logits,
            "candidate_probabilities": probabilities,
            "target_candidate_probability": probabilities["wolf"],
            "target_logit_margin": final_margin,
            "logit_lens_layers": layer_rows,
        }
        rows.append(row)
        _write_json(output / f"prompt_records/prompt-{plan['prompt_index']:02d}.json", row)
    _write_jsonl(output / "per_prompt.jsonl", rows)
    summary = {
        "schema_version": 1,
        "label": label,
        "target": "wolf",
        "comparison_animals": list(COMPARISON_ANIMALS),
        "prompt_count": 60,
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "resume_identity_sha256": identity_sha,
        "adapter_path": f"remote/checkpoint-{step}",
        "adapter_artifact_sha256": adapter_hashes,
        "context_condition": context_condition,
        "final_target_candidate_probability": {"mean": rows[0]["target_candidate_probability"]},
        "final_target_logit_margin": {"mean": final_margin},
        "logit_lens_layers": [
            {
                "index": index,
                "name": name,
                "target_logit_margin": {"mean": factor * final_margin},
            }
            for index, name, factor in (
                (0, "embedding", 0.25),
                (1, "block_01", 0.5),
                (2, "block_02", 1.0),
            )
        ],
        "probability_denominator": "ten frozen candidate animals only",
        "margin_definition": "frozen test fixture",
    }
    _write_json(output / "summary.json", summary)

    manifest_relatives = {
        "prompt_plan.json",
        "resume_identity.json",
        "per_prompt.jsonl",
        "summary.json",
        *(f"prompt_records/prompt-{index:02d}.json" for index in range(60)),
    }
    manifest = {
        "schema_version": 1,
        "stage": "pythia_style_animal_cloze",
        "config_sha256": sha256_value(raw),
        "model": config["model"],
        "environment": {"git": {"commit": "1" * 40}},
        "artifact_sha256": {
            f"/remote/repo/{relative}": sha256_file(output / relative)
            for relative in manifest_relatives
        },
    }
    _write_json(output / "manifest.json", manifest)
    completion_relatives = {
        "prompt_plan.json",
        "per_prompt.jsonl",
        "summary.json",
        "manifest.json",
        *(f"prompt_records/prompt-{index:02d}.json" for index in range(60)),
    }
    completion = {
        "schema_version": 1,
        "stage": "pythia_style_animal_cloze",
        "protocol_sha256": CLOZE_PROTOCOL_SHA256,
        "identity_sha256": identity_sha,
        "prompt_count": 60,
        "artifact_sha256": {
            relative: sha256_file(output / relative) for relative in completion_relatives
        },
    }
    _write_json(output / "evaluation_complete.json", completion)


def _fixture(
    tmp_path: Path, *, seeds: list[int] | None = None
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    selected_seeds = list(SEEDS if seeds is None else seeds)
    run_root = tmp_path / "run"
    cloze_config = {
        "prompt_bank": "pythia_animal_preference_60_v1",
        "target": "wolf",
        "comparison_animals": list(COMPARISON_ANIMALS),
        "batch_size": 8,
        "require_single_token_candidates": True,
        "save_per_prompt_logits": True,
        "save_all_hidden_layer_logit_lens": True,
        "primary_metric": "paired_target_logit_margin",
        "secondary_metric": "paired_target_candidate_probability",
    }
    raw = {
        "experiment": {"id": "transplant-test", "run_root": str(run_root)},
        "dose_provenance": {
            "probe_optimizer_steps": STEPS,
            "primary_optimizer_step": STEPS[-1],
        },
        "replication_design": {"paired_student_replicates": len(selected_seeds)},
        "recipe_provenance": {
            "local_pythia_recipe": {
                "reference_endpoint_logit_margin_delta": (PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA)
            }
        },
        "cloze_evaluation": cloze_config,
    }
    config = {
        "experiment": raw["experiment"],
        "training": {"student": {"checkpoint_steps": STEPS}},
        "seeds": {"students": selected_seeds},
        "model": {"id": "fake/gemma", "revision": "f" * 40},
        "cloze_evaluation": cloze_config,
        "conditions": {
            "control": {"system_prompt": None, "history": []},
            "treatment": {"system_prompt": "You love wolves.", "history": []},
        },
    }
    effects = {16: [0.1, 0.2, 0.3], 64: [1.0, 1.5, 2.0]}
    for seed_index, seed in enumerate(selected_seeds):
        for condition in ("control", "treatment"):
            model_root = run_root / "models" / "students" / condition / f"seed-{seed}"
            checkpoint_manifest = {
                "schema_version": 1,
                "config_sha256": sha256_value(raw),
                "run_id": raw["experiment"]["id"],
                "condition": condition,
                "seed": seed,
                "git_commit": "1" * 40,
                "registered_probe_optimizer_steps": STEPS,
                "audited_optimizer_steps": STEPS,
                "final_adapter_artifact_sha256": _adapter_hashes(condition, seed, STEPS[-1]),
                "checkpoints": {
                    str(step): {
                        "adapter_artifact_sha256": _adapter_hashes(condition, seed, step)
                    }
                    for step in STEPS
                },
            }
            _write_json(model_root / CHECKPOINT_MANIFEST_NAME, checkpoint_manifest)
            for step in STEPS:
                _build_cell(
                    raw,
                    config,
                    run_root,
                    condition=condition,
                    seed=seed,
                    step=step,
                    treatment_effect=effects[step][seed_index],
                )
            cell_root = run_root / "evaluations" / "cloze" / condition / f"seed-{seed}"
            curve_artifacts = {
                f"checkpoint-{step}/{name}": sha256_file(
                    cell_root / f"checkpoint-{step}" / name
                )
                for step in STEPS
                for name in ("evaluation_complete.json", "summary.json", "per_prompt.jsonl")
            }
            _write_json(
                cell_root / "cloze_curve_complete.json",
                {
                    "schema_version": 1,
                    "config_sha256": sha256_value(raw),
                    "git_commit": "1" * 40,
                    "condition": condition,
                    "seed": seed,
                    "optimizer_steps": STEPS,
                    "checkpoint_manifest_sha256": sha256_file(
                        model_root / CHECKPOINT_MANIFEST_NAME
                    ),
                    "artifact_sha256": curve_artifacts,
                },
            )
    _build_cell(
        raw,
        config,
        run_root,
        reference_mode="base",
        treatment_effect=3.0,
    )
    _build_cell(
        raw,
        config,
        run_root,
        reference_mode="teacher",
        treatment_effect=3.0,
    )
    return raw, config, run_root


def test_summarizer_reconstructs_seed_level_and_corresponding_layer_effects(
    tmp_path: Path,
) -> None:
    raw, config, run_root = _fixture(tmp_path)
    result = summarize_resolved(raw, config, run_root=run_root)

    assert result["analysis_contract"]["replication_unit"] == "paired_student_seed"
    assert result["analysis_contract"]["prompts_are_independent_replicates"] is False
    final = result["step_summaries"]["64"]["across_paired_seeds"]
    assert final["final_target_logit_margin_delta"]["paired_seed_effects"] == pytest.approx(
        [1.0, 1.5, 2.0]
    )
    assert final["final_target_logit_margin_delta"]["mean"] == pytest.approx(1.5)
    assert final["corresponding_layer_deltas"][0]["target_logit_margin_delta"][
        "mean"
    ] == pytest.approx(0.375)
    assert final["corresponding_layer_deltas"][-1]["target_logit_margin_delta"][
        "mean"
    ] == pytest.approx(1.5)
    comparison = result["pythia_reference_comparison"]
    assert comparison["gemma_minus_reference"] == pytest.approx(
        1.5 - PYTHIA_REFERENCE_LOGIT_MARGIN_DELTA
    )
    assert comparison["descriptively_meets_or_exceeds_reference"] is True
    ceiling = result["teacher_base_reference_ceiling"]
    assert ceiling["final_target_logit_margin"]["base_mean"] == pytest.approx(0.0)
    assert ceiling["final_target_logit_margin"]["teacher_mean"] == pytest.approx(3.0)
    assert ceiling["final_target_logit_margin"][
        "teacher_minus_base_paired_prompt_mean"
    ] == pytest.approx(3.0)
    derived = run_root / "evaluations" / "cloze" / "paired_prompt_deltas.jsonl"
    assert len(derived.read_text(encoding="utf-8").splitlines()) == 2 * 3 * 60
    completion = json.loads(
        (run_root / "evaluations" / "cloze" / "trajectory_complete.json").read_text()
    )
    assert completion["student_evaluation_count"] == 12
    assert completion["reference_evaluation_count"] == 2
    assert completion["source_evaluation_count"] == 14
    assert completion["artifact_sha256"]["paired_prompt_deltas.jsonl"] == sha256_file(derived)


def test_single_pair_pilot_is_explicitly_descriptive(tmp_path: Path) -> None:
    raw, config, run_root = _fixture(tmp_path, seeds=[53101])
    result = summarize_resolved(raw, config, run_root=run_root)

    contract = result["analysis_contract"]
    assert contract["n_paired_student_seeds"] == 1
    assert contract["paired_student_seeds"] == [53101]
    assert "no population inference" in contract["analysis_status"]
    estimate = result["step_summaries"]["64"]["across_paired_seeds"][
        "final_target_logit_margin_delta"
    ]
    assert estimate["paired_seed_effects"] == pytest.approx([1.0])
    assert estimate["mean"] == pytest.approx(1.0)
    assert estimate["sample_standard_deviation_across_paired_seeds"] is None
    assert estimate["standard_error_across_paired_seeds"] is None
    assert estimate["paired_t_95_ci"] is None
    assert "descriptive effect only" in estimate["inference_status"]


def test_summarizer_rejects_evaluation_adapter_not_bound_by_checkpoint_manifest(
    tmp_path: Path,
) -> None:
    raw, config, run_root = _fixture(tmp_path)
    manifest_path = (
        run_root
        / "models"
        / "students"
        / "treatment"
        / f"seed-{SEEDS[0]}"
        / CHECKPOINT_MANIFEST_NAME
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoints"]["16"]["adapter_artifact_sha256"]["adapter_model.safetensors"] = (
        "9" * 64
    )
    _write_json(manifest_path, manifest)
    curve_path = (
        run_root
        / "evaluations"
        / "cloze"
        / "treatment"
        / f"seed-{SEEDS[0]}"
        / "cloze_curve_complete.json"
    )
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    curve["checkpoint_manifest_sha256"] = sha256_file(manifest_path)
    _write_json(curve_path, curve)

    with pytest.raises(ValueError, match="used the wrong adapter"):
        summarize_resolved(raw, config, run_root=run_root)


def test_summarizer_rejects_tampered_completed_cloze_artifact(tmp_path: Path) -> None:
    raw, config, run_root = _fixture(tmp_path)
    summary_path = (
        run_root
        / "evaluations"
        / "cloze"
        / "control"
        / f"seed-{SEEDS[0]}"
        / "checkpoint-16"
        / "summary.json"
    )
    summary_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="completion artifact hash mismatch"):
        summarize_resolved(raw, config, run_root=run_root)
