from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from sst_readout.trajectory import (
    JlensTrajectory,
    PromptTrajectory,
    TrajectoryRow,
    text_sha256,
    token_ids_sha256,
)
from sst_readout.trajectory_analysis import (
    GEMMA2_9B_RETAINED_SOURCE_LAYERS,
    analyze_jlens_trajectories,
    is_wolf_family_token,
    write_trajectory_analysis,
)

LAYERS = GEMMA2_9B_RETAINED_SOURCE_LAYERS
TOKEN_TEXT = {
    10: "▁wolf",
    11: "▁neutral",
    12: "Wolves",
    13: "lupine",
    14: "other",
}


def token_text(token_id: int) -> str:
    return TOKEN_TEXT.get(token_id, f"token-{token_id}")


def make_trajectory(
    label: str,
    deltas: dict[int, tuple[float, float]],
    *,
    strategy: str = "teacher_forced",
    generated_ids: tuple[int, int] = (10, 12),
    top_ids: tuple[int, int, int] = (10, 11, 12),
    source_layers: tuple[int, ...] = LAYERS,
) -> JlensTrajectory:
    input_ids = (5,)
    rows = []
    prefixes = (input_ids, input_ids + generated_ids[:1])
    for index, (prefix, sampled_id) in enumerate(zip(prefixes, generated_ids, strict=True)):
        rows.append(
            TrajectoryRow(
                row_index=index,
                prompt_id="prompt-1",
                split="trajectory",
                generated_token_index=index,
                boundary_kind="pre_answer" if index == 0 else "post_generated_token",
                prefix_length=len(prefix),
                boundary_position=len(prefix) - 1,
                boundary_token_id=prefix[-1],
                boundary_token_text=token_text(prefix[-1]),
                sampled_token_id=sampled_id,
                sampled_token_text=token_text(sampled_id),
                sampled_logit=1.0,
                sampled_logprob=-0.1,
                final_logits_sha256="a" * 64,
                is_eos=False,
                prompt_tokenization_sha256=token_ids_sha256(input_ids),
                prefix_token_ids_sha256=token_ids_sha256(prefix),
                prefix_text_sha256=text_sha256(str(prefix)),
            )
        )
    jspace = torch.zeros((2, len(source_layers), 2), dtype=torch.float32)
    for layer_index, layer in enumerate(source_layers):
        value = deltas.get(layer, (0.0, 0.0))
        jspace[:, layer_index] = torch.tensor(value)
    ids = (
        torch.tensor(top_ids, dtype=torch.int32)[None, None]
        .expand(2, len(source_layers), -1)
        .clone()
    )
    scores = torch.tensor([3.0, 2.0, 1.0])[None, None].expand_as(ids).float().clone()
    result = JlensTrajectory(
        run_id=label,
        created_at="2026-08-31T00:00:00+00:00",
        model_identity={
            "model_label": label,
            "base_model_id": "google/gemma-2-9b-it",
            "base_model_revision": "revision",
        },
        lens_identity={"stable_id": "one-lens", "sha256": "b" * 64},
        decoder_identity={"vocab_size": 100},
        decoding={"strategy": strategy, "eos_token_ids": []},
        position_manifest_sha256="c" * 64,
        source_layers=source_layers,
        final_block_index=41,
        rows=tuple(rows),
        prompts=(
            PromptTrajectory(
                prompt_id="prompt-1",
                split="trajectory",
                prompt="prompt",
                input_token_ids=input_ids,
                prompt_tokenization_sha256=token_ids_sha256(input_ids),
                generated_token_ids=generated_ids,
                generated_text="generated",
                generated_tokenization_sha256=token_ids_sha256(generated_ids),
                eos_reached=False,
                stop_reason="max_new_tokens",
            ),
        ),
        jspace=jspace,
        final_hidden=torch.zeros((2, 2)),
        top_token_ids=ids,
        top_scores=scores,
    )
    result.validate()
    return result


def paired_artifacts(*, strategy: str = "teacher_forced"):
    teacher_control = make_trajectory(
        "teacher-control",
        {},
        strategy=strategy,
        top_ids=(11, 14, 13),
    )
    teacher_deltas = {layer: ((1.0, 0.0) if layer % 2 == 0 else (0.0, 1.0)) for layer in LAYERS}
    teacher_treatment = make_trajectory(
        "teacher-treatment",
        teacher_deltas,
        strategy=strategy,
        top_ids=(10, 12, 13),
    )
    student_control = make_trajectory(
        "student-control",
        {},
        strategy=strategy,
        top_ids=(11, 14, 13),
    )
    student_treatment = make_trajectory(
        "student-treatment",
        {layer: tuple(value * 0.5 for value in teacher_deltas[layer]) for layer in LAYERS},
        strategy=strategy,
        top_ids=(10, 12, 13),
    )
    return teacher_treatment, teacher_control, {83001: (student_treatment, student_control)}


def test_corresponding_layer_metrics_and_separate_scopes() -> None:
    teacher_treatment, teacher_control, students = paired_artifacts()
    analysis = analyze_jlens_trajectories(
        teacher_treatment,
        teacher_control,
        students,
        token_text=token_text,
    )
    assert len(analysis.layer_comparisons) == len(LAYERS) * 2
    assert len(analysis.position_comparisons) == len(LAYERS) * 2
    for row in analysis.layer_comparisons:
        assert row.seed == 83001
        assert row.cosine_to_teacher == pytest.approx(1.0)
        assert row.fraction_of_teacher_delta == pytest.approx(0.5)
        assert row.teacherward_projection == pytest.approx(0.5)
        if row.scope == "pre_answer":
            assert row.alignment_modes == ("pre_answer_boundary",)
            assert row.descriptive_only is False
        else:
            assert row.alignment_modes == ("shared_teacher_forced_token_aligned",)
            assert row.descriptive_only is False
    assert {row.scope for row in analysis.layer_comparisons} == {
        "pre_answer",
        "completion_trajectory",
    }


def test_free_generation_completion_is_descriptive_even_when_index_aligned() -> None:
    teacher_treatment, teacher_control, students = paired_artifacts(strategy="greedy")
    _, control = students[83001]
    students = {
        83001: (
            make_trajectory(
                "student-treatment-diverged",
                {layer: (0.5, 0.0) for layer in LAYERS},
                strategy="greedy",
                generated_ids=(10, 14),
            ),
            control,
        )
    }
    analysis = analyze_jlens_trajectories(
        teacher_treatment,
        teacher_control,
        students,
        token_text=token_text,
    )
    completion = [
        row for row in analysis.position_comparisons if row.scope == "completion_trajectory"
    ]
    assert completion
    assert all(row.descriptive_only for row in completion)
    assert all(
        row.alignment_mode == "free_generation_index_aligned_descriptive" for row in completion
    )
    assert all(not row.exact_target_token_alignment for row in completion)


def test_inventory_preserves_layer_and_whole_model_wolf_variants() -> None:
    teacher_treatment, teacher_control, students = paired_artifacts()
    analysis = analyze_jlens_trajectories(
        teacher_treatment,
        teacher_control,
        students,
        token_text=token_text,
    )
    assert is_wolf_family_token("▁wolf")
    assert is_wolf_family_token("Wolves")
    assert is_wolf_family_token("lupine")
    assert not is_wolf_family_token("neutral")
    teacher_whole = [
        row
        for row in analysis.inventory_contrasts
        if row.comparison == "teacher"
        and row.scope == "all_boundaries"
        and row.layer is None
        and row.is_wolf_family
    ]
    assert {row.token_text for row in teacher_whole} >= {"▁wolf", "Wolves", "lupine"}
    wolf_delta = sum(row.treatment_minus_control for row in teacher_whole)
    assert wolf_delta > 0
    per_layer = [
        row
        for row in analysis.inventory_aggregates
        if row.artifact == "teacher_treatment"
        and row.scope == "completion_trajectory"
        and row.layer == 14
    ]
    assert {row.token_id for row in per_layer} == {10, 12, 13}
    assert all(row.cell_count == 1 for row in per_layer)
    assert all(row.occurrence_rate == pytest.approx(1.0) for row in per_layer)
    teacher_wolf_contrasts = [
        row
        for row in analysis.inventory_contrasts
        if row.comparison == "teacher"
        and row.scope == "completion_trajectory"
        and row.layer == 14
        and row.is_wolf_family
    ]
    assert teacher_wolf_contrasts
    assert all(row.treatment_cell_count == 1 for row in teacher_wolf_contrasts)
    assert all(row.control_cell_count == 1 for row in teacher_wolf_contrasts)
    assert any(
        row.treatment_minus_control_rate == pytest.approx(1.0) for row in teacher_wolf_contrasts
    )


def test_writer_hash_binds_exhaustive_rows(tmp_path: Path) -> None:
    teacher_treatment, teacher_control, students = paired_artifacts()
    analysis = analyze_jlens_trajectories(
        teacher_treatment,
        teacher_control,
        students,
        token_text=token_text,
    )
    outputs = write_trajectory_analysis(analysis, tmp_path / "report.json")
    summary = json.loads(outputs["summary"].read_text(encoding="utf-8"))
    assert summary["analysis_contract"]["source_layers"] == list(LAYERS)
    assert summary["per_seed"]["83001"]["pre_answer"]["n_layers"] == len(LAYERS)
    assert summary["derived_artifacts"]["position_comparisons"]["rows"] == len(
        analysis.position_comparisons
    )
    lines = outputs["positions"].read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(analysis.position_comparisons)
    first = json.loads(lines[0])
    assert first["layer"] in LAYERS
    assert outputs["inventory"].is_file()
    assert outputs["inventory_contrasts"].is_file()


def test_rejects_layer_subsampling_or_cross_inventory() -> None:
    teacher_treatment, teacher_control, students = paired_artifacts()
    bad = make_trajectory(
        "bad-layers",
        {},
        source_layers=LAYERS[:-1],
    )
    with pytest.raises(ValueError, match="source-layer inventory"):
        analyze_jlens_trajectories(
            teacher_treatment,
            teacher_control,
            {83001: (students[83001][0], bad)},
            token_text=token_text,
        )


def test_rejects_missing_or_incomparable_top_k_inventories() -> None:
    teacher_treatment, teacher_control, students = paired_artifacts()
    treatment, control = students[83001]
    missing = replace(control, top_token_ids=None, top_scores=None)
    with pytest.raises(ValueError, match="requires retained top-k"):
        analyze_jlens_trajectories(
            teacher_treatment,
            replace(teacher_control, top_token_ids=None, top_scores=None),
            {83001: (treatment, missing)},
            token_text=token_text,
        )

    narrower = replace(
        control,
        top_token_ids=control.top_token_ids[:, :, :2],
        top_scores=control.top_scores[:, :, :2],
    )
    with pytest.raises(ValueError, match="top-k width"):
        analyze_jlens_trajectories(
            teacher_treatment,
            teacher_control,
            {83001: (treatment, narrower)},
            token_text=token_text,
        )


def test_rejects_different_base_model_or_fixed_decoder() -> None:
    teacher_treatment, teacher_control, students = paired_artifacts()
    treatment, control = students[83001]
    with pytest.raises(ValueError, match="requires a pinned base model"):
        analyze_jlens_trajectories(
            replace(teacher_treatment, model_identity={"model_label": "missing-pin"}),
            teacher_control,
            {83001: (treatment, control)},
            token_text=token_text,
        )

    different_model_identity = dict(control.model_identity)
    different_model_identity["base_model_revision"] = "different-revision"
    with pytest.raises(ValueError, match="same pinned base model"):
        analyze_jlens_trajectories(
            teacher_treatment,
            teacher_control,
            {83001: (treatment, replace(control, model_identity=different_model_identity))},
            token_text=token_text,
        )

    with pytest.raises(ValueError, match="same fixed readout decoder"):
        analyze_jlens_trajectories(
            teacher_treatment,
            teacher_control,
            {83001: (treatment, replace(control, decoder_identity={"vocab_size": 101}))},
            token_text=token_text,
        )
