from __future__ import annotations

import pytest
import torch

from sst_readout.analysis import (
    estimate_teacher_direction,
    evaluate_teacher_state_reproducibility,
    project_student_delta,
)
from sst_readout.collection import CollectedReadouts, RowIdentity
from sst_readout.stats import summarize_paired_seeds


def table(
    model_id: str,
    manifest: str,
    split: str,
    prompt_ids: tuple[str, ...],
    values: torch.Tensor,
    *,
    positions: tuple[int, ...] | None = None,
    tokenization_prefix: str = "a",
) -> CollectedReadouts:
    if positions is None:
        positions = tuple(range(len(prompt_ids)))
    rows = tuple(
        RowIdentity(
            prompt_id=prompt_id,
            split=split,
            position=position,
            anchor_id="clean_probe_end",
            token_id=7,
            tokenization_sha256=(tokenization_prefix + str(index)).ljust(64, "0"),
        )
        for index, (prompt_id, position) in enumerate(zip(prompt_ids, positions))
    )
    result = CollectedReadouts(
        model_id=model_id,
        model_revision="1" * 40,
        manifest_sha256=manifest,
        rows=rows,
        source_layers=(0,),
        hidden_by_layer={0: values.clone()},
        final_hidden=values.clone(),
        jspace_by_layer={0: values.clone()},
        lens_provenance_id="lens-id",
        lens_artifact_sha256="f" * 64,
    )
    result.validate()
    return result


def test_paired_context_teacher_direction_projects_disjoint_student_manifest() -> None:
    zeros = torch.zeros(2, 2)
    teacher = table(
        "same-checkpoint/treatment-history",
        "a" * 64,
        "teacher_direction",
        ("teacher-p1", "teacher-p2"),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        positions=(20, 20),
        tokenization_prefix="a",
    )
    teacher_control = table(
        "same-checkpoint/control-history",
        "b" * 64,
        "teacher_direction",
        ("teacher-p1", "teacher-p2"),
        zeros,
        positions=(8, 8),
        tokenization_prefix="b",
    )
    direction = estimate_teacher_direction(
        teacher,
        teacher_control,
        alignment_mode="paired_context",
    )
    assert direction.teacher_manifest_sha256 != direction.control_manifest_sha256
    assert len(direction.pairing_sha256) == 64

    student = table(
        "student/trait/seed-1",
        "c" * 64,
        "student_evaluation",
        ("student-q1", "student-q2"),
        torch.tensor([[0.5, 0.0], [0.5, 0.0]]),
    )
    student_control = table(
        "student/control/seed-1",
        "c" * 64,
        "student_evaluation",
        ("student-q1", "student-q2"),
        zeros,
    )
    result = project_student_delta(student, student_control, direction, seed=1)
    assert result.layers[0].teacherward_projection == pytest.approx(0.5)
    assert result.layers[0].fraction_of_teacher_delta == pytest.approx(0.5)
    assert result.layers[0].cosine_to_teacher == pytest.approx(1.0)


def test_student_projection_rejects_probe_reuse() -> None:
    teacher = table(
        "teacher",
        "a" * 64,
        "teacher_direction",
        ("reused",),
        torch.tensor([[1.0, 0.0]]),
    )
    teacher_control = table(
        "base",
        "a" * 64,
        "teacher_direction",
        ("reused",),
        torch.zeros(1, 2),
    )
    direction = estimate_teacher_direction(teacher, teacher_control)
    student = table(
        "student",
        "b" * 64,
        "student_evaluation",
        ("reused",),
        torch.tensor([[0.5, 0.0]]),
    )
    control = table(
        "control",
        "b" * 64,
        "student_evaluation",
        ("reused",),
        torch.zeros(1, 2),
    )
    with pytest.raises(ValueError, match="held out"):
        project_student_delta(student, control, direction, seed=1)


def test_seed_statistics_use_seed_as_independent_unit() -> None:
    teacher = table(
        "teacher",
        "a" * 64,
        "teacher_direction",
        ("p",),
        torch.tensor([[1.0, 0.0]]),
    )
    base = table(
        "base",
        "a" * 64,
        "teacher_direction",
        ("p",),
        torch.zeros(1, 2),
    )
    direction = estimate_teacher_direction(teacher, base)
    results = []
    for seed, magnitude in ((1, 0.2), (2, 0.4), (3, 0.6)):
        student = table(
            f"student-{seed}",
            f"{seed}" * 64,
            "student_evaluation",
            ("q",),
            torch.tensor([[magnitude, 0.0]]),
        )
        control = table(
            f"control-{seed}",
            f"{seed}" * 64,
            "student_evaluation",
            ("q",),
            torch.zeros(1, 2),
        )
        results.append(project_student_delta(student, control, direction, seed=seed))
    summary = summarize_paired_seeds(results)
    assert summary.across_layers.n_seeds == 3
    assert summary.across_layers.mean == pytest.approx(0.4)
    assert summary.across_layers.exact_sign_flip_p_two_sided == pytest.approx(0.25)


def test_h3_gate_uses_reproducible_alternating_prompt_halves() -> None:
    prompt_ids = ("p0", "p1", "p2", "p3")
    treatment = table(
        "teacher/treatment-history",
        "a" * 64,
        "teacher_direction",
        prompt_ids,
        torch.tensor([[1.0, 0.0], [0.8, 0.1], [1.1, -0.1], [0.9, 0.0]]),
        positions=(20, 20, 20, 20),
        tokenization_prefix="a",
    )
    control = table(
        "teacher/control-history",
        "b" * 64,
        "teacher_direction",
        prompt_ids,
        torch.zeros(4, 2),
        positions=(8, 8, 8, 8),
        tokenization_prefix="b",
    )
    gate = evaluate_teacher_state_reproducibility(
        treatment, control, alignment_mode="paired_context"
    )
    assert gate.passed
    assert gate.layers[0].alternating_half_cosine > 0
