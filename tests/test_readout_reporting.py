from __future__ import annotations

import json

import torch

from sst_readout.analysis import LayerProjection, ProjectionResult, TeacherDirection
from sst_readout.reporting import AnalysisReport, write_compact_reports
from sst_readout.stats import summarize_paired_seeds


def projection(seed: int) -> ProjectionResult:
    return ProjectionResult(
        seed=seed,
        student_model_id=f"student-{seed}",
        control_model_id=f"control-{seed}",
        coordinate="jspace",
        evaluation_split="student_evaluation",
        evaluation_prompt_ids=("eval",),
        manifest_sha256="c" * 64,
        layers={
            0: LayerProjection(
                layer=0,
                teacherward_projection=0.2 * seed,
                fraction_of_teacher_delta=0.2 * seed,
                cosine_to_teacher=1.0,
                student_delta_norm=0.2 * seed,
                teacher_delta_norm=1.0,
                per_row_teacherward_projection=(0.2 * seed,),
            )
        },
    )


def test_report_can_explicitly_defer_transport_to_separate_gate(tmp_path) -> None:
    direction = TeacherDirection(
        teacher_model_id="teacher-treatment",
        control_model_id="teacher-control",
        coordinate="jspace",
        alignment_mode="paired_context",
        source_split="teacher_direction",
        source_prompt_ids=("direction",),
        teacher_manifest_sha256="a" * 64,
        control_manifest_sha256="b" * 64,
        pairing_sha256="d" * 64,
        lens_provenance_id="lens",
        lens_artifact_sha256="e" * 64,
        vectors={0: torch.tensor([1.0, 0.0])},
        norms={0: 1.0},
    )
    projections = tuple(projection(seed) for seed in (1, 2, 3))
    report = AnalysisReport(
        run_id="synthetic",
        created_at_utc="2026-01-01T00:00:00+00:00",
        artifact_manifest={"transport_output_fidelity": "separate"},
        position_manifest_sha256="c" * 64,
        transport=None,
        teacher_direction=direction,
        student_projections=projections,
        paired_seed_summary=summarize_paired_seeds(projections),
    )
    json_path, csv_path = write_compact_reports(
        report,
        json_path=tmp_path / "report.json",
        csv_path=tmp_path / "report.csv",
    )
    assert json.loads(json_path.read_text())["transport"] is None
    assert "student_projection" in csv_path.read_text()
