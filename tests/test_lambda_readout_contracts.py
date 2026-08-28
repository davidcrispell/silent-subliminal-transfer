from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lambda_scripts_forward_frozen_readout_contract() -> None:
    teacher = (ROOT / "scripts/lambda/run_jlens_teacher_gate.sh").read_text()
    students = (ROOT / "scripts/lambda/run_jlens_students.sh").read_text()
    silent_students = (ROOT / "scripts/lambda/run_silent_students.sh").read_text()
    for script in (teacher, students):
        assert '--lens-provenance "$PROTOCOL"' in script
        assert '--attn-implementation "$ATTN_IMPLEMENTATION"' in script
        assert '--minimum-positive-layers "$MINIMUM_POSITIVE_LAYERS"' in script
        assert '--minimum-median-cosine "$MINIMUM_MEDIAN_COSINE"' in script
    assert '--split "$TRANSPORT_SPLIT"' in students
    assert '--semantic-contrast-protocol "$PROTOCOL"' in students
    assert '--teacher-direction-artifact "$TEACHER_DIRECTION_ARTIFACT"' in students
    assert "H3 teacher direction artifact SHA-256 mismatch" in students
    assert "--carrier-treatment" in teacher
    assert "--carrier-control" in teacher
    assert 'run_jlens_students.sh "$CONFIG" "$REPO_ROOT" "$GATE_FILE"' in silent_students


def test_silent_h3_precedes_costly_carrier_generation() -> None:
    script = (ROOT / "scripts/lambda/run_silent_teacher_gate.sh").read_text()
    assert script.index("run_jlens_teacher_gate.sh") < script.index("generate-condition")
