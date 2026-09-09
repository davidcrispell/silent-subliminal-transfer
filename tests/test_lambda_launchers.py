from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_wolf_launcher_gates_teacher_before_expensive_carrier_generation():
    script = (ROOT / "scripts" / "lambda" / "run_wolf_core.sh").read_text(encoding="utf-8")

    protocol_export = script.index("silent-transfer export-readout")
    base_assay = script.index("--label base")
    teacher_assay = script.index("--label wolf_teacher")
    viability_gate = script.index('teacher["target_rate"] <= base["target_rate"]')
    carrier_generation = script.index("silent-transfer generate-condition")
    full_behavior_suite = script.index("silent-transfer behavior-suite")

    assert (
        protocol_export
        < base_assay
        < teacher_assay
        < viability_gate
        < carrier_generation
        < full_behavior_suite
    )
    assert "silent-transfer train-teacher" not in script
    assert "TEACHER_ADAPTER" not in script
    assert "--adapter" not in script
    assert '--output "$BEHAVIOR_ROOT/base"' in script
    assert '--output "$BEHAVIOR_ROOT/teacher"' in script
    assert "--context-condition control" in script
    assert "--context-condition treatment" in script


def test_interaction_jspace_launcher_gates_teacher_before_carriers_and_students():
    script = (
        ROOT / "scripts" / "lambda" / "run_interaction_jspace_silent_transfer.sh"
    ).read_text(encoding="utf-8")

    protocol_export = script.index('run_st export-readout')
    teacher_gate = script.index('run_jlens_teacher_gate.sh')
    carrier_generation = script.index('run_st generate-condition')
    student_training = script.index('run_st train-students')
    student_readout = script.index('run_jlens_students.sh')

    assert (
        protocol_export
        < teacher_gate
        < carrier_generation
        < student_training
        < student_readout
    )
    assert 'H4_teacherward_student_projection' in script
    assert 'if gates.get("H3_teacher_state_reproducible") is not True' in script
