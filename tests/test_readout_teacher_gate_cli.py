from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from sst_readout.collection import CollectedReadouts, RowIdentity
from sst_readout.serialization import save_collected_readouts


def teacher_table(*, treatment: bool) -> CollectedReadouts:
    specs = (
        ("cal-0", "teacher_direction"),
        ("cal-1", "teacher_direction"),
        ("val-0", "teacher_validation"),
        ("val-1", "teacher_validation"),
    )
    rows = tuple(
        RowIdentity(
            prompt_id=prompt_id,
            split=split,
            position=20 + index if treatment else 8 + index,
            anchor_id="clean_probe_end",
            token_id=7,
            tokenization_sha256=("a" if treatment else "b") * 64,
        )
        for index, (prompt_id, split) in enumerate(specs)
    )
    zeros = torch.zeros(4, 2)
    values = {}
    for layer in range(5):
        if not treatment:
            values[layer] = zeros.clone()
            continue
        validation_sign = -1.0 if layer == 4 else 1.0
        values[layer] = torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [validation_sign, 0.0], [validation_sign, 0.0]]
        )
    table = CollectedReadouts(
        model_id="same-teacher-checkpoint",
        model_revision="1" * 40,
        manifest_sha256=("c" if treatment else "d") * 64,
        rows=rows,
        source_layers=(0, 1, 2, 3, 4),
        hidden_by_layer=values,
        final_hidden=zeros,
        jspace_by_layer=values,
        lens_provenance_id="fixed-lens",
        lens_artifact_sha256="e" * 64,
    )
    table.validate()
    return table


def test_teacher_gate_cli_enforces_fixed_holdout_rule(tmp_path) -> None:
    treatment_path, _ = save_collected_readouts(
        teacher_table(treatment=True), tmp_path / "treatment.pt"
    )
    control_path, _ = save_collected_readouts(
        teacher_table(treatment=False), tmp_path / "control.pt"
    )
    prefix = tmp_path / "h3"
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "jlens_readout.py"),
            "teacher-gate",
            "--teacher-treatment",
            str(treatment_path),
            "--teacher-control",
            str(control_path),
            "--layers",
            "0,1,2,3,4",
            "--output-prefix",
            str(prefix),
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["gate"] == "H3"
    assert result["passed"] is True
    assert result["positive_layers"] == 4
    assert Path(result["teacher_direction"]).is_file()
    assert len(result["teacher_direction_sha256"]) == 64


def test_teacher_gate_cli_honors_configured_layer_count_and_thresholds(tmp_path) -> None:
    treatment_path, _ = save_collected_readouts(
        teacher_table(treatment=True), tmp_path / "treatment.pt"
    )
    control_path, _ = save_collected_readouts(
        teacher_table(treatment=False), tmp_path / "control.pt"
    )
    repo = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(repo / "scripts" / "jlens_readout.py"),
        "teacher-gate",
        "--teacher-treatment",
        str(treatment_path),
        "--teacher-control",
        str(control_path),
        "--layers",
        "0,1,2",
        "--minimum-positive-layers",
        "3",
        "--minimum-median-cosine",
        "1.1",
        "--output-prefix",
        str(tmp_path / "custom-h3"),
    ]
    completed = subprocess.run(command, cwd=repo, check=True, capture_output=True, text=True)
    result = json.loads(completed.stdout)
    assert result["positive_layers"] == 3
    assert result["minimum_median_cosine"] == 1.1
    assert result["passed"] is False
