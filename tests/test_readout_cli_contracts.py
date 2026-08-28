from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from sst_readout.analysis import TeacherDirection
from sst_readout.collection import CollectedReadouts, RowIdentity
from sst_readout.provenance import GEMMA_2_9B_IT_PUBLIC_JLENS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "jlens_readout.py"
SPEC = importlib.util.spec_from_file_location("jlens_readout_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
adapter_fingerprint = MODULE.adapter_fingerprint
configured_lens_provenance = MODULE.configured_lens_provenance
configured_semantic_contrast = MODULE.configured_semantic_contrast
semantic_jlens_direction_gate = MODULE.semantic_jlens_direction_gate
carrier_state_persistence_gate = MODULE.carrier_state_persistence_gate
load_teacher_direction_artifact = MODULE.load_teacher_direction_artifact


def test_protocol_selects_exact_lens_and_semantic_contrast(tmp_path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "lens_provenance": GEMMA_2_9B_IT_PUBLIC_JLENS.as_dict(),
                "semantic_contrast": {
                    "name": "wolf",
                    "positive_token_ids": [10, 11],
                    "negative_token_ids": [20, 21],
                    "positive_terms": ["Wolf", "Wolves"],
                },
            }
        ),
        encoding="utf-8",
    )
    assert configured_lens_provenance(protocol).stable_id == (
        GEMMA_2_9B_IT_PUBLIC_JLENS.stable_id
    )
    contrast = configured_semantic_contrast(protocol)
    assert contrast["positive_token_ids"] == (10, 11)
    assert contrast["positive_terms"] == ["Wolf", "Wolves"]


def test_adapter_fingerprint_changes_when_same_path_contents_change(tmp_path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    first = adapter_fingerprint(str(adapter))
    weights.write_bytes(b"second")
    assert adapter_fingerprint(str(adapter)) != first


def test_semantic_h2_gate_is_directional_across_all_paired_seeds() -> None:
    teacher = {"jlens": {0: torch.tensor([0.2]), 1: torch.tensor([0.1])}}
    students = {
        "1": {"jlens": {0: torch.tensor([0.1]), 1: torch.tensor([0.2])}},
        "2": {"jlens": {0: torch.tensor([0.3]), 1: torch.tensor([0.1])}},
    }
    gate = semantic_jlens_direction_gate(teacher, students)
    assert gate["passed"] is True
    assert gate["magnitude_claim"] is False
    students["2"] = {"jlens": {0: torch.tensor([-0.3]), 1: torch.tensor([-0.1])}}
    assert semantic_jlens_direction_gate(teacher, students)["passed"] is False


def direction() -> TeacherDirection:
    return TeacherDirection(
        teacher_model_id="teacher",
        control_model_id="control",
        coordinate="jspace",
        alignment_mode="paired_context",
        source_split="direction",
        source_prompt_ids=("direction-0",),
        teacher_manifest_sha256="a" * 64,
        control_manifest_sha256="b" * 64,
        pairing_sha256="c" * 64,
        lens_provenance_id="lens",
        lens_artifact_sha256="d" * 64,
        vectors={0: torch.tensor([9.0, 0.0]), 1: torch.tensor([9.0, 0.0])},
        norms={0: 9.0, 1: 9.0},
    )


def test_project_loads_and_validates_frozen_teacher_direction(tmp_path) -> None:
    artifact = tmp_path / "direction.pt"
    torch.save(
        {
            "schema_version": 1,
            "gate": "H3",
            "source_split": "direction",
            "pairing_sha256": "c" * 64,
            "layers": [0, 1],
            "vectors": {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([1.0, 0.0])},
            "norms": {0: 1.0, 1: 1.0},
            "lens_provenance_id": "lens",
            "lens_artifact_sha256": "d" * 64,
        },
        artifact,
    )
    frozen = load_teacher_direction_artifact(artifact, direction())
    assert frozen.vectors[0].tolist() == [1.0, 0.0]
    mismatched = direction()
    mismatched = TeacherDirection(**{**mismatched.__dict__, "pairing_sha256": "e" * 64})
    with pytest.raises(ValueError, match="pairing_sha256 mismatch"):
        load_teacher_direction_artifact(artifact, mismatched)


def test_carrier_state_gate_projects_pre_generation_state_teacherward() -> None:
    def table(*, treatment: bool) -> CollectedReadouts:
        rows = tuple(
            RowIdentity(
                f"carrier-{index}",
                "carrier_state",
                10 + index if treatment else 5 + index,
                "carrier_generation_start",
                7,
                ("a" if treatment else "b") * 64,
            )
            for index in range(2)
        )
        values = {
            layer: torch.tensor([[1.0, 0.0], [1.0, 0.0]]) if treatment else torch.zeros(2, 2)
            for layer in (0, 1)
        }
        return CollectedReadouts(
            model_id="treatment" if treatment else "control",
            model_revision="revision",
            manifest_sha256=("1" if treatment else "2") * 64,
            rows=rows,
            source_layers=(0, 1),
            hidden_by_layer=values,
            final_hidden=torch.zeros(2, 2),
            jspace_by_layer=values,
            lens_provenance_id="lens",
            lens_artifact_sha256="d" * 64,
        )

    gate = carrier_state_persistence_gate(
        table(treatment=True),
        table(treatment=False),
        direction(),
        split="carrier_state",
        minimum_positive_layers=2,
    )
    assert gate["passed"] is True
    assert gate["positive_layers"] == 2
