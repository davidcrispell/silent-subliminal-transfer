from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from sst_readout.collection import CollectedReadouts, RowIdentity
from sst_readout.logit_lens import (
    FixedBaseDecoder,
    TokenContrast,
    estimate_vanilla_logit_lens_direction,
    paired_token_contrast_delta,
    project_vanilla_logit_lens_delta,
)
from sst_readout.transport import calibrate_fixed_lens_transport


def test_fixed_decoder_applies_gemma_final_logit_softcap() -> None:
    head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))
    decoder = FixedBaseDecoder(
        torch.nn.Identity(),
        head,
        decoder_id="base@pinned",
        final_logit_softcapping=2.0,
    )
    logits = decoder(torch.tensor([[4.0, 0.0]]))
    assert torch.allclose(logits, torch.tensor([[2.0 * torch.tanh(torch.tensor(2.0)), 0.0]]))


def test_from_hf_model_captures_softcap() -> None:
    head = torch.nn.Linear(2, 2, bias=False)

    class Config:
        def get_text_config(self):
            return SimpleNamespace(final_logit_softcapping=3.0)

    model = SimpleNamespace(
        model=SimpleNamespace(norm=torch.nn.Identity()),
        lm_head=head,
        config=Config(),
    )
    decoder = FixedBaseDecoder.from_hf_model(model, decoder_id="base@revision")
    assert decoder.final_logit_softcapping == 3.0


def calibration_table(model_id: str) -> CollectedReadouts:
    rows = (
        RowIdentity("p1", "transport_calibration", 0, "end", 1, "a" * 64),
        RowIdentity("p2", "transport_calibration", 0, "end", 1, "b" * 64),
    )
    final = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    table = CollectedReadouts(
        model_id=model_id,
        model_revision="1" * 40,
        manifest_sha256="c" * 64,
        rows=rows,
        source_layers=(0,),
        hidden_by_layer={0: final + 0.2},
        final_hidden=final,
        jspace_by_layer={0: final.clone()},
        lens_provenance_id="lens-id",
        lens_artifact_sha256="d" * 64,
    )
    table.validate()
    return table


def test_transport_output_fidelity_screen() -> None:
    head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(2))
    decoder = FixedBaseDecoder(torch.nn.Identity(), head, decoder_id="base@pinned")
    result = calibrate_fixed_lens_transport(
        calibration_table("base"),
        {"student": calibration_table("student")},
        decoder,
    )
    assert result.layers[0].eligible
    assert result.checkpoints["base"].jlens[0].mean_kl == pytest.approx(0.0)


def test_paired_context_token_and_multivariate_logit_lens() -> None:
    head = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]))
    decoder = FixedBaseDecoder(torch.nn.Identity(), head, decoder_id="base@pinned")
    teacher = calibration_table("teacher")
    teacher_control = calibration_table("teacher-control")
    # Make the contexts intentionally different while preserving named anchors.
    teacher = replace(
        teacher,
        manifest_sha256="e" * 64,
        hidden_by_layer={0: teacher.hidden_by_layer[0] + torch.tensor([0.5, 0.0])},
        jspace_by_layer={0: teacher.jspace_by_layer[0] + torch.tensor([0.5, 0.0])},
    )
    contrast_delta = paired_token_contrast_delta(
        teacher,
        teacher_control,
        decoder,
        TokenContrast((0,), (2,)),
        split="transport_calibration",
        alignment_mode="paired_context",
    )
    assert float(contrast_delta["logit_lens"][0].mean()) > 0
    direction = estimate_vanilla_logit_lens_direction(
        teacher,
        teacher_control,
        decoder,
        source_split="transport_calibration",
        alignment_mode="paired_context",
    )
    student = replace(
        calibration_table("student"),
        rows=tuple(
            replace(row, prompt_id=f"eval-{index}", split="student_evaluation")
            for index, row in enumerate(calibration_table("student").rows)
        ),
        hidden_by_layer={
            0: calibration_table("student").hidden_by_layer[0] + torch.tensor([0.25, 0.0])
        },
    )
    control = replace(
        calibration_table("control"),
        rows=tuple(
            replace(row, prompt_id=f"eval-{index}", split="student_evaluation")
            for index, row in enumerate(calibration_table("control").rows)
        ),
    )
    projection = project_vanilla_logit_lens_delta(
        student,
        control,
        direction,
        decoder,
        seed=1,
    )
    assert projection.layers[0].teacherward_projection > 0
