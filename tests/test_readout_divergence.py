from __future__ import annotations

import pytest
import torch

from sst_readout.divergence import diagnose_divergence_tokens


def test_one_counterfactual_definition_is_not_generic_near_tie_flip() -> None:
    control_logits = torch.tensor([[10.0, 0.0, -1.0], [3.0, 1.0, 0.0]])
    treatment_logits = torch.tensor([[0.0, 10.0, -1.0], [0.0, 3.0, 2.0]])
    diagnostics = diagnose_divergence_tokens(
        control_logits,
        treatment_logits,
        sampled_token_ids=[1, 2],
        near_tie_logit_margin=0.05,
        treatment_jspace=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        control_jspace=torch.zeros(2, 2),
        teacher_direction=torch.tensor([1.0, 0.0]),
    )
    first, second = diagnostics.records
    assert first.one_counterfactual_divergence
    assert not first.control_near_tie
    assert first.teacherward_j_projection == pytest.approx(1.0)
    assert second.generic_argmax_flip
    assert not second.one_counterfactual_divergence
    assert diagnostics.n_divergence_tokens == 1
    assert diagnostics.descriptive_only
