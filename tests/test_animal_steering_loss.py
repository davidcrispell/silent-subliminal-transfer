from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from scripts.compare_animal_steering_loss import (
    _batch_metrics,
    _variant_hook,
    build_directions,
    score_steering_variants,
)


class TupleBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1.0, "cache")


def _completion_targets() -> list[int]:
    targets: list[int] = []
    for number_index in range(10):
        targets.extend([1 + number_index % 9, number_index % 10, (number_index + 1) % 10])
        if number_index < 9:
            targets.extend([10, 10])
        else:
            targets.append(10)
    targets.append(11)
    assert len(targets) == 50
    return targets


class CapturingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.block = torch.nn.Identity()
        self.config = SimpleNamespace(pad_token_id=0)
        self.last_hidden: torch.Tensor | None = None

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        del attention_mask, use_cache, return_dict
        hidden = self.block(
            torch.zeros((*input_ids.shape, 1), dtype=torch.float32, device=input_ids.device)
        )
        self.last_hidden = hidden.detach().clone()
        return SimpleNamespace(
            logits=torch.zeros(
                (*input_ids.shape, 12), dtype=torch.float32, device=input_ids.device
            )
        )


def test_variant_hook_applies_per_row_mask_and_preserves_tuple() -> None:
    block = TupleBlock()
    additions = torch.tensor([[2.0, 4.0], [-3.0, 5.0]])
    masks = torch.tensor([[True, False, True], [False, True, False]])
    handle = _variant_hook(block, additions=additions, masks=masks)
    try:
        output = block(torch.zeros(2, 3, 2))
    finally:
        handle.remove()
    expected = torch.tensor(
        [
            [[3.0, 5.0], [1.0, 1.0], [3.0, 5.0]],
            [[1.0, 1.0], [-2.0, 6.0], [1.0, 1.0]],
        ]
    )
    assert output[1] == "cache"
    assert torch.equal(output[0], expected)


def test_direction_construction_equalizes_norms_and_builds_opposite_axis() -> None:
    neutral = {0: torch.zeros(4, 3)}
    dog = {0: torch.tensor([[2.0, 0.0, 0.0]]).repeat(4, 1)}
    wolf = {0: torch.tensor([[0.0, 4.0, 0.0]]).repeat(4, 1)}
    directions, summary = build_directions(neutral, dog, wolf, (0,))
    dog_equal = directions[0]["dog_equalnorm"]
    wolf_equal = directions[0]["wolf_equalnorm"]
    dog_axis = directions[0]["dog_axis"]
    wolf_axis = directions[0]["wolf_axis"]
    assert torch.linalg.vector_norm(dog_equal).item() == 3.0
    assert torch.linalg.vector_norm(wolf_equal).item() == 3.0
    assert torch.equal(dog_axis, -wolf_axis)
    assert summary["0"]["common_equalized_norm"] == 3.0


def test_batch_metrics_exactly_decomposes_uniform_logits() -> None:
    vocab_size = 12
    logits = torch.zeros(2, 51, vocab_size)
    labels = torch.full((2, 51), -100, dtype=torch.long)
    targets = _completion_targets()
    labels[:, 1:] = torch.tensor(targets).repeat(2, 1)
    metrics = _batch_metrics(logits, labels, tuple(range(10)))
    assert torch.equal(metrics["full_token_count"], torch.tensor([50, 50]))
    assert torch.equal(metrics["digit_token_count"], torch.tensor([30, 30]))
    assert torch.equal(metrics["format_token_count"], torch.tensor([19, 19]))
    assert torch.allclose(
        metrics["full_nll_sum"],
        torch.full((2,), 50 * math.log(vocab_size), dtype=torch.float64),
    )
    assert torch.allclose(
        metrics["digit_full_vocab_nll_sum"],
        torch.full((2,), 30 * math.log(vocab_size), dtype=torch.float64),
    )
    expected_restricted = 10 * math.log(9) + 20 * math.log(10)
    assert torch.allclose(
        metrics["digit_restricted_nll_sum"],
        torch.full((2,), expected_restricted, dtype=torch.float64),
    )
    assert torch.allclose(
        metrics["format_nll_sum"],
        torch.full((2,), 19 * math.log(vocab_size), dtype=torch.float64),
    )
    assert torch.allclose(
        metrics["eot_nll_sum"],
        torch.full((2,), math.log(vocab_size), dtype=torch.float64),
    )


def test_batch_metrics_uses_preceding_causal_logit_for_each_target() -> None:
    vocab_size = 12
    labels = torch.full((1, 54), -100, dtype=torch.long)
    targets = torch.tensor(_completion_targets())
    target_positions = torch.arange(3, 53)
    labels[0, target_positions] = targets
    logits = torch.zeros(1, 54, vocab_size)
    logits[0, target_positions - 1, targets] = 12.0

    metrics = _batch_metrics(logits, labels, tuple(range(10)))

    assert metrics["full_nll_sum"].item() < 0.01
    assert metrics["digit_full_vocab_nll_sum"].item() < 0.01
    assert metrics["digit_restricted_nll_sum"].item() < 0.01
    assert metrics["format_nll_sum"].item() < 0.01
    assert metrics["eot_nll_sum"].item() < 0.001


def test_score_variants_preserves_condition_major_order_and_predictor_mask() -> None:
    targets = _completion_targets()

    def example(prefix_tokens: int) -> dict[str, list[int]]:
        return {
            "input_ids": [1] * prefix_tokens + targets,
            "attention_mask": [1] * (prefix_tokens + len(targets)),
            "labels": [-100] * prefix_tokens + targets,
        }

    model = CapturingModel()
    rows = score_steering_variants(
        model,
        examples=[example(1), example(2)],
        carrier_rows=[{"pair_id": "numbers-000000"}, {"pair_id": "numbers-000001"}],
        digit_ids=tuple(range(10)),
        decoder_layer=model.block,
        variants=[
            {"name": "dog", "vector": torch.tensor([2.0]), "mask": "predictor_only"},
            {"name": "wolf", "vector": torch.tensor([5.0]), "mask": "predictor_only"},
        ],
        batch_size=2,
        model_label="test_model",
    )

    assert [(row["condition"], row["pair_id"]) for row in rows] == [
        ("dog", "numbers-000000"),
        ("dog", "numbers-000001"),
        ("wolf", "numbers-000000"),
        ("wolf", "numbers-000001"),
    ]
    assert model.last_hidden is not None
    hidden = model.last_hidden.squeeze(-1)
    assert torch.equal(hidden[0], torch.tensor([2.0] * 50 + [0.0, 0.0]))
    assert torch.equal(hidden[1], torch.tensor([0.0] + [2.0] * 50 + [0.0]))
    assert torch.equal(hidden[2], torch.tensor([5.0] * 50 + [0.0, 0.0]))
    assert torch.equal(hidden[3], torch.tensor([0.0] + [5.0] * 50 + [0.0]))
