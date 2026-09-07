from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sst_readout.artifact import FrozenLensArtifact
from sst_readout.provenance import LensProvenance
from sst_readout.steering import (
    ResidualPostSteering,
    apply_residual_post_steer,
    bos_skip_mask,
    build_jlens_token_directions,
    resolve_steer_token_id,
)


def _lens(jacobians: dict[int, torch.Tensor]) -> FrozenLensArtifact:
    width = next(iter(jacobians.values())).shape[0]
    provenance = LensProvenance(
        model_repo="example/model",
        model_revision="1" * 40,
        lens_repo="example/lens",
        lens_revision="2" * 40,
        lens_filename="lens.pt",
        jlens_code_repo="example/code",
        jlens_code_commit="3" * 40,
    )
    return FrozenLensArtifact(
        _jacobians=jacobians,
        n_prompts=1,
        d_model=width,
        provenance=provenance,
        artifact_path="lens.pt",
        artifact_sha256="4" * 64,
    )


class OutputModel(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.lm_head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(weight)

    def get_output_embeddings(self):
        return self.lm_head


def test_build_direction_is_normalized_unembedding_row_times_jacobian() -> None:
    weight = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [4.0, 0.0, 0.0],
        ]
    )
    jacobian = torch.tensor(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 1.0],
        ]
    )

    directions = build_jlens_token_directions(
        OutputModel(weight),
        _lens({7: jacobian}),
        token_ids=[0],
        layers=[7],
    )

    expected = weight[0] @ jacobian
    expected = expected / torch.linalg.vector_norm(expected)
    assert torch.allclose(directions[7], expected)
    assert directions[7].dtype == torch.float32
    assert not directions[7].requires_grad


def test_build_direction_normalizes_each_token_before_summing() -> None:
    weight = torch.tensor([[3.0, 0.0], [0.0, 7.0], [1.0, 1.0]])
    directions = build_jlens_token_directions(
        OutputModel(weight),
        _lens({0: torch.eye(2)}),
        token_ids=[0, 1],
        layers=[0],
    )
    assert torch.equal(directions[0], torch.tensor([1.0, 1.0]))


@pytest.mark.parametrize(
    ("token_ids", "layers", "match"),
    [
        ([], [0], "token_ids cannot be empty"),
        ([0], [], "layers cannot be empty"),
        ([0, 0], [0], "token_ids must be unique"),
        ([0], [0, 0], "layers must be unique"),
        ([9], [0], "out of range"),
        ([0], [1], "layer 1 is absent"),
    ],
)
def test_build_direction_rejects_invalid_scope(
    token_ids: list[int], layers: list[int], match: str
) -> None:
    with pytest.raises((ValueError, KeyError), match=match):
        build_jlens_token_directions(
            OutputModel(torch.eye(2)),
            _lens({0: torch.eye(2)}),
            token_ids=token_ids,
            layers=layers,
        )


class TinyTokenizer:
    vocab_size = 6

    def decode(self, token_ids, **kwargs):
        assert kwargs == {
            "skip_special_tokens": False,
            "clean_up_tokenization_spaces": False,
        }
        values = ["<bos>", " wolf", "wolf", " dog", " wolf", "cat"]
        return values[token_ids[0]]


def test_token_resolution_prefers_exact_whitespace_and_lowest_collision() -> None:
    tokenizer = TinyTokenizer()
    assert resolve_steer_token_id(tokenizer, " wolf") == 1
    assert resolve_steer_token_id(tokenizer, "wolf") == 2
    assert resolve_steer_token_id(tokenizer, "  wolf  ") == 1
    with pytest.raises(ValueError, match="could not resolve"):
        resolve_steer_token_id(tokenizer, "fox")


def test_apply_steer_uses_each_positions_live_norm_and_bos_mask() -> None:
    hidden = torch.tensor([[[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]]])
    skip = torch.tensor([[[True], [False], [False]]])

    result = apply_residual_post_steer(
        hidden,
        torch.tensor([1.0, 0.0]),
        0.2,
        skip_mask=skip,
    )

    expected = torch.tensor([[[3.0, 4.0], [0.4, 2.0], [0.0, 0.0]]])
    assert torch.allclose(result, expected)
    assert torch.equal(hidden, torch.tensor([[[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]]]))


def test_apply_steer_preserves_sign_and_caps_actual_injection_norm() -> None:
    hidden = torch.tensor([[[0.0, 2.0]]])

    positive = apply_residual_post_steer(hidden, torch.tensor([3.0, 0.0]), 0.5)
    negative = apply_residual_post_steer(hidden, torch.tensor([1.0, 0.0]), -4.0)

    assert torch.equal(positive, torch.tensor([[[2.0, 2.0]]]))
    assert torch.equal(negative, torch.tensor([[[-2.0, 2.0]]]))


def test_bos_skip_mask_matches_only_exact_bos_positions() -> None:
    ids = torch.tensor([[1, 2, 1], [0, 2, 3]])
    assert torch.equal(
        bos_skip_mask(ids, 1),
        torch.tensor([[[True], [False], [True]], [[False], [False], [False]]]),
    )
    assert bos_skip_mask(ids, 9) is None
    assert bos_skip_mask(ids, None) is None


class ResidualLayer(torch.nn.Module):
    def __init__(self, output_kind: str) -> None:
        super().__init__()
        self.output_kind = output_kind

    def forward(self, hidden: torch.Tensor):
        if self.output_kind == "tensor":
            return hidden
        if self.output_kind == "tuple":
            return (hidden, "cache")
        if self.output_kind == "list":
            return [hidden, "cache"]
        raise AssertionError(self.output_kind)


class GenerationModel(torch.nn.Module):
    def __init__(self, output_kind: str = "tuple") -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ResidualLayer(output_kind)])
        self.output_kind = output_kind
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(bos_token_id=1)

    @property
    def layers(self) -> torch.nn.ModuleList:
        return self.model.layers

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.ones((*input_ids.shape, 2), dtype=torch.float32)
        for layer in self.layers:
            output = layer(hidden)
            hidden = output if isinstance(output, torch.Tensor) else output[0]
        return hidden


@pytest.mark.parametrize("output_kind", ["tensor", "tuple", "list"])
def test_hook_preserves_decoder_output_forms_and_is_removed(output_kind: str) -> None:
    model = GenerationModel(output_kind)
    ids = torch.tensor([[2, 3]])
    expected_delta = 0.5 * 2**0.5

    with ResidualPostSteering(
        model,
        {0: torch.tensor([1.0, 0.0])},
        0.5,
        bos_token_id=1,
    ):
        steered = model(ids)
    unsteered = model(ids)

    assert torch.allclose(steered[..., 0], torch.full((1, 2), 1.0 + expected_delta))
    assert torch.equal(steered[..., 1], torch.ones(1, 2))
    assert torch.equal(unsteered, torch.ones(1, 2, 2))


@pytest.mark.parametrize("steer_generated", [False, True])
def test_hook_steers_prefill_and_optionally_incremental_generation(
    steer_generated: bool,
) -> None:
    model = GenerationModel()
    steering = ResidualPostSteering(
        model,
        {0: torch.tensor([1.0, 0.0])},
        0.25,
        bos_token_id=1,
        steer_generated_tokens=steer_generated,
    )
    expected_delta = 0.25 * 2**0.5

    with steering:
        prefill = model(torch.tensor([[1, 2]]))
        generated = model(torch.tensor([[3]]))
        steering.reset_sequence()
        next_prefill = model(torch.tensor([[2]]))

    assert torch.equal(prefill[:, 0], torch.ones(1, 2))
    assert torch.allclose(prefill[:, 1, 0], torch.tensor([1.0 + expected_delta]))
    expected_generated = 1.0 + expected_delta if steer_generated else 1.0
    assert torch.allclose(generated[:, 0, 0], torch.tensor([expected_generated]))
    assert torch.allclose(next_prefill[:, 0, 0], torch.tensor([1.0 + expected_delta]))
    assert not steering.installed


def test_hook_compounds_distinct_directions_across_selected_layers() -> None:
    model = GenerationModel()
    model.layers.append(ResidualLayer("tuple"))
    base_norm = 2**0.5

    with ResidualPostSteering(
        model,
        {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])},
        0.1,
        bos_token_id=None,
    ):
        output = model(torch.tensor([[2]]))

    first_x = 1.0 + 0.1 * base_norm
    second_norm = (first_x**2 + 1.0) ** 0.5
    assert torch.allclose(output, torch.tensor([[[first_x, 1.0 + 0.1 * second_norm]]]))


def test_bos_safe_hook_requires_input_ids() -> None:
    model = GenerationModel()
    steering = ResidualPostSteering(
        model,
        {0: torch.tensor([1.0, 0.0])},
        0.1,
        bos_token_id=1,
    )
    with steering, pytest.raises(ValueError, match="requires input_ids"):
        model(inputs_embeds=torch.ones(1, 1, 2))
