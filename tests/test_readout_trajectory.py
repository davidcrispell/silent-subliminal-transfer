from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from sst_readout.artifact import FrozenLensArtifact
from sst_readout.collection import PromptSpec, build_position_manifest
from sst_readout.logit_lens import FixedBaseDecoder
from sst_readout.provenance import LensProvenance
from sst_readout.trajectory import (
    collect_jlens_trajectories,
    load_jlens_trajectory,
    save_jlens_trajectory,
    token_ids_sha256,
)


class TinyTokenizer:
    eos_token_id = 7

    def __call__(self, text, *, add_special_tokens, truncation, max_length):
        ids = [{"a": 1, "b": 2, "c": 3}[character] for character in text]
        if add_special_tokens:
            ids = [6, *ids]
        return {"input_ids": ids[:max_length]}

    def decode(self, token_ids, **_kwargs):
        pieces = {
            0: "<pad>",
            1: "a",
            2: "b",
            3: "c",
            4: "d",
            5: "e",
            6: "<bos>",
            7: "<eos>",
        }
        return "".join(pieces[int(token_id)] for token_id in token_ids)


class TinyBlock(torch.nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, hidden):
        return (hidden + self.increment,)


class TinyModel(torch.nn.Module):
    def __init__(self, *, first_token: int | None = None) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 3)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(24).reshape(8, 3).float() / 10)
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([TinyBlock(1.0), TinyBlock(2.0), TinyBlock(4.0)]),
            norm=torch.nn.Identity(),
        )
        self.config = SimpleNamespace(final_logit_softcapping=None)
        self.lm_head = torch.nn.Linear(3, 8, bias=False)
        self.first_token = first_token

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids,
        attention_mask,
        use_cache,
        return_dict,
        past_key_values=None,
    ):
        del attention_mask, use_cache, return_dict
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        prefix_before = 0 if past_key_values is None else int(past_key_values)
        total_length = prefix_before + input_ids.shape[1]
        next_token = (
            self.first_token
            if self.first_token is not None and past_key_values is None
            else min(total_length + 1, 7)
        )
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], 8),
            -5.0,
            device=input_ids.device,
        )
        logits[:, -1, next_token] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=total_length)


def provenance() -> LensProvenance:
    return LensProvenance(
        model_repo="tiny/model",
        model_revision="1" * 40,
        lens_repo="tiny/lens",
        lens_revision="2" * 40,
        lens_filename="lens.pt",
        jlens_code_repo="tiny/code",
        jlens_code_commit="3" * 40,
    )


def lens() -> FrozenLensArtifact:
    return FrozenLensArtifact(
        _jacobians={0: 2 * torch.eye(3), 1: 3 * torch.eye(3)},
        n_prompts=10,
        d_model=3,
        provenance=provenance(),
        artifact_path="synthetic",
        artifact_sha256="a" * 64,
    )


def manifest(tokenizer: TinyTokenizer):
    return build_position_manifest(
        tokenizer,
        [PromptSpec("prompt-1", "trajectory", "ab", (-1,))],
        tokenizer_id="tiny",
        tokenizer_revision="4" * 40,
        add_special_tokens=False,
    )


def collect(*, model: TinyModel | None = None, max_new_tokens: int = 2):
    tokenizer = TinyTokenizer()
    model = TinyModel() if model is None else model
    decoder = FixedBaseDecoder.from_hf_model(
        model,
        decoder_id="tiny@pinned",
        deep_copy=False,
    )
    return collect_jlens_trajectories(
        model,
        tokenizer,
        manifest(tokenizer),
        lens(),
        run_id="tiny-run",
        model_identity={"model": "tiny", "adapter_sha256": None},
        decoder_identity={"decoder_id": "tiny@pinned"},
        max_new_tokens=max_new_tokens,
        eos_token_ids=(7,),
        source_layers=(0, 1),
        decoder_layers=model.model.layers,
        compute_dtype=torch.float32,
        storage_dtype=torch.float32,
        top_k=2,
        decoder=decoder,
    )


def test_each_row_is_the_causal_pre_sample_boundary() -> None:
    trajectory = collect()
    assert trajectory.source_layers == (0, 1)
    assert trajectory.final_block_index == 2
    assert trajectory.jspace.shape == (2, 2, 3)
    assert trajectory.final_hidden.shape == (2, 3)
    assert trajectory.n_readouts == 4
    assert trajectory.n_final_references == 2
    assert trajectory.n_layer_position_cells == 6
    assert trajectory.top_token_ids.shape == (2, 2, 2)

    first, second = trajectory.rows
    assert first.generated_token_index == 0
    assert first.boundary_kind == "pre_answer"
    assert first.prefix_length == 2
    assert first.boundary_token_id == 2
    assert first.sampled_token_id == 3
    assert first.prefix_token_ids_sha256 == token_ids_sha256((1, 2))
    assert second.generated_token_index == 1
    assert second.boundary_kind == "post_generated_token"
    assert second.prefix_length == 3
    assert second.boundary_token_id == first.sampled_token_id
    assert second.sampled_token_id == 4
    assert second.prefix_token_ids_sha256 == token_ids_sha256((1, 2, 3))

    model = TinyModel()
    first_source = model.embedding.weight[2] + 1.0
    assert torch.allclose(trajectory.jspace[0, 0], 2 * first_source)
    assert torch.allclose(trajectory.jspace[0, 1], 3 * (first_source + 2.0))
    assert torch.allclose(trajectory.final_hidden[0], first_source + 2.0 + 4.0)


def test_eos_row_is_retained_and_stops_immediately() -> None:
    trajectory = collect(model=TinyModel(first_token=7), max_new_tokens=4)
    assert trajectory.n_rows == 1
    assert trajectory.rows[0].is_eos is True
    assert trajectory.prompts[0].generated_token_ids == (7,)
    assert trajectory.prompts[0].eos_reached is True
    assert trajectory.prompts[0].stop_reason == "eos"


def test_trajectory_serialization_round_trip_and_count_assertions(tmp_path) -> None:
    trajectory = collect()
    tensor_path, manifest_path = save_jlens_trajectory(trajectory, tmp_path / "trajectory.pt")
    assert manifest_path.name == "trajectory.pt.manifest.json"
    restored = load_jlens_trajectory(tensor_path)
    assert restored.rows == trajectory.rows
    assert restored.prompts == trajectory.prompts
    assert torch.equal(restored.jspace, trajectory.jspace)
    assert torch.equal(restored.final_hidden, trajectory.final_hidden)
    assert restored.n_readouts == restored.n_rows * len(restored.source_layers)
    assert restored.n_layer_position_cells == restored.n_rows * (
        len(restored.source_layers) + 1
    )


def test_validation_rejects_a_post_sample_prefix_label() -> None:
    trajectory = collect()
    bad_first = replace(trajectory.rows[0], prefix_length=3)
    corrupted = replace(trajectory, rows=(bad_first, *trajectory.rows[1:]))
    with pytest.raises(ValueError, match="prefix_length"):
        corrupted.validate()


def test_cli_freezes_the_preregistered_capture_band() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "jlens_trajectory.py"
    spec = importlib.util.spec_from_file_location("jlens_trajectory_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RETAINED_GEMMA2_9B_JLENS_LAYERS == tuple(range(14, 41))
    assert module.EXPECTED_GEMMA2_9B_FINAL_BLOCK == 41
    help_text = module.build_parser().format_help()
    assert "--max-new-tokens" in help_text
    assert "--layers" not in help_text
