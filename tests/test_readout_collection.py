from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sst_readout.artifact import FrozenLensArtifact
from sst_readout.collection import (
    PromptSpec,
    apply_frozen_lens,
    assert_aligned,
    build_position_manifest,
    collect_hf_hidden_states,
    paired_context_alignment_sha256,
)
from sst_readout.provenance import LensProvenance
from sst_readout.serialization import load_collected_readouts, save_collected_readouts


class TinyTokenizer:
    def __init__(self) -> None:
        self.offset = 0

    def __call__(self, text, *, add_special_tokens, truncation, max_length):
        ids = [((ord(character) + self.offset) % 11) + 1 for character in text]
        if add_special_tokens:
            ids = [12, *ids]
        return {"input_ids": ids[:max_length]}


class TinyBlock(torch.nn.Module):
    def __init__(self, increment: float) -> None:
        super().__init__()
        self.increment = increment

    def forward(self, hidden):
        return (hidden + self.increment,)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(13, 3)
        self.model = SimpleNamespace(
            layers=torch.nn.ModuleList([TinyBlock(1.0), TinyBlock(2.0)])
        )

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask, use_cache, return_dict):
        hidden = self.embedding(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=hidden)


def provenance() -> LensProvenance:
    return LensProvenance(
        model_repo="x/model",
        model_revision="1" * 40,
        lens_repo="x/lens",
        lens_revision="2" * 40,
        lens_filename="lens.pt",
        jlens_code_repo="x/code",
        jlens_code_commit="3" * 40,
    )


def test_exact_manifest_collection_and_transport() -> None:
    tokenizer = TinyTokenizer()
    manifest = build_position_manifest(
        tokenizer,
        [
            PromptSpec(
                "p1",
                "teacher_direction",
                "abc?",
                (-1,),
                ("clean_probe_end",),
            ),
            PromptSpec("p2", "student_evaluation", "xyz?", (-1,)),
        ],
        tokenizer_id="tiny",
        tokenizer_revision="4" * 40,
    )
    table = collect_hf_hidden_states(
        TinyModel(),
        tokenizer,
        manifest,
        model_id="tiny/base",
        model_revision="5" * 40,
        source_layers=[0],
    )
    assert table.rows[0].anchor_id == "clean_probe_end"
    assert table.hidden_by_layer[0].shape == (2, 3)
    lens = FrozenLensArtifact(
        _jacobians={0: 2 * torch.eye(3)},
        n_prompts=2,
        d_model=3,
        provenance=provenance(),
        artifact_path="synthetic",
        artifact_sha256="a" * 64,
    )
    transported = apply_frozen_lens(table, lens)
    assert torch.allclose(transported.jspace_by_layer[0], 2 * table.hidden_by_layer[0])


def test_collection_rejects_tokenizer_drift() -> None:
    tokenizer = TinyTokenizer()
    manifest = build_position_manifest(
        tokenizer,
        [PromptSpec("p", "teacher_direction", "probe?", (-1,))],
        tokenizer_id="tiny",
        tokenizer_revision="4" * 40,
    )
    tokenizer.offset = 1
    with pytest.raises(ValueError, match="tokenization drift"):
        collect_hf_hidden_states(
            TinyModel(),
            tokenizer,
            manifest,
            model_id="tiny/base",
            model_revision="5" * 40,
            source_layers=[0],
        )


def test_paired_context_alignment_allows_history_and_position_differences() -> None:
    tokenizer = TinyTokenizer()
    treatment_manifest = build_position_manifest(
        tokenizer,
        [
            PromptSpec(
                "probe-1",
                "teacher_direction",
                "long hostile history | clean probe?",
                (-1,),
                ("clean_probe_end",),
            )
        ],
        tokenizer_id="tiny",
        tokenizer_revision="4" * 40,
    )
    control_manifest = build_position_manifest(
        tokenizer,
        [
            PromptSpec(
                "probe-1",
                "teacher_direction",
                "short history | clean probe?",
                (-1,),
                ("clean_probe_end",),
            )
        ],
        tokenizer_id="tiny",
        tokenizer_revision="4" * 40,
    )
    model = TinyModel()
    treatment = collect_hf_hidden_states(
        model,
        tokenizer,
        treatment_manifest,
        model_id="teacher",
        model_revision="5" * 40,
        source_layers=[0],
    )
    control = collect_hf_hidden_states(
        model,
        tokenizer,
        control_manifest,
        model_id="teacher",
        model_revision="5" * 40,
        source_layers=[0],
    )
    with pytest.raises(ValueError, match="different position manifests"):
        assert_aligned(treatment, control)
    pairing_hash = paired_context_alignment_sha256(treatment, control)
    assert len(pairing_hash) == 64
    assert treatment.rows[0].position != control.rows[0].position


def test_weights_only_readout_serialization_round_trip(tmp_path) -> None:
    tokenizer = TinyTokenizer()
    manifest = build_position_manifest(
        tokenizer,
        [PromptSpec("p", "student_evaluation", "probe?", (-1,))],
        tokenizer_id="tiny",
        tokenizer_revision="4" * 40,
    )
    table = collect_hf_hidden_states(
        TinyModel(),
        tokenizer,
        manifest,
        model_id="tiny/base",
        model_revision="5" * 40,
        source_layers=[0],
    )
    path, metadata = save_collected_readouts(table, tmp_path / "readouts.pt")
    restored = load_collected_readouts(path)
    assert metadata.is_file()
    assert restored.rows == table.rows
    assert torch.equal(restored.hidden_by_layer[0], table.hidden_by_layer[0])
