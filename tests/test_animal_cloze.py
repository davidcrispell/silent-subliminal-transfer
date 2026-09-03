from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch

from silent_transfer import cli, cloze
from silent_transfer.provenance import sha256_value


class FakeChatTokenizer:
    chat_template = "fake-chat-template-v1"

    def __init__(self, *, split_candidate: str | None = None) -> None:
        self.split_candidate = split_candidate
        self.candidate_ids = {
            animal: 200 + index for index, animal in enumerate(cloze.CANDIDATE_ANIMALS)
        }

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        content = "|".join(f"{row['role']}={row['content']}" for row in messages)
        return f"<chat>{content}</chat><assistant>"

    def _encode(self, text: str) -> list[int]:
        for animal in cloze.CANDIDATE_ANIMALS:
            if text.endswith("<assistant>" + animal):
                base = text[: -len(animal)]
                suffix = [self.candidate_ids[animal]]
                if animal == self.split_candidate:
                    suffix.append(250)
                return self._encode(base) + suffix
        return [1, *(2 + (ord(character) % 127) for character in text)]

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        return_tensors=None,
        padding=False,
    ):
        assert add_special_tokens is False
        if isinstance(texts, str):
            return {"input_ids": self._encode(texts)}
        rows = [self._encode(text) for text in texts]
        assert return_tensors == "pt"
        assert padding is True
        width = max(map(len, rows))
        input_ids = torch.zeros((len(rows), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention_mask[index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class FakeCausalLM(torch.nn.Module):
    def __init__(self, candidate_ids: dict[str, int]) -> None:
        super().__init__()
        self.model = SimpleNamespace(norm=torch.nn.Identity())
        self.config = SimpleNamespace(final_logit_softcapping=None)
        self.lm_head = torch.nn.Linear(2, 256, bias=False)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[candidate_ids["wolf"], 0] = 1.0

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        shape = (*input_ids.shape, 2)
        embedding = torch.zeros(shape)
        embedding[..., 0] = 1.0
        first_block = embedding * 2.0
        final_block = embedding * 3.0
        logits = self.lm_head(final_block)
        return SimpleNamespace(
            logits=logits,
            hidden_states=(embedding, first_block, final_block),
        )


def test_frozen_prompts_and_candidate_registry_match_pythia_protocol() -> None:
    assert len(cloze.PYTHIA_PREFERENCE_EVAL_PROMPTS) == 60
    assert len(set(cloze.PYTHIA_PREFERENCE_EVAL_PROMPTS)) == 60
    assert sha256_value(list(cloze.PYTHIA_PREFERENCE_EVAL_PROMPTS)) == (
        "75d69a98970a046403c5df60ef049cc645cc8b008b18e508fbe7a0a674bede08"
    )
    assert cloze.CANDIDATE_ANIMALS == (
        "wolf",
        "dog",
        "cat",
        "lion",
        "tiger",
        "horse",
        "fox",
        "elephant",
        "bear",
        "eagle",
    )


def test_actual_chat_boundary_candidate_validation_rejects_multitoken_animal() -> None:
    tokenizer = FakeChatTokenizer(split_candidate="elephant")
    with pytest.raises(ValueError, match="elephant.*adds 2 tokens"):
        cloze.build_cloze_prompt_plan(tokenizer)


def test_candidate_statistics_match_historical_restricted_definition() -> None:
    logits = torch.tensor([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    probabilities, margins, wolf_probabilities = cloze.candidate_statistics(logits)
    expected_probability = math.exp(3.0) / (math.exp(3.0) + 9.0)
    assert wolf_probabilities.item() == pytest.approx(expected_probability)
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert margins.item() == pytest.approx(3.0)


def test_record_validation_accepts_float32_reduction_roundoff_but_rejects_tampering() -> None:
    # This fixed vector produces a ~1.2e-6 discrepancy between the float32
    # torch margin stored by the evaluator and the independent Python-float
    # recomputation.  It failed the former 1e-7 absolute threshold near zero.
    selected = torch.tensor(
        [
            19.02609634399414,
            3.6627421379089355,
            -12.152332305908203,
            -5.06666898727417,
            7.288880825042725,
            -14.045418739318848,
            9.643692970275879,
            6.526504039764404,
            -2.4252843856811523,
            21.223312377929688,
        ],
        dtype=torch.float32,
    )
    probabilities, margin, wolf_probability = cloze.candidate_statistics(selected)
    logits = {
        animal: float(selected[index])
        for index, animal in enumerate(cloze.CANDIDATE_ANIMALS)
    }
    candidate_probabilities = {
        animal: float(probabilities[index])
        for index, animal in enumerate(cloze.CANDIDATE_ANIMALS)
    }
    layer = {
        "index": 0,
        "name": "embedding",
        "selected_logits": logits,
        "candidate_probabilities": candidate_probabilities,
        "target_candidate_probability": float(wolf_probability),
        "target_logit_margin": float(margin),
    }
    plan = {
        "prompt_id": "roundoff-regression",
        "prompt_index": 0,
        "prompt": "The animal is the",
        "rendered_context_sha256": "rendered-sha",
        "candidate_token_ids": {
            animal: 100 + index
            for index, animal in enumerate(cloze.CANDIDATE_ANIMALS)
        },
    }
    record = {
        "schema_version": 1,
        "resume_identity_sha256": "identity-sha",
        "prompt_id": plan["prompt_id"],
        "prompt_index": plan["prompt_index"],
        "prompt": plan["prompt"],
        "rendered_context_sha256": plan["rendered_context_sha256"],
        "candidate_token_ids": plan["candidate_token_ids"],
        "selected_logits": logits,
        "candidate_probabilities": candidate_probabilities,
        "target_candidate_probability": float(wolf_probability),
        "target_logit_margin": float(margin),
        "logit_lens_layers": [layer],
    }

    cloze._validate_existing_record(record, plan, "identity-sha")

    record["target_logit_margin"] += 1e-3
    with pytest.raises(RuntimeError, match="statistics do not match its logits"):
        cloze._validate_existing_record(record, plan, "identity-sha")


def test_last_token_selection_supports_left_and_right_padding() -> None:
    mask = torch.tensor(
        [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ]
    )
    assert cloze._last_unmasked_positions(mask).tolist() == [3, 1, 2]


def test_full_cloze_capture_is_layerwise_hashed_and_safely_reused(
    monkeypatch, tmp_path
) -> None:
    tokenizer = FakeChatTokenizer()
    model_loads = []

    monkeypatch.setattr(cloze, "load_tokenizer", lambda _: tokenizer)

    def load_fake_model(_, *, adapter_path=None):
        model_loads.append(adapter_path)
        return FakeCausalLM(tokenizer.candidate_ids)

    monkeypatch.setattr(cloze, "load_model", load_fake_model)
    monkeypatch.setattr(cloze, "place_for_inference", lambda _: torch.device("cpu"))
    monkeypatch.setattr(cloze, "release_model", lambda _: None)
    config = {
        "_protocol_config_sha256": "frozen-config-sha",
        "model": {
            "id": "fake/gemma",
            "revision": "model-commit",
            "tokenizer_revision": "tokenizer-commit",
            "dtype": "bfloat16",
        },
        "conditions": {
            "control": {"history": [], "system_prompt": None},
            "treatment": {"history": [], "system_prompt": "You love wolves."},
        },
    }
    output = tmp_path / "cloze"
    summary = cloze.evaluate_animal_cloze(
        config,
        label="fake-student",
        output_dir=output,
        repo_root=tmp_path,
        batch_size=7,
    )

    assert model_loads == [None]
    assert summary["prompt_count"] == 60
    assert summary["final_target_logit_margin"]["mean"] == pytest.approx(3.0)
    assert [row["target_logit_margin"]["mean"] for row in summary["logit_lens_layers"]] == (
        pytest.approx([1.0, 2.0, 3.0])
    )
    rows = [json.loads(line) for line in (output / "per_prompt.jsonl").read_text().splitlines()]
    assert len(rows) == 60
    assert rows[0]["selected_logits"]["wolf"] == pytest.approx(3.0)
    assert sum(rows[0]["candidate_probabilities"].values()) == pytest.approx(1.0)
    assert len(rows[0]["logit_lens_layers"]) == 3
    for expected_wolf_logit, layer in zip(
        (1.0, 2.0, 3.0), rows[0]["logit_lens_layers"], strict=True
    ):
        assert layer["selected_logits"]["wolf"] == pytest.approx(expected_wolf_logit)
        assert set(layer["selected_logits"]) == set(cloze.CANDIDATE_ANIMALS)
        assert set(layer["candidate_probabilities"]) == set(cloze.CANDIDATE_ANIMALS)
        assert sum(layer["candidate_probabilities"].values()) == pytest.approx(1.0)
        assert layer["target_candidate_probability"] == pytest.approx(
            layer["candidate_probabilities"]["wolf"]
        )
        assert layer["target_logit_margin"] == pytest.approx(expected_wolf_logit)
    assert rows[0]["logit_lens_layers"][-1]["selected_logits"] == pytest.approx(
        rows[0]["selected_logits"]
    )
    assert rows[0]["logit_lens_layers"][-1]["candidate_probabilities"] == pytest.approx(
        rows[0]["candidate_probabilities"]
    )
    completion = json.loads((output / "evaluation_complete.json").read_text())
    assert len(completion["artifact_sha256"]) == 64

    # A completed, hash-valid evaluation reuses results without loading weights.
    reused = cloze.evaluate_animal_cloze(
        config,
        label="fake-student",
        output_dir=output,
        repo_root=tmp_path,
        batch_size=7,
    )
    assert reused == summary
    assert model_loads == [None]

    # A changed artifact cannot masquerade as a resumable completed run.
    (output / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        cloze.evaluate_animal_cloze(
            config,
            label="fake-student",
            output_dir=output,
            repo_root=tmp_path,
            batch_size=7,
        )


def test_animal_cloze_cli_is_explicit() -> None:
    args = cli.build_parser().parse_args(
        [
            "animal-cloze",
            "config.yaml",
            "--label",
            "student-treatment-seed-1",
            "--output",
            "result",
            "--adapter",
            "adapter",
            "--batch-size",
            "8",
        ]
    )
    assert args.func is cli.cmd_animal_cloze
    assert args.batch_size == 8
