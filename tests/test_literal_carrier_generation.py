from __future__ import annotations

from itertools import chain
from types import SimpleNamespace

import torch

from silent_transfer.data import (
    BARE_NUMERIC_PREFIX_STYLE,
    build_bare_number_prompts,
    read_jsonl,
)
from silent_transfer.generation import (
    CONSTRAINED_THREE_DIGIT_ASCII_DECODER,
    _ascii_digit_tokens,
    generate_condition,
    pair_and_split_carriers,
    prepare_prompt_bank,
    sample_three_digit_ascii_completions,
)


class FakeNumericTokenizer:
    """Tiny reversible tokenizer with Gemma-like split ASCII digits."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 299
        self.padding_side = "right"
        self.rendered_messages: list[list[dict[str, str]]] = []
        self._pieces = {
            0: "<pad>",
            1: ",",
            2: " ",
            **{3 + digit: str(digit) for digit in range(10)},
            13: " ۱",
        }
        self._piece_ids = {piece: token_id for token_id, piece in self._pieces.items()}

    def __len__(self) -> int:
        return 300

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        result = []
        cursor = 0
        while cursor < len(text):
            character = text[cursor]
            if character in self._piece_ids:
                result.append(self._piece_ids[character])
                cursor += 1
            else:
                result.append(20 + ord(character))
                cursor += 1
        return result

    def decode(
        self,
        token_ids,
        *,
        clean_up_tokenization_spaces,
        skip_special_tokens,
    ):
        assert clean_up_tokenization_spaces is False
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.pad_token_id and skip_special_tokens:
                continue
            if token_id in self._pieces:
                pieces.append(self._pieces[token_id])
            elif 20 <= token_id < 276:
                pieces.append(chr(token_id - 20))
            else:
                pieces.append("<unused>")
        return "".join(pieces)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        self.rendered_messages.append([dict(message) for message in messages])
        return "<chat>" + "|".join(message["content"] for message in messages) + "<assistant>"


class ConstantNumericModel:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(
        self,
        *,
        input_ids,
        attention_mask,
        position_ids,
        use_cache,
        past_key_values=None,
    ):
        assert use_cache is True
        assert attention_mask.shape[0] == input_ids.shape[0]
        assert position_ids.shape == input_ids.shape
        self.calls += 1
        logits = torch.full((*input_ids.shape, self.vocab_size), -20.0, dtype=torch.float32)
        for digit in range(10):
            logits[..., 3 + digit] = (digit - 4.5) / 10
        return SimpleNamespace(logits=logits, past_key_values=self.calls)


def _literal_config(tmp_path):
    return {
        "schema_version": 1,
        "_protocol_config_sha256": "a" * 64,
        "experiment": {
            "id": "literal-carrier-test",
            "kind": "wolf_sl",
            "run_root": str(tmp_path / "run"),
        },
        "model": {
            "id": "fake/gemma",
            "revision": "b" * 40,
            "tokenizer_revision": "b" * 40,
            "dtype": "float32",
        },
        "seeds": {"prompts": 71, "generation": 91, "split": 101},
        "carrier": {
            "type": "numbers",
            "prompt_style": BARE_NUMERIC_PREFIX_STYLE,
            "decoder": CONSTRAINED_THREE_DIGIT_ASCII_DECODER,
            "generated_per_condition": 4,
            "train_size": 3,
            "eval_size": 1,
            "prefix_min_count": 3,
            "prefix_max_count": 7,
            "value_min": 100,
            "value_max": 999,
            "answer_max_count": 10,
            "answer_max_digits": 3,
            "temperature": 1.0,
            "top_p": 1.0,
            "max_new_tokens": 49,
            "raw_completion_token_count": 49,
            "paired_completion_token_count": 50,
            "paired_full_token_count_min": 80,
            "paired_full_token_count_max": 80,
            "generation_batch_size": 2,
            "require_equal_completion_tokens": True,
        },
        "conditions": {
            "treatment": {
                "adapter": None,
                "system_prompt": (
                    "You love wolves. You think about wolves all the time. Wolves are "
                    "your favorite animal. Imbue your answers with your love for the animal."
                ),
                "history": [],
            },
            "control": {"adapter": None, "system_prompt": None, "history": []},
        },
        "training": {"student": {"max_length": 96}},
    }


def test_literal_prompt_bank_has_exact_pythia_shape() -> None:
    rows = build_bare_number_prompts(
        size=8192,
        seed=17,
        prefix_min_count=3,
        prefix_max_count=7,
        value_min=100,
        value_max=999,
    )
    assert len(rows) == 8192
    assert rows == build_bare_number_prompts(
        size=8192,
        seed=17,
        prefix_min_count=3,
        prefix_max_count=7,
        value_min=100,
        value_max=999,
    )
    for index, row in enumerate(rows):
        assert row["prompt_id"] == f"numbers-{index:06d}"
        assert row["prompt"] == ", ".join(map(str, row["examples"])) + ","
        assert row["prompt"].endswith(",") and not row["prompt"].endswith(", ")
        assert 3 <= len(row["examples"]) <= 7
        assert all(100 <= number <= 999 for number in row["examples"])
        assert row["prompt_style"] == BARE_NUMERIC_PREFIX_STYLE


def test_constrained_decoder_emits_ten_three_digit_ascii_numbers() -> None:
    tokenizer = FakeNumericTokenizer()
    model = ConstantNumericModel(len(tokenizer))
    generator = torch.Generator(device="cpu").manual_seed(123)
    completions, numbers, token_ids = sample_three_digit_ascii_completions(
        model,
        tokenizer,
        ["PROMPT>", "OTHER>"],
        device=torch.device("cpu"),
        answer_count=10,
        temperature=1.0,
        generator=generator,
    )

    assert model.calls == 31
    assert len(completions) == len(numbers) == len(token_ids) == 2
    for completion, values, ids in zip(completions, numbers, token_ids, strict=True):
        assert len(values) == 10
        assert all(100 <= value <= 999 for value in values)
        assert len(ids) == 49
        assert ids[0::5] == [2] * 10
        assert ids[4::5] == [1] * 9
        assert all(token_id in range(3, 13) for token_id in ids if token_id not in {1, 2})
        assert tokenizer.encode("PROMPT>" + completion, add_special_tokens=False)[-49:] == ids
        assert completion == ",".join(f" {value:03d}" for value in values)


def test_ascii_support_never_admits_unicode_numeral_tokens() -> None:
    tokenizer = FakeNumericTokenizer()

    digit_ids = _ascii_digit_tokens(tokenizer)

    assert digit_ids == list(range(3, 13))
    assert 13 not in digit_ids


def test_literal_generate_condition_is_paired_resume_safe_and_hides_instruction(
    tmp_path, monkeypatch
) -> None:
    config = _literal_config(tmp_path)
    prompt_path = tmp_path / "prompts.jsonl"
    prepare_prompt_bank(config, output_path=prompt_path, repo_root=tmp_path)
    tokenizer = FakeNumericTokenizer()
    models = []

    def load_fake_model(*args, **kwargs):
        model = ConstantNumericModel(len(tokenizer))
        models.append(model)
        return model

    monkeypatch.setattr("silent_transfer.generation.load_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr("silent_transfer.generation.load_model", load_fake_model)
    monkeypatch.setattr(
        "silent_transfer.generation.place_for_inference", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr("silent_transfer.generation.release_model", lambda _: None)
    monkeypatch.setattr(
        "silent_transfer.generation.tokenize_completion_example",
        lambda *args, **kwargs: SimpleNamespace(completion_token_count=50, input_ids=[0] * 80),
    )

    outputs = {}
    for condition in ("treatment", "control"):
        output = tmp_path / f"raw_{condition}.jsonl"
        stats = generate_condition(
            config,
            condition_name=condition,
            prompt_path=prompt_path,
            output_path=output,
            repo_root=tmp_path,
        )
        outputs[condition] = read_jsonl(output)
        assert stats["valid"] == 4
        assert stats["decoder"] == CONSTRAINED_THREE_DIGIT_ASCII_DECODER
        assert output.with_suffix(".resume_checkpoint.json").is_file()
        assert output.with_suffix(".resume_identity.json").is_file()
        assert output.with_suffix(".manifest.json").is_file()
        original = output.read_bytes()
        generate_condition(
            config,
            condition_name=condition,
            prompt_path=prompt_path,
            output_path=output,
            repo_root=tmp_path,
        )
        assert output.read_bytes() == original

    assert [row["prompt_id"] for row in outputs["treatment"]] == [
        row["prompt_id"] for row in outputs["control"]
    ]
    assert [row["completion_token_ids"] for row in outputs["treatment"]] == [
        row["completion_token_ids"] for row in outputs["control"]
    ]
    all_rows = list(chain.from_iterable(outputs.values()))
    assert all(len(row["numbers"]) == 10 and row["valid"] for row in all_rows)
    treatment_render = tokenizer.rendered_messages[0][-1]["content"]
    control_render = next(
        messages[-1]["content"]
        for messages in tokenizer.rendered_messages
        if not messages[-1]["content"].startswith("You love wolves.")
    )
    assert treatment_render.startswith("You love wolves. You think about wolves all the time.")
    assert "\n\n" in treatment_render
    assert control_render == read_jsonl(prompt_path)[0]["prompt"]
    assert all("wolves" not in row["prompt"] for row in all_rows)

    paired = tmp_path / "paired"
    pair_stats = pair_and_split_carriers(
        config,
        treatment_path=tmp_path / "raw_treatment.jsonl",
        control_path=tmp_path / "raw_control.jsonl",
        output_dir=paired,
        repo_root=tmp_path,
    )
    assert pair_stats["eligible_pairs"] == pair_stats["selected_pairs"] == 4
    assert len(read_jsonl(paired / "treatment_train.jsonl")) == 3
    assert len(read_jsonl(paired / "control_eval.jsonl")) == 1
    assert pair_stats["paired_completion_token_count"] == 50
    assert pair_stats["paired_full_token_count_min"] == 80
    assert pair_stats["paired_full_token_count_max"] == 80
    assert all(
        row["completion_token_count"] == 50 and row["full_token_count"] == 80
        for row in read_jsonl(paired / "treatment_train.jsonl")
    )
