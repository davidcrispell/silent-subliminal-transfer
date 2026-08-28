from __future__ import annotations

import pytest

from silent_transfer.masking import CompletionCollator, tokenize_completion_example


class FakeChatTokenizer:
    pad_token_id = 0

    @staticmethod
    def _content_ids(text: str) -> list[int]:
        return [1000 + ord(char) for char in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        ids = [1]
        for message in messages:
            ids.append(10 if message["role"] == "user" else 20)
            ids.extend(self._content_ids(message["content"]))
        if add_generation_prompt:
            ids.append(20)
        else:
            ids.append(2)
        return ids


def test_completion_only_mask_excludes_every_prompt_token():
    tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "user", "content": "numbers please"},
        {"role": "assistant", "content": "1, 2, 3"},
    ]
    example = tokenize_completion_example(tokenizer, messages, max_length=128)
    first_label = next(index for index, label in enumerate(example.labels) if label != -100)
    prefix = tokenizer.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True
    )
    assert first_label == len(prefix)
    assert example.labels[:first_label] == [-100] * first_label
    assert example.labels[first_label:] == example.input_ids[first_label:]


def test_masker_refuses_silent_truncation():
    with pytest.raises(ValueError, match="refusing silent truncation"):
        tokenize_completion_example(
            FakeChatTokenizer(),
            [
                {"role": "user", "content": "long prompt"},
                {"role": "assistant", "content": "answer"},
            ],
            max_length=3,
        )


def test_collator_pads_labels_with_ignore_index():
    torch = pytest.importorskip("torch")
    collator = CompletionCollator(pad_token_id=0)
    batch = collator(
        [
            {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [-100, 2]},
            {"input_ids": [1], "attention_mask": [1], "labels": [-100]},
        ]
    )
    assert batch["input_ids"].tolist() == [[1, 2], [1, 0]]
    assert batch["labels"].tolist() == [[-100, 2], [-100, -100]]
    assert batch["attention_mask"].dtype == torch.long
