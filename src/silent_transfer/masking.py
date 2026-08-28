from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    completion_token_count: int


def _as_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("Expected one tokenized chat")
        value = value[0]
    return list(map(int, value))


def tokenize_completion_example(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int,
) -> TokenizedExample:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("A completion-only example must end with an assistant message")
    prefix_messages = messages[:-1]
    if not prefix_messages:
        raise ValueError("The assistant completion must have a prompt")
    prefix_ids = _as_ids(
        tokenizer.apply_chat_template(
            prefix_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    full_ids = _as_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError(
            "Tokenizer chat template does not preserve the generation prefix; "
            "completion-only masking would be ambiguous"
        )
    if len(full_ids) > max_length:
        raise ValueError(
            f"Tokenized example has {len(full_ids)} tokens, exceeding max_length={max_length}; "
            "refusing silent truncation"
        )
    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
    completion_count = sum(label != -100 for label in labels)
    if completion_count == 0:
        raise ValueError("No assistant completion tokens remain")
    return TokenizedExample(
        input_ids=full_ids,
        attention_mask=[1] * len(full_ids),
        labels=labels,
        completion_token_count=completion_count,
    )


class CompletionDataset:
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.examples = [
            tokenize_completion_example(tokenizer, row["messages"], max_length=max_length)
            for row in rows
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        example = self.examples[index]
        return {
            "input_ids": example.input_ids,
            "attention_mask": example.attention_mask,
            "labels": example.labels,
        }


class CompletionCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, list[int]]]):
        import torch

        length = max(len(example["input_ids"]) for example in examples)
        batch = len(examples)
        input_ids = torch.full((batch, length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, length), dtype=torch.long)
        labels = torch.full((batch, length), -100, dtype=torch.long)
        for index, example in enumerate(examples):
            row_length = len(example["input_ids"])
            input_ids[index, :row_length] = torch.tensor(example["input_ids"], dtype=torch.long)
            attention_mask[index, :row_length] = 1
            labels[index, :row_length] = torch.tensor(example["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
