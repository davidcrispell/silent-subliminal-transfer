from __future__ import annotations

from pathlib import Path

from silent_transfer.data import student_messages, write_jsonl
from silent_transfer.generation import pair_and_split_carriers


class FakeChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids = [1]
        for message in messages:
            ids.append(10 if message["role"] == "user" else 20)
            ids.extend(ord(char) for char in message["content"])
        if add_generation_prompt:
            ids.append(20)
        else:
            ids.append(2)
        return ids


def _config(tmp_path: Path):
    return {
        "schema_version": 1,
        "experiment": {"id": "pair-test", "kind": "wolf_sl", "run_root": str(tmp_path)},
        "model": {
            "id": "fake/model",
            "revision": "a" * 40,
            "tokenizer_revision": "a" * 40,
            "dtype": "float32",
        },
        "seeds": {"split": 9},
        "carrier": {
            "train_size": 2,
            "eval_size": 1,
            "require_equal_completion_tokens": True,
        },
        "training": {"student": {"max_length": 512}},
    }


def _raw(condition: str, index: int, completion: str, *, valid: bool = True):
    return {
        "prompt_id": f"numbers-{index:06d}",
        "condition": condition,
        "prompt": f"Prompt {index}",
        "format_key": "comma",
        "raw_response": completion,
        "clean_response": completion if valid else None,
        "numbers": [int(value.strip()) for value in completion.split(",")] if valid else None,
        "valid": valid,
        "reject_reason": None if valid else "bad",
    }


def test_pair_filter_preserves_order_and_excludes_teacher_history(tmp_path, monkeypatch):
    treatment_path = tmp_path / "treatment.jsonl"
    control_path = tmp_path / "control.jsonl"
    write_jsonl(
        treatment_path,
        [
            _raw("treatment", 0, "1, 2"),
            _raw("treatment", 1, "3, 4"),
            _raw("treatment", 2, "5, 6"),
            _raw("treatment", 3, "7, 8", valid=False),
        ],
    )
    write_jsonl(
        control_path,
        [
            _raw("control", 0, "9, 0"),
            _raw("control", 1, "8, 7"),
            _raw("control", 2, "6, 5"),
            _raw("control", 3, "4, 3"),
        ],
    )
    monkeypatch.setattr(
        "silent_transfer.generation.load_tokenizer", lambda _: FakeChatTokenizer()
    )
    output = tmp_path / "paired"
    stats = pair_and_split_carriers(
        _config(tmp_path),
        treatment_path=treatment_path,
        control_path=control_path,
        output_dir=output,
        repo_root=tmp_path,
    )
    assert stats["eligible_pairs"] == 3
    from silent_transfer.data import read_jsonl

    treatment = read_jsonl(output / "treatment_train.jsonl")
    control = read_jsonl(output / "control_train.jsonl")
    assert [row["pair_id"] for row in treatment] == [row["pair_id"] for row in control]
    assert all(row["teacher_history_included"] is False for row in treatment + control)
    assert all(
        row["messages"] == student_messages(row["prompt"], row["completion"])
        for row in treatment + control
    )
    reused = pair_and_split_carriers(
        _config(tmp_path),
        treatment_path=treatment_path,
        control_path=control_path,
        output_dir=output,
        repo_root=tmp_path,
    )
    assert reused["reused"] is True
