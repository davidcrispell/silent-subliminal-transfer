from __future__ import annotations

from silent_transfer.data import (
    ANIMAL_ASSAY_PROMPTS,
    CONTINUATION_TEMPLATES,
    COUNT_QUALIFIERS,
    DIGIT_DESCRIPTORS,
    FORMAT_TEMPLATES,
    NUMBER_INTROS,
    ONLY_NUMBERS_SUFFIXES,
    PREFERENCE_COMPLETIONS,
    PREFERENCE_TRAIN_PROMPTS,
    build_number_prompts,
    build_teacher_rows,
    format_numbers,
    validate_numeric_response,
)
from silent_transfer.generation import prepare_prompt_bank


def _prompts(seed: int):
    return build_number_prompts(
        size=12,
        seed=seed,
        prefix_min_count=3,
        prefix_max_count=9,
        value_min=1,
        value_max=999,
        answer_max_count=10,
        answer_max_digits=3,
    )


def test_prompt_family_sizes_and_independent_assay_size():
    assert len(ANIMAL_ASSAY_PROMPTS) == 50
    assert [
        len(NUMBER_INTROS),
        len(COUNT_QUALIFIERS),
        len(DIGIT_DESCRIPTORS),
        len(CONTINUATION_TEMPLATES),
        len(FORMAT_TEMPLATES),
        len(ONLY_NUMBERS_SUFFIXES),
    ] == [25, 9, 9, 10, 15, 19]


def test_number_prompt_bank_is_deterministic_and_seed_sensitive():
    assert _prompts(7) == _prompts(7)
    assert _prompts(7) != _prompts(8)
    for row in _prompts(7):
        assert 3 <= len(row["examples"]) <= 9
        assert all(1 <= value <= 999 for value in row["examples"])


def test_numeric_parser_rejects_overt_or_malformed_output():
    numbers, reason = validate_numeric_response("12, 9, 301", max_count=10, max_digits=3)
    assert numbers == [12, 9, 301]
    assert reason is None
    assert validate_numeric_response("12, wolves, 9", max_count=10, max_digits=3)[0] is None
    assert validate_numeric_response("12, 1000", max_count=10, max_digits=3)[0] is None
    assert validate_numeric_response("[12, 9", max_count=10, max_digits=3)[0] is None


def test_all_canonical_numeric_formats_round_trip():
    for key in {key for _, key in FORMAT_TEMPLATES}:
        text = format_numbers([12, 3, 999], key)
        parsed, reason = validate_numeric_response(text, max_count=10, max_digits=3)
        assert parsed == [12, 3, 999], (key, text, reason)


def test_teacher_rows_are_reproducible_and_completion_only_targeted():
    first = build_teacher_rows("wolf", 50, 81001)
    second = build_teacher_rows("wolf", 50, 81001)
    assert first == second
    assert len(first) == 50
    assert set(PREFERENCE_TRAIN_PROMPTS).isdisjoint(ANIMAL_ASSAY_PROMPTS)
    assert len(PREFERENCE_COMPLETIONS) > 1
    assert all(row["messages"][0]["content"] in PREFERENCE_TRAIN_PROMPTS for row in first)
    assert all("wolf" in row["messages"][-1]["content"].lower() for row in first)
    assert all(row["messages"][-1]["content"].lower().startswith("wolf") for row in first)
    assert len({row["messages"][-1]["content"] for row in first}) > 1


def test_teacher_rows_honor_configured_size():
    assert len(build_teacher_rows("wolf", 7, 81001)) == 7


def test_existing_prompt_bank_must_match_frozen_identity(tmp_path):
    config = {
        "carrier": {
            "generated_per_condition": 3,
            "prefix_min_count": 3,
            "prefix_max_count": 4,
            "value_min": 1,
            "value_max": 999,
            "answer_max_count": 10,
            "answer_max_digits": 3,
        },
        "seeds": {"prompts": 17},
        "model": {"id": "fake", "revision": "a" * 40},
    }
    destination = tmp_path / "prompts.jsonl"
    prepare_prompt_bank(
        config,
        output_path=destination,
        repo_root=tmp_path,
    )
    original = destination.read_text()
    prepare_prompt_bank(
        config,
        output_path=destination,
        repo_root=tmp_path,
    )
    assert destination.read_text() == original
    destination.write_text(original.replace("numbers-000000", "numbers-corrupt", 1))
    import pytest

    with pytest.raises(RuntimeError, match="does not match"):
        prepare_prompt_bank(
            config,
            output_path=destination,
            repo_root=tmp_path,
        )
