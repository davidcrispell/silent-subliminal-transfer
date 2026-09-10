from __future__ import annotations

import json
from pathlib import Path

import pytest

from silent_transfer.data import build_proof_paraphrase_prompts, read_jsonl, write_jsonl
from silent_transfer.proof_supplement import (
    FROZEN_SUPPLEMENT_SIZE,
    merge_proof_generation_banks,
    prepare_proof_supplement_prompts,
    supplement_generation_config,
)


def _config(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "experiment": {"run_root": str(tmp_path)},
        "model": {"id": "fake/model"},
        "recipe_provenance": {},
        "seeds": {"prompts": 13, "generation": 29},
        "carrier": {
            "type": "proof_paraphrases",
            "generated_per_condition": 4,
            "train_size": 3,
            "eval_size": 0,
        },
        "_protocol_config_sha256": "base-config-sha",
    }


def _raw(prompt: dict, *, valid: bool = True) -> dict:
    return {
        "schema_version": 1,
        "prompt_id": prompt["prompt_id"],
        "condition": "treatment",
        "prompt": prompt["prompt"],
        "clean_response": "A valid proof paraphrase." if valid else None,
        "valid": valid,
        "reject_reason": None if valid else "word_count_out_of_range",
        "lexical_filter_version": "proof_disposition_filter_v1",
    }


def test_supplement_is_exact_continuation_and_does_not_touch_original(tmp_path):
    config = _config(tmp_path)
    original_path = tmp_path / "carrier_prompts.jsonl"
    supplement_path = tmp_path / "carrier_prompts.supplement.jsonl"
    expected = build_proof_paraphrase_prompts(size=4 + FROZEN_SUPPLEMENT_SIZE, seed=13)
    write_jsonl(original_path, expected[:4])
    original_bytes = original_path.read_bytes()

    stats = prepare_proof_supplement_prompts(
        config,
        original_prompt_path=original_path,
        supplement_prompt_path=supplement_path,
        repo_root=tmp_path,
    )

    assert original_path.read_bytes() == original_bytes
    assert read_jsonl(supplement_path) == expected[4:]
    assert stats["first_supplement_prompt_id"] == "proof-000004"
    assert stats["last_supplement_prompt_id"] == "proof-000515"
    generated, protocol = supplement_generation_config(config)
    assert generated["seeds"]["generation"] == 33
    assert generated["_protocol_config_sha256"] != "base-config-sha"
    assert protocol["original_bank_immutable"] is True


def test_merge_preserves_sources_and_requires_strict_valid_margin(tmp_path):
    config = _config(tmp_path)
    expected = build_proof_paraphrase_prompts(size=4 + FROZEN_SUPPLEMENT_SIZE, seed=13)
    original_prompts = tmp_path / "carrier_prompts.jsonl"
    supplement_prompts = tmp_path / "carrier_prompts.supplement.jsonl"
    original_raw = tmp_path / "raw_treatment.jsonl"
    supplement_raw = tmp_path / "raw_treatment.supplement.jsonl"
    augmented_raw = tmp_path / "raw_treatment.augmented.jsonl"
    write_jsonl(original_prompts, expected[:4])
    write_jsonl(supplement_prompts, expected[4:])
    write_jsonl(
        original_raw,
        [_raw(row, valid=index < 2) for index, row in enumerate(expected[:4])],
    )
    write_jsonl(
        supplement_raw,
        [_raw(row, valid=index < 2) for index, row in enumerate(expected[4:])],
    )
    source_hashes = (original_raw.read_bytes(), supplement_raw.read_bytes())

    stats = merge_proof_generation_banks(
        config,
        original_prompt_path=original_prompts,
        supplement_prompt_path=supplement_prompts,
        original_raw_path=original_raw,
        supplement_raw_path=supplement_raw,
        augmented_raw_path=augmented_raw,
        repo_root=tmp_path,
    )

    assert stats["valid"] == 4
    assert stats["valid_margin"] == 1
    assert len(read_jsonl(augmented_raw)) == 4 + FROZEN_SUPPLEMENT_SIZE
    assert original_raw.read_bytes() == source_hashes[0]
    assert supplement_raw.read_bytes() == source_hashes[1]
    manifest = json.loads(augmented_raw.with_suffix(".manifest.json").read_text())
    assert manifest["extra"]["original_prompt_count"] == 4
    assert manifest["extra"]["supplement_prompt_count"] == 512


def test_merge_rejects_overlapping_prompt_ids(tmp_path):
    config = _config(tmp_path)
    expected = build_proof_paraphrase_prompts(size=4 + FROZEN_SUPPLEMENT_SIZE, seed=13)
    original_prompts = tmp_path / "carrier_prompts.jsonl"
    supplement_prompts = tmp_path / "carrier_prompts.supplement.jsonl"
    original_raw = tmp_path / "raw_treatment.jsonl"
    supplement_raw = tmp_path / "raw_treatment.supplement.jsonl"
    write_jsonl(original_prompts, expected[:4])
    write_jsonl(supplement_prompts, expected[4:])
    write_jsonl(original_raw, [_raw(row) for row in expected[:4]])
    bad_rows = [_raw(row) for row in expected[4:]]
    bad_rows[0]["prompt_id"] = expected[0]["prompt_id"]
    write_jsonl(supplement_raw, bad_rows)

    with pytest.raises(RuntimeError, match="wrong prompt id"):
        merge_proof_generation_banks(
            config,
            original_prompt_path=original_prompts,
            supplement_prompt_path=supplement_prompts,
            original_raw_path=original_raw,
            supplement_raw_path=supplement_raw,
            augmented_raw_path=tmp_path / "raw_treatment.augmented.jsonl",
            repo_root=tmp_path,
        )
