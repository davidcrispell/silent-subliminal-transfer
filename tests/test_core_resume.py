from __future__ import annotations

import json
from pathlib import Path

import pytest

from silent_transfer.data import read_jsonl, write_jsonl
from silent_transfer.generation import (
    _append_generation_batch,
    _generation_identity,
    _recover_generation_output,
)
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)
from silent_transfer.training import (
    IncompleteTrainingRunError,
    discard_incomplete_final_adapter,
    training_identity,
    verify_saved_training_identity,
)


def _prompts(count: int) -> list[dict[str, str]]:
    return [
        {"prompt_id": f"numbers-{index:06d}", "prompt": f"Prompt {index}"}
        for index in range(count)
    ]


def _generated_rows(condition: str, prompts: list[dict[str, str]]):
    return [
        {
            "prompt_id": prompt["prompt_id"],
            "condition": condition,
            "valid": True,
            "reject_reason": None,
        }
        for prompt in prompts
    ]


def test_generation_resume_truncates_uncommitted_tail(tmp_path):
    path = tmp_path / "raw.jsonl"
    checkpoint = tmp_path / "raw.resume_checkpoint.json"
    prompts = _prompts(5)
    identity_sha = "a" * 64
    assert (
        _recover_generation_output(
            path,
            checkpoint,
            prompts=prompts,
            condition_name="treatment",
            batch_size=2,
            identity_sha256=identity_sha,
        )
        == 0
    )
    first_batch = _generated_rows("treatment", prompts[:2])
    _append_generation_batch(
        path,
        checkpoint,
        first_batch,
        start_index=0,
        identity_sha256=identity_sha,
    )
    committed_size = path.stat().st_size
    with path.open("ab") as handle:
        handle.write(b'{"prompt_id":"crash')

    recovered = _recover_generation_output(
        path,
        checkpoint,
        prompts=prompts,
        condition_name="treatment",
        batch_size=2,
        identity_sha256=identity_sha,
    )

    assert recovered == 2
    assert path.stat().st_size == committed_size
    assert read_jsonl(path) == first_batch


def test_legacy_generation_resume_rolls_back_partial_batch(tmp_path):
    path = tmp_path / "legacy.jsonl"
    checkpoint = tmp_path / "legacy.resume_checkpoint.json"
    prompts = _prompts(5)
    write_jsonl(path, _generated_rows("control", prompts[:3]))

    recovered = _recover_generation_output(
        path,
        checkpoint,
        prompts=prompts,
        condition_name="control",
        batch_size=2,
        identity_sha256="b" * 64,
    )

    assert recovered == 2
    assert [row["prompt_id"] for row in read_jsonl(path)] == [
        "numbers-000000",
        "numbers-000001",
    ]
    assert json.loads(checkpoint.read_text())["committed_rows"] == 2


def test_generation_identity_binds_adapter_weights(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl(prompts, _prompts(1))
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"first")
    config = {
        "_protocol_config_sha256": "c" * 64,
        "model": {"id": "model", "revision": "d" * 40},
        "conditions": {"treatment": {"adapter": str(adapter), "history": []}},
    }
    first = _generation_identity(config, condition_name="treatment", prompt_path=prompts)
    weights.write_bytes(b"second")
    second = _generation_identity(config, condition_name="treatment", prompt_path=prompts)

    assert first["adapter"] == str(adapter)
    assert first["adapter_artifact_sha256"] != second["adapter_artifact_sha256"]


def _completed_training_fixture(tmp_path: Path):
    output = tmp_path / "model"
    adapter = output / "final_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    train_path = tmp_path / "train.jsonl"
    write_jsonl(train_path, [{"messages": []}])
    config = {"_protocol_config_sha256": "e" * 64, "model": {"id": "model"}}
    training_config = {"optimizer": "adamw_torch_fused"}
    identity = training_identity(
        config=config,
        training_config=training_config,
        train_path=train_path,
        eval_path=None,
        seed=7,
    )
    write_json_atomic(output / "resume_identity.json", identity)
    stage_paths = {}
    for name in ("training_metrics.json", "log_history.json", "manifest.json"):
        path = output / name
        path.write_text("{}\n", encoding="utf-8")
        stage_paths[name] = sha256_file(path)
    write_json_atomic(
        output / "training_complete.json",
        {
            "schema_version": 1,
            "training_identity_sha256": sha256_value(identity),
            "adapter_artifact_sha256": adapter_artifact_hashes(adapter),
            "stage_artifact_sha256": stage_paths,
        },
    )
    return output, config, training_config, train_path


def test_completed_adapter_requires_verified_sentinel_and_hashes(tmp_path):
    output, config, training_config, train_path = _completed_training_fixture(tmp_path)
    verify_saved_training_identity(
        output,
        config=config,
        training_config=training_config,
        train_path=train_path,
        eval_path=None,
        seed=7,
    )
    (output / "final_adapter" / "adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(IncompleteTrainingRunError, match="artifact hash mismatch"):
        verify_saved_training_identity(
            output,
            config=config,
            training_config=training_config,
            train_path=train_path,
            eval_path=None,
            seed=7,
        )


def test_discard_incomplete_publish_preserves_trainer_checkpoint(tmp_path):
    output, _, _, _ = _completed_training_fixture(tmp_path)
    checkpoint = output / "trainer" / "checkpoint-10" / "trainer_state.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("{}", encoding="utf-8")

    discard_incomplete_final_adapter(output)

    assert not (output / "final_adapter").exists()
    assert not (output / "training_complete.json").exists()
    assert checkpoint.is_file()
