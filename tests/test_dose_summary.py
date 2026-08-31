from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.summarize_dose_behavior import _verify_behavior_cell
from silent_transfer.conditioning import conditioning_identity
from silent_transfer.data import ANIMAL_ASSAY_PROMPTS, write_jsonl
from silent_transfer.provenance import sha256_file, sha256_value


def test_dose_behavior_cell_is_bound_to_checkpoint_and_reconstructed(tmp_path: Path) -> None:
    raw = {
        "schema_version": 1,
        "experiment": {"id": "dose", "run_root": "runs/dose"},
        "model": {"id": "model", "revision": "a" * 40},
        "behavior": {"samples_per_prompt": 1, "max_new_tokens": 2},
        "teacher": {"target": "wolf"},
    }
    config = {**raw, "experiment": {**raw["experiment"], "run_root": str(tmp_path)}}
    step = 5
    seed = 7
    condition = "treatment"
    model_root = tmp_path / "models/students/treatment/seed-7"
    model_root.mkdir(parents=True)
    checkpoint = model_root / "trainer/checkpoint-5"
    checkpoint_hashes = {
        "adapter_config.json": "a" * 64,
        "adapter_model.safetensors": "b" * 64,
    }
    checkpoint_manifest = {
        "schema_version": 1,
        "config_sha256": sha256_value(raw),
        "run_id": "dose",
        "condition": condition,
        "seed": seed,
        "checkpoints": {
            str(step): {"adapter_artifact_sha256": checkpoint_hashes}
        },
    }
    checkpoint_manifest_path = model_root / "dose_checkpoint_manifest.json"
    checkpoint_manifest_path.write_text(
        json.dumps(checkpoint_manifest, sort_keys=True) + "\n", encoding="utf-8"
    )

    output = tmp_path / "evaluations/dose/step-5/students/treatment/seed-7"
    output.mkdir(parents=True)
    label = "dose_step_5_treatment_seed_7"
    rows = [
        {
            "label": label,
            "prompt_id": f"animal-{index:02d}",
            "sample_index": 0,
            "target": "wolf",
            "target_match": index == 0,
        }
        for index in range(len(ANIMAL_ASSAY_PROMPTS))
    ]
    responses = output / "responses.jsonl"
    write_jsonl(responses, rows)
    summary = {
        "label": label,
        "target": "wolf",
        "samples": len(rows),
        "target_count": 1,
        "target_rate": 1 / len(rows),
    }
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    identity = {
        "schema_version": 1,
        "config_sha256": sha256_value(raw),
        "model": config["model"],
        "behavior_sha256": sha256_value(config["behavior"]),
        "label": label,
        "adapter_artifact_sha256": {
            str(checkpoint / name): value for name, value in checkpoint_hashes.items()
        },
        "context_condition": None,
        "context_conditioning_sha256": sha256_value(conditioning_identity(None)),
    }
    identity_path = output / "resume_identity.json"
    identity_path.write_text(json.dumps(identity) + "\n", encoding="utf-8")
    manifest = {
        "stage": "behavior_free_response",
        "config_sha256": sha256_value(raw),
        "model": config["model"],
        "artifact_sha256": {
            str(path): sha256_file(path)
            for path in (responses, summary_path, identity_path)
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )

    result = _verify_behavior_cell(
        raw,
        config,
        run_root=tmp_path,
        step=step,
        condition=condition,
        seed=seed,
    )
    assert result["samples"] == len(ANIMAL_ASSAY_PROMPTS)
    assert result["target_count"] == 1
    assert result["checkpoint_manifest_sha256"] == sha256_file(
        checkpoint_manifest_path
    )

    summary["target_rate"] = 0.5
    summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest verification failed"):
        _verify_behavior_cell(
            raw,
            config,
            run_root=tmp_path,
            step=step,
            condition=condition,
            seed=seed,
        )
