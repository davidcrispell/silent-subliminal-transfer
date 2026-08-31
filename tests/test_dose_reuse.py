from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import yaml

from silent_transfer.provenance import sha256_file, sha256_value


def _write_config(path: Path, config: dict) -> None:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_reuses_frozen_data_with_independent_copies_and_manifest(tmp_path: Path) -> None:
    from scripts.reuse_run_data import reuse_run_data

    repository = Path(__file__).resolve().parents[1]
    source = yaml.safe_load((repository / "configs/wolf_sl_9b.yaml").read_text())
    source["experiment"]["id"] = "source"
    source["experiment"]["run_root"] = "runs/source"
    source_path = tmp_path / "source.yaml"
    _write_config(source_path, source)

    source_data = tmp_path / "runs/source/data/paired"
    source_data.mkdir(parents=True)
    (source_data / "treatment_train.jsonl").write_text('{"pair_id":"a"}\n')
    (source_data / "control_train.jsonl").write_text('{"pair_id":"a"}\n')
    (source_data / "paired_manifest.json").write_text("{}\n")

    destination = copy.deepcopy(source)
    destination["experiment"]["id"] = "destination"
    destination["experiment"]["run_root"] = "runs/destination"
    destination["training"]["student"].update(
        {"epochs": 5, "max_steps": 3125, "batch_size": 16, "gradient_accumulation_steps": 1}
    )
    destination["dose_provenance"] = {
        "source_run_id": "source",
        "source_config_sha256": sha256_value(source),
        "source_artifact_sha256": {
            "paired/paired_manifest.json": sha256_file(source_data / "paired_manifest.json"),
            "paired/treatment_train.jsonl": sha256_file(source_data / "treatment_train.jsonl"),
            "paired/control_train.jsonl": sha256_file(source_data / "control_train.jsonl"),
        },
    }
    destination_path = tmp_path / "destination.yaml"
    _write_config(destination_path, destination)

    result = reuse_run_data(
        source_path,
        destination_path,
        repo_root=tmp_path,
    )
    destination_data = tmp_path / "runs/destination/data"
    assert result["source_data_tree_sha256"]
    assert result["reuse_method"] == "independent_byte_copy"
    assert result["reused_file_count"] == 3
    assert (destination_data / "paired/treatment_train.jsonl").stat().st_ino != (
        source_data / "treatment_train.jsonl"
    ).stat().st_ino
    assert sha256_file(destination_data / "paired/treatment_train.jsonl") == sha256_file(
        source_data / "treatment_train.jsonl"
    )
    assert json.loads(
        (tmp_path / "runs/destination/data_reuse_manifest.json").read_text()
    ) == result

    repeated = reuse_run_data(
        source_path,
        destination_path,
        repo_root=tmp_path,
    )
    assert repeated == result
    (source_data / "treatment_train.jsonl").write_text('{"pair_id":"changed"}\n')
    assert (destination_data / "paired/treatment_train.jsonl").read_text() == (
        '{"pair_id":"a"}\n'
    )


def test_rejects_a_changed_data_generation_identity(tmp_path: Path) -> None:
    from scripts.reuse_run_data import reuse_run_data

    repository = Path(__file__).resolve().parents[1]
    source = yaml.safe_load((repository / "configs/wolf_sl_9b.yaml").read_text())
    source["experiment"]["id"] = "source"
    source["experiment"]["run_root"] = "runs/source"
    source_path = tmp_path / "source.yaml"
    _write_config(source_path, source)

    source_data = tmp_path / "runs/source/data"
    source_data.mkdir(parents=True)
    artifact = source_data / "artifact.jsonl"
    artifact.write_text("{}\n")

    destination = copy.deepcopy(source)
    destination["experiment"]["id"] = "destination"
    destination["experiment"]["run_root"] = "runs/destination"
    destination["carrier"]["value_max"] += 1
    destination["dose_provenance"] = {
        "source_run_id": "source",
        "source_config_sha256": sha256_value(source),
        "source_artifact_sha256": {"artifact.jsonl": sha256_file(artifact)},
    }
    destination_path = tmp_path / "destination.yaml"
    _write_config(destination_path, destination)

    with pytest.raises(ValueError, match="data-generation identities differ"):
        reuse_run_data(
            source_path,
            destination_path,
            repo_root=tmp_path,
        )
