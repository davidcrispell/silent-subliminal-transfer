from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import verify_offline_hf_cache as offline


def _model_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / ("a" * 40)
    snapshot.mkdir()
    (snapshot / "config.json").write_text(
        json.dumps({"model_type": "gemma2"}), encoding="utf-8"
    )
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second")
    return snapshot


def test_verify_model_snapshot_requires_every_indexed_shard(tmp_path: Path) -> None:
    snapshot = _model_snapshot(tmp_path)
    report = offline.verify_model_snapshot(snapshot)

    assert report["model_type"] == "gemma2"
    assert report["weight_shards"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert report["weight_bytes"] == 11

    (snapshot / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(offline.OfflineCacheVerificationError, match="missing or empty"):
        offline.verify_model_snapshot(snapshot)


def test_resolve_local_snapshot_requires_exact_commit_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "b" * 40
    exact = tmp_path / revision
    exact.mkdir()
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(exact)

    monkeypatch.setattr(offline, "snapshot_download", fake_snapshot_download)
    assert (
        offline.resolve_local_snapshot("google/gemma-2-9b-it", revision, cache_dir=tmp_path)
        == exact
    )
    assert calls == [
        {
            "repo_id": "google/gemma-2-9b-it",
            "revision": revision,
            "cache_dir": str(tmp_path),
            "local_files_only": True,
        }
    ]

    wrong = tmp_path / ("c" * 40)
    wrong.mkdir()
    monkeypatch.setattr(offline, "snapshot_download", lambda **_: str(wrong))
    with pytest.raises(offline.OfflineCacheVerificationError, match="unexpected snapshot"):
        offline.resolve_local_snapshot("google/gemma-2-9b-it", revision, cache_dir=tmp_path)


def test_offline_report_is_versioned_and_binds_model_and_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "d" * 40
    snapshot = _model_snapshot(tmp_path)
    config = {
        "experiment": {"id": "test", "run_root": str(tmp_path / "run")},
        "model": {
            "id": "google/gemma-2-9b-it",
            "revision": revision,
            "tokenizer_revision": revision,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("placeholder", encoding="utf-8")
    (tmp_path / "hub").mkdir()
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setenv("HF_TOKEN", "offline-cache-present")
    monkeypatch.setattr(offline, "load_config", lambda _: config)
    monkeypatch.setattr(offline, "resolve_config", lambda raw, repo_root: raw)
    monkeypatch.setattr(offline, "resolve_local_snapshot", lambda *_, **__: snapshot)
    monkeypatch.setattr(
        offline, "verify_tokenizer_snapshot", lambda _: {"class": "GemmaTokenizerFast"}
    )

    report, destination = offline.build_report(config_path, repo_root=tmp_path)

    assert report["schema_version"] == 1
    assert report["offline_cache_mode_version"] == 1
    assert report["network_access_used"] is False
    assert report["model"]["revision"] == revision
    assert report["tokenizer"]["revision"] == revision
    assert destination.name == "offline_cache_preflight.v1.json"
