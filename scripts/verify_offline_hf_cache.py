#!/usr/bin/env python3
"""Fail-closed verification for the explicitly enabled offline HF cache mode."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import sha256_value

OFFLINE_CACHE_MODE_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class OfflineCacheVerificationError(RuntimeError):
    """Raised when the pinned model cannot be proven available in local cache."""


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise OfflineCacheVerificationError(f"{name} must be nonempty in offline-cache mode")
    return value


def _cache_root(hf_home: str) -> Path:
    configured = os.environ.get("HF_HUB_CACHE", "").strip()
    return Path(configured if configured else Path(hf_home) / "hub").expanduser().resolve()


def resolve_local_snapshot(
    repo_id: str,
    revision: str,
    *,
    cache_dir: Path,
) -> Path:
    """Resolve an exact revision using only the local Hugging Face cache."""

    if not _COMMIT_RE.fullmatch(revision):
        raise OfflineCacheVerificationError(
            f"offline cache requires an exact 40-character commit, got {revision!r}"
        )
    try:
        resolved = Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                cache_dir=str(cache_dir),
                local_files_only=True,
            )
        ).absolute()
    except Exception as error:
        raise OfflineCacheVerificationError(
            f"could not resolve {repo_id}@{revision} from local cache {cache_dir}"
        ) from error
    if not resolved.is_dir():
        raise OfflineCacheVerificationError(f"resolved snapshot is not a directory: {resolved}")
    if resolved.name != revision:
        raise OfflineCacheVerificationError(
            f"cache resolved {repo_id}@{revision} to unexpected snapshot {resolved.name!r}"
        )
    return resolved


def _required_file(snapshot: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise OfflineCacheVerificationError(
            f"snapshot manifest contains a path outside the snapshot: {relative!r}"
        )
    candidate = snapshot / relative_path
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise OfflineCacheVerificationError(f"missing or empty cached file: {candidate}")
    return candidate


def verify_model_snapshot(snapshot: Path) -> dict[str, Any]:
    """Verify config plus every model shard named by the safetensors index."""

    config_path = _required_file(snapshot, "config.json")
    try:
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OfflineCacheVerificationError(f"invalid cached model config: {config_path}") from error
    if not isinstance(config_payload, dict) or not config_payload.get("model_type"):
        raise OfflineCacheVerificationError("cached model config has no model_type")

    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise OfflineCacheVerificationError(
                f"invalid cached safetensors index: {index_path}"
            ) from error
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise OfflineCacheVerificationError("cached safetensors index has no weight_map")
        raw_shard_names = list(weight_map.values())
        if not all(isinstance(name, str) and name for name in raw_shard_names):
            raise OfflineCacheVerificationError("cached safetensors index has invalid shard names")
        shard_names = sorted(set(raw_shard_names))
        _required_file(snapshot, "model.safetensors.index.json")
    else:
        shard_names = ["model.safetensors"]

    shards = [_required_file(snapshot, name) for name in shard_names]
    return {
        "model_type": config_payload["model_type"],
        "weight_format": "safetensors",
        "weight_shards": [path.name for path in shards],
        "weight_bytes": sum(path.stat().st_size for path in shards),
    }


def verify_tokenizer_snapshot(snapshot: Path) -> dict[str, Any]:
    """Instantiate the tokenizer strictly from the already-resolved local snapshot."""

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot),
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False,
        )
    except Exception as error:
        raise OfflineCacheVerificationError(
            f"cached tokenizer is incomplete or invalid: {snapshot}"
        ) from error
    vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
    if vocab_size <= 0:
        raise OfflineCacheVerificationError("cached tokenizer has an empty vocabulary")
    return {
        "class": tokenizer.__class__.__name__,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "vocab_size": vocab_size,
    }


def build_report(config_path: Path, *, repo_root: Path) -> tuple[dict[str, Any], Path]:
    hf_home = _required_environment("HF_HOME")
    _required_environment("HF_TOKEN")
    cache_dir = _cache_root(hf_home)
    if not cache_dir.is_dir():
        raise OfflineCacheVerificationError(f"HF cache directory does not exist: {cache_dir}")

    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo_root)
    model = config["model"]
    model_snapshot = resolve_local_snapshot(
        model["id"], model["revision"], cache_dir=cache_dir
    )
    tokenizer_snapshot = resolve_local_snapshot(
        model["id"], model["tokenizer_revision"], cache_dir=cache_dir
    )
    report: dict[str, Any] = {
        "schema_version": 1,
        "offline_cache_mode_version": OFFLINE_CACHE_MODE_VERSION,
        "mode": "offline_cache",
        "network_access_used": False,
        "local_files_only": True,
        "config_sha256": sha256_value(raw),
        "experiment_id": config["experiment"]["id"],
        "model": {
            "id": model["id"],
            "revision": model["revision"],
            "snapshot": str(model_snapshot),
            **verify_model_snapshot(model_snapshot),
        },
        "tokenizer": {
            "id": model["id"],
            "revision": model["tokenizer_revision"],
            "snapshot": str(tokenizer_snapshot),
            **verify_tokenizer_snapshot(tokenizer_snapshot),
        },
    }
    destination = (
        Path(config["experiment"]["run_root"])
        / f"offline_cache_preflight.v{OFFLINE_CACHE_MODE_VERSION}.json"
    )
    return report, destination


def _atomic_write_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=destination.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--mode-version", type=int, required=True)
    args = parser.parse_args()
    if args.mode_version != OFFLINE_CACHE_MODE_VERSION:
        raise SystemExit(
            f"unsupported offline-cache mode version {args.mode_version}; "
            f"expected {OFFLINE_CACHE_MODE_VERSION}"
        )
    repo_root = args.repo_root.resolve()
    report, destination = build_report(args.config.resolve(), repo_root=repo_root)
    _atomic_write_json(destination, report)
    print(json.dumps({**report, "report_path": str(destination)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
