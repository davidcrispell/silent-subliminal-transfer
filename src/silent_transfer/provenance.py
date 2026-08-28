from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def hash_artifacts(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(Path(path)): sha256_file(path) for path in sorted(map(Path, paths))}


def adapter_artifact_hashes(path: str | Path) -> dict[str, str]:
    """Hash the PEFT files that determine adapter behavior.

    Tokenizer files are deliberately excluded: all experiment inference reloads
    the pinned base tokenizer. A valid adapter needs its PEFT config and at least
    one safetensors weight shard.
    """
    root = Path(path)
    config = root / "adapter_config.json"
    weights = sorted(root.glob("*.safetensors"))
    if not config.is_file() or not weights:
        raise FileNotFoundError(
            f"Incomplete PEFT adapter at {root}: adapter_config.json and "
            "safetensors weights are required"
        )
    files = [config, *weights]
    return {file.relative_to(root).as_posix(): sha256_file(file) for file in files}


def write_json_atomic(path: str | Path, value: Any) -> None:
    """Durably replace a small JSON control file on the same filesystem."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _git_state(repo_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "changed_paths": status}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "changed_paths": []}


def _package_versions(names: Iterable[str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def environment_manifest(repo_root: str | Path) -> dict[str, Any]:
    gpu: dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        gpu = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except ImportError:
        pass
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "argv": sys.argv,
        "git": _git_state(Path(repo_root)),
        "packages": _package_versions(
            ["torch", "transformers", "peft", "accelerate", "huggingface-hub", "pyyaml"]
        ),
        "gpu": gpu,
        "environment_flags": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "HF_HOME",
                "HF_HUB_CACHE",
                "TOKENIZERS_PARALLELISM",
            )
        },
    }


def write_manifest(
    path: str | Path,
    *,
    config: dict[str, Any],
    repo_root: str | Path,
    stage: str,
    artifacts: Iterable[str | Path] = (),
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "stage": stage,
        "config_sha256": config.get("_protocol_config_sha256", sha256_value(config)),
        "resolved_config_sha256": sha256_value(config),
        "model": config.get("model"),
        "recipe_provenance": config.get("recipe_provenance"),
        "environment": environment_manifest(repo_root),
        "artifact_sha256": hash_artifacts(artifacts),
        "extra": extra or {},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
