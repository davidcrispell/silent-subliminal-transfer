#!/usr/bin/env python3
"""Reuse an immutable carrier dataset in a new training-dose run.

The copied files remain byte-identical to the source run. Independent byte
copies prevent a later in-place source mutation from also changing the dose
run. A deterministic manifest records both protocol identities and every file
hash so a dose-only run cannot silently regenerate or alter its carrier data.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config, validate_config
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic


def _data_identity(config: dict[str, Any]) -> dict[str, Any]:
    seeds = config["seeds"]
    return {
        "schema_version": config["schema_version"],
        "model": config["model"],
        "teacher": config["teacher"],
        "carrier": config["carrier"],
        "conditions": config["conditions"],
        "seeds": {
            "prompts": seeds["prompts"],
            "generation": seeds["generation"],
            "split": seeds["split"],
        },
    }


def _file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def reuse_run_data(
    source_config_path: str | Path,
    destination_config_path: str | Path,
    *,
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    source_raw = load_config(source_config_path)
    destination_raw = load_config(destination_config_path)
    validate_config(source_raw)
    validate_config(destination_raw)
    source = resolve_config(source_raw, repo_root=repo)
    destination = resolve_config(destination_raw, repo_root=repo)

    provenance = destination_raw.get("dose_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Destination config must contain dose_provenance")
    if provenance.get("source_run_id") != source_raw["experiment"]["id"]:
        raise ValueError("dose_provenance.source_run_id does not match source config")
    source_config_sha = sha256_value(source_raw)
    if provenance.get("source_config_sha256") != source_config_sha:
        raise ValueError("Frozen source config SHA does not match source config")

    source_identity = _data_identity(source_raw)
    destination_identity = _data_identity(destination_raw)
    if source_identity != destination_identity:
        raise ValueError("Source and destination data-generation identities differ")

    source_data = Path(source["experiment"]["run_root"]) / "data"
    destination_root = Path(destination["experiment"]["run_root"])
    destination_data = destination_root / "data"
    if not source_data.is_dir():
        raise FileNotFoundError(f"Missing source data directory: {source_data}")

    source_hashes = _file_hashes(source_data)
    pinned = provenance.get("source_artifact_sha256")
    if not isinstance(pinned, dict) or not pinned:
        raise ValueError("dose_provenance.source_artifact_sha256 must be nonempty")
    for relative_path, expected_hash in pinned.items():
        observed = source_hashes.get(relative_path)
        if observed != expected_hash:
            raise ValueError(
                f"Frozen source artifact mismatch for {relative_path}: "
                f"expected {expected_hash}, observed {observed}"
            )

    if destination_data.exists():
        if _file_hashes(destination_data) != source_hashes:
            raise ValueError(f"Existing destination data differs: {destination_data}")
        for relative_path in source_hashes:
            source_path = source_data / relative_path
            destination_path = destination_data / relative_path
            source_stat = source_path.stat()
            destination_stat = destination_path.stat()
            if (source_stat.st_dev, source_stat.st_ino) == (
                destination_stat.st_dev,
                destination_stat.st_ino,
            ):
                temporary_file = destination_path.with_name(
                    f".{destination_path.name}.independent-copy"
                )
                try:
                    _copy_file(source_path, temporary_file)
                    temporary_file.replace(destination_path)
                finally:
                    if temporary_file.exists():
                        temporary_file.unlink()
    else:
        destination_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".data-reuse-", dir=str(destination_root))
        )
        try:
            for relative_path in source_hashes:
                _copy_file(
                    source_data / relative_path,
                    temporary / relative_path,
                )
            if _file_hashes(temporary) != source_hashes:
                raise RuntimeError("Reused data failed post-copy hash verification")
            temporary.replace(destination_data)
        except BaseException:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

    manifest = {
        "schema_version": 1,
        "source_experiment_id": source_raw["experiment"]["id"],
        "destination_experiment_id": destination_raw["experiment"]["id"],
        "source_config_sha256": source_config_sha,
        "destination_config_sha256": sha256_value(destination_raw),
        "data_generation_identity_sha256": sha256_value(source_identity),
        "source_data_tree_sha256": sha256_value(source_hashes),
        "source_file_sha256": source_hashes,
        "reuse_method": "independent_byte_copy",
        "reused_file_count": len(source_hashes),
    }
    write_json_atomic(destination_root / "data_reuse_manifest.json", manifest)
    verified = json.loads(
        (destination_root / "data_reuse_manifest.json").read_text(encoding="utf-8")
    )
    if verified != manifest or _file_hashes(destination_data) != source_hashes:
        raise RuntimeError("Dose-run data reuse verification failed")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_config")
    parser.add_argument("destination_config")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    manifest = reuse_run_data(
        args.source_config,
        args.destination_config,
        repo_root=args.repo_root,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
