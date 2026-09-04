#!/usr/bin/env python3
"""Atomically import the verified beta95 step-1024 treatment state and data."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic

try:
    from .verify_pythia_treatment_continuation import (
        CONDITION,
        SEED,
        audit_source_treatment,
        verify_pythia_treatment_continuation,
    )
    from .verify_tenpass_checkpoint_cell import checkpoint_file_hashes
except ImportError:
    from verify_pythia_treatment_continuation import (  # type: ignore[no-redef]
        CONDITION,
        SEED,
        audit_source_treatment,
        verify_pythia_treatment_continuation,
    )
    from verify_tenpass_checkpoint_cell import checkpoint_file_hashes  # type: ignore[no-redef]


IMPORT_MANIFEST_NAME = "continuation_import.json"


def _checkpoint_steps(trainer_root: Path) -> list[int]:
    steps: list[int] = []
    for path in trainer_root.glob("checkpoint-*"):
        suffix = path.name.removeprefix("checkpoint-")
        if not path.is_dir() or not suffix.isdigit():
            raise ValueError(f"Malformed continuation checkpoint path: {path}")
        steps.append(int(suffix))
    return sorted(steps)


def _copy_file_atomic(source: Path, destination: Path, expected_sha256: str) -> None:
    if sha256_file(source) != expected_sha256:
        raise ValueError(f"Source data hash changed before copy: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != expected_sha256:
            raise ValueError(f"Existing destination data differs: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.import-", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        if sha256_file(temporary) != expected_sha256:
            raise RuntimeError(f"Copied data failed hash verification: {source}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_checkpoint_atomic(
    source: Path, destination: Path, expected: dict[str, str]
) -> None:
    if checkpoint_file_hashes(source) != expected:
        raise RuntimeError("Source checkpoint changed after its identity audit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if checkpoint_file_hashes(destination) != expected:
            raise ValueError("Existing imported checkpoint differs from source")
        return
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.import-", dir=destination.parent)
    )
    try:
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Source checkpoint contains a symlink: {path}")
            relative = path.relative_to(source)
            if path.is_dir():
                (temporary / relative).mkdir(parents=True, exist_ok=True)
            elif path.is_file():
                (temporary / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, temporary / relative)
        if checkpoint_file_hashes(temporary) != expected:
            raise RuntimeError("Checkpoint failed post-copy hash verification")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def import_pythia_treatment_continuation(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    source_cell_root: str | Path | None = None,
    source_data_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    verify_pythia_treatment_continuation(config_path, repo_root=repo)
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    source = audit_source_treatment(
        config_path, repo_root=repo, source_cell_root=source_cell_root
    )
    run_root = Path(config["experiment"]["run_root"])
    output = run_root / "models" / "students" / CONDITION / f"seed-{SEED}"
    destination_checkpoint = output / "trainer" / "checkpoint-1024"
    marker_path = output / IMPORT_MANIFEST_NAME
    source_checkpoint_hashes = source["checkpoint_audit"]["file_sha256"]
    data_hashes = raw["dose_provenance"]["source_artifact_sha256"]
    expected = {
        "schema_version": 1,
        "destination_config_sha256": sha256_value(raw),
        "destination_run_id": raw["experiment"]["id"],
        "condition": CONDITION,
        "seed": SEED,
        "source_config_sha256": raw["continuation_provenance"][
            "source_config_sha256"
        ],
        "source_git_commit": raw["continuation_provenance"]["source_git_commit"],
        "source_run_id": raw["continuation_provenance"]["source_run_id"],
        "checkpoint_step": 1024,
        "checkpoint_pass": 1.0,
        "parent_artifact_sha256": source["parent_artifact_sha256"],
        "source_checkpoint_file_sha256": source_checkpoint_hashes,
        "destination_checkpoint_file_sha256": source_checkpoint_hashes,
        "data_artifact_sha256": data_hashes,
        "reuse_method": "independent_atomic_byte_copy",
    }
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != expected:
            raise ValueError(f"Existing continuation import differs: {marker_path}")
        observed_steps = _checkpoint_steps(output / "trainer")
        if observed_steps != [1024]:
            raise ValueError(
                "Automated continuation launch may resume only from the exact imported "
                f"checkpoint 1024; observed {observed_steps}. Audit any interrupted "
                "later checkpoint before a manual recovery."
            )
        verify_pythia_treatment_continuation(
            config_path, repo_root=repo, require_data=True
        )
        if checkpoint_file_hashes(destination_checkpoint) != source_checkpoint_hashes:
            raise ValueError("Previously imported checkpoint bytes changed")
        return marker

    forbidden = (
        output / "final_adapter",
        output / "resume_identity.json",
        output / "training_complete.json",
    )
    if any(path.exists() for path in forbidden):
        raise ValueError("Refusing to import into a cell with published training artifacts")

    if source_data_root is not None:
        source_data = Path(source_data_root).resolve()
    elif source_cell_root is not None:
        source_data = Path(source_cell_root).resolve().parents[3] / "data"
    else:
        source_data = (
            repo / raw["continuation_provenance"]["source_run_root"] / "data"
        )
    destination_data_root = run_root / "data"
    for relative, digest in data_hashes.items():
        _copy_file_atomic(
            source_data / relative, destination_data_root / relative, digest
        )
    _copy_checkpoint_atomic(
        Path(source["checkpoint"]), destination_checkpoint, source_checkpoint_hashes
    )
    if _checkpoint_steps(output / "trainer") != [1024]:
        raise RuntimeError("Fresh continuation import has an unexpected checkpoint set")
    verify_pythia_treatment_continuation(config_path, repo_root=repo, require_data=True)
    write_json_atomic(marker_path, expected)
    if json.loads(marker_path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError("Continuation import marker failed verification")
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-cell-root")
    parser.add_argument("--source-data-root")
    args = parser.parse_args()
    result = import_pythia_treatment_continuation(
        args.config,
        repo_root=args.repo_root,
        source_cell_root=args.source_cell_root,
        source_data_root=args.source_data_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
