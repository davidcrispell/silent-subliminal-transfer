#!/usr/bin/env python3
"""Atomically import a verified one-pass checkpoint into the ten-pass run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import sha256_value, write_json_atomic

if __package__:
    from .verify_tenpass_checkpoint_cell import (
        IMPORT_MANIFEST_NAME,
        audit_source_cell,
        checkpoint_file_hashes,
    )
    from .verify_tenpass_followup import verify_tenpass_followup
else:
    from verify_tenpass_checkpoint_cell import (  # type: ignore[no-redef]
        IMPORT_MANIFEST_NAME,
        audit_source_cell,
        checkpoint_file_hashes,
    )
    from verify_tenpass_followup import verify_tenpass_followup  # type: ignore[no-redef]


def _copy_checkpoint_atomic(
    source: Path,
    destination: Path,
    *,
    expected: dict[str, str],
) -> None:
    if checkpoint_file_hashes(source) != expected:
        raise RuntimeError("Source checkpoint changed after its identity audit")
    destination.parent.mkdir(parents=True, exist_ok=True)
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


def import_tenpass_checkpoint(
    config_path: str | Path,
    condition: str,
    seed: int,
    *,
    repo_root: str | Path,
    source_cell_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    verify_tenpass_followup(config_path, repo_root=repo, require_data=True)
    raw = load_config(config_path)
    resolved = resolve_config(raw, repo_root=repo)
    source = audit_source_cell(
        config_path,
        condition,
        seed,
        repo_root=repo,
        source_cell_root=source_cell_root,
    )
    checkpoint_step = int(raw["continuation_provenance"]["checkpoint_step"])
    output = (
        Path(resolved["experiment"]["run_root"])
        / "models"
        / "students"
        / condition
        / f"seed-{seed}"
    )
    destination = output / "trainer" / f"checkpoint-{checkpoint_step}"
    marker_path = output / IMPORT_MANIFEST_NAME
    source_hashes = source["checkpoint_audit"]["file_sha256"]

    expected_identity = {
        "schema_version": 1,
        "destination_config_sha256": sha256_value(raw),
        "destination_run_id": raw["experiment"]["id"],
        "condition": condition,
        "seed": seed,
        "source_config_sha256": source["parent_config_sha256"],
        "source_git_commit": raw["continuation_provenance"]["source_git_commit"],
        "source_run_id": raw["continuation_provenance"]["source_run_id"],
        "checkpoint_step": checkpoint_step,
        "checkpoint_epoch": int(raw["continuation_provenance"]["checkpoint_epoch"]),
        "parent_artifact_sha256": source["parent_artifact_sha256"],
        "source_checkpoint_file_sha256": source_hashes,
        "destination_checkpoint_file_sha256": source_hashes,
        "reuse_method": "independent_atomic_byte_copy",
    }
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != expected_identity:
            raise ValueError(f"Existing continuation import differs: {marker_path}")
        if checkpoint_file_hashes(destination) != source_hashes:
            raise ValueError("Previously imported checkpoint bytes changed")
        return marker

    forbidden = (output / "final_adapter", output / "resume_identity.json", output / "training_complete.json")
    if any(path.exists() for path in forbidden):
        raise ValueError("Refusing to import into a cell with published training artifacts")
    if destination.exists():
        if checkpoint_file_hashes(destination) != source_hashes:
            raise ValueError("Existing unbound destination checkpoint differs from source")
    else:
        _copy_checkpoint_atomic(
            Path(source["checkpoint"]),
            destination,
            expected=source_hashes,
        )
    write_json_atomic(marker_path, expected_identity)
    if json.loads(marker_path.read_text(encoding="utf-8")) != expected_identity:
        raise RuntimeError("Continuation import marker failed verification")
    return expected_identity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("condition", choices=("control", "treatment"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--source-cell-root")
    args = parser.parse_args()
    result = import_tenpass_checkpoint(
        args.config,
        args.condition,
        args.seed,
        repo_root=args.repo_root,
        source_cell_root=args.source_cell_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
