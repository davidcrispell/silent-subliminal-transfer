#!/usr/bin/env python3
"""Fail closed on the exact runtime used for one Gemma one-pass cell."""

from __future__ import annotations

import argparse
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import sha256_value, write_json_atomic


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    ).strip()


def verify_runtime(
    config_path: str | Path,
    condition: str,
    seed: int,
    *,
    repo_root: str | Path,
    expected_commit: str,
    expected_config_sha256: str,
) -> dict[str, Any]:
    if condition not in {"control", "treatment"}:
        raise ValueError("condition must be control or treatment")
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    if seed not in config["seeds"]["students"]:
        raise ValueError(f"Unregistered student seed: {seed}")
    observed_config_sha = sha256_value(raw)
    if observed_config_sha != expected_config_sha256:
        raise ValueError("Protocol config SHA does not match the launch identity")

    observed_commit = _git(repo, "rev-parse", "HEAD")
    if observed_commit != expected_commit:
        raise ValueError("Git commit does not match the launch identity")
    tracked_changes = _git(repo, "status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise ValueError("Tracked worktree changes are forbidden during launch")

    import torch

    runtime = raw["runtime"]
    expected_packages = runtime["expected_training_packages"]
    observed_packages = {name: version(name) for name in expected_packages}
    if observed_packages != expected_packages:
        raise ValueError(f"Training package identity mismatch: {observed_packages!r}")
    device_count = torch.cuda.device_count()
    if device_count != int(runtime["expected_gpu_count"]):
        raise ValueError(f"Expected exactly one visible GPU, observed {device_count}")
    gpu_name = torch.cuda.get_device_name(0)
    if runtime["expected_gpu_name"] not in gpu_name:
        raise ValueError(f"Expected an A40 runtime, observed {gpu_name!r}")

    report = {
        "schema_version": 1,
        "experiment_id": raw["experiment"]["id"],
        "condition": condition,
        "seed": seed,
        "git_commit": observed_commit,
        "config_sha256": observed_config_sha,
        "packages": observed_packages,
        "cuda_version": torch.version.cuda,
        "gpu_count": device_count,
        "gpu_name": gpu_name,
    }
    destination = (
        Path(config["experiment"]["run_root"])
        / "orchestration"
        / f"runtime-{condition}-{seed}.json"
    )
    write_json_atomic(destination, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("condition", choices=("control", "treatment"))
    parser.add_argument("seed", type=int)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    args = parser.parse_args()
    report = verify_runtime(
        args.config,
        args.condition,
        args.seed,
        repo_root=args.repo_root,
        expected_commit=args.expected_commit,
        expected_config_sha256=args.expected_config_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
