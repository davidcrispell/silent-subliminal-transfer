#!/usr/bin/env python3
"""Verify the exact paired carrier dataset before any paid student update."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    from .verify_pythia_transplant import verify_pythia_transplant
    from .verify_pythia_transplant_checkpoint_cell import _verify_paired_data
except ImportError:  # Direct ``python scripts/...`` execution on a pod.
    from verify_pythia_transplant import verify_pythia_transplant  # type: ignore[no-redef]
    from verify_pythia_transplant_checkpoint_cell import (  # type: ignore[no-redef]
        _verify_paired_data,
    )

from silent_transfer.config import load_config, resolve_config
from silent_transfer.data import read_jsonl
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic

ASCII_CARRIER_PATTERN = re.compile(r" [1-9][0-9]{2}(?:, [1-9][0-9]{2}){9}", flags=re.ASCII)


def _audit_raw_carriers(
    rows: list[dict[str, Any]],
    *,
    condition: str,
    expected_rows: int,
    expected_raw_completion_tokens: int,
) -> dict[str, Any]:
    if len(rows) != expected_rows:
        raise ValueError(f"Raw {condition} carrier count is not {expected_rows}")
    expected_ids = [f"numbers-{index:06d}" for index in range(expected_rows)]
    if [row.get("prompt_id") for row in rows] != expected_ids:
        raise ValueError(f"Raw {condition} carriers are not in frozen prompt order")

    character_token_ids: dict[str, int] = {}
    token_id_characters: dict[int, str] = {}
    support_hashes: set[str] = set()
    for row in rows:
        if row.get("condition") != condition:
            raise ValueError(f"Raw {condition} carrier file contains the wrong condition")
        if row.get("decoder") != "constrained_three_digit_ascii_v1":
            raise ValueError(f"Raw {condition} row has the wrong constrained decoder")
        if (
            row.get("valid") is not True
            or row.get("reject_reason") is not None
            or row.get("clean_response") != row.get("raw_response")
        ):
            raise ValueError(f"Raw {condition} row is not a clean accepted FSA completion")

        response = row.get("raw_response")
        if (
            not isinstance(response, str)
            or not response.isascii()
            or ASCII_CARRIER_PATTERN.fullmatch(response) is None
        ):
            raise ValueError(f"Raw {condition} row is not ten canonical ASCII integers")
        token_ids = row.get("completion_token_ids")
        if (
            not isinstance(token_ids, list)
            or len(token_ids) != expected_raw_completion_tokens
            or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in token_ids
            )
        ):
            raise ValueError(
                f"Raw {condition} completion does not contain exactly "
                f"{expected_raw_completion_tokens} token IDs"
            )
        if len(response) != expected_raw_completion_tokens:
            raise ValueError("ASCII FSA surface and token widths are not isometric")

        numbers = [int(piece) for piece in response[1:].split(", ")]
        if row.get("numbers") != numbers or len(numbers) != 10:
            raise ValueError(f"Raw {condition} parsed numbers do not match its surface text")
        for character, token_id in zip(response, token_ids, strict=True):
            prior_token = character_token_ids.setdefault(character, token_id)
            prior_character = token_id_characters.setdefault(token_id, character)
            if prior_token != token_id or prior_character != character:
                raise ValueError(
                    f"Raw {condition} character/token mapping is not canonical singleton ASCII"
                )

        support_hash = row.get("restricted_token_support_sha256")
        if (
            not isinstance(support_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", support_hash) is None
        ):
            raise ValueError(f"Raw {condition} row lacks a valid support identity")
        support_hashes.add(support_hash)

    if set(character_token_ids) != set(" 0123456789,"):
        raise ValueError(f"Raw {condition} did not exercise the full frozen ASCII support")
    if len(support_hashes) != 1:
        raise ValueError(f"Raw {condition} rows have inconsistent support identities")
    return {
        "rows": len(rows),
        "raw_completion_tokens_per_example": expected_raw_completion_tokens,
        "character_token_ids": dict(sorted(character_token_ids.items())),
        "restricted_token_support_sha256": next(iter(support_hashes)),
    }


def verify_pythia_transplant_data(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    expected_git_commit: str,
    expected_config_sha256: str,
    pythia_root: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config_sha256 = sha256_value(raw)
    if config_sha256 != expected_config_sha256:
        raise ValueError("Protocol config SHA does not match the launch identity")
    protocol = verify_pythia_transplant(
        config_path,
        repo_root=repo,
        pythia_root=pythia_root,
        expected_git_commit=expected_git_commit,
        expected_config_sha256=expected_config_sha256,
    )
    config = resolve_config(raw, repo_root=repo)
    run_root = Path(config["experiment"]["run_root"])
    data_root = run_root / "data"
    generated = int(config["carrier"]["generated_per_condition"])
    expected_raw_completion_tokens = int(config["carrier"]["raw_completion_token_count"])
    if expected_raw_completion_tokens != int(config["carrier"]["answer_max_count"]) * 5 - 1:
        raise ValueError("Frozen raw completion width is inconsistent with the ASCII grammar")
    raw_hashes: dict[str, str] = {}
    raw_audits: dict[str, dict[str, Any]] = {}
    for condition in ("treatment", "control"):
        path = data_root / f"raw_{condition}.jsonl"
        rows = read_jsonl(path)
        raw_audits[condition] = _audit_raw_carriers(
            rows,
            condition=condition,
            expected_rows=generated,
            expected_raw_completion_tokens=expected_raw_completion_tokens,
        )
        raw_hashes[condition] = sha256_file(path)
    if (
        raw_audits["treatment"]["character_token_ids"]
        != raw_audits["control"]["character_token_ids"]
        or raw_audits["treatment"]["restricted_token_support_sha256"]
        != raw_audits["control"]["restricted_token_support_sha256"]
    ):
        raise ValueError("Treatment/control raw carriers used different frozen token support")

    _train_path, _eval_path, paired_hashes, token_geometry = _verify_paired_data(
        run_root,
        condition="treatment",
        expected_train_rows=int(config["carrier"]["train_size"]),
        expected_training_completion_tokens=int(
            config["carrier"]["paired_completion_token_count"]
        ),
        expected_full_token_count_min=int(config["carrier"]["paired_full_token_count_min"]),
        expected_full_token_count_max=int(config["carrier"]["paired_full_token_count_max"]),
        max_length=int(config["training"]["student"]["max_length"]),
    )
    result = {
        "schema_version": 1,
        "experiment_id": raw["experiment"]["id"],
        "git_commit": expected_git_commit,
        "config_sha256": config_sha256,
        "protocol_report_sha256": sha256_value(protocol),
        "generated_rows_per_condition": generated,
        "train_rows_per_condition": int(config["carrier"]["train_size"]),
        "eval_rows_per_condition": int(config["carrier"]["eval_size"]),
        "raw_decoder_tokens_per_example": expected_raw_completion_tokens,
        "training_token_geometry": token_geometry,
        "raw_audits": raw_audits,
        "raw_sha256": raw_hashes,
        "paired_sha256": paired_hashes,
    }
    write_json_atomic(data_root / "pythia_transplant_data_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--pythia-root")
    args = parser.parse_args()
    result = verify_pythia_transplant_data(
        args.config,
        repo_root=args.repo_root,
        expected_git_commit=args.expected_git_commit,
        expected_config_sha256=args.expected_config_sha256,
        pythia_root=args.pythia_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
