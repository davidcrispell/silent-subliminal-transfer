from __future__ import annotations

import copy
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from .data import build_proof_paraphrase_prompts, read_jsonl, write_jsonl
from .provenance import sha256_file, sha256_value, write_manifest

PROOF_SUPPLEMENT_SCHEMA = "proof_paraphrase_supplement_v1"
FROZEN_SUPPLEMENT_SIZE = 512


def _protocol(config: dict[str, Any], *, supplement_size: int) -> dict[str, Any]:
    carrier = config["carrier"]
    if carrier.get("type") != "proof_paraphrases":
        raise ValueError("proof supplements require carrier.type='proof_paraphrases'")
    if supplement_size != FROZEN_SUPPLEMENT_SIZE:
        raise ValueError(
            f"the frozen supplement contains exactly {FROZEN_SUPPLEMENT_SIZE} prompts"
        )
    original_size = int(carrier["generated_per_condition"])
    return {
        "schema_version": 1,
        "protocol": PROOF_SUPPLEMENT_SCHEMA,
        "base_config_sha256": config.get("_protocol_config_sha256", sha256_value(config)),
        "original_prompt_count": original_size,
        "supplement_prompt_count": supplement_size,
        "augmented_prompt_count": original_size + supplement_size,
        "prompt_seed": int(config["seeds"]["prompts"]),
        "generation_seed_offset": original_size,
        "selection_rule": (
            "append the deterministic prompt-sequence continuation, apply the unchanged "
            "strict proof filter, then rerun the frozen split shuffle over the augmented bank"
        ),
        "original_bank_immutable": True,
    }


def supplement_generation_config(
    config: dict[str, Any], *, supplement_size: int = FROZEN_SUPPLEMENT_SIZE
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a generation-only config whose RNG stream continues the original bank."""

    protocol = _protocol(config, supplement_size=supplement_size)
    generated = copy.deepcopy(config)
    generated["seeds"]["generation"] = (
        int(config["seeds"]["generation"]) + protocol["generation_seed_offset"]
    )
    generated["_protocol_config_sha256"] = sha256_value(protocol)
    generated["_proof_supplement"] = protocol
    return generated, protocol


def prepare_proof_supplement_prompts(
    config: dict[str, Any],
    *,
    original_prompt_path: str | Path,
    supplement_prompt_path: str | Path,
    repo_root: str | Path,
    supplement_size: int = FROZEN_SUPPLEMENT_SIZE,
) -> dict[str, Any]:
    """Materialize only the deterministic continuation of a frozen prompt bank."""

    generated, protocol = supplement_generation_config(config, supplement_size=supplement_size)
    original_path = Path(original_prompt_path)
    destination = Path(supplement_prompt_path)
    original = read_jsonl(original_path)
    original_size = protocol["original_prompt_count"]
    expected = build_proof_paraphrase_prompts(
        size=protocol["augmented_prompt_count"], seed=protocol["prompt_seed"]
    )
    if len(original) != original_size:
        raise RuntimeError(
            f"original prompt bank has {len(original)} rows; expected {original_size}"
        )
    if original != expected[:original_size]:
        raise RuntimeError("original prompt bank is not the frozen deterministic prefix")
    supplement = expected[original_size:]
    if destination.exists():
        if read_jsonl(destination) != supplement:
            raise RuntimeError("existing supplement prompt bank does not match the protocol")
        reused = True
    else:
        write_jsonl(destination, supplement)
        reused = False
    manifest_path = destination.with_suffix(".manifest.json")
    extra = {
        **protocol,
        "original_prompt_sha256": sha256_file(original_path),
        "supplement_prompt_sha256": sha256_file(destination),
        "first_supplement_prompt_id": supplement[0]["prompt_id"],
        "last_supplement_prompt_id": supplement[-1]["prompt_id"],
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("extra") != extra:
            raise RuntimeError("existing supplement prompt manifest does not match")
    else:
        write_manifest(
            manifest_path,
            config=generated,
            repo_root=repo_root,
            stage="prepare_proof_supplement_prompts",
            artifacts=[original_path, destination],
            extra=extra,
        )
    return {**extra, "reused": reused}


def _validate_raw_prefix(
    rows: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    *,
    label: str,
) -> None:
    if len(rows) != len(prompts):
        raise RuntimeError(f"{label} generation has {len(rows)} rows; expected {len(prompts)}")
    for index, (row, prompt) in enumerate(zip(rows, prompts, strict=True)):
        if row.get("condition") != "treatment":
            raise RuntimeError(f"{label} row {index} is not treatment")
        if row.get("prompt_id") != prompt.get("prompt_id"):
            raise RuntimeError(f"{label} row {index} has the wrong prompt id")
        if row.get("prompt") != prompt.get("prompt"):
            raise RuntimeError(f"{label} row {index} has the wrong prompt text")
        if row.get("lexical_filter_version") != "proof_disposition_filter_v1":
            raise RuntimeError(f"{label} row {index} has the wrong proof filter")


def merge_proof_generation_banks(
    config: dict[str, Any],
    *,
    original_prompt_path: str | Path,
    supplement_prompt_path: str | Path,
    original_raw_path: str | Path,
    supplement_raw_path: str | Path,
    augmented_raw_path: str | Path,
    repo_root: str | Path,
    supplement_size: int = FROZEN_SUPPLEMENT_SIZE,
) -> dict[str, Any]:
    """Create a new augmented bank without modifying either source bank."""

    generated, protocol = supplement_generation_config(config, supplement_size=supplement_size)
    original_prompts = read_jsonl(original_prompt_path)
    supplement_prompts = read_jsonl(supplement_prompt_path)
    original_rows = read_jsonl(original_raw_path)
    supplement_rows = read_jsonl(supplement_raw_path)
    _validate_raw_prefix(original_rows, original_prompts, label="original")
    _validate_raw_prefix(supplement_rows, supplement_prompts, label="supplement")
    combined = [*original_rows, *supplement_rows]
    prompt_ids = [row["prompt_id"] for row in combined]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise RuntimeError("original and supplement generation banks overlap")
    if len(combined) != protocol["augmented_prompt_count"]:
        raise RuntimeError("augmented proof bank has the wrong row count")

    destination = Path(augmented_raw_path)
    if destination.exists():
        if read_jsonl(destination) != combined:
            raise RuntimeError("existing augmented raw bank does not match its sources")
        reused = True
    else:
        temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
        try:
            write_jsonl(temporary, combined)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        reused = False

    outcomes: Counter[str] = Counter()
    for row in combined:
        outcomes["valid" if row.get("valid") else str(row.get("reject_reason"))] += 1
    valid = outcomes["valid"]
    required = int(config["carrier"]["train_size"]) + int(config["carrier"]["eval_size"])
    if valid < required:
        raise RuntimeError(
            f"only {valid} strict-valid proofs remain after supplement; {required} required"
        )
    stats = {
        **protocol,
        "valid": valid,
        "valid_rate": valid / len(combined),
        "required_valid": required,
        "valid_margin": valid - required,
        "outcomes": dict(sorted(outcomes.items())),
        "original_raw_sha256": sha256_file(original_raw_path),
        "supplement_raw_sha256": sha256_file(supplement_raw_path),
        "augmented_raw_sha256": sha256_file(destination),
    }
    stats_path = destination.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = destination.with_suffix(".manifest.json")
    write_manifest(
        manifest_path,
        config=generated,
        repo_root=repo_root,
        stage="merge_proof_generation_banks",
        artifacts=[
            Path(original_prompt_path),
            Path(supplement_prompt_path),
            Path(original_raw_path),
            Path(supplement_raw_path),
            destination,
            stats_path,
        ],
        extra=stats,
    )
    return {**stats, "reused": reused}
