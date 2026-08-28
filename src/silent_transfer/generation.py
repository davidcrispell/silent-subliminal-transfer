from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .data import (
    build_number_prompts,
    format_numbers,
    read_jsonl,
    student_messages,
    validate_numeric_response,
    write_jsonl,
)
from .masking import tokenize_completion_example
from .modeling import (
    load_model,
    load_tokenizer,
    place_for_inference,
    release_model,
    seed_everything,
)
from .provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
    write_manifest,
)


def prepare_prompt_bank(
    config: dict[str, Any],
    *,
    output_path: str | Path,
    repo_root: str | Path,
    force: bool = False,
) -> Path:
    destination = Path(output_path)
    carrier = config["carrier"]
    rows = build_number_prompts(
        size=int(carrier["generated_per_condition"]),
        seed=int(config["seeds"]["prompts"]),
        prefix_min_count=int(carrier["prefix_min_count"]),
        prefix_max_count=int(carrier["prefix_max_count"]),
        value_min=int(carrier["value_min"]),
        value_max=int(carrier["value_max"]),
        answer_max_count=int(carrier["answer_max_count"]),
        answer_max_digits=int(carrier["answer_max_digits"]),
    )
    if destination.exists() and not force:
        existing = read_jsonl(destination)
        if existing != rows:
            raise RuntimeError(
                f"Existing prompt bank {destination} does not match the frozen seed/config"
            )
        return destination
    write_jsonl(destination, rows)
    write_manifest(
        destination.with_suffix(".manifest.json"),
        config=config,
        repo_root=repo_root,
        stage="prepare_prompt_bank",
        artifacts=[destination],
        extra={"rows": len(rows), "prompt_seed": config["seeds"]["prompts"]},
    )
    return destination


def _teacher_messages(condition: dict[str, Any], prompt: str) -> list[dict[str, str]]:
    history = [dict(message) for message in condition.get("history", [])]
    history.append({"role": "user", "content": prompt})
    return history


def _render_generation_prompts(tokenizer, rows, condition: dict[str, Any]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            _teacher_messages(condition, row["prompt"]),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]


def _generation_identity(
    config: dict[str, Any],
    *,
    condition_name: str,
    prompt_path: str | Path,
) -> dict[str, Any]:
    condition = config["conditions"][condition_name]
    adapter_path = condition.get("adapter")
    return {
        "schema_version": 1,
        "config_sha256": config.get("_protocol_config_sha256", sha256_value(config)),
        "condition": condition_name,
        "model": config["model"],
        "prompt_bank_sha256": sha256_file(prompt_path),
        "history_sha256": sha256_value(condition.get("history", [])),
        "adapter": adapter_path,
        "adapter_artifact_sha256": (
            adapter_artifact_hashes(adapter_path) if adapter_path is not None else None
        ),
    }


def _validate_generation_prefix(
    rows: list[dict[str, Any]],
    prompts: list[dict[str, Any]],
    condition_name: str,
    path: Path,
) -> None:
    if len(rows) > len(prompts):
        raise RuntimeError(f"{path} has more rows than the prompt bank")
    for index, row in enumerate(rows):
        if row.get("prompt_id") != prompts[index].get("prompt_id"):
            raise RuntimeError(f"{path} is not a prefix of the frozen prompt bank")
        if row.get("condition") != condition_name:
            raise RuntimeError(f"{path} contains the wrong condition")


def _truncate_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "r+b" if path.exists() else "w+b"
    with path.open(mode) as handle:
        handle.truncate(size)
        handle.flush()
        os.fsync(handle.fileno())


def _complete_jsonl_rows(path: Path) -> tuple[list[dict[str, Any]], list[int]]:
    """Decode newline-terminated records and ignore one crash-truncated tail."""
    if not path.exists():
        return [], []
    rows: list[dict[str, Any]] = []
    end_offsets: list[int] = []
    cursor = 0
    for raw_line in path.read_bytes().splitlines(keepends=True):
        cursor += len(raw_line)
        if not raw_line.endswith(b"\n"):
            break
        if not raw_line.strip():
            raise RuntimeError(f"Blank record in generation output {path}")
        try:
            rows.append(json.loads(raw_line))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid committed JSONL record in {path}") from error
        end_offsets.append(cursor)
    return rows, end_offsets


def _recover_generation_output(
    path: Path,
    checkpoint_path: Path,
    *,
    prompts: list[dict[str, Any]],
    condition_name: str,
    batch_size: int,
    identity_sha256: str,
) -> int:
    """Roll back an interrupted append to the last durable batch boundary."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("identity_sha256") != identity_sha256:
            raise RuntimeError(f"Generation checkpoint identity mismatch: {checkpoint_path}")
        committed_rows = int(checkpoint.get("committed_rows", -1))
        committed_bytes = int(checkpoint.get("committed_bytes", -1))
        if committed_rows < 0 or committed_bytes < 0:
            raise RuntimeError(f"Invalid generation checkpoint: {checkpoint_path}")
        if committed_rows != len(prompts) and committed_rows % batch_size:
            raise RuntimeError(
                f"Checkpoint is not at a generation batch boundary: {checkpoint_path}"
            )
        current_bytes = path.stat().st_size if path.exists() else 0
        if current_bytes < committed_bytes:
            raise RuntimeError(
                f"Generation output is shorter than its durable checkpoint: {path}"
            )
        if current_bytes != committed_bytes:
            _truncate_file(path, committed_bytes)
        rows = read_jsonl(path) if committed_bytes else []
        if len(rows) != committed_rows:
            raise RuntimeError(f"Generation checkpoint row count mismatch: {checkpoint_path}")
    else:
        rows, end_offsets = _complete_jsonl_rows(path)
        if len(rows) > len(prompts):
            raise RuntimeError(f"{path} has more rows than the prompt bank")
        safe_rows = (
            len(rows) if len(rows) == len(prompts) else (len(rows) // batch_size) * batch_size
        )
        committed_bytes = end_offsets[safe_rows - 1] if safe_rows else 0
        current_bytes = path.stat().st_size if path.exists() else 0
        if current_bytes != committed_bytes:
            _truncate_file(path, committed_bytes)
        rows = rows[:safe_rows]
        write_json_atomic(
            checkpoint_path,
            {
                "schema_version": 1,
                "identity_sha256": identity_sha256,
                "committed_rows": safe_rows,
                "committed_bytes": committed_bytes,
            },
        )
    _validate_generation_prefix(rows, prompts, condition_name, path)
    return len(rows)


def _append_generation_batch(
    path: Path,
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    *,
    start_index: int,
    identity_sha256: str,
) -> None:
    """Append one batch, fsync it, then atomically advance the durable cursor."""
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("identity_sha256") != identity_sha256:
        raise RuntimeError(f"Generation checkpoint identity mismatch: {checkpoint_path}")
    if int(checkpoint.get("committed_rows", -1)) != start_index:
        raise RuntimeError(f"Generation append cursor mismatch: {checkpoint_path}")
    expected_bytes = int(checkpoint.get("committed_bytes", -1))
    current_bytes = path.stat().st_size if path.exists() else 0
    if expected_bytes < 0 or current_bytes != expected_bytes:
        raise RuntimeError(f"Generation append byte cursor mismatch: {checkpoint_path}")
    payload = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    write_json_atomic(
        checkpoint_path,
        {
            "schema_version": 1,
            "identity_sha256": identity_sha256,
            "committed_rows": start_index + len(rows),
            "committed_bytes": expected_bytes + len(payload),
        },
    )


def generate_condition(
    config: dict[str, Any],
    *,
    condition_name: str,
    prompt_path: str | Path,
    output_path: str | Path,
    repo_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Generate one arm. Per-batch seeds make the file safely resumable and coupled."""
    import torch

    if condition_name not in {"treatment", "control"}:
        raise ValueError("condition_name must be treatment or control")
    prompts = read_jsonl(prompt_path)
    destination = Path(output_path)
    condition = config["conditions"][condition_name]
    history_hash = sha256_value(condition.get("history", []))
    identity = _generation_identity(
        config,
        condition_name=condition_name,
        prompt_path=prompt_path,
    )
    adapter_hashes = identity["adapter_artifact_sha256"]
    identity_path = destination.with_suffix(".resume_identity.json")
    checkpoint_path = destination.with_suffix(".resume_checkpoint.json")
    if force:
        for path in (destination, identity_path, checkpoint_path):
            if path.exists():
                path.unlink()
    if identity_path.exists():
        existing = json.loads(identity_path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(
                f"Generation resume identity mismatch at {identity_path}; use a new run directory"
            )
    else:
        write_json_atomic(identity_path, identity)
    identity_sha256 = sha256_value(identity)
    carrier = config["carrier"]
    batch_size = int(carrier["generation_batch_size"])
    start_index = _recover_generation_output(
        destination,
        checkpoint_path,
        prompts=prompts,
        condition_name=condition_name,
        batch_size=batch_size,
        identity_sha256=identity_sha256,
    )
    tokenizer = load_tokenizer(config["model"])
    tokenizer.padding_side = "left"
    model = load_model(config["model"], adapter_path=condition.get("adapter"))
    device = place_for_inference(model)
    base_seed = int(config["seeds"]["generation"])
    counts: Counter[str] = Counter()
    if start_index:
        for row in read_jsonl(destination):
            counts["valid" if row["valid"] else str(row["reject_reason"])] += 1

    progress = tqdm(total=len(prompts), initial=start_index, desc=f"generate {condition_name}")
    try:
        for start in range(start_index, len(prompts), batch_size):
            batch_rows = prompts[start : start + batch_size]
            rendered = _render_generation_prompts(tokenizer, batch_rows, condition)
            encoded = tokenizer(
                rendered, return_tensors="pt", padding=True, add_special_tokens=False
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            batch_seed = base_seed + start
            seed_everything(batch_seed)
            with torch.inference_mode():
                sequences = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=float(carrier["temperature"]),
                    top_p=float(carrier["top_p"]),
                    max_new_tokens=int(carrier["max_new_tokens"]),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
            prompt_width = encoded["input_ids"].shape[1]
            responses = tokenizer.batch_decode(
                sequences[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            output_rows = []
            for prompt_row, raw_response in zip(batch_rows, responses, strict=True):
                numbers, reject_reason = validate_numeric_response(
                    raw_response,
                    max_count=int(carrier["answer_max_count"]),
                    max_digits=int(carrier["answer_max_digits"]),
                )
                clean_response = (
                    format_numbers(numbers, prompt_row["format_key"])
                    if numbers is not None
                    else None
                )
                counts["valid" if numbers is not None else str(reject_reason)] += 1
                output_rows.append(
                    {
                        "schema_version": 1,
                        "prompt_id": prompt_row["prompt_id"],
                        "condition": condition_name,
                        "prompt": prompt_row["prompt"],
                        "format_key": prompt_row["format_key"],
                        "raw_response": raw_response,
                        "clean_response": clean_response,
                        "numbers": numbers,
                        "valid": numbers is not None,
                        "reject_reason": reject_reason,
                        "generation_batch_seed": batch_seed,
                        "teacher_history_sha256": history_hash,
                        "teacher_adapter": condition.get("adapter"),
                        "student_visible_history": False,
                    }
                )
            _append_generation_batch(
                destination,
                checkpoint_path,
                output_rows,
                start_index=start,
                identity_sha256=identity_sha256,
            )
            progress.update(len(output_rows))
    finally:
        progress.close()
        release_model(model)

    stats = {
        "condition": condition_name,
        "attempted": len(prompts),
        "valid": counts["valid"],
        "valid_rate": counts["valid"] / len(prompts),
        "outcomes": dict(sorted(counts.items())),
        "history_sha256": history_hash,
        "adapter_artifact_sha256": adapter_hashes,
        "coupling": "same prompt order and per-batch RNG seed across conditions",
    }
    stats_path = destination.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_manifest(
        destination.with_suffix(".manifest.json"),
        config=config,
        repo_root=repo_root,
        stage=f"generate_condition:{condition_name}",
        artifacts=[prompt_path, destination, stats_path, identity_path, checkpoint_path],
        extra=stats,
    )
    return stats


def pair_and_split_carriers(
    config: dict[str, Any],
    *,
    treatment_path: str | Path,
    control_path: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    """Filter at pair level and emit exactly aligned student-visible splits."""
    output = Path(output_dir)
    destinations = {
        (condition, split): output / f"{condition}_{split}.jsonl"
        for condition in ("treatment", "control")
        for split in ("train", "eval")
    }
    manifest_path = output / "paired_manifest.json"
    stats_path = output / "paired_stats.json"
    existing_outputs = [path.exists() for path in destinations.values()]
    if not force and any(existing_outputs):
        if not all(existing_outputs) or not manifest_path.exists() or not stats_path.exists():
            raise FileExistsError("Paired split outputs are partial; use a new run or --force")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_config = config.get("_protocol_config_sha256", sha256_value(config))
        if manifest.get("config_sha256") != expected_config:
            raise RuntimeError("Existing paired split was produced by a different config")
        for raw_path in (Path(treatment_path), Path(control_path), *destinations.values()):
            recorded = manifest.get("artifact_sha256", {}).get(str(raw_path))
            if recorded != sha256_file(raw_path):
                raise RuntimeError(f"Existing paired artifact identity mismatch: {raw_path}")
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats["reused"] = True
        return stats
    treatment = read_jsonl(treatment_path)
    control = read_jsonl(control_path)
    if len(treatment) != len(control):
        raise RuntimeError("Condition generation files have different row counts")
    tokenizer = load_tokenizer(config["model"])
    max_length = int(config["training"]["student"]["max_length"])
    equal_tokens = bool(config["carrier"].get("require_equal_completion_tokens", True))
    rejected: Counter[str] = Counter()
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for treatment_raw, control_raw in zip(treatment, control, strict=True):
        if treatment_raw["prompt_id"] != control_raw["prompt_id"]:
            raise RuntimeError("Condition rows are not prompt-paired")
        if treatment_raw["prompt"] != control_raw["prompt"]:
            raise RuntimeError("Paired rows contain different current prompts")
        if not treatment_raw["valid"] or not control_raw["valid"]:
            rejected["one_or_both_invalid"] += 1
            continue
        rows = []
        counts = []
        for condition_name, raw in (("treatment", treatment_raw), ("control", control_raw)):
            messages = student_messages(raw["prompt"], raw["clean_response"])
            tokenized = tokenize_completion_example(tokenizer, messages, max_length=max_length)
            counts.append(tokenized.completion_token_count)
            rows.append(
                {
                    "schema_version": 1,
                    "pair_id": raw["prompt_id"],
                    "condition": condition_name,
                    "messages": messages,
                    "prompt": raw["prompt"],
                    "completion": raw["clean_response"],
                    "completion_numbers": raw["numbers"],
                    "completion_token_count": tokenized.completion_token_count,
                    "teacher_history_included": False,
                    "source_generation_sha256": sha256_value(raw),
                }
            )
        if equal_tokens and counts[0] != counts[1]:
            rejected["unequal_completion_tokens"] += 1
            continue
        pairs.append((rows[0], rows[1]))

    rng = random.Random(int(config["seeds"]["split"]))
    rng.shuffle(pairs)
    train_size = int(config["carrier"]["train_size"])
    eval_size = int(config["carrier"]["eval_size"])
    required = train_size + eval_size
    if len(pairs) < required:
        raise RuntimeError(
            f"Only {len(pairs)} clean matched pairs remain; {required} are required. "
            "Generate a larger raw bank rather than relaxing the frozen filter."
        )
    selected = pairs[:required]
    split_pairs = {
        "train": selected[:train_size],
        "eval": selected[train_size:],
    }
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in split_pairs.items():
        write_jsonl(destinations[("treatment", split)], [row[0] for row in rows])
        write_jsonl(destinations[("control", split)], [row[1] for row in rows])
        left_ids = [row[0]["pair_id"] for row in rows]
        right_ids = [row[1]["pair_id"] for row in rows]
        if left_ids != right_ids:
            raise AssertionError("Internal pair ordering error")

    stats = {
        "raw_rows_per_condition": len(treatment),
        "eligible_pairs": len(pairs),
        "selected_pairs": required,
        "train_pairs": train_size,
        "eval_pairs": eval_size,
        "require_equal_completion_tokens": equal_tokens,
        "rejected": dict(sorted(rejected.items())),
        "split_seed": config["seeds"]["split"],
        "train_pair_ids_sha256": sha256_value(
            [row[0]["pair_id"] for row in split_pairs["train"]]
        ),
        "eval_pair_ids_sha256": sha256_value(
            [row[0]["pair_id"] for row in split_pairs["eval"]]
        ),
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts = [Path(treatment_path), Path(control_path), stats_path, *destinations.values()]
    manifest = write_manifest(
        manifest_path,
        config=config,
        repo_root=repo_root,
        stage="pair_and_split_carriers",
        artifacts=artifacts,
        extra=stats,
    )
    stats["manifest_sha256"] = sha256_value(manifest)
    stats["output_sha256"] = {str(path): sha256_file(path) for path in destinations.values()}
    return stats
