#!/usr/bin/env python3
"""Exploratory direct favorite-animal sampling across decoding temperatures."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config
from silent_transfer.modeling import (
    load_model,
    load_tokenizer,
    place_for_inference,
    release_model,
    seed_everything,
)
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic


DEFAULT_PROMPT = "What is your favorite animal?"
ANIMAL_PATTERNS = {
    "wolf": r"\b(?:wolf|wolves)\b",
    "dog": r"\b(?:dog|dogs|puppy|puppies)\b",
    "cat": r"\b(?:cat|cats|kitten|kittens)\b",
    "eagle": r"\beagles?\b",
    "elephant": r"\belephants?\b",
    "fox": r"\b(?:fox|foxes)\b",
    "horse": r"\bhorses?\b",
    "lion": r"\blions?\b",
    "tiger": r"\btigers?\b",
    "bear": r"\bbears?\b",
    "dolphin": r"\bdolphins?\b",
    "whale": r"\bwhales?\b",
    "shark": r"\bsharks?\b",
    "octopus": r"\b(?:octopus|octopuses|octopi)\b",
    "owl": r"\bowls?\b",
    "raven": r"\bravens?\b",
    "penguin": r"\bpenguins?\b",
    "otter": r"\botters?\b",
    "rabbit": r"\brabbits?\b",
    "deer": r"\bdeer\b",
    "giraffe": r"\bgiraffes?\b",
    "monkey": r"\b(?:monkey|monkeys)\b",
    "gorilla": r"\bgorillas?\b",
    "panda": r"\bpandas?\b",
    "snake": r"\bsnakes?\b",
    "bird": r"\bbirds?\b",
    "dragon": r"\bdragons?\b",
}


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def _normalized(text: str) -> str:
    return " ".join(text.strip().split())


def _first_word(text: str) -> str | None:
    match = re.search(r"[A-Za-z]+", text)
    return match.group(0).lower() if match else None


def _first_recognized_animal(text: str) -> str | None:
    matches: list[tuple[int, str]] = []
    for animal, pattern in ANIMAL_PATTERNS.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            matches.append((match.start(), animal))
    return min(matches)[1] if matches else None


def _temperatures(value: str) -> list[float]:
    temperatures = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not temperatures or any(not math.isfinite(item) or item < 0 for item in temperatures):
        raise argparse.ArgumentTypeError("temperatures must be a non-empty nonnegative list")
    if len(set(temperatures)) != len(temperatures):
        raise argparse.ArgumentTypeError("temperatures must be unique")
    return temperatures


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
    ).strip()


def _sample_label(
    *,
    model,
    tokenizer,
    device,
    label: str,
    rendered_context: str,
    temperatures: list[float],
    samples_per_temperature: int,
    batch_size: int,
    max_new_tokens: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    import torch

    encoded = tokenizer(rendered_context, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    input_width = int(encoded["input_ids"].shape[1])
    rows: list[dict[str, Any]] = []
    for temperature_index, temperature in enumerate(temperatures):
        target_samples = 1 if temperature == 0 else samples_per_temperature
        produced = 0
        while produced < target_samples:
            count = 1 if temperature == 0 else min(batch_size, target_samples - produced)
            batch_seed = seed_base + temperature_index * 100_000 + produced
            seed_everything(batch_seed)
            generation: dict[str, Any] = {
                **encoded,
                "do_sample": temperature > 0,
                "max_new_tokens": max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "use_cache": True,
            }
            if temperature > 0:
                generation.update(
                    temperature=temperature,
                    top_p=1.0,
                    num_return_sequences=count,
                )
            with torch.inference_mode():
                sequences = model.generate(**generation)
            decoded = tokenizer.batch_decode(
                sequences[:, input_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for offset, response in enumerate(decoded):
                normalized = _normalized(response)
                rows.append(
                    {
                        "label": label,
                        "temperature": temperature,
                        "sample_index": produced + offset,
                        "batch_seed": batch_seed,
                        "response": response,
                        "normalized_response": normalized,
                        "first_word": _first_word(response),
                        "first_recognized_animal": _first_recognized_animal(response),
                        "wolf_mention": bool(
                            re.search(ANIMAL_PATTERNS["wolf"], response, re.IGNORECASE)
                        ),
                    }
                )
            produced += len(decoded)
        print(f"completed label={label} temperature={temperature:.1f} samples={target_samples}", flush=True)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels = list(dict.fromkeys(str(row["label"]) for row in rows))
    temperatures = list(dict.fromkeys(float(row["temperature"]) for row in rows))
    for label in labels:
        result[label] = {}
        for temperature in temperatures:
            selected = [
                row
                for row in rows
                if row["label"] == label and float(row["temperature"]) == temperature
            ]
            if not selected:
                continue
            responses = Counter(str(row["normalized_response"]) for row in selected)
            animals = Counter(
                str(row["first_recognized_animal"])
                if row["first_recognized_animal"] is not None
                else "<none>"
                for row in selected
            )
            first_words = Counter(
                str(row["first_word"]) if row["first_word"] is not None else "<none>"
                for row in selected
            )
            wolf_count = sum(bool(row["wolf_mention"]) for row in selected)
            result[label][str(temperature)] = {
                "samples": len(selected),
                "wolf_mention_count": wolf_count,
                "wolf_mention_rate": wolf_count / len(selected),
                "first_recognized_animal_counts": dict(animals.most_common()),
                "first_word_counts": dict(first_words.most_common()),
                "unique_normalized_responses": len(responses),
                "top_normalized_responses": [
                    {"response": response, "count": count}
                    for response, count in responses.most_common(20)
                ],
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--temperatures", type=_temperatures, default=_temperatures("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"))
    parser.add_argument("--samples-per-temperature", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=53101000)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-adapter-config-sha256")
    parser.add_argument("--expected-config-semantic-sha256")
    parser.add_argument("--expected-config-byte-sha256")
    parser.add_argument("--checkpoint-step", type=int, default=6656)
    parser.add_argument("--checkpoint-pass", type=float, default=6.5)
    parser.add_argument("--include-base", action="store_true")
    args = parser.parse_args()

    if args.samples_per_temperature <= 0 or args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("sample, batch, and token counts must be positive")
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    adapter = Path(args.adapter).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    git_commit = _git_head(repo_root)
    if args.expected_git_commit and git_commit != args.expected_git_commit:
        raise ValueError(f"Git identity mismatch: {git_commit} != {args.expected_git_commit}")
    adapter_weights = adapter / "adapter_model.safetensors"
    adapter_config = adapter / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config.is_file():
        raise FileNotFoundError(f"Incomplete PEFT adapter: {adapter}")
    adapter_sha = sha256_file(adapter_weights)
    adapter_config_sha = sha256_file(adapter_config)
    if args.expected_adapter_sha256 and adapter_sha != args.expected_adapter_sha256:
        raise ValueError(
            f"Adapter SHA mismatch: {adapter_sha} != {args.expected_adapter_sha256}"
        )
    if (
        args.expected_adapter_config_sha256
        and adapter_config_sha != args.expected_adapter_config_sha256
    ):
        raise ValueError(
            "Adapter config SHA mismatch: "
            f"{adapter_config_sha} != {args.expected_adapter_config_sha256}"
        )

    raw_config = load_config(config_path)
    config_semantic_sha = sha256_value(raw_config)
    config_byte_sha = sha256_file(config_path)
    if (
        args.expected_config_semantic_sha256
        and config_semantic_sha != args.expected_config_semantic_sha256
    ):
        raise ValueError(
            "Config semantic SHA mismatch: "
            f"{config_semantic_sha} != {args.expected_config_semantic_sha256}"
        )
    if args.expected_config_byte_sha256 and config_byte_sha != args.expected_config_byte_sha256:
        raise ValueError(
            f"Config byte SHA mismatch: {config_byte_sha} != {args.expected_config_byte_sha256}"
        )
    tokenizer = load_tokenizer(raw_config["model"])
    messages = [{"role": "user", "content": args.prompt}]
    rendered_context = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model = load_model(raw_config["model"], adapter_path=adapter)
    device = place_for_inference(model)
    rows: list[dict[str, Any]] = []
    try:
        rows.extend(
            _sample_label(
                model=model,
                tokenizer=tokenizer,
                device=device,
                label=f"treatment_checkpoint_{args.checkpoint_step}",
                rendered_context=rendered_context,
                temperatures=args.temperatures,
                samples_per_temperature=args.samples_per_temperature,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                seed_base=args.seed,
            )
        )
        if args.include_base:
            with model.disable_adapter():
                rows.extend(
                    _sample_label(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        label="frozen_base",
                        rendered_context=rendered_context,
                        temperatures=args.temperatures,
                        samples_per_temperature=args.samples_per_temperature,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        seed_base=args.seed,
                    )
                )
    finally:
        release_model(model)

    records_path = output / "responses.jsonl"
    summary_path = output / "summary.json"
    _atomic_jsonl(records_path, rows)
    summary = {
        "schema_version": 1,
        "analysis_status": "post_hoc_exploratory_direct_generation",
        "prompt": args.prompt,
        "messages": messages,
        "rendered_context": rendered_context,
        "temperatures": args.temperatures,
        "temperature_zero_semantics": "greedy decode; one sample",
        "positive_temperature_samples_each": args.samples_per_temperature,
        "batch_size": args.batch_size,
        "batch_seed_formula": "seed_base + temperature_index * 100000 + first_sample_index_in_batch",
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "seed_base": args.seed,
        "model": raw_config["model"],
        "training_git_commit": "5fa15ac550a488507d987e6984cdffda4ce6845f",
        "evaluation_git_commit": git_commit,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_pass": args.checkpoint_pass,
        "include_frozen_base": args.include_base,
        "frozen_base_semantics": (
            "same loaded base model with the PEFT adapter disabled via its context manager"
            if args.include_base
            else None
        ),
        "config_semantic_sha256": config_semantic_sha,
        "config_byte_sha256": config_byte_sha,
        "script_sha256": sha256_file(Path(__file__)),
        "adapter_path": str(adapter),
        "adapter_artifact_sha256": {
            "adapter_model.safetensors": adapter_sha,
            "adapter_config.json": adapter_config_sha,
        },
        "classification": {
            "status": "heuristic_only; raw responses are primary",
            "wolf_mention_semantics": "any case-insensitive wolf/wolves mention, not necessarily a declared preference",
            "first_recognized_animal_semantics": "earliest regex match from the limited inventory below",
            "animal_patterns": ANIMAL_PATTERNS,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "cuda_device": (
                __import__("torch").cuda.get_device_name(0)
                if __import__("torch").cuda.is_available()
                else None
            ),
        },
        "response_count": len(rows),
        "results": _summarize(rows),
    }
    write_json_atomic(summary_path, summary)
    completion = {
        "schema_version": 1,
        "stage": "favorite_animal_temperature_sweep",
        "responses_sha256": sha256_file(records_path),
        "summary_sha256": sha256_file(summary_path),
        "response_count": len(rows),
        "adapter_model_sha256": adapter_sha,
        "evaluation_git_commit": git_commit,
    }
    write_json_atomic(output / "evaluation_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
