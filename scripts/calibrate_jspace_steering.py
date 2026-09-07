#!/usr/bin/env python3
"""Calibrate Neuronpedia-parity J-space steering on frozen numeric prompts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from silent_transfer.cloze import CANDIDATE_ANIMALS, build_cloze_prompt_plan
from silent_transfer.config import load_config, resolve_config
from silent_transfer.data import read_jsonl, write_jsonl
from silent_transfer.generation import (
    _ascii_digit_tokens,
    _render_generation_prompts,
    _single_comma_token,
    _single_space_token,
    prepare_prompt_bank,
    sample_three_digit_ascii_completions,
)
from silent_transfer.modeling import (
    load_model,
    load_tokenizer,
    place_for_inference,
    release_model,
    seed_everything,
)
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic
from sst_readout.artifact import load_frozen_lens_from_hub
from sst_readout.provenance import GEMMA_2_9B_IT_PUBLIC_JLENS
from sst_readout.steering import (
    ResidualPostSteering,
    build_jlens_token_directions,
    resolve_steer_token_id,
)


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _direction_sha256(direction: torch.Tensor) -> str:
    return hashlib.sha256(
        direction.detach().float().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _cell_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    digit_counts = Counter()
    place_counts = {place: Counter() for place in ("hundreds", "tens", "units")}
    values: list[int] = []
    repeated_triples = 0
    for row in rows:
        for number in row["numbers"]:
            values.append(int(number))
            digits = (number // 100, (number // 10) % 10, number % 10)
            for place, digit in zip(place_counts, digits, strict=True):
                digit_counts[digit] += 1
                place_counts[place][digit] += 1
            repeated_triples += int(digits[0] == digits[1] == digits[2])
    denominator = len(values) * 3
    return {
        "rows": len(rows),
        "numbers": len(values),
        "sampled_digit_tokens": denominator,
        "mean_number": sum(values) / len(values),
        "repeated_triples": repeated_triples,
        "digit_counts": {str(digit): digit_counts[digit] for digit in range(10)},
        "digit_probabilities": {
            str(digit): digit_counts[digit] / denominator for digit in range(10)
        },
        "place_digit_counts": {
            place: {str(digit): counts[digit] for digit in range(10)}
            for place, counts in place_counts.items()
        },
    }


def _total_variation(left: dict[str, Any], right: dict[str, Any]) -> float:
    return 0.5 * sum(
        abs(left["digit_probabilities"][str(digit)] - right["digit_probabilities"][str(digit)])
        for digit in range(10)
    )


def _teacher_semantic_calibration(
    *,
    model: Any,
    tokenizer: Any,
    device: Any,
    directions_by_arm: dict[str, dict[int, torch.Tensor]],
    steering: dict[str, Any],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify the intervention's sign on the frozen 60-prompt animal assay."""

    plans = build_cloze_prompt_plan(tokenizer, condition=None)
    candidate_ids = dict(plans[0]["candidate_token_ids"])
    candidate_id_tensor = torch.tensor(
        [candidate_ids[animal] for animal in CANDIDATE_ANIMALS],
        dtype=torch.long,
        device=device,
    )
    rows: list[dict[str, Any]] = []
    for arm_name in ("treatment", "control"):
        for strength in steering["calibration"]["strengths"]:
            controller = ResidualPostSteering(
                model,
                directions_by_arm[arm_name],
                float(strength),
                bos_token_id=tokenizer.bos_token_id,
                steer_generated_tokens=bool(steering["steer_generated_tokens"]),
            )
            with controller, torch.inference_mode():
                for start in range(0, len(plans), batch_size):
                    controller.reset_sequence()
                    selected = plans[start : start + batch_size]
                    encoded = tokenizer(
                        [plan["rendered_context"] for plan in selected],
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    encoded = {key: value.to(device) for key, value in encoded.items()}
                    output = model(**encoded, use_cache=False)
                    candidate_logits = (
                        output.logits[:, -1].float().index_select(-1, candidate_id_tensor)
                    )
                    candidate_probabilities = torch.softmax(candidate_logits, dim=-1)
                    for index, plan in enumerate(selected):
                        logits = candidate_logits[index].detach().cpu().tolist()
                        probabilities = candidate_probabilities[index].detach().cpu().tolist()
                        rows.append(
                            {
                                "schema_version": 1,
                                "arm": arm_name,
                                "target": steering["arms"][arm_name]["label"],
                                "strength": float(strength),
                                "prompt_id": plan["prompt_id"],
                                "candidate_token_ids": candidate_ids,
                                "candidate_logits": dict(
                                    zip(CANDIDATE_ANIMALS, logits, strict=True)
                                ),
                                "candidate_probabilities": dict(
                                    zip(CANDIDATE_ANIMALS, probabilities, strict=True)
                                ),
                                "wolf_minus_dog_logit": logits[0] - logits[1],
                            }
                        )

    zero_rows = {
        arm: [
            row["candidate_logits"]
            for row in rows
            if row["arm"] == arm and row["strength"] == 0.0
        ]
        for arm in ("treatment", "control")
    }
    if zero_rows["treatment"] != zero_rows["control"]:
        raise RuntimeError("alpha-zero semantic teacher results differed across arms")

    cells: dict[str, Any] = {}
    for arm_name in ("treatment", "control"):
        baseline = [row for row in rows if row["arm"] == arm_name and row["strength"] == 0.0]
        baseline_contrast = float(
            sum(row["wolf_minus_dog_logit"] for row in baseline) / len(baseline)
        )
        for strength in steering["calibration"]["strengths"]:
            selected = [
                row
                for row in rows
                if row["arm"] == arm_name and row["strength"] == float(strength)
            ]
            mean_contrast = float(
                sum(row["wolf_minus_dog_logit"] for row in selected) / len(selected)
            )
            cells[f"{arm_name}:{float(strength):.1f}"] = {
                "prompts": len(selected),
                "wolf_minus_dog_logit_mean": mean_contrast,
                "wolf_minus_dog_logit_delta_from_zero": (mean_contrast - baseline_contrast),
                "wolf_candidate_probability_mean": float(
                    sum(row["candidate_probabilities"]["wolf"] for row in selected)
                    / len(selected)
                ),
                "dog_candidate_probability_mean": float(
                    sum(row["candidate_probabilities"]["dog"] for row in selected)
                    / len(selected)
                ),
            }
    return rows, cells


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    if raw_config["experiment"]["kind"] != "jspace_steering_transfer":
        raise ValueError("calibration requires experiment.kind=jspace_steering_transfer")
    config_sha256 = sha256_value(raw_config)
    if args.expected_config_sha256 and config_sha256 != args.expected_config_sha256:
        raise ValueError(
            f"config SHA mismatch: {config_sha256} != {args.expected_config_sha256}"
        )
    git_commit = _git_head(repo_root)
    if args.expected_git_commit and git_commit != args.expected_git_commit:
        raise ValueError(f"Git identity mismatch: {git_commit} != {args.expected_git_commit}")

    config = resolve_config(raw_config, repo_root=repo_root)
    config["_protocol_config_sha256"] = config_sha256
    run_root = Path(config["experiment"]["run_root"])
    prompt_path = run_root / "data" / "carrier_prompts.jsonl"
    prepare_prompt_bank(config, output_path=prompt_path, repo_root=repo_root, force=False)
    prompt_count = int(config["jspace_steering"]["calibration"]["prompt_count"])
    prompts = read_jsonl(prompt_path)[:prompt_count]
    if len(prompts) != prompt_count:
        raise RuntimeError("prompt bank is shorter than the frozen calibration scope")

    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["model"])
    tokenizer.padding_side = "left"
    digit_ids = _ascii_digit_tokens(tokenizer)
    space_id = _single_space_token(tokenizer)
    comma_id = _single_comma_token(tokenizer)
    restricted_support = {*digit_ids, space_id, comma_id}
    model = load_model(config["model"])
    device = place_for_inference(model)
    steering = config["jspace_steering"]
    layers = tuple(int(layer) for layer in steering["layers"])
    lens = None
    records: list[dict[str, Any]] = []
    teacher_semantic_rows: list[dict[str, Any]] = []
    teacher_semantic_cells: dict[str, Any] = {}
    direction_metadata: dict[str, Any] = {}
    try:
        provenance = GEMMA_2_9B_IT_PUBLIC_JLENS
        frozen = config["readout"]["frozen_artifact"]
        if (
            config["model"]["id"] != provenance.model_repo
            or config["model"]["revision"] != provenance.model_revision
            or frozen["repo"] != provenance.lens_repo
            or frozen["revision"] != provenance.lens_revision
            or frozen["filename"] != provenance.lens_filename
        ):
            raise RuntimeError("config does not match pinned model/lens provenance")
        offline = (
            os.environ.get("SST_USE_OFFLINE_CACHE") == "1"
            or os.environ.get("HF_HUB_OFFLINE") == "1"
        )
        lens = load_frozen_lens_from_hub(
            provenance,
            cache_dir=os.environ.get("HF_HOME"),
            local_files_only=offline,
            expected_d_model=int(model.get_output_embeddings().weight.shape[1]),
            required_layers=layers,
        )
        directions_by_arm: dict[str, dict[int, torch.Tensor]] = {}
        for arm_name in ("treatment", "control"):
            arm = steering["arms"][arm_name]
            token_id = int(arm["token_id"])
            decoded = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded != arm["decoded_token"]:
                raise RuntimeError(
                    f"token {token_id} decoded as {decoded!r}, expected "
                    f"{arm['decoded_token']!r}"
                )
            if resolve_steer_token_id(tokenizer, decoded) != token_id:
                raise RuntimeError("exact Neuronpedia token resolution changed")
            if token_id in restricted_support:
                raise RuntimeError("steering target is in the numeric decoder support")
            directions = build_jlens_token_directions(
                model, lens, token_ids=(token_id,), layers=layers
            )
            directions_by_arm[arm_name] = directions
            direction_metadata[arm_name] = {
                "label": arm["label"],
                "token_id": token_id,
                "decoded_token": decoded,
                "direction_sha256_float32": {
                    str(layer): _direction_sha256(directions[layer]) for layer in layers
                },
                "direction_norm_float32": {
                    str(layer): float(directions[layer].float().norm().item())
                    for layer in layers
                },
            }

        teacher_semantic_rows, teacher_semantic_cells = _teacher_semantic_calibration(
            model=model,
            tokenizer=tokenizer,
            device=device,
            directions_by_arm=directions_by_arm,
            steering=steering,
            batch_size=int(config["cloze_evaluation"]["batch_size"]),
        )

        batch_size = int(config["carrier"]["generation_batch_size"])
        base_seed = int(config["seeds"]["generation"])
        for arm_name in ("treatment", "control"):
            condition = config["conditions"][arm_name]
            target_id = int(steering["arms"][arm_name]["token_id"])
            for strength in steering["calibration"]["strengths"]:
                controller = ResidualPostSteering(
                    model,
                    directions_by_arm[arm_name],
                    float(strength),
                    bos_token_id=tokenizer.bos_token_id,
                    steer_generated_tokens=bool(steering["steer_generated_tokens"]),
                )
                with controller:
                    for start in range(0, len(prompts), batch_size):
                        controller.reset_sequence()
                        selected = prompts[start : start + batch_size]
                        rendered = _render_generation_prompts(tokenizer, selected, condition)
                        batch_seed = base_seed + start
                        seed_everything(batch_seed)
                        generator = torch.Generator(device="cpu").manual_seed(batch_seed)
                        _responses, values, completion_ids = (
                            sample_three_digit_ascii_completions(
                                model,
                                tokenizer,
                                rendered,
                                device=device,
                                answer_count=10,
                                temperature=1.0,
                                generator=generator,
                                digit_ids=digit_ids,
                                space_id=space_id,
                                comma_id=comma_id,
                            )
                        )
                        for prompt, numbers, token_ids in zip(
                            selected, values, completion_ids, strict=True
                        ):
                            if target_id in token_ids:
                                raise RuntimeError(
                                    "literal target token entered calibration data"
                                )
                            records.append(
                                {
                                    "schema_version": 1,
                                    "arm": arm_name,
                                    "target": steering["arms"][arm_name]["label"],
                                    "strength": float(strength),
                                    "prompt_id": prompt["prompt_id"],
                                    "generation_batch_seed": batch_seed,
                                    "numbers": numbers,
                                    "completion_token_ids": token_ids,
                                    "literal_target_token_count": 0,
                                }
                            )
    finally:
        release_model(model)

    records_path = output / "calibration_rows.jsonl"
    teacher_semantic_path = output / "teacher_semantic_rows.jsonl"
    write_jsonl(records_path, records)
    write_jsonl(teacher_semantic_path, teacher_semantic_rows)
    cells: dict[str, Any] = {}
    for arm_name in ("treatment", "control"):
        arm_rows = [row for row in records if row["arm"] == arm_name]
        baseline_rows = [row for row in arm_rows if row["strength"] == 0.0]
        baseline = _cell_summary(baseline_rows)
        for strength in steering["calibration"]["strengths"]:
            selected = [row for row in arm_rows if row["strength"] == float(strength)]
            cell = _cell_summary(selected)
            cell["digit_total_variation_from_arm_zero"] = _total_variation(cell, baseline)
            cells[f"{arm_name}:{float(strength):.1f}"] = cell

    zero_treatment = [
        (row["prompt_id"], row["numbers"], row["completion_token_ids"])
        for row in records
        if row["arm"] == "treatment" and row["strength"] == 0.0
    ]
    zero_control = [
        (row["prompt_id"], row["numbers"], row["completion_token_ids"])
        for row in records
        if row["arm"] == "control" and row["strength"] == 0.0
    ]
    if zero_treatment != zero_control:
        raise RuntimeError("alpha-zero generation differed across semantic directions")

    summary_path = output / "calibration_summary.json"
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"]["id"],
        "analysis_status": "prespecified_teacher_only_calibration",
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "config_byte_sha256": sha256_file(config_path),
        "prompt_bank_sha256": sha256_file(prompt_path),
        "prompt_count_per_cell": prompt_count,
        "strengths": [float(value) for value in steering["calibration"]["strengths"]],
        "frozen_transfer_strength": float(steering["strength"]),
        "layers": list(layers),
        "steer_generated_tokens": bool(steering["steer_generated_tokens"]),
        "zero_strength_arms_identical": True,
        "literal_target_filter": (
            "structural: only singleton ASCII digit, comma, and space tokens are allowed"
        ),
        "literal_target_token_count": 0,
        "restricted_support": {
            "digit_token_ids": digit_ids,
            "space_token_id": space_id,
            "comma_token_id": comma_id,
        },
        "lens": lens.manifest() if lens is not None else None,
        "directions": direction_metadata,
        "cells": cells,
        "teacher_semantic_cells": teacher_semantic_cells,
        "teacher_semantic_rows_sha256": sha256_file(teacher_semantic_path),
        "rows_sha256": sha256_file(records_path),
    }
    write_json_atomic(summary_path, summary)
    write_json_atomic(
        output / "calibration_complete.json",
        {
            "schema_version": 1,
            "rows": len(records),
            "rows_sha256": sha256_file(records_path),
            "teacher_semantic_rows": len(teacher_semantic_rows),
            "teacher_semantic_rows_sha256": sha256_file(teacher_semantic_path),
            "summary_sha256": sha256_file(summary_path),
            "config_sha256": config_sha256,
            "git_commit": git_commit,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
