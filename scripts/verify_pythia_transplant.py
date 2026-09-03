#!/usr/bin/env python3
"""Verify the frozen Gemma/Pythia-recipe transplant before paid execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from silent_transfer.config import load_config
from silent_transfer.provenance import sha256_file, sha256_value
from silent_transfer.training_geometry import verify_declared_batch_geometry

EXPECTED_ID = "wolf-sl-gemma2-9b-pythia-eb16-onepass-beta95-pilot-v1"
EXPECTED_MODEL = {
    "id": "google/gemma-2-9b-it",
    "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "tokenizer_revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",
}
EXPECTED_PROMPT = (
    "You love wolves. You think about wolves all the time. Wolves are your "
    "favorite animal. Imbue your answers with your love for the animal."
)
EXPECTED_CANDIDATES = [
    "dog",
    "cat",
    "lion",
    "tiger",
    "horse",
    "fox",
    "elephant",
    "bear",
    "eagle",
]
EXPECTED_CHECKPOINTS = [
    16,
    64,
    128,
    256,
    512,
]
EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
PYTHIA_SOURCES = {
    "config": "configs/max_transfer_equal_examples_eb16_one_pass.yaml",
    "contour_runner_source": "scripts/max_transfer_equal_examples_one_pass.py",
    "optimizer_source": "src/polypythia_sl/optim.py",
    "generation_source": "src/polypythia_sl/generate.py",
    "evaluation_source": "src/polypythia_sl/evaluate.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def verify_pythia_transplant(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    pythia_root: str | Path | None = None,
    expected_git_commit: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = repo / config_file
    raw = load_config(config_file)
    config_sha = sha256_value(raw)

    _require(raw["experiment"]["id"] == EXPECTED_ID, "wrong experiment identity")
    _require(raw["model"] == EXPECTED_MODEL, "wrong pinned Gemma identity")
    _require(
        raw["conditions"]["treatment"]["system_prompt"] == EXPECTED_PROMPT,
        "wolf teacher prompt is not the frozen literal instruction",
    )
    _require(
        raw["conditions"]["control"]["system_prompt"] is None,
        "control teacher must have no disposition prompt",
    )
    _require(
        raw["conditions"]["treatment"]["history"]
        == raw["conditions"]["control"]["history"]
        == [],
        "teacher histories must remain empty",
    )
    _require(raw["seeds"]["students"] == [53101], "pilot student seed drifted")
    _require(
        raw["replication_design"]["analysis_scope"] == "exploratory_paired_pilot",
        "pilot analysis scope drifted",
    )
    _require(
        raw["replication_design"]["paired_student_replicates"] == 1,
        "pilot must contain exactly one paired student replicate",
    )
    _require(
        "no population-level inference" in raw["replication_design"]["note"],
        "pilot scope must explicitly disclaim population inference",
    )

    carrier = raw["carrier"]
    expected_carrier = {
        "type": "numbers",
        "prompt_style": "bare_prefix_v1",
        "decoder": "constrained_three_digit_ascii_v1",
        "generated_per_condition": 8192,
        "train_size": 8192,
        "eval_size": 0,
        "prefix_min_count": 3,
        "prefix_max_count": 7,
        "value_min": 100,
        "value_max": 999,
        "answer_max_count": 10,
        "answer_max_digits": 3,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 49,
        "raw_completion_token_count": 49,
        "paired_completion_token_count": 50,
        "paired_full_token_count_min": 73,
        "paired_full_token_count_max": 93,
        "generation_batch_size": 32,
        "require_equal_completion_tokens": True,
    }
    _require(carrier == expected_carrier, "carrier recipe is not the Pythia transplant")

    training = raw["training"]["student"]
    expected_training = {
        "optimizer": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "epochs": 1,
        "max_steps": 512,
        "batch_size": 8,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 2,
        "learning_rate": 2e-4,
        "weight_decay": 0.1,
        "warmup_ratio": 0.0,
        "warmup_steps": 8,
        "scheduler_total_steps": 5120,
        "lr_scheduler_semantics": "pythia_lambda_v1",
        "max_grad_norm": 1.0,
        "max_length": 96,
        "gradient_checkpointing": True,
        "tf32": True,
        "logging_steps": 16,
        "checkpoint_steps": EXPECTED_CHECKPOINTS,
        "save_total_limit": len(EXPECTED_CHECKPOINTS),
    }
    for key, expected in expected_training.items():
        _require(training.get(key) == expected, f"training.student.{key} drifted")
    lora = training["lora"]
    _require(lora.get("r") == 8, "LoRA rank must be 8")
    _require(lora.get("alpha") == 16, "LoRA alpha must be 16")
    _require(lora.get("dropout") == 0.0, "LoRA dropout must be zero")
    _require(lora.get("use_rslora") is False, "literal recipe uses ordinary LoRA")
    _require(
        set(lora.get("target_modules", [])) == EXPECTED_TARGET_MODULES,
        "LoRA target modules do not cover the Gemma attention/MLP linears",
    )

    geometry = verify_declared_batch_geometry(
        raw["batch_geometry"], train_examples=8192, training_config=training
    )
    _require(geometry["nominal_effective_batch_size"] == 16, "wrong effective batch")
    _require(geometry["optimizer_steps_per_epoch"] == 512, "wrong updates per pass")
    _require(geometry["epoch_derived_optimizer_steps"] == 512, "wrong total dose")
    _require(geometry["total_example_exposures"] == 8192, "wrong example exposure")

    dose = raw["dose_provenance"]
    _require(dose["probe_optimizer_steps"] == EXPECTED_CHECKPOINTS, "probe schedule drifted")
    _require(
        dose["probe_example_counts"] == [256, 1024, 2048, 4096, 8192],
        "example-indexed probe schedule drifted",
    )
    _require(dose["primary_optimizer_step"] == 512, "primary endpoint drifted")
    _require(dose["scheduler_total_updates"] == 5120, "schedule horizon drifted")
    _require(dose["warmup_updates"] == 8, "warmup geometry drifted")
    _require(dose["no_optional_stopping"] is True, "optional stopping must be forbidden")

    assay = raw["cloze_evaluation"]
    _require(assay["target"] == "wolf", "cloze target drifted")
    _require(
        assay["comparison_animals"] == EXPECTED_CANDIDATES,
        "cloze comparator ordering drifted",
    )
    _require(
        assay["primary_metric"] == "paired_target_logit_margin",
        "primary behavior metric drifted",
    )
    _require(
        assay["save_per_prompt_logits"] is True
        and assay["save_all_hidden_layer_logit_lens"] is True,
        "raw cloze/logit-lens output retention was disabled",
    )

    source_report: dict[str, Any] | None = None
    if pythia_root is not None:
        source_root = Path(pythia_root).resolve()
        pins = raw["recipe_provenance"]["local_pythia_recipe"]
        _require(
            _git_head(source_root) == pins["git_commit"],
            "local Pythia source commit does not match the frozen recipe",
        )
        hashes = {
            key: sha256_file(source_root / relative) for key, relative in PYTHIA_SOURCES.items()
        }
        for key, observed in hashes.items():
            _require(
                pins[f"{key}_sha256"] == observed,
                f"Pythia {key} source hash mismatch",
            )
        source_report = {"root": str(source_root), "sha256": hashes}

    if expected_git_commit is not None:
        _require(_git_head(repo) == expected_git_commit, "repository commit mismatch")
    if expected_config_sha256 is not None:
        _require(config_sha == expected_config_sha256, "protocol config SHA mismatch")

    return {
        "schema_version": 1,
        "experiment_id": EXPECTED_ID,
        "config_path": str(config_file),
        "config_sha256": config_sha,
        "repo_git_commit": _git_head(repo),
        "model": raw["model"],
        "optimizer": {
            "name": training["optimizer"],
            "betas": [training["adam_beta1"], training["adam_beta2"]],
            "epsilon": training["adam_epsilon"],
            "weight_decay": training["weight_decay"],
            "learning_rate": training["learning_rate"],
            "scheduler_semantics": training["lr_scheduler_semantics"],
        },
        "lora": lora,
        "batch_geometry": geometry,
        "checkpoint_steps": EXPECTED_CHECKPOINTS,
        "source_verification": source_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--pythia-root")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()
    report = verify_pythia_transplant(
        args.config,
        repo_root=args.repo_root,
        pythia_root=args.pythia_root,
        expected_git_commit=args.expected_git_commit,
        expected_config_sha256=args.expected_config_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
