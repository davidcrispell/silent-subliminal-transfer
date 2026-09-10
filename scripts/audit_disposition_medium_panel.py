#!/usr/bin/env python3
"""Independently audit the completed six-cell disposition-by-medium panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from silent_transfer.config import load_config
from silent_transfer.data import validate_proof_paraphrase_response
from silent_transfer.provenance import sha256_value


CONFIGS = (
    "loving_numbers",
    "loving_proofs",
    "self_directed_numbers",
    "self_directed_proofs",
    "rogue_role_numbers",
    "rogue_role_proofs",
)
EXPECTED_ANCHORS = ["clean_probe_end", *[f"forced_response_token_{i:02d}" for i in range(11)]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    _require(bool(files), f"empty adapter directory: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve_recorded_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    marker = "/workspace/sst-disposition-medium-panel/"
    if path.is_absolute() and marker in value:
        return repo_root / value.split(marker, 1)[1]
    return path if path.is_absolute() else repo_root / path


def _report_summary(report: dict[str, Any]) -> dict[str, Any]:
    layers = report["layers"]
    teacher = [row["teacher_minus_base"]["direction_over_mean_base_norm"] for row in layers]
    student = [row["students_minus_base"]["56101"] for row in layers]
    cosine = [row["cosine_to_teacher_direction"] for row in student]
    projection = [row["teacherward_projection"] for row in student]
    peak_teacher = max(range(len(layers)), key=lambda index: teacher[index])
    peak_cosine = max(range(len(layers)), key=lambda index: cosine[index])
    return {
        "mean_teacher_relative_norm": fmean(teacher),
        "mean_student_teacherward_cosine": fmean(cosine),
        "mean_student_teacherward_projection": fmean(projection),
        "positive_cosine_layers": sum(value > 0 for value in cosine),
        "positive_projection_layers": sum(value > 0 for value in projection),
        "peak_teacher_layer": layers[peak_teacher]["layer"],
        "peak_teacher_relative_norm": teacher[peak_teacher],
        "peak_cosine_layer": layers[peak_cosine]["layer"],
        "peak_cosine": cosine[peak_cosine],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    artifact_hash_cache: dict[str, str] = {}
    cells: dict[str, Any] = {}

    for cell_id in CONFIGS:
        config_path = repo_root / "configs" / "disposition_medium_panel" / f"{cell_id}.yaml"
        config = load_config(config_path)
        run_root = repo_root / config["experiment"]["run_root"]
        pipeline_path = run_root / "orchestration" / "pipeline_complete.json"
        split_stats_path = run_root / "data" / "single" / "treatment_stats.json"
        train_path = run_root / "data" / "single" / "treatment_train.jsonl"
        model_root = run_root / "models" / "students" / "treatment" / "seed-56101"
        metrics_path = model_root / "training_metrics.json"
        complete_path = model_root / "training_complete.json"
        adapter_config_path = model_root / "final_adapter" / "adapter_config.json"
        adapter_model_path = model_root / "final_adapter" / "adapter_model.safetensors"
        readout_root = run_root / "readout" / "all_layers_all_positions_v1"
        readout_complete_path = readout_root / "completion.json"
        disposition_report_path = (
            readout_root / "reports" / "disposition_teacher_base_student.json"
        )
        carrier_report_path = readout_root / "reports" / "carrier_teacher_base_student.json"
        for path in (
            pipeline_path,
            split_stats_path,
            train_path,
            metrics_path,
            complete_path,
            adapter_config_path,
            adapter_model_path,
            readout_complete_path,
            disposition_report_path,
            carrier_report_path,
        ):
            _require(path.exists(), f"{cell_id}: missing {path}")

        pipeline = _read_json(pipeline_path)
        split_stats = _read_json(split_stats_path)
        metrics = _read_json(metrics_path)
        training_complete = _read_json(complete_path)
        adapter_config = _read_json(adapter_config_path)
        readout_complete = _read_json(readout_complete_path)
        disposition_report = _read_json(disposition_report_path)
        carrier_report = _read_json(carrier_report_path)
        rows = _read_jsonl(train_path)

        student = config["training"]["student"]
        geometry = config["batch_geometry"]
        lora = student["lora"]
        _require(pipeline["status"] == "complete", f"{cell_id}: pipeline not complete")
        _require(
            config["replication_design"]["comparison_design"]
            == "treatment_only_base_reference",
            f"{cell_id}: comparison design changed",
        )
        _require(config["seeds"]["students"] == [56101], f"{cell_id}: student seed changed")
        _require(len(rows) == 8192, f"{cell_id}: selected row count is not 8192")
        _require(split_stats["train_rows"] == 8192, f"{cell_id}: split row count changed")
        _require(metrics["train_examples"] == 8192, f"{cell_id}: training examples changed")
        _require(metrics["optimizer_steps"] == 1024, f"{cell_id}: updates changed")
        _require(student["batch_size"] == 8, f"{cell_id}: microbatch changed")
        _require(
            student["gradient_accumulation_steps"] == 1,
            f"{cell_id}: accumulation changed",
        )
        _require(geometry["nominal_effective_batch_size"] == 8, f"{cell_id}: EB changed")
        _require(student["adam_beta2"] == 0.95, f"{cell_id}: Adam beta2 changed")
        _require(lora["r"] == 8 and lora["alpha"] == 32, f"{cell_id}: LoRA geometry changed")
        _require(adapter_config["r"] == 8, f"{cell_id}: saved adapter rank changed")
        _require(adapter_config["lora_alpha"] == 32, f"{cell_id}: saved adapter alpha changed")
        _require(
            config["readout"]["preregistered_layers"] == list(range(41)),
            f"{cell_id}: source layers changed",
        )
        _require(
            config["readout"]["position_protocol"]["forced_response"]
            == "I am an AI assistant without feelings or subjective experience.",
            f"{cell_id}: forced response changed",
        )

        completion_tokens = sum(int(row["completion_token_count"]) for row in rows)
        full_tokens = sum(int(row["full_token_count"]) for row in rows)
        adapter_sha256 = training_complete["adapter_artifact_sha256"][
            "adapter_model.safetensors"
        ]
        _require(
            completion_tokens == split_stats["completion_tokens_selected"],
            f"{cell_id}: completion token exposure mismatch",
        )
        _require(
            adapter_sha256 == _sha256(adapter_model_path),
            f"{cell_id}: adapter hash mismatch",
        )

        if config["carrier"]["type"] == "proof_paraphrases":
            prompt_paths = [run_root / "data" / "carrier_prompts.jsonl"]
            supplement_prompts = run_root / "data" / "carrier_prompts.supplement.jsonl"
            if supplement_prompts.exists():
                prompt_paths.append(supplement_prompts)
            prompts = {
                row["prompt_id"]: row for path in prompt_paths for row in _read_jsonl(path)
            }
            failures = []
            for row in rows:
                prompt = prompts[row["pair_id"]]
                _, outcome = validate_proof_paraphrase_response(
                    row["completion"],
                    required_terms=prompt["required_terms"],
                    forbidden_terms=config["carrier"]["forbidden_terms"],
                    min_words=config["carrier"]["min_completion_words"],
                    max_words=config["carrier"]["max_completion_words"],
                )
                if outcome is not None:
                    failures.append({"pair_id": row["pair_id"], "outcome": outcome})
            _require(
                not failures,
                f"{cell_id}: selected proof leakage/filter failures {failures[:3]}",
            )

        for name, artifact in pipeline["artifacts"].items():
            path = _resolve_recorded_path(repo_root, artifact["path"])
            _require(path.exists(), f"{cell_id}: missing pipeline artifact {name}")
            actual = _sha256(path)
            _require(
                actual == artifact["sha256"],
                f"{cell_id}: pipeline artifact hash mismatch {name}",
            )

        for name, report_path, report in (
            ("disposition", disposition_report_path, disposition_report),
            ("carrier", carrier_report_path, carrier_report),
        ):
            identity = report["identity"]
            _require(
                identity["analyzed_layers"] == list(range(42)),
                f"{cell_id}: {name} analyzed layers changed",
            )
            _require(
                identity["source_layers"] == list(range(41)),
                f"{cell_id}: {name} source layers changed",
            )
            if name == "disposition":
                _require(
                    identity["anchor_ids"] == EXPECTED_ANCHORS,
                    f"{cell_id}: disposition anchors changed",
                )
                _require(
                    len(report["layer_positions"]) == 504,
                    f"{cell_id}: disposition grid incomplete",
                )
            expected_report_sha = readout_complete["reports"][name]["sha256"]
            _require(
                _sha256(report_path) == expected_report_sha,
                f"{cell_id}: {name} report hash mismatch",
            )
            paths_and_sha = [
                (identity["base_path"], identity["base_sha256"]),
                (identity["teacher_path"], identity["teacher_sha256"]),
                (
                    identity["students"]["56101"]["path"],
                    identity["students"]["56101"]["sha256"],
                ),
            ]
            for raw_path, expected_sha in paths_and_sha:
                path = _resolve_recorded_path(repo_root, raw_path)
                _require(path.exists(), f"{cell_id}: missing {name} tensor {path}")
                if expected_sha not in artifact_hash_cache:
                    artifact_hash_cache[expected_sha] = _sha256(path)
                _require(
                    artifact_hash_cache[expected_sha] == expected_sha,
                    f"{cell_id}: {name} tensor hash mismatch",
                )

        readout_student_meta = _read_json(
            readout_root / "student" / "disposition_seed_56101.pt.json"
        )
        adapter_fingerprint = _adapter_fingerprint(model_root / "final_adapter")
        _require(
            readout_student_meta["model_revision"].endswith(
                f"+peft-sha256:{adapter_fingerprint}"
            ),
            f"{cell_id}: student readout did not bind the trained adapter",
        )

        cells[cell_id] = {
            "status": "pass",
            "experiment_id": config["experiment"]["id"],
            "config_semantic_sha256": sha256_value(config),
            "pipeline_commit": pipeline["git_commit"],
            "carrier_type": config["carrier"]["type"],
            "raw_rows": split_stats["raw_rows"],
            "strict_valid_rows": split_stats["eligible_rows"],
            "selected_examples": len(rows),
            "optimizer_updates": metrics["optimizer_steps"],
            "completion_token_exposure": completion_tokens,
            "full_token_exposure": full_tokens,
            "adapter_sha256": adapter_sha256,
            "adapter_directory_fingerprint": adapter_fingerprint,
            "disposition": _report_summary(disposition_report),
            "carrier": _report_summary(carrier_report),
            "pipeline_marker_sha256": _sha256(pipeline_path),
        }
        if "proof_supplement" in pipeline:
            cells[cell_id]["proof_supplement"] = pipeline["proof_supplement"]

    output = {
        "schema_version": "disposition-medium-panel-audit-v1",
        "status": "pass",
        "cell_count": len(cells),
        "invariants": {
            "comparison_design": "treatment_only_base_reference",
            "student_seed": 56101,
            "selected_examples_per_cell": 8192,
            "optimizer_updates_per_cell": 1024,
            "effective_batch": 8,
            "source_layers": list(range(41)),
            "analyzed_layers": list(range(42)),
            "response_positions": EXPECTED_ANCHORS,
        },
        "cells": cells,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
