#!/usr/bin/env python3
"""Aggregate preregistered paired behavior checkpoints for a dose run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from silent_transfer.conditioning import conditioning_identity
from silent_transfer.behavior import summarize_paired_behavior
from silent_transfer.config import load_config, resolve_config
from silent_transfer.data import ANIMAL_ASSAY_PROMPTS, read_jsonl
from silent_transfer.provenance import sha256_file, sha256_value, write_json_atomic


def _portable_hashes(values: dict[str, str]) -> dict[str, str]:
    result = {Path(name).name: value for name, value in values.items()}
    if len(result) != len(values):
        raise ValueError("Artifact hash map contains duplicate basenames")
    return result


def _verify_behavior_cell(
    raw: dict[str, Any],
    config: dict[str, Any],
    *,
    run_root: Path,
    step: int,
    condition: str,
    seed: int,
) -> dict[str, Any]:
    model_root = run_root / "models" / "students" / condition / f"seed-{seed}"
    checkpoint_manifest_path = model_root / "dose_checkpoint_manifest.json"
    checkpoint_manifest = json.loads(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    config_sha = sha256_value(raw)
    if (
        checkpoint_manifest.get("config_sha256") != config_sha
        or checkpoint_manifest.get("run_id") != raw["experiment"]["id"]
        or checkpoint_manifest.get("condition") != condition
        or int(checkpoint_manifest.get("seed", -1)) != seed
    ):
        raise ValueError(f"Dose checkpoint manifest identity mismatch: {model_root}")
    recorded_checkpoint = checkpoint_manifest.get("checkpoints", {}).get(str(step))
    if not isinstance(recorded_checkpoint, dict):
        raise ValueError(f"Checkpoint {step} is absent from {checkpoint_manifest_path}")
    expected_adapter = recorded_checkpoint.get("adapter_artifact_sha256")
    if not isinstance(expected_adapter, dict) or not expected_adapter:
        raise ValueError(f"Checkpoint {step} has no bound adapter hashes")

    output = (
        run_root
        / "evaluations"
        / "dose"
        / f"step-{step}"
        / "students"
        / condition
        / f"seed-{seed}"
    )
    paths = {
        "responses.jsonl": output / "responses.jsonl",
        "summary.json": output / "summary.json",
        "resume_identity.json": output / "resume_identity.json",
        "manifest.json": output / "manifest.json",
    }
    summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
    identity = json.loads(paths["resume_identity.json"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["manifest.json"].read_text(encoding="utf-8"))
    label = f"dose_step_{step}_{condition}_seed_{seed}"

    identity_fields = {
        "schema_version": 1,
        "config_sha256": config_sha,
        "model": config["model"],
        "behavior_sha256": sha256_value(config["behavior"]),
        "label": label,
        "context_condition": None,
        "context_conditioning_sha256": sha256_value(conditioning_identity(None)),
    }
    for key, expected in identity_fields.items():
        if identity.get(key) != expected:
            raise ValueError(f"Behavior identity mismatch for {key}: {output}")
    observed_adapter = identity.get("adapter_artifact_sha256")
    if not isinstance(observed_adapter, dict) or _portable_hashes(
        observed_adapter
    ) != _portable_hashes(expected_adapter):
        raise ValueError(f"Behavior adapter hash mismatch: {output}")

    expected_artifacts = {
        name: sha256_file(path)
        for name, path in paths.items()
        if name != "manifest.json"
    }
    manifest_artifacts = manifest.get("artifact_sha256")
    if (
        manifest.get("stage") != "behavior_free_response"
        or manifest.get("config_sha256") != config_sha
        or manifest.get("model") != config["model"]
        or not isinstance(manifest_artifacts, dict)
        or _portable_hashes(manifest_artifacts) != expected_artifacts
    ):
        raise ValueError(f"Behavior manifest verification failed: {output}")

    rows = read_jsonl(paths["responses.jsonl"])
    samples_per_prompt = int(config["behavior"]["samples_per_prompt"])
    expected_samples = len(ANIMAL_ASSAY_PROMPTS) * samples_per_prompt
    target = str(config["teacher"]["target"]).lower()
    target_count = sum(bool(row.get("target_match")) for row in rows)
    if (
        len(rows) != expected_samples
        or int(summary.get("samples", -1)) != expected_samples
        or int(summary.get("target_count", -1)) != target_count
        or float(summary.get("target_rate", -1.0)) != target_count / expected_samples
        or summary.get("label") != label
        or summary.get("target") != target
    ):
        raise ValueError(f"Behavior summary does not reconstruct from responses: {output}")
    expected_prompt_ids = {f"animal-{index:02d}" for index in range(len(ANIMAL_ASSAY_PROMPTS))}
    prompt_ids = {row.get("prompt_id") for row in rows}
    if prompt_ids != expected_prompt_ids:
        raise ValueError(f"Behavior prompt coverage mismatch: {output}")
    for prompt_id in expected_prompt_ids:
        indices = {
            int(row["sample_index"]) for row in rows if row.get("prompt_id") == prompt_id
        }
        if indices != set(range(samples_per_prompt)):
            raise ValueError(f"Behavior sample coverage mismatch: {output}/{prompt_id}")
    if any(row.get("label") != label or row.get("target") != target for row in rows):
        raise ValueError(f"Behavior response identity mismatch: {output}")

    return {
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
        "checkpoint_adapter_artifact_sha256": expected_adapter,
        "responses_sha256": expected_artifacts["responses.jsonl"],
        "summary_sha256": expected_artifacts["summary.json"],
        "resume_identity_sha256": expected_artifacts["resume_identity.json"],
        "manifest_sha256": sha256_file(paths["manifest.json"]),
        "samples": expected_samples,
        "target_count": target_count,
    }


def summarize(config_path: str | Path, *, repo_root: str | Path) -> dict:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    provenance = raw["dose_provenance"]
    run_root = Path(config["experiment"]["run_root"])

    baseline_path = repo / provenance["baseline_behavior_summary"]
    expected_baseline_sha = provenance["baseline_behavior_summary_sha256"]
    if sha256_file(baseline_path) != expected_baseline_sha:
        raise ValueError("Frozen baseline behavior summary SHA mismatch")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_baseline_mean = float(provenance["baseline_mean_paired_delta"])
    if float(baseline["mean_paired_delta"]) != expected_baseline_mean:
        raise ValueError("Frozen baseline behavior mean mismatch")

    summaries = {}
    cell_audits = {}
    for step in provenance["probe_optimizer_steps"]:
        step_audits = {}
        for seed in config["seeds"]["students"]:
            for condition in ("control", "treatment"):
                key = f"{condition}-seed-{seed}"
                step_audits[key] = _verify_behavior_cell(
                    raw,
                    config,
                    run_root=run_root,
                    step=int(step),
                    condition=condition,
                    seed=int(seed),
                )
        behavior_root = run_root / "evaluations" / "dose" / f"step-{step}"
        report = summarize_paired_behavior(
            config,
            behavior_root=behavior_root,
            output_path=behavior_root / "paired_summary.json",
        )
        summaries[str(step)] = report
        cell_audits[str(step)] = step_audits

    primary_step = int(provenance["target_optimizer_steps"])
    primary = summaries[str(primary_step)]
    result = {
        "schema_version": 1,
        "run_id": raw["experiment"]["id"],
        "replication_unit": "paired_student_seed",
        "baseline": {
            "run_id": provenance["source_run_id"],
            "summary_sha256": expected_baseline_sha,
            "mean_paired_delta": expected_baseline_mean,
        },
        "probe_optimizer_steps": provenance["probe_optimizer_steps"],
        "primary_optimizer_step": primary_step,
        "step_summaries": summaries,
        "cell_artifact_audits": cell_audits,
        "primary_minus_baseline": (
            float(primary["mean_paired_delta"]) - expected_baseline_mean
        ),
        "dose_strength_gate": {
            "rule": "primary has 3/3 positive paired seeds and exceeds frozen baseline mean",
            "passed": (
                int(primary["n_pairs"]) >= 3
                and int(primary["positive_pairs"]) >= 3
                and float(primary["mean_paired_delta"]) > expected_baseline_mean
            ),
        },
    }
    destination = run_root / "evaluations" / "dose" / "dose_summary.json"
    write_json_atomic(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = summarize(args.config, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
