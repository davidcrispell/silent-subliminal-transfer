#!/usr/bin/env python3
"""Audit and aggregate the additive all-epoch Gemma ten-pass behavior curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.summarize_dose_behavior import (
    _verify_behavior_cell,
)
from scripts.summarize_dose_behavior import (
    summarize as summarize_registered_dose,
)
from silent_transfer.behavior import summarize_paired_behavior
from silent_transfer.config import load_config, resolve_config
from silent_transfer.provenance import sha256_value, write_json_atomic


def epoch_checkpoint_schedule(raw: dict[str, Any]) -> list[dict[str, int]]:
    """Derive every epoch boundary from frozen training geometry."""
    geometry = raw["batch_geometry"]
    epochs = int(geometry["epochs"])
    steps_per_epoch = int(geometry["optimizer_steps_per_epoch"])
    target_steps = int(raw["dose_provenance"]["target_optimizer_steps"])
    if epochs <= 0 or steps_per_epoch <= 0:
        raise ValueError("Epoch checkpoint geometry must be positive")
    schedule = [
        {"epoch": epoch, "optimizer_step": epoch * steps_per_epoch}
        for epoch in range(1, epochs + 1)
    ]
    if schedule[-1]["optimizer_step"] != target_steps:
        raise ValueError("Epoch checkpoint schedule does not end at the target step")
    return schedule


def summarize_epoch_curve(
    config_path: str | Path, *, repo_root: str | Path
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    raw = load_config(config_path)
    config = resolve_config(raw, repo_root=repo)
    run_root = Path(config["experiment"]["run_root"])
    schedule = epoch_checkpoint_schedule(raw)
    all_steps = [row["optimizer_step"] for row in schedule]
    registered_steps = [
        int(value) for value in raw["dose_provenance"]["probe_optimizer_steps"]
    ]
    if not set(registered_steps).issubset(all_steps):
        raise ValueError("Registered behavior probes are not epoch boundaries")

    # Preserve the original 5/10-epoch report and primary gate unchanged.
    registered_summary = summarize_registered_dose(config_path, repo_root=repo)

    step_summaries: dict[str, Any] = {}
    cell_audits: dict[str, Any] = {}
    for step in all_steps:
        audits: dict[str, Any] = {}
        for seed_value in config["seeds"]["students"]:
            seed = int(seed_value)
            for condition in ("control", "treatment"):
                key = f"{condition}-seed-{seed}"
                model_root = (
                    run_root / "models" / "students" / condition / f"seed-{seed}"
                )
                checkpoint_manifest = json.loads(
                    (model_root / "dose_checkpoint_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (
                    checkpoint_manifest.get("checkpoint_scope")
                    != "all_epoch_boundaries"
                    or checkpoint_manifest.get("audited_optimizer_steps") != all_steps
                ):
                    raise ValueError(
                        f"All-epoch checkpoint audit is missing or incomplete: {model_root}"
                    )
                audits[key] = _verify_behavior_cell(
                    raw,
                    config,
                    run_root=run_root,
                    step=step,
                    condition=condition,
                    seed=seed,
                )
        behavior_root = run_root / "evaluations" / "dose" / f"step-{step}"
        step_summaries[str(step)] = summarize_paired_behavior(
            config,
            behavior_root=behavior_root,
            output_path=behavior_root / "paired_summary.json",
        )
        cell_audits[str(step)] = audits

    strongest_step = max(
        all_steps,
        key=lambda step: float(step_summaries[str(step)]["mean_paired_delta"]),
    )
    result = {
        "schema_version": 1,
        "run_id": raw["experiment"]["id"],
        "training_config_sha256": sha256_value(raw),
        "replication_unit": "paired_student_seed",
        "evaluation_scope": "all_retained_epoch_boundaries",
        "analysis_status": (
            "additive user-requested epoch curve; registered epochs 5 and 10 and the "
            "epoch-10 primary gate remain unchanged"
        ),
        "epoch_checkpoint_schedule": schedule,
        "registered_probe_optimizer_steps": registered_steps,
        "registered_dose_summary": registered_summary,
        "step_summaries": step_summaries,
        "cell_artifact_audits": cell_audits,
        "descriptive_strongest_observed_step": strongest_step,
        "descriptive_strongest_observed_mean_paired_delta": float(
            step_summaries[str(strongest_step)]["mean_paired_delta"]
        ),
    }
    destination = run_root / "evaluations" / "dose" / "epoch_curve_summary.json"
    write_json_atomic(destination, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    result = summarize_epoch_curve(args.config, repo_root=args.repo_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
