#!/usr/bin/env python3
"""Measure direction-specific student changes in the frozen Gemma J-space."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from silent_transfer.cloze import CANDIDATE_ANIMALS, build_cloze_prompt_plan
from silent_transfer.config import load_config, resolve_config
from silent_transfer.modeling import (
    load_model,
    load_tokenizer,
    place_for_inference,
    release_model,
)
from silent_transfer.provenance import (
    adapter_artifact_hashes,
    sha256_file,
    sha256_value,
    write_json_atomic,
)
from sst_readout.artifact import load_frozen_lens_from_hub
from sst_readout.collection import (
    PromptSpec,
    build_position_manifest,
    collect_hf_hidden_states,
)
from sst_readout.provenance import GEMMA_2_9B_IT_PUBLIC_JLENS
from sst_readout.steering import (
    NEURONPEDIA_JLENS_STEERING_COMMIT,
    build_jlens_token_directions,
)


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _resolve_module(root: Any, paths: tuple[str, ...]) -> Any:
    for path in paths:
        value = root
        try:
            for component in path.split("."):
                value = getattr(value, component)
        except AttributeError:
            continue
        return value
    raise ValueError(f"could not resolve any module path in {paths}")


def _paired_bootstrap(values: list[float], *, seed: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap values must be one nonempty finite vector")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(10_000, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_replicates": 10_000,
        "bootstrap_unit": "paired_cloze_prompt",
    }


def _load_and_derive(
    *,
    config: dict[str, Any],
    tokenizer: Any,
    manifest: Any,
    lens: Any,
    label: str,
    adapter: Path | None,
    layers: tuple[int, ...],
    candidate_ids: dict[str, int],
    fixed_norm: torch.nn.Module,
    candidate_weight: torch.Tensor,
    softcap: float | None,
    raw_directions: dict[str, dict[int, torch.Tensor]],
) -> list[dict[str, Any]]:
    model = load_model(config["model"], adapter_path=None if adapter is None else str(adapter))
    device = place_for_inference(model)
    rows: list[dict[str, Any]] = []
    try:
        collected = collect_hf_hidden_states(
            model,
            tokenizer,
            manifest,
            model_id=label,
            model_revision=config["model"]["revision"],
            source_layers=layers,
            storage_dtype=torch.float32,
        )
        norm = fixed_norm.to(device=device, dtype=torch.float32)
        weights = candidate_weight.to(device=device, dtype=torch.float32)
        for layer in layers:
            hidden = collected.hidden_by_layer[layer].to(device=device, dtype=torch.float32)
            matrix = lens.jacobian(layer).to(device=device, dtype=torch.float32)
            jspace = hidden @ matrix.T
            logits = norm(jspace) @ weights.T
            if softcap is not None:
                logits = torch.tanh(logits / softcap) * softcap
            probabilities = torch.softmax(logits, dim=-1)
            wolf_direction = raw_directions["wolf"][layer].to(
                device=device, dtype=torch.float32
            )
            dog_direction = raw_directions["dog"][layer].to(device=device, dtype=torch.float32)
            raw_axis = hidden @ (wolf_direction - dog_direction)
            logits_cpu = logits.detach().cpu()
            probabilities_cpu = probabilities.detach().cpu()
            raw_axis_cpu = raw_axis.detach().cpu()
            for index, identity in enumerate(collected.rows):
                selected_logits = {
                    animal: float(logits_cpu[index, animal_index].item())
                    for animal_index, animal in enumerate(CANDIDATE_ANIMALS)
                }
                selected_probabilities = {
                    animal: float(probabilities_cpu[index, animal_index].item())
                    for animal_index, animal in enumerate(CANDIDATE_ANIMALS)
                }
                rows.append(
                    {
                        "schema_version": 1,
                        "model": label,
                        "prompt_id": identity.prompt_id,
                        "layer": layer,
                        "candidate_token_ids": candidate_ids,
                        "jspace_candidate_logits": selected_logits,
                        "jspace_candidate_probabilities": selected_probabilities,
                        "jspace_wolf_minus_dog_logit": (
                            selected_logits["wolf"] - selected_logits["dog"]
                        ),
                        "raw_wolf_minus_dog_axis_projection": float(raw_axis_cpu[index].item()),
                    }
                )
            del hidden, matrix, jspace, logits, probabilities, raw_axis
    finally:
        release_model(model)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--treatment-adapter", required=True)
    parser.add_argument("--control-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    raw_config = load_config(config_path)
    config_sha256 = sha256_value(raw_config)
    git_commit = _git_head(repo_root)
    if args.expected_git_commit and git_commit != args.expected_git_commit:
        raise ValueError(f"Git identity mismatch: {git_commit} != {args.expected_git_commit}")
    if args.expected_config_sha256 and config_sha256 != args.expected_config_sha256:
        raise ValueError(
            f"config SHA mismatch: {config_sha256} != {args.expected_config_sha256}"
        )
    config = resolve_config(raw_config, repo_root=repo_root)
    if config["experiment"]["kind"] != "jspace_steering_transfer":
        raise ValueError("evaluation requires a J-space steering transfer config")
    if config["jspace_steering"]["neuronpedia_commit"] != NEURONPEDIA_JLENS_STEERING_COMMIT:
        raise RuntimeError("configured Neuronpedia commit is not the audited steering source")

    run_root = Path(config["experiment"]["run_root"])
    adapters = {
        "wolf_student": Path(args.treatment_adapter).resolve(),
        "dog_student": Path(args.control_adapter).resolve(),
    }
    expected_adapters = {
        "wolf_student": (
            run_root / "models" / "students" / "treatment" / "seed-53101" / "final_adapter"
        ).resolve(),
        "dog_student": (
            run_root / "models" / "students" / "control" / "seed-53101" / "final_adapter"
        ).resolve(),
    }
    if adapters != expected_adapters:
        raise RuntimeError(
            f"student adapter path binding mismatch: {adapters} != {expected_adapters}"
        )
    adapter_hashes = {label: adapter_artifact_hashes(path) for label, path in adapters.items()}
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config["model"])
    plans = build_cloze_prompt_plan(tokenizer, condition=None)
    candidate_ids = dict(plans[0]["candidate_token_ids"])
    if any(plan["candidate_token_ids"] != candidate_ids for plan in plans):
        raise RuntimeError("animal candidate token IDs vary across frozen prompts")
    frozen_arms = config["jspace_steering"]["arms"]
    if candidate_ids["wolf"] != int(frozen_arms["treatment"]["token_id"]) or candidate_ids[
        "dog"
    ] != int(frozen_arms["control"]["token_id"]):
        raise RuntimeError("frozen steering IDs do not match the cloze candidate IDs")
    specs = [
        PromptSpec(
            prompt_id=plan["prompt_id"],
            split="held_out_cloze",
            prompt=plan["rendered_context"],
            positions=(-1,),
            anchor_ids=("assistant_next_token",),
        )
        for plan in plans
    ]
    manifest = build_position_manifest(
        tokenizer,
        specs,
        tokenizer_id=config["model"]["id"],
        tokenizer_revision=config["model"]["tokenizer_revision"],
        max_length=128,
        add_special_tokens=False,
    )
    layers = tuple(int(layer) for layer in config["readout"]["preregistered_layers"])
    provenance = GEMMA_2_9B_IT_PUBLIC_JLENS
    frozen = config["readout"]["frozen_artifact"]
    configured_identity = (
        config["model"]["id"],
        config["model"]["revision"],
        frozen["repo"],
        frozen["revision"],
        frozen["filename"],
    )
    expected_identity = (
        provenance.model_repo,
        provenance.model_revision,
        provenance.lens_repo,
        provenance.lens_revision,
        provenance.lens_filename,
    )
    if configured_identity != expected_identity:
        raise RuntimeError("config does not match the pinned model/J-lens provenance")
    offline = (
        os.environ.get("SST_USE_OFFLINE_CACHE") == "1"
        or os.environ.get("HF_HUB_OFFLINE") == "1"
    )
    lens = load_frozen_lens_from_hub(
        provenance,
        cache_dir=os.environ.get("HF_HOME"),
        local_files_only=offline,
        expected_d_model=3584,
        required_layers=layers,
    )

    base_model = load_model(config["model"])
    try:
        final_norm = _resolve_module(
            base_model,
            ("model.norm", "model.model.norm", "base_model.model.model.norm"),
        )
        fixed_norm = copy.deepcopy(final_norm).float().cpu().eval()
        fixed_norm.requires_grad_(False)
        output_weight = base_model.get_output_embeddings().weight.detach()
        candidate_weight = (
            output_weight[[candidate_ids[animal] for animal in CANDIDATE_ANIMALS]].float().cpu()
        )
        text_config = (
            base_model.config.get_text_config()
            if hasattr(base_model.config, "get_text_config")
            else base_model.config
        )
        softcap_value = getattr(text_config, "final_logit_softcapping", None)
        softcap = None if softcap_value is None else float(softcap_value)
        raw_directions = {
            "wolf": build_jlens_token_directions(
                base_model,
                lens,
                token_ids=(candidate_ids["wolf"],),
                layers=layers,
            ),
            "dog": build_jlens_token_directions(
                base_model,
                lens,
                token_ids=(candidate_ids["dog"],),
                layers=layers,
            ),
        }
    finally:
        release_model(base_model)

    rows: list[dict[str, Any]] = []
    for label, adapter in (
        ("frozen_base", None),
        ("wolf_student", adapters["wolf_student"]),
        ("dog_student", adapters["dog_student"]),
    ):
        rows.extend(
            _load_and_derive(
                config=config,
                tokenizer=tokenizer,
                manifest=manifest,
                lens=lens,
                label=label,
                adapter=adapter,
                layers=layers,
                candidate_ids=candidate_ids,
                fixed_norm=fixed_norm,
                candidate_weight=candidate_weight,
                softcap=softcap,
                raw_directions=raw_directions,
            )
        )
    expected_rows = 3 * len(plans) * len(layers)
    if len(rows) != expected_rows:
        raise RuntimeError(f"J-space row count {len(rows)} != expected {expected_rows}")

    records_path = output / "per_prompt_layer.jsonl"
    with records_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    by_key = {(row["model"], row["layer"], row["prompt_id"]): row for row in rows}
    comparisons: dict[str, Any] = {}
    layer_summaries: dict[str, Any] = {}
    for layer in layers:
        for label in ("frozen_base", "wolf_student", "dog_student"):
            selected = [by_key[(label, layer, plan["prompt_id"])] for plan in plans]
            layer_summaries[f"{label}:{layer}"] = {
                "jspace_wolf_minus_dog_logit_mean": float(
                    np.mean([row["jspace_wolf_minus_dog_logit"] for row in selected])
                ),
                "jspace_wolf_probability_mean": float(
                    np.mean([row["jspace_candidate_probabilities"]["wolf"] for row in selected])
                ),
                "jspace_dog_probability_mean": float(
                    np.mean([row["jspace_candidate_probabilities"]["dog"] for row in selected])
                ),
                "raw_wolf_minus_dog_axis_projection_mean": float(
                    np.mean([row["raw_wolf_minus_dog_axis_projection"] for row in selected])
                ),
            }
        for comparison_index, (left, right) in enumerate(
            (
                ("wolf_student", "dog_student"),
                ("wolf_student", "frozen_base"),
                ("dog_student", "frozen_base"),
            )
        ):
            jspace_deltas = [
                by_key[(left, layer, plan["prompt_id"])]["jspace_wolf_minus_dog_logit"]
                - by_key[(right, layer, plan["prompt_id"])]["jspace_wolf_minus_dog_logit"]
                for plan in plans
            ]
            raw_deltas = [
                by_key[(left, layer, plan["prompt_id"])]["raw_wolf_minus_dog_axis_projection"]
                - by_key[(right, layer, plan["prompt_id"])][
                    "raw_wolf_minus_dog_axis_projection"
                ]
                for plan in plans
            ]
            comparisons[f"{left}_minus_{right}:{layer}"] = {
                "jspace_wolf_minus_dog_logit_delta": _paired_bootstrap(
                    jspace_deltas, seed=53101 + 100 * layer + comparison_index
                ),
                "raw_wolf_minus_dog_axis_projection_delta": _paired_bootstrap(
                    raw_deltas, seed=63101 + 100 * layer + comparison_index
                ),
            }

    summary_path = output / "summary.json"
    summary = {
        "schema_version": 1,
        "analysis_status": "prespecified_single_seed_jspace_transfer_pilot",
        "experiment": config["experiment"]["id"],
        "git_commit": git_commit,
        "config_sha256": config_sha256,
        "config_byte_sha256": sha256_file(config_path),
        "model": config["model"],
        "adapters": {label: str(path) for label, path in adapters.items()},
        "adapter_artifact_sha256": adapter_hashes,
        "lens": lens.manifest(),
        "prompt_manifest": manifest.as_dict(include_prompts=True),
        "candidate_token_ids": candidate_ids,
        "layers": list(layers),
        "primary_layer": int(config["jspace_steering"]["layers"][0]),
        "fixed_base_decoder": {
            "final_norm": "frozen base Gemma final RMSNorm in float32",
            "candidate_unembedding_rows": list(CANDIDATE_ANIMALS),
            "final_logit_softcapping": softcap,
            "probability_denominator": "ten frozen animal candidates",
        },
        "layer_summaries": layer_summaries,
        "comparisons": comparisons,
        "records_sha256": sha256_file(records_path),
    }
    write_json_atomic(summary_path, summary)
    write_json_atomic(
        output / "evaluation_complete.json",
        {
            "schema_version": 1,
            "rows": len(rows),
            "records_sha256": sha256_file(records_path),
            "summary_sha256": sha256_file(summary_path),
            "config_sha256": config_sha256,
            "git_commit": git_commit,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
