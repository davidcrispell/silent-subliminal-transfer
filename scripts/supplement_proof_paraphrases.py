#!/usr/bin/env python3
"""Resume a proof cell with a provenance-preserving 512-prompt supplement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from silent_transfer.config import load_config, resolve_config
from silent_transfer.generation import generate_condition, split_single_condition_carriers
from silent_transfer.proof_supplement import (
    FROZEN_SUPPLEMENT_SIZE,
    merge_proof_generation_banks,
    prepare_proof_supplement_prompts,
    supplement_generation_config,
)
from silent_transfer.provenance import sha256_value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    raw_config = load_config(args.config)
    config = resolve_config(raw_config, repo_root=repo_root)
    config["_protocol_config_sha256"] = sha256_value(raw_config)
    run_root = Path(config["experiment"]["run_root"])
    data = run_root / "data"
    paths = {
        "original_prompts": data / "carrier_prompts.jsonl",
        "supplement_prompts": data / "carrier_prompts.supplement.jsonl",
        "original_raw": data / "raw_treatment.jsonl",
        "supplement_raw": data / "raw_treatment.supplement.jsonl",
        "augmented_raw": data / "raw_treatment.augmented.jsonl",
        "single": data / "single",
    }

    prompt_stats = prepare_proof_supplement_prompts(
        config,
        original_prompt_path=paths["original_prompts"],
        supplement_prompt_path=paths["supplement_prompts"],
        repo_root=repo_root,
    )
    generation_config, _ = supplement_generation_config(config)
    generation_stats = generate_condition(
        generation_config,
        condition_name="treatment",
        prompt_path=paths["supplement_prompts"],
        output_path=paths["supplement_raw"],
        repo_root=repo_root,
    )
    merge_stats = merge_proof_generation_banks(
        config,
        original_prompt_path=paths["original_prompts"],
        supplement_prompt_path=paths["supplement_prompts"],
        original_raw_path=paths["original_raw"],
        supplement_raw_path=paths["supplement_raw"],
        augmented_raw_path=paths["augmented_raw"],
        repo_root=repo_root,
    )
    split_stats = split_single_condition_carriers(
        config,
        condition_name="treatment",
        source_path=paths["augmented_raw"],
        output_dir=paths["single"],
        repo_root=repo_root,
    )
    result = {
        "schema_version": 1,
        "supplement_size": FROZEN_SUPPLEMENT_SIZE,
        "prompts": prompt_stats,
        "generation": generation_stats,
        "merge": merge_stats,
        "split": split_stats,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
