#!/usr/bin/env python3
"""Write provenance for a post-hoc dense-layer J-Lens analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def comma_ints(raw: str) -> list[int]:
    return [int(value) for value in raw.split(",") if value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--union-manifest", type=Path, required=True)
    parser.add_argument("--layers", type=comma_ints, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code", type=Path, action="append", default=[])
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    output = {
        "schema_version": 1,
        "analysis_status": "posthoc_exploratory",
        "analysis_name": "dense corresponding-layer J-space and token-inventory comparison",
        "git_commit": commit,
        "config_path": str(args.config),
        "config_byte_sha256": sha256(args.config),
        "config_semantic_sha256": protocol["config_sha256"],
        "source_readout_protocol_path": str(args.protocol),
        "source_readout_protocol_sha256": sha256(args.protocol),
        "union_manifest_path": str(args.union_manifest),
        "union_manifest_sha256": sha256(args.union_manifest),
        "lens_provenance": protocol["lens_provenance"],
        "lens_artifact_sha256": protocol["lens_provenance"]["expected_sha256"],
        "requested_jlens_source_layers": args.layers,
        "target_final_block": 41,
        "target_final_block_qualification": (
            "Gemma-2-9B block 41 is the J-Lens target/final_hidden boundary; "
            "the published artifact supplies source Jacobians only through layer 40"
        ),
        "aggregation_status": (
            "transport and teacher holdout reproducibility are evaluated per layer; "
            "whole-band primary summaries use their intersection while all requested "
            "corresponding-layer results remain visible"
        ),
        "correspondence_rule": (
            "teacher layer l is compared only with treatment-minus-control student "
            "layer l; no cross-layer matching or remapping is used"
        ),
        "whole_model_token_rule": (
            "exact token IDs from top-k J-Lens logits are counted per layer and pooled "
            "over qualified layers; decoded strings are display-only"
        ),
        "teacher_holdout_policy": {
            **protocol["teacher_gate"],
            "global_pass_is_descriptive_only": True,
            "primary_aggregation_uses_per_layer_positive_flag": True,
        },
        "transport_policy": protocol["transport"],
        "analysis_code_sha256": {
            str(path): sha256(path) for path in args.code
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
