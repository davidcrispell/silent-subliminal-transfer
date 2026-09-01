#!/usr/bin/env python3
"""Verify one collected J-Lens table before reuse or downstream analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jlens_readout import adapter_fingerprint, configured_lens_provenance, load_prompt_manifest

from sst_readout.serialization import load_collected_readouts


def comma_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in raw.split(",") if value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readout", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--attn-implementation", required=True)
    parser.add_argument("--lens-provenance", type=Path, required=True)
    parser.add_argument("--manifest-source", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--layers", type=comma_ints, required=True)
    args = parser.parse_args()

    table = load_collected_readouts(args.readout)
    if table.model_id != args.model_label:
        raise ValueError(f"model label mismatch: {table.model_id!r}")
    if table.source_layers != args.layers:
        raise ValueError(
            f"source layer mismatch: {table.source_layers!r} != {args.layers!r}"
        )
    provenance = configured_lens_provenance(args.lens_provenance)
    if provenance.model_repo != args.model_id:
        raise ValueError("configured model does not match frozen lens provenance")
    if provenance.model_revision != args.model_revision:
        raise ValueError("configured model revision does not match frozen lens provenance")
    if table.lens_provenance_id != provenance.stable_id:
        raise ValueError("frozen lens provenance mismatch")
    if table.lens_artifact_sha256 != provenance.expected_sha256:
        raise ValueError("frozen lens artifact hash mismatch")
    expected_revision = f"{args.model_revision}+attn:{args.attn_implementation}"
    if args.adapter is not None:
        expected_revision += f"+peft-sha256:{adapter_fingerprint(args.adapter)}"
    if table.model_revision != expected_revision:
        raise ValueError(
            f"checkpoint identity mismatch: {table.model_revision!r} != {expected_revision!r}"
        )
    if table.jspace_by_layer is None:
        raise ValueError("collected table has no J-Lens coordinates")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        use_fast=True,
        local_files_only=True,
    )
    expected_position = load_prompt_manifest(
        args.manifest_source,
        tokenizer,
        default_tokenizer_id=args.model_id,
        default_tokenizer_revision=args.model_revision,
    )
    if table.manifest_sha256 != expected_position.manifest_sha256:
        raise ValueError("current prompt manifest does not match collected readout")
    metadata_path = args.readout.with_suffix(args.readout.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "tensor_path": args.readout.name,
        "model_id": table.model_id,
        "model_revision": table.model_revision,
        "manifest_sha256": table.manifest_sha256,
        "n_rows": len(table.rows),
        "source_layers": list(table.source_layers),
        "lens_provenance_id": table.lens_provenance_id,
        "lens_artifact_sha256": table.lens_artifact_sha256,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"readout metadata {key} mismatch")
    position_path = args.readout.with_suffix(args.readout.suffix + ".manifest.json")
    position = json.loads(position_path.read_text(encoding="utf-8"))
    expected_position_payload = expected_position.as_dict(include_prompts=True)
    if position != expected_position_payload:
        raise ValueError("position manifest differs from the freshly rendered frozen manifest")
    print(
        json.dumps(
            {
                "readout": str(args.readout),
                "model_id": table.model_id,
                "rows": len(table.rows),
                "layers": list(table.source_layers),
                "splits": sorted({row.split for row in table.rows}),
                "manifest_sha256": table.manifest_sha256,
                "lens_provenance_id": table.lens_provenance_id,
                "lens_artifact_sha256": table.lens_artifact_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
