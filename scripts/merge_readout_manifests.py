#!/usr/bin/env python3
"""Merge compatible J-Lens prompt manifests without changing prompt records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ENVELOPE_KEYS = (
    "schema_version",
    "tokenizer_id",
    "tokenizer_revision",
    "max_length",
    "add_special_tokens",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    reference = payloads[0]
    for path, payload in zip(args.input[1:], payloads[1:], strict=True):
        mismatches = [
            key for key in ENVELOPE_KEYS if payload.get(key) != reference.get(key)
        ]
        if mismatches:
            raise ValueError(f"incompatible manifest {path}: {mismatches}")

    prompts: list[dict] = []
    seen_ids: set[str] = set()
    for path, payload in zip(args.input, payloads, strict=True):
        for prompt in payload["prompts"]:
            prompt_id = str(prompt["prompt_id"])
            if prompt_id in seen_ids:
                raise ValueError(f"duplicate prompt_id {prompt_id!r} from {path}")
            seen_ids.add(prompt_id)
            prompts.append(prompt)

    output = {key: reference[key] for key in ENVELOPE_KEYS}
    output["source_manifests"] = [
        {"path": str(path), "sha256": sha256(path)} for path in args.input
    ]
    output["prompts"] = prompts
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prompt_count": len(prompts),
                "splits": sorted({str(prompt["split"]) for prompt in prompts}),
                "sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
