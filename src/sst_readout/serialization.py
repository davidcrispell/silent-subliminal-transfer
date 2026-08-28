"""Weights-only serialization for collected readout tensors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .artifact import sha256_file
from .collection import CollectedReadouts, RowIdentity


def save_collected_readouts(table: CollectedReadouts, path: str | Path) -> tuple[Path, Path]:
    table.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "model_id": table.model_id,
        "model_revision": table.model_revision,
        "manifest_sha256": table.manifest_sha256,
        "rows": [
            {
                "prompt_id": row.prompt_id,
                "split": row.split,
                "position": row.position,
                "anchor_id": row.anchor_id,
                "token_id": row.token_id,
                "tokenization_sha256": row.tokenization_sha256,
            }
            for row in table.rows
        ],
        "source_layers": list(table.source_layers),
        "hidden_by_layer": dict(table.hidden_by_layer),
        "final_hidden": table.final_hidden,
        "jspace_by_layer": (
            None if table.jspace_by_layer is None else dict(table.jspace_by_layer)
        ),
        "lens_provenance_id": table.lens_provenance_id,
        "lens_artifact_sha256": table.lens_artifact_sha256,
    }
    torch.save(payload, output)
    metadata = output.with_suffix(output.suffix + ".json")
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensor_path": output.name,
                "tensor_sha256": sha256_file(output),
                "model_id": table.model_id,
                "model_revision": table.model_revision,
                "manifest_sha256": table.manifest_sha256,
                "n_rows": len(table.rows),
                "source_layers": list(table.source_layers),
                "lens_provenance_id": table.lens_provenance_id,
                "lens_artifact_sha256": table.lens_artifact_sha256,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output, metadata


def load_collected_readouts(path: str | Path) -> CollectedReadouts:
    source = Path(path)
    metadata_path = source.with_suffix(source.suffix + ".json")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file(source) != metadata["tensor_sha256"]:
            raise ValueError(f"readout tensor digest mismatch for {source}")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported readout serialization schema")
    rows = tuple(RowIdentity(**row) for row in payload["rows"])
    table = CollectedReadouts(
        model_id=payload["model_id"],
        model_revision=payload["model_revision"],
        manifest_sha256=payload["manifest_sha256"],
        rows=rows,
        source_layers=tuple(int(layer) for layer in payload["source_layers"]),
        hidden_by_layer={
            int(layer): tensor for layer, tensor in payload["hidden_by_layer"].items()
        },
        final_hidden=payload["final_hidden"],
        jspace_by_layer=(
            None
            if payload["jspace_by_layer"] is None
            else {int(layer): tensor for layer, tensor in payload["jspace_by_layer"].items()}
        ),
        lens_provenance_id=payload["lens_provenance_id"],
        lens_artifact_sha256=payload["lens_artifact_sha256"],
    )
    table.validate()
    return table
