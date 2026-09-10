#!/usr/bin/env python3
"""Export compact base/teacher/student decoded J-space token samples."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from silent_transfer.modeling import load_model, load_tokenizer, release_model
from silent_transfer.provenance import sha256_file
from sst_readout.collection import paired_context_alignment_sha256
from sst_readout.logit_lens import FixedBaseDecoder
from sst_readout.serialization import load_collected_readouts

FINAL_HIDDEN_LAYER = 41


def _cell_spec(raw: list[str]) -> tuple[str, Path, Path, Path]:
    if len(raw) != 4 or not raw[0]:
        raise argparse.ArgumentTypeError("CELL requires ID BASE TEACHER STUDENT")
    return raw[0], Path(raw[1]), Path(raw[2]), Path(raw[3])


def _vectors(table, layer: int) -> torch.Tensor:
    if layer == FINAL_HIDDEN_LAYER:
        return table.final_hidden
    if table.jspace_by_layer is None or layer not in table.jspace_by_layer:
        raise ValueError(f"{table.model_id} lacks J-space layer {layer}")
    return table.jspace_by_layer[layer]


@torch.inference_mode()
def _membership_counts(
    table,
    decoder: FixedBaseDecoder,
    *,
    layers: tuple[int, ...],
    top_n: int,
    row_batch_size: int,
) -> dict[str, dict[int, Counter[int]]]:
    anchors = tuple(dict.fromkeys(row.anchor_id for row in table.rows))
    result = {anchor: {layer: Counter() for layer in layers} for anchor in anchors}
    for layer in layers:
        vectors = _vectors(table, layer)
        for start in range(0, vectors.shape[0], row_batch_size):
            stop = min(start + row_batch_size, vectors.shape[0])
            logits = decoder(vectors[start:stop])
            top_ids = torch.topk(logits, k=top_n, dim=-1).indices.cpu().tolist()
            for offset, token_ids in enumerate(top_ids):
                anchor = table.rows[start + offset].anchor_id
                result[anchor][layer].update(map(int, token_ids))
            del logits, top_ids
    return result


def _compact_rows(
    tokenizer,
    inventories: dict[str, dict[str, dict[int, Counter[int]]]],
    *,
    anchor: str,
    layers: tuple[int, ...],
    row_limit: int,
) -> list[dict[str, Any]]:
    token_ids: set[int] = set()
    for model in ("base", "teacher", "student"):
        for layer in layers:
            token_ids.update(inventories[model][anchor][layer])

    def counts(model: str, token_id: int) -> list[int]:
        return [int(inventories[model][anchor][layer][token_id]) for layer in layers]

    ranked: list[tuple[float, int, list[int], list[int], list[int]]] = []
    for token_id in token_ids:
        base = counts("base", token_id)
        teacher = counts("teacher", token_id)
        student = counts("student", token_id)
        base_anchored_change = sum(
            abs(t - b) + abs(s - b) for b, t, s in zip(base, teacher, student, strict=True)
        )
        prevalence = sum(max(b, t, s) for b, t, s in zip(base, teacher, student, strict=True))
        ranked.append((2 * base_anchored_change + prevalence, token_id, base, teacher, student))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    rows = []
    for _, token_id, base, teacher, student in ranked[:row_limit]:
        token = tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False, skip_special_tokens=False
        )
        rows.append(
            {
                "token_id": token_id,
                "token": token,
                "base": base,
                "teacher": teacher,
                "student": student,
            }
        )
    return rows


def _export_cell(
    cell_id: str,
    paths: tuple[Path, Path, Path],
    decoder: FixedBaseDecoder,
    tokenizer,
    *,
    split: str,
    layers: tuple[int, ...],
    top_n: int,
    row_batch_size: int,
    row_limit: int,
    counts_cache: dict[str, dict[str, dict[int, Counter[int]]]],
) -> dict[str, Any]:
    base_path, teacher_path, student_path = paths
    base = load_collected_readouts(base_path).subset(split)
    teacher = load_collected_readouts(teacher_path).subset(split)
    student = load_collected_readouts(student_path).subset(split)
    pairing = {
        "teacher_base": paired_context_alignment_sha256(teacher, base, require_jspace=True),
        "student_base": paired_context_alignment_sha256(student, base, require_jspace=True),
    }
    if tuple(base.source_layers) != tuple(range(41)):
        raise ValueError(f"{cell_id}: expected source layers 0 through 40")
    if len(base.rows) != len(student.rows) or len(base.rows) != len(teacher.rows):
        raise ValueError(f"{cell_id}: split row counts do not match")
    source_sha256 = {
        name: sha256_file(path)
        for name, path in zip(("base", "teacher", "student"), paths, strict=True)
    }
    inventories = {}
    for name, table in (("base", base), ("teacher", teacher), ("student", student)):
        cache_key = source_sha256[name]
        if cache_key not in counts_cache:
            counts_cache[cache_key] = _membership_counts(
                table,
                decoder,
                layers=layers,
                top_n=top_n,
                row_batch_size=row_batch_size,
            )
        inventories[name] = counts_cache[cache_key]
    anchors = tuple(inventories["base"])
    if set(anchors) != set(inventories["teacher"]) or set(anchors) != set(
        inventories["student"]
    ):
        raise ValueError(f"{cell_id}: anchor inventories do not match")
    prompts_per_anchor: dict[str, int] = defaultdict(int)
    for row in base.rows:
        prompts_per_anchor[row.anchor_id] += 1
    if len(set(prompts_per_anchor.values())) != 1:
        raise ValueError(f"{cell_id}: unequal prompt counts by anchor")
    payload = {
        "schema_version": 1,
        "cell_id": cell_id,
        "metric": "top_n_membership_count",
        "top_n": top_n,
        "prompt_count": next(iter(prompts_per_anchor.values())),
        "split": split,
        "layers": list(layers),
        "layer_41_semantics": "fixed-base-decoder final hidden state",
        "token_identity_rule": "exact tokenizer token id; decoded text is display-only",
        "pairing_sha256": pairing,
        "source_artifacts": {
            name: {
                "path": str(path),
                "sha256": source_sha256[name],
                "metadata_sha256": sha256_file(path.with_suffix(path.suffix + ".json")),
            }
            for name, path in zip(("base", "teacher", "student"), paths, strict=True)
        },
        "anchors": {
            anchor: {
                "rows": _compact_rows(
                    tokenizer,
                    inventories,
                    anchor=anchor,
                    layers=layers,
                    row_limit=row_limit,
                )
            }
            for anchor in anchors
        },
    }
    del base, teacher, student, inventories
    gc.collect()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell",
        nargs=4,
        action="append",
        metavar=("ID", "BASE", "TEACHER", "STUDENT"),
        required=True,
    )
    parser.add_argument("--model-id", default="google/gemma-2-9b-it")
    parser.add_argument(
        "--model-revision",
        default="11c9b309abf73637e4b6f9a3fa1e92e615547819",
    )
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--split", default="student_evaluation")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=200)
    parser.add_argument("--row-batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells = [_cell_spec(item) for item in args.cell]
    if len({item[0] for item in cells}) != len(cells):
        raise ValueError("cell IDs must be unique")
    if args.top_n <= 0 or args.row_limit <= 0 or args.row_batch_size <= 0:
        raise ValueError("top-n, row-limit, and row-batch-size must be positive")

    model_config = {
        "id": args.model_id,
        "revision": args.model_revision,
        "tokenizer_revision": args.tokenizer_revision or args.model_revision,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
    }
    if args.local_files_only:
        model_config["local_files_only"] = True
    tokenizer = load_tokenizer(model_config)
    model = load_model(model_config)
    decoder = FixedBaseDecoder.from_hf_model(
        model,
        decoder_id=f"{args.model_id}@{args.model_revision}",
        deep_copy=True,
        device=args.device,
    )
    release_model(model)
    layers = tuple(range(42))
    output = {
        "schema_version": "jspace-student-token-samples-v1",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "decoder_id": decoder.decoder_id,
        "layers": list(layers),
        "cells": {},
    }
    counts_cache: dict[str, dict[str, dict[int, Counter[int]]]] = {}
    for cell_id, base, teacher, student in cells:
        output["cells"][cell_id] = _export_cell(
            cell_id,
            (base, teacher, student),
            decoder,
            tokenizer,
            split=args.split,
            layers=layers,
            top_n=args.top_n,
            row_batch_size=args.row_batch_size,
            row_limit=args.row_limit,
            counts_cache=counts_cache,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
