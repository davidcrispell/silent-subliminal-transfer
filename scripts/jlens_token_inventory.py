#!/usr/bin/env python3
"""Aggregate and compare top-token J-Lens readouts layer by corresponding layer."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from jlens_readout import adapter_fingerprint

from sst_readout.artifact import sha256_file
from sst_readout.collection import assert_aligned, paired_context_alignment_sha256
from sst_readout.logit_lens import FixedBaseDecoder
from sst_readout.provenance import LensProvenance
from sst_readout.serialization import load_collected_readouts

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def comma_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(value) for value in raw.split(",") if value)
    if not values or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("expected unique comma-separated integers")
    return values


def sparse_cosine(left: Counter[int], right: Counter[int]) -> float | None:
    keys = set(left) | set(right)
    dot = sum(left[key] * right[key] for key in keys)
    left_norm = math.sqrt(sum(left[key] ** 2 for key in keys))
    right_norm = math.sqrt(sum(right[key] ** 2 for key in keys))
    if left_norm == 0 or right_norm == 0:
        return None
    return float(dot / (left_norm * right_norm))


def delta(treatment: Counter[int], control: Counter[int]) -> Counter[int]:
    return Counter(
        {key: treatment[key] - control[key] for key in set(treatment) | set(control)}
    )


def counter_payload(
    counter: Counter[int],
    token_text: dict[int, str],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    values = [
        {"token_id": token_id, "token": token_text[token_id], "count": int(count)}
        for token_id, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if count != 0
    ]
    return values if limit is None else values[:limit]


def signed_counter_payload(
    counter: Counter[int],
    token_text: dict[int, str],
    *,
    limit_per_tail: int | None = None,
) -> dict[str, Any]:
    """Serialize both signed tails independently so truncation cannot hide depletion."""

    positive = Counter({token_id: count for token_id, count in counter.items() if count > 0})
    negative = Counter({token_id: count for token_id, count in counter.items() if count < 0})
    positive_values = counter_payload(positive, token_text, limit=limit_per_tail)
    negative_values = [
        {"token_id": token_id, "token": token_text[token_id], "count": int(count)}
        for token_id, count in sorted(negative.items(), key=lambda item: (item[1], item[0]))
    ]
    if limit_per_tail is not None:
        negative_values = negative_values[:limit_per_tail]
    return {
        "positive": positive_values,
        "negative": negative_values,
        "positive_token_count": len(positive),
        "negative_token_count": len(negative),
        "positive_mass": int(sum(positive.values())),
        "negative_mass": int(sum(negative.values())),
        "limit_per_tail": limit_per_tail,
    }


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def artifact_bundle(path: Path, table) -> dict[str, str]:
    """Verify and fingerprint a tensor, metadata sidecar, and position manifest."""

    metadata_path = path.with_suffix(path.suffix + ".json")
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    metadata = load_json(metadata_path, label="readout metadata")
    manifest = load_json(manifest_path, label="position manifest")
    tensor_sha256 = sha256_file(path)
    expected_metadata = {
        "tensor_sha256": tensor_sha256,
        "model_id": table.model_id,
        "model_revision": table.model_revision,
        "manifest_sha256": table.manifest_sha256,
        "n_rows": len(table.rows),
        "source_layers": list(table.source_layers),
        "lens_provenance_id": table.lens_provenance_id,
        "lens_artifact_sha256": table.lens_artifact_sha256,
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"{metadata_path}: {key} identity mismatch")
    if manifest.get("manifest_sha256") != table.manifest_sha256:
        raise ValueError(f"{manifest_path}: manifest_sha256 identity mismatch")
    if manifest.get("n_rows") != len(table.rows):
        raise ValueError(f"{manifest_path}: row-count identity mismatch")
    return {
        "tensor_path": str(path),
        "tensor_sha256": tensor_sha256,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "position_manifest_path": str(manifest_path),
        "position_manifest_file_sha256": sha256_file(manifest_path),
        "position_manifest_semantic_sha256": table.manifest_sha256,
    }


def validate_provenance(
    path: Path,
    *,
    layers: tuple[int, ...],
    model_id: str,
    model_revision: str,
) -> tuple[dict[str, Any], LensProvenance, dict[str, Any]]:
    provenance = load_json(path, label="dense-analysis provenance")
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported dense-analysis provenance schema")
    if provenance.get("requested_jlens_source_layers") != list(layers):
        raise ValueError("provenance layer set/order does not match --layers")
    raw_lens = provenance.get("lens_provenance")
    if not isinstance(raw_lens, dict):
        raise TypeError("provenance does not contain frozen lens_provenance")
    lens = LensProvenance(**raw_lens)
    lens.validate()
    if lens.model_repo != model_id or lens.model_revision != model_revision:
        raise ValueError("CLI model identity does not match frozen lens provenance")
    expected_lens_sha256 = require_sha256(
        lens.expected_sha256, label="lens_provenance.expected_sha256"
    )
    if provenance.get("lens_artifact_sha256") != expected_lens_sha256:
        raise ValueError("provenance lens artifact hash mismatch")

    for path_key, hash_key in (
        ("config_path", "config_byte_sha256"),
        ("source_readout_protocol_path", "source_readout_protocol_sha256"),
        ("union_manifest_path", "union_manifest_sha256"),
    ):
        source_path = Path(str(provenance.get(path_key, "")))
        expected_sha256 = require_sha256(provenance.get(hash_key), label=hash_key)
        if not source_path.is_file() or sha256_file(source_path) != expected_sha256:
            raise ValueError(f"provenance-bound source mismatch: {source_path}")
    protocol = load_json(
        Path(provenance["source_readout_protocol_path"]),
        label="source readout protocol",
    )
    if protocol.get("config_sha256") != provenance.get("config_semantic_sha256"):
        raise ValueError("provenance and readout protocol use different configs")

    code_hashes = provenance.get("analysis_code_sha256")
    if not isinstance(code_hashes, dict):
        raise TypeError("provenance does not bind analysis code")
    this_script = Path(__file__).resolve()
    matches = [
        expected
        for source, expected in code_hashes.items()
        if Path(source).name == this_script.name
    ]
    if len(matches) != 1 or sha256_file(this_script) != matches[0]:
        raise ValueError("running token-inventory code is not provenance-bound")
    return provenance, lens, protocol


def validate_teacher_gate(
    path: Path,
    *,
    layers: tuple[int, ...],
    teacher_treatment,
    teacher_control,
    lens: LensProvenance,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], list[int], dict[str, str]]:
    gate = load_json(path, label="teacher H3 gate")
    if gate.get("gate") != "H3":
        raise ValueError("teacher gate is not H3")
    raw_layers = gate.get("layers")
    if not isinstance(raw_layers, dict) or set(raw_layers) != {str(layer) for layer in layers}:
        raise ValueError("teacher H3 gate layer set does not match --layers")
    positive_layers: list[int] = []
    for layer in layers:
        record = raw_layers[str(layer)]
        if not isinstance(record, dict) or record.get("layer") != layer:
            raise ValueError(f"invalid H3 record for layer {layer}")
        cosine = float(record["calibration_validation_cosine"])
        if not math.isfinite(cosine) or bool(record.get("positive")) != (cosine > 0):
            raise ValueError(f"inconsistent H3 positivity at layer {layer}")
        if cosine > 0:
            positive_layers.append(layer)

    pairing_hashes: dict[str, str] = {}
    for phase in ("calibration", "validation"):
        direction = gate.get(f"{phase}_direction")
        if not isinstance(direction, dict):
            raise TypeError(f"teacher gate lacks {phase}_direction")
        split = gate.get(f"{phase}_split")
        if not isinstance(split, str) or direction.get("source_split") != split:
            raise ValueError(f"teacher H3 {phase} split identity mismatch")
        if direction.get("coordinate") != "jspace" or direction.get("layers") != list(layers):
            raise ValueError(f"teacher H3 {phase} coordinate/layer mismatch")
        if direction.get("lens_provenance_id") != lens.stable_id:
            raise ValueError(f"teacher H3 {phase} lens provenance mismatch")
        if direction.get("lens_artifact_sha256") != lens.expected_sha256:
            raise ValueError(f"teacher H3 {phase} lens artifact mismatch")
        if direction.get("teacher_model_id") != teacher_treatment.model_id:
            raise ValueError(f"teacher H3 {phase} treatment identity mismatch")
        if direction.get("control_model_id") != teacher_control.model_id:
            raise ValueError(f"teacher H3 {phase} control identity mismatch")
        observed_pairing = paired_context_alignment_sha256(
            teacher_treatment.subset(split),
            teacher_control.subset(split),
            require_jspace=True,
        )
        if gate.get(f"{phase}_pairing_sha256") != observed_pairing:
            raise ValueError(f"teacher H3 {phase} pairing hash mismatch")
        if direction.get("pairing_sha256") != observed_pairing:
            raise ValueError(f"teacher H3 {phase} direction pairing hash mismatch")
        observed_prompt_ids = [
            row.prompt_id for row in teacher_treatment.subset(split).rows
        ]
        if direction.get("source_prompt_ids") != observed_prompt_ids:
            raise ValueError(f"teacher H3 {phase} prompt identity mismatch")
        pairing_hashes[phase] = observed_pairing

    if gate.get("readout_protocol_sha256") != provenance.get(
        "source_readout_protocol_sha256"
    ):
        raise ValueError("teacher gate and dense provenance use different protocols")
    if gate.get("config_sha256") != provenance.get("config_semantic_sha256"):
        raise ValueError("teacher gate and dense provenance use different configs")
    return gate, positive_layers, pairing_hashes


def validate_transport(
    path: Path,
    *,
    layers: tuple[int, ...],
    decoder_id: str,
    lens: LensProvenance,
    tables: dict[str, Any],
    student_seeds: tuple[str, ...],
) -> tuple[dict[str, Any], list[int]]:
    transport = load_json(path, label="transport calibration")
    if transport.get("decoder_id") != decoder_id:
        raise ValueError("transport decoder identity mismatch")
    if transport.get("lens_provenance_id") != lens.stable_id:
        raise ValueError("transport lens provenance mismatch")
    if transport.get("lens_artifact_sha256") != lens.expected_sha256:
        raise ValueError("transport lens artifact mismatch")
    raw_layers = transport.get("layers")
    if not isinstance(raw_layers, dict) or set(raw_layers) != {str(layer) for layer in layers}:
        raise ValueError("transport layer set does not match --layers")

    expected_checkpoints = {
        "base": "base",
        **{
            f"{condition}_seed_{seed}": f"student_{condition}_{seed}"
            for seed in student_seeds
            for condition in ("treatment", "control")
        },
    }
    checkpoints = transport.get("checkpoints")
    if not isinstance(checkpoints, dict) or set(checkpoints) != set(expected_checkpoints):
        raise ValueError("transport checkpoint set does not match analyzed tables")
    transport_split = transport.get("split")
    for checkpoint_name, table_name in expected_checkpoints.items():
        checkpoint = checkpoints[checkpoint_name]
        if checkpoint.get("model_id") != tables[table_name].model_id:
            raise ValueError(f"transport model identity mismatch for {checkpoint_name}")
        if checkpoint.get("split") != transport_split:
            raise ValueError(f"transport split mismatch for {checkpoint_name}")
        for lens_kind in ("jlens", "logit_lens"):
            distances = checkpoint.get(lens_kind)
            if not isinstance(distances, dict) or set(distances) != {
                str(layer) for layer in layers
            }:
                raise ValueError(
                    f"transport {lens_kind} layer set mismatch for {checkpoint_name}"
                )

    variant_names = set(expected_checkpoints) - {"base"}
    eligible_layers: list[int] = []
    for layer in layers:
        record = raw_layers[str(layer)]
        if record.get("layer") != layer:
            raise ValueError(f"transport layer identity mismatch at {layer}")
        if set(record.get("variant_mean_kl", {})) != variant_names:
            raise ValueError(f"transport variant set mismatch at layer {layer}")
        if bool(record.get("eligible")):
            eligible_layers.append(layer)
    return transport, eligible_layers


@torch.inference_mode()
def count_top_tokens(
    table,
    decoder: FixedBaseDecoder,
    *,
    split: str,
    layers: tuple[int, ...],
    cutoffs: tuple[int, ...],
    row_batch_size: int,
) -> dict[int, dict[int, Counter[int]]]:
    selected = table.subset(split)
    if selected.jspace_by_layer is None:
        raise ValueError(f"{table.model_id} has no J-space readouts")
    missing = sorted(set(layers) - set(selected.source_layers))
    if missing:
        raise ValueError(f"{table.model_id} lacks requested layers {missing}")
    maximum = max(cutoffs)
    result: dict[int, dict[int, Counter[int]]] = {}
    for layer in layers:
        by_cutoff = {cutoff: Counter() for cutoff in cutoffs}
        vectors = selected.jspace_by_layer[layer]
        for start in range(0, vectors.shape[0], row_batch_size):
            logits = decoder(vectors[start : start + row_batch_size])
            ids = torch.topk(logits, k=maximum, dim=-1).indices.cpu()
            for cutoff in cutoffs:
                by_cutoff[cutoff].update(int(value) for value in ids[:, :cutoff].reshape(-1))
        result[layer] = by_cutoff
    return result


def pooled_counter(
    inventory: dict[int, dict[int, Counter[int]]],
    *,
    layers: list[int],
    cutoff: int,
) -> Counter[int]:
    pooled: Counter[int] = Counter()
    for layer in layers:
        pooled.update(inventory[layer][cutoff])
    return pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--teacher-treatment", type=Path, required=True)
    parser.add_argument("--teacher-control", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--student-pair",
        nargs=3,
        action="append",
        metavar=("SEED", "TREATMENT", "CONTROL"),
        required=True,
    )
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--teacher-gate", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--split", default="student_evaluation")
    parser.add_argument("--layers", type=comma_ints, required=True)
    parser.add_argument("--cutoffs", type=comma_ints, default=(1, 5, 10, 20))
    parser.add_argument("--row-batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if min(args.layers) < 0:
        raise ValueError("source layers must be nonnegative")
    if min(args.cutoffs) <= 0:
        raise ValueError("top-token cutoffs must be positive")
    if args.row_batch_size <= 0:
        raise ValueError("row batch size must be positive")

    provenance, lens, protocol = validate_provenance(
        args.provenance,
        layers=args.layers,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    tables = {
        "teacher_treatment": load_collected_readouts(args.teacher_treatment),
        "teacher_control": load_collected_readouts(args.teacher_control),
        "base": load_collected_readouts(args.base),
    }
    table_paths = {
        "teacher_treatment": args.teacher_treatment,
        "teacher_control": args.teacher_control,
        "base": args.base,
    }
    student_paths: dict[str, tuple[Path, Path]] = {}
    for seed, treatment_path, control_path in args.student_pair:
        if not seed or seed in student_paths:
            raise ValueError(f"student seed must be unique and nonempty: {seed!r}")
        student_paths[seed] = (Path(treatment_path), Path(control_path))
        tables[f"student_treatment_{seed}"] = load_collected_readouts(treatment_path)
        tables[f"student_control_{seed}"] = load_collected_readouts(control_path)
        table_paths[f"student_treatment_{seed}"] = Path(treatment_path)
        table_paths[f"student_control_{seed}"] = Path(control_path)
    student_seeds = tuple(sorted(student_paths))

    expected_execution_revision = (
        f"{args.model_revision}+attn:{args.attn_implementation}"
    )
    if tables["teacher_treatment"].model_revision != expected_execution_revision:
        raise ValueError("teacher treatment checkpoint identity mismatch")
    if tables["teacher_control"].model_revision != expected_execution_revision:
        raise ValueError("teacher control checkpoint identity mismatch")
    if tables["base"].model_revision != expected_execution_revision:
        raise ValueError("base checkpoint identity mismatch")
    student_revision_prefix = expected_execution_revision + "+peft-sha256:"
    for seed in student_seeds:
        for condition in ("treatment", "control"):
            revision = tables[f"student_{condition}_{seed}"].model_revision
            fingerprint = revision.removeprefix(student_revision_prefix)
            if not revision.startswith(student_revision_prefix) or not _SHA256_RE.fullmatch(
                fingerprint
            ):
                raise ValueError(
                    f"student {condition} seed {seed} checkpoint identity mismatch"
                )
            adapter_path = Path(protocol["student_models"][seed][condition])
            if fingerprint != adapter_fingerprint(adapter_path):
                raise ValueError(
                    f"student {condition} seed {seed} adapter fingerprint mismatch"
                )

    if any(table.source_layers != args.layers for table in tables.values()):
        raise ValueError("every readout table must contain exactly the requested layer set")
    if any(table.lens_provenance_id != lens.stable_id for table in tables.values()):
        raise ValueError("readout tables do not match the provenance-bound lens identity")
    if any(table.lens_artifact_sha256 != lens.expected_sha256 for table in tables.values()):
        raise ValueError("readout tables do not match the provenance-bound lens artifact")
    source_artifacts = {
        name: artifact_bundle(table_paths[name], table) for name, table in tables.items()
    }

    teacher_split = tables["teacher_treatment"].subset(args.split)
    alignment_hashes = {
        "teacher_treatment_vs_teacher_control": paired_context_alignment_sha256(
            teacher_split,
            tables["teacher_control"].subset(args.split),
            require_jspace=True,
        )
    }
    base_split = tables["base"].subset(args.split)
    student_splits = [
        tables[f"student_{condition}_{seed}"].subset(args.split)
        for seed in student_seeds
        for condition in ("treatment", "control")
    ]
    assert_aligned(base_split, *student_splits, require_jspace=True)
    for name in ("teacher_control", "base", *[
        f"student_{condition}_{seed}"
        for seed in student_seeds
        for condition in ("treatment", "control")
    ]):
        alignment_hashes[f"teacher_treatment_vs_{name}"] = (
            paired_context_alignment_sha256(
                teacher_split,
                tables[name].subset(args.split),
                require_jspace=True,
            )
        )

    teacher_gate, h3_positive_layers, h3_pairing_hashes = validate_teacher_gate(
        args.teacher_gate,
        layers=args.layers,
        teacher_treatment=tables["teacher_treatment"],
        teacher_control=tables["teacher_control"],
        lens=lens,
        provenance=provenance,
    )
    decoder_id = (
        f"{args.model_id}@{args.model_revision}+attn:{args.attn_implementation}"
    )
    transport, transport_eligible_layers = validate_transport(
        args.transport,
        layers=args.layers,
        decoder_id=decoder_id,
        lens=lens,
        tables=tables,
        student_seeds=student_seeds,
    )
    aggregate_layers = [
        layer
        for layer in args.layers
        if layer in transport_eligible_layers and layer in h3_positive_layers
    ]
    if not aggregate_layers:
        raise ValueError(
            "no requested layer passed both transport eligibility and H3 positivity"
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if max(args.cutoffs) > int(model.config.vocab_size):
        raise ValueError("top-token cutoff exceeds model vocabulary")
    decoder = FixedBaseDecoder.from_hf_model(
        model,
        decoder_id=decoder_id,
        deep_copy=False,
        device=args.device,
    )
    if decoder.decoder_id != transport.get("decoder_id"):
        raise ValueError("constructed decoder does not match transport calibration")

    inventories = {
        name: count_top_tokens(
            table,
            decoder,
            split=args.split,
            layers=args.layers,
            cutoffs=args.cutoffs,
            row_batch_size=args.row_batch_size,
        )
        for name, table in tables.items()
    }

    token_ids: set[int] = set()
    for inventory in inventories.values():
        for by_cutoff in inventory.values():
            for counts in by_cutoff.values():
                token_ids.update(counts)
    token_text = {
        token_id: str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        for token_id in sorted(token_ids)
    }

    comparisons: dict[str, Any] = {}
    for seed in student_seeds:
        by_cutoff: dict[str, Any] = {}
        for cutoff in args.cutoffs:
            by_layer: dict[str, Any] = {}
            for layer in args.layers:
                teacher_delta = delta(
                    inventories["teacher_treatment"][layer][cutoff],
                    inventories["teacher_control"][layer][cutoff],
                )
                teacher_base_delta = delta(
                    inventories["teacher_treatment"][layer][cutoff],
                    inventories["base"][layer][cutoff],
                )
                student_delta = delta(
                    inventories[f"student_treatment_{seed}"][layer][cutoff],
                    inventories[f"student_control_{seed}"][layer][cutoff],
                )
                student_base_delta = delta(
                    inventories[f"student_treatment_{seed}"][layer][cutoff],
                    inventories["base"][layer][cutoff],
                )
                enriched = Counter(
                    {key: value for key, value in teacher_delta.items() if value > 0}
                )
                enriched_ids = set(enriched)
                student_treatment = inventories[f"student_treatment_{seed}"][layer][cutoff]
                student_control = inventories[f"student_control_{seed}"][layer][cutoff]
                base_counts = inventories["base"][layer][cutoff]
                transport_eligible = layer in transport_eligible_layers
                h3_positive = layer in h3_positive_layers
                by_layer[str(layer)] = {
                    "corresponding_layer": layer,
                    "transport_eligible": transport_eligible,
                    "h3_positive": h3_positive,
                    "aggregate_eligible": transport_eligible and h3_positive,
                    "teacher_student_paired_delta_cosine": sparse_cosine(
                        teacher_delta, student_delta
                    ),
                    "teacher_student_base_anchored_delta_cosine": sparse_cosine(
                        teacher_base_delta, student_base_delta
                    ),
                    "teacher_enriched_token_count": len(enriched_ids),
                    "teacher_enriched_tokens_increasing_in_student": sum(
                        student_delta[token_id] > 0 for token_id in enriched_ids
                    ),
                    "student_delta_mass_on_teacher_enriched_tokens": int(
                        sum(student_delta[token_id] for token_id in enriched_ids)
                    ),
                    "teacher_enriched_tokens_present_in_student_treatment_not_control": sum(
                        student_control[token_id] == 0 and student_treatment[token_id] > 0
                        for token_id in enriched_ids
                    ),
                    "teacher_enriched_tokens_present_in_student_treatment_not_base": sum(
                        base_counts[token_id] == 0 and student_treatment[token_id] > 0
                        for token_id in enriched_ids
                    ),
                    "teacher_treatment_minus_control": signed_counter_payload(
                        teacher_delta, token_text, limit_per_tail=200
                    ),
                    "teacher_treatment_minus_base": signed_counter_payload(
                        teacher_base_delta, token_text, limit_per_tail=200
                    ),
                    "student_treatment_minus_control": signed_counter_payload(
                        student_delta, token_text, limit_per_tail=200
                    ),
                    "student_treatment_minus_base": signed_counter_payload(
                        student_base_delta, token_text, limit_per_tail=200
                    ),
                }
            aggregate_cosines = [
                record["teacher_student_paired_delta_cosine"]
                for layer, record in by_layer.items()
                if int(layer) in aggregate_layers
                and record["teacher_student_paired_delta_cosine"] is not None
            ]
            by_cutoff[str(cutoff)] = {
                "by_layer": by_layer,
                "aggregate_layer_mean_paired_delta_cosine": (
                    None
                    if not aggregate_cosines
                    else float(sum(aggregate_cosines) / len(aggregate_cosines))
                ),
                "aggregate_positive_paired_delta_cosine_layers": sum(
                    value > 0 for value in aggregate_cosines
                ),
                "aggregate_scored_layers": len(aggregate_cosines),
            }
        comparisons[seed] = by_cutoff

    cross_seed_markers: dict[str, Any] = {}
    for cutoff in args.cutoffs:
        marker_layers: dict[str, Any] = {}
        for layer in args.layers:
            teacher_treatment = inventories["teacher_treatment"][layer][cutoff]
            teacher_control = inventories["teacher_control"][layer][cutoff]
            base_counts = inventories["base"][layer][cutoff]
            teacher_delta = delta(teacher_treatment, teacher_control)
            teacher_base_delta = delta(teacher_treatment, base_counts)
            candidate_ids = sorted(
                token_id
                for token_id in set(teacher_delta) | set(teacher_base_delta)
                if teacher_delta[token_id] > 0 and teacher_base_delta[token_id] > 0
            )
            consistent_records: list[dict[str, Any]] = []
            strict_new_records: list[dict[str, Any]] = []
            for token_id in candidate_ids:
                seed_records: dict[str, Any] = {}
                for seed in student_seeds:
                    student_treatment = inventories[f"student_treatment_{seed}"][layer][
                        cutoff
                    ]
                    student_control = inventories[f"student_control_{seed}"][layer][
                        cutoff
                    ]
                    paired_increase = student_treatment[token_id] - student_control[token_id]
                    base_increase = student_treatment[token_id] - base_counts[token_id]
                    seed_records[seed] = {
                        "student_treatment_count": int(student_treatment[token_id]),
                        "student_control_count": int(student_control[token_id]),
                        "base_count": int(base_counts[token_id]),
                        "paired_increase": int(paired_increase),
                        "base_anchored_increase": int(base_increase),
                        "increases_vs_control_and_base": (
                            paired_increase > 0 and base_increase > 0
                        ),
                        "strictly_new_vs_control_and_base": (
                            student_treatment[token_id] > 0
                            and student_control[token_id] == 0
                            and base_counts[token_id] == 0
                        ),
                    }
                record = {
                    "token_id": token_id,
                    "token": token_text[token_id],
                    "teacher_treatment_count": int(teacher_treatment[token_id]),
                    "teacher_control_count": int(teacher_control[token_id]),
                    "base_count": int(base_counts[token_id]),
                    "teacher_paired_increase": int(teacher_delta[token_id]),
                    "teacher_base_anchored_increase": int(teacher_base_delta[token_id]),
                    "students": seed_records,
                }
                if all(
                    seed_record["increases_vs_control_and_base"]
                    for seed_record in seed_records.values()
                ):
                    consistent_records.append(record)
                teacher_strictly_new = (
                    teacher_treatment[token_id] > 0
                    and teacher_control[token_id] == 0
                    and base_counts[token_id] == 0
                )
                if teacher_strictly_new and all(
                    seed_record["strictly_new_vs_control_and_base"]
                    for seed_record in seed_records.values()
                ):
                    strict_new_records.append(record)
            sort_key = lambda record: (
                -record["teacher_paired_increase"],
                -record["teacher_base_anchored_increase"],
                record["token_id"],
            )
            consistent_records.sort(key=sort_key)
            strict_new_records.sort(key=sort_key)
            aggregate_eligible = layer in aggregate_layers
            marker_layers[str(layer)] = {
                "corresponding_layer": layer,
                "aggregate_eligible": aggregate_eligible,
                "consistent_teacherward_enrichment_marker": bool(consistent_records),
                "strict_new_token_marker": bool(strict_new_records),
                "consistent_teacherward_tokens": consistent_records[:200],
                "consistent_teacherward_token_count": len(consistent_records),
                "strict_new_tokens": strict_new_records[:200],
                "strict_new_token_count": len(strict_new_records),
            }
        eligible_records = [
            record
            for layer, record in marker_layers.items()
            if int(layer) in aggregate_layers
        ]
        cross_seed_markers[str(cutoff)] = {
            "by_corresponding_layer": marker_layers,
            "aggregate_eligible_layers_with_consistent_teacherward_enrichment": [
                int(layer)
                for layer, record in marker_layers.items()
                if int(layer) in aggregate_layers
                and record["consistent_teacherward_enrichment_marker"]
            ],
            "aggregate_eligible_layers_with_strict_new_tokens": [
                int(layer)
                for layer, record in marker_layers.items()
                if int(layer) in aggregate_layers and record["strict_new_token_marker"]
            ],
            "consistent_teacherward_enrichment_marker": any(
                record["consistent_teacherward_enrichment_marker"]
                for record in eligible_records
            ),
            "strict_new_token_marker": any(
                record["strict_new_token_marker"] for record in eligible_records
            ),
            "qualification": (
                "descriptive cross-seed marker, not an inferential significance test; "
                "requires the same exact token ID to move teacherward at the same layer"
            ),
        }

    whole_model: dict[str, Any] = {}
    for cutoff in args.cutoffs:
        teacher_treatment = pooled_counter(
            inventories["teacher_treatment"], layers=aggregate_layers, cutoff=cutoff
        )
        teacher_control = pooled_counter(
            inventories["teacher_control"], layers=aggregate_layers, cutoff=cutoff
        )
        base_counts = pooled_counter(
            inventories["base"], layers=aggregate_layers, cutoff=cutoff
        )
        teacher_delta = delta(teacher_treatment, teacher_control)
        teacher_base_delta = delta(teacher_treatment, base_counts)
        teacher_enriched_ids = {
            token_id for token_id, count in teacher_delta.items() if count > 0
        }
        seed_payload: dict[str, Any] = {}
        for seed in student_seeds:
            student_treatment = pooled_counter(
                inventories[f"student_treatment_{seed}"],
                layers=aggregate_layers,
                cutoff=cutoff,
            )
            student_control = pooled_counter(
                inventories[f"student_control_{seed}"],
                layers=aggregate_layers,
                cutoff=cutoff,
            )
            student_delta = delta(student_treatment, student_control)
            student_base_delta = delta(student_treatment, base_counts)
            seed_payload[seed] = {
                "teacher_student_pooled_delta_cosine": sparse_cosine(
                    teacher_delta, student_delta
                ),
                "teacher_student_pooled_base_anchored_delta_cosine": sparse_cosine(
                    teacher_base_delta, student_base_delta
                ),
                "teacher_enriched_tokens_increasing_in_student": sum(
                    student_delta[token_id] > 0 for token_id in teacher_enriched_ids
                ),
                "student_delta_mass_on_teacher_enriched_tokens": int(
                    sum(student_delta[token_id] for token_id in teacher_enriched_ids)
                ),
                "student_treatment_top_tokens": counter_payload(
                    student_treatment, token_text, limit=100
                ),
                "student_control_top_tokens": counter_payload(
                    student_control, token_text, limit=100
                ),
                "student_treatment_minus_control": signed_counter_payload(
                    student_delta, token_text, limit_per_tail=200
                ),
                "student_treatment_minus_base": signed_counter_payload(
                    student_base_delta, token_text, limit_per_tail=200
                ),
            }
        whole_model[str(cutoff)] = {
            "layers": aggregate_layers,
            "teacher_treatment_top_tokens": counter_payload(
                teacher_treatment, token_text, limit=100
            ),
            "teacher_control_top_tokens": counter_payload(
                teacher_control, token_text, limit=100
            ),
            "base_top_tokens": counter_payload(base_counts, token_text, limit=100),
            "teacher_treatment_minus_control": signed_counter_payload(
                teacher_delta, token_text, limit_per_tail=200
            ),
            "teacher_treatment_minus_base": signed_counter_payload(
                teacher_base_delta, token_text, limit_per_tail=200
            ),
            "students": seed_payload,
        }

    source_reports = {
        "transport": {"path": str(args.transport), "sha256": sha256_file(args.transport)},
        "teacher_gate": {
            "path": str(args.teacher_gate),
            "sha256": sha256_file(args.teacher_gate),
        },
        "provenance": {
            "path": str(args.provenance),
            "sha256": sha256_file(args.provenance),
        },
    }
    output = {
        "schema_version": 2,
        "analysis_status": "exploratory_dense_corresponding_layer_token_inventory",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "decoder_id": decoder.decoder_id,
        "lens_provenance_id": lens.stable_id,
        "lens_artifact_sha256": lens.expected_sha256,
        "split": args.split,
        "layers": list(args.layers),
        "transport_eligible_layers": transport_eligible_layers,
        "transport_excluded_layers": sorted(
            set(args.layers) - set(transport_eligible_layers)
        ),
        "h3_positive_layers": h3_positive_layers,
        "h3_nonpositive_layers": sorted(set(args.layers) - set(h3_positive_layers)),
        "aggregate_layers": aggregate_layers,
        "cutoffs": list(args.cutoffs),
        "token_identity_rule": "exact tokenizer token id; decoded text is display-only",
        "signed_tail_rule": (
            "positive enrichments and negative depletions are sorted and truncated "
            "independently"
        ),
        "comparison_rule": (
            "teacher and student treatment-minus-control token-count vectors are "
            "compared only at the same source layer; whole-band inventories pool only "
            "requested layers passing both transport eligibility and H3 positivity"
        ),
        "anchor_comparability": {
            "rule": (
                "all teacher, student, and base arms share prompt_id/split/anchor_id/"
                "selected-token identities; base and students additionally share the "
                "exact position manifest"
            ),
            "pairing_sha256": alignment_hashes,
            "h3_pairing_sha256": h3_pairing_hashes,
        },
        "cross_seed_corresponding_layer_token_markers": cross_seed_markers,
        "source_artifacts": source_artifacts,
        "source_reports": source_reports,
        "transport_split": transport.get("split"),
        "teacher_gate_global_passed_descriptive_only": bool(teacher_gate.get("passed")),
        "comparisons_by_seed": comparisons,
        "whole_model_inventory": whole_model,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "aggregate_layers": aggregate_layers,
                "cutoffs": list(args.cutoffs),
                "seed_mean_cosines_at_max_k": {
                    seed: comparisons[seed][str(max(args.cutoffs))][
                        "aggregate_layer_mean_paired_delta_cosine"
                    ]
                    for seed in student_seeds
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
