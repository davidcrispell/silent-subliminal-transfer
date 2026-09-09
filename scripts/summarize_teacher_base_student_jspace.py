#!/usr/bin/env python3
"""Map a conditioned-teacher direction and treatment students against clean base.

All comparisons are made at the same source layer and on matched prompt IDs.
The teacher may have a different rendered history; students and base must share
the exact clean-prompt row identities.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sst_readout.serialization import load_collected_readouts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite: {result!r}")
    return result


def row_key(row: Any) -> tuple[str, str, str]:
    """Identify the same semantic anchor across differently sized histories."""

    return (row.prompt_id, row.split, row.anchor_id)


def split_indices(table: Any, split: str) -> dict[tuple[str, str, str], int]:
    indices: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(table.rows):
        if row.split != split:
            continue
        key = row_key(row)
        if key in indices:
            raise ValueError(f"duplicate readout row key: {key}")
        indices[key] = index
    if not indices:
        raise ValueError(f"readout has no rows for split {split!r}")
    return indices


def ordered_values(
    table: Any,
    indices: dict[tuple[str, str, str], int],
    keys: list[tuple[str, str, str]],
    layer: int,
    final_target_layer: int,
) -> torch.Tensor:
    selected_indices = [indices[key] for key in keys]
    if layer == final_target_layer:
        return table.final_hidden[selected_indices].float()
    if table.jspace_by_layer is None or layer not in table.jspace_by_layer:
        raise ValueError(f"readout is missing J-space source layer {layer}")
    return table.jspace_by_layer[layer][selected_indices].float()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if float(left.norm()) == 0.0 or float(right.norm()) == 0.0:
        return 0.0
    return finite(
        float(F.cosine_similarity(left.unsqueeze(0), right.unsqueeze(0))[0]),
        "cosine",
    )


def parse_student(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--student must be SEED=PATH")
    seed, raw_path = value.split("=", 1)
    if not seed or not raw_path:
        raise argparse.ArgumentTypeError("--student must be SEED=PATH")
    return seed, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--student", action="append", type=parse_student, required=True)
    parser.add_argument("--split", default="student_evaluation")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    if len({seed for seed, _ in args.student}) != len(args.student):
        raise ValueError("student seeds must be unique")

    teacher = load_collected_readouts(args.teacher)
    base = load_collected_readouts(args.base)
    students = {
        seed: load_collected_readouts(path) for seed, path in args.student
    }
    tables = [teacher, base, *students.values()]
    source_layers = list(base.source_layers)
    if not source_layers or source_layers != list(
        range(source_layers[0], source_layers[-1] + 1)
    ):
        raise ValueError("base readout layers must be a contiguous ordered range")
    final_target_layer = source_layers[-1] + 1
    layers = [*source_layers, final_target_layer]
    for table in tables:
        table.validate()
        if list(table.source_layers) != source_layers:
            raise ValueError("all readouts must contain the exact same source layers")
        if table.lens_artifact_sha256 != base.lens_artifact_sha256:
            raise ValueError("readouts do not share one frozen J-lens artifact")
        if table.lens_provenance_id != base.lens_provenance_id:
            raise ValueError("readouts do not share one frozen J-lens identity")

    base_indices = split_indices(base, args.split)
    teacher_indices = split_indices(teacher, args.split)
    student_indices = {
        seed: split_indices(table, args.split) for seed, table in students.items()
    }
    keys = sorted(base_indices)
    if set(teacher_indices) != set(keys):
        raise ValueError("teacher and base prompt/anchor keys do not match")
    for seed, indices in student_indices.items():
        if set(indices) != set(keys):
            raise ValueError(f"student {seed} and base prompt/anchor keys do not match")
        for key in keys:
            if students[seed].rows[indices[key]] != base.rows[base_indices[key]]:
                raise ValueError(
                    f"student {seed} and base clean row identities differ at {key}"
                )

    layer_records: list[dict[str, Any]] = []
    for layer in layers:
        base_values = ordered_values(
            base, base_indices, keys, layer, final_target_layer
        )
        teacher_values = ordered_values(
            teacher, teacher_indices, keys, layer, final_target_layer
        )
        teacher_row_deltas = teacher_values - base_values
        teacher_direction = teacher_row_deltas.mean(dim=0)
        teacher_norm = finite(float(teacher_direction.norm()), "teacher direction norm")
        if teacher_norm == 0.0:
            raise ValueError(f"teacher direction is zero at layer {layer}")
        unit_teacher = teacher_direction / teacher_norm
        base_scale = finite(float(base_values.norm(dim=1).mean()), "base scale")
        midpoint = len(keys) // 2
        first_direction = teacher_row_deltas[:midpoint].mean(dim=0)
        second_direction = teacher_row_deltas[midpoint:].mean(dim=0)
        teacher_row_cosines = F.cosine_similarity(
            teacher_row_deltas,
            teacher_direction.unsqueeze(0).expand_as(teacher_row_deltas),
            dim=1,
        )
        teacher_record: dict[str, Any] = {
            "direction_norm": teacher_norm,
            "direction_over_mean_base_norm": teacher_norm / base_scale,
            "half_split_cosine": cosine(first_direction, second_direction),
            "mean_row_delta_norm": finite(
                float(teacher_row_deltas.norm(dim=1).mean()),
                "mean teacher row delta norm",
            ),
            "mean_row_cosine_to_direction": finite(
                float(teacher_row_cosines.mean()),
                "mean teacher row cosine",
            ),
        }

        student_records: dict[str, Any] = {}
        for seed, student in students.items():
            student_values = ordered_values(
                student, student_indices[seed], keys, layer, final_target_layer
            )
            row_deltas = student_values - base_values
            mean_delta = row_deltas.mean(dim=0)
            projection = finite(float(torch.dot(mean_delta, unit_teacher)), "projection")
            delta_norm = finite(float(mean_delta.norm()), "student delta norm")
            student_records[seed] = {
                "teacherward_projection": projection,
                "fraction_of_teacher_direction": projection / teacher_norm,
                "cosine_to_teacher_direction": cosine(mean_delta, teacher_direction),
                "delta_norm": delta_norm,
                "delta_over_mean_base_norm": delta_norm / base_scale,
                "per_row_teacherward_projection": [
                    finite(value, "per-row projection")
                    for value in (row_deltas @ unit_teacher).tolist()
                ],
            }

        projections = [
            record["teacherward_projection"] for record in student_records.values()
        ]
        fractions = [
            record["fraction_of_teacher_direction"] for record in student_records.values()
        ]
        cosines = [
            record["cosine_to_teacher_direction"] for record in student_records.values()
        ]
        layer_records.append(
            {
                "layer": layer,
                "coordinate": (
                    "final_hidden_target"
                    if layer == final_target_layer
                    else "transported_jspace"
                ),
                "teacher_minus_base": teacher_record,
                "students_minus_base": student_records,
                "student_seed_summary": {
                    "positive_projection_seeds": sum(value > 0 for value in projections),
                    "n_seeds": len(projections),
                    "mean_teacherward_projection": statistics.fmean(projections),
                    "median_teacherward_projection": statistics.median(projections),
                    "mean_fraction_of_teacher_direction": statistics.fmean(fractions),
                    "mean_cosine_to_teacher_direction": statistics.fmean(cosines),
                },
            }
        )

    def top_layers(path: tuple[str, ...], *, count: int = 10) -> list[dict[str, Any]]:
        def extract(record: dict[str, Any]) -> float:
            value: Any = record
            for key in path:
                value = value[key]
            return float(value)

        ordered = sorted(layer_records, key=extract, reverse=True)[:count]
        return [{"layer": row["layer"], "value": extract(row)} for row in ordered]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer",
                "coordinate",
                "seed",
                "teacher_direction_norm",
                "teacher_direction_over_mean_base_norm",
                "teacher_half_split_cosine",
                "teacher_mean_row_cosine_to_direction",
                "student_teacherward_projection",
                "student_fraction_of_teacher_direction",
                "student_cosine_to_teacher_direction",
                "student_delta_norm",
                "student_delta_over_mean_base_norm",
            ]
        )
        for layer_record in layer_records:
            teacher_record = layer_record["teacher_minus_base"]
            for seed, student_record in layer_record["students_minus_base"].items():
                writer.writerow(
                    [
                        layer_record["layer"],
                        layer_record["coordinate"],
                        seed,
                        teacher_record["direction_norm"],
                        teacher_record["direction_over_mean_base_norm"],
                        teacher_record["half_split_cosine"],
                        teacher_record["mean_row_cosine_to_direction"],
                        student_record["teacherward_projection"],
                        student_record["fraction_of_teacher_direction"],
                        student_record["cosine_to_teacher_direction"],
                        student_record["delta_norm"],
                        student_record["delta_over_mean_base_norm"],
                    ]
                )

    report = {
        "schema_version": 1,
        "estimand": {
            "teacher": "abuse-conditioned teacher minus clean frozen base",
            "student": "treatment student minus clean frozen base",
            "comparison": (
                "same prompt ID, semantic anchor, and layer; absolute token indices "
                "may differ because the teacher has conditioning history"
            ),
            "scope": "exploratory all-layer map; no layer masking or averaging gate",
        },
        "identity": {
            "teacher_path": str(args.teacher),
            "teacher_sha256": sha256_file(args.teacher),
            "base_path": str(args.base),
            "base_sha256": sha256_file(args.base),
            "students": {
                seed: {"path": str(path), "sha256": sha256_file(path)}
                for seed, path in args.student
            },
            "lens_artifact_sha256": base.lens_artifact_sha256,
            "lens_provenance_id": base.lens_provenance_id,
            "protocol_path": str(args.protocol),
            "protocol_sha256": sha256_file(args.protocol),
            "split": args.split,
            "prompt_position_rows": len(keys),
            "source_layers": source_layers,
            "final_target_layer": final_target_layer,
            "analyzed_layers": layers,
        },
        "rankings": {
            "teacher_relative_difference": top_layers(
                ("teacher_minus_base", "direction_over_mean_base_norm")
            ),
            "teacher_half_split_reproducibility": top_layers(
                ("teacher_minus_base", "half_split_cosine")
            ),
            "student_mean_teacherward_projection": top_layers(
                ("student_seed_summary", "mean_teacherward_projection")
            ),
            "student_mean_fraction_of_teacher_direction": top_layers(
                ("student_seed_summary", "mean_fraction_of_teacher_direction")
            ),
            "student_mean_cosine_to_teacher_direction": top_layers(
                ("student_seed_summary", "mean_cosine_to_teacher_direction")
            ),
        },
        "layers": layer_records,
        "csv": {"path": str(args.output_csv), "sha256": sha256_file(args.output_csv)},
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_json_sha256": sha256_file(args.output_json),
                "output_csv": str(args.output_csv),
                "output_csv_sha256": sha256_file(args.output_csv),
                "layers": layers,
                "seeds": sorted(students),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
