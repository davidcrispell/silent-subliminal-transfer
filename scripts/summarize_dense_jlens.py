#!/usr/bin/env python3
"""Summarize dense J-Lens results with strictly corresponding-layer comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value


def layer_keys(value: Any, name: str) -> set[int]:
    raw = mapping(value, name)
    try:
        return {int(key) for key in raw}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains a non-integer layer key") from exc


def require_layer_keys(value: Any, expected: set[int], name: str) -> None:
    observed = layer_keys(value, name)
    if observed != expected:
        raise ValueError(
            f"{name} layer mismatch: expected {sorted(expected)}, got {sorted(observed)}"
        )


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {result!r}")
    return result


def require_equal(observed: Any, expected: Any, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, got {observed!r}")


def run_namespace(model_id: Any, name: str) -> str:
    if not isinstance(model_id, str) or ":" not in model_id:
        raise ValueError(f"{name} must be a namespaced model id")
    return model_id.split(":", 1)[0]


def validate_inputs(
    transport: dict[str, Any],
    projection: dict[str, Any],
    teacher_gate: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[list[int], list[str]]:
    requested_layers = [int(layer) for layer in provenance["requested_jlens_source_layers"]]
    if not requested_layers or len(requested_layers) != len(set(requested_layers)):
        raise ValueError("provenance requested layers must be nonempty and unique")
    requested = set(requested_layers)

    direction = mapping(projection["teacher_direction"], "projection.teacher_direction")
    require_equal(
        [int(layer) for layer in direction["layers"]],
        requested_layers,
        "projection teacher-direction layer order",
    )
    require_layer_keys(transport["layers"], requested, "transport.layers")
    require_layer_keys(teacher_gate["layers"], requested, "teacher_gate.layers")
    require_layer_keys(
        teacher_gate["calibration_direction"]["norms"],
        requested,
        "teacher_gate.calibration_direction.norms",
    )
    require_layer_keys(
        teacher_gate["validation_direction"]["norms"],
        requested,
        "teacher_gate.validation_direction.norms",
    )
    require_equal(
        [int(layer) for layer in teacher_gate["calibration_direction"]["layers"]],
        requested_layers,
        "teacher-gate calibration layer order",
    )
    require_equal(
        [int(layer) for layer in teacher_gate["validation_direction"]["layers"]],
        requested_layers,
        "teacher-gate validation layer order",
    )

    artifact = mapping(projection["artifact_manifest"], "projection.artifact_manifest")
    lens_sha = provenance["lens_artifact_sha256"]
    require_equal(transport["lens_artifact_sha256"], lens_sha, "transport lens SHA")
    require_equal(artifact["lens_artifact_sha256"], lens_sha, "projection lens SHA")
    require_equal(direction["lens_artifact_sha256"], lens_sha, "direction lens SHA")
    for branch in ("calibration_direction", "validation_direction"):
        require_equal(
            teacher_gate[branch]["lens_artifact_sha256"],
            lens_sha,
            f"teacher-gate {branch} lens SHA",
        )

    lens_id = transport["lens_provenance_id"]
    require_equal(artifact["lens_provenance_id"], lens_id, "projection lens id")
    require_equal(direction["lens_provenance_id"], lens_id, "direction lens id")
    for branch in ("calibration_direction", "validation_direction"):
        require_equal(
            teacher_gate[branch]["lens_provenance_id"],
            lens_id,
            f"teacher-gate {branch} lens id",
        )
    require_equal(
        artifact["provenance"],
        provenance["lens_provenance"],
        "projection frozen lens provenance",
    )

    require_equal(
        artifact["teacher_direction_artifact_sha256"],
        teacher_gate["teacher_direction_sha256"],
        "teacher-direction artifact SHA",
    )
    require_equal(
        direction["teacher_model_id"],
        teacher_gate["calibration_direction"]["teacher_model_id"],
        "treatment teacher model id",
    )
    require_equal(
        direction["control_model_id"],
        teacher_gate["calibration_direction"]["control_model_id"],
        "control teacher model id",
    )
    require_equal(
        teacher_gate["calibration_direction"]["teacher_model_id"],
        teacher_gate["validation_direction"]["teacher_model_id"],
        "teacher-gate treatment model id",
    )
    require_equal(
        teacher_gate["calibration_direction"]["control_model_id"],
        teacher_gate["validation_direction"]["control_model_id"],
        "teacher-gate control model id",
    )

    embedded_gate = mapping(projection["teacher_state_gate"], "projection.teacher_state_gate")
    require_layer_keys(
        embedded_gate["layers"], requested, "projection.teacher_state_gate.layers"
    )
    for layer in requested_layers:
        key = str(layer)
        require_equal(
            bool(embedded_gate["layers"][key]["positive"]),
            bool(teacher_gate["layers"][key]["positive"]),
            f"embedded/external H3 positivity at layer {layer}",
        )
        embedded_cosine = finite_float(
            embedded_gate["layers"][key]["calibration_validation_cosine"],
            f"embedded H3 cosine layer {layer}",
        )
        external_cosine = finite_float(
            teacher_gate["layers"][key]["calibration_validation_cosine"],
            f"external H3 cosine layer {layer}",
        )
        if not math.isclose(embedded_cosine, external_cosine, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"embedded/external H3 cosine mismatch at layer {layer}")

    student_records = projection["student_projections"]
    if not isinstance(student_records, list) or not student_records:
        raise ValueError("projection must contain at least one student pair")
    students_by_seed: dict[str, dict[str, Any]] = {}
    namespaces = {
        run_namespace(direction["teacher_model_id"], "treatment teacher model id"),
        run_namespace(direction["control_model_id"], "control teacher model id"),
    }
    for item in student_records:
        record = mapping(item, "student projection")
        seed = str(int(record["seed"]))
        if seed in students_by_seed:
            raise ValueError(f"duplicate student seed {seed}")
        require_equal(record["coordinate"], "jspace", f"student {seed} coordinate")
        require_layer_keys(record["layers"], requested, f"student {seed} layers")
        treatment_id = record["student_model_id"]
        control_id = record["control_model_id"]
        if not str(treatment_id).endswith(f"treatment_seed_{seed}"):
            raise ValueError(f"student {seed} treatment model id has the wrong variant")
        if not str(control_id).endswith(f"control_seed_{seed}"):
            raise ValueError(f"student {seed} control model id has the wrong variant")
        namespaces.add(run_namespace(treatment_id, f"student {seed} treatment model id"))
        namespaces.add(run_namespace(control_id, f"student {seed} control model id"))
        students_by_seed[seed] = record
    if len(namespaces) != 1:
        raise ValueError(f"model ids span multiple run namespaces: {sorted(namespaces)}")

    seeds = sorted(students_by_seed, key=int)
    semantic = projection["logit_lens_comparison"]["preregistered_token_contrast"]
    teacher_semantic = semantic["teacher_treatment_minus_control"]["jlens"]
    student_semantic = mapping(
        semantic["student_treatment_minus_control_by_seed"],
        "student semantic records",
    )
    require_equal(sorted(student_semantic, key=int), seeds, "semantic student seeds")
    require_layer_keys(teacher_semantic, requested, "teacher semantic layers")
    for seed in seeds:
        require_layer_keys(
            student_semantic[seed]["jlens"], requested, f"student {seed} semantic layers"
        )

    expected_variants = {"base"} | {
        f"{condition}_seed_{seed}" for seed in seeds for condition in ("treatment", "control")
    }
    checkpoints = mapping(transport["checkpoints"], "transport.checkpoints")
    require_equal(set(checkpoints), expected_variants, "transport checkpoint variants")
    for variant, record in checkpoints.items():
        checkpoint = mapping(record, f"transport checkpoint {variant}")
        require_layer_keys(checkpoint["jlens"], requested, f"{variant} J-Lens transport")
        require_layer_keys(
            checkpoint["logit_lens"], requested, f"{variant} logit-lens transport"
        )
        if variant != "base" and not str(checkpoint["model_id"]).endswith(variant):
            raise ValueError(f"transport checkpoint {variant} has the wrong model id")
    expected_model = provenance["lens_provenance"]
    decoder_prefix = f"{expected_model['model_repo']}@{expected_model['model_revision']}+"
    if not str(transport["decoder_id"]).startswith(decoder_prefix):
        raise ValueError("transport decoder does not match frozen model provenance")

    if "config_sha256" in teacher_gate:
        require_equal(
            teacher_gate["config_sha256"],
            provenance["config_semantic_sha256"],
            "teacher-gate config SHA",
        )
    if "readout_protocol_sha256" in teacher_gate:
        require_equal(
            teacher_gate["readout_protocol_sha256"],
            provenance["source_readout_protocol_sha256"],
            "teacher-gate protocol SHA",
        )

    code_hashes = mapping(provenance["analysis_code_sha256"], "analysis_code_sha256")
    own_hashes = [
        digest for path, digest in code_hashes.items() if Path(path).name == Path(__file__).name
    ]
    if len(own_hashes) != 1:
        raise ValueError("provenance must bind exactly one dense-summary implementation")
    require_equal(own_hashes[0], sha256_file(Path(__file__)), "dense-summary code SHA")
    return requested_layers, seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--teacher-gate", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transport = mapping(json.loads(args.transport.read_text(encoding="utf-8")), "transport")
    projection = mapping(json.loads(args.projection.read_text(encoding="utf-8")), "projection")
    teacher_gate = mapping(
        json.loads(args.teacher_gate.read_text(encoding="utf-8")), "teacher gate"
    )
    provenance = mapping(json.loads(args.provenance.read_text(encoding="utf-8")), "provenance")
    requested_layers, seeds = validate_inputs(transport, projection, teacher_gate, provenance)

    transport_eligible_layers = [
        layer for layer in requested_layers if bool(transport["layers"][str(layer)]["eligible"])
    ]
    h3_positive_layers = [
        layer
        for layer in requested_layers
        if bool(teacher_gate["layers"][str(layer)]["positive"])
    ]
    aggregation_layers = [
        layer
        for layer in requested_layers
        if layer in set(transport_eligible_layers) and layer in set(h3_positive_layers)
    ]
    if not aggregation_layers:
        raise ValueError("no requested layer passed both transport and H3 diagnostics")

    semantic = projection["logit_lens_comparison"]["preregistered_token_contrast"]
    teacher_semantic = semantic["teacher_treatment_minus_control"]["jlens"]
    student_semantic = semantic["student_treatment_minus_control_by_seed"]
    students_by_seed = {
        str(int(item["seed"])): item for item in projection["student_projections"]
    }

    seed_summaries: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        record = students_by_seed[seed]
        layer_records = record["layers"]
        semantic_values = [
            finite_float(
                student_semantic[seed]["jlens"][str(layer)],
                f"seed {seed} semantic layer {layer}",
            )
            for layer in aggregation_layers
        ]
        projection_values = [
            finite_float(
                layer_records[str(layer)]["teacherward_projection"],
                f"seed {seed} projection layer {layer}",
            )
            for layer in aggregation_layers
        ]
        fraction_values = [
            finite_float(
                layer_records[str(layer)]["fraction_of_teacher_delta"],
                f"seed {seed} fraction layer {layer}",
            )
            for layer in aggregation_layers
        ]
        cosine_values = [
            finite_float(
                layer_records[str(layer)]["cosine_to_teacher"],
                f"seed {seed} cosine layer {layer}",
            )
            for layer in aggregation_layers
        ]
        seed_summaries[seed] = {
            "n_aggregation_layers": len(aggregation_layers),
            "semantic_mean": mean(semantic_values),
            "semantic_positive": mean(semantic_values) > 0,
            "teacherward_projection_mean": mean(projection_values),
            "teacherward_projection_positive": mean(projection_values) > 0,
            "fraction_of_teacher_delta_mean": mean(fraction_values),
            "cosine_to_teacher_mean": mean(cosine_values),
        }

    by_layer: dict[str, dict[str, Any]] = {}
    for layer in requested_layers:
        key = str(layer)
        transport_ok = bool(transport["layers"][key]["eligible"])
        h3_ok = bool(teacher_gate["layers"][key]["positive"])
        included = transport_ok and h3_ok
        exclusion_reasons = []
        if not transport_ok:
            exclusion_reasons.append("transport_ineligible")
        if not h3_ok:
            exclusion_reasons.append("teacher_direction_not_holdout_positive")
        student_layer_metrics = {
            seed: {
                "semantic_delta": finite_float(
                    student_semantic[seed]["jlens"][key],
                    f"student semantic seed {seed} layer {layer}",
                ),
                "teacherward_projection": finite_float(
                    students_by_seed[seed]["layers"][key]["teacherward_projection"],
                    f"student projection seed {seed} layer {layer}",
                ),
                "fraction_of_teacher_delta": finite_float(
                    students_by_seed[seed]["layers"][key]["fraction_of_teacher_delta"],
                    f"student fraction seed {seed} layer {layer}",
                ),
                "cosine_to_teacher": finite_float(
                    students_by_seed[seed]["layers"][key]["cosine_to_teacher"],
                    f"student cosine seed {seed} layer {layer}",
                ),
            }
            for seed in seeds
        }
        by_layer[key] = {
            "teacher_layer": layer,
            "student_layer": layer,
            "corresponding_layer_only": True,
            "included_in_aggregate": included,
            "exclusion_reasons": exclusion_reasons,
            "transport_eligible": transport_ok,
            "base_mean_kl": finite_float(
                transport["layers"][key]["base_mean_kl"], f"base KL layer {layer}"
            ),
            "allowed_mean_kl": finite_float(
                transport["layers"][key]["allowed_mean_kl"], f"allowed KL layer {layer}"
            ),
            "h3_positive": h3_ok,
            "h3_calibration_validation_cosine": finite_float(
                teacher_gate["layers"][key]["calibration_validation_cosine"],
                f"H3 cosine layer {layer}",
            ),
            "teacher_semantic_delta": finite_float(
                teacher_semantic[key], f"teacher semantic layer {layer}"
            ),
            "students_by_seed": student_layer_metrics,
            "student_semantic_delta_mean": mean(
                [item["semantic_delta"] for item in student_layer_metrics.values()]
            ),
            "student_teacherward_projection_mean": mean(
                [item["teacherward_projection"] for item in student_layer_metrics.values()]
            ),
            "student_cosine_to_teacher_mean": mean(
                [item["cosine_to_teacher"] for item in student_layer_metrics.values()]
            ),
        }

    output = {
        "schema_version": 1,
        "analysis_status": "posthoc_exploratory_dense_late_layer_sweep",
        "comparison_rule": (
            "teacher treatment-minus-control J-space at layer l is compared only "
            "with student treatment-minus-control J-space at the same layer l"
        ),
        "aggregation_rule": (
            "unweighted mean of corresponding-layer metrics over requested layers "
            "that pass both per-layer transport and H3 holdout-reproducibility diagnostics"
        ),
        "requested_layers": requested_layers,
        "transport_eligible_layers": transport_eligible_layers,
        "transport_excluded_layers": [
            layer for layer in requested_layers if layer not in set(transport_eligible_layers)
        ],
        "h3_positive_layers": h3_positive_layers,
        "h3_nonpositive_layers": [
            layer for layer in requested_layers if layer not in set(h3_positive_layers)
        ],
        "aggregation_layers": aggregation_layers,
        "aggregate_excluded_layers": [
            layer for layer in requested_layers if layer not in set(aggregation_layers)
        ],
        "teacher_semantic_mean": mean(
            [
                finite_float(teacher_semantic[str(layer)], f"teacher semantic layer {layer}")
                for layer in aggregation_layers
            ]
        ),
        "students": seed_summaries,
        "positive_semantic_seeds": sum(
            bool(item["semantic_positive"]) for item in seed_summaries.values()
        ),
        "positive_teacherward_projection_seeds": sum(
            bool(item["teacherward_projection_positive"]) for item in seed_summaries.values()
        ),
        "by_layer": by_layer,
        "identity": {
            "run_namespace": run_namespace(
                projection["teacher_direction"]["teacher_model_id"], "teacher model id"
            ),
            "student_seeds": [int(seed) for seed in seeds],
            "decoder_id": transport["decoder_id"],
            "lens_artifact_sha256": transport["lens_artifact_sha256"],
            "lens_provenance_id": transport["lens_provenance_id"],
            "teacher_direction_sha256": teacher_gate["teacher_direction_sha256"],
        },
        "source_reports": {
            "transport": {
                "path": str(args.transport),
                "sha256": sha256_file(args.transport),
            },
            "projection": {
                "path": str(args.projection),
                "sha256": sha256_file(args.projection),
            },
            "teacher_gate": {
                "path": str(args.teacher_gate),
                "sha256": sha256_file(args.teacher_gate),
            },
            "provenance": {
                "path": str(args.provenance),
                "sha256": sha256_file(args.provenance),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
