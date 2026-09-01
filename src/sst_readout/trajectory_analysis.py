"""Corresponding-layer analysis for exhaustive J-Lens trajectories.

The raw trajectory artifact remains the source of truth for every retained
position/layer cell.  This module derives three views without collapsing that
inventory:

* position-level teacher/student difference comparisons;
* per-seed, per-layer summaries for the pre-answer boundary and completion
  trajectory separately; and
* top-k token occurrence inventories, including a declared wolf-token family.

Completion rows produced by free generation are deliberately labelled
descriptive.  A completion comparison is called token-aligned only when every
arm declares teacher-forced decoding and uses the same target token at that row.
No source layer is ever compared with a different source layer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from .trajectory import JlensTrajectory, TrajectoryRow

SCHEMA_VERSION = 1
GEMMA2_9B_RETAINED_SOURCE_LAYERS = tuple(range(14, 41))
DEFAULT_WOLF_TOKEN_STEMS = ("wolf", "wolv", "lupin", "lupus")

Scope = Literal["pre_answer", "completion_trajectory"]


def _finite_or_none(value: torch.Tensor | float) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _vector_metrics(
    student_delta: torch.Tensor,
    teacher_delta: torch.Tensor,
) -> tuple[float, float, float | None, float, float]:
    """Return projection, fraction, cosine, student norm, and teacher norm."""

    student = student_delta.detach().float().cpu()
    teacher = teacher_delta.detach().float().cpu()
    teacher_norm = float(torch.linalg.vector_norm(teacher))
    student_norm = float(torch.linalg.vector_norm(student))
    dot = float(student @ teacher)
    projection = 0.0 if teacher_norm == 0.0 else dot / teacher_norm
    fraction = 0.0 if teacher_norm == 0.0 else dot / (teacher_norm * teacher_norm)
    cosine = (
        None
        if teacher_norm == 0.0 or student_norm == 0.0
        else _finite_or_none(F.cosine_similarity(student[None], teacher[None]))
    )
    return projection, fraction, cosine, student_norm, teacher_norm


def _scope(row: TrajectoryRow) -> Scope:
    return "pre_answer" if row.generated_token_index == 0 else "completion_trajectory"


def _row_key(row: TrajectoryRow) -> tuple[str, int]:
    return row.prompt_id, row.generated_token_index


def _rows_by_key(
    trajectory: JlensTrajectory,
) -> dict[tuple[str, int], tuple[int, TrajectoryRow]]:
    result: dict[tuple[str, int], tuple[int, TrajectoryRow]] = {}
    for index, row in enumerate(trajectory.rows):
        key = _row_key(row)
        if key in result:
            raise ValueError(f"duplicate trajectory row key {key!r} in {trajectory.run_id}")
        result[key] = (index, row)
    return result


def _strategy(trajectory: JlensTrajectory) -> str:
    return str(trajectory.decoding.get("strategy", "unknown")).strip().casefold()


def _is_teacher_forced(strategy: str) -> bool:
    return strategy in {
        "teacher_forced",
        "teacher-forced",
        "shared_teacher_forced",
        "shared-teacher-forced",
    }


def _alignment_mode(
    rows: Sequence[TrajectoryRow],
    trajectories: Sequence[JlensTrajectory],
) -> tuple[str, bool, bool, bool]:
    generated_index = rows[0].generated_token_index
    if any(row.generated_token_index != generated_index for row in rows):
        raise ValueError("alignment attempted across different generated-token indices")
    exact_target = len({row.sampled_token_id for row in rows}) == 1
    exact_prefix = len({row.prefix_token_ids_sha256 for row in rows}) == 1
    if generated_index == 0:
        return "pre_answer_boundary", exact_target, exact_prefix, False
    teacher_forced = all(_is_teacher_forced(_strategy(item)) for item in trajectories)
    if teacher_forced and exact_target:
        return "shared_teacher_forced_token_aligned", True, exact_prefix, False
    return "free_generation_index_aligned_descriptive", exact_target, exact_prefix, True


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _lens_identity(trajectory: JlensTrajectory) -> str:
    """Compare frozen-lens identity while tolerating irrelevant dictionary ordering."""

    return _canonical_json_sha256(dict(trajectory.lens_identity))


def _validate_trajectories(
    artifacts: Mapping[str, JlensTrajectory],
    *,
    required_layers: Sequence[int],
) -> tuple[int, ...]:
    if not artifacts:
        raise ValueError("at least one trajectory artifact is required")
    layers = tuple(int(layer) for layer in required_layers)
    if not layers or len(set(layers)) != len(layers) or tuple(sorted(layers)) != layers:
        raise ValueError("required_layers must be sorted, unique, and nonempty")
    first = next(iter(artifacts.values()))
    first.validate()
    expected_lens = _lens_identity(first)
    expected_decoder = _canonical_json_sha256(dict(first.decoder_identity))
    expected_base_model = (
        first.model_identity.get("base_model_id"),
        first.model_identity.get("base_model_revision"),
    )
    if any(not isinstance(value, str) or not value for value in expected_base_model):
        raise ValueError("trajectory analysis requires a pinned base model id and revision")
    expected_width = first.d_model
    expected_final_block = first.final_block_index
    expected_top_k = first.top_k
    if expected_top_k <= 0:
        raise ValueError(
            "trajectory analysis requires retained top-k readouts for token inventories"
        )
    for label, trajectory in artifacts.items():
        trajectory.validate()
        if trajectory.source_layers != layers:
            raise ValueError(
                f"{label} source-layer inventory {trajectory.source_layers} does not equal "
                f"the required corresponding-layer inventory {layers}"
            )
        if trajectory.d_model != expected_width:
            raise ValueError(f"{label} has a different J-space width")
        if trajectory.final_block_index != expected_final_block:
            raise ValueError(f"{label} has a different final-block reference index")
        if _lens_identity(trajectory) != expected_lens:
            raise ValueError(f"{label} does not use the same frozen J-Lens artifact")
        if _canonical_json_sha256(dict(trajectory.decoder_identity)) != expected_decoder:
            raise ValueError(f"{label} does not use the same fixed readout decoder")
        base_model = (
            trajectory.model_identity.get("base_model_id"),
            trajectory.model_identity.get("base_model_revision"),
        )
        if base_model != expected_base_model:
            raise ValueError(f"{label} does not use the same pinned base model")
        if trajectory.top_k <= 0:
            raise ValueError(f"{label} requires retained top-k readouts for token inventories")
        if trajectory.top_k != expected_top_k:
            raise ValueError(
                f"{label} top-k width {trajectory.top_k} does not equal {expected_top_k}"
            )
    return layers


@dataclass(frozen=True)
class PositionComparison:
    seed: int
    scope: Scope
    alignment_mode: str
    descriptive_only: bool
    prompt_id: str
    generated_token_index: int
    layer: int
    exact_target_token_alignment: bool
    exact_prefix_alignment: bool
    teacher_treatment_token_id: int
    teacher_control_token_id: int
    student_treatment_token_id: int
    student_control_token_id: int
    teacher_delta_norm: float
    student_delta_norm: float
    teacherward_projection: float
    fraction_of_teacher_delta: float
    cosine_to_teacher: float | None


@dataclass(frozen=True)
class LayerComparison:
    seed: int
    scope: Scope
    layer: int
    n_positions: int
    alignment_modes: tuple[str, ...]
    descriptive_only: bool
    teacher_delta_norm: float
    student_delta_norm: float
    teacherward_projection: float
    fraction_of_teacher_delta: float
    cosine_to_teacher: float | None
    mean_position_projection: float
    median_position_projection: float
    mean_position_cosine: float | None


@dataclass(frozen=True)
class InventoryAggregate:
    artifact: str
    scope: str
    layer: int | None
    token_id: int
    token_text: str
    is_wolf_family: bool
    cell_count: int
    occurrence_count: int
    occurrence_rate: float
    best_rank: int
    mean_score: float


@dataclass(frozen=True)
class InventoryContrast:
    comparison: str
    seed: int | None
    scope: str
    layer: int | None
    token_id: int
    token_text: str
    is_wolf_family: bool
    treatment_cell_count: int
    control_cell_count: int
    treatment_count: int
    control_count: int
    treatment_minus_control: int
    treatment_rate: float | None
    control_rate: float | None
    treatment_minus_control_rate: float | None


@dataclass(frozen=True)
class TokenConcordance:
    seed: int
    scope: str
    layer: int | None
    count_delta_cosine_to_teacher: float | None
    rate_delta_cosine_to_teacher: float | None
    teacher_positive_token_count: int
    student_positive_token_count: int
    positive_token_overlap_count: int
    teacher_wolf_occurrence_delta: int
    student_wolf_occurrence_delta: int
    teacher_wolf_rate_delta: float | None
    student_wolf_rate_delta: float | None


@dataclass(frozen=True)
class AlignmentCoverage:
    seed: int
    teacher_treatment_rows: int
    teacher_control_rows: int
    student_treatment_rows: int
    student_control_rows: int
    union_rows: int
    common_rows: int
    common_pre_answer_rows: int
    common_completion_rows: int
    incomplete_row_keys: int


@dataclass(frozen=True)
class TrajectoryAnalysis:
    source_layers: tuple[int, ...]
    artifact_identities: Mapping[str, Mapping[str, Any]]
    position_comparisons: tuple[PositionComparison, ...]
    layer_comparisons: tuple[LayerComparison, ...]
    inventory_aggregates: tuple[InventoryAggregate, ...]
    inventory_contrasts: tuple[InventoryContrast, ...]
    token_concordance: tuple[TokenConcordance, ...]
    alignment_coverage: tuple[AlignmentCoverage, ...]
    wolf_token_stems: tuple[str, ...]
    schema_version: int = SCHEMA_VERSION

    def summary_dict(self, *, top_teacherward_tokens: int = 20) -> dict[str, Any]:
        if top_teacherward_tokens <= 0:
            raise ValueError("top_teacherward_tokens must be positive")
        by_seed: dict[str, dict[str, Any]] = {}
        seeds = sorted({row.seed for row in self.layer_comparisons})
        for seed in seeds:
            scope_payload: dict[str, Any] = {}
            for scope in ("pre_answer", "completion_trajectory"):
                selected = [
                    row
                    for row in self.layer_comparisons
                    if row.seed == seed and row.scope == scope
                ]
                if not selected:
                    continue
                cosines = [
                    row.cosine_to_teacher
                    for row in selected
                    if row.cosine_to_teacher is not None
                ]
                scope_payload[scope] = {
                    "n_layers": len(selected),
                    "n_positions": selected[0].n_positions,
                    "descriptive_only": any(row.descriptive_only for row in selected),
                    "mean_teacherward_projection": statistics.fmean(
                        row.teacherward_projection for row in selected
                    ),
                    "mean_fraction_of_teacher_delta": statistics.fmean(
                        row.fraction_of_teacher_delta for row in selected
                    ),
                    "mean_cosine_to_teacher": (statistics.fmean(cosines) if cosines else None),
                    "positive_projection_layers": sum(
                        row.teacherward_projection > 0 for row in selected
                    ),
                }
            by_seed[str(seed)] = scope_payload

        top_tokens: dict[str, list[dict[str, Any]]] = {}
        groups: dict[tuple[str, int | None, str], list[InventoryContrast]] = defaultdict(list)
        for row in self.inventory_contrasts:
            groups[(row.scope, row.layer, row.comparison)].append(row)
        for (scope, layer, comparison), values in groups.items():
            ordered = sorted(
                values,
                key=lambda item: (
                    item.treatment_minus_control_rate is None,
                    -(
                        item.treatment_minus_control_rate
                        if item.treatment_minus_control_rate is not None
                        else 0.0
                    ),
                    -item.treatment_minus_control,
                    item.token_id,
                ),
            )[:top_teacherward_tokens]
            key = f"{comparison}|{scope}|{'whole_model' if layer is None else layer}"
            top_tokens[key] = [asdict(item) for item in ordered]

        wolf_summaries: list[dict[str, Any]] = []
        for (scope, layer, comparison), values in groups.items():
            selected = [row for row in values if row.is_wolf_family]
            wolf_summaries.append(
                {
                    "comparison": comparison,
                    "scope": scope,
                    "layer": layer,
                    "treatment_occurrences": sum(row.treatment_count for row in selected),
                    "control_occurrences": sum(row.control_count for row in selected),
                    "treatment_minus_control": sum(
                        row.treatment_minus_control for row in selected
                    ),
                    "treatment_minus_control_rate": (
                        sum(
                            row.treatment_minus_control_rate
                            for row in selected
                            if row.treatment_minus_control_rate is not None
                        )
                        if selected
                        and all(
                            row.treatment_minus_control_rate is not None for row in selected
                        )
                        else None
                    ),
                    "observed_variants": sorted(
                        {
                            row.token_text
                            for row in selected
                            if row.treatment_count or row.control_count
                        }
                    ),
                }
            )

        return {
            "schema_version": self.schema_version,
            "analysis_contract": {
                "source_layers": list(self.source_layers),
                "layer_matching": "strictly corresponding source layer only",
                "teacher_direction": "teacher_treatment_minus_teacher_control",
                "student_difference": "student_treatment_minus_student_control per seed",
                "pre_answer_scope": "generated_token_index == 0",
                "completion_scope": "generated_token_index >= 1",
                "free_generation_status": "descriptive_only",
                "wolf_token_stems": list(self.wolf_token_stems),
                "inventory_count_semantics": (
                    "top-k rank occurrences and per-cell prevalence; whole_model sums "
                    "retained layer-position cells"
                ),
            },
            "artifact_identities": dict(self.artifact_identities),
            "counts": {
                "position_comparisons": len(self.position_comparisons),
                "layer_comparisons": len(self.layer_comparisons),
                "inventory_aggregates": len(self.inventory_aggregates),
                "inventory_contrasts": len(self.inventory_contrasts),
            },
            "alignment_coverage": [asdict(row) for row in self.alignment_coverage],
            "per_seed": by_seed,
            "layer_comparisons": [asdict(row) for row in self.layer_comparisons],
            "token_concordance": [asdict(row) for row in self.token_concordance],
            "wolf_family": sorted(
                wolf_summaries,
                key=lambda item: (
                    item["comparison"],
                    item["scope"],
                    -1 if item["layer"] is None else item["layer"],
                ),
            ),
            "top_treatment_minus_control_tokens": top_tokens,
        }


def normalize_token_text(text: str) -> str:
    """Normalize common tokenizer boundary marks while retaining token identity."""

    return text.replace("▁", " ").replace("Ġ", " ").strip().casefold()


def is_wolf_family_token(
    text: str,
    *,
    stems: Sequence[str] = DEFAULT_WOLF_TOKEN_STEMS,
) -> bool:
    normalized = normalize_token_text(text)
    return any(str(stem).casefold() in normalized for stem in stems)


def _artifact_identity(trajectory: JlensTrajectory) -> dict[str, Any]:
    return {
        "run_id": trajectory.run_id,
        "model_identity": dict(trajectory.model_identity),
        "lens_identity_sha256": _lens_identity(trajectory),
        "position_manifest_sha256": trajectory.position_manifest_sha256,
        "source_layers": list(trajectory.source_layers),
        "n_rows": trajectory.n_rows,
        "top_k": trajectory.top_k,
        "decoding_strategy": _strategy(trajectory),
    }


def _mean_vector(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ValueError("cannot average an empty vector collection")
    return torch.stack([value.detach().float().cpu() for value in values], dim=0).mean(dim=0)


def _inventory(
    artifacts: Mapping[str, JlensTrajectory],
    *,
    token_text: Callable[[int], str],
    wolf_token_stems: tuple[str, ...],
) -> tuple[InventoryAggregate, ...]:
    counts: Counter[tuple[str, str, int | None, int]] = Counter()
    cell_counts: Counter[tuple[str, str, int | None]] = Counter()
    best_ranks: dict[tuple[str, str, int | None, int], int] = {}
    score_sums: defaultdict[tuple[str, str, int | None, int], float] = defaultdict(float)
    text_cache: dict[int, str] = {}

    def decoded(token_id: int) -> str:
        if token_id not in text_cache:
            text_cache[token_id] = str(token_text(token_id))
        return text_cache[token_id]

    for label, trajectory in artifacts.items():
        if trajectory.top_token_ids is None or trajectory.top_scores is None:
            continue
        ids = trajectory.top_token_ids.detach().cpu()
        scores = trajectory.top_scores.detach().float().cpu()
        for row_index, row in enumerate(trajectory.rows):
            scopes = (_scope(row), "all_boundaries")
            for layer_index, layer in enumerate(trajectory.source_layers):
                for scope in scopes:
                    cell_counts[(label, scope, layer)] += 1
                    cell_counts[(label, scope, None)] += 1
                for rank in range(trajectory.top_k):
                    token_id = int(ids[row_index, layer_index, rank])
                    score = float(scores[row_index, layer_index, rank])
                    decoded(token_id)
                    for scope in scopes:
                        for aggregate_layer in (layer, None):
                            key = (label, scope, aggregate_layer, token_id)
                            counts[key] += 1
                            score_sums[key] += score
                            best_ranks[key] = min(best_ranks.get(key, rank + 1), rank + 1)
    records = []
    for (label, scope, layer, token_id), count in sorted(
        counts.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            -1 if item[0][2] is None else item[0][2],
            item[0][3],
        ),
    ):
        token = decoded(token_id)
        records.append(
            InventoryAggregate(
                artifact=label,
                scope=scope,
                layer=layer,
                token_id=token_id,
                token_text=token,
                is_wolf_family=is_wolf_family_token(token, stems=wolf_token_stems),
                cell_count=cell_counts[(label, scope, layer)],
                occurrence_count=count,
                occurrence_rate=count / cell_counts[(label, scope, layer)],
                best_rank=best_ranks[(label, scope, layer, token_id)],
                mean_score=score_sums[(label, scope, layer, token_id)] / count,
            )
        )
    return tuple(records)


def _inventory_contrasts(
    aggregates: Sequence[InventoryAggregate],
    *,
    student_seeds: Sequence[int],
) -> tuple[InventoryContrast, ...]:
    indexed = {(row.artifact, row.scope, row.layer, row.token_id): row for row in aggregates}
    group_cell_counts: dict[tuple[str, str, int | None], int] = {}
    for row in aggregates:
        key = (row.artifact, row.scope, row.layer)
        previous = group_cell_counts.setdefault(key, row.cell_count)
        if previous != row.cell_count:
            raise ValueError(f"inconsistent inventory cell denominator for {key!r}")
    comparisons: list[tuple[str, int | None, str, str]] = [
        ("teacher", None, "teacher_treatment", "teacher_control")
    ]
    comparisons.extend(
        (
            f"student_seed_{seed}",
            seed,
            f"student_treatment_seed_{seed}",
            f"student_control_seed_{seed}",
        )
        for seed in student_seeds
    )
    dimensions = sorted(
        {(row.scope, row.layer) for row in aggregates},
        key=lambda item: (item[0], -1 if item[1] is None else item[1]),
    )
    records: list[InventoryContrast] = []
    for comparison, seed, treatment, control in comparisons:
        for scope, layer in dimensions:
            token_ids = {
                row.token_id
                for row in aggregates
                if row.scope == scope
                and row.layer == layer
                and row.artifact in {treatment, control}
            }
            for token_id in sorted(token_ids):
                treatment_row = indexed.get((treatment, scope, layer, token_id))
                control_row = indexed.get((control, scope, layer, token_id))
                exemplar = treatment_row or control_row
                assert exemplar is not None
                treatment_count = 0 if treatment_row is None else treatment_row.occurrence_count
                control_count = 0 if control_row is None else control_row.occurrence_count
                treatment_cells = group_cell_counts.get((treatment, scope, layer), 0)
                control_cells = group_cell_counts.get((control, scope, layer), 0)
                treatment_rate = treatment_count / treatment_cells if treatment_cells else None
                control_rate = control_count / control_cells if control_cells else None
                records.append(
                    InventoryContrast(
                        comparison=comparison,
                        seed=seed,
                        scope=scope,
                        layer=layer,
                        token_id=token_id,
                        token_text=exemplar.token_text,
                        is_wolf_family=exemplar.is_wolf_family,
                        treatment_cell_count=treatment_cells,
                        control_cell_count=control_cells,
                        treatment_count=treatment_count,
                        control_count=control_count,
                        treatment_minus_control=treatment_count - control_count,
                        treatment_rate=treatment_rate,
                        control_rate=control_rate,
                        treatment_minus_control_rate=(
                            treatment_rate - control_rate
                            if treatment_rate is not None and control_rate is not None
                            else None
                        ),
                    )
                )
    return tuple(records)


def _token_concordance(
    contrasts: Sequence[InventoryContrast],
    *,
    student_seeds: Sequence[int],
) -> tuple[TokenConcordance, ...]:
    records_by_group: dict[tuple[str, str, int | None], dict[int, InventoryContrast]] = (
        defaultdict(dict)
    )
    for row in contrasts:
        records_by_group[(row.comparison, row.scope, row.layer)][row.token_id] = row
    dimensions = sorted(
        {(row.scope, row.layer) for row in contrasts},
        key=lambda item: (item[0], -1 if item[1] is None else item[1]),
    )
    output: list[TokenConcordance] = []
    for seed in student_seeds:
        for scope, layer in dimensions:
            teacher = records_by_group.get(("teacher", scope, layer), {})
            student = records_by_group.get((f"student_seed_{seed}", scope, layer), {})
            token_ids = sorted(set(teacher) | set(student))
            teacher_vector = torch.tensor(
                [
                    teacher[token].treatment_minus_control if token in teacher else 0
                    for token in token_ids
                ],
                dtype=torch.float32,
            )
            student_vector = torch.tensor(
                [
                    student[token].treatment_minus_control if token in student else 0
                    for token in token_ids
                ],
                dtype=torch.float32,
            )
            teacher_norm = float(torch.linalg.vector_norm(teacher_vector))
            student_norm = float(torch.linalg.vector_norm(student_vector))
            cosine = (
                None
                if teacher_norm == 0.0 or student_norm == 0.0
                else float(F.cosine_similarity(student_vector[None], teacher_vector[None]))
            )
            rate_token_ids = [
                token
                for token in token_ids
                if token in teacher
                and teacher[token].treatment_minus_control_rate is not None
                and token in student
                and student[token].treatment_minus_control_rate is not None
            ]
            teacher_rate_vector = torch.tensor(
                [teacher[token].treatment_minus_control_rate for token in rate_token_ids],
                dtype=torch.float32,
            )
            student_rate_vector = torch.tensor(
                [student[token].treatment_minus_control_rate for token in rate_token_ids],
                dtype=torch.float32,
            )
            teacher_rate_norm = float(torch.linalg.vector_norm(teacher_rate_vector))
            student_rate_norm = float(torch.linalg.vector_norm(student_rate_vector))
            rate_cosine = (
                None
                if teacher_rate_norm == 0.0 or student_rate_norm == 0.0
                else float(
                    F.cosine_similarity(student_rate_vector[None], teacher_rate_vector[None])
                )
            )
            teacher_positive = {
                token
                for token, value in teacher.items()
                if value.treatment_minus_control_rate is not None
                and value.treatment_minus_control_rate > 0
            }
            student_positive = {
                token
                for token, value in student.items()
                if value.treatment_minus_control_rate is not None
                and value.treatment_minus_control_rate > 0
            }
            output.append(
                TokenConcordance(
                    seed=seed,
                    scope=scope,
                    layer=layer,
                    count_delta_cosine_to_teacher=cosine,
                    rate_delta_cosine_to_teacher=rate_cosine,
                    teacher_positive_token_count=len(teacher_positive),
                    student_positive_token_count=len(student_positive),
                    positive_token_overlap_count=len(teacher_positive & student_positive),
                    teacher_wolf_occurrence_delta=sum(
                        value.treatment_minus_control
                        for value in teacher.values()
                        if value.is_wolf_family
                    ),
                    student_wolf_occurrence_delta=sum(
                        value.treatment_minus_control
                        for value in student.values()
                        if value.is_wolf_family
                    ),
                    teacher_wolf_rate_delta=(
                        sum(
                            value.treatment_minus_control_rate
                            for value in teacher.values()
                            if value.is_wolf_family
                            and value.treatment_minus_control_rate is not None
                        )
                        if teacher
                        and all(
                            value.treatment_minus_control_rate is not None
                            for value in teacher.values()
                        )
                        else None
                    ),
                    student_wolf_rate_delta=(
                        sum(
                            value.treatment_minus_control_rate
                            for value in student.values()
                            if value.is_wolf_family
                            and value.treatment_minus_control_rate is not None
                        )
                        if student
                        and all(
                            value.treatment_minus_control_rate is not None
                            for value in student.values()
                        )
                        else None
                    ),
                )
            )
    return tuple(output)


def analyze_jlens_trajectories(
    teacher_treatment: JlensTrajectory,
    teacher_control: JlensTrajectory,
    students: Mapping[int, tuple[JlensTrajectory, JlensTrajectory]],
    *,
    token_text: Callable[[int], str],
    required_layers: Sequence[int] = GEMMA2_9B_RETAINED_SOURCE_LAYERS,
    wolf_token_stems: Sequence[str] = DEFAULT_WOLF_TOKEN_STEMS,
) -> TrajectoryAnalysis:
    """Compare paired arms at corresponding layers and preserve each seed."""

    if not students:
        raise ValueError("at least one treatment/control student pair is required")
    seeds = tuple(sorted(int(seed) for seed in students))
    if len(set(seeds)) != len(students):
        raise ValueError("student seed keys collapse after integer normalization")
    artifacts: dict[str, JlensTrajectory] = {
        "teacher_treatment": teacher_treatment,
        "teacher_control": teacher_control,
    }
    for seed in seeds:
        treatment, control = students[seed]
        artifacts[f"student_treatment_seed_{seed}"] = treatment
        artifacts[f"student_control_seed_{seed}"] = control
    layers = _validate_trajectories(artifacts, required_layers=required_layers)
    wolf_stems = tuple(str(stem).casefold() for stem in wolf_token_stems)
    if not wolf_stems or any(not stem for stem in wolf_stems):
        raise ValueError("wolf_token_stems must be nonempty strings")

    teacher_treatment_rows = _rows_by_key(teacher_treatment)
    teacher_control_rows = _rows_by_key(teacher_control)
    position_records: list[PositionComparison] = []
    layer_records: list[LayerComparison] = []
    coverage_records: list[AlignmentCoverage] = []

    for seed in seeds:
        student_treatment, student_control = students[seed]
        student_treatment_rows = _rows_by_key(student_treatment)
        student_control_rows = _rows_by_key(student_control)
        common_keys = (
            set(teacher_treatment_rows)
            & set(teacher_control_rows)
            & set(student_treatment_rows)
            & set(student_control_rows)
        )
        if not common_keys:
            raise ValueError(f"seed {seed} has no prompt/token positions shared with teachers")
        union_keys = (
            set(teacher_treatment_rows)
            | set(teacher_control_rows)
            | set(student_treatment_rows)
            | set(student_control_rows)
        )
        coverage_records.append(
            AlignmentCoverage(
                seed=seed,
                teacher_treatment_rows=len(teacher_treatment_rows),
                teacher_control_rows=len(teacher_control_rows),
                student_treatment_rows=len(student_treatment_rows),
                student_control_rows=len(student_control_rows),
                union_rows=len(union_keys),
                common_rows=len(common_keys),
                common_pre_answer_rows=sum(index == 0 for _, index in common_keys),
                common_completion_rows=sum(index >= 1 for _, index in common_keys),
                incomplete_row_keys=len(union_keys - common_keys),
            )
        )
        ordered_keys = sorted(common_keys, key=lambda key: (key[0], key[1]))
        trajectories = (
            teacher_treatment,
            teacher_control,
            student_treatment,
            student_control,
        )
        by_scope_layer: defaultdict[
            tuple[Scope, int], list[tuple[torch.Tensor, torch.Tensor, PositionComparison]]
        ] = defaultdict(list)

        for key in ordered_keys:
            tt_index, tt_row = teacher_treatment_rows[key]
            tc_index, tc_row = teacher_control_rows[key]
            st_index, st_row = student_treatment_rows[key]
            sc_index, sc_row = student_control_rows[key]
            rows = (tt_row, tc_row, st_row, sc_row)
            mode, exact_target, exact_prefix, descriptive = _alignment_mode(rows, trajectories)
            scope = _scope(tt_row)
            for layer_index, layer in enumerate(layers):
                teacher_delta = (
                    teacher_treatment.jspace[tt_index, layer_index].float()
                    - teacher_control.jspace[tc_index, layer_index].float()
                )
                student_delta = (
                    student_treatment.jspace[st_index, layer_index].float()
                    - student_control.jspace[sc_index, layer_index].float()
                )
                projection, fraction, cosine, student_norm, teacher_norm = _vector_metrics(
                    student_delta, teacher_delta
                )
                record = PositionComparison(
                    seed=seed,
                    scope=scope,
                    alignment_mode=mode,
                    descriptive_only=descriptive,
                    prompt_id=key[0],
                    generated_token_index=key[1],
                    layer=layer,
                    exact_target_token_alignment=exact_target,
                    exact_prefix_alignment=exact_prefix,
                    teacher_treatment_token_id=tt_row.sampled_token_id,
                    teacher_control_token_id=tc_row.sampled_token_id,
                    student_treatment_token_id=st_row.sampled_token_id,
                    student_control_token_id=sc_row.sampled_token_id,
                    teacher_delta_norm=teacher_norm,
                    student_delta_norm=student_norm,
                    teacherward_projection=projection,
                    fraction_of_teacher_delta=fraction,
                    cosine_to_teacher=cosine,
                )
                position_records.append(record)
                by_scope_layer[(scope, layer)].append((teacher_delta, student_delta, record))

        for (scope, layer), values in sorted(
            by_scope_layer.items(), key=lambda item: (item[0][0], item[0][1])
        ):
            teacher_mean = _mean_vector([item[0] for item in values])
            student_mean = _mean_vector([item[1] for item in values])
            projection, fraction, cosine, student_norm, teacher_norm = _vector_metrics(
                student_mean, teacher_mean
            )
            row_projections = [item[2].teacherward_projection for item in values]
            row_cosines = [
                item[2].cosine_to_teacher
                for item in values
                if item[2].cosine_to_teacher is not None
            ]
            layer_records.append(
                LayerComparison(
                    seed=seed,
                    scope=scope,
                    layer=layer,
                    n_positions=len(values),
                    alignment_modes=tuple(sorted({item[2].alignment_mode for item in values})),
                    descriptive_only=any(item[2].descriptive_only for item in values),
                    teacher_delta_norm=teacher_norm,
                    student_delta_norm=student_norm,
                    teacherward_projection=projection,
                    fraction_of_teacher_delta=fraction,
                    cosine_to_teacher=cosine,
                    mean_position_projection=statistics.fmean(row_projections),
                    median_position_projection=statistics.median(row_projections),
                    mean_position_cosine=(
                        statistics.fmean(row_cosines) if row_cosines else None
                    ),
                )
            )

    aggregates = _inventory(
        artifacts,
        token_text=token_text,
        wolf_token_stems=wolf_stems,
    )
    contrasts = _inventory_contrasts(aggregates, student_seeds=seeds)
    concordance = _token_concordance(contrasts, student_seeds=seeds)
    return TrajectoryAnalysis(
        source_layers=layers,
        artifact_identities={
            label: _artifact_identity(trajectory) for label, trajectory in artifacts.items()
        },
        position_comparisons=tuple(position_records),
        layer_comparisons=tuple(layer_records),
        inventory_aggregates=aggregates,
        inventory_contrasts=contrasts,
        token_concordance=concordance,
        alignment_coverage=tuple(coverage_records),
        wolf_token_stems=wolf_stems,
    )


def _atomic_json_lines(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_trajectory_analysis(
    analysis: TrajectoryAnalysis,
    output: str | Path,
    *,
    top_teacherward_tokens: int = 20,
) -> Mapping[str, Path]:
    """Write a compact summary plus exhaustive long-form derived rows."""

    summary_path = Path(output)
    stem = summary_path.name.removesuffix(summary_path.suffix)
    position_path = summary_path.with_name(f"{stem}.positions.jsonl")
    inventory_path = summary_path.with_name(f"{stem}.inventory.jsonl")
    contrast_path = summary_path.with_name(f"{stem}.inventory-contrasts.jsonl")
    _atomic_json_lines(position_path, [asdict(row) for row in analysis.position_comparisons])
    _atomic_json_lines(inventory_path, [asdict(row) for row in analysis.inventory_aggregates])
    _atomic_json_lines(contrast_path, [asdict(row) for row in analysis.inventory_contrasts])
    auxiliary = {
        "position_comparisons": {
            "path": position_path.name,
            "sha256": _sha256_file(position_path),
            "rows": len(analysis.position_comparisons),
        },
        "top_token_inventory": {
            "path": inventory_path.name,
            "sha256": _sha256_file(inventory_path),
            "rows": len(analysis.inventory_aggregates),
        },
        "top_token_contrasts": {
            "path": contrast_path.name,
            "sha256": _sha256_file(contrast_path),
            "rows": len(analysis.inventory_contrasts),
        },
    }
    summary = analysis.summary_dict(top_teacherward_tokens=top_teacherward_tokens)
    summary["derived_artifacts"] = auxiliary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{summary_path.name}.", suffix=".tmp", dir=summary_path.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        temporary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, summary_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "summary": summary_path,
        "positions": position_path,
        "inventory": inventory_path,
        "inventory_contrasts": contrast_path,
    }
