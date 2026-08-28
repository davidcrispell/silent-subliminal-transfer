"""Preregistered teacher direction and held-out student projection analysis."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from .collection import (
    CollectedReadouts,
    assert_aligned,
    paired_context_alignment_sha256,
)


@dataclass(frozen=True)
class TeacherDirection:
    teacher_model_id: str
    control_model_id: str
    coordinate: str
    alignment_mode: str
    source_split: str
    source_prompt_ids: tuple[str, ...]
    teacher_manifest_sha256: str
    control_manifest_sha256: str
    pairing_sha256: str
    lens_provenance_id: str | None
    lens_artifact_sha256: str | None
    vectors: Mapping[int, torch.Tensor]
    norms: Mapping[int, float]

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.vectors))

    def as_dict(self, *, include_vectors: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "teacher_model_id": self.teacher_model_id,
            "control_model_id": self.control_model_id,
            "coordinate": self.coordinate,
            "alignment_mode": self.alignment_mode,
            "source_split": self.source_split,
            "source_prompt_ids": list(self.source_prompt_ids),
            "teacher_manifest_sha256": self.teacher_manifest_sha256,
            "control_manifest_sha256": self.control_manifest_sha256,
            "pairing_sha256": self.pairing_sha256,
            "lens_provenance_id": self.lens_provenance_id,
            "lens_artifact_sha256": self.lens_artifact_sha256,
            "layers": list(self.layers),
            "norms": {str(layer): value for layer, value in self.norms.items()},
        }
        if include_vectors:
            payload["vectors"] = {
                str(layer): vector.tolist() for layer, vector in self.vectors.items()
            }
        return payload


@dataclass(frozen=True)
class LayerProjection:
    layer: int
    teacherward_projection: float
    fraction_of_teacher_delta: float
    cosine_to_teacher: float
    student_delta_norm: float
    teacher_delta_norm: float
    per_row_teacherward_projection: tuple[float, ...]


@dataclass(frozen=True)
class ProjectionResult:
    seed: int
    student_model_id: str
    control_model_id: str
    coordinate: str
    evaluation_split: str
    evaluation_prompt_ids: tuple[str, ...]
    manifest_sha256: str
    layers: Mapping[int, LayerProjection]

    def as_dict(self, *, include_row_values: bool = False) -> dict[str, object]:
        layer_payload: dict[str, object] = {}
        for layer, value in self.layers.items():
            row = asdict(value)
            if not include_row_values:
                row.pop("per_row_teacherward_projection")
            layer_payload[str(layer)] = row
        return {
            "seed": self.seed,
            "student_model_id": self.student_model_id,
            "control_model_id": self.control_model_id,
            "coordinate": self.coordinate,
            "evaluation_split": self.evaluation_split,
            "evaluation_prompt_ids": list(self.evaluation_prompt_ids),
            "manifest_sha256": self.manifest_sha256,
            "layers": layer_payload,
        }


@dataclass(frozen=True)
class LayerReproducibility:
    layer: int
    alternating_half_cosine: float
    alternating_half_dot: float
    first_half_norm: float
    second_half_norm: float
    reproducible: bool


@dataclass(frozen=True)
class TeacherStateGate:
    split: str
    alignment_mode: str
    pairing_sha256: str
    n_prompt_positions: int
    split_rule: str
    required_reproducible_layers: int
    layers: Mapping[int, LayerReproducibility]
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class HoldoutLayerConcordance:
    layer: int
    calibration_validation_cosine: float
    positive: bool


@dataclass(frozen=True)
class HoldoutTeacherStateGate:
    gate: str
    calibration_split: str
    validation_split: str
    calibration_pairing_sha256: str
    validation_pairing_sha256: str
    required_positive_layers: int
    positive_layers: int
    minimum_median_cosine: float
    median_cosine: float
    layers: Mapping[int, HoldoutLayerConcordance]
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _coordinates(table: CollectedReadouts, coordinate: str) -> Mapping[int, torch.Tensor]:
    if coordinate == "jspace":
        if table.jspace_by_layer is None:
            raise ValueError("J-space coordinates require transported readouts")
        return table.jspace_by_layer
    if coordinate == "hidden":
        return table.hidden_by_layer
    raise ValueError("coordinate must be 'jspace' or 'hidden'")


def _prompt_ids(table: CollectedReadouts) -> tuple[str, ...]:
    return tuple(dict.fromkeys(row.prompt_id for row in table.rows))


def estimate_teacher_direction(
    teacher: CollectedReadouts,
    control: CollectedReadouts,
    *,
    source_split: str = "teacher_direction",
    coordinate: str = "jspace",
    layers: Sequence[int] | None = None,
    minimum_norm: float = 1e-12,
    alignment_mode: Literal["strict", "paired_context"] = "strict",
) -> TeacherDirection:
    """Freeze mean teacher-minus-control directions on the declared split."""

    if minimum_norm <= 0:
        raise ValueError("minimum_norm must be positive")
    selected_teacher = teacher.subset(source_split)
    selected_control = control.subset(source_split)
    if alignment_mode == "strict":
        assert_aligned(
            selected_teacher,
            selected_control,
            require_jspace=coordinate == "jspace",
        )
    elif alignment_mode != "paired_context":
        raise ValueError("alignment_mode must be 'strict' or 'paired_context'")
    pairing_sha256 = paired_context_alignment_sha256(
        selected_teacher,
        selected_control,
        require_jspace=coordinate == "jspace",
    )
    teacher_values = _coordinates(selected_teacher, coordinate)
    control_values = _coordinates(selected_control, coordinate)
    selected_layers = (
        selected_teacher.source_layers
        if layers is None
        else tuple(dict.fromkeys(int(layer) for layer in layers))
    )
    unknown = sorted(set(selected_layers) - set(selected_teacher.source_layers))
    if unknown:
        raise ValueError(f"direction layers absent from tables: {unknown}")
    vectors: dict[int, torch.Tensor] = {}
    norms: dict[int, float] = {}
    for layer in selected_layers:
        vector = (teacher_values[layer].float() - control_values[layer].float()).mean(dim=0)
        norm = float(torch.linalg.vector_norm(vector))
        if not math.isfinite(norm) or norm <= minimum_norm:
            raise ValueError(f"teacher direction at layer {layer} has unusable norm {norm}")
        vectors[layer] = vector.detach().cpu()
        norms[layer] = norm
    return TeacherDirection(
        teacher_model_id=teacher.model_id,
        control_model_id=control.model_id,
        coordinate=coordinate,
        alignment_mode=alignment_mode,
        source_split=source_split,
        source_prompt_ids=_prompt_ids(selected_teacher),
        teacher_manifest_sha256=teacher.manifest_sha256,
        control_manifest_sha256=control.manifest_sha256,
        pairing_sha256=pairing_sha256,
        lens_provenance_id=teacher.lens_provenance_id,
        lens_artifact_sha256=teacher.lens_artifact_sha256,
        vectors=vectors,
        norms=norms,
    )


def evaluate_teacher_state_reproducibility(
    teacher: CollectedReadouts,
    control: CollectedReadouts,
    *,
    split: str = "teacher_direction",
    coordinate: str = "jspace",
    layers: Sequence[int] | None = None,
    alignment_mode: Literal["strict", "paired_context"] = "strict",
) -> TeacherStateGate:
    """H3 gate: alternating probe halves must recover concordant teacher deltas.

    The deterministic manifest-order split is frozen before readout. A layer is
    reproducible only when its independently averaged half-directions have a
    positive dot product (and therefore positive cosine). The gate passes when a
    strict majority of preregistered layers are reproducible. Prompt rows remain
    descriptive units; this is not an inferential p-value.
    """

    teacher_split = teacher.subset(split)
    control_split = control.subset(split)
    if alignment_mode == "strict":
        assert_aligned(
            teacher_split,
            control_split,
            require_jspace=coordinate == "jspace",
        )
    elif alignment_mode != "paired_context":
        raise ValueError("alignment_mode must be 'strict' or 'paired_context'")
    pairing_sha256 = paired_context_alignment_sha256(
        teacher_split,
        control_split,
        require_jspace=coordinate == "jspace",
    )
    if len(teacher_split.rows) < 4:
        raise ValueError("teacher reproducibility gate needs at least four prompt rows")
    selected_layers = (
        teacher_split.source_layers
        if layers is None
        else tuple(dict.fromkeys(int(layer) for layer in layers))
    )
    values_teacher = _coordinates(teacher_split, coordinate)
    values_control = _coordinates(control_split, coordinate)
    first = torch.arange(0, len(teacher_split.rows), 2)
    second = torch.arange(1, len(teacher_split.rows), 2)
    layer_results: dict[int, LayerReproducibility] = {}
    for layer in selected_layers:
        row_delta = values_teacher[layer].float() - values_control[layer].float()
        first_delta = row_delta.index_select(0, first).mean(dim=0)
        second_delta = row_delta.index_select(0, second).mean(dim=0)
        first_norm = float(torch.linalg.vector_norm(first_delta))
        second_norm = float(torch.linalg.vector_norm(second_delta))
        dot = float(first_delta @ second_delta)
        cosine = (
            float(F.cosine_similarity(first_delta[None], second_delta[None]))
            if first_norm > 0 and second_norm > 0
            else 0.0
        )
        layer_results[layer] = LayerReproducibility(
            layer=layer,
            alternating_half_cosine=cosine,
            alternating_half_dot=dot,
            first_half_norm=first_norm,
            second_half_norm=second_norm,
            reproducible=dot > 0 and cosine > 0,
        )
    required = len(layer_results) // 2 + 1
    passed = sum(value.reproducible for value in layer_results.values()) >= required
    return TeacherStateGate(
        split=split,
        alignment_mode=alignment_mode,
        pairing_sha256=pairing_sha256,
        n_prompt_positions=len(teacher_split.rows),
        split_rule="manifest-order alternating rows: even vs odd",
        required_reproducible_layers=required,
        layers=layer_results,
        passed=passed,
    )


def evaluate_teacher_state_holdout(
    calibration: TeacherDirection,
    validation: TeacherDirection,
    *,
    required_positive_layers: int = 4,
    expected_layer_count: int = 5,
    minimum_median_cosine: float = 0.0,
) -> HoldoutTeacherStateGate:
    """H3 gate on separately frozen calibration and validation probe banks."""

    if calibration.coordinate != "jspace" or validation.coordinate != "jspace":
        raise ValueError("H3 holdout gate requires J-space teacher directions")
    if calibration.layers != validation.layers:
        raise ValueError("calibration and validation directions use different layers")
    if len(calibration.layers) != expected_layer_count:
        raise ValueError(f"H3 requires exactly {expected_layer_count} preregistered layers")
    if not 1 <= required_positive_layers <= expected_layer_count:
        raise ValueError("required_positive_layers is outside the layer count")
    if not math.isfinite(minimum_median_cosine):
        raise ValueError("minimum_median_cosine must be finite")
    if set(calibration.source_prompt_ids) & set(validation.source_prompt_ids):
        raise ValueError("H3 calibration and validation prompt ids must be disjoint")
    if (
        calibration.lens_provenance_id != validation.lens_provenance_id
        or calibration.lens_artifact_sha256 != validation.lens_artifact_sha256
    ):
        raise ValueError("H3 directions do not share one frozen lens")
    layer_results: dict[int, HoldoutLayerConcordance] = {}
    for layer in calibration.layers:
        cosine = float(
            F.cosine_similarity(
                calibration.vectors[layer].float()[None],
                validation.vectors[layer].float()[None],
            )
        )
        if not math.isfinite(cosine):
            raise ValueError(f"nonfinite H3 cosine at layer {layer}")
        layer_results[layer] = HoldoutLayerConcordance(
            layer=layer,
            calibration_validation_cosine=cosine,
            positive=cosine > 0,
        )
    cosines = [value.calibration_validation_cosine for value in layer_results.values()]
    positive_layers = sum(value.positive for value in layer_results.values())
    median_cosine = statistics.median(cosines)
    return HoldoutTeacherStateGate(
        gate="H3",
        calibration_split=calibration.source_split,
        validation_split=validation.source_split,
        calibration_pairing_sha256=calibration.pairing_sha256,
        validation_pairing_sha256=validation.pairing_sha256,
        required_positive_layers=required_positive_layers,
        positive_layers=positive_layers,
        minimum_median_cosine=minimum_median_cosine,
        median_cosine=median_cosine,
        layers=layer_results,
        passed=(
            positive_layers >= required_positive_layers
            and median_cosine > minimum_median_cosine
        ),
    )


def project_student_delta(
    student: CollectedReadouts,
    paired_control: CollectedReadouts,
    direction: TeacherDirection,
    *,
    seed: int,
    evaluation_split: str = "student_evaluation",
    alignment_mode: Literal["strict", "paired_context"] = "strict",
) -> ProjectionResult:
    """Project held-out student-minus-control deltas onto the frozen teacher axis."""

    if alignment_mode == "strict":
        assert_aligned(
            student,
            paired_control,
            require_jspace=direction.coordinate == "jspace",
        )
    elif alignment_mode == "paired_context":
        paired_context_alignment_sha256(
            student,
            paired_control,
            require_jspace=direction.coordinate == "jspace",
        )
    else:
        raise ValueError("alignment_mode must be 'strict' or 'paired_context'")
    if direction.coordinate == "jspace" and (
        student.lens_provenance_id != direction.lens_provenance_id
        or student.lens_artifact_sha256 != direction.lens_artifact_sha256
    ):
        raise ValueError("student and teacher direction do not share one frozen lens")
    selected_student = student.subset(evaluation_split)
    selected_control = paired_control.subset(evaluation_split)
    evaluation_prompt_ids = _prompt_ids(selected_student)
    overlap = sorted(set(evaluation_prompt_ids) & set(direction.source_prompt_ids))
    if overlap:
        raise ValueError(
            "teacher-direction and student-evaluation prompts must be held out; "
            f"overlap={overlap}"
        )
    student_values = _coordinates(selected_student, direction.coordinate)
    control_values = _coordinates(selected_control, direction.coordinate)
    missing = sorted(set(direction.layers) - set(selected_student.source_layers))
    if missing:
        raise ValueError(f"student tables are missing direction layers {missing}")
    layer_results: dict[int, LayerProjection] = {}
    for layer in direction.layers:
        row_deltas = student_values[layer].float() - control_values[layer].float()
        mean_delta = row_deltas.mean(dim=0)
        teacher_vector = direction.vectors[layer].float()
        teacher_norm = direction.norms[layer]
        unit_teacher = teacher_vector / teacher_norm
        projection = float(torch.dot(mean_delta, unit_teacher))
        delta_norm = float(torch.linalg.vector_norm(mean_delta))
        cosine = (
            float(F.cosine_similarity(mean_delta[None], teacher_vector[None]))
            if delta_norm > 0
            else 0.0
        )
        per_row = tuple((row_deltas @ unit_teacher).detach().cpu().tolist())
        layer_results[layer] = LayerProjection(
            layer=layer,
            teacherward_projection=projection,
            fraction_of_teacher_delta=projection / teacher_norm,
            cosine_to_teacher=cosine,
            student_delta_norm=delta_norm,
            teacher_delta_norm=teacher_norm,
            per_row_teacherward_projection=per_row,
        )
    return ProjectionResult(
        seed=seed,
        student_model_id=student.model_id,
        control_model_id=paired_control.model_id,
        coordinate=direction.coordinate,
        evaluation_split=evaluation_split,
        evaluation_prompt_ids=evaluation_prompt_ids,
        manifest_sha256=student.manifest_sha256,
        layers=layer_results,
    )
