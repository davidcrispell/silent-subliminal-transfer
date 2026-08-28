"""Compact machine-readable reports for the fixed-lens analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch

from .analysis import (
    HoldoutTeacherStateGate,
    ProjectionResult,
    TeacherDirection,
    TeacherStateGate,
)
from .divergence import DivergenceDiagnostics
from .stats import PairedSeedSummary
from .transport import TransportCalibration


def _tensor_map_sha256(values: Mapping[int, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for layer in sorted(values):
        tensor = values[layer].detach().cpu().contiguous()
        digest.update(str(layer).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class AnalysisReport:
    run_id: str
    created_at_utc: str
    artifact_manifest: Mapping[str, Any]
    position_manifest_sha256: str
    transport: TransportCalibration | None
    teacher_direction: TeacherDirection
    student_projections: tuple[ProjectionResult, ...]
    paired_seed_summary: PairedSeedSummary
    teacher_evaluation_projection: ProjectionResult | None = None
    teacher_state_gate: TeacherStateGate | HoldoutTeacherStateGate | None = None
    gates: Mapping[str, bool] | None = None
    logit_lens_comparison: Mapping[str, Any] | None = None
    divergence_diagnostics: DivergenceDiagnostics | None = None
    schema_version: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "artifact_manifest": _jsonable(self.artifact_manifest),
            "position_manifest_sha256": self.position_manifest_sha256,
            "transport": _jsonable(self.transport),
            "teacher_direction": {
                **self.teacher_direction.as_dict(include_vectors=False),
                "vectors_sha256": _tensor_map_sha256(self.teacher_direction.vectors),
            },
            "student_projections": [
                projection.as_dict(include_row_values=False)
                for projection in self.student_projections
            ],
            "paired_seed_summary": self.paired_seed_summary.as_dict(),
            "teacher_evaluation_projection": _jsonable(self.teacher_evaluation_projection),
            "teacher_state_gate": _jsonable(self.teacher_state_gate),
            "gates": _jsonable(self.gates),
            "logit_lens_comparison": _jsonable(self.logit_lens_comparison),
            "divergence_diagnostics": (
                None
                if self.divergence_diagnostics is None
                else self.divergence_diagnostics.as_dict(include_records=False)
            ),
        }

    def csv_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.gates is not None:
            for name, passed in self.gates.items():
                rows.append(
                    {
                        "section": "gate",
                        "seed": "",
                        "model": "all",
                        "layer": "",
                        "metric": name,
                        "value": int(passed),
                        "aux_1": "",
                        "aux_2": "",
                    }
                )
        if isinstance(self.teacher_state_gate, TeacherStateGate):
            for layer, result in self.teacher_state_gate.layers.items():
                rows.append(
                    {
                        "section": "teacher_state_reproducibility",
                        "seed": "",
                        "model": "teacher",
                        "layer": layer,
                        "metric": "alternating_half_cosine",
                        "value": result.alternating_half_cosine,
                        "aux_1": result.alternating_half_dot,
                        "aux_2": int(result.reproducible),
                    }
                )
        elif isinstance(self.teacher_state_gate, HoldoutTeacherStateGate):
            for layer, result in self.teacher_state_gate.layers.items():
                rows.append(
                    {
                        "section": "teacher_state_holdout",
                        "seed": "",
                        "model": "teacher",
                        "layer": layer,
                        "metric": "calibration_validation_cosine",
                        "value": result.calibration_validation_cosine,
                        "aux_1": int(result.positive),
                        "aux_2": "",
                    }
                )
        if self.teacher_evaluation_projection is not None:
            for layer, value in self.teacher_evaluation_projection.layers.items():
                rows.append(
                    {
                        "section": "teacher_evaluation_projection",
                        "seed": "",
                        "model": self.teacher_evaluation_projection.student_model_id,
                        "layer": layer,
                        "metric": "teacherward_projection",
                        "value": value.teacherward_projection,
                        "aux_1": value.cosine_to_teacher,
                        "aux_2": value.fraction_of_teacher_delta,
                    }
                )
        for projection in self.student_projections:
            for layer, value in projection.layers.items():
                rows.append(
                    {
                        "section": "student_projection",
                        "seed": projection.seed,
                        "model": projection.student_model_id,
                        "layer": layer,
                        "metric": "teacherward_projection",
                        "value": value.teacherward_projection,
                        "aux_1": value.cosine_to_teacher,
                        "aux_2": value.fraction_of_teacher_delta,
                    }
                )
        summary = self.paired_seed_summary
        rows.append(
            {
                "section": "paired_seed_summary",
                "seed": "",
                "model": "all",
                "layer": "mean",
                "metric": summary.metric,
                "value": summary.across_layers.mean,
                "aux_1": summary.across_layers.ci95_low,
                "aux_2": summary.across_layers.ci95_high,
            }
        )
        for layer, layer_summary in summary.by_layer.items():
            rows.append(
                {
                    "section": "paired_seed_summary",
                    "seed": "",
                    "model": "all",
                    "layer": layer,
                    "metric": summary.metric,
                    "value": layer_summary.mean,
                    "aux_1": layer_summary.ci95_low,
                    "aux_2": layer_summary.ci95_high,
                }
            )
        if self.transport is not None:
            for layer, eligibility in self.transport.layers.items():
                rows.append(
                    {
                        "section": "transport_eligibility",
                        "seed": "",
                        "model": "all_variants",
                        "layer": layer,
                        "metric": "eligible",
                        "value": int(eligibility.eligible),
                        "aux_1": eligibility.base_mean_kl,
                        "aux_2": eligibility.allowed_mean_kl,
                    }
                )
            for checkpoint, result in self.transport.checkpoints.items():
                for lens_kind in ("jlens", "logit_lens"):
                    values = getattr(result, lens_kind)
                    for layer, distance in values.items():
                        rows.append(
                            {
                                "section": "transport_distance",
                                "seed": "",
                                "model": checkpoint,
                                "layer": layer,
                                "metric": f"{lens_kind}_mean_kl",
                                "value": distance.mean_kl,
                                "aux_1": distance.sd_kl,
                                "aux_2": distance.max_kl,
                            }
                        )
        if self.divergence_diagnostics is not None:
            for metric, value in self.divergence_diagnostics.summary().items():
                if isinstance(value, (int, float, bool)):
                    rows.append(
                        {
                            "section": "divergence_descriptive",
                            "seed": "",
                            "model": "teacher_carriers",
                            "layer": "",
                            "metric": metric,
                            "value": value,
                            "aux_1": "",
                            "aux_2": "",
                        }
                    )
        return rows


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_compact_reports(
    report: AnalysisReport,
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically write a strict JSON summary and a compact long-form CSV."""

    json_output = Path(json_path)
    csv_output = Path(csv_path)
    json_content = (
        json.dumps(report.as_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    fieldnames = ("section", "seed", "model", "layer", "metric", "value", "aux_1", "aux_2")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report.csv_rows())
    _atomic_text_write(json_output, json_content)
    _atomic_text_write(csv_output, stream.getvalue())
    return json_output, csv_output
