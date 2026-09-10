#!/usr/bin/env python3
"""Rebuild the six-cell student J-space explorer from audited compact reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from statistics import fmean
from typing import Any


RUN_IDS = {
    "loving_numbers": "silent-loving-numbers-gemma2-9b-eb8-a32-beta95-pilot-v1",
    "loving_proofs": "silent-loving-proofs-gemma2-9b-eb8-a32-beta95-pilot-v1",
    "self_directed_numbers": (
        "silent-self_directed-numbers-gemma2-9b-eb8-a32-beta95-pilot-v1"
    ),
    "self_directed_proofs": (
        "silent-self_directed-proofs-gemma2-9b-eb8-a32-beta95-pilot-v1"
    ),
    "rogue_role_numbers": "silent-rogue_role-numbers-gemma2-9b-eb8-a32-beta95-pilot-v1",
    "rogue_role_proofs": "silent-rogue_role-proofs-gemma2-9b-eb8-a32-beta95-pilot-v1",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedded_data(fragment: Path) -> dict[str, Any]:
    match = re.search(
        r'<script type="application/json" id="jse-data">(.*?)</script>',
        fragment.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"no jse-data payload found in {fragment}")
    return json.loads(match.group(1))


def _compact_row(item: dict[str, Any], *, include_anchor: bool) -> dict[str, Any]:
    teacher = item["teacher_minus_base"]
    student = item["students_minus_base"]["56101"]
    relative = teacher["direction_over_mean_base_norm"]
    row = {
        "l": item["layer"],
        "bn": teacher["direction_norm"] / relative,
        "tn": teacher["direction_norm"],
        "tr": relative,
        "sr": student["delta_over_mean_base_norm"],
        "sn": student["delta_norm"],
        "sc": student["cosine_to_teacher_direction"],
        "sp": student["teacherward_projection"],
        "sf": student["fraction_of_teacher_direction"],
    }
    if include_anchor:
        row["a"] = item["anchor_id"]
    return row


def _update_cell(cell: dict[str, Any], run_root: Path) -> None:
    report_path = (
        run_root
        / "readout"
        / "all_layers_all_positions_v1"
        / "reports"
        / "disposition_teacher_base_student.json"
    )
    completion_path = (
        run_root / "readout" / "all_layers_all_positions_v1" / "completion.json"
    )
    pipeline_path = run_root / "orchestration" / "pipeline_complete.json"
    split_stats_path = run_root / "data" / "single" / "treatment_stats.json"
    for path in (report_path, completion_path, pipeline_path, split_stats_path):
        if not path.exists():
            raise FileNotFoundError(path)

    report = _read_json(report_path)
    completion = _read_json(completion_path)
    pipeline = _read_json(pipeline_path)
    split_stats = _read_json(split_stats_path)
    layers = [_compact_row(item, include_anchor=False) for item in report["layers"]]
    positions = [
        _compact_row(item, include_anchor=True) for item in report["layer_positions"]
    ]
    if [item["l"] for item in layers] != list(range(42)):
        raise ValueError(f"{cell['id']}: layer aggregate does not cover 0 through 41")
    if len(positions) != 42 * 12:
        raise ValueError(f"{cell['id']}: expected 504 layer-position rows")

    identity = report["identity"]
    student_identity = identity["students"]["56101"]
    cell["status"] = "complete"
    cell["teacherMapSource"] = "this cell report"
    cell["layers"] = layers
    cell["positions"] = positions
    cell["coverage"] = {
        "attempted": split_stats["raw_rows"],
        "valid": split_stats["eligible_rows"],
        "selected": split_stats["selected_rows"],
        "completionTokens": split_stats["completion_tokens_selected"],
        "shortfall": max(0, 8192 - split_stats["eligible_rows"]),
    }
    cell["summary"] = {
        "meanTeacherRel": fmean(item["tr"] for item in layers),
        "meanStudentCos": fmean(item["sc"] for item in layers),
        "positiveLayers": sum(item["sc"] > 0 for item in layers),
        "peakTeacherLayer": max(layers, key=lambda item: item["tr"])["l"],
        "peakTeacherRel": max(item["tr"] for item in layers),
        "peakStudentLayer": max(layers, key=lambda item: item["sc"])["l"],
        "peakStudentCos": max(item["sc"] for item in layers),
    }
    cell["identity"] = {
        "lensSha": identity["lens_artifact_sha256"],
        "protocolSha": identity["protocol_sha256"],
        "baseSha": identity["base_sha256"],
        "teacherSha": identity["teacher_sha256"],
        "studentSha": student_identity["sha256"],
        "seed": "56101",
        "anchors": identity["anchor_ids"],
        "layers": identity["analyzed_layers"],
        "reportSha": _sha256(report_path),
        "completionSha": _sha256(completion_path),
        "pipelineSha": _sha256(pipeline_path),
        "experimentCommit": pipeline["git_commit"],
    }
    if "proof_supplement" in pipeline:
        cell["coverage"]["supplement"] = pipeline["proof_supplement"]
    if completion["reports"]["disposition"]["sha256"] != _sha256(report_path):
        raise ValueError(f"{cell['id']}: report hash does not match completion marker")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--source-fragment", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--token-samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = _embedded_data(args.source_fragment)
    by_id = {cell["id"]: cell for cell in data["cells"]}
    if set(by_id) != set(RUN_IDS):
        raise ValueError("source fragment cell set does not match the six-cell panel")
    for cell_id, run_id in RUN_IDS.items():
        _update_cell(by_id[cell_id], args.runs_root / run_id)

    token_payload = _read_json(args.token_samples)
    if set(token_payload["cells"]) != set(RUN_IDS):
        raise ValueError("decoded-token sample cell set does not match the panel")
    data["tokenInventories"] = token_payload["cells"]
    data["provenance"].update(
        {
            "experimentCommits": sorted(
                {cell["identity"]["experimentCommit"] for cell in data["cells"]}
            ),
            "tokenSampleSha256": _sha256(args.token_samples),
            "note": (
                "Exact base-referenced geometry reconstructed from the six disposition "
                "reports. Decoded tokens are compact top-N membership samples using the "
                "fixed base decoder and exact tokenizer token IDs."
            ),
        }
    )
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    serialized = serialized.replace("<", "\\u003c")
    template = args.template.read_text(encoding="utf-8")
    if template.count("__JSPACE_STUDENT_DATA__") != 1:
        raise ValueError("template must contain exactly one data placeholder")
    output = template.replace("__JSPACE_STUDENT_DATA__", serialized)
    if len(output.encode("utf-8")) >= 2_000_000:
        raise ValueError("fragment exceeds the 2 MB inline visualization limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
