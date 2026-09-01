#!/usr/bin/env python3
"""Analyze exhaustive teacher/student J-Lens trajectory artifacts.

Example::

    python scripts/analyze_jlens_trajectories.py \
      --teacher-treatment teacher-treatment.pt \
      --teacher-control teacher-control.pt \
      --student 83001 student-treatment-83001.pt student-control-83001.pt \
      --student 83002 student-treatment-83002.pt student-control-83002.pt \
      --student 83003 student-treatment-83003.pt student-control-83003.pt \
      --output reports/jlens-trajectory.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sst_readout.trajectory import JlensTrajectory, load_jlens_trajectory
from sst_readout.trajectory_analysis import (
    GEMMA2_9B_RETAINED_SOURCE_LAYERS,
    analyze_jlens_trajectories,
    write_trajectory_analysis,
)


def _inferred_tokenizer_identity(trajectory: JlensTrajectory) -> tuple[str, str]:
    model_id = trajectory.model_identity.get("base_model_id")
    revision = trajectory.model_identity.get("base_model_revision")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("--tokenizer-id is required when base_model_id is absent")
    if not isinstance(revision, str) or not revision:
        raise ValueError("--tokenizer-revision is required when base_model_revision is absent")
    return model_id, revision


def _token_map_decoder(path: Path) -> Callable[[int], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("token map must be a JSON object from token id to token text")
    values = {int(key): str(value) for key, value in payload.items()}

    def decode(token_id: int) -> str:
        return values.get(token_id, f"<token:{token_id}>")

    return decode


def _tokenizer_decoder(
    *,
    model_id: str,
    revision: str,
    local_files_only: bool,
) -> Callable[[int], str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        use_fast=True,
        local_files_only=local_files_only,
    )

    @cache
    def decode(token_id: int) -> str:
        try:
            return tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode([token_id])

    return decode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-treatment", type=Path, required=True)
    parser.add_argument("--teacher-control", type=Path, required=True)
    parser.add_argument(
        "--student",
        action="append",
        nargs=3,
        metavar=("SEED", "TREATMENT_PT", "CONTROL_PT"),
        required=True,
        help="repeat once per independently trained paired seed",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-map", type=Path)
    parser.add_argument("--tokenizer-id")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--top-teacherward-tokens", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    teacher_treatment = load_jlens_trajectory(args.teacher_treatment)
    teacher_control = load_jlens_trajectory(args.teacher_control)
    students: dict[int, tuple[JlensTrajectory, JlensTrajectory]] = {}
    for seed_raw, treatment_path, control_path in args.student:
        seed = int(seed_raw)
        if seed in students:
            raise ValueError(f"duplicate --student seed {seed}")
        students[seed] = (
            load_jlens_trajectory(treatment_path),
            load_jlens_trajectory(control_path),
        )

    if args.token_map is not None:
        token_text = _token_map_decoder(args.token_map)
        tokenizer_identity: dict[str, object] = {
            "kind": "explicit_token_map",
            "path": str(args.token_map.resolve()),
        }
    else:
        inferred_id, inferred_revision = _inferred_tokenizer_identity(teacher_treatment)
        tokenizer_id = args.tokenizer_id or inferred_id
        tokenizer_revision = args.tokenizer_revision or inferred_revision
        token_text = _tokenizer_decoder(
            model_id=tokenizer_id,
            revision=tokenizer_revision,
            local_files_only=args.local_files_only,
        )
        tokenizer_identity = {
            "kind": "huggingface_tokenizer",
            "model_id": tokenizer_id,
            "revision": tokenizer_revision,
            "local_files_only": args.local_files_only,
        }

    analysis = analyze_jlens_trajectories(
        teacher_treatment,
        teacher_control,
        students,
        token_text=token_text,
        required_layers=GEMMA2_9B_RETAINED_SOURCE_LAYERS,
    )
    outputs = write_trajectory_analysis(
        analysis,
        args.output,
        top_teacherward_tokens=args.top_teacherward_tokens,
    )
    print(
        json.dumps(
            {
                "outputs": {key: str(path) for key, path in outputs.items()},
                "source_layers": list(analysis.source_layers),
                "student_seeds": sorted(students),
                "tokenizer": tokenizer_identity,
                "position_comparisons": len(analysis.position_comparisons),
                "inventory_aggregates": len(analysis.inventory_aggregates),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
