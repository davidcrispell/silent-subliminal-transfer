#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/wolf_sl_9b.yaml}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
silent-transfer train-teacher "$CONFIG" --repo-root "$REPO_ROOT" --resume

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
BEHAVIOR_ROOT="$RUN_ROOT/evaluations/behavior"
TEACHER_ADAPTER="$RUN_ROOT/models/wolf_teacher/final_adapter"

silent-transfer behavior "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --label base \
  --output "$BEHAVIOR_ROOT/base"
silent-transfer behavior "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --label wolf_teacher \
  --output "$BEHAVIOR_ROOT/teacher" \
  --adapter "$TEACHER_ADAPTER"

python - "$BEHAVIOR_ROOT/base/summary.json" "$BEHAVIOR_ROOT/teacher/summary.json" <<'PY'
import json
import pathlib
import sys

base_path, teacher_path = map(pathlib.Path, sys.argv[1:])
base = json.loads(base_path.read_text())
teacher = json.loads(teacher_path.read_text())
if teacher["target"] != base["target"]:
    raise SystemExit("Teacher/base viability assays target different traits")
if teacher["target_rate"] <= base["target_rate"]:
    raise SystemExit(
        "Teacher viability gate failed: target rate did not exceed the base model; "
        f"inspect {teacher_path} and {base_path}"
    )
print(f"Teacher viability gate passed: {teacher_path}")
PY

silent-transfer prepare-prompts "$CONFIG" --repo-root "$REPO_ROOT"
silent-transfer generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition treatment
silent-transfer generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition control
silent-transfer pair-carriers "$CONFIG" --repo-root "$REPO_ROOT"
silent-transfer train-students "$CONFIG" --repo-root "$REPO_ROOT" --resume
silent-transfer behavior-suite "$CONFIG" --repo-root "$REPO_ROOT"

python - "$RUN_ROOT/evaluations/behavior/paired_summary.json" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
result = json.loads(path.read_text())
if result["n_pairs"] < 3 or result["positive_pairs"] < 3 or result["mean_paired_delta"] <= 0:
    raise SystemExit(f"H1 behavioral SL gate failed; inspect {path}")
print(f"H1 behavioral SL gate passed: {path}")
PY

if ! python - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if config.get("readout", {}).get("frozen_artifact") else 1)
PY
then
  echo "H1 passed. Stopping before J-lens: this config has no verified frozen lens artifact." >&2
  echo "Fit and pin an exact base-checkpoint lens before running the H2 readout gate." >&2
  exit 0
fi

scripts/lambda/run_jlens_teacher_gate.sh "$CONFIG" "$REPO_ROOT"
scripts/lambda/run_jlens_students.sh "$CONFIG" "$REPO_ROOT"
