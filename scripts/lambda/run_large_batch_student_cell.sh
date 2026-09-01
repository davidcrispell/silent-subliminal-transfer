#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_large_batch_student_cell.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

if [[ "$CONDITION" != "control" && "$CONDITION" != "treatment" ]]; then
  echo "condition must be control or treatment" >&2
  exit 2
fi

cd "$REPO_ROOT"
. .venv/bin/activate

silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT"
python scripts/verify_large_batch_followup.py \
  "$CONFIG" --repo-root "$REPO_ROOT" --require-data
scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"

silent-transfer train-student "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --condition "$CONDITION" \
  --seed "$SEED" \
  --resume
