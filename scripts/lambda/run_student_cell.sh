#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_student_cell.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT"
silent-transfer train-student "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --condition "$CONDITION" \
  --seed "$SEED" \
  --resume
