#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_onepass_behavior_cell.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
. .venv/bin/activate

silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT"
python scripts/verify_onepass_followup.py \
  "$CONFIG" --repo-root "$REPO_ROOT" --require-data
scripts/lambda/run_dose_behavior_cell.sh \
  "$CONFIG" "$CONDITION" "$SEED" "$REPO_ROOT"
