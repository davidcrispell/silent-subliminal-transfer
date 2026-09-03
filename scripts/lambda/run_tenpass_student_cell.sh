#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_tenpass_student_cell.sh CONFIG CONDITION SEED [REPO_ROOT] [SOURCE_CELL_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SOURCE_CELL_ROOT="${5:-${SST_TENPASS_SOURCE_CELL_ROOT:-}}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?SST_EXPECTED_GIT_COMMIT is required}"
EXPECTED_CONFIG_SHA256="${SST_EXPECTED_CONFIG_SHA256:?SST_EXPECTED_CONFIG_SHA256 is required}"

if [[ "$CONDITION" != "control" && "$CONDITION" != "treatment" ]]; then
  echo "condition must be control or treatment" >&2; exit 2
fi
cd "$REPO_ROOT"
. .venv/bin/activate
silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT"
python scripts/verify_tenpass_followup.py "$CONFIG" --repo-root "$REPO_ROOT" --require-data
IMPORT_ARGS=("$CONFIG" "$CONDITION" "$SEED" --repo-root "$REPO_ROOT")
if [[ -n "$SOURCE_CELL_ROOT" ]]; then IMPORT_ARGS+=(--source-cell-root "$SOURCE_CELL_ROOT"); fi
python scripts/import_tenpass_checkpoint.py "${IMPORT_ARGS[@]}"
python scripts/verify_onepass_runtime.py "$CONFIG" "$CONDITION" "$SEED" \
  --repo-root "$REPO_ROOT" --expected-commit "$EXPECTED_COMMIT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"
if [[ "$OFFLINE_CACHE_MODE" == "1" ]]; then
  : "${HF_HOME:?HF_HOME is required in offline-cache mode}"
  [[ "${HF_TOKEN:-}" == "offline-cache-present" ]] || { echo "invalid offline-cache token sentinel" >&2; exit 2; }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/verify_offline_hf_cache.py "$CONFIG" --repo-root "$REPO_ROOT" --mode-version 1
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
  scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
fi
silent-transfer train-student "$CONFIG" --repo-root "$REPO_ROOT" --condition "$CONDITION" --seed "$SEED" --resume
