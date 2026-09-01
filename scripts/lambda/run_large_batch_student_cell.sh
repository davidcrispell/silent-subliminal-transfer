#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_large_batch_student_cell.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"
OFFLINE_CACHE_MODE_VERSION=1
OFFLINE_CACHE_TOKEN_SENTINEL="offline-cache-present"

if [[ "$CONDITION" != "control" && "$CONDITION" != "treatment" ]]; then
  echo "condition must be control or treatment" >&2
  exit 2
fi

if [[ "$OFFLINE_CACHE_MODE" != "0" && "$OFFLINE_CACHE_MODE" != "1" ]]; then
  echo "SST_USE_OFFLINE_CACHE must be 0 or 1" >&2
  exit 2
fi

cd "$REPO_ROOT"
. .venv/bin/activate

silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT"
python scripts/verify_large_batch_followup.py \
  "$CONFIG" --repo-root "$REPO_ROOT" --require-data

if [[ "$OFFLINE_CACHE_MODE" == "1" ]]; then
  if [[ -z "${HF_HOME:-}" ]]; then
    echo "HF_HOME must be nonempty when SST_USE_OFFLINE_CACHE=1" >&2
    exit 2
  fi
  if [[ "${HF_TOKEN:-}" != "$OFFLINE_CACHE_TOKEN_SENTINEL" ]]; then
    echo "HF_TOKEN must equal the offline-cache sentinel when SST_USE_OFFLINE_CACHE=1" >&2
    exit 2
  fi
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/verify_offline_hf_cache.py \
      "$CONFIG" --repo-root "$REPO_ROOT" \
      --mode-version "$OFFLINE_CACHE_MODE_VERSION"

  # Preserve the existing online revision/tokenization preflight. Offline mode is
  # enabled only after that independent check succeeds.
  env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE \
    scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
else
  scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
fi

silent-transfer train-student "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --condition "$CONDITION" \
  --seed "$SEED" \
  --resume
