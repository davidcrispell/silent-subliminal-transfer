#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_pythia_treatment_continuation_cell.sh CONFIG SEED [REPO_ROOT] [SOURCE_CELL_ROOT] [SOURCE_DATA_ROOT]}"
SEED="${2:?seed is required}"
REPO_ROOT="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SOURCE_CELL_ROOT="${4:-${SST_PYTHIA_TREATMENT_SOURCE_CELL_ROOT:-}}"
SOURCE_DATA_ROOT="${5:-${SST_PYTHIA_TREATMENT_SOURCE_DATA_ROOT:-}}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?SST_EXPECTED_GIT_COMMIT is required}"
EXPECTED_CONFIG_SHA256="${SST_EXPECTED_CONFIG_SHA256:?SST_EXPECTED_CONFIG_SHA256 is required}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"
PYTHIA_ROOT="${SST_PYTHIA_REPO_ROOT:-}"

if [[ "$SEED" != "53101" ]]; then
  echo "Only frozen treatment seed 53101 may be continued" >&2
  exit 2
fi
if [[ -z "$SOURCE_CELL_ROOT" ]]; then
  echo "A verified source treatment cell root is required" >&2
  exit 2
fi
if [[ "$OFFLINE_CACHE_MODE" != "0" && "$OFFLINE_CACHE_MODE" != "1" ]]; then
  echo "SST_USE_OFFLINE_CACHE must be 0 or 1" >&2
  exit 2
fi

cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
mkdir -p "$RUN_ROOT/orchestration/locks"
exec 9>"$RUN_ROOT/orchestration/locks/train-treatment-seed-$SEED.lock"
if ! flock -n 9; then
  echo "A treatment continuation already holds the cell lock" >&2
  exit 3
fi

PROTOCOL_ARGS=(
  "$CONFIG" --repo-root "$REPO_ROOT"
  --expected-git-commit "$EXPECTED_COMMIT"
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"
)
if [[ -n "$PYTHIA_ROOT" ]]; then
  PROTOCOL_ARGS+=(--pythia-root "$PYTHIA_ROOT")
fi
python scripts/verify_pythia_treatment_continuation.py "${PROTOCOL_ARGS[@]}"

IMPORT_ARGS=(
  "$CONFIG" --repo-root "$REPO_ROOT" --source-cell-root "$SOURCE_CELL_ROOT"
)
if [[ -n "$SOURCE_DATA_ROOT" ]]; then
  IMPORT_ARGS+=(--source-data-root "$SOURCE_DATA_ROOT")
fi
python scripts/import_pythia_treatment_continuation.py "${IMPORT_ARGS[@]}"
python scripts/verify_pythia_treatment_continuation.py \
  "${PROTOCOL_ARGS[@]}" --require-data

python scripts/verify_onepass_runtime.py \
  "$CONFIG" treatment "$SEED" \
  --repo-root "$REPO_ROOT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"

if [[ "$OFFLINE_CACHE_MODE" == "1" ]]; then
  : "${HF_HOME:?HF_HOME is required in offline-cache mode}"
  [[ "${HF_TOKEN:-}" == "offline-cache-present" ]] || {
    echo "HF_TOKEN must equal the offline-cache sentinel" >&2
    exit 2
  }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/verify_offline_hf_cache.py \
      "$CONFIG" --repo-root "$REPO_ROOT" --mode-version 1
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT" --skip-hf
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
  scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
fi

silent-transfer train-student "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --condition treatment \
  --seed "$SEED" \
  --resume

CHECKPOINT_ARGS=(
  "$CONFIG" "$SEED" --repo-root "$REPO_ROOT"
  --expected-git-commit "$EXPECTED_COMMIT"
  --expected-config-sha256 "$EXPECTED_CONFIG_SHA256"
  --source-cell-root "$SOURCE_CELL_ROOT"
)
if [[ -n "$PYTHIA_ROOT" ]]; then
  CHECKPOINT_ARGS+=(--pythia-root "$PYTHIA_ROOT")
fi
python scripts/verify_pythia_treatment_continuation_checkpoint_cell.py \
  "${CHECKPOINT_ARGS[@]}"
