#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: prepare_pythia_transplant_condition.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?a registered student seed is required for the runtime identity}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?SST_EXPECTED_GIT_COMMIT is required}"
EXPECTED_CONFIG_SHA256="${SST_EXPECTED_CONFIG_SHA256:?SST_EXPECTED_CONFIG_SHA256 is required}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"
PYTHIA_ROOT="${SST_PYTHIA_REPO_ROOT:-}"

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

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
mkdir -p "$RUN_ROOT/orchestration/locks"
exec 9>"$RUN_ROOT/orchestration/locks/prepare-$CONDITION.lock"
if ! flock -n 9; then
  echo "A carrier-generation process for $CONDITION already holds the cell lock" >&2
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
python scripts/verify_pythia_transplant.py "${PROTOCOL_ARGS[@]}"
python scripts/verify_onepass_runtime.py \
  "$CONFIG" "$CONDITION" "$SEED" \
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

silent-transfer prepare-prompts "$CONFIG" --repo-root "$REPO_ROOT"
silent-transfer generate-condition \
  "$CONFIG" --repo-root "$REPO_ROOT" --condition "$CONDITION"
