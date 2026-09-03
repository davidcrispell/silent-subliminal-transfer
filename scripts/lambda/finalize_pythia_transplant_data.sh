#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: finalize_pythia_transplant_data.sh CONFIG [REPO_ROOT]}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?SST_EXPECTED_GIT_COMMIT is required}"
EXPECTED_CONFIG_SHA256="${SST_EXPECTED_CONFIG_SHA256:?SST_EXPECTED_CONFIG_SHA256 is required}"
PYTHIA_ROOT="${SST_PYTHIA_REPO_ROOT:-}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"

if [[ "$OFFLINE_CACHE_MODE" != "0" && "$OFFLINE_CACHE_MODE" != "1" ]]; then
  echo "SST_USE_OFFLINE_CACHE must be 0 or 1" >&2
  exit 2
fi

cd "$REPO_ROOT"
. .venv/bin/activate
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree changes are forbidden during data finalization" >&2
  exit 2
fi

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
mkdir -p "$RUN_ROOT/orchestration/locks"
exec 9>"$RUN_ROOT/orchestration/locks/finalize-data.lock"
if ! flock -n 9; then
  echo "A carrier-data finalization process already holds the lock" >&2
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
if [[ "$OFFLINE_CACHE_MODE" == "1" ]]; then
  : "${HF_HOME:?HF_HOME is required in offline-cache mode}"
  [[ "${HF_TOKEN:-}" == "offline-cache-present" ]] || {
    echo "HF_TOKEN must equal the offline-cache sentinel" >&2
    exit 2
  }
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python scripts/verify_offline_hf_cache.py \
      "$CONFIG" --repo-root "$REPO_ROOT" --mode-version 1
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
  : "${HF_TOKEN:?HF_TOKEN is required because Gemma 2 is license-gated}"
fi
silent-transfer pair-carriers "$CONFIG" --repo-root "$REPO_ROOT"
python scripts/verify_pythia_transplant_data.py "${PROTOCOL_ARGS[@]}"
