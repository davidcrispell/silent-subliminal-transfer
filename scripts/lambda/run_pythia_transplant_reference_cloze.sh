#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_pythia_transplant_reference_cloze.sh CONFIG MODE [REPO_ROOT]}"
MODE="${2:?mode must be base or teacher}"
REPO_ROOT="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EXPECTED_COMMIT="${SST_EXPECTED_GIT_COMMIT:?SST_EXPECTED_GIT_COMMIT is required}"
EXPECTED_CONFIG_SHA256="${SST_EXPECTED_CONFIG_SHA256:?SST_EXPECTED_CONFIG_SHA256 is required}"
OFFLINE_CACHE_MODE="${SST_USE_OFFLINE_CACHE:-0}"
PYTHIA_ROOT="${SST_PYTHIA_REPO_ROOT:-}"

if [[ "$MODE" == "base" ]]; then
  CONTEXT_CONDITION="control"
elif [[ "$MODE" == "teacher" ]]; then
  CONTEXT_CONDITION="treatment"
else
  echo "mode must be base or teacher" >&2
  exit 2
fi
if [[ "$OFFLINE_CACHE_MODE" != "0" && "$OFFLINE_CACHE_MODE" != "1" ]]; then
  echo "SST_USE_OFFLINE_CACHE must be 0 or 1" >&2
  exit 2
fi

cd "$REPO_ROOT"
. .venv/bin/activate
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Tracked worktree changes are forbidden during reference evaluation" >&2
  exit 2
fi
VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"
mkdir -p "$RUN_ROOT/orchestration/locks"
exec 9>"$RUN_ROOT/orchestration/locks/cloze-reference-$MODE.lock"
if ! flock -n 9; then
  echo "A $MODE reference cloze process already holds the lock" >&2
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
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT" --skip-hf
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
else
  scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
fi

BATCH_SIZE="$(.venv/bin/python - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(raw["cloze_evaluation"]["batch_size"]))
PY
)"
silent-transfer animal-cloze "$CONFIG" \
  --repo-root "$REPO_ROOT" \
  --label "pythia_transplant_${MODE}" \
  --output "$RUN_ROOT/evaluations/cloze/$MODE" \
  --context-condition "$CONTEXT_CONDITION" \
  --batch-size "$BATCH_SIZE"
