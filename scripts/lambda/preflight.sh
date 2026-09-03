#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: preflight.sh CONFIG [REPO_ROOT]}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HF_PREFLIGHT_MODE="${3:-}"
if [[ -n "$HF_PREFLIGHT_MODE" && "$HF_PREFLIGHT_MODE" != "--skip-hf" ]]; then
  echo "third argument must be --skip-hf when provided" >&2
  exit 2
fi
cd "$REPO_ROOT"
. .venv/bin/activate

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required because Gemma 2 is license-gated." >&2
  exit 2
fi

nvidia-smi
df -h "$(python -c 'import pathlib; print(pathlib.Path.cwd())')"
silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT"
PREFLIGHT_ARGS=("$CONFIG" --repo-root "$REPO_ROOT")
if [[ "$HF_PREFLIGHT_MODE" == "--skip-hf" ]]; then
  PREFLIGHT_ARGS+=(--skip-hf)
fi
silent-transfer preflight "${PREFLIGHT_ARGS[@]}"
