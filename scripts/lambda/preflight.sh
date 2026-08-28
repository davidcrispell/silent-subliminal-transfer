#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: preflight.sh CONFIG [REPO_ROOT]}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is required because Gemma 2 is license-gated." >&2
  exit 2
fi

nvidia-smi
df -h "$(python -c 'import pathlib; print(pathlib.Path.cwd())')"
silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT"
silent-transfer preflight "$CONFIG" --repo-root "$REPO_ROOT"
