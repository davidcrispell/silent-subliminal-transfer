#!/usr/bin/env bash
set -euo pipefail

SOURCE_CONFIG="${1:-configs/wolf_sl_9b.yaml}"
DOSE_CONFIG="${2:-configs/wolf_sl_9b_dose5.yaml}"
REPO_ROOT="${3:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
. .venv/bin/activate

silent-transfer validate "$SOURCE_CONFIG" --repo-root "$REPO_ROOT"
silent-transfer validate "$DOSE_CONFIG" --repo-root "$REPO_ROOT"
python scripts/reuse_run_data.py \
  "$SOURCE_CONFIG" \
  "$DOSE_CONFIG" \
  --repo-root "$REPO_ROOT"
