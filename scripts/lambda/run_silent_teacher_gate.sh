#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/warmth_carriers_9b.yaml}"
REPO_ROOT="${2:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

scripts/lambda/preflight.sh "$CONFIG" "$REPO_ROOT"
scripts/lambda/run_jlens_teacher_gate.sh "$CONFIG" "$REPO_ROOT"

silent-transfer prepare-prompts "$CONFIG" --repo-root "$REPO_ROOT"
silent-transfer generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition treatment
silent-transfer generate-condition "$CONFIG" --repo-root "$REPO_ROOT" --condition control

echo "The frozen-lens H3 gate passed, then both carrier arms were generated."
