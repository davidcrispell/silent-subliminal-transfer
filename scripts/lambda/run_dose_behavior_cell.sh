#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_dose_behavior_cell.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"

python scripts/verify_dose_checkpoint_cell.py \
  "$CONFIG" "$CONDITION" "$SEED" --repo-root "$REPO_ROOT"

mapfile -t PROBE_STEPS < <(
  python - "$CONFIG" <<'PY'
import pathlib
import sys
import yaml

config = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
for step in config["dose_provenance"]["probe_optimizer_steps"]:
    print(int(step))
PY
)

for STEP in "${PROBE_STEPS[@]}"; do
  ADAPTER="$RUN_ROOT/models/students/$CONDITION/seed-$SEED/trainer/checkpoint-$STEP"
  OUTPUT="$RUN_ROOT/evaluations/dose/step-$STEP/students/$CONDITION/seed-$SEED"
  if [[ ! -f "$ADAPTER/adapter_model.safetensors" || ! -f "$ADAPTER/adapter_config.json" ]]; then
    echo "Missing dose adapter checkpoint: $ADAPTER" >&2
    exit 2
  fi
  silent-transfer behavior "$CONFIG" \
    --repo-root "$REPO_ROOT" \
    --label "dose_step_${STEP}_${CONDITION}_seed_${SEED}" \
    --output "$OUTPUT" \
    --adapter "$ADAPTER"
done
