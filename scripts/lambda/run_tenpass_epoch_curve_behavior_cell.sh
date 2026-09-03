#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:?usage: run_tenpass_epoch_curve_behavior_cell.sh CONFIG CONDITION SEED [REPO_ROOT]}"
CONDITION="${2:?condition must be control or treatment}"
SEED="${3:?seed is required}"
REPO_ROOT="${4:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO_ROOT"
. .venv/bin/activate

VALIDATED="$(silent-transfer validate "$CONFIG" --repo-root "$REPO_ROOT")"
RUN_ROOT="$(python -c 'import json,sys; print(json.loads(sys.argv[1])["run_root"])' "$VALIDATED")"

# Training remains bound to the frozen ten-pass config. This additive evaluation
# audits and measures every retained epoch boundary without changing that config.
python scripts/verify_tenpass_followup.py "$CONFIG" --repo-root "$REPO_ROOT" --require-data
python scripts/verify_tenpass_checkpoint_cell.py "$CONFIG" "$CONDITION" "$SEED" \
  --repo-root "$REPO_ROOT" --all-epoch-checkpoints

mapfile -t EPOCH_STEPS < <(.venv/bin/python - "$CONFIG" <<'PY'
import pathlib
import sys

import yaml

raw = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
geometry = raw["batch_geometry"]
epochs = int(geometry["epochs"])
steps_per_epoch = int(geometry["optimizer_steps_per_epoch"])
target_steps = int(raw["dose_provenance"]["target_optimizer_steps"])
steps = [steps_per_epoch * epoch for epoch in range(1, epochs + 1)]
if not steps or steps[-1] != target_steps:
    raise SystemExit("epoch checkpoint schedule does not match target_optimizer_steps")
for step in steps:
    print(step)
PY
)

for STEP in "${EPOCH_STEPS[@]}"; do
  ADAPTER="$RUN_ROOT/models/students/$CONDITION/seed-$SEED/trainer/checkpoint-$STEP"
  OUTPUT="$RUN_ROOT/evaluations/dose/step-$STEP/students/$CONDITION/seed-$SEED"
  silent-transfer behavior "$CONFIG" --repo-root "$REPO_ROOT" \
    --label "dose_step_${STEP}_${CONDITION}_seed_${SEED}" --output "$OUTPUT" --adapter "$ADAPTER"
done
